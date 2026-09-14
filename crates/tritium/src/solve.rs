//! 1D cell-centred finite-volume + theta-stepping solver for (T1–T2).
//!
//! The slab `0 ≤ x ≤ L` carries `cells` finite volumes (`dx = L / cells`)
//! holding the mobile concentration `c_m` plus one trapped population per
//! trap species. Diffusion assembles the tridiagonal rate matrix `A` with
//! `dc/dt = A c + b` (`b` folds Dirichlet/Sieverts/Henry face values and
//! the volumetric source); each implicit theta step
//!
//! ```text
//! (I − dt·θ·A) cⁿ⁺¹ = (I + dt·(1−θ)·A) cⁿ + dt·b − Σⱼ(ctⱼ* − ctⱼⁿ)
//! ```
//!
//! solves through [`nucleide_linalg::tridiag`] (Thomas, O(N)) while the
//! per-cell trap populations update by the exact backward-Euler map of
//! (T2) at the latest mobile iterate:
//!
//! ```text
//! ctⱼⁿ⁺¹ = (ctⱼⁿ + dt·kⱼ·cⁿ⁺¹·Nⱼ) / (1 + dt·(kⱼ·cⁿ⁺¹ + pⱼ)),
//! ```
//!
//! which is positivity-preserving and bounded by `Nⱼ`. Mobile and traps
//! couple by Picard iteration to `rtol`/`atol` (the kinetics `solve.rs`
//! pattern: options struct with `validate()`, exact stepping onto output
//! times, a hard step budget); trap-free systems take a single solve per
//! step. Temperature is the caller-supplied steady profile from
//! [`TransportParams`](crate::params::TransportParams) — no heat solve in
//! v1. Recombination ends are rejected with
//! [`Error::RecombinationOpen`](crate::error::Error) (G5 named-open).
//!
//! Closed-form helpers pin the algebraic gates without a solve:
//! [`time_lag`] (G2-lag), [`breakthrough_ratio`] (G2 series),
//! [`equilibrium_trapped`] (T2-eq, G3b), [`effective_diffusivity`] (G3a),
//! [`irreversible_fill`] (G3c), [`sieverts_concentration`] (G4).

use crate::bc::Boundary;
use crate::error::Error;
use crate::params::TransportParams;

/// Theta-method selector: `1/2` (Crank–Nicolson, default) or `1` (backward
/// Euler).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theta {
    /// Second-order A-stable Crank–Nicolson (default).
    #[default]
    CrankNicolson,
    /// First-order L-stable backward Euler (extra damping of fast trap
    /// transients; larger phase error on the breakthrough tail).
    BackwardEuler,
}

impl Theta {
    /// Theta value (`1/2` or `1`).
    pub fn value(self) -> f64 {
        match self {
            Theta::CrankNicolson => 0.5,
            Theta::BackwardEuler => 1.0,
        }
    }
}

/// Solver tolerances and budgets.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverOptions {
    /// Implicit integration method (theta value).
    pub theta: Theta,
    /// Relative tolerance for the per-step trap-coupling Picard iteration.
    pub rtol: f64,
    /// Absolute tolerance (in concentration units) for the Picard iteration.
    pub atol: f64,
    /// Smallest allowed internal step \[s\].
    pub dt_min: f64,
    /// Largest allowed internal step \[s\].
    pub dt_max: f64,
    /// Maximum accepted internal steps over the whole grid.
    pub max_steps: usize,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            theta: Theta::CrankNicolson,
            rtol: 1e-9,
            atol: 1e-12,
            dt_min: 1e-14,
            dt_max: f64::INFINITY,
            max_steps: 1_000_000,
        }
    }
}

impl SolverOptions {
    /// Validate tolerances and budgets (finite, positive, ordered).
    pub fn validate(&self) -> Result<(), Error> {
        if !self.rtol.is_finite() || self.rtol <= 0.0 {
            return Err(Error::BadOption("rtol must be finite and > 0"));
        }
        if !self.atol.is_finite() || self.atol <= 0.0 {
            return Err(Error::BadOption("atol must be finite and > 0"));
        }
        if !self.dt_min.is_finite() || self.dt_min <= 0.0 {
            return Err(Error::BadOption("dt_min must be finite and > 0"));
        }
        if self.dt_max <= 0.0 || self.dt_max < self.dt_min {
            return Err(Error::BadOption("need 0 < dt_min <= dt_max"));
        }
        if self.max_steps == 0 {
            return Err(Error::BadOption("max_steps must be > 0"));
        }
        Ok(())
    }
}

/// Output time grid: strictly increasing times `> 0`.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeGrid {
    /// Output times \[s\].
    pub times: Vec<f64>,
}

impl TimeGrid {
    /// Validate: at least one time, all finite and `> 0`, strictly
    /// increasing.
    pub fn new(times: Vec<f64>) -> Result<Self, Error> {
        if times.is_empty() {
            return Err(Error::BadGrid("time grid must not be empty"));
        }
        let mut prev = 0.0_f64;
        for t in &times {
            if !t.is_finite() || *t <= prev {
                return Err(Error::BadGrid(
                    "times must be finite, > 0, strictly increasing",
                ));
            }
            prev = *t;
        }
        Ok(Self { times })
    }
}

/// Initial state at `t = 0`: mobile profile plus per-cell trapped loads.
#[derive(Debug, Clone, PartialEq)]
pub struct InitialState {
    /// Mobile concentration per cell \[mol/m³\] (`>= 0`, finite).
    pub mobile: Vec<f64>,
    /// Trapped concentrations `[cell][trap]` \[mol/m³\] (`0 <= ct <= N_j`).
    pub trapped: Vec<Vec<f64>>,
}

impl InitialState {
    /// Zero initial state for the params geometry (mobile and traps zero).
    pub fn zeros(params: &TransportParams) -> Self {
        Self {
            mobile: vec![0.0; params.cells],
            trapped: vec![vec![0.0; params.traps.len()]; params.cells],
        }
    }

    /// Uniform mobile value with per-species uniform trapped loads.
    pub fn uniform(
        params: &TransportParams,
        mobile_value: f64,
        trapped_values: Vec<f64>,
    ) -> Result<Self, Error> {
        if trapped_values.len() != params.traps.len() {
            return Err(Error::BadState(
                "trapped values length must match trap count",
            ));
        }
        Self::new(
            params,
            vec![mobile_value; params.cells],
            vec![trapped_values; params.cells],
        )
    }

    /// Validate raw profiles against the params geometry: mobile length
    /// matches the cell count (finite, `>= 0`); trapped is
    /// `[cell][trap]`-shaped (finite, `0 <= ct <= N_j`).
    pub fn new(
        params: &TransportParams,
        mobile: Vec<f64>,
        trapped: Vec<Vec<f64>>,
    ) -> Result<Self, Error> {
        if mobile.len() != params.cells {
            return Err(Error::BadState("mobile length must match cell count"));
        }
        if mobile.iter().any(|c| !c.is_finite() || *c < 0.0) {
            return Err(Error::BadState("mobile must be finite and >= 0"));
        }
        if trapped.len() != params.cells {
            return Err(Error::BadState(
                "trapped outer length must match cell count",
            ));
        }
        for (cell, row) in trapped.iter().enumerate() {
            if row.len() != params.traps.len() {
                return Err(Error::BadState(
                    "trapped inner length must match trap count",
                ));
            }
            for (j, ct) in row.iter().enumerate() {
                let n = params.traps[j].site_density;
                if !ct.is_finite() || *ct < 0.0 || *ct > n {
                    return Err(Error::BadState("trapped loads must satisfy 0 <= ct <= N_j"));
                }
            }
            let _ = cell;
        }
        Ok(Self { mobile, trapped })
    }
}

/// Full transient: one mobile/trapped row plus outward surface fluxes per
/// output time (the `t = 0` state is the caller-supplied
/// [`InitialState`], echoed here as `initial`).
#[derive(Debug, Clone, PartialEq)]
pub struct Solution {
    /// Output times \[s\] (echo of the grid).
    pub times: Vec<f64>,
    /// Mobile concentration at each output time (`[time][cell]`).
    pub mobile: Vec<Vec<f64>>,
    /// Trapped concentrations (`[time][cell][trap]`).
    pub trapped: Vec<Vec<Vec<f64>>>,
    /// Outward flux at `x = 0` \[mol/m²/s\] (positive leaves the slab).
    pub flux_left: Vec<f64>,
    /// Outward flux at `x = L` \[mol/m²/s\] (positive leaves the slab).
    pub flux_right: Vec<f64>,
    /// Cell width `dx` \[m\] (for inventory integrals).
    pub dx: f64,
    /// The `t = 0` state.
    pub initial: InitialState,
}

impl Solution {
    /// Mobile inventory per unit area at each output time \[mol/m²\].
    pub fn inventory_mobile(&self) -> Vec<f64> {
        self.mobile
            .iter()
            .map(|row| row.iter().sum::<f64>() * self.dx)
            .collect()
    }

    /// Trapped inventory per unit area at each output time \[mol/m²\].
    pub fn inventory_trapped(&self) -> Vec<f64> {
        self.trapped
            .iter()
            .map(|rows| rows.iter().map(|row| row.iter().sum::<f64>()).sum::<f64>() * self.dx)
            .collect()
    }

    /// Total (mobile + trapped) inventory per unit area \[mol/m²\].
    pub fn inventory_total(&self) -> Vec<f64> {
        let m = self.inventory_mobile();
        let t = self.inventory_trapped();
        m.iter().zip(&t).map(|(a, b)| a + b).collect()
    }
}

/// Trap-free-style steady state: mobile profile, Langmuir trapped loads,
/// boundary fluxes, and inventories.
///
/// At steady state `dct/dt = 0`, so every trap sits on its Langmuir
/// isotherm (T2-eq) and the mobile profile solves the trap-free system
/// `A c + b = 0` (G1/G4); the trapped loads follow pointwise (G3b).
#[derive(Debug, Clone, PartialEq)]
pub struct SteadyState {
    /// Cell-centre positions \[m\].
    pub centres: Vec<f64>,
    /// Mobile concentration per cell \[mol/m³\].
    pub mobile: Vec<f64>,
    /// Trapped concentrations `[cell][trap]` \[mol/m³\].
    pub trapped: Vec<Vec<f64>>,
    /// Outward flux at `x = 0` \[mol/m²/s\].
    pub flux_left: f64,
    /// Outward flux at `x = L` \[mol/m²/s\].
    pub flux_right: f64,
    /// Mobile inventory per unit area \[mol/m²\].
    pub inventory_mobile: f64,
    /// Trapped inventory per unit area \[mol/m²\].
    pub inventory_trapped: f64,
}

// ---------------------------------------------------------------------------
// Closed-form helpers (algebraic gates)
// ---------------------------------------------------------------------------

/// Permeation time lag `t_lag = L²/6D` \[s\] (G2-lag).
///
/// Rejects non-finite or non-positive inputs.
pub fn time_lag(length: f64, diffusivity: f64) -> Result<f64, Error> {
    if !length.is_finite() || length <= 0.0 {
        return Err(Error::BadData("length must be finite and > 0"));
    }
    if !diffusivity.is_finite() || diffusivity <= 0.0 {
        return Err(Error::BadData("diffusivity must be finite and > 0"));
    }
    Ok(length * length / (6.0 * diffusivity))
}

/// Normalized outlet flux `J(L,t)/J_ss` of the G2 permeation transient
/// (trap-free slab, `c(0,t) = c0`, `c(L,t) = 0`, `c(x,0) = 0`):
///
/// ```text
/// J/Jss = 1 + 2 Σ_{n≥1} (−1)ⁿ exp(−D n² π² t / L²).
/// ```
///
/// The series truncates once `|term| < 1e-18` (the fixture convention).
/// Rejects non-finite/non-positive `D`/`L` and non-finite/non-positive `t`.
pub fn breakthrough_ratio(diffusivity: f64, length: f64, t: f64) -> Result<f64, Error> {
    if !diffusivity.is_finite() || diffusivity <= 0.0 {
        return Err(Error::BadData("diffusivity must be finite and > 0"));
    }
    if !length.is_finite() || length <= 0.0 {
        return Err(Error::BadData("length must be finite and > 0"));
    }
    if !t.is_finite() || t <= 0.0 {
        return Err(Error::BadData("time must be finite and > 0"));
    }
    let alpha = diffusivity * std::f64::consts::PI.powi(2) * t / length.powi(2);
    let mut sum = 0.0;
    let mut n = 1_u32;
    loop {
        let term = 2.0 * (-1.0_f64).powi(n as i32) * (-alpha * (n as f64).powi(2)).exp();
        sum += term;
        if term.abs() < 1e-18 {
            break;
        }
        n += 1;
        debug_assert!(n < 10_000_000, "G2 series failed to converge");
    }
    Ok(1.0 + sum)
}

/// Langmuir equilibrium load `c_t = N K c / (1 + K c)` \[mol/m³\] (T2-eq).
///
/// Rejects non-finite/negative inputs.
pub fn equilibrium_trapped(
    site_density: f64,
    equilibrium_constant: f64,
    c_mobile: f64,
) -> Result<f64, Error> {
    if !site_density.is_finite() || site_density < 0.0 {
        return Err(Error::BadData("site density must be finite and >= 0"));
    }
    if !equilibrium_constant.is_finite() || equilibrium_constant < 0.0 {
        return Err(Error::BadData(
            "equilibrium constant must be finite and >= 0",
        ));
    }
    if !c_mobile.is_finite() || c_mobile < 0.0 {
        return Err(Error::BadData(
            "mobile concentration must be finite and >= 0",
        ));
    }
    let kc = equilibrium_constant * c_mobile;
    Ok(site_density * kc / (1.0 + kc))
}

/// Oriani effective diffusivity `D_eff = D / (1 + K N)` \[m²/s\] (G3a):
/// the low-occupancy (`K c ≪ 1`) local-equilibrium limit of (T1–T2).
///
/// Rejects non-finite/negative inputs.
pub fn effective_diffusivity(
    diffusivity: f64,
    equilibrium_constant: f64,
    site_density: f64,
) -> Result<f64, Error> {
    if !diffusivity.is_finite() || diffusivity < 0.0 {
        return Err(Error::BadData("diffusivity must be finite and >= 0"));
    }
    if !equilibrium_constant.is_finite() || equilibrium_constant < 0.0 {
        return Err(Error::BadData(
            "equilibrium constant must be finite and >= 0",
        ));
    }
    if !site_density.is_finite() || site_density < 0.0 {
        return Err(Error::BadData("site density must be finite and >= 0"));
    }
    Ok(diffusivity / (1.0 + equilibrium_constant * site_density))
}

/// Irreversible-trap fill `c_t(t) = N (1 − e^{−k c t})` \[mol/m³\] (G3c):
/// (T2) at `p → 0` with mobile `c` held fixed and `c_t(0) = 0`.
///
/// Rejects non-finite/negative inputs.
pub fn irreversible_fill(
    rate_k: f64,
    c_mobile: f64,
    site_density: f64,
    t: f64,
) -> Result<f64, Error> {
    if !rate_k.is_finite() || rate_k < 0.0 {
        return Err(Error::BadData("trapping rate must be finite and >= 0"));
    }
    if !c_mobile.is_finite() || c_mobile < 0.0 {
        return Err(Error::BadData(
            "mobile concentration must be finite and >= 0",
        ));
    }
    if !site_density.is_finite() || site_density < 0.0 {
        return Err(Error::BadData("site density must be finite and >= 0"));
    }
    if !t.is_finite() || t < 0.0 {
        return Err(Error::BadData("time must be finite and >= 0"));
    }
    Ok(site_density * (1.0 - (-rate_k * c_mobile * t).exp()))
}

/// Sieverts surface concentration `c = K_S sqrt(p)` \[mol/m³\] (G4).
///
/// Rejects non-finite/negative inputs.
pub fn sieverts_concentration(solubility: f64, pressure: f64) -> Result<f64, Error> {
    if !solubility.is_finite() || solubility < 0.0 {
        return Err(Error::BadData("solubility must be finite and >= 0"));
    }
    if !pressure.is_finite() || pressure < 0.0 {
        return Err(Error::BadData("pressure must be finite and >= 0"));
    }
    Ok(solubility * pressure.sqrt())
}

// ---------------------------------------------------------------------------
// Discrete operator
// ---------------------------------------------------------------------------

/// Tridiagonal rate system `dc/dt = A c + rhs` with `A` in
/// sub/diag/sup form and face diffusivities for flux evaluation.
struct DiffusionSystem {
    sub: Vec<f64>,
    diag: Vec<f64>,
    sup: Vec<f64>,
    rhs: Vec<f64>,
    face_d: Vec<f64>,
}

/// Reject recombination ends (G5 named-open) before assembly.
fn reject_recombination(left: &Boundary, right: &Boundary) -> Result<(), Error> {
    if left.is_recombination() || right.is_recombination() {
        return Err(Error::RecombinationOpen);
    }
    Ok(())
}

/// Face concentration imposed by an equilibrium end; `None` for zero flux.
fn face_concentration(bc: &Boundary) -> Option<f64> {
    bc.surface_concentration()
}

/// Assemble the cell-centred finite-volume diffusion operator.
///
/// Interior faces use the arithmetic-mean diffusivity; boundary faces the
/// adjacent cell value. A Dirichlet/Sieverts/Henry face of value `cf` with
/// conductance `g = 2 D/dx` contributes `−g/dx` to the diagonal and
/// `g cf/dx` to the right-hand side; a zero-flux face contributes nothing.
fn assemble(
    params: &TransportParams,
    left: &Boundary,
    right: &Boundary,
) -> Result<DiffusionSystem, Error> {
    reject_recombination(left, right)?;
    let n = params.cells;
    let dx = params.dx();
    let d_cell = params.diffusivities()?;
    let mut face_d = vec![0.0; n + 1];
    face_d[0] = d_cell[0];
    face_d[n] = d_cell[n - 1];
    for i in 1..n {
        face_d[i] = 0.5 * (d_cell[i - 1] + d_cell[i]);
    }
    let mut sub = vec![0.0; n.saturating_sub(1)];
    let mut diag = vec![0.0; n];
    let mut sup = vec![0.0; n.saturating_sub(1)];
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        // West coupling.
        let (gw, cw) = if i > 0 {
            (face_d[i] / dx, None)
        } else {
            match face_concentration(left) {
                Some(cf) => (2.0 * face_d[0] / dx, Some(cf)),
                None => (0.0, None),
            }
        };
        // East coupling.
        let (ge, ce) = if i + 1 < n {
            (face_d[i + 1] / dx, None)
        } else {
            match face_concentration(right) {
                Some(cf) => (2.0 * face_d[n] / dx, Some(cf)),
                None => (0.0, None),
            }
        };
        diag[i] = -(gw + ge) / dx;
        if i > 0 {
            sub[i - 1] = gw / dx;
        }
        if i + 1 < n {
            sup[i] = ge / dx;
        }
        let mut b = params.source_at(i);
        if let Some(cf) = cw {
            b += gw * cf / dx;
        }
        if let Some(cf) = ce {
            b += ge * cf / dx;
        }
        rhs[i] = b;
    }
    Ok(DiffusionSystem {
        sub,
        diag,
        sup,
        rhs,
        face_d,
    })
}

/// Outward surface flux \[mol/m²/s\] from the boundary-adjacent cell value
/// (`D (c_cell − c_face) / (dx/2)` for equilibrium ends, `0` for zero flux).
fn outward_flux(bc: &Boundary, c_cell: f64, d_face: f64, dx: f64) -> f64 {
    match face_concentration(bc) {
        Some(cf) => d_face * (c_cell - cf) / (dx * 0.5),
        None => 0.0,
    }
}

/// Exact backward-Euler map of (T2) at fixed mobile `c` over `dt`:
///
/// ```text
/// ctⱼⁿᵉʷ = (ctⱼᵒˡᵈ + dt·kⱼ·c·Nⱼ) / (1 + dt·(kⱼ·c + pⱼ)).
/// ```
fn trap_update(ct_old: f64, c: f64, k: f64, p: f64, site_density: f64, dt: f64) -> f64 {
    (ct_old + dt * k * c * site_density) / (1.0 + dt * (k * c + p))
}

fn tridiag_err(e: nucleide_linalg::TridiagError) -> Error {
    Error::Tridiag(e.to_string())
}

/// Cap on the per-step trap-coupling Picard iterations.
const MAX_PICARD: usize = 100;

// ---------------------------------------------------------------------------
// Steady state (G1/G3b/G4)
// ---------------------------------------------------------------------------

/// Trap-free-style steady state of (T1–T2).
///
/// Solves `A c + b = 0` through [`nucleide_linalg::tridiag`] (exact to
/// roundoff for the G1/G4 linear profiles) and evaluates the Langmuir
/// isotherm (T2-eq) pointwise for the trapped loads (G3b). Rejects
/// recombination ends with [`Error::RecombinationOpen`].
pub fn steady_state(
    params: &TransportParams,
    left: &Boundary,
    right: &Boundary,
) -> Result<SteadyState, Error> {
    let sys = assemble(params, left, right)?;
    let n = params.cells;
    let dx = params.dx();
    let neg_sub: Vec<f64> = sys.sub.iter().map(|v| -v).collect();
    let neg_diag: Vec<f64> = sys.diag.iter().map(|v| -v).collect();
    let neg_sup: Vec<f64> = sys.sup.iter().map(|v| -v).collect();
    let mobile = nucleide_linalg::tridiag::solve(&neg_sub, &neg_diag, &neg_sup, &sys.rhs)
        .map_err(tridiag_err)?;
    // Langmuir trapped loads at the cell temperatures.
    let mut trapped = vec![vec![0.0; params.traps.len()]; n];
    for i in 0..n {
        let rates = params.trap_rates_at(i)?;
        for (j, (k, p)) in rates.iter().enumerate() {
            let keq = k / p;
            trapped[i][j] = equilibrium_trapped(params.traps[j].site_density, keq, mobile[i])?;
        }
    }
    let inventory_mobile: f64 = mobile.iter().sum::<f64>() * dx;
    let inventory_trapped: f64 = trapped
        .iter()
        .map(|row| row.iter().sum::<f64>())
        .sum::<f64>()
        * dx;
    Ok(SteadyState {
        centres: params.cell_centres(),
        flux_left: outward_flux(left, mobile[0], sys.face_d[0], dx),
        flux_right: outward_flux(right, mobile[n - 1], sys.face_d[n], dx),
        mobile,
        trapped,
        inventory_mobile,
        inventory_trapped,
    })
}

// ---------------------------------------------------------------------------
// Transient (G2 + invariants)
// ---------------------------------------------------------------------------

/// Solve the (T1–T2) transient over `grid` from `initial`.
///
/// Between consecutive event times (`t = 0` plus every output time) the
/// stepper takes `ceil(span / dt_max)` theta steps of equal width (each
/// `>= dt_min`, else [`Error::BadOption`]); output rows land exactly on
/// the grid times. Mobile and traps couple by Picard iteration to
/// `rtol`/`atol` (trap-free steps take one solve). Exceeding `max_steps`
/// returns [`Error::StepBudget`]; recombination ends return
/// [`Error::RecombinationOpen`].
pub fn solve(
    params: &TransportParams,
    left: &Boundary,
    right: &Boundary,
    grid: &TimeGrid,
    initial: &InitialState,
    opts: &SolverOptions,
) -> Result<Solution, Error> {
    opts.validate()?;
    reject_recombination(left, right)?;
    if initial.mobile.len() != params.cells
        || initial.trapped.len() != params.cells
        || initial
            .trapped
            .iter()
            .any(|row| row.len() != params.traps.len())
    {
        return Err(Error::BadState(
            "initial state shape must match params geometry",
        ));
    }
    let sys = assemble(params, left, right)?;
    let n = params.cells;
    let ntraps = params.traps.len();
    let dx = params.dx();
    let theta = opts.theta.value();

    let mut c = initial.mobile.clone();
    let mut ct = initial.trapped.clone();

    let mut mobile_out = Vec::with_capacity(grid.times.len());
    let mut trapped_out = Vec::with_capacity(grid.times.len());
    let mut flux_left = Vec::with_capacity(grid.times.len());
    let mut flux_right = Vec::with_capacity(grid.times.len());

    let mut t_prev = 0.0_f64;
    let mut steps = 0_usize;
    // Per-cell trap rates are time-independent (steady T profile): cache.
    let mut rates: Vec<Vec<(f64, f64)>> = Vec::with_capacity(n);
    for i in 0..n {
        rates.push(params.trap_rates_at(i)?);
    }

    for &t_out in &grid.times {
        let span = t_out - t_prev;
        let n_sub = ((span / opts.dt_max).ceil() as usize).max(1);
        let dt = span / n_sub as f64;
        if dt < opts.dt_min {
            return Err(Error::BadOption("output spacing needs a step below dt_min"));
        }
        for _ in 0..n_sub {
            // Theta-step matrices: M = I − dt·θ·A (strictly dominant, so
            // the Thomas pivots never vanish for well-posed inputs).
            let m_sub: Vec<f64> = sys.sub.iter().map(|v| -dt * theta * v).collect();
            let m_diag: Vec<f64> = sys.diag.iter().map(|v| 1.0 - dt * theta * v).collect();
            let m_sup: Vec<f64> = sys.sup.iter().map(|v| -dt * theta * v).collect();
            // Explicit half: e = (I + dt·(1−θ)·A) c + dt·b.
            let mut e = vec![0.0; n];
            for i in 0..n {
                let mut a_c = sys.diag[i] * c[i];
                if i > 0 {
                    a_c += sys.sub[i - 1] * c[i - 1];
                }
                if i + 1 < n {
                    a_c += sys.sup[i] * c[i + 1];
                }
                e[i] = c[i] + dt * (1.0 - theta) * a_c + dt * sys.rhs[i];
            }
            if ntraps == 0 {
                c = nucleide_linalg::tridiag::solve(&m_sub, &m_diag, &m_sup, &e)
                    .map_err(tridiag_err)?;
            } else {
                // Picard: trap sink from the latest trapped iterate, then
                // the exact per-cell backward-Euler trap map at the new
                // mobile iterate, until consecutive iterates agree to
                // rtol/atol (the first pass always runs predict + correct).
                let mut c_iter = c.clone();
                let mut ct_iter = ct.clone();
                let mut converged = false;
                for _ in 0..MAX_PICARD {
                    let mut rhs = e.clone();
                    for i in 0..n {
                        let old_total: f64 = ct[i].iter().sum();
                        let star_total: f64 = ct_iter[i].iter().sum();
                        rhs[i] -= star_total - old_total;
                    }
                    let c_next = nucleide_linalg::tridiag::solve(&m_sub, &m_diag, &m_sup, &rhs)
                        .map_err(tridiag_err)?;
                    let mut ct_next = ct.clone();
                    for i in 0..n {
                        for j in 0..ntraps {
                            let (k, p) = rates[i][j];
                            ct_next[i][j] = trap_update(
                                ct[i][j],
                                c_next[i],
                                k,
                                p,
                                params.traps[j].site_density,
                                dt,
                            );
                        }
                    }
                    let mut err = 0.0_f64;
                    for i in 0..n {
                        let scale = opts.atol + opts.rtol * c_next[i].abs().max(c_iter[i].abs());
                        err = err.max((c_next[i] - c_iter[i]).abs() / scale);
                        for j in 0..ntraps {
                            let scale_t = opts.atol
                                + opts.rtol * ct_next[i][j].abs().max(ct_iter[i][j].abs());
                            err = err.max((ct_next[i][j] - ct_iter[i][j]).abs() / scale_t);
                        }
                    }
                    c_iter = c_next;
                    ct_iter = ct_next;
                    if err <= 1.0 {
                        converged = true;
                        break;
                    }
                }
                if !converged {
                    return Err(Error::NotConverged);
                }
                c = c_iter;
                ct = ct_iter;
            }
            steps += 1;
            if steps > opts.max_steps {
                return Err(Error::StepBudget(opts.max_steps));
            }
        }
        t_prev = t_out;
        mobile_out.push(c.clone());
        trapped_out.push(ct.clone());
        flux_left.push(outward_flux(left, c[0], sys.face_d[0], dx));
        flux_right.push(outward_flux(right, c[n - 1], sys.face_d[n], dx));
    }

    Ok(Solution {
        times: grid.times.clone(),
        mobile: mobile_out,
        trapped: trapped_out,
        flux_left,
        flux_right,
        dx,
        initial: initial.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::tests::no_traps;

    fn tight(theta: Theta) -> SolverOptions {
        SolverOptions {
            theta,
            rtol: 1e-10,
            atol: 1e-14,
            dt_min: 1e-14,
            dt_max: 0.1,
            max_steps: 10_000_000,
        }
    }

    #[test]
    fn helpers_match_closed_forms() {
        // G2-lag: L^2/6D for the G1/G2 slab.
        assert!((time_lag(1e-3, 1e-9).unwrap() - 166.666_666_666_666_66).abs() < 1e-9);
        // G2 series endpoints: pre-breakthrough ~0, late ~1.
        assert!(breakthrough_ratio(1e-9, 1e-3, 1.0).unwrap().abs() < 1e-9);
        assert!(
            (breakthrough_ratio(1e-9, 1e-3, 1000.0).unwrap() - 0.999_896_553_627_592_4).abs()
                < 1e-12
        );
        // G3a: D_eff = D/4 at KN = 3.
        assert_eq!(effective_diffusivity(1e-9, 1.0, 3.0).unwrap(), 2.5e-10);
        // T2-eq: N K c/(1 + K c).
        assert!((equilibrium_trapped(2.0, 1.0, 0.5).unwrap() - 2.0 / 3.0).abs() < 1e-15);
        // G3c: N(1 − e^{−kct}) at t = 0 and late.
        assert_eq!(irreversible_fill(0.05, 1.0, 2.0, 0.0).unwrap(), 0.0);
        assert!((irreversible_fill(0.05, 1.0, 2.0, 1000.0).unwrap() - 2.0).abs() < 1e-12);
        // G4: K_S sqrt(p).
        assert_eq!(sieverts_concentration(2.0, 16.0).unwrap(), 8.0);
        assert!(time_lag(0.0, 1e-9).is_err());
        assert!(time_lag(1e-3, -1.0).is_err());
        assert!(breakthrough_ratio(0.0, 1e-3, 1.0).is_err());
        assert!(breakthrough_ratio(1e-9, 0.0, 1.0).is_err());
        assert!(breakthrough_ratio(1e-9, 1e-3, 0.0).is_err());
        assert!(breakthrough_ratio(1e-9, 1e-3, -1.0).is_err());
        assert!(effective_diffusivity(-1.0, 1.0, 3.0).is_err());
        assert!(effective_diffusivity(1e-9, -1.0, 3.0).is_err());
        assert!(effective_diffusivity(1e-9, 1.0, -3.0).is_err());
        assert!(equilibrium_trapped(-1.0, 1.0, 0.5).is_err());
        assert!(equilibrium_trapped(2.0, -1.0, 0.5).is_err());
        assert!(equilibrium_trapped(2.0, 1.0, -0.5).is_err());
        assert!(irreversible_fill(0.05, 1.0, 2.0, -1.0).is_err());
        assert!(irreversible_fill(-0.05, 1.0, 2.0, 1.0).is_err());
        assert!(irreversible_fill(0.05, -1.0, 2.0, 1.0).is_err());
        assert!(irreversible_fill(0.05, 1.0, -2.0, 1.0).is_err());
        assert!(sieverts_concentration(-1.0, 16.0).is_err());
        assert!(sieverts_concentration(2.0, -16.0).is_err());
    }

    #[test]
    fn steady_dirichlet_is_linear() {
        // G1 smoke: coarse grid recovers the linear profile to roundoff.
        let p = no_traps(1e-3, 8, 1e-9);
        let s = steady_state(
            &p,
            &Boundary::dirichlet(1.0).unwrap(),
            &Boundary::dirichlet(0.0).unwrap(),
        )
        .unwrap();
        for (i, c) in s.mobile.iter().enumerate() {
            let x = (i as f64 + 0.5) / 8.0;
            assert!((c - (1.0 - x)).abs() < 1e-12, "cell {i}: {c}");
        }
        assert!((s.flux_left + 1e-6).abs() < 1e-15);
        assert!((s.flux_right - 1e-6).abs() < 1e-15);
        assert!((s.inventory_mobile - 5e-4).abs() < 1e-15);
    }

    #[test]
    fn transient_holds_steady_state() {
        // Invariant: starting on the G1 profile with matching ends stays put.
        let p = no_traps(1e-3, 16, 1e-9);
        let s = steady_state(
            &p,
            &Boundary::dirichlet(1.0).unwrap(),
            &Boundary::dirichlet(0.0).unwrap(),
        )
        .unwrap();
        let init = InitialState::new(&p, s.mobile.clone(), vec![vec![]; 16]).unwrap();
        let sol = solve(
            &p,
            &Boundary::dirichlet(1.0).unwrap(),
            &Boundary::dirichlet(0.0).unwrap(),
            &TimeGrid::new(vec![10.0, 100.0]).unwrap(),
            &init,
            &tight(Theta::CrankNicolson),
        )
        .unwrap();
        for row in &sol.mobile {
            for (got, want) in row.iter().zip(&s.mobile) {
                assert!((got - want).abs() / want.max(1e-300) < 1e-9);
            }
        }
    }

    #[test]
    fn pure_neumann_steady_has_no_solution() {
        // Zero-flux ends with no source: −A is singular, so the Thomas
        // path surfaces its elimination error instead of a profile.
        let p = no_traps(1e-3, 8, 1e-9);
        assert!(steady_state(&p, &Boundary::ZeroFlux, &Boundary::ZeroFlux).is_err());
    }

    #[test]
    fn dirichlet_zeroflux_is_flat() {
        // ZeroFlux right end with uniform Dirichlet left: flat profile.
        let p = no_traps(1e-3, 8, 1e-9);
        let s = steady_state(&p, &Boundary::dirichlet(2.0).unwrap(), &Boundary::ZeroFlux).unwrap();
        for c in &s.mobile {
            assert!((c - 2.0).abs() < 1e-12);
        }
        assert!(s.flux_left.abs() < 1e-15);
        assert!(s.flux_right.abs() < 1e-15);
    }

    #[test]
    fn recombination_is_named_open() {
        let p = no_traps(1e-3, 8, 1e-9);
        let rec = Boundary::recombination(1.0).unwrap();
        let dir = Boundary::dirichlet(1.0).unwrap();
        assert_eq!(steady_state(&p, &dir, &rec), Err(Error::RecombinationOpen));
        assert_eq!(steady_state(&p, &rec, &dir), Err(Error::RecombinationOpen));
        let init = InitialState::zeros(&p);
        let grid = TimeGrid::new(vec![1.0]).unwrap();
        assert_eq!(
            solve(&p, &dir, &rec, &grid, &init, &SolverOptions::default()),
            Err(Error::RecombinationOpen)
        );
    }

    #[test]
    fn rejects_bad_options_and_states() {
        let p = no_traps(1e-3, 8, 1e-9);
        assert!(TimeGrid::new(vec![]).is_err());
        assert!(TimeGrid::new(vec![1.0, 1.0]).is_err());
        assert!(TimeGrid::new(vec![-1.0]).is_err());
        assert!(InitialState::new(&p, vec![0.0; 7], vec![vec![]; 8]).is_err());
        assert!(InitialState::new(&p, vec![-1.0; 8], vec![vec![]; 8]).is_err());
        assert!(InitialState::new(&p, vec![f64::NAN; 8], vec![vec![]; 8]).is_err());
        // Trapped-shape branches need a trap-bearing geometry.
        let tp = TransportParams::new(
            1e-3,
            8,
            1e-9,
            0.0,
            vec![crate::TrapSpec::new(0.05, 0.0, 0.01, 0.0, 2.0).unwrap()],
            vec![500.0],
            vec![],
        )
        .unwrap();
        assert!(InitialState::new(&tp, vec![1.0; 8], vec![vec![0.0]; 7]).is_err());
        assert!(InitialState::new(&tp, vec![1.0; 8], vec![vec![]; 8]).is_err());
        assert!(InitialState::new(&tp, vec![1.0; 8], vec![vec![3.0]; 8]).is_err());
        assert!(InitialState::new(&tp, vec![1.0; 8], vec![vec![-0.5]; 8]).is_err());
        assert!(InitialState::uniform(&tp, f64::NAN, vec![0.0]).is_err());
        assert!(SolverOptions {
            rtol: -1.0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SolverOptions {
            max_steps: 0,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(SolverOptions {
            dt_min: 1.0,
            dt_max: 0.5,
            ..Default::default()
        }
        .validate()
        .is_err());
        // A dt_min above the output spacing is rejected, not stalled on.
        let init = InitialState::zeros(&p);
        let grid = TimeGrid::new(vec![0.5]).unwrap();
        let opts = SolverOptions {
            dt_min: 1.0,
            ..Default::default()
        };
        assert!(solve(
            &p,
            &Boundary::dirichlet(1.0).unwrap(),
            &Boundary::dirichlet(0.0).unwrap(),
            &grid,
            &init,
            &opts,
        )
        .is_err());
        // A hand-built state whose trapped rows miss the trap count is
        // rejected at solve time even though it type-checks.
        let bad_trapped = InitialState {
            mobile: vec![0.0; 8],
            trapped: vec![vec![]; 8],
        };
        let one_trap = TransportParams::new(
            1e-3,
            8,
            1e-9,
            0.0,
            vec![crate::TrapSpec::new(0.05, 0.0, 0.01, 0.0, 2.0).unwrap()],
            vec![500.0],
            vec![],
        )
        .unwrap();
        assert!(InitialState::uniform(&one_trap, 1.0, vec![]).is_err());
        assert!(solve(
            &one_trap,
            &Boundary::ZeroFlux,
            &Boundary::ZeroFlux,
            &TimeGrid::new(vec![1.0]).unwrap(),
            &bad_trapped,
            &SolverOptions::default(),
        )
        .is_err());
    }

    #[test]
    fn step_budget_and_picard_failure_surface() {
        let p = no_traps(1e-3, 8, 1e-9);
        let init = InitialState::zeros(&p);
        // Four internal steps needed, one allowed.
        let budgeted = SolverOptions {
            dt_max: 0.5,
            max_steps: 1,
            ..Default::default()
        };
        assert_eq!(
            solve(
                &p,
                &Boundary::dirichlet(1.0).unwrap(),
                &Boundary::dirichlet(0.0).unwrap(),
                &TimeGrid::new(vec![1.0, 2.0]).unwrap(),
                &init,
                &budgeted,
            ),
            Err(Error::StepBudget(1))
        );
        // A bounded Picard 2-cycle defeats the iteration cap: with
        // dt·k·N = 1.5 and dt·k = 0.5 the trap map is f(c) = 1.5c/(1+0.5c)
        // and the sealed uniform slab iterates h(c) = 1 − f(c), whose exact
        // {0, 1} 2-cycle never meets rtol/atol.
        let stiff = TransportParams::new(
            1e-3,
            8,
            1e-9,
            0.0,
            vec![crate::TrapSpec::new(0.5, 0.0, 1e-12, 0.0, 3.0).unwrap()],
            vec![500.0],
            vec![],
        )
        .unwrap();
        let init_s = InitialState::uniform(&stiff, 1.0, vec![0.0]).unwrap();
        assert_eq!(
            solve(
                &stiff,
                &Boundary::ZeroFlux,
                &Boundary::ZeroFlux,
                &TimeGrid::new(vec![1.0]).unwrap(),
                &init_s,
                &SolverOptions {
                    dt_max: 1.0,
                    rtol: 1e-12,
                    atol: 1e-15,
                    ..Default::default()
                },
            ),
            Err(Error::NotConverged)
        );
    }
}
