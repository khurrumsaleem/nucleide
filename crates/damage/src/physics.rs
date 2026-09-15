//! Closed-form primary-damage physics: the Lindhard energy partition
//! (Robinson fit), the NRT displacement function, and the arc-dpa
//! efficiency correction.
//!
//! All equations are implemented clean-room from the published literature;
//! no code or tables are ported from other implementations:
//!
//! - **NRT-dpa** — Norgett, Robinson & Torrens, Nucl. Eng. Des. 33 (1975)
//!   50–54, DOI 10.1016/0029-5493(75)90035-7: modified Kinchin–Pease with
//!   the displacement efficiency κ = 0.8 and the Lindhard damage energy.
//! - **Lindhard partition, Robinson fit** — Lindhard, Nielsen, Scharff &
//!   Thomsen, Kgl. Danske Vid. Selsk. Mat. Fys. Medd. 33(10) (1963), as
//!   fitted by Robinson (1970); constants quoted here follow Griffin,
//!   SAND2016-2269 §3.2 (US-gov PD), which also documents the NJOY manual's
//!   corrected `k_L` mass exponent (3/2, not 2/3).
//! - **arc-dpa** — Nordlund et al., Nat. Commun. 9 (2018) 1084
//!   (CC BY 4.0), DOI 10.1038/s41467-018-03415-5, Eqs. (5)–(7): the
//!   piecewise NRT form multiplied by the efficiency function
//!   ξ(T_d) = (1 − c)·(T_d/(2E_d/κ))^b + c fitted to MD cascade data.
//!
//! Conventions (documented, tested):
//!
//! - Energies are electron-volts (`*_ev` suffixes) throughout; the spectral
//!   folds in [`crate::fold`] use MeV group bounds because neutron group
//!   structures are tabulated in MeV.
//! - The damage energy `T_dam` is the Lindhard partition evaluated at the
//!   PKA energy (the SPECTER/NJOY convention, lower integration bound zero:
//!   energy below `E_d` still counts toward damage energy, it just does not
//!   displace — see the LI discussion in Griffin SAND2016-2269 §2.5).
//! - The piecewise boundaries sit on the **PKA energy** `T` at `E_d` and
//!   `2·E_d/κ`, while the high branch evaluates `κ·T_dam/(2·E_d)` on the
//!   **damage energy** (Nordlund 2018, Eq. (1) and Eq. (5) verbatim). The
//!   small downward step at `T = 2E_d/κ` (where `T_dam < T`) is the known
//!   NRT construction, not a bug.
//! - Everything is a pure function over caller-supplied material constants
//!   (`E_d`, `b_arc`, `c_arc`) and nuclide keys; no evaluated data, no
//!   tables, no PKA-spectrum solving (explicitly OUT of scope: the
//!   fispact-org PKA evaluator is GPL-3.0 and is never read).

use nucleide_nuclei::NuclideId;

use crate::error::{Error, Result};

/// NRT displacement efficiency κ (Nordlund 2018 Eq. (1); NRT 1975).
pub const NRT_EFFICIENCY: f64 = 0.8;

/// Coefficient of the Lindhard reduced-energy scale `E_L` (Robinson 1970,
/// in eV): `E_L = LINDHARD_E_COEFF · Z_R·Z_L·(Z_R^{2/3}+Z_L^{2/3})^{1/2}·(A_R+A_L)/A_L`.
pub const LINDHARD_E_COEFF: f64 = 30.724;

/// Coefficient of the Lindhard partition constant `k_L` (Robinson 1970):
/// `k_L = LINDHARD_K_COEFF · Z_R^{2/3}·Z_L^{1/2}·(A_R+A_L)^{3/2} /
/// ((Z_R^{2/3}+Z_L^{2/3})^{3/4}·A_R^{3/2}·A_L^{1/2})`.
pub const LINDHARD_K_COEFF: f64 = 0.0793;

/// Coefficient of the linear term of the Robinson `g(ε)` fit.
pub const ROBINSON_G1: f64 = 1.0;
/// Coefficient of the `ε^{1/6}` term of the Robinson `g(ε)` fit.
pub const ROBINSON_G16: f64 = 3.4008;
/// Coefficient of the `ε^{3/4}` term of the Robinson `g(ε)` fit.
pub const ROBINSON_G34: f64 = 0.40244;

/// arc-dpa efficiency constants (`b_arc`, `c_arc` of Nordlund 2018 Eq. (7)).
///
/// `b_arc < 0` is the MD-fitted regime (efficiency decreasing with damage
/// energy); `c_arc` is the high-energy saturation value in `(0, 1)`. Both
/// are material constants the caller supplies from MD or experiment; this
/// crate deliberately ships no fitted table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcParams {
    /// Power-law exponent (must be finite and negative).
    pub b_arc: f64,
    /// Saturation efficiency (must be finite, strictly between 0 and 1).
    pub c_arc: f64,
}

impl ArcParams {
    /// Validate and build arc-dpa efficiency constants.
    pub fn new(b_arc: f64, c_arc: f64) -> Result<Self> {
        if !b_arc.is_finite() || !c_arc.is_finite() {
            return Err(Error::NonFinite("arc constants"));
        }
        if b_arc >= 0.0 || !(0.0..1.0).contains(&c_arc) {
            return Err(Error::InvalidArcParams { b_arc, c_arc });
        }
        Ok(Self { b_arc, c_arc })
    }
}

/// Validate a damage-relevant nuclide key (any valid `NuclideId`; the
/// metastable state index is ignored — damage physics sees only Z and A).
fn checked_nuclide(id: &NuclideId) -> Result<()> {
    if !id.is_valid() {
        return Err(Error::BadNuclide(id.nucid()));
    }
    Ok(())
}

/// Robinson fit `g(ε) = ε + 3.4008·ε^{1/6} + 0.40244·ε^{3/4}` (P4).
fn robinson_g(eps: f64) -> f64 {
    ROBINSON_G1 * eps + ROBINSON_G16 * eps.powf(1.0 / 6.0) + ROBINSON_G34 * eps.powf(0.75)
}

/// Lindhard partition constant `k_L` (P2) for a recoil `(Z_R, A_R)` in a
/// lattice `(Z_L, A_L)`.
fn lindhard_k(recoil: &NuclideId, lattice: &NuclideId) -> f64 {
    let zr = recoil.z() as f64;
    let ar = recoil.a() as f64;
    let zl = lattice.z() as f64;
    let al = lattice.a() as f64;
    LINDHARD_K_COEFF * zr.powf(2.0 / 3.0) * zl.sqrt() * (ar + al).powf(1.5)
        / ((zr.powf(2.0 / 3.0) + zl.powf(2.0 / 3.0)).powf(0.75) * ar.powf(1.5) * al.sqrt())
}

/// Lindhard reduced-energy scale `E_L` in eV (P3) for a recoil `(Z_R, A_R)`
/// in a lattice `(Z_L, A_L)`.
fn lindhard_e_scale(recoil: &NuclideId, lattice: &NuclideId) -> f64 {
    let zr = recoil.z() as f64;
    let ar = recoil.a() as f64;
    let zl = lattice.z() as f64;
    let al = lattice.a() as f64;
    LINDHARD_E_COEFF * zr * zl * (zr.powf(2.0 / 3.0) + zl.powf(2.0 / 3.0)).sqrt() * (ar + al) / al
}

/// Lindhard partition fraction `P(ε) = 1/(1 + k_L·g(ε))` (P1): the fraction
/// of the PKA energy available for displacements after electronic losses.
///
/// `P → 1` as `T → 0` and `P → 0` as `T → ∞` (damage energy saturates at
/// `E_L/k_L`, the asymptote the in-crate tests pin independently).
pub fn lindhard_partition(t_ev: f64, recoil: &NuclideId, lattice: &NuclideId) -> Result<f64> {
    if !t_ev.is_finite() || t_ev < 0.0 {
        return Err(Error::Negative("pka_energy_ev"));
    }
    checked_nuclide(recoil)?;
    checked_nuclide(lattice)?;
    let eps = t_ev / lindhard_e_scale(recoil, lattice);
    Ok(1.0 / (1.0 + lindhard_k(recoil, lattice) * robinson_g(eps)))
}

/// Lindhard damage energy `T_dam = T·P(ε)` in eV (P1).
pub fn damage_energy(t_ev: f64, recoil: &NuclideId, lattice: &NuclideId) -> Result<f64> {
    Ok(t_ev * lindhard_partition(t_ev, recoil, lattice)?)
}

/// NRT displacement function `N_d(T)` (N1) for a self-recoil `target` with
/// average threshold displacement energy `ed_ev`:
///
/// ```text
/// N_d(T) = 0                    , T <  E_d
/// N_d(T) = 1                    , E_d <= T < 2·E_d/κ
/// N_d(T) = κ·T_dam(T)/(2·E_d)   , T >= 2·E_d/κ        with κ = 0.8
/// ```
pub fn nrt_displacements(t_ev: f64, ed_ev: f64, target: &NuclideId) -> Result<f64> {
    nrt_displacements_for(t_ev, ed_ev, target, target)
}

/// NRT displacement function for an arbitrary recoil/lattice pair (the
/// general SPECTER case, e.g. a transmutation recoil stopped in a host
/// lattice). Same piecewise form as [`nrt_displacements`].
pub fn nrt_displacements_for(
    t_ev: f64,
    ed_ev: f64,
    recoil: &NuclideId,
    lattice: &NuclideId,
) -> Result<f64> {
    validate_damage_inputs(t_ev, ed_ev)?;
    checked_nuclide(recoil)?;
    checked_nuclide(lattice)?;
    let branch = 2.0 * ed_ev / NRT_EFFICIENCY;
    if t_ev < ed_ev {
        return Ok(0.0);
    }
    if t_ev < branch {
        return Ok(1.0);
    }
    let t_dam = damage_energy(t_ev, recoil, lattice)?;
    Ok(NRT_EFFICIENCY * t_dam / (2.0 * ed_ev))
}

/// arc-dpa efficiency `ξ(T_d)` (A1, Nordlund 2018 Eq. (7)):
///
/// ```text
/// ξ(T_d) = (1 − c_arc)·(T_d/(2·E_d/κ))^{b_arc} + c_arc
/// ```
///
/// By construction `ξ(2E_d/κ) = 1` (junction with the middle branch) and
/// `ξ → c_arc` as `T_d → ∞` (MD saturation); `b_arc < 0` makes it monotone
/// decreasing. `T_d` is a damage energy in eV.
pub fn arc_efficiency(t_dam_ev: f64, ed_ev: f64, params: &ArcParams) -> Result<f64> {
    if !t_dam_ev.is_finite() || t_dam_ev < 0.0 {
        return Err(Error::Negative("t_dam_ev"));
    }
    if !ed_ev.is_finite() || ed_ev <= 0.0 {
        return Err(Error::NonPositive("ed_ev"));
    }
    let junction = 2.0 * ed_ev / NRT_EFFICIENCY;
    Ok((1.0 - params.c_arc) * (t_dam_ev / junction).powf(params.b_arc) + params.c_arc)
}

/// arc-dpa displacement function (A2, Nordlund 2018 Eq. (5)): the NRT
/// piecewise form with the high branch multiplied by `ξ_arc(T_dam)`:
///
/// ```text
/// N_d,arc(T) = 0                                  , T <  E_d
/// N_d,arc(T) = 1                                  , E_d <= T < 2·E_d/κ
/// N_d,arc(T) = κ·T_dam(T)·ξ_arc(T_dam)/(2·E_d)    , T >= 2·E_d/κ
/// ```
pub fn arc_displacements(
    t_ev: f64,
    ed_ev: f64,
    target: &NuclideId,
    params: &ArcParams,
) -> Result<f64> {
    arc_displacements_for(t_ev, ed_ev, target, target, params)
}

/// arc-dpa displacement function for an arbitrary recoil/lattice pair.
pub fn arc_displacements_for(
    t_ev: f64,
    ed_ev: f64,
    recoil: &NuclideId,
    lattice: &NuclideId,
    params: &ArcParams,
) -> Result<f64> {
    validate_damage_inputs(t_ev, ed_ev)?;
    checked_nuclide(recoil)?;
    checked_nuclide(lattice)?;
    let branch = 2.0 * ed_ev / NRT_EFFICIENCY;
    if t_ev < ed_ev {
        return Ok(0.0);
    }
    if t_ev < branch {
        return Ok(1.0);
    }
    let t_dam = damage_energy(t_ev, recoil, lattice)?;
    let xi = arc_efficiency(t_dam, ed_ev, params)?;
    Ok(NRT_EFFICIENCY * t_dam * xi / (2.0 * ed_ev))
}

fn validate_damage_inputs(t_ev: f64, ed_ev: f64) -> Result<()> {
    if !t_ev.is_finite() || t_ev < 0.0 {
        return Err(Error::Negative("pka_energy_ev"));
    }
    if !ed_ev.is_finite() || ed_ev <= 0.0 {
        return Err(Error::NonPositive("ed_ev"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic Fe-56 target (Z/A from the nucid key; no evaluated data).
    fn fe56() -> NuclideId {
        NuclideId::new(26, 56, 0).unwrap()
    }

    /// Hand-computed anchor: the damage-energy saturation limit of the
    /// Lindhard partition. As `T → ∞`, `g(ε) → ε`, so
    /// `T_dam → T/(1 + k_L·T/E_L) → E_L/k_L`. Dividing the published `E_L`
    /// and `k_L` expressions term by term gives a closed form that shares
    /// no intermediate with the implementation:
    ///
    /// ```text
    /// E_L/k_L = (30.724/0.0793) · Z_R^{1/3} · Z_L^{1/2}
    ///           · (Z_R^{2/3} + Z_L^{2/3})^{5/4}
    ///           · A_R^{3/2} / (A_L^{1/2} · (A_R + A_L)^{1/2})
    /// ```
    fn saturation_limit(recoil: &NuclideId, lattice: &NuclideId) -> f64 {
        let zr = recoil.z() as f64;
        let ar = recoil.a() as f64;
        let zl = lattice.z() as f64;
        let al = lattice.a() as f64;
        (LINDHARD_E_COEFF / LINDHARD_K_COEFF)
            * zr.powf(1.0 / 3.0)
            * zl.sqrt()
            * (zr.powf(2.0 / 3.0) + zl.powf(2.0 / 3.0)).powf(5.0 / 4.0)
            * ar.powf(1.5)
            / (al.sqrt() * (ar + al).sqrt())
    }

    #[test]
    fn partition_limits_are_pinned() {
        let fe = fe56();
        // P -> 1 as T -> 0, but only logarithmically (g(ε) ~ ε^{1/6}):
        // at 1 µeV the ε^{1/6} term already costs ~0.6%.
        let p = lindhard_partition(1.0e-6, &fe, &fe).unwrap();
        assert!((0.99..1.0).contains(&p), "P(1e-6 eV) = {p}");
        // Monotonic decrease over 12 decades.
        let mut prev = 1.0;
        for t in [1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12] {
            let p = lindhard_partition(t, &fe, &fe).unwrap();
            assert!(
                p < prev && p > 0.0,
                "partition not decreasing at T={t}: {p}"
            );
            prev = p;
        }
        // High-energy saturation: at 1e12 eV the ε^{3/4} correction to
        // g(ε) ≈ ε is below 1%, so T_dam must sit within 1% of E_L/k_L.
        let t_dam = damage_energy(1e12, &fe, &fe).unwrap();
        let limit = saturation_limit(&fe, &fe);
        assert!(
            (t_dam - limit).abs() / limit < 1e-2,
            "T_dam(1e12) = {t_dam}, saturation limit {limit}"
        );
        // Damage energy never exceeds the PKA energy.
        for t in [0.5, 100.0, 1e5, 1e8, 1e11] {
            assert!(damage_energy(t, &fe, &fe).unwrap() <= t);
        }
    }

    #[test]
    fn partition_matches_transcribed_spot() {
        // Self-recoil spot recomputed from the published constants in a
        // second spelling: Fe-56, T = 10 keV. ε = T/E_L with
        // E_L = 30.724·Z²·(2·Z^{2/3})^{1/2}·2 and k_L = 0.0793·2^{3/4}·Z^{2/3}/A^{1/2}.
        let fe = fe56();
        let t = 1.0e4;
        let z = 26.0f64;
        let a = 56.0f64;
        let e_l = LINDHARD_E_COEFF * z * z * (2.0 * z.powf(2.0 / 3.0)).sqrt() * 2.0;
        let k_l = LINDHARD_K_COEFF * 2.0f64.powf(0.75) * z.powf(2.0 / 3.0) / a.sqrt();
        let eps = t / e_l;
        let g = eps + ROBINSON_G16 * eps.powf(1.0 / 6.0) + ROBINSON_G34 * eps.powf(0.75);
        let want = 1.0 / (1.0 + k_l * g);
        let got = lindhard_partition(t, &fe, &fe).unwrap();
        // Bit-identical up to the rounding difference between the general
        // and self-recoil algebraic spellings of k_L/E_L.
        assert!(
            (got - want).abs() <= 1e-12 * want,
            "partition spot {got} vs {want}"
        );
        // And the damage energy keeps a sensible fraction at 10 keV.
        assert!((0.6..0.85).contains(&got), "P(10 keV, Fe) = {got}");
    }

    #[test]
    fn nrt_piecewise_boundaries() {
        let fe = fe56();
        let ed = 40.0;
        // Below threshold: no displacements.
        assert_eq!(nrt_displacements(0.0, ed, &fe).unwrap(), 0.0);
        assert_eq!(nrt_displacements(ed * 0.999, ed, &fe).unwrap(), 0.0);
        // Modified-Kinchin–Pease plateau: exactly one displacement.
        assert_eq!(nrt_displacements(ed, ed, &fe).unwrap(), 1.0);
        assert_eq!(nrt_displacements(2.0 * ed, ed, &fe).unwrap(), 1.0);
        // Just below the branch boundary the plateau still holds.
        let branch = 2.0 * ed / NRT_EFFICIENCY;
        assert_eq!(
            nrt_displacements(branch * (1.0 - 1e-12), ed, &fe).unwrap(),
            1.0
        );
        // On the high branch the value equals κ·T_dam/(2·E_d) with the
        // Lindhard partition applied — recomputed independently here.
        let t = 1.0e5;
        let t_dam = damage_energy(t, &fe, &fe).unwrap();
        assert_eq!(
            nrt_displacements(t, ed, &fe).unwrap(),
            NRT_EFFICIENCY * t_dam / (2.0 * ed)
        );
        // The high branch is continuous from above in T (no jumps).
        let a = nrt_displacements(1.0e7, ed, &fe).unwrap();
        let b = nrt_displacements(1.0e7 * (1.0 + 1e-9), ed, &fe).unwrap();
        assert!((a - b).abs() / a < 1e-6);
    }

    #[test]
    fn arc_efficiency_matches_construction() {
        let fe = fe56();
        let ed = 40.0;
        let params = ArcParams::new(-0.55, 0.3).unwrap();
        let junction = 2.0 * ed / NRT_EFFICIENCY;
        // ξ(2E_d/κ) = 1 exactly (the continuity construction).
        assert_eq!(arc_efficiency(junction, ed, &params).unwrap(), 1.0);
        // Saturation: ξ → c_arc at high damage energy.
        let xi_hi = arc_efficiency(1.0e9, ed, &params).unwrap();
        assert!((xi_hi - params.c_arc).abs() < 1e-3, "xi(1 GeV) = {xi_hi}");
        // Monotone decreasing on a sweep (b_arc < 0).
        let mut prev = f64::INFINITY;
        for t_d in [1.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7] {
            let xi = arc_efficiency(t_d, ed, &params).unwrap();
            assert!(xi < prev, "efficiency not decreasing at T_d={t_d}");
            prev = xi;
        }
        // The arc damage function is N_d,NRT × ξ on the high branch.
        let t = 1.0e5;
        let t_dam = damage_energy(t, &fe, &fe).unwrap();
        let want =
            NRT_EFFICIENCY * t_dam * arc_efficiency(t_dam, ed, &params).unwrap() / (2.0 * ed);
        assert_eq!(arc_displacements(t, ed, &fe, &params).unwrap(), want);
        // Below threshold and on the plateau the two models agree exactly.
        assert_eq!(arc_displacements(ed * 0.5, ed, &fe, &params).unwrap(), 0.0);
        assert_eq!(arc_displacements(ed * 1.5, ed, &fe, &params).unwrap(), 1.0);
        // arc-dpa never exceeds NRT-dpa by more than the junction overshoot:
        // on the high branch arc/nrt = ξ(T_dam(T)), ξ is monotone decreasing
        // and T_dam is increasing, so the ratio peaks at the junction
        // T = 2E_d/κ (where it exceeds 1 because T_dam < T there).
        let overshoot = arc_efficiency(
            damage_energy(2.0 * ed / NRT_EFFICIENCY, &fe, &fe).unwrap(),
            ed,
            &params,
        )
        .unwrap();
        assert!(overshoot > 1.0, "junction overshoot = {overshoot}");
        for t in [2.0 * ed / NRT_EFFICIENCY, 1e5, 1e7, 1e9] {
            let nrt = nrt_displacements(t, ed, &fe).unwrap();
            let arc = arc_displacements(t, ed, &fe, &params).unwrap();
            assert!(
                arc <= nrt * overshoot * (1.0 + 1e-12),
                "arc {arc} exceeds NRT {nrt} × {overshoot} at T={t}"
            );
        }
    }

    #[test]
    fn bad_inputs_name_their_cause() {
        let fe = fe56();
        assert!(matches!(
            nrt_displacements(-1.0, 40.0, &fe),
            Err(Error::Negative("pka_energy_ev"))
        ));
        assert!(matches!(
            nrt_displacements(f64::NAN, 40.0, &fe),
            Err(Error::Negative("pka_energy_ev"))
        ));
        assert!(matches!(
            nrt_displacements(100.0, 0.0, &fe),
            Err(Error::NonPositive("ed_ev"))
        ));
        assert!(matches!(
            lindhard_partition(100.0, &NuclideId::from_nucid(999_999_999), &fe),
            Err(Error::BadNuclide(_))
        ));
        assert!(matches!(
            ArcParams::new(0.5, 0.3),
            Err(Error::InvalidArcParams { .. })
        ));
        assert!(matches!(
            ArcParams::new(-0.5, 1.2),
            Err(Error::InvalidArcParams { .. })
        ));
        assert!(matches!(
            ArcParams::new(f64::NAN, 0.3),
            Err(Error::NonFinite("arc constants"))
        ));
        assert!(matches!(
            arc_efficiency(-1.0, 40.0, &ArcParams::new(-0.5, 0.3).unwrap()),
            Err(Error::Negative("t_dam_ev"))
        ));
    }
}
