//! Fixture-pinned oracle gates (`fixtures/tritium/`).
//!
//! These tests replay the hand-built synthetic fixtures through the public
//! API; they fail if the implementation drifts from the recorded
//! closed-form values.

use crate::{
    breakthrough_ratio, effective_diffusivity, equilibrium_trapped, irreversible_fill,
    sieverts_concentration, steady_state, time_lag, Boundary, InitialState, Solution,
    SolverOptions, Theta, TimeGrid, TransportParams, TrapSpec,
};

fn fixture(name: &str) -> serde_json::Value {
    let path = format!(
        "{}/../../fixtures/tritium/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn vec_of(v: &serde_json::Value, key: &str) -> Vec<f64> {
    serde_json::from_value(v[key].clone()).unwrap()
}

#[test]
fn g1_steady_linear_profile() {
    let v = fixture("g1_steady.json");
    let tol: f64 = serde_json::from_value(v["tolerance"].clone()).unwrap();
    let p = TransportParams::new(
        v["params"]["L"].as_f64().unwrap(),
        32,
        v["params"]["D"].as_f64().unwrap(),
        0.0,
        vec![],
        vec![500.0],
        vec![],
    )
    .unwrap();
    let s = steady_state(
        &p,
        &Boundary::dirichlet(v["params"]["c0"].as_f64().unwrap()).unwrap(),
        &Boundary::dirichlet(v["params"]["cL"].as_f64().unwrap()).unwrap(),
    )
    .unwrap();
    let xs = vec_of(&v, "x_over_L");
    let want = vec_of(&v, "c");
    for (i, c) in s.mobile.iter().enumerate() {
        // Cell centres sit half a cell off the fixture nodes: compare
        // against the closed form, not the node vector.
        let x = (i as f64 + 0.5) / s.mobile.len() as f64;
        let closed = 1.0 - x;
        assert!((c - closed).abs() < tol, "cell {i}: {c} vs {closed}");
    }
    // Fixture nodes land exactly on the closed form too.
    for (x, w) in xs.iter().zip(&want) {
        assert!(((1.0 - x) - w).abs() < tol);
    }
    let jss = v["J_ss"].as_f64().unwrap();
    // Through-flux runs left → right: outward-left is −J_ss (inflow),
    // outward-right is +J_ss (outflow).
    assert!((s.flux_left + jss).abs() < tol, "{}", s.flux_left);
    assert!((s.flux_right - jss).abs() < tol, "{}", s.flux_right);
    assert!((s.inventory_mobile - v["inventory"].as_f64().unwrap()).abs() < tol);
    assert!(s.mobile.iter().all(|c| *c >= 0.0));
}

#[test]
fn g2_timelag_and_breakthrough_curve() {
    let v = fixture("g2_timelag.json");
    let tol: f64 = serde_json::from_value(v["tolerance"].clone()).unwrap();
    let d = v["params"]["D"].as_f64().unwrap();
    let l = v["params"]["L"].as_f64().unwrap();
    let c0 = v["params"]["c0"].as_f64().unwrap();
    // Algebraic leg: t_lag = L^2/6D to 1e-12 against the recorded value.
    let lag = time_lag(l, d).unwrap();
    assert!((lag - v["t_lag"].as_f64().unwrap()).abs() < 1e-12);
    // Series helper reproduces the recorded curve far inside the gate.
    let ts = vec_of(&v, "t");
    let want = vec_of(&v, "J_over_Jss");
    for (t, w) in ts.iter().zip(&want) {
        let got = breakthrough_ratio(d, l, *t).unwrap();
        assert!(
            (got - w).abs() <= 1e-12 * w.abs().max(1.0),
            "t={t}: {got} vs {w}"
        );
    }
    // Solver leg: transient replay at the pinned 1e-6 tolerance.
    let cells = 1000;
    let p = TransportParams::new(l, cells, d, 0.0, vec![], vec![500.0], vec![]).unwrap();
    let left = Boundary::dirichlet(c0).unwrap();
    let right = Boundary::dirichlet(0.0).unwrap();
    let init = InitialState::zeros(&p);
    let opts = SolverOptions {
        dt_max: 0.1,
        rtol: 1e-10,
        atol: 1e-14,
        ..Default::default()
    };
    let sol: Solution = crate::solve::solve(
        &p,
        &left,
        &right,
        &TimeGrid::new(ts.clone()).unwrap(),
        &init,
        &opts,
    )
    .unwrap();
    let jss = d * c0 / l;
    for (k, w) in want.iter().enumerate() {
        let got = sol.flux_right[k] / jss;
        assert!(
            (got - w).abs() <= tol * w.abs().max(1.0),
            "t={}: {got} vs {w}",
            ts[k]
        );
    }
    // Breakthrough intercept: cumulative permeation asymptotes to
    // J_ss (t − t_lag). The seven fixture times are too coarse for the
    // quadrature, so a dedicated dense-grid run carries this leg.
    let dense_times: Vec<f64> = (1..=300).map(|k| 5.0 * k as f64).collect();
    let dense = crate::solve::solve(
        &p,
        &left,
        &right,
        &TimeGrid::new(dense_times.clone()).unwrap(),
        &InitialState::zeros(&p),
        &SolverOptions {
            dt_max: 0.5,
            rtol: 1e-10,
            atol: 1e-14,
            ..Default::default()
        },
    )
    .unwrap();
    let mut cum = 0.0;
    let mut prev_t = 0.0;
    let mut prev_j = 0.0;
    for (k, t) in dense_times.iter().enumerate() {
        cum += 0.5 * (prev_j + dense.flux_right[k]) * (t - prev_t);
        prev_t = *t;
        prev_j = dense.flux_right[k];
    }
    let lag_est = dense_times[dense_times.len() - 1] - cum / jss;
    assert!(
        (lag_est - lag).abs() / lag < 0.02,
        "intercept {lag_est} vs {lag}"
    );
    // Positivity on the transient rows.
    for row in &sol.mobile {
        assert!(row.iter().all(|c| *c >= 0.0));
    }
}

#[test]
fn g2_backward_euler_agrees_on_tail() {
    // Method option: backward Euler lands on the same late-time flux.
    let v = fixture("g2_timelag.json");
    let d = v["params"]["D"].as_f64().unwrap();
    let l = v["params"]["L"].as_f64().unwrap();
    let p = TransportParams::new(l, 200, d, 0.0, vec![], vec![500.0], vec![]).unwrap();
    let left = Boundary::dirichlet(1.0).unwrap();
    let right = Boundary::dirichlet(0.0).unwrap();
    let init = InitialState::zeros(&p);
    let grid = TimeGrid::new(vec![316.227_766_016_837_96, 1000.0]).unwrap();
    let cn = crate::solve::solve(
        &p,
        &left,
        &right,
        &grid,
        &init,
        &SolverOptions {
            dt_max: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    let be = crate::solve::solve(
        &p,
        &left,
        &right,
        &grid,
        &init,
        &SolverOptions {
            theta: Theta::BackwardEuler,
            dt_max: 0.5,
            ..Default::default()
        },
    )
    .unwrap();
    for (a, b) in cn.flux_right.iter().zip(&be.flux_right) {
        assert!((a - b).abs() / a < 1e-3, "{a} vs {b}");
    }
}

#[test]
fn g3a_oriani_effective_diffusivity() {
    let v = fixture("g3a_oriani.json");
    let tol: f64 = serde_json::from_value(v["tolerance"].clone()).unwrap();
    let d = v["params"]["D"].as_f64().unwrap();
    let k = v["params"]["K"].as_f64().unwrap();
    let n = v["params"]["N"].as_f64().unwrap();
    let got = effective_diffusivity(d, k, n).unwrap();
    assert!((got - v["D_eff"].as_f64().unwrap()).abs() < tol);
    assert!((d / got - v["retardation"].as_f64().unwrap()).abs() < tol);
    // The fixture regime really is low-occupancy: K c_ref << 1.
    let kc: f64 = serde_json::from_value(v["Kc_ref"].clone()).unwrap();
    assert!(kc < 1e-2);
}

#[test]
fn g3b_saturated_trapped_profile_and_inventory() {
    let v = fixture("g3b_saturated.json");
    let tol: f64 = serde_json::from_value(v["tolerance"].clone()).unwrap();
    let l = v["params"]["L"].as_f64().unwrap();
    let d = v["params"]["D"].as_f64().unwrap();
    let k = v["params"]["K"].as_f64().unwrap();
    let n = v["params"]["N"].as_f64().unwrap();
    // Isotherm helper reproduces the recorded trapped nodes.
    for (cm, wt) in vec_of(&v, "c_mobile").iter().zip(vec_of(&v, "c_trapped")) {
        let got = equilibrium_trapped(n, k, *cm).unwrap();
        assert!((got - wt).abs() < tol, "{got} vs {wt}");
    }
    // Occupancies match K c/(1 + K c).
    for (cm, wo) in vec_of(&v, "c_mobile").iter().zip(vec_of(&v, "occupancy")) {
        assert!(((k * cm / (1.0 + k * cm)) - wo).abs() < tol);
    }
    // Steady solve with a saturated trap: mobile is G1, trapped follows
    // the isotherm pointwise, inventories match the closed forms.
    let trap = TrapSpec::new(k, 0.0, 1.0, 0.0, n).unwrap();
    let p = TransportParams::new(l, 64, d, 0.0, vec![trap], vec![500.0], vec![]).unwrap();
    let s = steady_state(
        &p,
        &Boundary::dirichlet(v["params"]["c0"].as_f64().unwrap()).unwrap(),
        &Boundary::dirichlet(v["params"]["cL"].as_f64().unwrap()).unwrap(),
    )
    .unwrap();
    assert!((s.inventory_mobile - v["inventory_mobile"].as_f64().unwrap()).abs() < tol);
    // The 64-cell midpoint rule on the smooth isotherm lands inside 1e-6
    // of the exact 2L(1 − ln 2); the algebraic pin above holds 1e-12.
    assert!(
        (s.inventory_trapped - v["inventory_trapped"].as_f64().unwrap()).abs() < 1e-6,
        "{}",
        s.inventory_trapped
    );
    assert!(s.mobile.iter().all(|c| *c >= 0.0));
    assert!(s.trapped.iter().all(|row| row[0] >= 0.0 && row[0] <= n));
}

#[test]
fn g3c_irreversible_fill_and_positivity() {
    let v = fixture("g3c_irreversible.json");
    let tol: f64 = serde_json::from_value(v["tolerance"].clone()).unwrap();
    let k = v["params"]["k"].as_f64().unwrap();
    let c = v["params"]["c"].as_f64().unwrap();
    let n = v["params"]["N"].as_f64().unwrap();
    for (t, w) in vec_of(&v, "t").iter().zip(vec_of(&v, "c_trapped")) {
        let got = irreversible_fill(k, c, n, *t).unwrap();
        assert!((got - w).abs() < tol, "t={t}: {got} vs {w}");
    }
    // Solver leg: uniform mobile held by zero-flux ends fills traps
    // monotonically inside [0, N] with mobile staying nonneg.
    let trap = TrapSpec::new(k, 0.0, 1e-30, 0.0, n).unwrap();
    let p = TransportParams::new(1e-3, 8, 1e-9, 0.0, vec![trap], vec![500.0], vec![]).unwrap();
    let init = InitialState::uniform(&p, c, vec![0.0]).unwrap();
    let sol = crate::solve::solve(
        &p,
        &Boundary::ZeroFlux,
        &Boundary::ZeroFlux,
        &TimeGrid::new(vec![10.0, 50.0, 200.0]).unwrap(),
        &init,
        &SolverOptions {
            theta: Theta::BackwardEuler,
            dt_max: 1.0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut prev = 0.0;
    for rows in &sol.trapped {
        for row in rows {
            assert!(row[0] >= 0.0 && row[0] <= n);
            assert!(row[0] >= prev - 1e-12, "fill must not decrease");
            prev = row[0];
        }
    }
    for row in &sol.mobile {
        assert!(row.iter().all(|m| *m >= 0.0));
    }
}

#[test]
fn g4_sieverts_steady_state() {
    let v = fixture("g4_sieverts.json");
    let tol: f64 = serde_json::from_value(v["tolerance"].clone()).unwrap();
    let d = v["params"]["D"].as_f64().unwrap();
    let l = v["params"]["L"].as_f64().unwrap();
    let ks = v["params"]["K_S"].as_f64().unwrap();
    let p1 = v["params"]["p1"].as_f64().unwrap();
    let p2 = v["params"]["p2"].as_f64().unwrap();
    // Sieverts ends map to the recorded surface concentrations.
    let c0 = sieverts_concentration(ks, p1).unwrap();
    let cl = sieverts_concentration(ks, p2).unwrap();
    assert!((c0 - v["c0"].as_f64().unwrap()).abs() < tol);
    assert!((cl - v["cL"].as_f64().unwrap()).abs() < tol);
    let p = TransportParams::new(l, 32, d, 0.0, vec![], vec![500.0], vec![]).unwrap();
    let s = steady_state(
        &p,
        &Boundary::sieverts(ks, p1).unwrap(),
        &Boundary::sieverts(ks, p2).unwrap(),
    )
    .unwrap();
    let xs = vec_of(&v, "x_over_L");
    let want = vec_of(&v, "c");
    for (i, c) in s.mobile.iter().enumerate() {
        let x = (i as f64 + 0.5) / s.mobile.len() as f64;
        assert!((c - (c0 + (cl - c0) * x)).abs() < tol, "cell {i}: {c}");
    }
    for (x, w) in xs.iter().zip(&want) {
        assert!(((c0 + (cl - c0) * x) - w).abs() < tol);
    }
    assert!((s.flux_right - v["J"].as_f64().unwrap()).abs() < tol);
    assert!((s.inventory_mobile - v["inventory"].as_f64().unwrap()).abs() < tol);
    // Permeability Φ = D K_S recovers the flux from the root pressures.
    let phi = Boundary::sieverts(ks, p1).unwrap().permeability(d).unwrap();
    assert!((phi - v["permeability"].as_f64().unwrap()).abs() < tol);
    assert!((phi * (p1.sqrt() - p2.sqrt()) / l - v["J"].as_f64().unwrap()).abs() < 1e-17);
}

#[test]
fn mass_conservation_to_roundoff() {
    // Invariant: d/dt I = F_in(left) − F_out(right) + ∫S dx. Three legs,
    // each exact for the discrete scheme (no quadrature in the gate):
    // (a) zero-flux ends + nonuniform initial profile: diffusion only
    // redistributes, so the inventory is constant to roundoff;
    // (b) zero-flux ends + uniform source: I(t) = I0 + S L t;
    // (c) Dirichlet ends held on the G1 profile: net flux zero, inventory
    // constant to the stepper tolerance.
    let p = TransportParams::new(1e-3, 32, 1e-9, 0.0, vec![], vec![500.0], vec![]).unwrap();
    let opts = SolverOptions {
        dt_max: 0.5,
        ..Default::default()
    };
    // (a) Sine bump between impermeable walls.
    let bump: Vec<f64> = (0..32)
        .map(|i| 1.0 + 0.5 * (std::f64::consts::PI * (i as f64 + 0.5) / 32.0).sin())
        .collect();
    let i0: f64 = bump.iter().sum::<f64>() * p.dx();
    let init = InitialState::new(&p, bump, vec![vec![]; 32]).unwrap();
    let times = vec![5.0, 20.0, 80.0];
    let sol = crate::solve::solve(
        &p,
        &Boundary::ZeroFlux,
        &Boundary::ZeroFlux,
        &TimeGrid::new(times).unwrap(),
        &init,
        &opts,
    )
    .unwrap();
    for (k, row) in sol.mobile.iter().enumerate() {
        assert!(row.iter().all(|c| *c >= 0.0), "row {k} positivity");
    }
    for inv in sol.inventory_total() {
        assert!((inv - i0).abs() < 1e-12 * i0, "{inv} vs {i0}");
    }
    assert!(sol.flux_left.iter().all(|f| *f == 0.0));
    assert!(sol.flux_right.iter().all(|f| *f == 0.0));
    // (b) Uniform source fills the sealed slab linearly.
    let ps = TransportParams::new(1e-3, 16, 1e-9, 0.0, vec![], vec![500.0], vec![2.0]).unwrap();
    let init_s = InitialState::uniform(&ps, 1.0, vec![]).unwrap();
    let sol_s = crate::solve::solve(
        &ps,
        &Boundary::ZeroFlux,
        &Boundary::ZeroFlux,
        &TimeGrid::new(vec![10.0, 40.0]).unwrap(),
        &init_s,
        &opts,
    )
    .unwrap();
    for (k, t) in [10.0, 40.0].iter().enumerate() {
        let want = 1.0 * 1e-3 + 2.0 * 1e-3 * t;
        assert!((sol_s.inventory_total()[k] - want).abs() < 1e-12 * want);
    }
    // (c) G1 profile held by matching Dirichlet ends.
    let s = steady_state(
        &p,
        &Boundary::dirichlet(1.0).unwrap(),
        &Boundary::dirichlet(0.0).unwrap(),
    )
    .unwrap();
    let init_h = InitialState::new(&p, s.mobile.clone(), vec![vec![]; 32]).unwrap();
    let sol_h = crate::solve::solve(
        &p,
        &Boundary::dirichlet(1.0).unwrap(),
        &Boundary::dirichlet(0.0).unwrap(),
        &TimeGrid::new(vec![10.0, 50.0]).unwrap(),
        &init_h,
        &opts,
    )
    .unwrap();
    for (k, inv) in sol_h.inventory_total().iter().enumerate() {
        assert!(
            (inv - s.inventory_mobile).abs() < 1e-9 * s.inventory_mobile,
            "row {k}"
        );
        assert!((sol_h.flux_left[k] + sol_h.flux_right[k]).abs() < 1e-12);
    }
}
