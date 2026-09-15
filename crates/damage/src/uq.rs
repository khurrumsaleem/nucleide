//! Uncertainty propagation through the spectral folds over caller-supplied
//! MVN blocks — the UQ-lite hook, gated like the landed U1–U4/U7 pattern.
//!
//! [`fold_uq`] perturbs the stacked `[flux, response]` vector with seeded
//! MVN draws from `linalg::sample::sample_mvn` (Cholesky primary,
//! eigen-clip fallback, pinned `seed` → bit-identical streams) and refolds
//! the metric per draw. Deltas are **relative-unit** perturbations applied
//! as `nominal·(1 + δ)` (the SANDY convention); a draw leaving the physical
//! domain (negative flux or response) surfaces as the fold's named error,
//! never a silent clip — callers keep covariances in the small-perturbation
//! regime where this cannot trigger.
//!
//! The linear folds (dpa, appm) admit moment analytics the gate compares
//! at `k` standard errors:
//!
//! - `E[m] = m₀ + scale·(fᵀμr + rᵀμf + μfᵀμr + Σ_g cov[g][G+g])` — exact
//!   for the bilinear fold of a Gaussian (`E[δf_g·δr_g] = μf_g·μr_g +
//!   cov[g][G+g]`).
//! - `std[m] = scale·sqrt(J·C·Jᵀ)` with `J = [(f⊙r)⊙(1+μr) | (f⊙r)⊙(1+μf)]`
//!   — the first-order propagation about the block mean (for relative
//!   perturbations the per-group sensitivity is the nominal product
//!   `f_g·r_g` times the other block's mean shift); for small relative
//!   blocks the neglected quartic term sits orders of magnitude inside the
//!   gate width (the in-crate tests size it at ~3e-4 of the linear term).
//!
//! No new sampling machinery lives here: `sample_mvn` and the perturbation
//! conventions are `linalg`'s.

use nucleide_linalg::sample::{apply_perturbation, sample_mvn, PerturbConvention};

use crate::error::{Error, Result};
use crate::fold::{fold_scaled, BARNS_TO_CM2};

/// Which spectral fold the UQ sweep refolds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoldMetric {
    /// [`crate::fold::nrt_dpa`].
    NrtDpa,
    /// [`crate::fold::arc_dpa`].
    ArcDpa,
    /// [`crate::fold::gas_appm`].
    GasAppm,
    /// [`crate::fold::he_dpa_ratio`] — not a linear fold; UQ on the ratio
    /// needs a delta-method (or sampling) treatment of a nonlinear
    /// statistic, which is out of the v1 scope.
    HeDpaRatio,
}

impl FoldMetric {
    /// Stable short name for reports and the Python facade.
    pub fn name(&self) -> &'static str {
        match self {
            FoldMetric::NrtDpa => "nrt_dpa",
            FoldMetric::ArcDpa => "arc_dpa",
            FoldMetric::GasAppm => "gas_appm",
            FoldMetric::HeDpaRatio => "he_dpa_ratio",
        }
    }
}

/// Convergence report from [`fold_uq`].
#[derive(Debug, Clone, PartialEq)]
pub struct UqSummary {
    /// Metric being propagated.
    pub metric: FoldMetric,
    /// Nominal (unperturbed) metric value.
    pub nominal: f64,
    /// Sample mean over the `n` perturbed folds.
    pub mean: f64,
    /// Unbiased sample standard deviation over the draws.
    pub std: f64,
    /// Exact expectation of the bilinear fold under the Gaussian block.
    pub expected: f64,
    /// First-order propagated standard deviation.
    pub analytic_std: f64,
    /// Gate width multiplier supplied by the caller.
    pub k: f64,
    /// Draw count.
    pub n: usize,
    /// Pinned seed that produced the draws.
    pub seed: u64,
    /// Both moment gates passed at `k` standard errors.
    pub passed: bool,
}

/// Scale factor mapping the raw fold sum to the metric's units.
fn metric_scale(metric: FoldMetric, seconds: f64) -> f64 {
    match metric {
        FoldMetric::NrtDpa | FoldMetric::ArcDpa => BARNS_TO_CM2 * seconds,
        FoldMetric::GasAppm => 1.0e-18 * seconds,
        FoldMetric::HeDpaRatio => 1.0,
    }
}

/// Propagate caller-block uncertainty through one spectral fold.
///
/// `mean_delta`/`cov` describe relative perturbations of the stacked
/// `[flux, response]` vector (dimension `2G`). Draws are seeded and
/// reproducible; the summary carries both the sample moments and the
/// analytic propagation, with the `k`-standard-error gates evaluated and
/// echoed back. `FoldMetric::HeDpaRatio` is a loud
/// [`Error::NotYetSupported`].
#[allow(clippy::too_many_arguments)] // the fold tuple plus the UQ block is the natural call shape
pub fn fold_uq(
    metric: FoldMetric,
    flux: &[f64],
    response: &[f64],
    bounds: &[f64],
    seconds: f64,
    mean_delta: &[f64],
    cov: &[Vec<f64>],
    n: usize,
    seed: u64,
    k: f64,
) -> Result<UqSummary> {
    if matches!(metric, FoldMetric::HeDpaRatio) {
        return Err(Error::NotYetSupported(
            "UQ on the He/dpa ratio (nonlinear statistic; delta method out of v1 scope)",
        ));
    }
    if !k.is_finite() || k <= 0.0 {
        return Err(Error::NonPositive("k"));
    }
    let g = flux.len();
    if g == 0 {
        return Err(Error::Empty);
    }
    if mean_delta.len() != 2 * g {
        return Err(Error::DimensionMismatch {
            what: "mean_delta",
            expected: 2 * g,
            got: mean_delta.len(),
        });
    }
    // Validate the nominal fold up front (bounds, signs, seconds).
    let is_appm = matches!(metric, FoldMetric::GasAppm);
    let nominal = fold_scaled(flux, response, bounds, seconds, is_appm)?;
    // Draw once up front: linalg validates the block (dimension, symmetry,
    // finiteness, PSD-adjacency) and pins the factorisation path before
    // any per-draw work happens.
    let set = sample_mvn(mean_delta, cov, n, seed)?;
    let scale = metric_scale(metric, seconds);

    // Exact expectation of the bilinear fold under δ ~ N(μ, C) with
    // relative perturbations: E[(f(1+δf))(r(1+δr))] = f·r·(1 + μf + μr +
    // μf·μr + cov_fr), since E[δf·δr] = μf·μr + cov_fr. The block shape is
    // guaranteed (2G)×(2G) by sample_mvn's validation above.
    let (mu_f, mu_r) = mean_delta.split_at(g);
    let mut expected = 0.0;
    for i in 0..g {
        expected +=
            flux[i] * response[i] * (1.0 + mu_f[i] + mu_r[i] + mu_f[i] * mu_r[i] + cov[i][g + i]);
    }
    expected *= scale;

    // First-order variance about the block mean. For relative perturbations
    // the sensitivity of the group product f(1+δf)·r(1+δr) to δf is
    // f·r·(1+μr) (and symmetrically for δr), so
    // J = scale·[(f⊙r)⊙(1+μr) | (f⊙r)⊙(1+μf)] and Var ≈ scale²·J·C·Jᵀ.
    let j_at = |i: usize| -> f64 {
        let g_i = i % g;
        let other_mean = if i < g { mu_r[g_i] } else { mu_f[g_i] };
        scale * flux[g_i] * response[g_i] * (1.0 + other_mean)
    };
    let mut jcj = 0.0;
    for (i, row) in cov.iter().enumerate() {
        let j_i = j_at(i);
        for (j, c_ij) in row.iter().enumerate() {
            jcj += j_i * c_ij * j_at(j);
        }
    }
    let analytic_std = jcj.max(0.0).sqrt();

    let mut draws: Vec<f64> = Vec::with_capacity(n);
    for delta in &set.samples {
        let (d_f, d_r) = delta.split_at(g);
        let pert_f = apply_perturbation(flux, d_f, PerturbConvention::Relative)?;
        let pert_r = apply_perturbation(response, d_r, PerturbConvention::Relative)?;
        draws.push(fold_scaled(&pert_f, &pert_r, bounds, seconds, is_appm)?);
    }

    // Unbiased sample moments (1/(n-1) variance), like linalg's estimators.
    let n_f = n as f64;
    let mean = draws.iter().sum::<f64>() / n_f;
    let std = if n < 2 {
        0.0
    } else {
        let var = draws.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n_f - 1.0);
        var.max(0.0).sqrt()
    };

    // Honest k-SE gates: the sample mean against the exact expectation, and
    // the sample standard deviation against the first-order propagation
    // (whose own normal-theory standard error is σ/√(2(n−1))).
    let mean_se = analytic_std / n_f.sqrt();
    let std_se = if n >= 2 {
        analytic_std / (2.0 * (n as f64 - 1.0)).sqrt()
    } else {
        0.0
    };
    let passed = (mean - expected).abs() <= k * mean_se && (std - analytic_std).abs() <= k * std_se;

    Ok(UqSummary {
        metric,
        nominal,
        mean,
        std,
        expected,
        analytic_std,
        k,
        n,
        seed,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 20260915;

    /// Synthetic two-group problem tuple: (flux, response, bounds, mean, cov).
    type Problem = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>);

    /// Synthetic two-group fold and a small relative block (1–5%
    /// standard deviations — the small-perturbation regime the gate
    /// documents). No evaluated data anywhere.
    fn problem() -> Problem {
        let flux = vec![1.0e13, 3.0e12];
        let resp = vec![80.0, 160.0];
        let bounds = vec![0.0, 0.5, 20.0];
        let mean = vec![0.0, 0.0, 0.0, 0.0];
        let cov = vec![
            vec![0.0004, 0.0, 0.0, 0.0],
            vec![0.0, 0.0009, 0.0, 0.0],
            vec![0.0, 0.0, 0.0025, 0.0],
            vec![0.0, 0.0, 0.0, 0.0001],
        ];
        (flux, resp, bounds, mean, cov)
    }

    #[test]
    fn mvn_uq_recovers_propagated_moments_within_k_se() {
        let (flux, resp, bounds, mean, cov) = problem();
        let n = 20_000usize;
        let k = 5.0;
        for metric in [FoldMetric::NrtDpa, FoldMetric::ArcDpa, FoldMetric::GasAppm] {
            let s = fold_uq(metric, &flux, &resp, &bounds, 2.0, &mean, &cov, n, SEED, k).unwrap();
            assert!(s.passed, "{metric:?}: {s:?}");
            assert_eq!(s.n, n);
            assert_eq!(s.seed, SEED);
            // With zero-mean block-diagonal perturbations the exact
            // expectation is the nominal fold itself.
            let nominal = fold_scaled(
                &flux,
                &resp,
                &bounds,
                2.0,
                matches!(metric, FoldMetric::GasAppm),
            )
            .unwrap();
            assert_eq!(s.nominal, nominal);
            assert_eq!(s.expected, nominal);
            // The reported moments sit inside the gate widths the summary
            // itself echoes (mean_se = analytic_std/sqrt(n), etc.).
            let mean_se = s.analytic_std / (n as f64).sqrt();
            assert!((s.mean - s.expected).abs() <= k * mean_se);
            let std_se = s.analytic_std / (2.0 * (n as f64 - 1.0)).sqrt();
            assert!((s.std - s.analytic_std).abs() <= k * std_se);
        }
        // Determinism under the pinned seed.
        let a = fold_uq(
            FoldMetric::NrtDpa,
            &flux,
            &resp,
            &bounds,
            2.0,
            &mean,
            &cov,
            64,
            SEED,
            5.0,
        )
        .unwrap();
        let b = fold_uq(
            FoldMetric::NrtDpa,
            &flux,
            &resp,
            &bounds,
            2.0,
            &mean,
            &cov,
            64,
            SEED,
            5.0,
        )
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn mean_bias_is_caught_exactly() {
        // A relative mean bias on the response block shifts E[m] by
        // scale·Σ f·μr — the analytic expectation must track it. The bias
        // is sized well above the f64 summation noise of the big terms.
        let (flux, resp, bounds, mut mean, cov) = problem();
        mean[2] = 0.5; // +50% response bias, group 1
        let s = fold_uq(
            FoldMetric::NrtDpa,
            &flux,
            &resp,
            &bounds,
            1.0,
            &mean,
            &cov,
            4_000,
            SEED,
            5.0,
        )
        .unwrap();
        let bias = 1.0e-24 * flux[0] * resp[0] * 0.5;
        assert!((s.expected - (s.nominal + bias)).abs() <= 1e-12 * s.nominal.abs());
        assert!(s.passed);
    }

    #[test]
    fn ratio_uq_is_a_loud_named_open() {
        let (flux, resp, bounds, mean, cov) = problem();
        assert!(matches!(
            fold_uq(
                FoldMetric::HeDpaRatio,
                &flux,
                &resp,
                &bounds,
                1.0,
                &mean,
                &cov,
                16,
                SEED,
                5.0
            ),
            Err(Error::NotYetSupported(_))
        ));
        assert_eq!(FoldMetric::HeDpaRatio.name(), "he_dpa_ratio");
    }

    #[test]
    fn bad_blocks_name_their_cause() {
        let (flux, resp, bounds, mean, cov) = problem();
        assert!(matches!(
            fold_uq(
                FoldMetric::NrtDpa,
                &flux,
                &resp,
                &bounds,
                1.0,
                &[0.0; 3],
                &cov,
                8,
                SEED,
                5.0
            ),
            Err(Error::DimensionMismatch {
                what: "mean_delta",
                ..
            })
        ));
        assert!(matches!(
            fold_uq(
                FoldMetric::NrtDpa,
                &flux,
                &resp,
                &bounds,
                1.0,
                &mean,
                &[vec![1.0]],
                8,
                SEED,
                5.0
            ),
            Err(Error::Sampling(
                nucleide_linalg::SampleError::DimensionMismatch { .. },
            ))
        ));
        assert!(matches!(
            fold_uq(
                FoldMetric::NrtDpa,
                &flux,
                &resp,
                &bounds,
                1.0,
                &mean,
                &cov,
                8,
                SEED,
                0.0
            ),
            Err(Error::NonPositive("k"))
        ));
    }
}
