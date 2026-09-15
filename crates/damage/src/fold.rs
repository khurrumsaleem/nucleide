//! Spectral folding: caller-supplied multigroup flux folded with
//! caller-supplied response functions into engineering damage/gas metrics.
//!
//! Folding convention (documented, tested — the crate does no spectral
//! weighting of its own):
//!
//! - `bounds` holds `G + 1` group boundaries in MeV, strictly increasing.
//!   Group `g` spans `[bounds[g], bounds[g+1])`. Bounds are validated and
//!   document the grid; the fold itself is piecewise-constant per group, so
//!   a group's width never enters the sum.
//! - `flux[g]` is the per-group integrated flux (n/cm²/s through group g),
//!   i.e. `∫_group φ(E) dE` — not a per-lethargy or per-unit-energy density.
//! - `response[g]` is the group response cross section in barns, condensed
//!   by the caller consistently with the flux spectrum (e.g. an NJOY/
//!   SPECTER-class pipeline for displacement cross sections, or an
//!   activation library for gas production). The kernel multiplies; it never
//!   re-weights within a group.
//! - The fold is `metric = scale · Σ_g flux[g]·response[g]` with
//!   `scale = 1e-24·seconds` for dpa (barns → cm², per target atom) and
//!   `scale = 1e-18·seconds` for gas appm (adds the per-atom → parts-per-
//!   million factor 1e6). `seconds` is the exposure time at the tabulated
//!   flux; pass `1.0` with a fluence in `flux` for a fluence fold.
//!
//! Edge conventions:
//!
//! - Zero-flux groups contribute exactly `0.0` — the kernel never divides
//!   by a group flux, so a dead group cannot produce `NaN`/`inf` even with
//!   a nonzero response.
//! - Negative flux or response is a loud named error, never clamped: such
//!   input means the caller's condensation already went wrong.
//! - The He/dpa ratio at exactly zero dpa is [`Error::ZeroDpa`], never
//!   `f64::INFINITY` (the `ZeroMaxFlux` precedent in `vr-tools` `magic`).
//!
//! What is deliberately NOT here: PKA-spectrum solving, displacement-table
//! vendoring, transport solving. Callers build `response` with their own
//! nuclear-data pipeline; the in-crate physics ([`crate::physics`]) supplies
//! the closed-form damage functions that pipeline would tabulate.

use crate::error::{Error, Result};

/// barns → cm².
pub const BARNS_TO_CM2: f64 = 1.0e-24;

/// Validate one response fold's shared shape and values.
///
/// Returns the group count `G`.
fn validate_fold(flux: &[f64], response: &[f64], bounds: &[f64], seconds: f64) -> Result<usize> {
    if flux.is_empty() {
        return Err(Error::Empty);
    }
    if response.len() != flux.len() {
        return Err(Error::DimensionMismatch {
            what: "response",
            expected: flux.len(),
            got: response.len(),
        });
    }
    if bounds.len() != flux.len() + 1 {
        return Err(Error::DimensionMismatch {
            what: "bounds",
            expected: flux.len() + 1,
            got: bounds.len(),
        });
    }
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(Error::NonPositive("seconds"));
    }
    if !flux.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite("flux"));
    }
    if !response.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite("response"));
    }
    if flux.iter().any(|v| *v < 0.0) {
        return Err(Error::Negative("flux"));
    }
    if response.iter().any(|v| *v < 0.0) {
        return Err(Error::Negative("response"));
    }
    if !bounds.iter().all(|v| v.is_finite()) {
        return Err(Error::NonFinite("bounds"));
    }
    for (i, w) in bounds.windows(2).enumerate() {
        if w[0] >= w[1] {
            return Err(Error::NonMonotonicBounds { index: i });
        }
    }
    Ok(flux.len())
}

/// Piecewise-constant-per-group fold with the caller's scale.
pub(crate) fn fold_scaled(
    flux: &[f64],
    response: &[f64],
    bounds: &[f64],
    seconds: f64,
    appm: bool,
) -> Result<f64> {
    validate_fold(flux, response, bounds, seconds)?;
    let scale = if appm { 1.0e-18 } else { BARNS_TO_CM2 };
    let mut acc = 0.0;
    for (f, r) in flux.iter().zip(response.iter()) {
        // Zero-flux groups contribute exactly 0.0 (no division anywhere).
        acc += f * r;
    }
    Ok(scale * seconds * acc)
}

/// NRT-dpa: fold the caller's dpa cross sections (built with the NRT
/// displacement function, e.g. by a SPECTER/NJOY-class pipeline) over the
/// group flux for `seconds` of exposure.
///
/// `dpa = seconds · 1e-24 · Σ_g flux[g]·response[g]`.
pub fn nrt_dpa(flux: &[f64], response: &[f64], bounds: &[f64], seconds: f64) -> Result<f64> {
    fold_scaled(flux, response, bounds, seconds, false)
}

/// arc-dpa: same fold with arc-corrected dpa cross sections (the
/// [`crate::physics::arc_displacements`] damage function in the pipeline).
pub fn arc_dpa(flux: &[f64], response: &[f64], bounds: &[f64], seconds: f64) -> Result<f64> {
    fold_scaled(flux, response, bounds, seconds, false)
}

/// Gas production in atomic parts per million (He or H, whichever gas the
/// caller's `response` counts — one nuclide's production cross section per
/// target atom basis).
///
/// `appm = seconds · 1e-18 · Σ_g flux[g]·response[g]`.
pub fn gas_appm(flux: &[f64], response: &[f64], bounds: &[f64], seconds: f64) -> Result<f64> {
    fold_scaled(flux, response, bounds, seconds, true)
}

/// He/dpa ratio in appm per dpa, from one spectral fold of the He
/// production cross sections and the damage cross sections over the same
/// flux.
///
/// The ratio is computed from the two folds, then checked: a damage fold
/// of exactly zero is [`Error::ZeroDpa`] — never `inf`.
pub fn he_dpa_ratio(
    flux: &[f64],
    he_response: &[f64],
    damage_response: &[f64],
    bounds: &[f64],
    seconds: f64,
) -> Result<f64> {
    let he = gas_appm(flux, he_response, bounds, seconds)?;
    let dpa = nrt_dpa(flux, damage_response, bounds, seconds)?;
    if dpa == 0.0 {
        return Err(Error::ZeroDpa);
    }
    Ok(he / dpa)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three-group synthetic grid (MeV) — hand-picked round bounds.
    fn bounds3() -> Vec<f64> {
        vec![0.0, 0.1, 1.0, 20.0]
    }

    #[test]
    fn flat_response_hand_sum() {
        let bounds = bounds3();
        let flux = vec![1.0e12, 2.0e12, 4.0e12];
        let resp = vec![100.0, 200.0, 50.0];
        // Hand sum: 1e12*100 + 2e12*200 + 4e12*50 = 7e14 barns·n/cm²/s.
        let want = 1.0e-24 * 2.0 * 7.0e14;
        assert_eq!(nrt_dpa(&flux, &resp, &bounds, 2.0).unwrap(), want);
        assert_eq!(arc_dpa(&flux, &resp, &bounds, 2.0).unwrap(), want);
        // appm differs from dpa by exactly the 1e6 per-atom factor.
        let want_appm = 1.0e-18 * 2.0 * 7.0e14;
        assert_eq!(gas_appm(&flux, &resp, &bounds, 2.0).unwrap(), want_appm);
    }

    #[test]
    fn zero_flux_groups_contribute_exactly_zero() {
        let bounds = bounds3();
        let flux = vec![0.0, 2.0e12, 0.0];
        let resp = vec![1.0e3, 1.5e2, 7.0e3];
        let want = 1.0e-24 * 1.0 * (2.0e12 * 1.5e2);
        assert_eq!(nrt_dpa(&flux, &resp, &bounds, 1.0).unwrap(), want);
        // All-zero flux: exactly 0 dpa -> the ratio is the named error.
        let dead = vec![0.0, 0.0, 0.0];
        assert_eq!(nrt_dpa(&dead, &resp, &bounds, 1.0).unwrap(), 0.0);
        assert!(matches!(
            he_dpa_ratio(&dead, &resp, &resp, &bounds, 1.0),
            Err(Error::ZeroDpa)
        ));
        // Zero damage response with live flux is the same named error.
        let zero_resp = vec![0.0; 3];
        assert!(matches!(
            he_dpa_ratio(&flux, &resp, &zero_resp, &bounds, 1.0),
            Err(Error::ZeroDpa)
        ));
    }

    #[test]
    fn ratio_uses_one_flux_fold() {
        let bounds = bounds3();
        let flux = vec![1.0e13, 1.0e12, 1.0e11];
        let he = vec![0.2, 0.4, 0.6];
        let dpa_xs = vec![50.0, 100.0, 200.0];
        let he_appm = gas_appm(&flux, &he, &bounds, 3.0).unwrap();
        let dpa = nrt_dpa(&flux, &dpa_xs, &bounds, 3.0).unwrap();
        assert_eq!(
            he_dpa_ratio(&flux, &he, &dpa_xs, &bounds, 3.0).unwrap(),
            he_appm / dpa
        );
    }

    #[test]
    fn fluence_fold_convention() {
        // seconds = 1 with the fluence parked in `flux` is the documented
        // fluence-fold convention used by the SPECTER oracle spots.
        let bounds = vec![0.0, 20.0];
        let fluence = vec![4.78373e22];
        let xs = vec![1.9118e2];
        let dpa = nrt_dpa(&fluence, &xs, &bounds, 1.0).unwrap();
        assert!((dpa - 9.145535).abs() < 1e-5, "dpa = {dpa}");
    }

    #[test]
    fn malformed_folds_name_their_cause() {
        let bounds = bounds3();
        let flux = vec![1.0, 2.0, 3.0];
        let resp = vec![1.0, 2.0, 3.0];
        assert!(matches!(nrt_dpa(&[], &[], &[], 1.0), Err(Error::Empty)));
        assert!(matches!(
            nrt_dpa(&flux, &[1.0, 2.0], &bounds, 1.0),
            Err(Error::DimensionMismatch {
                what: "response",
                ..
            })
        ));
        assert!(matches!(
            nrt_dpa(&flux, &resp, &[0.0, 1.0], 1.0),
            Err(Error::DimensionMismatch { what: "bounds", .. })
        ));
        assert!(matches!(
            nrt_dpa(&flux, &resp, &bounds, 0.0),
            Err(Error::NonPositive("seconds"))
        ));
        assert!(matches!(
            nrt_dpa(&[f64::INFINITY, 1.0, 1.0], &resp, &bounds, 1.0),
            Err(Error::NonFinite("flux"))
        ));
        assert!(matches!(
            nrt_dpa(&[-1.0, 1.0, 1.0], &resp, &bounds, 1.0),
            Err(Error::Negative("flux"))
        ));
        assert!(matches!(
            nrt_dpa(&flux, &[1.0, -1.0, 1.0], &bounds, 1.0),
            Err(Error::Negative("response"))
        ));
        assert!(matches!(
            nrt_dpa(&flux, &resp, &[0.0, 0.5, 0.5, 1.0], 1.0),
            Err(Error::NonMonotonicBounds { index: 1 })
        ));
        assert!(matches!(
            nrt_dpa(&flux, &resp, &[f64::NAN, 1.0, 2.0, 3.0], 1.0),
            Err(Error::NonFinite("bounds"))
        ));
    }
}
