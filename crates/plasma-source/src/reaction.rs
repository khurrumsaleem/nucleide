//! Fusion reactions and their temperature-broadened neutron spectra.
//!
//! The spectrum of thermonuclear neutrons is Gaussian to good approximation
//! with a width proportional to `sqrt(T_i)` — Brysk, Plasma Phys. **15**
//! (1973) 611. The closed-form coefficients used here are the published fits
//! of Ballabio et al., Nucl. Fusion **38** (1998) 1723, Table III, which
//! refine the Brysk scaling with a weakly temperature-dependent width and a
//! small mean-energy shift:
//!
//! ```text
//! mean(T_i) = E0 + a1·Ti^(2/3) / (1 + a2·Ti^a3) + a4·Ti          [keV]
//! FWHM(T_i) = w0 · (1 + b1·Ti^(2/3) / (1 + b2·Ti^b3) + b4·Ti) · sqrt(T_i)  [keV]
//! sigma     = FWHM / (2·sqrt(2·ln 2))
//! ```
//!
//! with `T_i` in keV. At `T_i = 0` both corrections vanish and the spectrum
//! degenerates to the nominal monoenergetic line `E0` (14.021 MeV for D-T,
//! 2.4495 MeV for D-D).

use crate::{Error, Result};

/// Gaussian FWHM-to-sigma factor `2·sqrt(2·ln 2)`.
const FWHM_OVER_SIGMA: f64 = 2.354_820_045_030_949_3;

/// Table III fit coefficients of Ballabio et al. 1998 for one reaction.
#[derive(Debug, Clone, Copy)]
struct BallabioFit {
    /// Nominal (cold) neutron energy \[MeV\].
    e0_mev: f64,
    /// Mean-shift coefficients `(a1, a2, a3, a4)` with `T_i` in keV.
    mean: (f64, f64, f64, f64),
    /// Width coefficients `(w0, b1, b2, b3, b4)` with `T_i` in keV.
    width: (f64, f64, f64, f64, f64),
}

/// D(d,n)³He fit (neutron branch only; the twin D(d,p)T proton branch is out
/// of scope for a neutron source crate).
const DD: BallabioFit = BallabioFit {
    e0_mev: 2.449_5,
    mean: (4.695_15, -0.040_729, 0.47, 0.818_44),
    width: (82.542, 1.701_3e-3, 0.168_88, 0.49, 7.946_0e-4),
};

/// T(d,n)⁴He fit.
const DT: BallabioFit = BallabioFit {
    e0_mev: 14.021,
    mean: (5.305_09, 2.473_6e-3, 1.84, 1.381_8),
    width: (177.259, 5.106_8e-4, 7.622_3e-3, 1.78, 8.769_1e-5),
};

/// Fusion reaction driving a neutron source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionReaction {
    /// T(d,n)⁴He — the 14.1 MeV deuterium–tritium neutron.
    Dt,
    /// D(d,n)³He — the 2.45 MeV deuterium–deuterium neutron.
    Dd,
}

impl FusionReaction {
    /// Published Table III coefficient set for this reaction.
    fn fit(self) -> &'static BallabioFit {
        match self {
            FusionReaction::Dt => &DT,
            FusionReaction::Dd => &DD,
        }
    }

    /// Human-readable reaction label (`"D-T"`, `"D-D"`).
    pub fn label(self) -> &'static str {
        match self {
            FusionReaction::Dt => "D-T",
            FusionReaction::Dd => "D-D",
        }
    }

    /// Nominal (cold, `T_i = 0`) neutron energy \[MeV\]: 14.021 for D-T,
    /// 2.4495 for D-D — the "14.1 MeV" / "2.45 MeV" lines of the literature.
    pub fn nominal_energy_mev(self) -> f64 {
        self.fit().e0_mev
    }

    /// Mean neutron energy \[MeV\] at ion temperature `ti_kev` \[keV\]
    /// (Ballabio et al. 1998, Table III mean fit).
    ///
    /// Errors: [`Error::NonFinite`] / [`Error::NegativeIonTemperature`].
    pub fn mean_energy_mev(self, ti_kev: f64) -> Result<f64> {
        let fit = self.fit();
        let ti = validate_ti(ti_kev)?;
        let (a1, a2, a3, a4) = fit.mean;
        let delta_kev = a1 * ti.powf(2.0 / 3.0) / (1.0 + a2 * ti.powf(a3)) + a4 * ti;
        Ok(fit.e0_mev + delta_kev / 1000.0)
    }

    /// Gaussian standard deviation \[MeV\] at ion temperature `ti_kev`
    /// \[keV\] (Ballabio et al. 1998, Table III width fit; the Brysk
    /// `sigma ∝ sqrt(T_i)` scaling with a weakly temperature-dependent
    /// coefficient). Zero at `T_i = 0`.
    ///
    /// Errors: [`Error::NonFinite`] / [`Error::NegativeIonTemperature`].
    pub fn sigma_mev(self, ti_kev: f64) -> Result<f64> {
        let fit = self.fit();
        let ti = validate_ti(ti_kev)?;
        if ti == 0.0 {
            return Ok(0.0);
        }
        let (w0, b1, b2, b3, b4) = fit.width;
        let delta = b1 * ti.powf(2.0 / 3.0) / (1.0 + b2 * ti.powf(b3)) + b4 * ti;
        let fwhm_kev = w0 * (1.0 + delta) * ti.sqrt();
        Ok(fwhm_kev / 1000.0 / FWHM_OVER_SIGMA)
    }

    /// Mean and sigma together (one validation pass).
    pub fn moments_mev(self, ti_kev: f64) -> Result<(f64, f64)> {
        Ok((self.mean_energy_mev(ti_kev)?, self.sigma_mev(ti_kev)?))
    }
}

/// Shared ion-temperature validation for the moment functions.
fn validate_ti(ti_kev: f64) -> Result<f64> {
    if !ti_kev.is_finite() {
        return Err(Error::NonFinite("ion temperature"));
    }
    if ti_kev < 0.0 {
        return Err(Error::NegativeIonTemperature(ti_kev));
    }
    Ok(ti_kev)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden moments recomputed independently from the published Table III
    /// coefficients (Python float64, documented in
    /// `validation/plasma_source_vs_openmc.py` gate P2). They pin both the
    /// coefficient transcription and the unit conventions (keV in, MeV out).
    #[test]
    fn ballabio_moments_match_independent_recomputation() {
        // (reaction, ti_kev, mean MeV, sigma MeV) — full-precision values
        // recomputed independently from the published Table III coefficients
        // (float64, documented in `validation/plasma_source_vs_openmc.py`
        // gate P2). They pin both the coefficient transcription and the unit
        // conventions (keV in, MeV out).
        let gold: &[(FusionReaction, f64, f64, f64)] = &[
            (
                FusionReaction::Dd,
                1.0,
                2.455_212_938_009_43,
                0.035_131_231_189_825_3,
            ),
            (
                FusionReaction::Dd,
                10.0,
                2.482_454_746_525_07,
                0.112_301_222_672_679,
            ),
            (
                FusionReaction::Dd,
                20.0,
                2.507_372_988_406_42,
                0.160_384_037_736_95,
            ),
            (
                FusionReaction::Dt,
                1.0,
                14.027_673_799_709_5,
                0.075_319_718_060_101_5,
            ),
            (
                FusionReaction::Dt,
                10.0,
                14.055_843_863_041,
                0.238_635_740_911_14,
            ),
            (
                FusionReaction::Dt,
                20.0,
                14.072_874_253_353_2,
                0.337_721_762_921_614,
            ),
        ];
        for &(reaction, ti, mean, sigma) in gold {
            let (m, s) = reaction.moments_mev(ti).unwrap();
            assert!(
                (m - mean).abs() < 1e-12,
                "{reaction:?} Ti={ti}: mean {m} vs {mean}"
            );
            assert!(
                (s - sigma).abs() < 1e-12,
                "{reaction:?} Ti={ti}: sigma {s} vs {sigma}"
            );
        }
    }

    #[test]
    fn zero_temperature_is_monoenergetic_at_nominal() {
        for reaction in [FusionReaction::Dt, FusionReaction::Dd] {
            assert_eq!(
                reaction.moments_mev(0.0).unwrap(),
                (reaction.nominal_energy_mev(), 0.0)
            );
        }
    }

    #[test]
    fn width_scales_like_sqrt_ti_asymptotically() {
        // Brysk scaling: sigma/sqrt(Ti) drifts only via the weak (1+delta)
        // correction, so adjacent decades must bracket a common constant.
        let s1 = FusionReaction::Dt.sigma_mev(4.0).unwrap();
        let s2 = FusionReaction::Dt.sigma_mev(16.0).unwrap();
        let ratio = s2 / s1;
        assert!(
            (ratio - 2.0).abs() < 0.05,
            "sigma(16)/sigma(4) = {ratio}, ~2 expected"
        );
    }

    #[test]
    fn bad_temperatures_are_loud() {
        assert_eq!(
            FusionReaction::Dt.mean_energy_mev(-1.0),
            Err(Error::NegativeIonTemperature(-1.0))
        );
        assert_eq!(
            FusionReaction::Dt.sigma_mev(f64::NAN),
            Err(Error::NonFinite("ion temperature"))
        );
        assert_eq!(
            FusionReaction::Dd.mean_energy_mev(f64::INFINITY),
            Err(Error::NonFinite("ion temperature"))
        );
    }
}
