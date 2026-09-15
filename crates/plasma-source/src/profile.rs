//! Caller-supplied plasma profiles over the minor radius — the L/H/A-mode
//! parametrizations of Fausser et al., Fus. Eng. Des. **87** (2012) 787
//! (public form of the ITER/EU-DEMO source model), in the spelling common to
//! that paper and the MIT `openmc-plasma-source` package.
//!
//! Profiles are *inputs*: the caller supplies the centre/peaking/pedestal/
//! separatrix values and the confinement mode; nothing here computes or
//! fits profiles, and no equilibrium is solved.
//!
//! ```text
//! L mode, density and temperature:
//!     f(r) = f_c · (1 − (r/a)²)^α
//!
//! H/A mode (same formulas; the modes differ by parameter choice):
//!     core   (r ≤ r_ped): f_ped + (f_c − f_ped) · (1 − (r/r_ped)²)^α
//!     edge   (r > r_ped): linear from f_ped at r_ped to f_sep at a
//!     temperature core uses (1 − (r/r_ped)^β_T)^α_T
//! ```
//!
//! with `a` the minor radius and `r_ped` the pedestal radius. Density is in
//! m⁻³ (Fausser convention) and temperature in keV; both are converted
//! internally where cm conventions need them (density → cm⁻³, a factor 1e-6).
//!
//! Gates pinned: definition-level goldens at hand radii for all three modes,
//! C⁰ continuity at the pedestal foot, non-negativity over `[0, a]`, and
//! L-mode `(r/a)²` limit values.

use crate::{Error, Result};

/// Confinement-mode profile family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfileMode {
    /// L-mode: single power-law falloff, no pedestal.
    L,
    /// H-mode: cored power law plus a pedestal and linear edge.
    H,
    /// A-mode: same formulas as H in this public parametrization (Fausser
    /// distinguishes the modes by parameter choice, not algebra).
    A,
}

impl ProfileMode {
    /// Parse a mode label (`"L"`, `"H"`, `"A"`; case-insensitive).
    pub fn parse(label: &str) -> Result<Self> {
        match label.to_ascii_uppercase().as_str() {
            "L" => Ok(ProfileMode::L),
            "H" => Ok(ProfileMode::H),
            "A" => Ok(ProfileMode::A),
            _ => Err(Error::InvalidProfile("mode must be one of L, H, A")),
        }
    }
}

/// Ion-density profile parameters (Fausser convention, m⁻³).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityProfile {
    /// Centre density `n_c` \[m⁻³\]; strictly positive.
    pub centre_m3: f64,
    /// Peaking exponent `α_n` (dimensionless); non-negative.
    pub peaking_factor: f64,
    /// Pedestal density `n_ped` \[m⁻³\] (H/A); non-negative.
    pub pedestal_m3: f64,
    /// Separatrix density `n_sep` \[m⁻³\] (H/A); non-negative.
    pub separatrix_m3: f64,
}

/// Ion-temperature profile parameters (keV).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TemperatureProfile {
    /// Centre temperature `T_c` \[keV\]; strictly positive.
    pub centre_kev: f64,
    /// Peaking exponent `α_T` (dimensionless); non-negative.
    pub peaking_factor: f64,
    /// Beta exponent `β_T` in `(1 − (r/r_ped)^β_T)` (dimensionless); strictly
    /// positive (H/A).
    pub beta: f64,
    /// Pedestal temperature `T_ped` \[keV\] (H/A); non-negative.
    pub pedestal_kev: f64,
    /// Separatrix temperature `T_sep` \[keV\] (H/A); non-negative.
    pub separatrix_kev: f64,
}

/// Density in m⁻³ at minor radius `r` \[cm\]; `mode` selects the family.
pub fn density(
    mode: ProfileMode,
    p: &DensityProfile,
    minor_radius: f64,
    pedestal_radius: f64,
    r: f64,
) -> f64 {
    match mode {
        ProfileMode::L => p.centre_m3 * (1.0 - (r / minor_radius).powi(2)).powf(p.peaking_factor),
        ProfileMode::H | ProfileMode::A => {
            if r < pedestal_radius {
                let core = (1.0 - (r / pedestal_radius).powi(2)).powf(p.peaking_factor);
                p.pedestal_m3 + (p.centre_m3 - p.pedestal_m3) * core
            } else {
                p.separatrix_m3
                    + (p.pedestal_m3 - p.separatrix_m3) * (minor_radius - r)
                        / (minor_radius - pedestal_radius)
            }
        }
    }
}

/// Ion temperature in keV at minor radius `r` \[cm\]; `mode` selects the family.
pub fn temperature(
    mode: ProfileMode,
    p: &TemperatureProfile,
    minor_radius: f64,
    pedestal_radius: f64,
    r: f64,
) -> f64 {
    match mode {
        ProfileMode::L => p.centre_kev * (1.0 - (r / minor_radius).powi(2)).powf(p.peaking_factor),
        ProfileMode::H | ProfileMode::A => {
            if r < pedestal_radius {
                let arg = 1.0 - (r / pedestal_radius).powf(p.beta);
                p.pedestal_kev + (p.centre_kev - p.pedestal_kev) * arg.powf(p.peaking_factor)
            } else {
                p.separatrix_kev
                    + (p.pedestal_kev - p.separatrix_kev) * (minor_radius - r)
                        / (minor_radius - pedestal_radius)
            }
        }
    }
}

/// Validate the profile parameters for a minor/pedestal radius pair.
pub fn validate_profiles(
    mode: ProfileMode,
    density_profile: &DensityProfile,
    temperature_profile: &TemperatureProfile,
    minor_radius: f64,
    pedestal_radius: f64,
) -> Result<()> {
    let d = density_profile;
    let t = temperature_profile;
    for (field, value) in [
        ("centre density", d.centre_m3),
        ("density peaking factor", d.peaking_factor),
        ("pedestal density", d.pedestal_m3),
        ("separatrix density", d.separatrix_m3),
        ("centre temperature", t.centre_kev),
        ("temperature peaking factor", t.peaking_factor),
        ("temperature beta", t.beta),
        ("pedestal temperature", t.pedestal_kev),
        ("separatrix temperature", t.separatrix_kev),
    ] {
        if !value.is_finite() {
            return Err(Error::InvalidProfile(field));
        }
    }
    if d.centre_m3 <= 0.0 {
        return Err(Error::InvalidProfile("centre density must be > 0"));
    }
    if d.peaking_factor < 0.0 {
        return Err(Error::InvalidProfile("density peaking factor must be >= 0"));
    }
    if d.pedestal_m3 < 0.0 || d.separatrix_m3 < 0.0 {
        return Err(Error::InvalidProfile(
            "pedestal/separatrix density must be >= 0",
        ));
    }
    if t.centre_kev <= 0.0 {
        return Err(Error::InvalidProfile("centre temperature must be > 0"));
    }
    if t.peaking_factor < 0.0 {
        return Err(Error::InvalidProfile(
            "temperature peaking factor must be >= 0",
        ));
    }
    if t.beta <= 0.0 {
        return Err(Error::InvalidProfile("temperature beta must be > 0"));
    }
    if t.pedestal_kev < 0.0 || t.separatrix_kev < 0.0 {
        return Err(Error::InvalidProfile(
            "pedestal/separatrix temperature must be >= 0",
        ));
    }
    if mode != ProfileMode::L
        && pedestal_radius.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
    {
        return Err(Error::InvalidProfile(
            "pedestal radius must lie in (0, minor radius)",
        ));
    }
    if mode != ProfileMode::L && pedestal_radius >= minor_radius {
        return Err(Error::InvalidProfile(
            "pedestal radius must lie in (0, minor radius)",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: f64 = 200.0;
    const R_PED: f64 = 150.0;

    fn density_params() -> DensityProfile {
        DensityProfile {
            centre_m3: 1.2e20,
            peaking_factor: 1.1,
            pedestal_m3: 4.0e19,
            separatrix_m3: 3.0e19,
        }
    }

    fn temperature_params() -> TemperatureProfile {
        TemperatureProfile {
            centre_kev: 28.0,
            peaking_factor: 2.5,
            beta: 2.0,
            pedestal_kev: 4.0,
            separatrix_kev: 0.1,
        }
    }

    // --- G-prof: definition-level goldens (recomputed independently) --------

    #[test]
    fn l_mode_density_golden() {
        let p = density_params();
        let n = density(ProfileMode::L, &p, A, R_PED, 80.0);
        assert!((n - 9.905_775_034_943_809e19).abs() < 1e5, "n {n}");
        let edge = density(ProfileMode::L, &p, A, R_PED, A);
        assert!((edge - 0.0).abs() < 1e-30, "separatrix {edge}");
        let center = density(ProfileMode::L, &p, A, R_PED, 0.0);
        assert_eq!(center, p.centre_m3);
    }

    #[test]
    fn h_mode_core_and_edge_goldens() {
        let p = density_params();
        let core = density(ProfileMode::H, &p, A, R_PED, 100.0);
        assert!((core - 8.190_735_310_433_901e19).abs() < 1e5, "core {core}");
        let edge = density(ProfileMode::H, &p, A, R_PED, 180.0);
        assert!((edge - 3.4e19).abs() < 1e5, "edge {edge}");
        // A-mode shares the H algebra.
        assert_eq!(
            density(ProfileMode::A, &p, A, R_PED, 100.0),
            density(ProfileMode::H, &p, A, R_PED, 100.0)
        );
    }

    #[test]
    fn h_mode_temperature_golden() {
        let t = temperature_params();
        let core = temperature(ProfileMode::H, &t, A, R_PED, 100.0);
        assert!((core - 9.521_155_499_999_482).abs() < 1e-9, "T core {core}");
        let edge = temperature(ProfileMode::H, &t, A, R_PED, 175.0);
        assert!((edge - 2.05).abs() < 1e-12, "T edge {edge}");
        let l_mode = temperature(ProfileMode::L, &t, A, R_PED, 80.0);
        assert!(
            (l_mode - 18.107_406_298_020_706).abs() < 1e-9,
            "T L {l_mode}"
        );
    }

    // --- G-cont: pedestal C0 continuity and non-negativity -------------------

    #[test]
    fn pedestal_is_continuous_and_nonnegative() {
        let d = density_params();
        let t = temperature_params();
        let n_left = density(ProfileMode::H, &d, A, R_PED, 150.0 - 1e-9);
        let n_right = density(ProfileMode::H, &d, A, R_PED, 150.0 + 1e-9);
        assert!((n_left - n_right).abs() < 1e-9 * d.centre_m3);
        let t_left = temperature(ProfileMode::H, &t, A, R_PED, 150.0 - 1e-9);
        let t_right = temperature(ProfileMode::H, &t, A, R_PED, 150.0 + 1e-9);
        assert!((t_left - t_right).abs() < 1e-9);
        for i in 0..=400 {
            let r = A * i as f64 / 400.0;
            assert!(density(ProfileMode::H, &d, A, R_PED, r) >= 0.0);
            assert!(temperature(ProfileMode::H, &t, A, R_PED, r) >= 0.0);
        }
    }

    #[test]
    fn invalid_profiles_are_loud() {
        let mut d = density_params();
        d.centre_m3 = 0.0;
        assert!(matches!(
            validate_profiles(ProfileMode::H, &d, &temperature_params(), A, R_PED),
            Err(Error::InvalidProfile(_))
        ));
        let mut t = temperature_params();
        t.beta = 0.0;
        assert!(matches!(
            validate_profiles(ProfileMode::H, &density_params(), &t, A, R_PED),
            Err(Error::InvalidProfile(_))
        ));
        assert!(matches!(
            validate_profiles(
                ProfileMode::H,
                &density_params(),
                &temperature_params(),
                A,
                0.0
            ),
            Err(Error::InvalidProfile(_))
        ));
        assert!(ProfileMode::parse("q").is_err());
        assert_eq!(ProfileMode::parse("h").unwrap(), ProfileMode::H);
    }
}
