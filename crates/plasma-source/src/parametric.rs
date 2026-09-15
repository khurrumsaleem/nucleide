//! Parametric tokamak plasma source: caller-supplied profiles over Miller
//! geometry, reactivity-weighted neutron emission, seeded sampling, and
//! histogram source-card emission.
//!
//! This is the second Cycle-01 landing. The source model is the public
//! ITER/EU-DEMO parametrization of Fausser et al., Fus. Eng. Des. **87**
//! (2012) 787 — [`MillerGeometry`] flux surfaces ([`crate::miller`]) with
//! L/H/A-mode density/temperature profiles ([`crate::profile`]) — and the
//! neutron emission strength
//!
//! ```text
//! S(r) = f_fuel · n_i(r)² · ⟨σv⟩(T_i(r))          [neutrons/s/cm³, relative]
//! ```
//!
//! with `⟨σv⟩` the Bosch & Hale fits ([`crate::reactivity`]) and
//! `f_fuel = 1/4` for equimolar D-T (`n_D = n_T = n_i/2`) or `1/2` for pure
//! D-D (the ½ avoids double-counting identical reactant pairs, and follows
//! the Fausser/`openmc-plasma-source` strength convention). Birth positions
//! are drawn ∝ `S(r)·R·|J|` — the volume element of [`crate::miller`] —
//! and birth energies from the local-ion-temperature Ballabio Gaussian of
//! [`crate::reaction`].
//!
//! Profiles are caller inputs; nothing here computes profiles or solves an
//! equilibrium. Arbitrary (non-Maxwellian, non-equimolar) reactant
//! distributions — the Eriksson et al., Comput. Phys. Commun. **199**
//! (2016) 40 generalization — are the documented loud boundary
//! ([`Error::NotYetSupported`]): the strength model above assumes Maxwellian
//! reactants at a common `T_i`. Toroidal sectors stay caller-side via
//! rejection of the sampled `φ`.
//!
//! # Card emission
//!
//! Transport source cards cannot represent the correlated `(r, z)` joint
//! (nor the position–energy correlation), so [`emission_histograms`] renders
//! the *marginals*: radial and vertical histograms plus the global marginal
//! energy spectrum. The drift report quantifies what the card preserves and
//! what it loses (truncation; the joint-correlation distance).

use std::f64::consts::PI;

use crate::miller::MillerGeometry;
use crate::profile::{self, DensityProfile, ProfileMode, TemperatureProfile};
use crate::sample::{Particle, Rng};
use crate::{Error, FusionReaction, Result};

/// Fine-grid resolution for the radial weight table.
const R_GRID: usize = 256;
/// Poloidal quadrature points per radial cell.
const THETA_GRID: usize = 256;

/// Parametric tokamak plasma source configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParametricPlasmaConfig {
    /// Miller flux-surface geometry.
    pub geometry: MillerGeometry,
    /// Confinement-mode profile family (L/H/A).
    pub mode: ProfileMode,
    /// Ion-density profile parameters (Fausser convention, m⁻³).
    pub ion_density: DensityProfile,
    /// Ion-temperature profile parameters (keV).
    pub ion_temperature: TemperatureProfile,
    /// Pedestal radius `r_ped` \[cm\]; required in `(0, a_minor)` for H/A.
    pub pedestal_radius_cm: f64,
    /// Fuel reaction: D-T (equimolar) or D-D. Mixtures are the loud
    /// boundary (Eriksson-weighted reactant distributions).
    pub fuel: FusionReaction,
    /// Particle weight carried by the sampler and the emitted cards.
    pub weight: f64,
}

impl ParametricPlasmaConfig {
    /// Validate the full configuration.
    pub fn validate(&self) -> Result<()> {
        self.geometry.validate()?;
        profile::validate_profiles(
            self.mode,
            &self.ion_density,
            &self.ion_temperature,
            self.geometry.minor_radius_cm,
            self.pedestal_radius_cm,
        )?;
        if !self.pedestal_radius_cm.is_finite() {
            return Err(Error::InvalidProfile("pedestal radius"));
        }
        if !self.weight.is_finite() || self.weight <= 0.0 {
            return Err(Error::NonPositiveWeight(self.weight));
        }
        Ok(())
    }

    /// Ion density \[m⁻³\] at minor radius `r` \[cm\].
    pub fn density_m3(&self, r: f64) -> f64 {
        profile::density(
            self.mode,
            &self.ion_density,
            self.geometry.minor_radius_cm,
            self.pedestal_radius_cm,
            r,
        )
    }

    /// Ion temperature \[keV\] at minor radius `r` \[cm\].
    pub fn temperature_kev(&self, r: f64) -> f64 {
        profile::temperature(
            self.mode,
            &self.ion_temperature,
            self.geometry.minor_radius_cm,
            self.pedestal_radius_cm,
            r,
        )
    }

    /// Relative neutron source density \[neutrons/s/cm³, arbitrary global
    /// scale\] at minor radius `r` \[cm\]: `f_fuel·n²·⟨σv⟩` with density in
    /// cm⁻³ and reactivity in cm³/s. Zero temperature ⇒ zero strength (cold
    /// separatrix makes no neutrons).
    pub fn strength_density(&self, r: f64) -> Result<f64> {
        let n_m3 = self.density_m3(r);
        let ti_kev = self.temperature_kev(r);
        let reactivity = self.fuel.reactivity_m3_per_s(ti_kev)? * 1e6; // m³/s → cm³/s
        let n_cm3 = n_m3 * 1e-6;
        let fuel_factor = match self.fuel {
            FusionReaction::Dt => 0.25,
            FusionReaction::Dd => 0.5,
        };
        Ok(fuel_factor * n_cm3 * n_cm3 * reactivity)
    }

    /// Total relative source strength ∭ S·R·|J| da dθ dφ (the toroidal
    /// integral contributes the factor `2π`). Arbitrary global scale, but
    /// ratios between configurations are meaningful.
    pub fn total_strength(&self) -> Result<f64> {
        let table = WeightTable::build(self)?;
        Ok(2.0 * PI * table.masses.iter().sum::<f64>())
    }
}

/// Piecewise-constant inverse-CDF table over `[0, a_minor]`: cell masses and
/// per-cell poloidal CDFs (so θ sampling needs no per-particle quadrature).
struct WeightTable {
    /// Cell edges (R_GRID + 1).
    edges: Vec<f64>,
    /// Cell masses `m_i ≈ S(r_i)·∫R|J|dθ·Δr` (unnormalized).
    masses: Vec<f64>,
    /// Cumulative mass (R_GRID + 1, first 0).
    cumulative: Vec<f64>,
    /// Per-cell poloidal CDFs: THETA_GRID + 1 entries each (first 0, last 1).
    theta_cdfs: Vec<Vec<f64>>,
}

impl WeightTable {
    fn build(config: &ParametricPlasmaConfig) -> Result<Self> {
        let a = config.geometry.minor_radius_cm;
        let dr = a / R_GRID as f64;
        let mut edges = Vec::with_capacity(R_GRID + 1);
        let mut masses = Vec::with_capacity(R_GRID);
        let mut cumulative = Vec::with_capacity(R_GRID + 1);
        let mut theta_cdfs = Vec::with_capacity(R_GRID);
        let mut acc = 0.0;
        cumulative.push(0.0);
        for i in 0..R_GRID {
            let r = (i as f64 + 0.5) * dr;
            edges.push(i as f64 * dr);
            let strength = config.strength_density(r)?;
            // Poloidal marginal ∫ R|J| dθ at cell midpoint (uniform θ grid,
            // trapezoid on a periodic integrand).
            let dtheta = 2.0 * PI / THETA_GRID as f64;
            let mut weights = Vec::with_capacity(THETA_GRID);
            let mut w_acc = 0.0;
            for j in 0..THETA_GRID {
                let theta = (j as f64 + 0.5) * dtheta;
                let w = config.geometry.volume_element(r, theta);
                weights.push(w);
                w_acc += w;
            }
            let mass = strength * w_acc * dtheta * dr;
            masses.push(mass);
            acc += mass;
            cumulative.push(acc);
            let mut cdf = Vec::with_capacity(THETA_GRID + 1);
            let mut c = 0.0;
            cdf.push(0.0);
            for w in weights {
                c += w / w_acc;
                cdf.push(c);
            }
            theta_cdfs.push(cdf);
        }
        edges.push(a);
        if acc.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !acc.is_finite() {
            return Err(Error::InvalidProfile(
                "total parametric source strength is zero (check profiles and fuel)",
            ));
        }
        Ok(Self {
            edges,
            masses,
            cumulative,
            theta_cdfs,
        })
    }

    /// Sample a minor radius from the cell masses.
    fn sample_r(&self, u: f64) -> f64 {
        let total = self.cumulative[R_GRID];
        let target = u * total;
        let idx = match self
            .cumulative
            .binary_search_by(|v| v.partial_cmp(&target).unwrap())
        {
            Ok(i) => i.min(R_GRID - 1),
            Err(i) => i.saturating_sub(1).min(R_GRID - 1),
        };
        let lo = self.edges[idx];
        let hi = self.edges[idx + 1];
        let frac = (target - self.cumulative[idx])
            / (self.cumulative[idx + 1] - self.cumulative[idx]).max(1e-300);
        lo + frac.min(1.0) * (hi - lo)
    }

    /// Sample a poloidal angle from the cell's poloidal CDF.
    fn sample_theta(&self, cell: usize, u: f64) -> f64 {
        let cdf = &self.theta_cdfs[cell];
        let idx = match cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap()) {
            Ok(i) => i.min(THETA_GRID - 1),
            Err(i) => i.saturating_sub(1).min(THETA_GRID - 1),
        };
        let dtheta = 2.0 * PI / THETA_GRID as f64;
        let lo = idx as f64 * dtheta;
        let frac = (u - cdf[idx]) / (cdf[idx + 1] - cdf[idx]).max(1e-300);
        lo + frac.min(1.0) * dtheta
    }
}

/// Deterministic sampler for a parametric plasma source.
pub struct ParametricSampler {
    config: ParametricPlasmaConfig,
    table: WeightTable,
    rng: Rng,
}

impl ParametricSampler {
    /// New sampler over `config` pinned to `seed`.
    pub fn new(config: ParametricPlasmaConfig, seed: u64) -> Result<Self> {
        config.validate()?;
        let table = WeightTable::build(&config)?;
        Ok(Self {
            config,
            table,
            rng: Rng::new(seed),
        })
    }

    /// Borrow the underlying configuration.
    pub fn config(&self) -> &ParametricPlasmaConfig {
        &self.config
    }

    /// Sample one particle: `r` from the strength-weighted radial CDF, `θ`
    /// from the cell's poloidal CDF, `φ` uniform; energy from the local
    /// ion-temperature Ballabio Gaussian; isotropic direction.
    pub fn sample(&mut self) -> Particle {
        let r = self.table.sample_r(self.rng.uniform());
        let dr = self.config.geometry.minor_radius_cm / R_GRID as f64;
        let cell = ((r / dr).floor() as usize).min(R_GRID - 1);
        let theta = self.table.sample_theta(cell, self.rng.uniform());
        let phi = 2.0 * PI * self.rng.uniform();
        let (big_r, z) = self.config.geometry.map(r, theta);
        let ti_kev = self.config.temperature_kev(r);
        let energy_mev = match FusionReaction::Dt.moments_mev(ti_kev) {
            Ok((mean, sigma)) if sigma > 0.0 => mean + sigma * self.rng.standard_normal(),
            Ok((mean, _)) => mean,
            Err(_) => 0.0,
        };
        Particle {
            position_cm: [big_r * phi.cos(), big_r * phi.sin(), z],
            direction: self.rng.isotropic_direction(),
            energy_mev,
            weight: self.config.weight,
        }
    }

    /// Sample `n` particles.
    pub fn sample_n(&mut self, n: usize) -> Vec<Particle> {
        (0..n).map(|_| self.sample()).collect()
    }
}

/// A binned marginal distribution for source-card emission.
#[derive(Debug, Clone, PartialEq)]
pub struct BinnedDistribution {
    /// Bin centers, ascending.
    pub centers: Vec<f64>,
    /// Normalized bin masses (sum 1.0).
    pub masses: Vec<f64>,
}

/// Marginal histograms of the parametric source for card emission:
/// radial birth profile, vertical birth profile, and the global marginal
/// energy spectrum (built on the fine grid; `bins` controls the output
/// binning of each marginal).
#[derive(Debug, Clone, PartialEq)]
pub struct EmissionHistograms {
    /// Radial marginal (birth minor-radius profile).
    pub radial: BinnedDistribution,
    /// Vertical marginal (birth Z profile over `[-z_max, z_max]`).
    pub vertical: BinnedDistribution,
    /// Global marginal energy spectrum \[MeV\].
    pub energy: BinnedDistribution,
    /// Captured energy probability mass (1 − tail truncation).
    pub energy_coverage: f64,
    /// Half the L1 distance between the true `(r, z)` birth joint and the
    /// product of the two marginals — the correlation information a
    /// product-form source card cannot carry (0 = independent).
    pub joint_correlation: f64,
}

/// Energy tabulation half-width in sigma (same convention as the ring/point
/// spectrum tabulation).
const ENERGY_WIDTH_SIGMA: f64 = 4.0;

/// Build the marginal histograms for one configuration.
pub fn emission_histograms(
    config: &ParametricPlasmaConfig,
    bins: usize,
) -> Result<EmissionHistograms> {
    config.validate()?;
    let bins = bins.max(4);
    let g = &config.geometry;
    let a = g.minor_radius_cm;
    let dr = a / R_GRID as f64;
    let dtheta = 2.0 * PI / THETA_GRID as f64;

    let mut radial_m = vec![0.0f64; bins];
    let mut vertical_m = vec![0.0f64; bins];
    let z_max = g.z_max();
    let mut energy_lo = f64::INFINITY;
    let mut energy_hi = f64::NEG_INFINITY;
    let mut cell_weights: Vec<f64> = Vec::with_capacity(R_GRID * THETA_GRID);
    let mut cell_r: Vec<f64> = Vec::with_capacity(R_GRID * THETA_GRID);
    let mut cell_z: Vec<f64> = Vec::with_capacity(R_GRID * THETA_GRID);
    let mut total = 0.0f64;
    for i in 0..R_GRID {
        let r = (i as f64 + 0.5) * dr;
        let strength = config.strength_density(r)?;
        let ti = config.temperature_kev(r);
        let (mu, sigma) = config.fuel.moments_mev(ti)?;
        energy_lo = energy_lo.min(mu - ENERGY_WIDTH_SIGMA * sigma);
        energy_hi = energy_hi.max(mu + ENERGY_WIDTH_SIGMA * sigma);
        for j in 0..THETA_GRID {
            let theta = (j as f64 + 0.5) * dtheta;
            let (_, z) = g.map(r, theta);
            let w = strength * g.volume_element(r, theta) * dtheta * dr;
            total += w;
            cell_weights.push(w);
            cell_r.push(r);
            cell_z.push(z);
        }
    }
    if total.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(Error::InvalidProfile(
            "total parametric source strength is zero",
        ));
    }
    let energy_lo = if energy_lo.is_finite() {
        energy_lo
    } else {
        0.0
    };
    let energy_hi = if energy_hi.is_finite() {
        energy_hi
    } else {
        1.0
    };

    // Bin the spatial marginals and accumulate the joint for the
    // correlation distance.
    let mut joint = vec![vec![0.0f64; bins]; bins];
    for k in 0..cell_weights.len() {
        let (w, r, z) = (cell_weights[k], cell_r[k], cell_z[k]);
        let ri = ((r / a) * bins as f64).floor() as usize;
        let ri = ri.min(bins - 1);
        let zi = (((z + z_max) / (2.0 * z_max)) * bins as f64).floor() as usize;
        let zi = zi.min(bins - 1);
        radial_m[ri] += w;
        vertical_m[zi] += w;
        joint[ri][zi] += w / total;
    }
    let radial: Vec<f64> = radial_m.iter().map(|m| m / total).collect();
    let vertical: Vec<f64> = vertical_m.iter().map(|m| m / total).collect();
    // joint_correlation = 1/2 Σ |p_true(i,j) − p_r(i) p_z(j)|
    let mut corr = 0.0;
    for i in 0..bins {
        for j in 0..bins {
            corr += (joint[i][j] - radial[i] * vertical[j]).abs();
        }
    }
    corr *= 0.5;

    // Marginal energy spectrum: Σ_cells w_cell · [Φ(hi; μ,σ) − Φ(lo; μ,σ)].
    let de = (energy_hi - energy_lo) / bins as f64;
    let mut energy_m = vec![0.0f64; bins];
    let mut energy_coverage = 0.0f64;
    for i in 0..R_GRID {
        let r = (i as f64 + 0.5) * dr;
        let ti = config.temperature_kev(r);
        let (mu, sigma) = config.fuel.moments_mev(ti)?;
        let cell_mass: f64 = cell_weights[i * THETA_GRID..(i + 1) * THETA_GRID]
            .iter()
            .sum();
        for (b, mass) in energy_m.iter_mut().enumerate() {
            let lo = energy_lo + b as f64 * de;
            let hi = lo + de;
            let p = if sigma > 0.0 {
                crate::spectrum::normal_cdf((hi - mu) / sigma)
                    - crate::spectrum::normal_cdf((lo - mu) / sigma)
            } else if lo < mu && mu <= hi {
                1.0
            } else {
                0.0
            };
            *mass += cell_mass * p;
        }
        let covered = if sigma > 0.0 {
            crate::spectrum::normal_cdf((energy_hi - mu) / sigma)
                - crate::spectrum::normal_cdf((energy_lo - mu) / sigma)
        } else {
            1.0
        };
        energy_coverage += cell_mass * covered;
    }
    energy_coverage /= total;
    let energy_centers: Vec<f64> = (0..bins)
        .map(|b| energy_lo + (b as f64 + 0.5) * de)
        .collect();
    let energy_masses: Vec<f64> = energy_m.iter().map(|m| m / total).collect();

    Ok(EmissionHistograms {
        radial: BinnedDistribution {
            centers: (0..bins)
                .map(|b| (b as f64 + 0.5) * (a / bins as f64))
                .collect(),
            masses: radial,
        },
        vertical: BinnedDistribution {
            centers: (0..bins)
                .map(|b| -z_max + (b as f64 + 0.5) * (2.0 * z_max / bins as f64))
                .collect(),
            masses: vertical,
        },
        energy: BinnedDistribution {
            centers: energy_centers,
            masses: energy_masses,
        },
        energy_coverage,
        joint_correlation: corr,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::miller::MillerGeometry;

    /// ITER-ish synthetic H-mode case (hand round numbers).
    pub(crate) fn iter_h_mode() -> ParametricPlasmaConfig {
        ParametricPlasmaConfig {
            geometry: MillerGeometry {
                major_radius_cm: 620.0,
                minor_radius_cm: 200.0,
                elongation: 1.85,
                triangularity: 0.35,
                shafranov_factor_cm: 15.0,
            },
            mode: ProfileMode::H,
            ion_density: DensityProfile {
                centre_m3: 1.2e20,
                peaking_factor: 1.1,
                pedestal_m3: 4.0e19,
                separatrix_m3: 3.0e19,
            },
            ion_temperature: TemperatureProfile {
                centre_kev: 28.0,
                peaking_factor: 2.5,
                beta: 2.0,
                pedestal_kev: 4.0,
                separatrix_kev: 0.1,
            },
            pedestal_radius_cm: 150.0,
            fuel: FusionReaction::Dt,
            weight: 1.0,
        }
    }

    /// Flat-profile configuration (constant n and T): strength ∝ volume
    /// element, so moments have closed forms (see module docs and gates).
    fn flat_config(kappa: f64, delta: f64, esh: f64) -> ParametricPlasmaConfig {
        let mut c = iter_h_mode();
        c.mode = ProfileMode::L;
        c.geometry.elongation = kappa;
        c.geometry.triangularity = delta;
        c.geometry.shafranov_factor_cm = esh;
        c.ion_density = DensityProfile {
            centre_m3: 1.0e20,
            peaking_factor: 0.0,
            pedestal_m3: 1.0,
            separatrix_m3: 1.0,
        };
        c.ion_temperature = TemperatureProfile {
            centre_kev: 20.0,
            peaking_factor: 0.0,
            beta: 1.0,
            pedestal_kev: 20.0,
            separatrix_kev: 20.0,
        };
        c
    }

    // --- G-strength: hand strength cases -------------------------------------

    #[test]
    fn strength_density_hand_cases() {
        let c = iter_h_mode();
        // n(0) = 1.2e20 m-3, T(0) = 28 keV.
        let s0 = c.strength_density(0.0).unwrap();
        let n_cm3 = 1.2e20 * 1e-6;
        let sv = FusionReaction::Dt.reactivity_m3_per_s(28.0).unwrap() * 1e6;
        let want = 0.25 * n_cm3 * n_cm3 * sv;
        assert!((s0 - want).abs() < 1e-12 * want, "{s0} vs {want}");
        // Cold separatrix: T(a) = 0.1 keV, tiny but non-zero strength.
        assert!(c.strength_density(200.0).unwrap() > 0.0);
        // L-mode flat density 1e20, T 20: exact factor 1/4·(1e14)^2·<σv>(20).
        let f = flat_config(1.85, 0.35, 15.0);
        let sf = f.strength_density(123.0).unwrap();
        let sv20 = FusionReaction::Dt.reactivity_m3_per_s(20.0).unwrap() * 1e6;
        let wantf = 0.25 * 1e28 * sv20;
        assert!((sf - wantf).abs() < 1e-12 * wantf, "{sf} vs {wantf}");
    }

    #[test]
    fn total_strength_is_positive_and_scales_with_reactivity() {
        let hot = flat_config(1.0, 0.0, 0.0);
        let mut cold = hot;
        cold.ion_temperature.centre_kev = 10.0;
        cold.ion_temperature.peaking_factor = 0.0;
        cold.ion_temperature.pedestal_kev = 10.0;
        cold.ion_temperature.separatrix_kev = 10.0;
        let s_hot = hot.total_strength().unwrap();
        let s_cold = cold.total_strength().unwrap();
        let ratio = s_hot / s_cold;
        let want = FusionReaction::Dt.reactivity_m3_per_s(20.0).unwrap()
            / FusionReaction::Dt.reactivity_m3_per_s(10.0).unwrap();
        assert!((ratio - want).abs() < 1e-3 * want, "{ratio} vs {want}");
    }

    // --- G-flat: sampled moments vs closed forms at δ = 0 ---------------------

    const N: usize = 200_000;

    #[test]
    fn flat_profile_sampled_moments_match_closed_form() {
        // δ = 0, esh = 0: J = κr, volume weight = R·κ·r.
        // <r²> = a²/2 (closed); <Z> = 0; <R> = R0 (flat strength).
        let config = flat_config(1.85, 0.0, 0.0);
        let mut sampler = ParametricSampler::new(config, 11).unwrap();
        let particles = sampler.sample_n(N);
        let a = config.geometry.minor_radius_cm;
        let r0 = config.geometry.major_radius_cm;
        let mut r2 = 0.0;
        let mut z_sum = 0.0;
        let mut big_r = 0.0;
        let kappa = config.geometry.elongation;
        for p in &particles {
            let x = p.position_cm[0];
            let y = p.position_cm[1];
            // Recover the sampled minor radius: at δ = esh = 0 the map
            // gives (R − R0) = r·cos θ and Z = κ·r·sin θ.
            let r = ((x.hypot(y) - r0).powi(2) + (p.position_cm[2] / kappa).powi(2)).sqrt();
            r2 += r * r;
            z_sum += p.position_cm[2];
            big_r += x.hypot(y);
        }
        let tol = 8.0 * a / (2.0_f64 * N as f64).sqrt();
        assert!(
            (r2 / N as f64 - a * a / 2.0).abs() < tol * a,
            "<r²> {}",
            r2 / N as f64
        );
        assert!(z_sum.abs() / (N as f64) < tol);
        // Birth major radius is volume-element weighted: <R> = ∫∫R²J/∫∫RJ
        // = R₀ + a²/(4R₀) exactly at δ = esh = 0 (outboard volume bias).
        let want_r = r0 + a * a / (4.0 * r0);
        assert!(
            (big_r / N as f64 - want_r).abs() < tol,
            "<R> {} vs {want_r}",
            big_r / N as f64
        );
    }

    #[test]
    fn sampled_mean_energy_matches_strength_weighted_temperature() {
        // Global mean birth energy ≈ ⟨μ(T_i(r))⟩ weighted by S·R·J —
        // computed here by independent fine quadrature.
        let config = iter_h_mode();
        let mut sampler = ParametricSampler::new(config, 5).unwrap();
        let particles = sampler.sample_n(N);
        let e_mean: f64 = particles.iter().map(|p| p.energy_mev).sum::<f64>() / N as f64;
        let quad = mean_energy_quadrature(&config);
        let sigma_max = 0.4;
        let tol = 8.0 * sigma_max / (N as f64).sqrt();
        assert!(
            (e_mean - quad).abs() < tol,
            "mean E {e_mean} vs quadrature {quad}"
        );
    }

    /// Independent fine-quadrature reference for the strength-weighted mean
    /// Ballabio mean energy.
    fn mean_energy_quadrature(config: &ParametricPlasmaConfig) -> f64 {
        let g = &config.geometry;
        let n_r = 400;
        let n_t = 400;
        let dr = g.minor_radius_cm / n_r as f64;
        let dt = 2.0 * PI / n_t as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n_r {
            let r = (i as f64 + 0.5) * dr;
            let s = config.strength_density(r).unwrap();
            let ti = config.temperature_kev(r);
            let (mu, _) = config.fuel.moments_mev(ti).unwrap();
            for j in 0..n_t {
                let theta = (j as f64 + 0.5) * dt;
                let w = s * g.volume_element(r, theta) * dr * dt;
                num += w * mu;
                den += w;
            }
        }
        num / den
    }

    #[test]
    fn determinism_and_cell_consistency() {
        let config = iter_h_mode();
        let a = ParametricSampler::new(config, 99).unwrap().sample_n(64);
        let b = ParametricSampler::new(config, 99).unwrap().sample_n(64);
        assert_eq!(a, b);
        let c = ParametricSampler::new(config, 98).unwrap().sample_n(64);
        assert_ne!(a, c);
    }

    // --- G-hist: emission histograms -------------------------------------------

    #[test]
    fn emission_histograms_are_normalized_and_consistent() {
        let config = iter_h_mode();
        let hist = emission_histograms(&config, 21).unwrap();
        for dist in [&hist.radial, &hist.vertical] {
            let sum: f64 = dist.masses.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "masses sum {sum}");
            assert!(dist.centers.windows(2).all(|w| w[0] < w[1]));
            assert!(dist.masses.iter().all(|&m| m >= 0.0));
        }
        // The energy marginal keeps the tabulation truncation: its masses
        // sum to the captured coverage (same convention as the ring/point
        // spectrum tabulation), reported as drift.
        let energy_sum: f64 = hist.energy.masses.iter().sum();
        assert!(
            (energy_sum - hist.energy_coverage).abs() < 1e-12,
            "energy {energy_sum}"
        );
        assert!(hist.energy.centers.windows(2).all(|w| w[0] < w[1]));
        // Radial mass concentrates in the core (peaked H-mode profiles).
        let core_mass: f64 = hist.radial.masses[..7].iter().sum();
        assert!(core_mass > 0.3, "core mass {core_mass}");
        // Vertical span matches the geometry: centers within ±κa.
        let z_max = config.geometry.z_max();
        assert!(hist.vertical.centers.iter().all(|&z| z.abs() <= z_max));
        // Energy spectrum covers the DT line with truncation drift only.
        assert!(hist.energy_coverage > 0.999);
        // Joint correlation distance is a probability metric in [0, 1).
        assert!(hist.joint_correlation >= 0.0 && hist.joint_correlation < 1.0);
    }

    #[test]
    fn invalid_configs_are_loud() {
        let mut c = iter_h_mode();
        c.pedestal_radius_cm = 250.0;
        assert!(matches!(c.validate(), Err(Error::InvalidProfile(_))));
        let mut c = iter_h_mode();
        c.geometry.triangularity = 1.5;
        assert!(matches!(c.validate(), Err(Error::InvalidGeometry(_))));
        let mut c = iter_h_mode();
        c.ion_density.centre_m3 = -1.0;
        assert!(matches!(c.validate(), Err(Error::InvalidProfile(_))));
    }
}
