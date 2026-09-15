//! Miller-geometry flux-surface map and its Jacobian — the parametric
//! tokamak cross-section of Fausser et al., Fus. Eng. Des. **87** (2012)
//! 787 (public form of the ITER/EU-DEMO source model), in the spelling
//! common to that paper and the MIT `openmc-plasma-source` package:
//!
//! ```text
//! Δ(a)   = esh · (1 − (a/a_minor)²)                    (Shafranov shift)
//! R(a,θ) = R₀ + a·cos(θ + δ·sin θ) + Δ(a)
//! Z(a,θ) = κ·a·sin θ
//! ```
//!
//! with major radius `R₀`, minor radius `a ∈ [0, a_minor]`, elongation `κ`,
//! triangularity `δ`, and Shafranov factor `esh`. All lengths in cm.
//!
//! # Volume element (the flagged-risk part)
//!
//! The torus volume element in `(a, θ, φ)` coordinates is
//!
//! ```text
//! dV = R(a,θ) · |J(a,θ)| · da dθ dφ
//! J  = ∂R/∂a·∂Z/∂θ − ∂R/∂θ·∂Z/∂a
//! ```
//!
//! — the poloidal Jacobian times the toroidal radius `R` (without the `R`
//! factor the core is over-represented). The analytic gates pinned here:
//!
//! - `κ=1, δ=0, esh=0`: `J == r` and `dV ∝ r` exactly (plain torus limit).
//! - `δ=0` (any `κ`, `esh`): `J == κ·r·(1 − 2·esh·r·cos θ/a_minor²)`
//!   exactly, the poloidal cross-section area is `π·κ·a²`, and the torus
//!   volume is `2·π²·κ·a²·R₀` — the Shafranov terms integrate out, so the
//!   shift moves surfaces outboard without changing the volume.
//! - general `δ`: `J` matches central differences of the map.
//!
//! Profiles are *not* part of this module — see [`crate::profile`].

use crate::{Error, Result};

/// Miller tokamak geometry (flux-surface shape parameters).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MillerGeometry {
    /// Plasma major radius `R₀` \[cm\]; must exceed `minor_radius + |esh|`.
    pub major_radius_cm: f64,
    /// Plasma minor radius `a_minor` \[cm\]; strictly positive.
    pub minor_radius_cm: f64,
    /// Elongation `κ` (dimensionless); strictly positive.
    pub elongation: f64,
    /// Triangularity `δ` (dimensionless); must lie in `(-1, 1)`.
    pub triangularity: f64,
    /// Shafranov factor `esh` \[cm\]; must satisfy `|esh| < 0.5·a_minor`.
    pub shafranov_factor_cm: f64,
}

impl MillerGeometry {
    /// Validate the geometry; errors are loud and named.
    pub fn validate(&self) -> Result<()> {
        let (r0, a, kappa, delta, esh) = (
            self.major_radius_cm,
            self.minor_radius_cm,
            self.elongation,
            self.triangularity,
            self.shafranov_factor_cm,
        );
        for (field, value) in [
            ("major radius", r0),
            ("minor radius", a),
            ("elongation", kappa),
            ("triangularity", delta),
            ("shafranov factor", esh),
        ] {
            if !value.is_finite() {
                return Err(Error::InvalidGeometry(field));
            }
        }
        if a <= 0.0 {
            return Err(Error::InvalidGeometry("minor radius must be > 0"));
        }
        if r0 <= 0.0 {
            return Err(Error::InvalidGeometry("major radius must be > 0"));
        }
        if kappa <= 0.0 {
            return Err(Error::InvalidGeometry("elongation must be > 0"));
        }
        if !(delta > -1.0 && delta < 1.0) {
            return Err(Error::InvalidGeometry("triangularity must lie in (-1, 1)"));
        }
        if esh.abs() >= 0.5 * a {
            return Err(Error::InvalidGeometry(
                "|shafranov factor| must be < 0.5 * minor radius",
            ));
        }
        if r0 <= a + esh.abs() {
            return Err(Error::InvalidGeometry(
                "major radius must exceed minor radius + |shafranov factor|",
            ));
        }
        Ok(())
    }

    /// Shafranov shift `Δ(a) = esh·(1 − (a/a_minor)²)` \[cm\].
    pub fn shafranov_shift(&self, a: f64) -> f64 {
        self.shafranov_factor_cm * (1.0 - (a / self.minor_radius_cm).powi(2))
    }

    /// Forward map `(a, θ) → (R, Z)` \[cm\]. `a` is the minor-radius label
    /// and `θ` the poloidal angle (radians).
    pub fn map(&self, a: f64, theta: f64) -> (f64, f64) {
        let theta_m = theta + self.triangularity * theta.sin();
        let r = self.major_radius_cm + a * theta_m.cos() + self.shafranov_shift(a);
        let z = self.elongation * a * theta.sin();
        (r, z)
    }

    /// Poloidal Jacobian `J = ∂R/∂a·∂Z/∂θ − ∂R/∂θ·∂Z/∂a` (closed form).
    ///
    /// With `θ_m = θ + δ·sin θ`:
    /// `∂R/∂a = cos θ_m − 2·esh·a/a_minor²`,
    /// `∂R/∂θ = −a·sin θ_m·(1 + δ·cos θ)`,
    /// `∂Z/∂a = κ·sin θ`, `∂Z/∂θ = κ·a·cos θ`.
    pub fn jacobian(&self, a: f64, theta: f64) -> f64 {
        let delta = self.triangularity;
        let kappa = self.elongation;
        let esh = self.shafranov_factor_cm;
        let a_minor = self.minor_radius_cm;
        let theta_m = theta + delta * theta.sin();
        let dtheta_dtheta = 1.0 + delta * theta.cos();
        let dr_da = theta_m.cos() - 2.0 * esh * a / (a_minor * a_minor);
        let dr_dtheta = -a * theta_m.sin() * dtheta_dtheta;
        let dz_da = kappa * theta.sin();
        let dz_dtheta = kappa * a * theta.cos();
        dr_da * dz_dtheta - dr_dtheta * dz_da
    }

    /// Absolute poloidal Jacobian (the map can fold for extreme parameters;
    /// the volume element uses `|J|`).
    pub fn jacobian_abs(&self, a: f64, theta: f64) -> f64 {
        self.jacobian(a, theta).abs()
    }

    /// Torus volume element `dV/(da dθ dφ) = R·|J|`.
    pub fn volume_element(&self, a: f64, theta: f64) -> f64 {
        let (r, _) = self.map(a, theta);
        r * self.jacobian_abs(a, theta)
    }

    /// Vertical extent: `max |Z| = κ·a_minor` (the map carries `Z = κ·a·sin θ`).
    pub fn z_max(&self) -> f64 {
        self.elongation * self.minor_radius_cm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// ITER-ish hand geometry (round numbers, cm).
    fn iter_like() -> MillerGeometry {
        MillerGeometry {
            major_radius_cm: 620.0,
            minor_radius_cm: 200.0,
            elongation: 1.85,
            triangularity: 0.35,
            shafranov_factor_cm: 15.0,
        }
    }

    /// Central-difference reference for the Jacobian.
    fn jacobian_fd(g: &MillerGeometry, a: f64, theta: f64, h: f64) -> f64 {
        let d = |f: &dyn Fn(f64, f64) -> f64, x: f64, y: f64, dx: f64, dy: f64| {
            (f(x + dx, y + dy) - f(x - dx, y - dy)) / (2.0 * (if dx != 0.0 { dx } else { dy }))
        };
        let r = |a: f64, t: f64| g.map(a, t).0;
        let z = |a: f64, t: f64| g.map(a, t).1;
        let dr_da = d(&r, a, theta, h, 0.0);
        let dr_dt = d(&r, a, theta, 0.0, h);
        let dz_da = d(&z, a, theta, h, 0.0);
        let dz_dt = d(&z, a, theta, 0.0, h);
        dr_da * dz_dt - dr_dt * dz_da
    }

    // --- G-tor: torus limit -------------------------------------------------

    #[test]
    fn torus_limit_jacobian_is_exactly_r() {
        let g = MillerGeometry {
            elongation: 1.0,
            triangularity: 0.0,
            shafranov_factor_cm: 0.0,
            ..iter_like()
        };
        g.validate().unwrap();
        for i in 0..17 {
            let a = 200.0 * i as f64 / 16.0;
            for j in 0..24 {
                let theta = 2.0 * PI * j as f64 / 24.0;
                assert!(
                    (g.jacobian(a, theta) - a).abs() < 1e-9 * a.max(1.0),
                    "a={a} θ={theta}"
                );
            }
        }
    }

    #[test]
    fn no_triangularity_jacobian_closed_form() {
        // δ = 0: J = κ·a·(1 − 2·esh·a·cosθ/a_minor²) exactly (the esh=0
        // specialization is J == κ·a, the plain scaled-torus limit).
        for (kappa, esh) in [(1.85, 15.0), (1.0, 0.0), (2.4, -8.0)] {
            let g = MillerGeometry {
                elongation: kappa,
                triangularity: 0.0,
                shafranov_factor_cm: esh,
                ..iter_like()
            };
            let am2 = g.minor_radius_cm.powi(2);
            for i in 1..=16 {
                let a = 200.0 * i as f64 / 16.0;
                for j in 0..16 {
                    let theta = 2.0 * PI * j as f64 / 16.0;
                    let want = kappa * a * (1.0 - 2.0 * esh * a * theta.cos() / am2);
                    assert!(
                        (g.jacobian(a, theta) - want).abs() < 1e-9 * want.abs().max(1.0),
                        "κ={kappa} esh={esh}: a={a} θ={theta}"
                    );
                }
            }
        }
    }

    // --- G-area / G-volume: closed forms at δ = 0 ---------------------------

    /// Simpson integrate `f` over [0, 2π].
    fn simpson_periodic(f: impl Fn(f64) -> f64, n: usize) -> f64 {
        let h = 2.0 * PI / n as f64;
        let mut sum = f(0.0) + f(2.0 * PI);
        for i in 1..n {
            let x = i as f64 * h;
            sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
        }
        sum * h / 3.0
    }

    #[test]
    fn cross_section_area_is_exactly_pi_kappa_a2() {
        for (kappa, esh) in [(1.85, 15.0), (1.0, 0.0), (2.4, -8.0)] {
            let g = MillerGeometry {
                elongation: kappa,
                triangularity: 0.0,
                shafranov_factor_cm: esh,
                ..iter_like()
            };
            // Area = ∫∫ J da dθ (no R factor — poloidal plane).
            let n_r = 200;
            let h = g.minor_radius_cm / n_r as f64;
            let mut area = 0.0;
            for i in 0..n_r {
                let a = (i as f64 + 0.5) * h;
                let edge = simpson_periodic(|t| g.jacobian(a, t), 96);
                area += edge * h;
            }
            let want = PI * kappa * g.minor_radius_cm.powi(2);
            assert!(
                (area - want).abs() < 1e-6 * want,
                "κ={kappa} esh={esh}: {area} vs {want}"
            );
        }
    }

    #[test]
    fn torus_volume_closed_form_any_shafranov_shift() {
        // V = 2π²·κ·a²·R₀ exactly at δ = 0 for ANY esh: the Shafranov terms
        // (∫∫ Δ·J and the esh part of J) cancel over the poloidal integral
        // (see module docs; verified here at three shifts, signs included).
        for (kappa, esh) in [(1.85, 15.0), (1.0, 0.0), (2.4, -8.0)] {
            let g = MillerGeometry {
                elongation: kappa,
                triangularity: 0.0,
                shafranov_factor_cm: esh,
                ..iter_like()
            };
            let n_r = 200;
            let h = g.minor_radius_cm / n_r as f64;
            let mut integral = 0.0;
            for i in 0..n_r {
                let a = (i as f64 + 0.5) * h;
                let edge = simpson_periodic(|t| g.volume_element(a, t), 96);
                integral += edge * h;
            }
            let want = 2.0 * PI.powi(2) * kappa * g.minor_radius_cm.powi(2) * g.major_radius_cm;
            let volume = 2.0 * PI * integral; // the toroidal integral
            assert!(
                (volume - want).abs() < 1e-6 * want,
                "κ={kappa} esh={esh}: {volume} vs {want}"
            );
        }
    }

    // --- G-fd: general Jacobian vs finite differences ------------------------

    #[test]
    fn triangular_jacobian_matches_finite_differences() {
        let g = iter_like();
        let h = 1e-6 * g.minor_radius_cm;
        for &(a, theta) in &[(50.0, 0.3), (120.0, 1.9), (200.0, 4.2), (175.0, 5.9)] {
            let closed = g.jacobian(a, theta);
            let fd = jacobian_fd(&g, a, theta, h);
            assert!(
                (closed - fd).abs() < 1e-5 * fd.abs().max(1.0),
                "a={a} θ={theta}: closed {closed} vs fd {fd}"
            );
        }
    }

    // --- G-sym: up-down symmetry ---------------------------------------------

    #[test]
    fn volume_weight_is_up_down_symmetric() {
        let g = iter_like();
        for &(a, theta) in &[(37.0, 0.8), (150.0, 2.6), (200.0, 5.1)] {
            let up = g.volume_element(a, theta);
            let down = g.volume_element(a, -theta);
            assert!((up - down).abs() < 1e-9 * up.max(1.0), "a={a} θ={theta}");
        }
    }

    // --- G-map: hand golden of the forward map --------------------------------

    #[test]
    fn map_matches_hand_golden() {
        // Independently recomputed from the published formulas (float64).
        let g = iter_like();
        let (r, z) = g.map(120.0, 1.1);
        assert!((r - 648.584_749_021_081_5).abs() < 1e-9, "R {r}");
        assert!((z - 197.848_033_933_638_67).abs() < 1e-9, "Z {z}");
        let j = g.jacobian(120.0, 1.1);
        assert!((j - 233.239_118_184_499_06).abs() < 1e-7, "J {j}");
    }

    #[test]
    fn invalid_geometry_is_loud() {
        let mut g = iter_like();
        g.minor_radius_cm = 0.0;
        assert_eq!(
            g.validate(),
            Err(Error::InvalidGeometry("minor radius must be > 0"))
        );
        let mut g = iter_like();
        g.triangularity = 1.0;
        assert!(matches!(g.validate(), Err(Error::InvalidGeometry(_))));
        let mut g = iter_like();
        g.shafranov_factor_cm = 101.0;
        assert!(matches!(g.validate(), Err(Error::InvalidGeometry(_))));
        let mut g = iter_like();
        g.major_radius_cm = 210.0; // < a + esh
        assert!(matches!(g.validate(), Err(Error::InvalidGeometry(_))));
        let mut g = iter_like();
        g.major_radius_cm = f64::NAN;
        assert!(matches!(g.validate(), Err(Error::InvalidGeometry(_))));
    }
}
