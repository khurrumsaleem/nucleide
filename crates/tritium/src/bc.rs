//! Surface-law taxonomy: Dirichlet, Sieverts/Henry equilibrium,
//! recombination flux, and zero flux.
//!
//! At each surface (`n` the outward normal, `J = −D∂c_m/∂n` the outward
//! flux): Dirichlet prescribes the mobile concentration, Sieverts (or its
//! linear Henry variant) maps a gas pressure to one through the solubility,
//! recombination prescribes the nonlinear flux `J = K_r c_m²`, and zero flux
//! models a symmetry midplane or an impermeable wall. Sieverts/Henry ends
//! reduce the trap-free steady state to the G1 linear profile (G4); the
//! recombination steady state is the named-open G5 gate.

use crate::error::Error;

/// Surface law at one slab end.
#[derive(Debug, Clone, PartialEq)]
pub enum Boundary {
    /// Prescribed surface concentration `c_m = value` \[mol/m³\] (`>= 0`).
    Dirichlet(f64),
    /// Diatomic-gas equilibrium `c_m = K_S sqrt(p)` with solubility `K_S`
    /// \[mol/m³/Pa¹ᐟ²\] (`> 0`) at gas pressure `p` \[Pa\] (`>= 0`).
    Sieverts {
        /// Solubility `K_S` \[mol/m³/Pa¹ᐟ²\].
        solubility: f64,
        /// Gas pressure `p` \[Pa\].
        pressure: f64,
    },
    /// Linear (Henry) variant `c_m = K_H p` with `K_H` \[mol/m³/Pa\] (`> 0`)
    /// at gas pressure `p` \[Pa\] (`>= 0`).
    Henry {
        /// Solubility `K_H` \[mol/m³/Pa\].
        solubility: f64,
        /// Gas pressure `p` \[Pa\].
        pressure: f64,
    },
    /// Surface recombination flux `J = K_r c_m²` with rate `K_r`
    /// \[m⁴/mol/s\] (`> 0`): nonlinear Robin condition. The steady state
    /// (G5) closes it by Picard iteration on the face value; the transient
    /// stays named-open.
    Recombination {
        /// Recombination rate `K_r` \[m⁴/mol/s\].
        rate: f64,
    },
    /// Symmetry midplane or impermeable wall (`J = 0`).
    ZeroFlux,
}

impl Boundary {
    /// Validate constructor arguments (finite, correctly signed).
    pub fn dirichlet(value: f64) -> Result<Self, Error> {
        if !value.is_finite() || value < 0.0 {
            return Err(Error::BadBoundary(
                "Dirichlet value must be finite and >= 0",
            ));
        }
        Ok(Self::Dirichlet(value))
    }

    /// Validate a Sieverts end (finite, positive solubility, non-negative
    /// pressure).
    pub fn sieverts(solubility: f64, pressure: f64) -> Result<Self, Error> {
        if !solubility.is_finite() || solubility <= 0.0 {
            return Err(Error::BadBoundary(
                "Sieverts solubility must be finite and > 0",
            ));
        }
        if !pressure.is_finite() || pressure < 0.0 {
            return Err(Error::BadBoundary(
                "Sieverts pressure must be finite and >= 0",
            ));
        }
        Ok(Self::Sieverts {
            solubility,
            pressure,
        })
    }

    /// Validate a Henry end (finite, positive solubility, non-negative
    /// pressure).
    pub fn henry(solubility: f64, pressure: f64) -> Result<Self, Error> {
        if !solubility.is_finite() || solubility <= 0.0 {
            return Err(Error::BadBoundary(
                "Henry solubility must be finite and > 0",
            ));
        }
        if !pressure.is_finite() || pressure < 0.0 {
            return Err(Error::BadBoundary("Henry pressure must be finite and >= 0"));
        }
        Ok(Self::Henry {
            solubility,
            pressure,
        })
    }

    /// Validate a recombination end (finite, positive rate).
    pub fn recombination(rate: f64) -> Result<Self, Error> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(Error::BadBoundary(
                "recombination rate must be finite and > 0",
            ));
        }
        Ok(Self::Recombination { rate })
    }

    /// Surface concentration \[mol/m³\] for the equilibrium ends
    /// (Dirichlet value, `K_S sqrt(p)`, `K_H p`); `None` for the flux ends
    /// (recombination, zero flux).
    pub fn surface_concentration(&self) -> Option<f64> {
        match self {
            Boundary::Dirichlet(v) => Some(*v),
            Boundary::Sieverts {
                solubility,
                pressure,
            } => Some(solubility * pressure.sqrt()),
            Boundary::Henry {
                solubility,
                pressure,
            } => Some(solubility * pressure),
            Boundary::Recombination { .. } | Boundary::ZeroFlux => None,
        }
    }

    /// Whether this end is a recombination law (named-open in the
    /// transient; closed by Picard iteration in the steady state).
    pub fn is_recombination(&self) -> bool {
        matches!(self, Boundary::Recombination { .. })
    }

    /// Recombination rate `K_r` for a recombination end; `None` otherwise.
    pub fn recombination_rate(&self) -> Option<f64> {
        match self {
            Boundary::Recombination { rate } => Some(*rate),
            _ => None,
        }
    }

    /// Validate a recombination end from an Arrhenius rate
    /// `K_r = kr0 * exp(-e_r / R / temp)` at face temperature `temp` \[K\]
    /// (same validation as [`crate::params::arrhenius`], then the
    /// positivity check of [`Boundary::recombination`]).
    pub fn recombination_arrhenius(kr0: f64, e_r: f64, temp: f64) -> Result<Self, Error> {
        Self::recombination(crate::params::arrhenius(kr0, e_r, temp)?)
    }

    /// Permeability `Φ = D K_S` \[mol/m/s/Pa¹ᐟ²\] for a Sieverts end at
    /// diffusivity `D` (G4); `None` for every other law.
    pub fn permeability(&self, diffusivity: f64) -> Option<f64> {
        match self {
            Boundary::Sieverts { solubility, .. } => Some(diffusivity * solubility),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_concentrations() {
        assert_eq!(
            Boundary::dirichlet(1.5).unwrap().surface_concentration(),
            Some(1.5)
        );
        // G4 fixture: K_S = 2.0, p1 = 16 Pa -> c0 = 8.0.
        assert_eq!(
            Boundary::sieverts(2.0, 16.0)
                .unwrap()
                .surface_concentration(),
            Some(8.0)
        );
        assert_eq!(
            Boundary::henry(0.5, 4.0).unwrap().surface_concentration(),
            Some(2.0)
        );
        assert_eq!(Boundary::ZeroFlux.surface_concentration(), None);
        assert_eq!(
            Boundary::recombination(1.0)
                .unwrap()
                .surface_concentration(),
            None
        );
        // G4 permeability: D * K_S = 1e-9 * 2.0.
        assert_eq!(
            Boundary::sieverts(2.0, 16.0).unwrap().permeability(1e-9),
            Some(2e-9)
        );
        assert_eq!(Boundary::dirichlet(1.0).unwrap().permeability(1e-9), None);
        assert!(Boundary::recombination(1.0).unwrap().is_recombination());
        assert!(!Boundary::ZeroFlux.is_recombination());
    }

    #[test]
    fn rejects_bad_boundaries() {
        assert!(Boundary::dirichlet(-1.0).is_err());
        assert!(Boundary::dirichlet(f64::NAN).is_err());
        assert!(Boundary::sieverts(0.0, 1.0).is_err());
        assert!(Boundary::sieverts(2.0, -1.0).is_err());
        assert!(Boundary::henry(-0.5, 1.0).is_err());
        assert!(Boundary::henry(0.5, -1.0).is_err());
        assert!(Boundary::recombination(0.0).is_err());
        assert!(Boundary::recombination(f64::INFINITY).is_err());
    }
}
