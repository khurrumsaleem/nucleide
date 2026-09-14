//! Transport and trap data (T-Arr): caller-supplied diffusivities, trap
//! kinetics, site densities, temperature profile, and volumetric source.
//!
//! The kernel ships **no material property tables** — the same stance as
//! kinetics delayed data. Every coefficient is caller data: either a plain
//! constant or an Arrhenius law evaluated in the caller-supplied
//! temperature. Trap site densities `N_j` are static in v1 (extrinsic
//! traps); damage-created intrinsic trap evolution is out of scope.

use crate::error::Error;

/// Universal gas constant `R` \[J/mol/K\] used by the Arrhenius laws (T-Arr).
pub const GAS_CONSTANT: f64 = 8.314;

/// Evaluate an Arrhenius law `pre * exp(-energy / R / temp)`.
///
/// Rejects non-finite or non-positive `pre`, non-finite or negative
/// `energy`, and non-finite or non-positive `temp`.
pub fn arrhenius(pre: f64, energy: f64, temp: f64) -> Result<f64, Error> {
    if !pre.is_finite() || pre <= 0.0 {
        return Err(Error::BadData(
            "Arrhenius pre-factor must be finite and > 0",
        ));
    }
    if !energy.is_finite() || energy < 0.0 {
        return Err(Error::BadData("Arrhenius energy must be finite and >= 0"));
    }
    if !temp.is_finite() || temp <= 0.0 {
        return Err(Error::BadData("temperature must be finite and > 0"));
    }
    Ok(pre * (-energy / GAS_CONSTANT / temp).exp())
}

/// One extrinsic McNabb–Foster trap species (T2).
///
/// Trapping/detrapping rates are caller data: either plain constants
/// (`*_energy = 0` with the `*_0` value) or Arrhenius laws in the
/// caller-supplied temperature. The site density `N_j` is static in v1.
#[derive(Debug, Clone, PartialEq)]
pub struct TrapSpec {
    /// Trapping pre-factor `k_{j0}` \[m³/mol/s\] (`> 0`).
    pub k0: f64,
    /// Trapping activation energy `E_{kj}` \[J/mol\] (`>= 0`).
    pub e_k: f64,
    /// Detrapping pre-factor `p_{j0}` \[1/s\] (`> 0`).
    pub p0: f64,
    /// Detrapping activation energy `E_{pj}` \[J/mol\] (`>= 0`).
    pub e_p: f64,
    /// Static site density `N_j` \[mol/m³\] (`> 0`).
    pub site_density: f64,
}

impl TrapSpec {
    /// Validate a trap spec (finite, positive rates and site density,
    /// non-negative energies).
    pub fn new(k0: f64, e_k: f64, p0: f64, e_p: f64, site_density: f64) -> Result<Self, Error> {
        if !k0.is_finite() || k0 <= 0.0 {
            return Err(Error::BadTrap("k0 must be finite and > 0"));
        }
        if !e_k.is_finite() || e_k < 0.0 {
            return Err(Error::BadTrap("E_k must be finite and >= 0"));
        }
        if !p0.is_finite() || p0 <= 0.0 {
            return Err(Error::BadTrap("p0 must be finite and > 0"));
        }
        if !e_p.is_finite() || e_p < 0.0 {
            return Err(Error::BadTrap("E_p must be finite and >= 0"));
        }
        if !site_density.is_finite() || site_density <= 0.0 {
            return Err(Error::BadTrap("site density must be finite and > 0"));
        }
        Ok(Self {
            k0,
            e_k,
            p0,
            e_p,
            site_density,
        })
    }

    /// Trapping (`k`) and detrapping (`p`) rates at temperature `temp` \[K\].
    pub fn rates(&self, temp: f64) -> Result<(f64, f64), Error> {
        Ok((
            arrhenius(self.k0, self.e_k, temp)?,
            arrhenius(self.p0, self.e_p, temp)?,
        ))
    }

    /// Equilibrium constant `K = k / p` at temperature `temp` \[K\].
    pub fn equilibrium_constant(&self, temp: f64) -> Result<f64, Error> {
        let (k, p) = self.rates(temp)?;
        Ok(k / p)
    }
}

/// 1D slab transport data over `0 ≤ x ≤ L` with `cells` finite volumes.
///
/// Diffusivity is caller data (`d0`, constant when `e_d = 0`, else
/// Arrhenius in the caller-supplied temperature). The temperature profile
/// is caller data: either uniform or one value per cell. The volumetric
/// source `S` defaults to zero (uniform or one value per cell).
#[derive(Debug, Clone, PartialEq)]
pub struct TransportParams {
    /// Slab length `L` \[m\] (`> 0`).
    pub length: f64,
    /// Cell count (`>= 2`).
    pub cells: usize,
    /// Diffusivity pre-factor `D_0` \[m²/s\] (`> 0`).
    pub d0: f64,
    /// Diffusivity activation energy `E_D` \[J/mol\] (`>= 0`).
    pub e_d: f64,
    /// Trap species (possibly empty: pure Fickian diffusion).
    pub traps: Vec<TrapSpec>,
    /// Temperature \[K\]: one value (uniform) or one per cell.
    pub temperature: Vec<f64>,
    /// Volumetric source `S` \[mol/m³/s\]: empty (zero), one value
    /// (uniform), or one per cell.
    pub source: Vec<f64>,
}

impl TransportParams {
    /// Validate slab data: positive length, at least two cells, finite
    /// positive diffusivity data, finite positive temperatures (uniform or
    /// per-cell), and a zero/uniform/per-cell source.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        length: f64,
        cells: usize,
        d0: f64,
        e_d: f64,
        traps: Vec<TrapSpec>,
        temperature: Vec<f64>,
        source: Vec<f64>,
    ) -> Result<Self, Error> {
        if !length.is_finite() || length <= 0.0 {
            return Err(Error::BadData("length must be finite and > 0"));
        }
        if cells < 2 {
            return Err(Error::BadGrid("need at least 2 cells"));
        }
        if !d0.is_finite() || d0 <= 0.0 {
            return Err(Error::BadData("D0 must be finite and > 0"));
        }
        if !e_d.is_finite() || e_d < 0.0 {
            return Err(Error::BadData("E_D must be finite and >= 0"));
        }
        if temperature.len() != 1 && temperature.len() != cells {
            return Err(Error::BadData(
                "temperature must be uniform (1 value) or per-cell",
            ));
        }
        if temperature.iter().any(|t| !t.is_finite() || *t <= 0.0) {
            return Err(Error::BadData("temperatures must be finite and > 0"));
        }
        if !source.is_empty() && source.len() != 1 && source.len() != cells {
            return Err(Error::BadData(
                "source must be empty (zero), uniform (1 value), or per-cell",
            ));
        }
        if source.iter().any(|s| !s.is_finite()) {
            return Err(Error::BadData("source values must be finite"));
        }
        Ok(Self {
            length,
            cells,
            d0,
            e_d,
            traps,
            temperature,
            source,
        })
    }

    /// Cell width `dx = L / cells` \[m\].
    pub fn dx(&self) -> f64 {
        self.length / self.cells as f64
    }

    /// Cell-centre positions \[m\] (`(i + 1/2) dx`).
    pub fn cell_centres(&self) -> Vec<f64> {
        let dx = self.dx();
        (0..self.cells).map(|i| (i as f64 + 0.5) * dx).collect()
    }

    /// Temperature at cell `i` \[K\] (uniform broadcasts).
    pub fn temp_at(&self, i: usize) -> f64 {
        if self.temperature.len() == 1 {
            self.temperature[0]
        } else {
            self.temperature[i]
        }
    }

    /// Diffusivity at cell `i` \[m²/s\].
    pub fn diffusivity_at(&self, i: usize) -> Result<f64, Error> {
        arrhenius(self.d0, self.e_d, self.temp_at(i))
    }

    /// Diffusivity at every cell \[m²/s\].
    pub fn diffusivities(&self) -> Result<Vec<f64>, Error> {
        (0..self.cells).map(|i| self.diffusivity_at(i)).collect()
    }

    /// Source at cell `i` \[mol/m³/s\] (empty means zero, uniform broadcasts).
    pub fn source_at(&self, i: usize) -> f64 {
        if self.source.is_empty() {
            0.0
        } else if self.source.len() == 1 {
            self.source[0]
        } else {
            self.source[i]
        }
    }

    /// Trap `(k, p)` rates at cell `i` for every species.
    pub fn trap_rates_at(&self, i: usize) -> Result<Vec<(f64, f64)>, Error> {
        let t = self.temp_at(i);
        self.traps.iter().map(|trap| trap.rates(t)).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn no_traps(length: f64, cells: usize, d: f64) -> TransportParams {
        TransportParams::new(length, cells, d, 0.0, vec![], vec![500.0], vec![]).unwrap()
    }

    #[test]
    fn arrhenius_constant_when_zero_energy() {
        assert_eq!(arrhenius(1e-9, 0.0, 500.0).unwrap(), 1e-9);
        let v = arrhenius(1e-7, 20_000.0, 1000.0).unwrap();
        let want = 1e-7 * (-20_000.0 / GAS_CONSTANT / 1000.0).exp();
        assert!((v - want).abs() / want < 1e-15);
    }

    #[test]
    fn arrhenius_rejects_bad_inputs() {
        assert!(arrhenius(0.0, 0.0, 500.0).is_err());
        assert!(arrhenius(f64::NAN, 0.0, 500.0).is_err());
        assert!(arrhenius(1e-9, -1.0, 500.0).is_err());
        assert!(arrhenius(1e-9, 0.0, 0.0).is_err());
        assert!(arrhenius(1e-9, 0.0, f64::INFINITY).is_err());
    }

    #[test]
    fn trap_spec_rates_and_equilibrium() {
        let trap = TrapSpec::new(0.05, 0.0, 0.01, 0.0, 2.0).unwrap();
        assert_eq!(trap.rates(500.0).unwrap(), (0.05, 0.01));
        assert_eq!(trap.equilibrium_constant(500.0).unwrap(), 5.0);
        assert!(TrapSpec::new(0.0, 0.0, 0.01, 0.0, 2.0).is_err());
        assert!(TrapSpec::new(0.05, 0.0, 0.0, 0.0, 2.0).is_err());
        assert!(TrapSpec::new(0.05, 0.0, 0.01, -1.0, 2.0).is_err());
        assert!(TrapSpec::new(0.05, 0.0, 0.01, 0.0, 0.0).is_err());
        assert!(TrapSpec::new(0.05, -1.0, 0.01, 0.0, 2.0).is_err());
    }

    #[test]
    fn params_geometry_and_broadcasts() {
        let p = TransportParams::new(1e-3, 4, 1e-9, 0.0, vec![], vec![500.0], vec![1.0]).unwrap();
        assert_eq!(p.dx(), 2.5e-4);
        assert_eq!(p.cell_centres().len(), 4);
        assert_eq!(p.diffusivity_at(2).unwrap(), 1e-9);
        assert_eq!(p.source_at(0), 1.0);
        let z = no_traps(1e-3, 4, 1e-9);
        assert_eq!(z.source_at(3), 0.0);
        assert_eq!(z.temp_at(1), 500.0);
    }

    #[test]
    fn per_cell_temperature_and_source_broadcast() {
        let p = TransportParams::new(
            1e-3,
            4,
            1e-9,
            0.0,
            vec![],
            vec![500.0, 600.0, 700.0, 800.0],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();
        assert_eq!(p.temp_at(2), 700.0);
        assert_eq!(p.diffusivity_at(2).unwrap(), 1e-9);
        assert_eq!(p.source_at(3), 4.0);
        assert_eq!(p.trap_rates_at(0).unwrap(), vec![]);
    }

    #[test]
    fn params_rejects_bad_inputs() {
        assert!(TransportParams::new(0.0, 4, 1e-9, 0.0, vec![], vec![500.0], vec![]).is_err());
        assert!(TransportParams::new(1e-3, 1, 1e-9, 0.0, vec![], vec![500.0], vec![]).is_err());
        assert!(TransportParams::new(1e-3, 4, 0.0, 0.0, vec![], vec![500.0], vec![]).is_err());
        assert!(TransportParams::new(1e-3, 4, 1e-9, -1.0, vec![], vec![500.0], vec![]).is_err());
        assert!(
            TransportParams::new(1e-3, 4, 1e-9, 0.0, vec![], vec![500.0, 600.0], vec![]).is_err()
        );
        assert!(TransportParams::new(1e-3, 4, 1e-9, 0.0, vec![], vec![-5.0], vec![]).is_err());
        assert!(
            TransportParams::new(1e-3, 4, 1e-9, 0.0, vec![], vec![500.0], vec![1.0, 2.0]).is_err()
        );
        assert!(
            TransportParams::new(1e-3, 4, 1e-9, 0.0, vec![], vec![500.0], vec![f64::NAN]).is_err()
        );
    }
}
