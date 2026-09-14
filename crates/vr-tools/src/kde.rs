//! Gaussian kernel-density source sampling (KDSource-class, clean-room).
//!
//! [`KdeSampler`] fits an axis-aligned Gaussian KDE over caller-supplied
//! particle vectors (energy/position/direction rows, e.g. projected from
//! MCPL records by the caller) and resamples synthetic particles with the
//! same density to boost downstream statistics. The KDE equations
//! (Gaussian product kernel, Silverman bandwidth rule) are textbook
//! statistics implemented here from scratch; no upstream code or data is
//! read, ported, or vendored.
//!
//! Draws are deterministic in the house style ([`crate::sampling`]:
//! caller-supplied randoms, no RNG inside): [`KdeSampler::draw`] takes one
//! uniform `u` selecting the kernel centre plus one standard normal per
//! dimension. MCPL projection stays caller-side (a ~10-line map from
//! particle fields to rows); this module never depends on `mcpl-io`.

use crate::{Error, Result};

/// sqrt(2π): Gaussian normalization factor (no std const exists).
const SQRT_2PI: f64 = 2.506_628_274_631_000_2;

/// Bandwidth rule for [`KdeSampler::fit`].
#[derive(Debug, Clone, PartialEq)]
pub enum Bandwidth {
    /// Silverman's rule per dimension,
    /// `h = σ (4 / (d + 2) / n)^(1 / (d + 4))` with the unbiased
    /// per-dimension standard deviation `σ`. Zero-variance dimensions are
    /// an [`Error`] (a zero bandwidth is a delta spike, never a density);
    /// use [`Bandwidth::Fixed`] with an explicit width instead.
    Silverman,
    /// Caller-supplied per-dimension widths (all finite and positive).
    Fixed(Vec<f64>),
}

/// Axis-aligned Gaussian KDE over `n` samples in `d` dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct KdeSampler {
    samples: Vec<Vec<f64>>,
    widths: Vec<f64>,
}

impl KdeSampler {
    /// Fit a KDE over `samples` (non-empty, rectangular, all finite).
    pub fn fit(samples: &[Vec<f64>], bandwidth: Bandwidth) -> Result<Self> {
        if samples.is_empty() {
            return Err(Error::EmptyPdf);
        }
        let dims = samples[0].len();
        if dims == 0 {
            return Err(Error::LengthMismatch {
                expected: 1,
                got: 0,
            });
        }
        for (i, row) in samples.iter().enumerate() {
            if row.len() != dims {
                return Err(Error::LengthMismatch {
                    expected: dims,
                    got: row.len(),
                });
            }
            if row.iter().any(|v| !v.is_finite()) {
                return Err(Error::NonFiniteTally {
                    field: "kde sample",
                    index: i,
                });
            }
        }
        let n = samples.len() as f64;
        let d = dims as f64;
        let widths = match bandwidth {
            Bandwidth::Fixed(w) => {
                if w.len() != dims {
                    return Err(Error::LengthMismatch {
                        expected: dims,
                        got: w.len(),
                    });
                }
                if w.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                    return Err(Error::NonFiniteTally {
                        field: "kde bandwidth",
                        index: w
                            .iter()
                            .position(|v| !v.is_finite() || *v <= 0.0)
                            .unwrap_or(0),
                    });
                }
                w
            }
            Bandwidth::Silverman => {
                let factor = (4.0 / (d + 2.0) / n).powf(1.0 / (d + 4.0));
                let mut widths = Vec::with_capacity(dims);
                for j in 0..dims {
                    let mean = samples.iter().map(|row| row[j]).sum::<f64>() / n;
                    let var = samples
                        .iter()
                        .map(|row| (row[j] - mean).powi(2))
                        .sum::<f64>()
                        / (n - 1.0).max(1.0);
                    let sigma = var.sqrt();
                    if sigma <= 0.0 {
                        return Err(Error::ZeroVarianceDim { dim: j });
                    }
                    widths.push(sigma * factor);
                }
                widths
            }
        };
        Ok(Self {
            samples: samples.to_vec(),
            widths,
        })
    }

    /// Number of fitted samples.
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Sample dimensionality.
    pub fn dims(&self) -> usize {
        self.widths.len()
    }

    /// Fitted per-dimension bandwidths.
    pub fn bandwidths(&self) -> &[f64] {
        &self.widths
    }

    /// KDE density at `point` (Gaussian product kernel, length-checked).
    pub fn pdf(&self, point: &[f64]) -> Result<f64> {
        if point.len() != self.dims() {
            return Err(Error::LengthMismatch {
                expected: self.dims(),
                got: point.len(),
            });
        }
        if point.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFiniteTally {
                field: "kde point",
                index: point.iter().position(|v| !v.is_finite()).unwrap_or(0),
            });
        }
        let norm = self.widths.iter().map(|h| h * SQRT_2PI).product::<f64>();
        let mut density = 0.0;
        for row in &self.samples {
            let mut z2 = 0.0;
            for ((x, c), h) in point.iter().zip(row).zip(&self.widths) {
                let z = (x - c) / h;
                z2 += z * z;
            }
            density += (-0.5 * z2).exp();
        }
        Ok(density / norm / self.samples.len() as f64)
    }

    /// Resample one synthetic particle: `u` in `[0, 1)` selects kernel
    /// centre `floor(u * n)`; `normals` (one finite standard normal per
    /// dimension) perturbs it by `width * z` per dimension.
    pub fn draw(&self, u: f64, normals: &[f64]) -> Result<Vec<f64>> {
        if !(0.0..1.0).contains(&u) {
            return Err(Error::BadDraw { value: u });
        }
        if normals.len() != self.dims() {
            return Err(Error::LengthMismatch {
                expected: self.dims(),
                got: normals.len(),
            });
        }
        if normals.iter().any(|v| !v.is_finite()) {
            return Err(Error::NonFiniteTally {
                field: "kde normal",
                index: normals.iter().position(|v| !v.is_finite()).unwrap_or(0),
            });
        }
        let centre = &self.samples[(u * self.samples.len() as f64) as usize];
        Ok(centre
            .iter()
            .zip(&self.widths)
            .zip(normals)
            .map(|((c, h), z)| c + h * z)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic test stream (LCG + Box-Muller); test-only, never shipped.
    struct TestRng(u64);

    impl TestRng {
        fn uniform(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        fn normal(&mut self) -> f64 {
            let (u1, u2) = (self.uniform().max(1e-300), self.uniform());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn gaussian_samples(mean: f64, sigma: f64, n: usize) -> Vec<Vec<f64>> {
        let mut rng = TestRng(0x1234_5678_9abc_def0);
        (0..n).map(|_| vec![mean + sigma * rng.normal()]).collect()
    }

    #[test]
    fn input_errors_are_loud() {
        assert!(KdeSampler::fit(&[], Bandwidth::Silverman).is_err());
        assert!(KdeSampler::fit(&[vec![]], Bandwidth::Silverman).is_err());
        assert!(KdeSampler::fit(&[vec![1.0], vec![1.0, 2.0]], Bandwidth::Silverman).is_err());
        assert!(KdeSampler::fit(&[vec![f64::NAN]], Bandwidth::Silverman).is_err());
        assert!(KdeSampler::fit(&[vec![1.0]], Bandwidth::Fixed(vec![])).is_err());
        assert!(KdeSampler::fit(&[vec![1.0]], Bandwidth::Fixed(vec![0.0])).is_err());
        assert!(KdeSampler::fit(&[vec![1.0]], Bandwidth::Fixed(vec![-2.0])).is_err());
        // Zero variance under Silverman is a delta spike, not a density.
        assert!(KdeSampler::fit(&vec![vec![3.0]; 8], Bandwidth::Silverman).is_err());
        // ...but an explicit fixed width is the caller's choice.
        assert!(KdeSampler::fit(&vec![vec![3.0]; 8], Bandwidth::Fixed(vec![0.5])).is_ok());
    }

    #[test]
    fn draw_is_exact_and_deterministic() {
        let kde = KdeSampler::fit(
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
            Bandwidth::Fixed(vec![0.5, 2.0]),
        )
        .unwrap();
        // u = 0.75 selects centre 1; perturbation is width * z per dim.
        let got = kde.draw(0.75, &[1.0, -0.5]).unwrap();
        assert_eq!(got, vec![3.5, 3.0]);
        assert_eq!(kde.draw(0.75, &[1.0, -0.5]).unwrap(), got);
        assert!(kde.draw(1.0, &[0.0, 0.0]).is_err());
        assert!(kde.draw(-0.1, &[0.0, 0.0]).is_err());
        assert!(kde.draw(0.5, &[0.0]).is_err());
        assert!(kde.draw(0.5, &[0.0, f64::INFINITY]).is_err());
    }

    #[test]
    fn gaussian_recovery_and_normalization() {
        // 20000 draws from N(5, 2²): KDE mean within 5 SE, pdf at the mode
        // within 2% of the closed form, integral within 1e-3 of 1.
        let samples = gaussian_samples(5.0, 2.0, 20_000);
        let kde = KdeSampler::fit(&samples, Bandwidth::Silverman).unwrap();
        let h = kde.bandwidths()[0];
        assert!(h > 0.0 && h < 1.0, "silverman width {h}");
        let mut draws = Vec::with_capacity(4096);
        let mut rng = TestRng(0xabcd);
        for _ in 0..4096 {
            draws.push(kde.draw(rng.uniform(), &[rng.normal()]).unwrap()[0]);
        }
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        let se = 2.0 / (draws.len() as f64).sqrt();
        assert!((mean - 5.0).abs() < 5.0 * se, "mean {mean}");
        let closed = (-0.5 * ((5.0f64 - 5.0) / 2.0).powi(2)).exp() / (2.0 * SQRT_2PI);
        let got = kde.pdf(&[5.0]).unwrap();
        assert!(
            (got - closed).abs() / closed < 0.02,
            "pdf {got} vs {closed}"
        );
        // Trapezoid integral over ±8σ.
        let (lo, hi, m) = (-11.0, 21.0, 2048);
        let mut area = 0.0;
        let mut prev = kde.pdf(&[lo]).unwrap();
        for i in 1..=m {
            let x = lo + (hi - lo) * i as f64 / m as f64;
            let cur = kde.pdf(&[x]).unwrap();
            area += 0.5 * (prev + cur) * (hi - lo) / m as f64;
            prev = cur;
        }
        assert!((area - 1.0).abs() < 1e-3, "integral {area}");
    }

    #[test]
    fn bandwidth_is_deterministic() {
        let samples = gaussian_samples(0.0, 1.0, 512);
        let a = KdeSampler::fit(&samples, Bandwidth::Silverman).unwrap();
        let b = KdeSampler::fit(&samples, Bandwidth::Silverman).unwrap();
        assert_eq!(a.bandwidths(), b.bandwidths());
    }
}
