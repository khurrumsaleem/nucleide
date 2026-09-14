//! Real tridiagonal solve (Thomas algorithm, O(N)) over caller vectors.
//!
//! Shared kernel: solves `A x = d` for a tridiagonal `A` with sub-diagonal
//! `a` (length `n - 1`), diagonal `b` (length `n`), and super-diagonal `c`
//! (length `n - 1`). The forward-elimination / back-substitution pass is
//! hand-rolled (no backend call): the implicit 1D finite-volume stepper in
//! `nucleide-tritium` needs one `O(N)` solve per time step, and routing that
//! through the complex sparse LU core would pay symbolic/numeric overhead per
//! step for a real symmetric-positive-definite system with a closed form.
//!
//! Consumers: `nucleide-tritium` (`solve.rs` builds the theta-method systems
//! and the trap-free steady system, then calls [`solve`] here). No evaluated
//! data lives here: every entry is a caller input, validated for shape and
//! finiteness before the elimination runs.

/// Errors surfaced by [`solve`]. Every rejection names its cause.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TridiagError {
    /// No unknowns supplied (`b` empty).
    Empty,
    /// `a` length is not `n - 1` (`n` unknowns).
    SubLength {
        /// Unknown count.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// `c` length is not `n - 1` (`n` unknowns).
    SuperLength {
        /// Unknown count.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// `d` length does not match the `b` (unknown) count.
    RhsLength {
        /// Unknown count.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// A non-finite value in the named input (`"a"`, `"b"`, `"c"`, `"d"`).
    NonFinite(&'static str),
    /// A zero (or denormal-vanishing) pivot at the given elimination stage:
    /// the system is singular.
    ZeroPivot {
        /// Elimination stage (0-based row index).
        stage: usize,
    },
}

impl std::fmt::Display for TridiagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TridiagError::Empty => write!(f, "tridiag: no unknowns supplied"),
            TridiagError::SubLength { expected, got } => write!(
                f,
                "tridiag: sub-diagonal length {got} does not match {expected} unknowns"
            ),
            TridiagError::SuperLength { expected, got } => write!(
                f,
                "tridiag: super-diagonal length {got} does not match {expected} unknowns"
            ),
            TridiagError::RhsLength { expected, got } => write!(
                f,
                "tridiag: right-hand side length {got} does not match {expected} unknowns"
            ),
            TridiagError::NonFinite(name) => {
                write!(f, "tridiag: non-finite value in `{name}`")
            }
            TridiagError::ZeroPivot { stage } => {
                write!(f, "tridiag: zero pivot at elimination stage {stage}")
            }
        }
    }
}

impl std::error::Error for TridiagError {}

/// Result alias for the `tridiag` module.
pub type Result<T> = std::result::Result<T, TridiagError>;

/// Solve the tridiagonal system `A x = d` by the Thomas algorithm (O(N)).
///
/// `a[i]` is the sub-diagonal entry coupling row `i + 1` to column `i`
/// (length `n - 1`), `b` the diagonal (length `n`), `c[i]` the
/// super-diagonal entry coupling row `i` to column `i + 1` (length
/// `n - 1`), and `d` the right-hand side (length `n`). Inputs are borrowed;
/// the elimination works on local copies. Rejects empty systems, length
/// mismatches, non-finite entries, and vanishing pivots.
pub fn solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
    let n = b.len();
    if n == 0 {
        return Err(TridiagError::Empty);
    }
    if a.len() + 1 != n {
        return Err(TridiagError::SubLength {
            expected: n,
            got: a.len(),
        });
    }
    if c.len() + 1 != n {
        return Err(TridiagError::SuperLength {
            expected: n,
            got: c.len(),
        });
    }
    if d.len() != n {
        return Err(TridiagError::RhsLength {
            expected: n,
            got: d.len(),
        });
    }
    for (values, name) in [(a, "a"), (b, "b"), (c, "c"), (d, "d")] {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(TridiagError::NonFinite(name));
        }
    }
    if n == 1 {
        if b[0] == 0.0 {
            return Err(TridiagError::ZeroPivot { stage: 0 });
        }
        return Ok(vec![d[0] / b[0]]);
    }
    // Forward elimination: fold the sub-diagonal into modified super-diagonal
    // `cp` and right-hand side `dp`.
    let mut cp = vec![0.0; n - 1];
    let mut dp = vec![0.0; n];
    let mut denom = b[0];
    if denom == 0.0 {
        return Err(TridiagError::ZeroPivot { stage: 0 });
    }
    cp[0] = c[0] / denom;
    dp[0] = d[0] / denom;
    for i in 1..n {
        denom = b[i] - a[i - 1] * cp[i - 1];
        if denom == 0.0 {
            return Err(TridiagError::ZeroPivot { stage: i });
        }
        if i < n - 1 {
            cp[i] = c[i] / denom;
        }
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom;
    }
    // Back substitution.
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn residual(a: &[f64], b: &[f64], c: &[f64], x: &[f64], d: &[f64]) -> f64 {
        let n = b.len();
        let mut worst = 0.0_f64;
        for i in 0..n {
            let mut row = b[i] * x[i];
            if i > 0 {
                row += a[i - 1] * x[i - 1];
            }
            if i + 1 < n {
                row += c[i] * x[i + 1];
            }
            worst = worst.max((row - d[i]).abs());
        }
        worst
    }

    #[test]
    fn solves_diagonally_dominant_system() {
        // [-1 2 -1] Laplacian row pattern with x = [1, 2, 3, 4].
        let n = 4;
        let a = vec![-1.0; n - 1];
        let b = vec![2.0; n];
        let c = vec![-1.0; n - 1];
        let x_want = vec![1.0, 2.0, 3.0, 4.0];
        let d = vec![0.0, 0.0, 0.0, 5.0];
        let x = solve(&a, &b, &c, &d).unwrap();
        for (got, want) in x.iter().zip(&x_want) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn solves_nonsymmetric_system() {
        // Hand-built 5x5 with distinct diagonals; gate is the residual.
        let a = vec![0.5, -1.5, 2.0, -0.25];
        let b = vec![4.0, 3.0, 5.0, 2.0, 4.0];
        let c = vec![-1.0, 0.75, -2.0, 1.25];
        let d = vec![1.0, -2.0, 3.0, 0.5, -1.0];
        let x = solve(&a, &b, &c, &d).unwrap();
        assert!(residual(&a, &b, &c, &x, &d) < 1e-12);
    }

    #[test]
    fn handles_single_unknown() {
        let x = solve(&[], &[2.0], &[], &[6.0]).unwrap();
        assert_eq!(x, vec![3.0]);
        assert!(solve(&[], &[0.0], &[], &[6.0]).is_err());
    }

    #[test]
    fn rejects_bad_shapes_and_values() {
        let a = vec![-1.0; 2];
        let b = vec![2.0; 3];
        let c = vec![-1.0; 2];
        let d = vec![1.0; 3];
        assert_eq!(
            solve(&a[..1], &b, &c, &d),
            Err(TridiagError::SubLength {
                expected: 3,
                got: 1
            })
        );
        assert_eq!(
            solve(&a, &b, &c[..1], &d),
            Err(TridiagError::SuperLength {
                expected: 3,
                got: 1
            })
        );
        assert_eq!(
            solve(&a, &b, &c, &d[..2]),
            Err(TridiagError::RhsLength {
                expected: 3,
                got: 2
            })
        );
        assert_eq!(solve(&[], &[], &[], &[]), Err(TridiagError::Empty));
        assert_eq!(
            solve(&a, &b, &c, &[1.0, f64::NAN, 3.0]),
            Err(TridiagError::NonFinite("d"))
        );
        assert_eq!(
            solve(&a, &[f64::INFINITY, 2.0, 2.0], &c, &d),
            Err(TridiagError::NonFinite("b"))
        );
        // Singular: zero leading pivot.
        assert_eq!(
            solve(&[1.0], &[0.0, 1.0], &[1.0], &[1.0, 2.0]),
            Err(TridiagError::ZeroPivot { stage: 0 })
        );
        // Singular mid-elimination: rows [1 1; 1 1] hit a zero denominator.
        assert_eq!(
            solve(&[1.0], &[1.0, 1.0], &[1.0], &[2.0, 2.0]),
            Err(TridiagError::ZeroPivot { stage: 1 })
        );
    }

    #[test]
    fn error_display_strings() {
        assert_eq!(
            TridiagError::Empty.to_string(),
            "tridiag: no unknowns supplied"
        );
        assert_eq!(
            TridiagError::SubLength {
                expected: 3,
                got: 1
            }
            .to_string(),
            "tridiag: sub-diagonal length 1 does not match 3 unknowns"
        );
        assert_eq!(
            TridiagError::SuperLength {
                expected: 3,
                got: 1
            }
            .to_string(),
            "tridiag: super-diagonal length 1 does not match 3 unknowns"
        );
        assert_eq!(
            TridiagError::RhsLength {
                expected: 3,
                got: 2
            }
            .to_string(),
            "tridiag: right-hand side length 2 does not match 3 unknowns"
        );
        assert_eq!(
            TridiagError::NonFinite("c").to_string(),
            "tridiag: non-finite value in `c`"
        );
        assert_eq!(
            TridiagError::ZeroPivot { stage: 1 }.to_string(),
            "tridiag: zero pivot at elimination stage 1"
        );
    }

    #[test]
    fn error_implements_std_error_trait() {
        let err: &dyn std::error::Error = &TridiagError::Empty;
        assert!(err.source().is_none());
    }
}
