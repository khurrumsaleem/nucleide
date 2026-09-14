//! Error type for the tritium crate.

use thiserror::Error;

/// Result alias for the `tritium` crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised while validating transport data, trap specs, boundary
/// conditions, grids, or solver options — and when a requested analysis is
/// named-open in v1 (recombination steady state, G5).
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A transport datum is non-finite or out of range: `{0}`.
    #[error("tritium: invalid transport data: {0}")]
    BadData(&'static str),
    /// A trap spec entry is invalid: `{0}`.
    #[error("tritium: invalid trap spec: {0}")]
    BadTrap(&'static str),
    /// A boundary-condition parameter is invalid: `{0}`.
    #[error("tritium: invalid boundary condition: {0}")]
    BadBoundary(&'static str),
    /// A grid entry is invalid: `{0}`.
    #[error("tritium: invalid grid: {0}")]
    BadGrid(&'static str),
    /// A solver option is invalid: `{0}`.
    #[error("tritium: invalid solver option: {0}")]
    BadOption(&'static str),
    /// An initial-state entry is invalid: `{0}`.
    #[error("tritium: invalid initial state: {0}")]
    BadState(&'static str),
    /// The recombination steady state (G5) is named-open in v1: the surface
    /// law `J = K_r c_m^2` makes the steady problem nonlinear and no gate
    /// pins the nonlinear-solve tolerance yet.
    #[error("tritium: recombination steady state (G5) is named-open in v1")]
    RecombinationOpen,
    /// The per-step trap-coupling Picard iteration did not converge within
    /// its iteration cap.
    #[error("tritium: trap coupling did not converge within its iteration cap")]
    NotConverged,
    /// The implicit stepper exceeded `{0}` accepted steps.
    #[error("tritium: step budget exhausted ({0} steps)")]
    StepBudget(usize),
    /// The tridiagonal elimination failed: `{0}`.
    #[error("tritium: tridiagonal solve failed: {0}")]
    Tridiag(String),
}
