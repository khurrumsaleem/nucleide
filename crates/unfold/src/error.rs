//! Error type for the `unfold` crate.

use thiserror::Error;

/// Result alias for the `unfold` crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised while validating response matrices, measured rates, guess
/// spectra, or solver options — and when the SAND-II adjustment fails to
/// converge within its iteration cap.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A response-matrix entry is invalid: `{0}`.
    #[error("unfold: invalid response matrix: {0}")]
    BadResponse(&'static str),
    /// A measured-rate entry is invalid: `{0}`.
    #[error("unfold: invalid measured rates: {0}")]
    BadRates(&'static str),
    /// A guess-spectrum entry is invalid: `{0}`.
    #[error("unfold: invalid guess spectrum: {0}")]
    BadGuess(&'static str),
    /// An input length is inconsistent: `{what}` holds `{got}` entries, expected `{expected}`.
    #[error("unfold: shape mismatch: {what} holds {got} entries, expected {expected}")]
    BadShape {
        /// Name of the offending input.
        what: &'static str,
        /// Length the input must carry.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// A solver option is invalid: `{0}`.
    #[error("unfold: invalid solver option: {0}")]
    BadOption(&'static str),
    /// No strictly positive spectrum can fold to the measured rates: detector
    /// `{detector}` either carries a zero response row against a nonzero
    /// measurement (rejected up front) or was left with a zero fold because
    /// every group it responds to was pinned to zero by other measurements
    /// (detected mid-iteration).
    #[error("unfold: measured rates are unreachable: detector {detector} cannot be satisfied by any positive spectrum")]
    RatesUnreachable {
        /// Index of the offending detector (response row).
        detector: usize,
    },
    /// The SAND-II adjustment exhausted its iteration cap without meeting the
    /// per-group relative-change tolerance. Hard fail — never a silent partial
    /// spectrum (the `tritium` face-Newton precedent).
    #[error("unfold: SAND-II adjustment did not converge within its iteration cap")]
    NotConverged,
}
