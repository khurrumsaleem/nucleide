//! Error type for the plasma-source crate.

use thiserror::Error;

/// Result alias for the `plasma-source` crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors raised while validating source configurations, tabulating spectra,
/// or rendering source cards.
///
/// Everything outside the v1 scope (parametric Miller-geometry plasma
/// profiles, pedestal modes, mixed-fuel spectra, toroidal sectors, the
/// D(d,p)T proton branch) is reported through [`Error::NotYetSupported`] —
/// a loud named error, never a panic or a silent fallback.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum Error {
    /// A source-model field is `NaN` or infinite: `{0}`.
    #[error("plasma-source: non-finite value in {0}")]
    NonFinite(&'static str),
    /// The ring radius must be finite and strictly positive; got `{0}` cm.
    #[error("plasma-source: ring radius must be > 0 cm, got {0}")]
    NonPositiveRadius(f64),
    /// The ion temperature must be finite and non-negative; got `{0}` keV.
    #[error("plasma-source: ion temperature must be >= 0 keV, got {0}")]
    NegativeIonTemperature(f64),
    /// The particle weight must be finite and strictly positive; got `{0}`.
    #[error("plasma-source: particle weight must be > 0, got {0}")]
    NonPositiveWeight(f64),
    /// A spectrum-tabulation request is invalid: `{0}`.
    #[error("plasma-source: invalid spectrum tabulation: {0}")]
    BadTabulation(&'static str),
    /// The MCNP designator version must be 5 or 6; got `{0}`.
    #[error("plasma-source: unsupported MCNP version {0} (supported: 5, 6)")]
    UnsupportedMcnpVersion(u32),
    /// A Miller-geometry field fails its documented range check: `{0}`.
    #[error("plasma-source: invalid Miller geometry: {0}")]
    InvalidGeometry(&'static str),
    /// A profile parameter fails its documented range check: `{0}`.
    #[error("plasma-source: invalid plasma profile: {0}")]
    InvalidProfile(&'static str),
    /// The emitted card failed to re-parse through the typed `SDEF` reader
    /// (an internal emission invariant; surfaced loudly rather than
    /// delivered unverified).
    #[error("plasma-source: emitted card failed the SDEF reader round trip: {0}")]
    CardRoundTrip(String),
    /// The requested capability is outside the v1 scope: `{0}`.
    #[error("plasma-source: not yet supported: {0}")]
    NotYetSupported(&'static str),
}
