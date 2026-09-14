#![warn(missing_docs)]
//! Tokamak fusion-neutron sources: ring and point geometry over the D-D
//! (2.45 MeV) and D-T (14.1 MeV) reactions, sampled to particle vectors and
//! emitted as MCNP `SDEF` / Serpent `src` cards with a drift report.
//!
//! This is the first of two landings: ring/point sources only. The
//! parametric (Miller-geometry) tokamak plasma — density/temperature
//! profiles, pedestal modes, reactant mixtures — is the follow-up landing's
//! scope; nothing here computes profiles and everything outside v1 fails
//! loudly through [`Error::NotYetSupported`], never a panic or a guess.
//!
//! # Physics
//!
//! Thermonuclear neutron spectra are Gaussian with a width proportional to
//! `sqrt(T_i)` (Brysk, Plasma Phys. **15** (1973) 611). [`FusionReaction`]
//! implements the closed-form coefficient fits of Ballabio et al., Nucl.
//! Fusion **38** (1998) 1723, Table III (mean shift and weakly
//! temperature-dependent FWHM), which refine the Brysk scaling; at
//! `T_i = 0` the spectrum is the monoenergetic nominal line. Ring/point
//! spatial moments are closed form (radius/height/azimuth), which the
//! analytic gates in `validation/plasma_source_vs_openmc.py` check against
//! sampled particle vectors.
//!
//! # Units and dialects
//!
//! Lengths are centimetres, energies MeV, ion temperature keV — the
//! transport-code card convention. MCNP `SDEF` cards render through the
//! typed `nucleide-mcnp-io` reader and round-trip byte-identically (the
//! spectroscopy E9 precedent); Serpent `src` cards follow the public Serpent
//! input-manual spelling and are analytic-by-design in the drift report.
//!
//! # Layering
//!
//! Dependencies are `nucleide-mcnp-io` (the `SDEF` dialect) and
//! `nucleide-nuclei` (`PAR=` designators) only — no `mcpl-io`. MCPL
//! projection stays caller-side, the same rule as `vr-tools` `KdeSampler`:
//! this crate outputs particle vectors and card strings; the caller writes
//! files with `nucleide-mcpl-io` / `nucleide-mcnp-io` when it wants them.
//!
//! # Example
//!
//! ```rust
//! use nucleide_plasma_source::{
//!     emit_sdef, emit_serpent, FusionReaction, SourceSampler, PlasmaSourceConfig,
//! };
//!
//! // D-T tokamak ring, R = 300 cm at the midplane, T_i = 20 keV.
//! let config = PlasmaSourceConfig::ring(300.0, 0.0, FusionReaction::Dt, 20.0);
//!
//! // Deterministic particle vector (seeded); MCPL stays caller-side.
//! let particles = SourceSampler::new(config, 42).unwrap().sample_n(1000);
//! assert_eq!(particles.len(), 1000);
//!
//! // Cards + drift report.
//! let sdef = emit_sdef(&config, 5, 21).unwrap();
//! sdef.verify_round_trip().unwrap();
//! let serpent = emit_serpent(&config, 21).unwrap();
//! assert!(sdef.text.contains("RAD=D1"));
//! assert!(serpent.text.contains("src 1 rad d1"));
//! ```

pub mod emit_sdef;
pub mod emit_serpent;
pub mod error;
pub mod reaction;
pub mod report;
pub mod sample;
pub mod source;
pub mod spectrum;

pub use emit_sdef::emit_sdef;
pub use emit_sdef::EmittedCard;
pub use emit_serpent::emit_serpent;
pub use error::{Error, Result};
pub use reaction::FusionReaction;
pub use report::{DriftReport, DriftRow};
pub use sample::{Particle, SourceSampler};
pub use source::{PlasmaSourceConfig, PointSource, RingSource, SourceModel};
pub use spectrum::{SpectrumSpec, SpectrumTable};
