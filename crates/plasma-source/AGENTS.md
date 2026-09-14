# `crates/plasma-source` AGENTS.md

## Purpose

Tokamak fusion-neutron source creation: ring and point sources over the D-D
(2.45 MeV) and D-T (14.1 MeV) reactions with ion-temperature-broadened
Gaussian spectra, a seeded sampler to particle vectors, and MCNP `SDEF` +
Serpent `src` card emission with a drift report.

## Ownership

Owns `crates/plasma-source/src/` (`reaction.rs`, `spectrum.rs`, `source.rs`,
`sample.rs`, `emit_sdef.rs`, `emit_serpent.rs`, `report.rs`, `error.rs`), the
Python surface (`nucleide.plasma_source`, `plasma_source_*` in `_internal`),
`tests/test_plasma_source.py`, and the `plasma_source_vs_openmc.py`
validation oracle.

## Local Contracts

- Provenance: physics is clean-room from published closed forms — Brysk,
  Plasma Phys. 15 (1973) 611 (Gaussian spectrum, σ ∝ √Tᵢ) with the
  coefficient fits of Ballabio et al., Nucl. Fusion 38 (1998) 1723, Table
  III, transcribed as plain published facts and pinned by golden tests.
  The MIT `openmc-plasma-source` package is a runtime oracle only (never
  ported); GPL upstreams (KDSource) stay never-read.
- Layering: depends on `nucleide-mcnp-io` (the `SDEF` dialect) and
  `nucleide-nuclei` (`PAR=` designators) only — never on `mcpl-io` or
  bindings. MCPL projection stays caller-side (the `vr-tools` KDE rule):
  the crate outputs particle vectors and card strings; the caller writes
  files with `nucleide-mcpl-io` / `nucleide-mcnp-io`.
- The MCNP `SDEF` accepted subset carries the ring keywords `AXS`/`RAD`/
  `EXT` (owned by the `mcnp-io` reader); emitted cards must round-trip
  byte-identically through `parse_sdef_text` (the E9 precedent).
  Serpent `src` rows are analytic by design (no Serpent source reader in
  the workspace).
- Units: cm, MeV, keV ion temperature (card convention); the validation
  oracle converts at the openmc-plasma-source boundary (m, eV).
- Loud boundary: parametric (Miller-geometry) plasma profiles, pedestal
  modes, mixed-fuel spectra, toroidal sectors, and the D(d,p)T proton
  branch are the second landing's scope — callers hitting them get
  `Error::NotYetSupported` / a `ValueError`, never a panic or a guess.
- Sampler determinism: pinned seed reproduces the stream on a given
  platform (libm functions); golden vectors in `sample.rs` pin the RNG
  across refactors.

## Work Guidance

- The parametric-plasma landing adds profiles/pedestals as caller inputs,
  reusing the sampler/report modules; reactivity-weighted reactant
  mixtures need a documented normalization before they leave the
  `NotYetSupported` boundary.
- New emission dialects follow `emit_serpent.rs`: golden card tests plus
  drift rows (`reparsed` false when no reader exists).

## Verification

- `cargo test -p nucleide-plasma-source` (analytic moment gates, golden
  cards, reader round trips).
- `pytest tests/test_plasma_source.py` after `maturin develop`.
- `validation/plasma_source_vs_openmc.py` runs inside `run_all.sh`
  (two-part: P1–P4 always, O1–O3 vs openmc-plasma-source container-only
  with loud SKIP outside; the Containerfile layer installs
  `openmc-plasma-source` + `NeSST` and shims `scipy.integrate.cumtrapz`
  in the oracle process for scipy ≥ 1.14).

## Child NAD Index

None.
