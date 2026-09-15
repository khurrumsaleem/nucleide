# `crates/plasma-source` AGENTS.md

## Purpose

Tokamak fusion-neutron source creation: ring and point sources plus a
parametric Miller-geometry plasma, over the D-D (2.45 MeV) and D-T
(14.1 MeV) reactions with ion-temperature-broadened Gaussian spectra, a
seeded sampler to particle vectors, and MCNP `SDEF` + Serpent `src` card
emission with a drift report.

## Ownership

Owns `crates/plasma-source/src/` (`reaction.rs`, `reactivity.rs`,
`spectrum.rs`, `source.rs`, `sample.rs`, `miller.rs`, `profile.rs`,
`parametric.rs`, `emit_sdef.rs`, `emit_serpent.rs`, `report.rs`,
`error.rs`), the Python surface (`nucleide.plasma_source`,
`plasma_source_*` in `_internal`), `tests/test_plasma_source.py`, and the
`plasma_source_vs_openmc.py` validation oracle.

## Local Contracts

- Provenance: physics is clean-room from published closed forms — Brysk,
  Plasma Phys. 15 (1973) 611 + Ballabio et al., Nucl. Fusion 38 (1998) 1723
  Table III (spectra); Fausser et al., Fus. Eng. Des. 87 (2012) 787
  (Miller map, L/H/A profiles); Bosch & Hale, Nucl. Fusion 32 (1992) 611
  in the Atzeni–Meyer-ter-Vehn parametrization (reactivity). Coefficients
  are transcribed as plain published facts and pinned by golden tests. The
  MIT `openmc-plasma-source`/`NeSST` packages are runtime oracles only
  (never ported); GPL upstreams (KDSource) stay never-read.
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
- Parametric model: profiles are caller inputs (never computed, no
  equilibrium); emission strength is `f_fuel·n²·⟨σv⟩` with equimolar D-T
  (`f=1/4`) or pure D-D (`f=1/2`, the Fausser/openmc-plasma-source
  convention); birth positions weight `S·R·|J|` (the `R` toroidal factor
  is mandatory — without it the core is over-represented). Card emission
  for parametric sources is product-form: radial/vertical/energy
  *marginals* as discrete histograms, and the drift report must carry the
  joint-correlation distance (half the L1 distance between the true
  `(r, z)` joint and the product of marginals).
- Loud boundary (`Error::NotYetSupported` / `ValueError`, never a guess):
  fuel mixtures (Eriksson et al., CPC 199 (2016) 40-weighted reactant
  distributions), toroidal sectors (`start_angle`/`rotation_angle`), the
  D(d,p)T proton branch, and profile self-consistency (zero total
  strength).
- Units: cm, MeV, keV (card convention); profile density is m⁻³ (Fausser
  convention, converted internally); reactivity is exposed in m³/s. The
  validation oracle converts at the openmc-plasma-source boundary (m, eV).
- Sampler determinism: pinned seed reproduces the stream on a given
  platform (libm functions); golden vectors in `sample.rs` pin the RNG
  across refactors, and the parametric inverse-CDF tables are built at
  construction (no per-particle quadrature).

## Work Guidance

- Miller/Jacobian changes must keep the pinned gates green: `J == r` (torus
  limit), `J == κ·r·(1 − 2·esh·r·cosθ/a²)` at δ=0, area `π·κ·a²`, volume
  `2π²·κ·a²·R₀` (any esh), finite-difference agreement at general δ, and
  the strength ratio gates (`<σv>` scaling).
- New emission dialects follow `emit_serpent.rs`: golden card tests plus
  drift rows (`reparsed` false when no reader exists).
- Adding mixtures later requires a documented normalization (Eriksson
  weighting) before the `NotYetSupported` boundary moves.

## Verification

- `cargo test -p nucleide-plasma-source` (map/Jacobian closed forms,
  profile/reactivity goldens, sampler moment gates, golden cards, reader
  round trips).
- `pytest tests/test_plasma_source.py` after `maturin develop`.
- `validation/plasma_source_vs_openmc.py` runs inside `run_all.sh`
  (two-part: P1–P7 always, O1–O7 vs openmc-plasma-source/NeSST
  container-only with loud SKIP outside; the Containerfile layer installs
  `openmc-plasma-source` + `NeSST` and shims `scipy.integrate.cumtrapz`
  in the oracle process for scipy ≥ 1.14).

## Child NAD Index

None.
