# `crates/vr-tools` AGENTS.md

## Purpose

Monte Carlo variance-reduction tools: MAGIC weight windows, mesh source
sampling (alias tables), and Gaussian KDE source resampling
(KDSource-class, clean-room). No transport, no MOAB.

## Ownership

Owns `crates/vr-tools/src/` (`magic.rs`, `sampling.rs`, `kde.rs`), the
Python surface (`nucleide.vr`, `magic`/`AliasTable`/`MeshSourceSampler`/
`KdeSampler` in `_internal`), and `tests/test_serpent_fluka_vr.py` (MAGIC
+ sampling + KDE sections).

## Local Contracts

- `magic`: flux/`(2*max)` weight-window bounds over meshtal data.
- `sampling`: Walker/Vose alias tables + `MeshSourceSampler`
  (ANALOG/UNIFORM/USER bias modes, birth weights); draws take
  caller-supplied uniforms, no RNG inside.
- `kde`: axis-aligned Gaussian KDE over caller particle vectors
  (Silverman or fixed bandwidths; zero-variance Silverman dims are loud);
  deterministic `draw(u, normals)`/`pdf`, no RNG inside, no `mcpl-io`
  dependency (MCPL projection stays caller-side).
- KDE equations are textbook statistics implemented from scratch; no
  upstream code or data read, ported, or vendored (GPL upstream).
- Synthetic fixtures only; golden values live in unit tests with
  recorded provenance (test-only LCG seed).
- Bindings stay thin; error enum stays `#[non_exhaustive]`.

## Work Guidance

- New samplers add a module plus Python/tests in the same change.
- Keep the layering: depends on `nucleide-mcnp-io` (meshtal) only.

## Verification

- `cargo test -p nucleide-vr-tools` (includes KDE recovery gates).
- `pytest tests/test_serpent_fluka_vr.py` after `maturin develop`.

## Child NAD Index

None.
