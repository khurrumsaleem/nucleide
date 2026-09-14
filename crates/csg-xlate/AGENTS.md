# `crates/csg-xlate` AGENTS.md

## Purpose

MCNP CSG translation (scoped v2) to OpenMC `geometry.xml`, Serpent input
(`surf`/`cell` cards), and PHITS input (`[Surface]`/`[Cell]` sections):
surfaces, cells, and simple nested universes plus a material stub.
Clean-room from format docs (MCNP manual grammar, OpenMC XML spec, public
Serpent wiki/manual, public PHITS manual); EUPL upstream never read
for code.

## Ownership

Owns `crates/csg-xlate/src/lib.rs` (`deck_csg_to_openmc_xml`,
`deck_csg_to_serpent_input`, `deck_csg_to_phits_input`, `Error`,
`DriftTable`), the Python facades (`parse_csg_to_openmc`/`read_csg_to_openmc`,
`parse_csg_to_serpent`/`read_csg_to_serpent`,
`parse_csg_to_phits`/`read_csg_to_phits`), `tests/test_csg_xlate.py`,
`fixtures/mcnp/inp/deck_csg_*.txt`, and the CSG section of
`validation/parsers_vs_refs.py`. Replays (never rewrites) the CSG fixtures.

## Local Contracts

- GO scope: 34 surface kinds (planes, on-axis cylinders, spheres, `SPH`,
  axis-aligned `RPP`/`RCC` expansions), cell CSG (`:`/`#`/parens) with De
  Morgan complement inlining over flat intersections only, reflective +
  periodic boundaries, cell `U=k` + single-universe `FILL n` (cell param or
  data-block card) mapping to `universe=`/`fill=` (filled cells omit
  `material`).
- Serpent direction: same v2 scope with native simplifications — `RPP` to
  `cuboid`, axis-aligned `RCC` to truncated cylinders, `#n` passes through
  as Serpent's native cell complement, empty regions synthesize an `inf`
  surface; materials render as `m<n>`/`void` names (caller supplies `mat`
  cards). Reflecting/periodic boundaries are loud
  `SerpentBoundaryOutOfScope` (Serpent `set bc` is global; per-surface
  mapping unverified).
- PHITS direction: same v2 scope with manual-verified identical symbols
  (`PX/Y/Z`, `SO/SX/SY/SZ/S`, `CX/CY/CZ`, `SPH`, `RPP`, `RCC` any
  orientation, axis-aligned `BOX`) passing coefficients verbatim; `#n`
  passes through, `U=`/`FILL=` render as cell params, `*` markers render
  as reflective surfaces, densities pass through verbatim (shared sign
  convention). Void union/complement cells emit as outer void `-1` with an
  `outer-void-assigned` drift note (heuristic — review it for
  union-shaped interior voids). Periodic pointers are loud
  `PhitsBoundaryOutOfScope`; empty regions are loud (no `inf` spelling).
- Loud errors, never silent mistranslation: `LAT` lattices, matrix or
  transformed fills, `U=-n`, `TRCL`/`TRn`, cones/quadrics/tori, tallies,
  sources, `READ` includes. Every `Error` variant has an end-to-end Python
  reject test.
- Drift actions: `macrobody-expansion`, `complement-expansion`,
  `reflective-applied`, `periodic-link`, `universe-assigned`,
  `fill-applied`, `universe-data-card`, `dropped-cell-param`,
  `dropped-data-card`.
- Structural assertions in tests (never byte-gold vs OpenMC output);
  validation cross-checks `Region.from_expression` in the container only.
- Bindings stay thin; synthetic fixtures only; no OpenMC dependency in
  Rust (XML hand-emitted via `quick-xml`).

## Work Guidance

- New surface/universe coverage lands with fixtures + Rust unit + Python
  tests + validation probe rows in the same change.
- Keep the layering: depends on `nucleide-mcnp-io` semantic views only;
  `mcnp-io` never learns OpenMC concepts.
- Error enum stays `#[non_exhaustive]`; crate `Result` alias; full rustdoc.

## Verification

- `cargo test -p nucleide-csg-xlate` (unit + doctest).
- `pytest tests/test_csg_xlate.py` after `maturin develop`.
- Container `parsers_vs_refs.py` CSG section (structural + OpenMC probes).

## Child NAD Index

None.
