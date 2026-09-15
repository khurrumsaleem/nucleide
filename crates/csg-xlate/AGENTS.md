# `crates/csg-xlate` AGENTS.md

## Purpose

MCNP CSG translation (scoped v3) to OpenMC `geometry.xml`, Serpent input
(`surf`/`cell` cards), PHITS input (`[Surface]`/`[Cell]` sections), and GDML
(Geant4 geometry, schema version 3.1.7 pinned from the Geant4 `v11.4.2` tag;
the format is described by the CHEP 2005 paper, CERN-CDS-1023367): surfaces,
cells, and simple nested universes plus a material stub. Clean-room from
format docs (MCNP manual grammar, OpenMC XML spec, public Serpent
wiki/manual, public PHITS manual, public Geant4 GDML schema); EUPL upstream
never read for code.

## Ownership

Owns `crates/csg-xlate/src/lib.rs` (`deck_csg_to_openmc_xml`,
`deck_csg_to_serpent_input`, `deck_csg_to_phits_input`,
`deck_csg_to_gdml`, `Error`, `DriftTable`), `crates/csg-xlate/src/gdml.rs`
(GDML direction internals), the Python facades
(`parse_csg_to_openmc`/`read_csg_to_openmc`,
`parse_csg_to_serpent`/`read_csg_to_serpent`,
`parse_csg_to_phits`/`read_csg_to_phits`,
`parse_csg_to_gdml`/`read_csg_to_gdml`), `tests/test_csg_xlate.py`,
`fixtures/mcnp/inp/deck_csg_*.txt`, and the CSG section of
`validation/parsers_vs_refs.py`. Replays (never rewrites) the CSG fixtures.

## Local Contracts

- GO scope: 34 surface kinds (planes, on-axis cylinders, spheres, `SPH`,
  axis-aligned `RPP`/`RCC` expansions), cell CSG (`:`/`#`/parens) with De
  Morgan complement inlining over flat intersections only, reflective +
  periodic boundaries (OpenMC only), cell `U=k` + single-universe `FILL n`
  (cell param or data-block card) mapping to `universe=`/`fill=` (filled
  cells omit `material`), and `LAT=1` rectangular lattices with a full
  matrix `FILL` — pitch and lower-left derive only from the lattice cell's
  `RPP` or axis-plane-box bounds — in all four directions.
- Serpent direction: same v3 scope with native simplifications — `RPP` to
  `cuboid`, axis-aligned `RCC` to truncated cylinders, `#n` passes through
  as Serpent's native cell complement, empty regions synthesize an `inf`
  surface; materials render as `m<n>`/`void` names (caller supplies `mat`
  cards). A `LAT=1` matrix `FILL` becomes a cuboidal `lat` card (type 11)
  filled from the lattice cell via `fill <lattice id>`. Reflecting/periodic
  boundaries are loud `SerpentBoundaryOutOfScope` (Serpent `set bc` is
  global; per-surface mapping unverified).
- PHITS direction: same v3 scope with manual-verified identical symbols
  (`PX/Y/Z`, `SO/SX/SY/SZ/S`, `CX/CY/CZ`, `SPH`, `RPP`, `RCC` any
  orientation, axis-aligned `BOX`) passing coefficients verbatim; `#n`
  passes through, `U=`/`FILL=` render as cell params, `*` markers render
  as reflective surfaces, densities pass through verbatim (shared sign
  convention). Void union/complement cells emit as outer void `-1` with an
  `outer-void-assigned` drift note (heuristic — review it for
  union-shaped interior voids). Rectangular lattices keep `LAT=1` with a
  matrix `FILL` (ranges plus the universe list in MCNP order verbatim).
  Periodic pointers are loud `PhitsBoundaryOutOfScope`; empty regions are
  loud (no `inf` spelling).
- GDML direction: same v3 scope expressed in GDML's solid-based model —
  every cell becomes a named boolean solid plus a `<volume>`, every
  universe an `<assembly>`, the deck a `world` `<volume>` over the shared
  cutoff box, and `<setup>` points at it. Native closed solids where exact:
  spheres to `orb`, `RPP`/axis-aligned `BOX` to `box`, axis-aligned `RCC`
  to a finite tube plus two cap half-space boxes (named boolean `sl<n>`,
  `macrobody-expansion` drift). Axis planes map to half-space box pairs and
  infinite cylinder axes to bounded `tube`s (per-deck cutoff `L`, one
  `halfspace-bounded` deck note; CX/CY add a single-axis `firstrotation`).
  `#n` inlines by De Morgan exactly like OpenMC; unions mixing an exterior
  rewrite through `subtraction(bigbox, …)`, exact within the cutoff.
  Rectangular lattices expand to one `<physvol>` per element at the element
  center (`lattice-expanded` drift note). Materials emit as `mat_<n>` stubs
  (`<D>`/`<atom>` placeholders, one `material-stub` deck note; the caller
  replaces them). Reflecting and periodic boundaries are loud
  `GdmlBoundaryOutOfScope` (Geant4 expresses boundaries through wrapper
  code, not geometry markup). Emitted documents carry `version="3.1.7"`;
  the v1 element subset is pinned in the `gdml.rs` module docs.
- Loud errors, never silent mistranslation: hexagonal `LAT=2` lattices,
  lattice `0`-holes, non-`RPP`/axis-plane-bounded lattice cells,
  single-universe fills of lattice type, matrix fills without `LAT=1`,
  transformed fills, `U=-n`, `TRCL`/`TRn`, cones/quadrics/tori, tallies,
  sources, `READ` includes — in all four directions. Canted `RCC`/`BOX`
  are loud `MacrobodyOutOfScope` in the GDML direction. Every `Error`
  variant has an end-to-end Python reject test.
- Drift actions: `macrobody-expansion`, `complement-expansion`,
  `reflective-applied`, `periodic-link`, `universe-assigned`,
  `fill-applied`, `lattice-emitted`, `universe-data-card`,
  `dropped-cell-param`, `dropped-data-card` (shared);
  `halfspace-bounded`, `lattice-expanded`, `material-stub` (GDML).
- Structural assertions in tests (never byte-gold vs any code's output);
  validation cross-checks `Region.from_expression` plus full
  `Geometry.from_xml` loads (lattice decks verified geometrically against
  the source deck's FILL matrix) in the container only, and validates every
  translated GDML document against the runtime hash-pinned Geant4 XSD
  wherever `lxml` is importable (loud SKIP otherwise; a Geant4 load gate
  stays OUT as a disproportionate dependency).
- Bindings stay thin; synthetic fixtures only; no OpenMC or Geant4
  dependency in Rust (XML hand-emitted via `quick-xml`).

## Work Guidance

- New surface/universe coverage lands with fixtures + Rust unit + Python
  tests + validation probe rows in the same change.
- Keep the layering: depends on `nucleide-mcnp-io` semantic views only;
  `mcnp-io` never learns target-code concepts.
- Error enum stays `#[non_exhaustive]`; crate `Result` alias; full rustdoc.
- GDML schema details (element subset, rotation convention
  `R = Rz·Ry·Rx` per `G4GDMLReadDefine`, required `name` attributes on
  every position/rotation) live in the `gdml.rs` module docs — pin changes
  there, never vendored.

## Verification

- `cargo test -p nucleide-csg-xlate` (unit + doctest).
- `pytest tests/test_csg_xlate.py` after `maturin develop`.
- Container `parsers_vs_refs.py` CSG section (structural + OpenMC probes +
  GDML XSD validation).

## Child NAD Index

None.
