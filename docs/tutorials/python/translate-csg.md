---
title: Translate CSG
sidebar:
  order: 16
---

Nucleide translates a scoped subset of MCNP constructive-solid-geometry decks
to OpenMC `geometry.xml`, Serpent `surf`/`cell` cards, PHITS
`[Surface]`/`[Cell]` sections, or a GDML (Geant4) document with the
`nucleide-csg-xlate` crate. The scope
is deliberately narrow — everything inside it translates; everything outside
it raises a `ValueError` instead of guessing. Each call returns the emitted
text plus a drift report: a list of every approximation the translator made
(expanded macrobodies, dropped parameters, …).

## Translate a deck

`parse_csg_to_*` takes deck text; `read_csg_to_*` takes a path and returns
the identical result for the same deck:

```python
import xml.etree.ElementTree as ET

from nucleide import mcnp

deck = """lattice demo
1x2x1 LAT=1 lattice in RPP bounds, every element filled
1 1 -1.0 -1 u=1
2 1 -1.0 -2 u=2
10 0 -100 lat=1 fill=0:0 0:1 0:0 1 2
20 0 #10

1 sph 1 1 1 0.5
2 sph 3 1 1 0.5
100 rpp 0 4 0 4 0 2

m1 92235 1.0
"""

xml, drift = mcnp.parse_csg_to_openmc(deck)
root = ET.fromstring(xml)
lat = root.find("lattice")
assert lat is not None and lat.get("type") == "rectangular"
assert lat.findtext("dimension") == "1 2 1"
by_id = {c.get("id"): c for c in root.findall("cell")}
assert by_id["10"].get("fill") == lat.get("id")
assert by_id["10"].get("material") is None  # filled cells carry no material
assert any(d["action"] == "lattice-emitted" and d["target"] == "10" for d in drift)
```

Assert on structure, never on emitted bytes — the text is a rendering detail;
the drift report is the authoritative record (see below). The same deck renders to the
other codes with the scoped per-code simplifications:

```python
text, _ = mcnp.parse_csg_to_serpent(deck)
assert "cell 10 0 fill 21 -100" in text.splitlines()  # cuboidal lat 21 behind the fill

text, _ = mcnp.parse_csg_to_phits(deck)
assert "10  0  -100  LAT=1  FILL=0:0 0:1 0:0 1 2" in text.splitlines()  # matrix FILL verbatim

xml, _ = mcnp.parse_csg_to_gdml(deck)
gdml = ET.fromstring(xml)
vol10 = next(v for v in gdml.find("structure") if v.get("name") == "vol10")
assert len(vol10.findall("physvol")) == 2  # one placement per lattice element
```

## What translates

The scoped v3 subset covers 34 surface kinds (planes, on-axis cylinders,
spheres, `SPH`, and axis-aligned `RPP`/`RCC` macrobodies), cell regions with
`:` unions, `#` complements, and parentheses, plus nested universes — cell
`U=k` assignments and single-universe `FILL n` — and rectangular `LAT=1`
lattices whose `FILL` gives a full matrix (no holes). A lattice's pitch and
lower-left come only from its cell's `RPP` or axis-plane box bounds.
Macrobodies expand (`RPP` to six planes, `RCC` to a cylinder plus caps), `#n`
complements inline as the negated region, and filled cells emit without a
material. Materials stay a stub: cell cards carry `material="1"` /
`m<n>` / `1` / `mat_<n>` names and you supply the actual definitions
yourself. The GDML direction renders cells as named boolean solids inside a
`world` volume, universes as assemblies, and lattices as one placement per
element; because Geant4 has no infinite solids, infinite half-spaces are
bounded by a per-deck cutoff recorded in the drift report.

## The drift report

Every accepted-but-not-lossless step lands in the drift report as a dict with
string keys `scope` (`"cell"`, `"surface"`, or `"deck"`), `target` (the
cell/surface number, `"0"` for deck scope), `action`, and a human-readable
`reason`:

```python
for entry in drift:
    print(entry["scope"], entry["target"], entry["action"], entry["reason"])
```

Actions you will meet: `macrobody-expansion` (RPP/RCC broken into primitive
surfaces), `complement-expansion` (`#n` inlined), `universe-assigned` and
`fill-applied` (universe plumbing), `lattice-emitted` (the matrix `FILL`
became a lattice element), `lattice-expanded` (GDML per-element placements),
`halfspace-bounded` and `material-stub` (GDML cutoff and placeholder
materials), `reflective-applied` / `periodic-link` (boundary mapping), and
`dropped-cell-param` / `dropped-data-card` (tokens with no target-code
spelling — listed in the report instead of vanishing silently).

## What is not translated

Out-of-scope geometry fails with a `ValueError` that names the offending
cell, in all four directions. Hexagonal `LAT=2` lattices, lattice `FILL`
matrices with `0` holes, and lattice cells whose bounds are not an `RPP`
interior or an axis-plane box are all rejected:

```python
try:
    mcnp.parse_csg_to_openmc(deck.replace("lat=1", "lat=2"))
except ValueError as e:
    print(e)
# universes/lattices/fills are out of v1 scope: cell 10 parameter `lat=2`
# needs lattice translation (rectangular LAT=1 only)
```

Transforms are rejected the same way on any card that carries them:

```python
try:
    mcnp.parse_csg_to_openmc("msg\ntitle\n1 0 -1 trcl=1\n\n1 so 10\n\ntr1 0 0 0\n")
except ValueError as e:
    print(e)
# transforms are out of v1 scope: cell 1 parameter `trcl=1` needs transform translation
```

The same funnel rejects cones, quadrics, and tori, `U=-n`, transformed fills,
matrix fills without `LAT=1`, tallies, source cards, and `READ` includes.
Serpent and PHITS add their own direction-specific unsupported cases —
reflecting and periodic boundaries have no verified Serpent mapping, and
periodic pointers have no PHITS spelling — the GDML direction rejects both
boundary kinds (Geant4 expresses boundaries through wrapper code, not
geometry markup) — and each raises a clear error.

## See also

- `tests/test_csg_xlate.py` for the full structural-assertion and reject
  suites, and `fixtures/mcnp/inp/deck_csg_*.txt` for the sample decks they
  read.
- [Parse MCNP output](parse-mcnp-output.md) for the deck reader underneath
  (`read_deck` / `parse_deck`).
