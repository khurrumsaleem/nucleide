"""ENDF/B-VIII.0 fission-yield store + the depletion-chain fallback to it.

Spot values are read off the ENDF/B-VIII.0 fission-yield tapes themselves
(England et al. ENDF-349 evaluations; Pu-239 is Chadwick-Kawano); the
depletion fallback tests use synthetic chain XML written to ``tmp_path``.
"""

import math
from pathlib import Path

import pytest

import nucleide


def test_u235_thermal_sets_and_spots() -> None:
    sets = nucleide.nuclei.fission_yields("U235")
    assert [energy for energy, _ in sets] == [0.0253, 500000.0, 14000000.0]
    energy, products = sets[0]
    assert energy == 0.0253
    by_daughter = {name: (y, dy) for name, y, dy in products}
    # Independent thermal Xe135, verbatim from the tape.
    y_xe, dy_xe = by_daughter["Xe135"]
    assert y_xe == pytest.approx(0.000785125, rel=1e-9)
    assert dy_xe == pytest.approx(4.71075e-05, rel=1e-9)
    # FPS = 1 maps to the GNDS _m1 suffix.
    y_m1, _ = by_daughter["Xe135_m1"]
    assert y_m1 == pytest.approx(0.00178122, rel=1e-9)
    # Independent yields sum to ~2.0 (two fragments per fission); the
    # A <= 116 light peak carries ~1.0 of that (asymmetric split).
    total = sum(y for _, y, _ in products)
    assert total == pytest.approx(2.0, abs=1e-6)
    light = sum(y for name, y, _ in products if nucleide.nuclei.Nuclide(name).a <= 116)
    assert light == pytest.approx(1.0, abs=1e-3)


def test_cumulative_kind_and_origins() -> None:
    # Cumulative (MT459) Xe135 is the classic ~6.6% chain value.
    (energy, products), *_ = nucleide.nuclei.fission_yields("U235", kind="cumulative")
    assert energy == 0.0253
    by_daughter = {name: (y, dy) for name, y, dy in products}
    y_cum, dy_cum = by_daughter["Xe135"]
    assert y_cum == pytest.approx(0.065385, rel=1e-9)
    assert dy_cum == pytest.approx(0.000457695, rel=1e-9)
    # Cumulative sets accumulate precursor feed: sums far above 2.0.
    assert sum(y for _, y, _ in products) > 4.0
    # Spontaneous fission: Cf-252 has a single E = 0 set.
    (sf_energy, sf_products), *_ = nucleide.nuclei.fission_yields("Cf252", origin="sf")
    assert sf_energy == 0.0
    sf_by_daughter = {name: y for name, y, _ in sf_products}
    assert sf_by_daughter["Xe135"] == pytest.approx(0.00186145, rel=1e-9)
    # U-238 lives in both sublibraries; its neutron-induced sets start at
    # 500 keV (there is no thermal nfy evaluation for it).
    n_sets = nucleide.nuclei.fission_yields("U238")
    assert n_sets[0][0] == 500000.0
    assert len(n_sets) == 2


def test_singular_lookup_and_missing() -> None:
    assert nucleide.nuclei.fission_yield("U235", "Xe135") == pytest.approx(0.000785125, rel=1e-9)
    # U-238's default set is the 500 keV neutron-induced one.
    assert nucleide.nuclei.fission_yield("U238", "Xe135") == pytest.approx(0.000111541, rel=1e-9)
    assert nucleide.nuclei.fission_yield("U235", "Fe56") is None
    assert nucleide.nuclei.fission_yields("Fe56") == []
    assert nucleide.nuclei.fission_yields("Cm247") == []
    with pytest.raises(ValueError, match="origin"):
        nucleide.nuclei.fission_yields("U235", origin="x")
    with pytest.raises(ValueError, match="kind"):
        nucleide.nuclei.fission_yields("U235", kind="x")
    with pytest.raises(ValueError):
        nucleide.nuclei.fission_yields("Nope1")


def test_pu239_chadwick_kawano_evaluation() -> None:
    # Pu-239 is the one re-evaluated parent (EVAL-NOV11): four energy sets.
    sets = nucleide.nuclei.fission_yields("Pu239")
    assert [energy for energy, _ in sets] == [0.0253, 500000.0, 2000000.0, 14000000.0]
    _, products = sets[0]
    by_daughter = {name: y for name, y, _ in products}
    assert by_daughter["Xe135"] == pytest.approx(0.00314131, rel=1e-9)
    assert sum(by_daughter.values()) == pytest.approx(2.0, abs=1e-6)


def _write_chain(tmp_path: Path, text: str) -> str:
    path = tmp_path / "chain_fy.xml"
    path.write_text(text)
    return str(path)


# Chain with a fissionable parent and NO <neutron_fission_yields>: the
# parser must fall back to the built-in library at the lowest-energy
# independent set (OpenMC get_default_fission_yields convention).
FALLBACK_CHAIN = """<depletion_chain>
  <nuclide name="U235" reactions="1">
    <reaction type="fission" Q="2.0e8"/>
  </nuclide>
  <nuclide name="I135" reactions="0"/>
  <nuclide name="Xe135" reactions="0"/>
</depletion_chain>
"""


def test_chain_falls_back_to_builtin_fission_yields(tmp_path: Path) -> None:
    chain = nucleide.depletion.read_chain(_write_chain(tmp_path, FALLBACK_CHAIN))
    n0 = {"U235": 1e15}
    rates = {"U235:fission": 1e-5}
    out = nucleide.depletion.deplete(chain, n0, 1e5, rates=rates)
    # Exact Bateman gain: N0 * y * (1 - exp(-rate * dt)) for stable products.
    produced = 1.0 - math.exp(-1.0)
    assert out["I135"] == pytest.approx(1e15 * 0.0292737 * produced, rel=1e-9)
    assert out["Xe135"] == pytest.approx(1e15 * 0.000785125 * produced, rel=1e-9)
    assert out["U235"] == pytest.approx(1e15 * math.exp(-1.0), rel=1e-9)


def test_chain_fission_without_library_parent_errors(tmp_path: Path) -> None:
    # Cm-247 has no ENDF/B-VIII.0 fission-yield evaluation: the chain is
    # structurally incomplete (the crate's BadStructure error class).
    xml = """<depletion_chain>
  <nuclide name="Cm247" reactions="1">
    <reaction type="fission" Q="2.0e8"/>
  </nuclide>
</depletion_chain>
"""
    with pytest.raises(ValueError, match="Cm247"):
        nucleide.depletion.read_chain(_write_chain(tmp_path, xml))
