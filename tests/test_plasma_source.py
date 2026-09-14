"""Python-side tests for the tokamak fusion-source core (analytic gates + errors)."""

import math

import pytest

import nucleide.mcnp as mcnp
import nucleide.plasma_source as ps

RADIUS_CM = 300.0
HEIGHT_CM = 25.0
N = 200_000

RING_SPEC = {
    "kind": "ring",
    "radius": RADIUS_CM,
    "height": HEIGHT_CM,
    "reaction": "dt",
    "ion_temperature_kev": 20.0,
}


def _ballabio_moments(reaction: str, ti_kev: float) -> tuple[float, float]:
    """Independent transcription of the Ballabio et al. 1998 Table III fits."""
    if reaction == "dd":
        a1, a2, a3, a4 = 4.69515, -0.040729, 0.47, 0.81844
        m0 = 2.4495e6
        w0, b1, b2, b3, b4 = 82.542, 1.7013e-3, 0.16888, 0.49, 7.9460e-4
    else:
        a1, a2, a3, a4 = 5.30509, 2.4736e-3, 1.84, 1.3818
        m0 = 14.021e6
        w0, b1, b2, b3, b4 = 177.259, 5.1068e-4, 7.6223e-3, 1.78, 8.7691e-5
    fwhm_over_sigma = 2 * math.sqrt(2 * math.log(2))
    mean = m0 + 1e3 * (a1 * ti_kev ** (2 / 3) / (1 + a2 * ti_kev**a3) + a4 * ti_kev)
    delta = b1 * ti_kev ** (2 / 3) / (1 + b2 * ti_kev**b3) + b4 * ti_kev
    sigma = w0 * (1 + delta) * math.sqrt(ti_kev) / fwhm_over_sigma
    return mean / 1e6, sigma / 1e3


def test_ring_spatial_moments_match_closed_form() -> None:
    out = ps.particles(RING_SPEC, N, seed=42)
    radius = (out["x"] ** 2 + out["y"] ** 2) ** 0.5
    assert bool((out["z"] == HEIGHT_CM).all())
    assert radius == pytest.approx(RADIUS_CM, rel=1e-12)
    se = RADIUS_CM / math.sqrt(2 * N)
    assert abs(float(out["x"].mean())) < 8 * se
    assert abs(float(out["y"].mean())) < 8 * se
    direction_norm = (out["u"] ** 2 + out["v"] ** 2 + out["w"] ** 2) ** 0.5
    assert direction_norm == pytest.approx(1.0, rel=1e-12)
    assert abs(float(out["w"].mean())) < 0.01


def test_spectrum_moments_match_closed_form() -> None:
    out = ps.particles(RING_SPEC, N, seed=7)
    mean, sigma = _ballabio_moments("dt", 20.0)
    assert float(out["energy"].mean()) == pytest.approx(mean, abs=5e-3)
    assert float(out["energy"].std()) == pytest.approx(sigma, abs=5e-3)


def test_spectrum_moments_helper_matches_independent_fit() -> None:
    for reaction in ("dd", "dt"):
        for ti in (1.0, 10.0, 20.0):
            got = ps.spectrum_moments(reaction, ti)
            mean, sigma = _ballabio_moments(reaction, ti)
            assert got["mean_mev"] == pytest.approx(mean, abs=1e-12)
            assert got["sigma_mev"] == pytest.approx(sigma, abs=1e-12)


def test_zero_temperature_is_monoenergetic() -> None:
    spec = {"kind": "point", "position": [0, 0, 0], "reaction": "dd", "ion_temperature_kev": 0.0}
    out = ps.particles(spec, 100, seed=1)
    assert out["energy"] == pytest.approx(2.4495, rel=1e-12)
    moments = ps.spectrum_moments("dd", 0.0)
    assert moments["sigma_mev"] == 0.0
    assert moments["nominal_mev"] == pytest.approx(2.4495)


def test_pinned_seed_reproduces_the_stream() -> None:
    a = ps.particles(RING_SPEC, 512, seed=1234)
    b = ps.particles(RING_SPEC, 512, seed=1234)
    for key in ("x", "y", "z", "u", "v", "w", "energy", "weight"):
        assert bool((a[key] == b[key]).all())
    c = ps.particles(RING_SPEC, 512, seed=1235)
    assert not bool((a["energy"] == c["energy"]).all())


def test_sdef_card_round_trips_through_typed_reader() -> None:
    out = ps.emit_source_cards(RING_SPEC)
    card = out["sdef"]["card"]
    parsed = mcnp.parse_sdef(card)
    assert parsed["card"] == card
    assert parsed["rad"] == "D1"
    assert parsed["axs"] == "0 0 1"
    assert parsed["pos"] == "0 0 25"
    assert parsed["erg"] == "D2"
    assert len(parsed["distributions"]) == 2
    assert parsed["distributions"][0]["si"] == "300 300"
    assert out["sdef"]["drift"][0]["reparsed"] is True


def test_sdef_point_monoenergetic_card() -> None:
    spec = {"kind": "point", "position": [1, 2, 3], "reaction": "dd", "ion_temperature_kev": 0.0}
    out = ps.emit_source_cards(spec)
    assert out["sdef"]["card"] == "SDEF POS=1 2 3\n     ERG=2.4495\n     WGT=1\n     PAR=n"
    assert out["spectrum"]["mono"] is True
    assert out["sdef"]["drift"][0]["rel_drift"] == 0.0


def test_serpent_card_structure() -> None:
    out = ps.emit_source_cards(RING_SPEC)
    card = out["serpent"]["card"]
    lines = card.splitlines()
    assert lines[0] == "src 1 pos 0 0 25"
    assert lines[1] == "src 1 rad d1"
    assert lines[2] == "src 1 erg d2"
    assert lines[3] == "src 1 wgt 1"
    assert lines[4] == "SI1 300 300"
    assert lines[5] == "SP1 1"
    assert lines[6].startswith("SI2 ")
    assert lines[7].startswith("SP2 ")
    assert out["serpent"]["drift"][0]["reparsed"] is False


def test_gaussian_tabulation_drift_is_reported() -> None:
    out = ps.emit_source_cards(RING_SPEC)
    drift = out["sdef"]["drift"][0]
    assert drift["quantity"] == "emission probability"
    assert drift["rel_drift"] > 0.0
    assert drift["rel_drift"] == pytest.approx(1.0 - drift["accounted"], rel=1e-12)
    # +/-4 sigma tail mass.
    assert drift["rel_drift"] == pytest.approx(1.0 - (math.erf(4 / math.sqrt(2))), rel=1e-3)


def test_mcnp_version_six_and_bad_version() -> None:
    # Neutrons carry designator "n" in both dialects; version 6 must emit an
    # identical card and an unknown version must raise loudly.
    mono = {
        "kind": "point",
        "position": [0, 0, 0],
        "reaction": "dt",
        "ion_temperature_kev": 0.0,
        "mcnp_version": 6,
    }
    out = ps.emit_source_cards(mono)
    assert out["sdef"]["card"] == "SDEF POS=0 0 0\n     ERG=14.021\n     WGT=1\n     PAR=n"
    bad = dict(RING_SPEC, mcnp_version=4)
    with pytest.raises(ValueError, match="version"):
        ps.emit_source_cards(bad)


def test_validation_errors_are_loud() -> None:
    with pytest.raises(ValueError, match="radius"):
        ps.particles(dict(RING_SPEC, radius=0.0), 4, seed=0)
    with pytest.raises(ValueError, match="ion temperature"):
        ps.particles(dict(RING_SPEC, ion_temperature_kev=-1.0), 4, seed=0)
    with pytest.raises(ValueError, match="reaction"):
        ps.particles(dict(RING_SPEC, reaction="tt"), 4, seed=0)
    with pytest.raises(ValueError, match="kind"):
        ps.particles(dict(RING_SPEC, kind="torus"), 4, seed=0)
    with pytest.raises(ValueError, match="not yet supported"):
        ps.particles(dict(RING_SPEC, fuel={"D": 0.5, "T": 0.5}), 4, seed=0)
    with pytest.raises(ValueError, match="not yet supported"):
        ps.particles(dict(RING_SPEC, rotation_angle=1.57), 4, seed=0)
    with pytest.raises(ValueError, match="position"):
        ps.particles(
            {"kind": "point", "position": [0, 0], "reaction": "dt", "ion_temperature_kev": 0.0},
            4,
            seed=0,
        )
