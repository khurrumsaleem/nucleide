"""Python-side tests for MCPL interchange + MCTAL tally bodies (synthetic only).

All particle records are hand-built axis vectors with closed-form packing
math; no upstream files are read. Gzip transport is covered semantically
(compressed bytes are encoder-dependent and never asserted).
"""

import gzip
import os
from pathlib import Path

import pytest

import nucleide.mcpl as mcpl
from nucleide.mcnp import read_mctal, read_ssw

FIXTURE = os.path.join(
    os.path.dirname(__file__),
    "..",
    "fixtures",
    "mcnp",
    "mctal",
    "synthetic_tally_bodies.mctal",
)

SSW_REF = os.path.join(
    os.path.dirname(__file__),
    "..",
    "fixtures",
    "mcpl",
    "ssw_conversion",
    "reference.w",
)

SSW_SURFS = [100, 200]
SSW_KINDS = ["neutron", "gamma"]

HEADER = {
    "srcname": "nucleide-test",
    "comments": ["synthetic probe"],
    "has_userflags": True,
    "has_polarisation": False,
    "double_prec": False,
    "universal_pdgcode": None,
    "universal_weight": None,
    "blobs": [],
}

PARTICLES = [
    {
        "ekin": 2.5,
        "polarisation": [0.0, 0.0, 0.0],
        "position": [1.0, -2.0, 0.5],
        "direction": [0.0, 0.0, 1.0],
        "time": 3.0,
        "weight": 1.0,
        "pdgcode": 2112,
        "userflags": 0,
    },
    {
        "ekin": 0.662,
        "polarisation": [0.0, 0.0, 0.0],
        "position": [0.0, 0.0, 0.0],
        "direction": [1.0, 0.0, 0.0],
        "time": 0.0,
        "weight": 0.5,
        "pdgcode": 22,
        "userflags": 7,
    },
]


def test_mcpl_round_trip_single(tmp_path: Path) -> None:
    path = str(tmp_path / "probe.mcpl")
    mcpl.write_mcpl(path, HEADER, PARTICLES)
    back = mcpl.read_mcpl(path)
    assert back.version == 3
    assert back.nparticles == 2
    assert back.srcname == "nucleide-test"
    assert back.comments == ["synthetic probe"]
    assert back.has_userflags is True
    got = back.particles()
    assert len(got) == 2
    assert got[0]["ekin"] == pytest.approx(2.5, abs=1e-6)
    assert got[0]["direction"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    assert got[0]["pdgcode"] == 2112
    assert got[1]["pdgcode"] == 22
    assert got[1]["userflags"] == 7
    assert got[1]["weight"] == pytest.approx(0.5, abs=1e-6)


def test_mcpl_round_trip_double_universal(tmp_path: Path) -> None:
    header = {
        **HEADER,
        "double_prec": True,
        "has_polarisation": True,
        "universal_pdgcode": 2112,
        "universal_weight": 1.5,
    }
    path = str(tmp_path / "probe.mcpl")
    mcpl.write_mcpl(path, header, PARTICLES)
    back = mcpl.read_mcpl(path)
    assert back.double_prec is True
    assert back.universal_pdgcode == 2112
    assert back.universal_weight == pytest.approx(1.5)
    for p in back.particles():
        assert p["pdgcode"] == 2112
        assert p["weight"] == pytest.approx(1.5)


def test_mcpl_gzip_transparent(tmp_path: Path) -> None:
    path = str(tmp_path / "probe.mcpl.gz")
    mcpl.write_mcpl(path, HEADER, PARTICLES)
    # Raw bytes carry the gzip magic (encoder settings differ across
    # writers, so only the magic is asserted, never the full byte stream).
    with open(path, "rb") as fh:
        assert fh.read(2) == b"\x1f\x8b"
    # gzip.open reads through the layer transparently (decompressed magic).
    with gzip.open(path, "rb") as fh:
        assert fh.read(4) == b"MCPL"
    back = mcpl.read_mcpl(path)
    assert back.nparticles == 2
    assert back.particles()[0]["ekin"] == pytest.approx(2.5, abs=1e-6)


def test_mcpl_rejects_bad_inputs(tmp_path: Path) -> None:
    bad = [dict(PARTICLES[0], direction=[1.0, 1.0, 1.0])]
    with pytest.raises(ValueError):
        mcpl.write_mcpl(str(tmp_path / "x.mcpl"), HEADER, bad)
    bad_e = [dict(PARTICLES[0], ekin=-1.0)]
    with pytest.raises(ValueError):
        mcpl.write_mcpl(str(tmp_path / "y.mcpl"), HEADER, bad_e)


def test_mctal_bodies_fixture_closed_form() -> None:
    m = read_mctal(FIXTURE)
    assert m.tally_nums == [4, 14]
    bodies = {t["number"]: t for t in m.tallies}
    assert set(bodies) == {4, 14}
    t4 = bodies[4]
    assert t4["f"] == {
        "count": 2,
        "values": pytest.approx([1.0, 2.0]),
        "variant": None,
        "flag": None,
    }
    assert t4["e"] == {
        "count": 2,
        "values": pytest.approx([0.5, 2.0]),
        "variant": None,
        "flag": None,
    }
    assert t4["tfc"] is None
    assert [v for v, _ in t4["vals"]] == pytest.approx([11.0, 12.0, 21.0, 22.0])
    assert t4["total"] == pytest.approx(66.0)
    assert m.tally_vals_array(4).shape == (4, 2)
    t14 = bodies[14]
    assert [v for v, _ in t14["vals"]] == pytest.approx([7.0])
    with pytest.raises(ValueError):
        m.tally_vals_array(999)


def test_ssw2mcpl_fixture_closed_form(tmp_path: Path) -> None:
    out = str(tmp_path / "conv.mcpl")
    assert mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS) == 2
    back = mcpl.read_mcpl(out)
    assert back.nparticles == 2
    assert back.srcname == "ssw2mcpl"
    assert back.has_userflags is True
    got = back.particles()
    # Track 1 (neutron): energy verbatim, 3.0e5 shakes -> 3.0 ms, surf 100.
    assert got[0]["ekin"] == pytest.approx(2.5)
    assert got[0]["time"] == pytest.approx(3.0)
    assert got[0]["pdgcode"] == 2112
    assert got[0]["userflags"] == 100
    assert got[0]["position"] == pytest.approx([1.0, -2.0, 0.5])
    assert got[0]["direction"] == pytest.approx([0.0, 0.0, 1.0])
    assert got[0]["weight"] == pytest.approx(1.0)
    # Track 2 (gamma): 0.662 MeV through single precision, surf 200.
    assert got[1]["ekin"] == pytest.approx(0.662, abs=1e-6)
    assert got[1]["time"] == pytest.approx(0.0)
    assert got[1]["pdgcode"] == 22
    assert got[1]["userflags"] == 200
    assert got[1]["direction"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_ssw2mcpl_options(tmp_path: Path) -> None:
    out = str(tmp_path / "conv.mcpl")
    mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS, {"surf_to_userflags": False})
    back = mcpl.read_mcpl(out)
    assert back.has_userflags is False
    assert [p["userflags"] for p in back.particles()] == [0, 0]

    mcpl.ssw2mcpl(
        SSW_REF,
        out,
        SSW_SURFS,
        SSW_KINDS,
        {
            "double_prec": True,
            "srcname": "probe",
            "comments": ["synthetic"],
            "deck_blob": ("ssw_deck", b"c synthetic deck"),
        },
    )
    back = mcpl.read_mcpl(out)
    assert back.double_prec is True
    assert back.srcname == "probe"
    assert back.comments == ["synthetic"]
    assert back.blobs == [("ssw_deck", b"c synthetic deck")]

    gz = str(tmp_path / "conv.mcpl.gz")
    mcpl.ssw2mcpl(SSW_REF, gz, SSW_SURFS, SSW_KINDS, {"gzip": True})
    with open(gz, "rb") as fh:
        assert fh.read(2) == b"\x1f\x8b"
    assert mcpl.read_mcpl(gz).nparticles == 2


def test_ssw2mcpl_rejects_bad_inputs(tmp_path: Path) -> None:
    out = str(tmp_path / "conv.mcpl")
    with pytest.raises(ValueError):
        mcpl.ssw2mcpl(SSW_REF, out, [100], SSW_KINDS)
    with pytest.raises(ValueError):
        mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, ["neutron", "pion"])
    with pytest.raises(ValueError):
        mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS, {"double_prec": "yes"})
    with pytest.raises(ValueError):
        mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS, ["not-a-dict"])  # type: ignore[arg-type]


def test_ssw2mcpl_extended_kinds_fixture_closed_form(tmp_path: Path) -> None:
    out = str(tmp_path / "ext.mcpl")
    assert mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, ["electron", "positron"]) == 2
    got = mcpl.read_mcpl(out).particles()
    assert [p["pdgcode"] for p in got] == [11, -11]
    assert got[0]["ekin"] == pytest.approx(2.5)
    assert got[0]["direction"] == pytest.approx([0.0, 0.0, 1.0])
    assert got[1]["direction"] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)
    # Proton over the same reference geometry.
    assert mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, ["proton", "neutron"]) == 2
    assert [p["pdgcode"] for p in mcpl.read_mcpl(out).particles()] == [2212, 2112]
    # Committed golden for the electron/positron pairing.
    golden = os.path.join(
        os.path.dirname(__file__),
        "..",
        "fixtures",
        "mcpl",
        "ssw_conversion",
        "ssw2mcpl_extended_expected.mcpl",
    )
    mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, ["electron", "positron"])
    with open(out, "rb") as fh_out, open(golden, "rb") as fh_golden:
        assert fh_out.read() == fh_golden.read()


def test_ssw2mcpl_polarisation_and_universal(tmp_path: Path) -> None:
    out = str(tmp_path / "conv.mcpl")
    mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS, {"polarisation": [0.1, 0.2, 0.3]})
    back = mcpl.read_mcpl(out)
    assert back.has_polarisation is True
    for p in back.particles():
        assert p["polarisation"] == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)
    with pytest.raises(ValueError):
        mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS, {"polarisation": [0.0, 0.0]})
    # Universal PDG over a single-kind pairing; mixed kinds are loud.
    mcpl.ssw2mcpl(SSW_REF, out, [100, 200], ["neutron", "neutron"], {"universal_pdg": True})
    assert mcpl.read_mcpl(out).universal_pdgcode == 2112
    with pytest.raises(ValueError):
        mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS, {"universal_pdg": True})
    # Universal weight needs equal weights; the reference pair differs.
    with pytest.raises(ValueError):
        mcpl.ssw2mcpl(SSW_REF, out, SSW_SURFS, SSW_KINDS, {"universal_weight": True})


def test_mcpl2ssw_round_trip_closed_form(tmp_path: Path) -> None:
    probe = str(tmp_path / "probe.mcpl")
    assert mcpl.ssw2mcpl(SSW_REF, probe, SSW_SURFS, SSW_KINDS) == 2
    out = str(tmp_path / "back.w")
    assert mcpl.mcpl2ssw(probe, SSW_REF, out) == 2
    tracks = read_ssw(out).tracks()
    assert len(tracks) == 2
    # Neutron: 3.0 ms -> 3.0e5 shakes, geometry verbatim, no cosine forced.
    assert tracks[0]["erg"] == pytest.approx(2.5)
    assert tracks[0]["tme"] == pytest.approx(3.0e5, rel=1e-6)
    assert tracks[0]["wgt"] == pytest.approx(1.0)
    assert (tracks[0]["x"], tracks[0]["y"], tracks[0]["z"]) == pytest.approx((1.0, -2.0, 0.5))
    assert (tracks[0]["u"], tracks[0]["v"], tracks[0]["cs"]) == pytest.approx((0.0, 0.0, 1.0))
    # Gamma.
    assert tracks[1]["erg"] == pytest.approx(0.662, abs=1e-6)
    assert (tracks[1]["u"], tracks[1]["v"], tracks[1]["cs"]) == pytest.approx(
        (1.0, 0.0, 0.0), abs=1e-6
    )
    # Reference header passes through (counts patched to the converted
    # tally); re-parse is stable.
    ref = read_ssw(SSW_REF)
    again = read_ssw(out)
    assert (again.nrss, again.np1, again.orignp1) == (2, 2, -2)
    assert again.kod == ref.kod
    assert again.ver == ref.ver


def test_mcpl2ssw_surface_override(tmp_path: Path) -> None:
    probe = str(tmp_path / "probe.mcpl")
    mcpl.ssw2mcpl(SSW_REF, probe, SSW_SURFS, SSW_KINDS)
    out = str(tmp_path / "back.w")
    # Override wins over userflags; the write still succeeds (surface ids
    # live outside the track records in this file layout).
    assert mcpl.mcpl2ssw(probe, SSW_REF, out, surface=7) == 2
    assert len(read_ssw(out).tracks()) == 2
    with pytest.raises(ValueError):
        mcpl.mcpl2ssw(probe, SSW_REF, out, surface=1_000_000)


def test_mcpl2ssw_rejects_unsupported_pdg(tmp_path: Path) -> None:
    probe = str(tmp_path / "pion.mcpl")
    mcpl.write_mcpl(probe, HEADER, [dict(PARTICLES[0], pdgcode=211)])
    with pytest.raises(ValueError):
        mcpl.mcpl2ssw(probe, SSW_REF, str(tmp_path / "back.w"))


def test_mcpl2ssw_force_cs_to_one(tmp_path: Path) -> None:
    probe = str(tmp_path / "probe.mcpl")
    mcpl.ssw2mcpl(SSW_REF, probe, SSW_SURFS, SSW_KINDS)
    out = str(tmp_path / "back.w")
    assert mcpl.mcpl2ssw(probe, SSW_REF, out) == 2
    assert read_ssw(out).tracks()[1]["cs"] == pytest.approx(0.0, abs=1e-6)
    assert mcpl.mcpl2ssw(probe, SSW_REF, out, force_cs_to_one=True) == 2
    tracks = read_ssw(out).tracks()
    assert [t["cs"] for t in tracks] == pytest.approx([1.0, 1.0])
    assert (tracks[1]["u"], tracks[1]["v"]) == pytest.approx((1.0, 0.0), abs=1e-6)


def test_mcpl2ssw_niss_passthrough_and_override(tmp_path: Path) -> None:
    probe = str(tmp_path / "probe.mcpl")
    mcpl.ssw2mcpl(SSW_REF, probe, SSW_SURFS, SSW_KINDS)
    out = str(tmp_path / "back.w")
    mcpl.mcpl2ssw(probe, SSW_REF, out)
    assert read_ssw(out).niss == read_ssw(SSW_REF).niss
    mcpl.mcpl2ssw(probe, SSW_REF, out, niss=5)
    assert read_ssw(out).niss == 5
    with pytest.raises(ValueError):
        mcpl.mcpl2ssw(probe, SSW_REF, out, niss=-1)


def test_mcpl2ssw_polarisation_gate(tmp_path: Path) -> None:
    probe = str(tmp_path / "pol.mcpl")
    mcpl.write_mcpl(
        probe,
        {**HEADER, "has_polarisation": True},
        [dict(PARTICLES[1], polarisation=[0.0, 0.0, 1.0])],
    )
    out = str(tmp_path / "back.w")
    with pytest.raises(ValueError):
        mcpl.mcpl2ssw(probe, SSW_REF, out)
    assert mcpl.mcpl2ssw(probe, SSW_REF, out, allow_polarisation=True) == 1


# ── Particle-list utilities: merge / extract / stats / repair ──────────

UTILS = os.path.join(os.path.dirname(__file__), "..", "fixtures", "mcpl", "utils")


def _utils(name: str) -> str:
    return os.path.join(UTILS, name)


def test_merge_mcpl_golden_concat_order(tmp_path: Path) -> None:
    out = str(tmp_path / "merged.mcpl")
    n = mcpl.merge_mcpl([_utils("merge_a.mcpl"), _utils("merge_b.mcpl")], out)
    assert n == 4
    with open(out, "rb") as fh_out, open(_utils("merge_ab.mcpl"), "rb") as fh_golden:
        assert fh_out.read() == fh_golden.read()
    # Field-level: first-file header wins plus the provenance comment.
    merged = mcpl.read_mcpl(out)
    assert merged.srcname == "merge-a"
    assert merged.comments == ["first file", "Merged by nucleide-mcpl-io merge_mcpl from 2 files"]
    assert merged.double_prec is False
    got = merged.particles()
    assert [p["userflags"] for p in got] == [100, 200, 300, 0]
    assert [p["ekin"] for p in got[:3]] == pytest.approx([2.5, 0.662, 1.0], abs=1e-6)


def test_merge_mcpl_promotes_precision_golden(tmp_path: Path) -> None:
    out = str(tmp_path / "merged.mcpl")
    n = mcpl.merge_mcpl([_utils("merge_a.mcpl"), _utils("merge_c.mcpl")], out)
    assert n == 3
    with open(out, "rb") as fh_out, open(_utils("merge_ac.mcpl"), "rb") as fh_golden:
        assert fh_out.read() == fh_golden.read()
    merged = mcpl.read_mcpl(out)
    assert merged.double_prec is True
    # The single-precision input decodes exactly into the double output.
    ekins = [p["ekin"] for p in merged.particles()]
    assert ekins[0] == 2.5
    assert ekins[1] == pytest.approx(0.6620000004768372)
    assert ekins[2] == 14.1


def test_merge_mcpl_rejects_incompatible_and_empty(tmp_path: Path) -> None:
    out = str(tmp_path / "merged.mcpl")
    polarised = str(tmp_path / "pol.mcpl")
    mcpl.write_mcpl(polarised, {**HEADER, "has_polarisation": True}, PARTICLES)
    with pytest.raises(ValueError):
        mcpl.merge_mcpl([_utils("merge_a.mcpl"), polarised], out)
    with pytest.raises(ValueError):
        mcpl.merge_mcpl([], out)
    # Two double-precision inputs with different universal codes disagree.
    u1 = str(tmp_path / "u1.mcpl")
    u2 = str(tmp_path / "u2.mcpl")
    mcpl.write_mcpl(u1, {**HEADER, "universal_pdgcode": 2112}, PARTICLES)
    mcpl.write_mcpl(u2, {**HEADER, "universal_pdgcode": 22}, PARTICLES)
    with pytest.raises(ValueError):
        mcpl.merge_mcpl([u1, u2], out)


def test_merge_mcpl_never_synthesizes_statsum(tmp_path: Path) -> None:
    src = str(tmp_path / "sum.mcpl")
    mcpl.write_mcpl(
        src,
        {**HEADER, "comments": ["run a", mcpl.mcpl_statsum_comment("nps", 2.0)]},
        PARTICLES,
    )
    out = str(tmp_path / "merged.mcpl")
    mcpl.merge_mcpl([src, _utils("merge_b.mcpl")], out)
    comments = mcpl.read_mcpl(out).comments
    # First-file stat:sum rides along verbatim, no sums synthesized or updated.
    assert comments[1] == mcpl.mcpl_statsum_comment("nps", 2.0)
    assert sum(c.startswith("stat:sum:") for c in comments) == 1
    assert comments[-1].startswith("Merged by nucleide-mcpl-io merge_mcpl")


def test_extract_mcpl_range_golden_header_verbatim(tmp_path: Path) -> None:
    out = str(tmp_path / "sub.mcpl")
    n = mcpl.extract_mcpl(_utils("extract_src.mcpl"), out, {"start": 1, "stop": 3})
    assert n == 2
    with open(out, "rb") as fh_out, open(_utils("extract_range.mcpl"), "rb") as fh_golden:
        assert fh_out.read() == fh_golden.read()
    sub = mcpl.read_mcpl(out)
    src = mcpl.read_mcpl(_utils("extract_src.mcpl"))
    assert sub.srcname == src.srcname == "extract-src"
    assert sub.comments == src.comments == ["extract source"]
    assert sub.blobs == src.blobs == [("meta", b"\x01\x02\x03")]
    assert sub.double_prec is True
    got = sub.particles()
    assert [p["pdgcode"] for p in got] == [22, 2112]
    assert [p["ekin"] for p in got] == pytest.approx([0.662, 14.1])


def test_extract_mcpl_predicate_golden(tmp_path: Path) -> None:
    out = str(tmp_path / "sub.mcpl")
    n = mcpl.extract_mcpl(
        _utils("extract_src.mcpl"),
        out,
        {"predicate": lambda p: p["pdgcode"] == 2112},
    )
    assert n == 2
    with open(out, "rb") as fh_out, open(_utils("extract_pdg.mcpl"), "rb") as fh_golden:
        assert fh_out.read() == fh_golden.read()
    assert [p["ekin"] for p in mcpl.read_mcpl(out).particles()] == pytest.approx([2.5, 14.1])


def test_extract_mcpl_options_and_errors(tmp_path: Path) -> None:
    src = _utils("extract_src.mcpl")
    out = str(tmp_path / "sub.mcpl")
    # Default copies the whole file; open-ended ranges slice like Python.
    assert mcpl.extract_mcpl(src, out) == 4
    assert mcpl.extract_mcpl(src, out, {"stop": 2}) == 2
    assert mcpl.extract_mcpl(src, out, {"start": 3}) == 1
    # Round-trip both directions: extracted subset merges back losslessly.
    mcpl.extract_mcpl(src, out, {"start": 1, "stop": 3})
    back = str(tmp_path / "back.mcpl")
    mcpl.merge_mcpl([out], back)
    assert [p["ekin"] for p in mcpl.read_mcpl(back).particles()] == pytest.approx([0.662, 14.1])
    with pytest.raises(ValueError):
        mcpl.extract_mcpl(src, out, {"start": 3, "stop": 99})
    with pytest.raises(ValueError):
        mcpl.extract_mcpl(src, out, {"start": -1})
    with pytest.raises(ValueError):
        mcpl.extract_mcpl(src, out, {"predicate": "not-callable"})
    with pytest.raises(ValueError):
        mcpl.extract_mcpl(src, out, {"start": 0, "predicate": lambda p: True})
    with pytest.raises(ValueError):
        mcpl.extract_mcpl(src, out, "not-a-dict")  # type: ignore[arg-type]


def test_extract_mcpl_predicate_exception_propagates(tmp_path: Path) -> None:
    src = _utils("extract_src.mcpl")
    out = str(tmp_path / "sub.mcpl")

    def boom(p: dict[str, object]) -> bool:
        raise RuntimeError("predicate blew up")

    with pytest.raises(RuntimeError, match="predicate blew up"):
        mcpl.extract_mcpl(src, out, {"predicate": boom})


def test_mcpl_stats_hand_moments(tmp_path: Path) -> None:
    s = mcpl.mcpl_stats(_utils("extract_src.mcpl"))
    assert s["nparticles"] == 4
    assert s["ekin_sum"] == pytest.approx(2.5 + 0.662 + 14.1 + 5.0)
    assert s["ekin_min"] == pytest.approx(0.662)
    assert s["ekin_max"] == pytest.approx(14.1)
    assert s["ekin_mean"] == pytest.approx((2.5 + 0.662 + 14.1 + 5.0) / 4.0)
    assert s["weight_sum"] == pytest.approx(1.0 + 0.5 + 2.0 + 1.5)
    assert s["pdg_counts"] == [(22, 1), (2112, 2), (2212, 1)]


def test_mcpl_stats_empty_and_universal(tmp_path: Path) -> None:
    empty = str(tmp_path / "empty.mcpl")
    mcpl.write_mcpl(empty, HEADER, [])
    s = mcpl.mcpl_stats(empty)
    assert s["nparticles"] == 0
    assert s["ekin_min"] is None
    assert s["ekin_max"] is None
    assert s["ekin_mean"] is None
    assert s["ekin_sum"] == 0.0
    assert s["weight_sum"] == 0.0
    assert s["pdg_counts"] == []

    uni = str(tmp_path / "uni.mcpl")
    mcpl.write_mcpl(uni, {**HEADER, "universal_pdgcode": 2112, "universal_weight": 1.5}, PARTICLES)
    s = mcpl.mcpl_stats(uni)
    assert s["pdg_counts"] == [(2112, 2)]
    assert s["weight_sum"] == pytest.approx(3.0)


def test_repair_mcpl_unpatched_count(tmp_path: Path) -> None:
    probe = tmp_path / "broken.mcpl"
    mcpl.write_mcpl(str(probe), HEADER, PARTICLES)
    good = probe.read_bytes()
    # Simulate an interrupted job: the count field never got patched.
    broken = bytearray(good)
    broken[8:16] = (1).to_bytes(8, "little")
    probe.write_bytes(bytes(broken))
    with pytest.raises(ValueError):
        mcpl.read_mcpl(str(probe)).particles()
    n = mcpl.repair_mcpl(str(probe))
    assert n == 2
    assert probe.read_bytes() == good
    assert mcpl.read_mcpl(str(probe)).particles()[0]["ekin"] == pytest.approx(2.5)


def test_repair_mcpl_drops_partial_trailing_record(tmp_path: Path) -> None:
    probe = tmp_path / "broken.mcpl"
    mcpl.write_mcpl(str(probe), HEADER, PARTICLES)
    good = probe.read_bytes()
    # Particle record size lives in the 7th header option field (offset 40).
    particle_size = int.from_bytes(good[40:44], "little")
    header_len = len(good) - 2 * particle_size
    # An interrupted final write leaves a half-record at the end.
    probe.write_bytes(good[: len(good) - particle_size // 2])
    n = mcpl.repair_mcpl(str(probe))
    assert n == 1
    repaired = mcpl.read_mcpl(str(probe))
    assert repaired.nparticles == 1
    assert repaired.particles()[0]["ekin"] == pytest.approx(2.5)
    # Only the complete records are kept: header plus one record.
    assert len(probe.read_bytes()) == header_len + particle_size


def test_repair_mcpl_gzip_transparent(tmp_path: Path) -> None:
    path = str(tmp_path / "broken.mcpl.gz")
    mcpl.write_mcpl(path, HEADER, PARTICLES)
    with gzip.open(path, "rb") as fh:
        good = fh.read()
    broken = bytearray(good)
    broken[8:16] = (9).to_bytes(8, "little")
    with gzip.open(path, "wb") as fh:
        fh.write(bytes(broken))
    n = mcpl.repair_mcpl(path)
    assert n == 2
    with gzip.open(path, "rb") as fh:
        assert fh.read() == good
