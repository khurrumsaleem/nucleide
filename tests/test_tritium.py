"""Python-side tests for the 1D tritium-transport kernel (G1-G4 gates + input errors)."""

import json
from pathlib import Path

import pytest

import nucleide.tritium as tri

FIX = Path(__file__).parent.parent / "fixtures" / "tritium"

DIR0 = {"kind": "dirichlet", "value": 1.0}
DIR1 = {"kind": "dirichlet", "value": 0.0}


def test_g1_steady_linear() -> None:
    oracle = json.loads((FIX / "g1_steady.json").read_text())
    p = oracle["params"]
    out = tri.steady(p["L"], 32, p["D"], DIR0, DIR1)
    n = len(out["mobile"])
    for i, c in enumerate(out["mobile"]):
        assert c == pytest.approx(1.0 - (i + 0.5) / n, rel=1e-12)
    assert out["flux_left"] == pytest.approx(-oracle["J_ss"], rel=1e-12)
    assert out["flux_right"] == pytest.approx(oracle["J_ss"], rel=1e-12)
    assert out["inventory_mobile"] == pytest.approx(oracle["inventory"], rel=1e-12)
    assert all(c >= 0.0 for c in out["mobile"])


def test_g2_time_lag_and_breakthrough() -> None:
    oracle = json.loads((FIX / "g2_timelag.json").read_text())
    p = oracle["params"]
    assert tri.time_lag(p["L"], p["D"]) == pytest.approx(oracle["t_lag"], rel=1e-12)
    got = tri.breakthrough(p["D"], p["L"], oracle["t"])
    for g, w in zip(got, oracle["J_over_Jss"], strict=True):
        assert g == pytest.approx(w, rel=1e-12, abs=1e-12)
    out = tri.transient(
        p["L"], 1000, p["D"], DIR0, DIR1, oracle["t"], dt_max=0.1, rtol=1e-10, atol=1e-14
    )
    jss = p["D"] * p["c0"] / p["L"]
    for f, w in zip(out["flux_right"], oracle["J_over_Jss"], strict=True):
        assert f / jss == pytest.approx(w, rel=1e-6, abs=1e-6)
    assert all(c >= 0.0 for row in out["mobile"] for c in row)


def test_g3a_oriani() -> None:
    oracle = json.loads((FIX / "g3a_oriani.json").read_text())
    p = oracle["params"]
    assert tri.oriani(p["D"], p["K"], p["N"]) == pytest.approx(oracle["D_eff"], rel=1e-12)


def test_g3b_saturated() -> None:
    oracle = json.loads((FIX / "g3b_saturated.json").read_text())
    p = oracle["params"]
    for cm, wt in zip(oracle["c_mobile"], oracle["c_trapped"], strict=True):
        assert tri.langmuir(p["N"], p["K"], cm) == pytest.approx(wt, rel=1e-12)
    out = tri.steady(
        p["L"],
        512,
        p["D"],
        {"kind": "dirichlet", "value": p["c0"]},
        {"kind": "dirichlet", "value": p["cL"]},
        traps=[{"k0": p["K"], "p0": 1.0, "site_density": p["N"]}],
    )
    assert out["inventory_mobile"] == pytest.approx(oracle["inventory_mobile"], rel=1e-12)
    assert out["inventory_trapped"] == pytest.approx(oracle["inventory_trapped"], rel=1e-6)


def test_g3c_irreversible() -> None:
    oracle = json.loads((FIX / "g3c_irreversible.json").read_text())
    p = oracle["params"]
    got = tri.irreversible_fill(p["k"], p["c"], p["N"], oracle["t"])
    for g, w in zip(got, oracle["c_trapped"], strict=True):
        assert g == pytest.approx(w, rel=1e-12)


def test_g4_sieverts() -> None:
    oracle = json.loads((FIX / "g4_sieverts.json").read_text())
    p = oracle["params"]
    assert tri.sieverts(p["K_S"], p["p1"]) == pytest.approx(oracle["c0"], rel=1e-12)
    assert tri.sieverts(p["K_S"], p["p2"]) == pytest.approx(oracle["cL"], rel=1e-12)
    out = tri.steady(
        p["L"],
        32,
        p["D"],
        {"kind": "sieverts", "solubility": p["K_S"], "pressure": p["p1"]},
        {"kind": "sieverts", "solubility": p["K_S"], "pressure": p["p2"]},
    )
    n = len(out["mobile"])
    for i, c in enumerate(out["mobile"]):
        x = (i + 0.5) / n
        assert c == pytest.approx(oracle["c0"] + (oracle["cL"] - oracle["c0"]) * x, rel=1e-12)
    assert out["flux_right"] == pytest.approx(oracle["J"], rel=1e-12)
    assert out["inventory_mobile"] == pytest.approx(oracle["inventory"], rel=1e-12)


def test_g5a_dirichlet_recombination() -> None:
    # G5a closed form: D=1e-9, L=1e-3, c0=1.0, K_r=1e-6 ->
    # c_s=(sqrt(5)-1)/2, J=K_r c_s^2, I=(c0+c_s)L/2.
    cs = (5.0**0.5 - 1.0) / 2.0
    out = tri.steady(1e-3, 512, 1e-9, DIR0, {"kind": "recombination", "rate": 1e-6})
    n = len(out["mobile"])
    for i, c in enumerate(out["mobile"]):
        assert c == pytest.approx(1.0 + (cs - 1.0) * (i + 0.5) / n, rel=1e-12)
    assert out["flux_right"] == pytest.approx(1e-6 * cs * cs, rel=1e-12)
    assert out["flux_left"] == pytest.approx(-1e-6 * cs * cs, rel=1e-12)
    assert out["inventory_mobile"] == pytest.approx((1.0 + cs) * 1e-3 / 2.0, rel=1e-12)
    assert all(c >= 0.0 for c in out["mobile"])


def test_g5b_large_rate_recovers_dirichlet() -> None:
    out = tri.steady(1e-3, 64, 1e-9, DIR0, {"kind": "recombination", "rate": 1e12})
    assert out["flux_right"] == pytest.approx(1e-6, rel=1e-6)
    assert out["inventory_mobile"] == pytest.approx(5e-4, rel=1e-6)


def test_g5_recombination_rate_helper() -> None:
    assert tri.recombination_rate(1.0, 0.0, 500.0) == pytest.approx(1.0, rel=1e-15)
    import math

    assert tri.recombination_rate(2.0, 8.314 * 500.0, 500.0) == pytest.approx(
        2.0 / math.e, rel=1e-12
    )
    with pytest.raises(ValueError):
        tri.recombination_rate(-1.0, 0.0, 500.0)
    with pytest.raises(ValueError):
        tri.recombination_rate(1.0, 0.0, 0.0)


def test_recombination_transient_is_named_open() -> None:
    rec = {"kind": "recombination", "rate": 1.0}
    with pytest.raises(ValueError, match="named-open"):
        tri.transient(1e-3, 8, 1e-9, rec, DIR1, [1.0])


def test_mass_conservation_sealed_source() -> None:
    out = tri.transient(
        1e-3,
        16,
        1e-9,
        {"kind": "zero_flux"},
        {"kind": "zero_flux"},
        [10.0, 40.0],
        source=[2.0],
        mobile0=[1.0] * 16,
    )
    dx = 1e-3 / 16
    for row, t in zip(out["mobile"], [10.0, 40.0], strict=True):
        assert sum(row) * dx == pytest.approx(1e-3 + 2.0 * 1e-3 * t, rel=1e-12)


def test_input_errors() -> None:
    with pytest.raises(ValueError):
        tri.steady(1e-3, 1, 1e-9, DIR0, DIR1)
    with pytest.raises(ValueError):
        tri.steady(1e-3, 8, 1e-9, {"kind": "dirichlet", "value": -1.0}, DIR1)
    with pytest.raises(ValueError):
        tri.steady(1e-3, 8, 1e-9, {"kind": "nope"}, DIR1)
    with pytest.raises(ValueError):
        tri.transient(1e-3, 8, 1e-9, DIR0, DIR1, [1.0, 1.0])
    with pytest.raises(ValueError):
        tri.transient(1e-3, 8, 1e-9, DIR0, DIR1, [1.0], method="rk4")
    with pytest.raises(ValueError):
        tri.transient(1e-3, 8, 1e-9, DIR0, DIR1, [1.0], mobile0=[-1.0] * 8)
    with pytest.raises(ValueError):
        tri.time_lag(1e-3, 0.0)
    with pytest.raises(ValueError):
        tri.breakthrough(1e-9, 1e-3, [0.0])
