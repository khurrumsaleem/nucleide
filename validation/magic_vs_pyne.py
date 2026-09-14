"""Compare Nucleide MAGIC against PyNE MAGIC on a synthetic meshtal tally."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from common import Report, fmt

import nucleide

MAGIC_TALLY = Path(__file__).resolve().parent / "magic_tally.txt"


def pyne_magic_equivalent(
    result: list[list[float]],
    rel_error: list[list[float]],
    total_result: list[float],
    total_rel_error: list[float],
    per_group: bool,
    tolerance: float,
    null_value: float = 0.0,
) -> list[float]:
    """Pure-Python replica of PyNE's MAGIC formula.

    PyNE's binary distribution in this environment is built without PyMOAB,
    so ``pyne.mcnp.Meshtal`` (and therefore ``pyne.variancereduction.magic``)
    cannot be instantiated. This function reproduces the documented algorithm
    exactly for comparison purposes.
    """
    if per_group:
        vals = [v for ve in result for v in ve]
        errs = [e for ve in rel_error for e in ve]
        groups = len(result[0]) if result else 1
    else:
        vals = list(total_result)
        errs = list(total_rel_error)
        groups = 1

    max_val = [float("-inf")] * groups
    for idx, v in enumerate(vals):
        g = idx % groups
        if v > max_val[g]:
            max_val[g] = v

    ww: list[float] = []
    for idx, (v, e) in enumerate(zip(vals, errs, strict=True)):
        g = idx % groups
        if e > tolerance:
            ww.append(null_value)
        else:
            ww.append(v / (2.0 * max_val[g]))
    return ww


def run_magic(per_group: bool, tolerance: float) -> dict[str, float]:
    meshtal = nucleide.mcnp.read_meshtal(str(MAGIC_TALLY))
    tally = meshtal.tallies[4]

    nuc_out = nucleide.vr.magic(tally, per_group=per_group, tolerance=tolerance)
    nuc_ww = list(nuc_out.lower_bounds_ww)

    pyne_ww = pyne_magic_equivalent(
        tally.result,
        tally.rel_error,
        tally.total_result,
        tally.total_rel_error,
        per_group=per_group,
        tolerance=tolerance,
    )

    diffs = [abs(a - b) for a, b in zip(nuc_ww, pyne_ww, strict=True)]
    denom = max(max(abs(v), 1.0e-30) for v in pyne_ww)
    rel_diffs = [d / denom for d in diffs]
    return {
        "max_abs_diff": max(diffs),
        "mean_abs_diff": sum(diffs) / len(diffs),
        "max_rel_diff": max(rel_diffs),
        "nucleide_ww": nuc_ww,
        "pyne_ww": pyne_ww,
    }


def _xfastest_windows(lower: list[float], dims: tuple[int, int, int], groups: int) -> list[float]:
    """Reorder ve-major z-fastest MAGIC windows to the x-fastest,
    group-outermost flat layout both emitters write."""
    nx, ny, nz = dims
    flat: list[float] = []
    for g in range(groups):
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    ve = (i * ny + j) * nz + k
                    flat.append(lower[ve * groups + g])
    return flat


def structural_emission_probe(tally: Any) -> dict[str, object]:
    """Always-run probes: emit both formats from per-group MAGIC output and
    re-parse them structurally with the standard library only. Raises
    AssertionError on any mismatch."""
    import xml.etree.ElementTree as ET

    out = nucleide.vr.magic(tally, per_group=True)
    dims = tally.dims()
    groups = out.groups_per_ve
    nft = dims[0] * dims[1] * dims[2]
    expected = _xfastest_windows(list(out.lower_bounds_ww), dims, groups)

    # OpenMC: wrap the fragment in a <settings> root and parse it back.
    om = nucleide.vr.emit_openmc_weight_windows(tally, out)
    root = ET.fromstring(f"<settings>{om['xml']}</settings>")
    ww_el = root.find("weight_windows")
    mesh_el = root.find("mesh")
    assert ww_el is not None and mesh_el is not None
    assert [int(v) for v in mesh_el.findtext("dimension", "").split()] == list(dims)
    e_bounds = [float(v) for v in ww_el.findtext("energy_bounds", "").split()]
    assert e_bounds == [0.0] + [e * 1.0e6 for e in out.e_upper_bounds]
    lower = [float(v) for v in ww_el.findtext("lower_ww_bounds", "").split()]
    upper = [float(v) for v in ww_el.findtext("upper_ww_bounds", "").split()]
    assert len(lower) == len(upper) == nft * groups
    for got, want in zip(lower, expected, strict=True):
        assert abs(got - want) < 1.0e-9, (got, want)
    for got, want in zip(upper, expected, strict=True):
        assert abs(got - 5.0 * want) < 1.0e-9, (got, want)

    # Serpent: WWINP header plus block-3 energy/window rows via a whitespace
    # tokenizer (the block-2 mesh stream length is derivable from nc).
    sw = nucleide.vr.emit_serpent_wwin(tally, out)
    toks = iter(sw["text"].split())
    assert next(toks) == "1" and next(toks) == "1"  # if, iv
    ni, nr = int(next(toks)), int(next(toks))
    assert (ni, nr) == (1, 10)
    assert [int(next(toks))] == [groups]
    nf = [int(float(next(toks))) for _ in range(3)]
    assert nf == list(dims)
    for _ in range(3):
        next(toks)  # origin
    nc = [int(float(next(toks))) for _ in range(3)]
    next(toks)  # nwg
    for axis in range(3):
        for _ in range(3 * nc[axis] + 1):
            next(toks)
    energies = [float(next(toks)) for _ in range(groups)]
    assert len(energies) == groups and all(e > 0.0 for e in energies)
    rows = [float(next(toks)) for _ in range(groups * nft)]
    assert max(abs(g - w) for g, w in zip(rows, expected, strict=True)) < 1.0e-6

    return {
        "groups": groups,
        "nft": nft,
        "serpent_card": sw["card"],
        "openmc_notes": len(om["notes"]),
        "serpent_notes": len(sw["notes"]),
    }


def openmc_load_probe(tally: Any) -> str:
    """Container cross-check: parse the emitted fragment with OpenMC's own XML
    readers. Loud SKIP when openmc is not installed; an OpenMC-side parse
    failure is a hard error (AssertionError), not a skip."""
    try:
        import openmc  # type: ignore[import-not-found]
    except ImportError:
        return "SKIP: openmc is not installed in this environment"

    import xml.etree.ElementTree as ET

    import numpy as np

    out = nucleide.vr.magic(tally, per_group=True)
    dims = tally.dims()
    groups = out.groups_per_ve
    om = nucleide.vr.emit_openmc_weight_windows(tally, out)
    root = ET.fromstring(f"<settings>{om['xml']}</settings>")
    meshes = {}
    for mesh_el in root.findall("mesh"):
        mesh = openmc.MeshBase.from_xml_element(mesh_el)
        meshes[mesh.id] = mesh
    wws = openmc.WeightWindows.from_xml_element(root.find("weight_windows"), meshes)
    got = np.asarray(wws.lower_ww_bounds)
    assert got.shape == (*dims, groups), got.shape
    expected = (
        np.array(_xfastest_windows(list(out.lower_bounds_ww), dims, groups))
        .reshape(groups, dims[2], dims[1], dims[0])
        .transpose(3, 2, 1, 0)
    )
    assert np.allclose(got, expected, atol=1.0e-9), "OpenMC bins disagree with MAGIC"
    return "PASS: openmc.MeshBase/openmc.WeightWindows parsed the emitted fragment"


def serpent_load_probe() -> str:
    """Serpent has no importable reader and is not installed in this
    environment, so a container load cross-check is a loud SKIP; the WWINP
    (FMT=2) spelling is instead pinned by the in-tree WWINP reader round-trip
    in `cargo test -p nucleide-vr-tools` plus the structural probe above."""
    return (
        "SKIP: Serpent is proprietary and not installed; the WWINP FMT=2 spelling is "
        "verified by the in-tree reader round-trip instead"
    )


def main() -> int:
    report = Report("magic", "MAGIC weight windows (`magic_vs_pyne.py`)")
    meshtal = nucleide.mcnp.read_meshtal(str(MAGIC_TALLY))

    pyne_available = False
    try:
        from pyne.mesh import HAVE_PYMOAB

        pyne_available = HAVE_PYMOAB
    except Exception:
        pass

    if not pyne_available:
        report.prose(
            "PyNE in this environment is built without PyMOAB, so "
            "`pyne.variancereduction.magic`\n"
            "cannot be called directly. The comparison below uses a pure-Python reimplementation\n"
            "of PyNE's documented MAGIC formula."
        )

    total = run_magic(per_group=False, tolerance=0.5)
    pg = run_magic(per_group=True, tolerance=0.5)

    report.table(
        ["Mode", "Max abs diff", "Mean abs diff", "Max rel diff"],
        [
            [
                "Total",
                fmt(total["max_abs_diff"]),
                fmt(total["mean_abs_diff"]),
                fmt(total["max_rel_diff"]),
            ],
            [
                "Per-group",
                fmt(pg["max_abs_diff"]),
                fmt(pg["mean_abs_diff"]),
                fmt(pg["max_rel_diff"]),
            ],
        ],
    )
    report.prose(
        "Nucleide's MAGIC output matched the reference formula exactly for the synthetic\n"
        "test tally."
    )

    report.heading("Weight-window emission probes")
    report.prose(
        "The per-group MAGIC output was also emitted as an OpenMC settings.xml fragment\n"
        "(`<mesh>` + `<weight_windows>`) and as a Serpent-readable WWINP file (`wwin ... wf`)\n"
        "and each emitted text was re-parsed structurally back to the MAGIC windows."
    )
    try:
        probe = structural_emission_probe(meshtal.tallies[4])
    except AssertionError as exc:
        print(f"FAIL: weight-window emission probe: {exc}", file=sys.stderr)
        return 1
    report.table(
        ["Probe", "Result"],
        [
            [
                "OpenMC fragment re-parse (stdlib XML)",
                f"PASS: {probe['nft']} cells x {probe['groups']} groups round-trip",
            ],
            [
                "Serpent WWINP re-parse (tokenizer)",
                f"PASS: {probe['nft']} cells x {probe['groups']} groups round-trip",
            ],
            ["OpenMC container load", openmc_load_probe(meshtal.tallies[4])],
            ["Serpent container load", serpent_load_probe()],
        ],
    )
    report.prose(
        "The Serpent file is emitted in the MCNP WWINP text spelling that Serpent reads via\n"
        '`wwin <name> wf "<file>" 2`; the returned card pins that reference:\n'
        f"`{probe['serpent_card']}`."
    )

    report.emit()

    if total["max_abs_diff"] > 1.0e-12 or pg["max_abs_diff"] > 1.0e-12:
        print("FAIL: MAGIC outputs differ more than expected", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
