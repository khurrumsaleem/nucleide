"""Tokamak fusion-source cross-check (`nucleide.plasma_source` vs analytic gates +
openmc-plasma-source).

Two parts:

1. Analytic gates (always run): P1 ring/point spatial moments against the
   closed form (radius/height/azimuth), P2 Gaussian spectrum mean/width
   against an independent in-script transcription of the Ballabio et al.
   1998 Nucl. Fusion 38 1723 Table III fits, P3 sampler determinism under a
   pinned seed, P4 card emission (SDEF byte round trip through the typed
   reader, drift rows, Serpent structure).
2. openmc-plasma-source cross-check (container oracle): O1 coefficient
   transcription vs the upstream package's published-fit helpers, O2 D-D
   energy moments on sampled particles from the upstream ring source, O3
   ring geometry moments vs the upstream `CylindricalIndependent` space.
   The upstream package is an optional oracle dependency: if it cannot be
   imported, the oracle checks are reported as SKIP with their reason (never
   silently).

Units: Nucleide works in cm/MeV/keV; openmc-plasma-source in m/eV — the
oracle converts at the boundary.
"""

from __future__ import annotations

import math
import sys

import numpy as np
from common import Report, fmt, rel_diff

import nucleide.mcnp as mcnp
import nucleide.plasma_source as ps

FAILURES = 0

# Tokamak ring: R = 3 m at z = 0.5 m, D-T, T_i = 20 keV (ITER-ish numbers,
# hand-picked round values).
RADIUS_CM = 300.0
HEIGHT_CM = 50.0
TI_KEV = 20.0
N = 200_000

RING_SPEC = {
    "kind": "ring",
    "radius": RADIUS_CM,
    "height": HEIGHT_CM,
    "reaction": "dt",
    "ion_temperature_kev": TI_KEV,
}


def _check(ok: bool, label: str) -> str:
    global FAILURES
    if not ok:
        FAILURES += 1
        print(f"FAIL: {label}", file=sys.stderr)
    return "PASS" if ok else "FAIL"


def _ballabio_moments(reaction: str, ti_kev: float) -> tuple[float, float]:
    """Independent in-script transcription of the Table III fits (keV in, MeV out)."""
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


def analytic_gates() -> tuple[list[list[str]], list[str]]:
    """P1-P4 on the synthetic ring spec; returns (gate rows, prose notes)."""
    rows: list[list[str]] = []
    notes: list[str] = []

    out = ps.particles(RING_SPEC, N, seed=42)
    radius = np.hypot(out["x"], out["y"])
    notes.append(
        f"P1 ring sampling: n={N}, radius min/max {radius.min():.6f}/{radius.max():.6f} cm."
    )
    ok = bool(np.all(out["z"] == HEIGHT_CM)) and bool(
        np.all(np.abs(radius - RADIUS_CM) < 1e-9 * RADIUS_CM)
    )
    rows.append(["P1 ring radius/height", "-", "exact", _check(ok, "P1 radius/height")])
    se = RADIUS_CM / math.sqrt(2 * N)
    for label, value in (
        ("P1 <x>", float(np.mean(out["x"]))),
        ("P1 <y>", float(np.mean(out["y"]))),
    ):
        err = abs(value) / se
        rows.append([label, fmt(value), "< 8 sigma", _check(err < 8.0, label)])
    mean_w = float(np.mean(out["w"]))
    rows.append(
        [
            "P1 direction isotropy <w>",
            fmt(mean_w),
            "< 1e-2",
            _check(abs(mean_w) < 1e-2, "P1 isotropy"),
        ]
    )

    for reaction in ("dd", "dt"):
        for ti in (1.0, 10.0, 20.0):
            got = ps.spectrum_moments(reaction, ti)
            mean, sigma = _ballabio_moments(reaction, ti)
            worst = max(rel_diff(got["mean_mev"], mean), rel_diff(got["sigma_mev"], sigma))
            rows.append(
                [
                    f"P2 {reaction.upper()} Ti={ti:g} keV moments",
                    fmt(worst),
                    "< 1e-12",
                    _check(worst < 1e-12, f"P2 {reaction} {ti}"),
                ]
            )
    notes.append("P2 spectrum moments vs in-script Ballabio 1998 Table III transcription.")

    a = ps.particles(RING_SPEC, 512, seed=1234)
    b = ps.particles(RING_SPEC, 512, seed=1234)
    c = ps.particles(RING_SPEC, 512, seed=1235)
    ok = all(bool(np.array_equal(a[k], b[k])) for k in a) and not np.array_equal(
        a["energy"], c["energy"]
    )
    rows.append(["P3 pinned-seed determinism", "-", "identical", _check(ok, "P3 determinism")])

    emitted = ps.emit_source_cards(RING_SPEC)
    card = emitted["sdef"]["card"]
    parsed = mcnp.parse_sdef(card)
    ok = parsed["card"] == card and parsed["rad"] == "D1" and parsed["erg"] == "D2"
    rows.append(
        ["P4 SDEF reader round trip", "-", "byte-identical", _check(ok, "P4 SDEF round trip")]
    )
    drift = emitted["sdef"]["drift"][0]
    tail = 1.0 - math.erf(4.0 / math.sqrt(2.0))
    rows.append(
        [
            "P4 tabulation tail drift",
            fmt(drift["rel_drift"]),
            "~ +/-4 sigma tail",
            _check(rel_diff(drift["rel_drift"], tail) < 1e-3, "P4 tail drift"),
        ]
    )
    serpent = emitted["serpent"]["card"].splitlines()
    ok = (
        serpent[1] == "src 1 rad d1"
        and serpent[4] == f"SI1 {RADIUS_CM:g} {RADIUS_CM:g}"
        and emitted["serpent"]["drift"][0]["reparsed"] is False
    )
    rows.append(["P4 Serpent src structure", "-", "shape", _check(ok, "P4 Serpent structure")])
    notes.append("P4 Serpent drift rows are analytic by design (no reader in the workspace).")
    return rows, notes


def oracle_check_openmc_plasma_source() -> tuple[list[list[str]], list[str], bool]:
    """O1-O3 vs the upstream MIT-licensed package. Returns (rows, notes, skipped)."""
    # NeSST (< 1.2) still imports scipy.integrate.cumtrapz, removed in scipy
    # 1.14; the modern spelling is a drop-in for its usage. Shim the alias in
    # this oracle process only (never in the shipped library).
    try:
        import scipy.integrate
        from scipy.integrate import cumulative_trapezoid

        scipy.integrate.cumtrapz = cumulative_trapezoid  # type: ignore[attr-defined]
    except ImportError:
        pass
    try:
        from openmc_plasma_source import fusion_point_source, fusion_ring_source
        from openmc_plasma_source.fuel_types import (
            neutron_energy_mean,
            neutron_energy_std_dev,
        )
    except Exception as exc:  # noqa: BLE001 — oracle is optional; reason recorded
        note = f"Oracle check (openmc-plasma-source) SKIPPED: {exc}"
        print(note)
        return [], [note], True

    rows: list[list[str]] = []
    notes: list[str] = []

    # O1: coefficient transcription vs the upstream published-fit helpers.
    worst = 0.0
    for reaction, upstream in (("dd", "DD"), ("dt", "DT")):
        for ti_kev in (1.0, 10.0, 20.0):
            got = ps.spectrum_moments(reaction, ti_kev)
            mean_ev = neutron_energy_mean(ion_temperature=ti_kev * 1e3, reaction=upstream)
            std_ev = neutron_energy_std_dev(ion_temperature=ti_kev * 1e3, reaction=upstream)
            err = max(
                rel_diff(got["mean_mev"], mean_ev / 1e6),
                rel_diff(got["sigma_mev"], std_ev / 1e6),
            )
            worst = max(worst, err)
    rows.append(
        ["O1 Ballabio fit transcription", fmt(worst), "< 1e-12", _check(worst < 1e-12, "O1 fits")]
    )
    notes.append(
        "O1 compares the Rust coefficient transcription against the upstream "
        "package's Table III helpers at three ion temperatures."
    )

    # O2: D-D energy moments on sampled particles (upstream fuel={"D": 1.0} is
    # the pure D-D Gaussian; mixtures are out of v1 scope here).
    ti_ev = TI_KEV * 1e3
    (upstream,) = fusion_ring_source(
        radius=RADIUS_CM / 100.0,
        z_placement=HEIGHT_CM / 100.0,
        temperature=ti_ev,
        fuel={"D": 1.0},
    )
    rng = np.random.default_rng(2026)
    try:
        upstream_e = np.asarray(upstream.energy.sample(N, seed=1)) / 1e6  # eV -> MeV
    except (AttributeError, TypeError):
        mean_ev = neutron_energy_mean(ion_temperature=ti_ev, reaction="DD")
        std_ev = neutron_energy_std_dev(ion_temperature=ti_ev, reaction="DD")
        upstream_e = rng.normal(mean_ev, std_ev, N) / 1e6
    ours = ps.particles(dict(RING_SPEC, reaction="dd"), N, seed=42)
    err_mean = rel_diff(float(np.mean(ours["energy"])), float(np.mean(upstream_e)))
    err_std = rel_diff(float(np.std(ours["energy"])), float(np.std(upstream_e)))
    rows.append(["O2 DD energy mean", fmt(err_mean), "< 1e-2", _check(err_mean < 1e-2, "O2 mean")])
    rows.append(["O2 DD energy std", fmt(err_std), "< 1e-2", _check(err_std < 1e-2, "O2 std")])
    notes.append(
        f"O2 D-D sampled moments at T_i={TI_KEV:g} keV, n={N} per side "
        "(upstream space object sampled analytically where OpenMC exposes no sampler)."
    )

    # O3: ring geometry moments vs the upstream CylindricalIndependent space
    # (geometry is fuel-independent; positions compared in cm).
    (upstream_dt,) = fusion_ring_source(
        radius=RADIUS_CM / 100.0,
        z_placement=HEIGHT_CM / 100.0,
        temperature=ti_ev,
        fuel={"D": 0.5, "T": 0.5},
    )
    space = upstream_dt.space
    r_ref = float(np.asarray(space.r.x)[0]) * 100.0  # m -> cm
    z_ref = float(np.asarray(space.z.x)[0]) * 100.0
    phi_a, phi_b = float(space.phi.a), float(space.phi.b)
    phi = rng.uniform(phi_a, phi_b, N)
    up_x = r_ref * np.cos(phi)
    up_y = r_ref * np.sin(phi)
    err_r = max(
        rel_diff(
            float(np.mean(np.hypot(ours["x"], ours["y"]))), float(np.mean(np.hypot(up_x, up_y)))
        ),
        rel_diff(r_ref, RADIUS_CM),
    )
    rows.append(
        ["O3 ring radius moments", fmt(err_r), "< 1e-12", _check(err_r < 1e-12, "O3 radius")]
    )
    rows.append(
        [
            "O3 ring height moments",
            fmt(rel_diff(float(np.mean(ours["z"])), z_ref)),
            "< 1e-12",
            _check(rel_diff(float(np.mean(ours["z"])), z_ref) < 1e-12, "O3 height"),
        ]
    )
    # Second moment <x^2> = R^2/2 for a uniform azimuth; comparing the two
    # independent samplings catches a biased/shrunk azimuth distribution
    # (a first-moment comparison of two noisy ~0 means is meaningless).
    err_phi = rel_diff(float(np.mean(ours["x"] ** 2)), float(np.mean(up_x**2)))
    rows.append(["O3 azimuth <x^2>", fmt(err_phi), "< 1e-2", _check(err_phi < 1e-2, "O3 azimuth")])
    notes.append(
        "O3 geometry cross-check uses the upstream ring's own CylindricalIndependent "
        "parameters (r delta, z delta, uniform azimuth) against Nucleide's sampled ring."
    )

    # O3b: point source geometry.
    (up_point,) = fusion_point_source(
        coordinate=(1.0, -2.0, 0.5), temperature=ti_ev, fuel={"D": 1.0}
    )
    px, py, pz = (float(v) * 100.0 for v in up_point.space.xyz)
    ours_p = ps.particles(
        {
            "kind": "point",
            "position": [px, py, pz],
            "reaction": "dd",
            "ion_temperature_kev": TI_KEV,
        },
        128,
        seed=5,
    )
    ok = bool(np.all(ours_p["x"] == px)) and bool(np.all(ours_p["z"] == pz))
    rows.append(["O3 point position", "-", "exact", _check(ok, "O3 point")])
    return rows, notes, False


def main() -> int:
    report = Report("plasma_source", "Tokamak fusion sources (`plasma_source_vs_openmc.py`)")
    report.prose(
        "Two-part oracle for `nucleide.plasma_source`: analytic gates P1-P4 on "
        "the synthetic ring spec (closed-form ring/spectrum moments, sampler "
        "determinism, card emission round trip; always run), and a "
        "cross-check against the upstream openmc-plasma-source package "
        "(MIT; Ballabio-fit helpers + sampled ring/point sources; oracle "
        "checks O1-O3, container only)."
    )
    rows1, notes1 = analytic_gates()
    for note in notes1:
        report.prose(note)
    report.table(["Gate", "Value", "Tol", "Status"], rows1)
    rows2, notes2, skipped = oracle_check_openmc_plasma_source()
    for note in notes2:
        report.prose(note)
    if skipped:
        skip_rows = [
            [f"{gate}", "SKIP (openmc-plasma-source unavailable)"] for gate in ("O1", "O2", "O3")
        ]
        report.table(["Gate", "Status"], skip_rows)
    else:
        report.table(["Gate", "Rel err", "Tol", "Status"], rows2)
    report.emit()

    if FAILURES:
        print(f"FAIL: {FAILURES} plasma-source check(s) failed", file=sys.stderr)
        return 1
    print("All plasma-source checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
