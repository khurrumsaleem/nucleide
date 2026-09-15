"""Neutron spectrum unfolding cross-check (`nucleide.unfold` vs analytic gates).

Two parts, both always run (no external unfolding oracle exists — PyNE/OpenMC
ship no SAND-II counterpart — so no container dependency is added and nothing
here can SKIP):

1. Analytic gates U1-U4: forward-fold against a hand matvec, exact-guess
   fixed point, determined-system recovery from a biased guess at pinned
   tolerances, and underdetermined rate reproduction + error reduction.
2. IRDFF-II analytical benchmark-shape probes U5: the analytical fields
   named in Trkov et al., Nucl. Data Sheets 163 (2020) 1 (open access,
   arXiv:1909.03336) — a 293.6 K thermal Maxwellian, a pure 1/E field, and a
   25 keV Maxwellian — reused as published facts with citation. The IRDFF-II
   tabulated group spectra are IAEA-copyright data files and are deliberately
   NOT used; provenance stays caller-supplied.

All response matrices are hand-built synthetics (Gaussian bumps over
log-spaced groups), never evaluated data.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable

from common import Report, fmt

import nucleide.unfold as unfold

FAILURES = 0


def _check(ok: bool, label: str) -> str:
    global FAILURES
    if not ok:
        FAILURES += 1
        print(f"FAIL: {label}", file=sys.stderr)
    return "PASS" if ok else "FAIL"


def _synthetic_response(n_det: int, n_groups: int, width: float) -> list[list[float]]:
    """Hand-built synthetic response: Gaussian bumps + a small uniform tail."""
    rows = []
    for i in range(n_det):
        center = i * (n_groups - 1) / max(n_det - 1, 1)
        rows.append([math.exp(-(((j - center) / width) ** 2)) + 1e-3 for j in range(n_groups)])
    return rows


def _log_midpoints(n_groups: int, lo: float, hi: float) -> list[float]:
    step = math.log(hi / lo) / n_groups
    return [math.exp(math.log(lo) + (j + 0.5) * step) for j in range(n_groups)]


def _maxwellian_plus_inv_e(midpoints: list[float], kt: float) -> list[float]:
    return [e * math.exp(-e / kt) + 1e-6 / e for e in midpoints]


def _max_rel_err(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / y for x, y in zip(a, b, strict=True))


def analytic_gates() -> tuple[list[list[str]], list[str]]:
    """U1-U4 on synthetic forward folds; returns (gate rows, prose notes)."""
    rows: list[list[str]] = []
    notes: list[str] = []

    response = _synthetic_response(3, 4, 1.5)
    spectrum = [1.0, 2.0, 3.0, 4.0]
    rates = unfold.forward_fold(response, spectrum)
    worst = max(
        abs(r - sum(x * p for x, p in zip(row, spectrum, strict=True)))
        for r, row in zip(rates, response, strict=True)
    )
    rows.append(["U1 forward fold", fmt(worst), "< 1e-12", _check(worst < 1e-12, "U1 fold")])

    response = _synthetic_response(4, 12, 2.0)
    midpoints = _log_midpoints(12, 1e-6, 10.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    out = unfold.sandii(response, rates, truth)
    ok = (
        out["iterations"] == 1
        and out["max_rel_change"] == 0.0
        and out["spectrum"] == truth
        and max(abs(f - 1.0) for f in out["rate_factors"]) < 1e-12
    )
    rows.append(["U2 exact fixed point", "-", "1 no-op adjust", _check(ok, "U2 fixed point")])
    notes.append("U2: an exact guess is a fixed point (one confirming no-op adjustment).")

    response = _synthetic_response(6, 6, 0.9)
    midpoints = _log_midpoints(6, 1e-6, 10.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    guess = [3.0 * p for p in truth]
    out = unfold.sandii(response, rates, guess, tolerance=1e-12, max_iterations=10_000)
    worst = _max_rel_err(out["spectrum"], truth)
    rate_err = max(abs(f - 1.0) for f in out["rate_factors"])
    ok = worst < 1e-6 and rate_err < 1e-9
    rows.append(
        ["U3 determined recovery", fmt(worst), "< 1e-6", _check(worst < 1e-6, "U3 recovery")]
    )
    rows.append(
        [
            "U3 rate factors",
            fmt(rate_err),
            "< 1e-9",
            _check(rate_err < 1e-9, "U3 rate factors"),
        ]
    )
    notes.append(
        f"U3 determined 6x6 recovery from a 3x-biased guess: {out['iterations']} adjustments."
    )

    n_groups = 24
    response = _synthetic_response(6, n_groups, 3.0)
    midpoints = _log_midpoints(n_groups, 1e-6, 12.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    # Non-uniform bias: part of it lies in the response null space and must
    # keep the guess; the recoverable part must still converge.
    guess = [p * (1.3 + 0.7 * (j % 5) / 4.0) for j, p in enumerate(truth)]
    out = unfold.sandii(response, rates, guess, tolerance=1e-9, max_iterations=50_000)
    rate_err = max(abs(f - 1.0) for f in out["rate_factors"])
    recovered_err = _max_rel_err(out["spectrum"], truth)
    guess_err = _max_rel_err(guess, truth)
    rows.append(
        [
            "U4 underdetermined rate repro",
            fmt(rate_err),
            "< 1e-6",
            _check(rate_err < 1e-6, "U4 rate repro"),
        ]
    )
    rows.append(
        [
            "U4 spectrum error reduction",
            fmt(recovered_err),
            "< guess err / 2",
            _check(recovered_err < guess_err / 2.0, "U4 error reduction"),
        ]
    )
    notes.append(
        f"U4 underdetermined case (6 detectors, {n_groups} groups): rates reproduced; "
        f"spectrum error {recovered_err:.3e} vs guess error {guess_err:.3e} — the "
        "recoverable bias component converges, the null-space part keeps the guess."
    )
    return rows, notes


def irdff_ii_probes() -> tuple[list[list[str]], list[str]]:
    """U5 IRDFF-II analytical benchmark-shape recovery (published facts)."""
    rows: list[list[str]] = []
    notes: list[str] = []
    n_groups = 6
    midpoints = _log_midpoints(n_groups, 1e-9, 20.0)
    thermal_kt = 8.617333262e-5 * 293.6  # 293.6 K in MeV
    shapes = {
        "thermal Maxwellian 293.6 K": [e * math.exp(-e / thermal_kt) + 1e-30 for e in midpoints],
        "pure 1/E": [1.0 / e + 1e-30 for e in midpoints],
        "Maxwellian 25 keV": [math.sqrt(e) * math.exp(-e / 2.5e-2) + 1e-30 for e in midpoints],
    }
    for name, shape in shapes.items():
        response = _synthetic_response(n_groups, n_groups, 0.8)
        rates = unfold.forward_fold(response, shape)
        guess = [5.0 * p for p in shape]
        out = unfold.sandii(response, rates, guess, tolerance=1e-11, max_iterations=100_000)
        worst = _max_rel_err(out["spectrum"], shape)
        rows.append([f"U5 {name}", fmt(worst), "< 1e-5", _check(worst < 1e-5, f"U5 {name}")])
    notes.append(
        "U5 shapes are the IRDFF-II analytical benchmark fields named in "
        "Trkov et al., Nucl. Data Sheets 163 (2020) 1 (arXiv:1909.03336), "
        "re-derived from their closed forms; the IAEA-copyright tabulated "
        "spectra are never used."
    )
    return rows, notes


def contract_gates() -> tuple[list[list[str]], list[str]]:
    """U6-U8 convergence and validation contract probes."""
    rows: list[list[str]] = []
    notes: list[str] = []

    response = _synthetic_response(4, 8, 1.5)
    midpoints = _log_midpoints(8, 1e-6, 10.0)
    truth = _maxwellian_plus_inv_e(midpoints, 2.53e-5)
    rates = unfold.forward_fold(response, truth)
    guess = [2.0 * p for p in truth]
    try:
        unfold.sandii(response, rates, guess, tolerance=1e-12, max_iterations=1)
        ok = False
    except ValueError:
        ok = True
    rows.append(["U6 cap is a hard fail", "-", "ValueError", _check(ok, "U6 hard fail")])

    out = unfold.sandii([[1.0, 0.0, 0.0]], [4.0], [2.0, 7.0, 3.0], tolerance=1e-12)
    ok = out["spectrum"][0] == 4.0 and out["spectrum"][1:] == [7.0, 3.0]
    rows.append(["U7 unconstrained keeps guess", "-", "bit-exact", _check(ok, "U7 unconstrained")])

    failures: list[str] = []

    def _expect(label: str, fn: Callable[[], object], needle: str) -> None:
        try:
            fn()
            failures.append(label)
        except ValueError as exc:
            if needle not in str(exc):
                failures.append(f"{label} (wrong message: {exc})")

    _expect("rates length", lambda: unfold.sandii(response, [1.0], guess), "shape mismatch")
    _expect(
        "negative response",
        lambda: unfold.sandii([[1.0, -1.0], [1.0, 1.0]], [1.0, 1.0], [1.0, 1.0]),
        "negative entry",
    )
    _expect(
        "zero guess",
        lambda: unfold.sandii([[1.0, 1.0]], [1.0], [1.0, 0.0]),
        "strictly positive",
    )
    _expect(
        "unreachable rate",
        lambda: unfold.sandii([[0.0, 0.0]], [1.0], [1.0, 1.0]),
        "unreachable",
    )
    _expect(
        "fold shape",
        lambda: unfold.forward_fold([[1.0, 1.0]], [1.0]),
        "shape mismatch",
    )
    rows.append(["U8 named input errors", "-", "5 cases", _check(not failures, "U8 errors")])
    if failures:
        notes.append(f"U8 missing/weak rejections: {', '.join(failures)}.")
    notes.append(
        "U6-U8: non-convergence raises (never a partial spectrum), unconstrained "
        "groups keep the guess bit-exactly, and every malformed input names its cause."
    )
    return rows, notes


def main() -> int:
    report = Report("unfold", "Neutron spectrum unfolding (`unfold_vs_analytic.py`)")
    report.prose(
        "Two-part oracle for `nucleide.unfold`, both parts always run: analytic "
        "gates U1-U4 on synthetic forward folds (fold identity, exact fixed "
        "point, determined and underdetermined round-trips at pinned "
        "tolerances) and IRDFF-II analytical benchmark-shape probes U5 "
        "(Trkov et al. 2020, published facts with citation). No external "
        "unfolding oracle exists, so no container dependency is added and no "
        "check can SKIP."
    )
    rows1, notes1 = analytic_gates()
    for note in notes1:
        report.prose(note)
    report.table(["Gate", "Value", "Tol", "Status"], rows1)
    rows2, notes2 = irdff_ii_probes()
    for note in notes2:
        report.prose(note)
    report.table(["Gate", "Rel err", "Tol", "Status"], rows2)
    rows3, notes3 = contract_gates()
    for note in notes3:
        report.prose(note)
    report.table(["Gate", "Value", "Tol", "Status"], rows3)
    report.emit()

    if FAILURES:
        print(f"FAIL: {FAILURES} unfold check(s) failed", file=sys.stderr)
        return 1
    print("All unfold checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
