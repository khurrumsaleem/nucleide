"""Library assessment oracle for the vendored fission-yield pack (JADE-class).

Compares nothing against external codes: the ENDF/B-VIII.0
independent/cumulative sublibraries committed in
`crates/nuclei/src/data/fission_yields.tsv` are checked for internal
consistency and completeness, in the style of a nuclear-data V&V
assessment stage (cross-sublibrary consistency plus coverage statistics
over fixed synthetic thresholds). No tapes are read, no NJOY runs, no
second library is vendored.

Gates (always run, deterministic over the committed TSV):

- L1 data hygiene (exact): every row finite, `yield >= 0`, `dY >= 0`.
- L2 independent sums: each (parent, origin, energy) independent block
  sums to 2.0 (two fragments per fission) within 1e-4.
- L3 coverage (exact): every independent daughter has a cumulative row
  for the same (parent, origin, energy).
- L4 diagnostics (reported, never failing): zeroed cumulative entries
  (evaluation shape: untracked isomers) and worst deviations per block.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from common import Report, fmt

REPO_ROOT = Path(__file__).resolve().parent.parent
TSV = REPO_ROOT / "crates" / "nuclei" / "src" / "data" / "fission_yields.tsv"

FAILURES = 0

SUM_TOL = 1e-4


def _check(ok: bool, label: str) -> str:
    global FAILURES
    if not ok:
        FAILURES += 1
        print(f"FAIL: {label}", file=sys.stderr)
    return "PASS" if ok else "FAIL"


def _load() -> tuple[
    dict[tuple[str, str, str, str], float],
    dict[tuple[str, str, str, str], float],
    int,
    int,
]:
    """Load (independent, cumulative, rows, hygiene violations)."""
    ind: dict[tuple[str, str, str, str], float] = {}
    cum: dict[tuple[str, str, str, str], float] = defaultdict(float)
    rows = 0
    bad = 0
    with open(TSV, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["# parent_GNDS"].startswith("#"):
                continue
            rows += 1
            try:
                v = float(row["yield"])
                dy = float(row["dY"])
            except ValueError:
                bad += 1
                continue
            import math

            if not (math.isfinite(v) and math.isfinite(dy)) or v < 0.0 or dy < 0.0:
                bad += 1
                continue
            key = (row["# parent_GNDS"], row["origin"], row["energy_eV"], row["daughter_GNDS"])
            if row["kind"].strip().lower().startswith("indep"):
                ind[key] = v
            else:
                cum[key] += v
    return ind, cum, rows, bad


def main() -> int:
    report = Report("library", "Fission-yield library assessment (`library_vs_refs.py`)")
    report.prose(
        "JADE-class library assessment over the committed ENDF/B-VIII.0"
        " fission-yield pack (no external oracle, no tapes): data hygiene,"
        " independent-block sums to 2.0, and cumulative coverage, plus"
        " reported diagnostics on evaluation-shape quirks."
    )
    ind, cum, rows, bad = _load()

    gate_rows: list[list[str]] = []
    gate_rows.append(
        ["L1 rows finite, yields/dY >= 0", str(rows), str(bad), _check(bad == 0, "L1")]
    )

    blocks: dict[tuple[str, str, str], float] = defaultdict(float)
    for (parent, origin, energy, _), v in ind.items():
        blocks[(parent, origin, energy)] += v
    worst_key = max(blocks, key=lambda k: abs(blocks[k] - 2.0))
    worst_dev = abs(blocks[worst_key] - 2.0)
    gate_rows.append(
        [
            "L2 independent sums to 2.0",
            f"{len(blocks)} blocks",
            f"worst dev {fmt(worst_dev)} at {worst_key[0]}/{worst_key[2]}",
            _check(worst_dev < SUM_TOL, "L2"),
        ]
    )

    missing = sum(1 for k in ind if k not in cum)
    gate_rows.append(
        [
            "L3 cumulative row per independent daughter",
            str(len(ind)),
            f"{missing} missing",
            _check(missing == 0, "L3"),
        ]
    )
    report.table(["Gate", "Expected", "Got", "Status"], gate_rows)

    zeroed = sum(1 for k in ind if cum.get(k, 0.0) == 0.0 and ind[k] > 0.0)
    worst_ind = sorted(((abs(total - 2.0), key) for key, total in blocks.items()), reverse=True)[:5]
    diag_rows = [
        ["independent blocks", str(len(blocks)), "—"],
        ["independent daughters", str(len(ind)), "—"],
        ["zeroed cumulative entries (untracked isomers)", str(zeroed), "reported"],
    ]
    for dev, key in worst_ind:
        diag_rows.append([f"dev {key[0]} {key[1]} {key[2]}", fmt(dev), "reported"])
    report.table(["Diagnostic", "Value", "Status"], diag_rows)
    report.prose(
        f"Zeroed cumulative entries ({zeroed}) are evaluation shape — short-lived"
        " isomers the cumulative evaluation does not track — recorded here so a"
        " library regeneration that changes the count fails loudly at review,"
        " not silently."
    )
    report.emit()
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
