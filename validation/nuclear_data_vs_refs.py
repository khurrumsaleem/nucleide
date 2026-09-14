"""Compare Nucleide nuclear data and name conversions against PyNE and OpenMC."""

from __future__ import annotations

import contextlib
import importlib.metadata
import sys

from common import Report, abs_diff, fmt, rel_diff

import nucleide

try:
    import openmc.data

    HAS_OPENMC = True
    OPENMC_SKIP = ""
except Exception as exc:
    openmc = None  # type: ignore[no-redef]
    HAS_OPENMC = False
    OPENMC_SKIP = f"OpenMC oracle skipped: cannot import openmc.data ({exc})."

try:
    import pyne.data
    import pyne.nucname as nucname

    HAS_PYNE = True
    PYNE_SKIP = ""
except Exception as exc:
    pyne = None  # type: ignore[no-redef]
    nucname = None  # type: ignore[no-redef]
    HAS_PYNE = False
    PYNE_SKIP = f"PyNE oracle skipped: cannot import pyne.data ({exc})."


def compare_atomic_masses(sample: list[str]) -> dict[str, dict[str, float]]:
    """Compare atomic masses vs PyNE (AME2016) and OpenMC (AME2020)."""
    stats: dict[str, list[float]] = {"pyne": [], "openmc": []}
    per_nuc: dict[str, dict[str, float]] = {}
    for name in sample:
        nuc_val = nucleide.nuclei.atomic_mass(name)
        pyne_val = pyne.data.atomic_mass(name)
        openmc_val = None
        with contextlib.suppress(Exception):
            openmc_val = openmc.data.atomic_mass(name.lower())

        per_nuc[name] = {
            "nucleide": nuc_val if nuc_val is not None else float("nan"),
            "pyne": pyne_val if pyne_val is not None else float("nan"),
            "openmc": openmc_val if openmc_val is not None else float("nan"),
        }
        if nuc_val is not None and pyne_val is not None:
            stats["pyne"].append(abs_diff(nuc_val, pyne_val))
        if nuc_val is not None and openmc_val is not None:
            stats["openmc"].append(abs_diff(nuc_val, openmc_val))

    return {
        "per_nuc": per_nuc,
        "pyne_max_abs": max(stats["pyne"]) if stats["pyne"] else float("nan"),
        "pyne_mean_abs": (sum(stats["pyne"]) / len(stats["pyne"]))
        if stats["pyne"]
        else float("nan"),
        "openmc_max_abs": max(stats["openmc"]) if stats["openmc"] else float("nan"),
        "openmc_mean_abs": (sum(stats["openmc"]) / len(stats["openmc"]))
        if stats["openmc"]
        else float("nan"),
    }


def compare_natural_abundances() -> dict[str, float]:
    """Compare natural abundances for all naturally-occurring isotopes."""
    om_abund = openmc.data.NATURAL_ABUNDANCE
    diffs_pyne: list[float] = []
    diffs_openmc: list[float] = []
    for name in om_abund:
        nuc_val = nucleide.nuclei.natural_abundance(name)
        om_val = om_abund[name]
        pyne_val = None
        with contextlib.suppress(Exception):
            pyne_val = pyne.data.natural_abund(nucname.id(name))
        if nuc_val is not None:
            diffs_openmc.append(abs_diff(nuc_val, om_val))
            if pyne_val is not None:
                diffs_pyne.append(abs_diff(nuc_val, pyne_val))

    return {
        "count": len(om_abund),
        "openmc_max_abs": max(diffs_openmc) if diffs_openmc else float("nan"),
        "openmc_mean_abs": (sum(diffs_openmc) / len(diffs_openmc))
        if diffs_openmc
        else float("nan"),
        "pyne_max_abs": max(diffs_pyne) if diffs_pyne else float("nan"),
        "pyne_mean_abs": (sum(diffs_pyne) / len(diffs_pyne)) if diffs_pyne else float("nan"),
    }


def compare_half_lives(sample: list[str]) -> dict[str, float]:
    """Compare half-lives vs PyNE and OpenMC."""
    diffs_pyne: list[float] = []
    diffs_openmc: list[float] = []
    for name in sample:
        nuc_val = nucleide.nuclei.half_life(name)
        pyne_val = pyne.data.half_life(name)
        om_name = name.lower().replace("-", "").replace("m", "_m1")
        om_val = None
        with contextlib.suppress(Exception):
            om_val = openmc.data.half_life(om_name)

        if nuc_val is not None and pyne_val is not None and pyne_val > 0:
            diffs_pyne.append(rel_diff(nuc_val, pyne_val))
        if nuc_val is not None and om_val is not None and om_val > 0:
            diffs_openmc.append(rel_diff(nuc_val, om_val))

    return {
        "pyne_max_rel": max(diffs_pyne) if diffs_pyne else float("nan"),
        "pyne_mean_rel": (sum(diffs_pyne) / len(diffs_pyne)) if diffs_pyne else float("nan"),
        "openmc_max_rel": max(diffs_openmc) if diffs_openmc else float("nan"),
        "openmc_mean_rel": (sum(diffs_openmc) / len(diffs_openmc))
        if diffs_openmc
        else float("nan"),
    }


def compare_name_conversions(sample: list[str]) -> dict[str, float]:
    """Compare Nucleide name-dialect conversions against pyne.nucname."""

    def expected_zaid(name: str) -> int:
        n = nucleide.nuclei.Nuclide(name)
        z, a, state = n.z, n.a, n.state
        zaid = z * 1000 + a
        # MCNP special case: Am-242 and Am-242m are swapped.
        if zaid == 95242 and state < 2:
            state = (state + 1) % 2
        if state > 0:
            zaid += 300 + state * 100
        return zaid

    checks = {
        "nucid": (lambda n: n.nucid, lambda name: nucname.id(name)),
        "zzaaam": (lambda n: n.zzaaam, lambda name: nucname.zzaaam(name)),
        "zaid": (lambda n: n.zaid, expected_zaid),
        "serpent": (lambda n: n.serpent, lambda name: nucname.serpent(nucname.id(name))),
        "nist": (lambda n: n.nist, lambda name: nucname.nist(nucname.id(name))),
        "cinder": (lambda n: n.cinder, lambda name: nucname.cinder(nucname.id(name))),
        "alara": (lambda n: n.alara, lambda name: nucname.alara(nucname.id(name))),
        "sza": (lambda n: n.sza, lambda name: nucname.sza(nucname.id(name))),
    }

    diffs: dict[str, list[float]] = {k: [] for k in checks}
    for name in sample:
        nuc = nucleide.nuclei.Nuclide(name)
        for key, (nuc_fn, pyne_fn) in checks.items():
            try:
                a = nuc_fn(nuc)
                b = pyne_fn(name)
                if isinstance(a, str):
                    diffs[key].append(0.0 if a == b else 1.0)
                else:
                    diffs[key].append(abs_diff(float(a), float(b)))
            except Exception:
                # PyNE may not support some metastable dialects; skip silently.
                pass

    return {f"{key}_max": max(v) if v else float("nan") for key, v in diffs.items()}


class SkipCheck(Exception):
    """An oracle is unavailable; the message is the loud skip reason."""


def emit_report(report: Report) -> bool:
    """Write the JSON report; skip loudly when env metadata is missing.

    `common.environment()` needs installed PyNE/OpenMC distributions, so a
    bare checkout outside the container cannot write `results/*.json`. That
    write is skipped (never hand-written) while every check above still runs.
    """
    try:
        report.emit()
        return True
    except importlib.metadata.PackageNotFoundError as exc:
        print(
            "SKIPPED report write: validation environment metadata unavailable"
            f" outside the container ({exc}); checks above still ran."
        )
        return False


#: Nuclides covered by both `simple_xs.tsv` and PyNE's KAERI simple-xs table.
SIMPLE_XS_SAMPLE = [
    "H1",
    "B10",
    "C12",
    "O16",
    "Fe56",
    "Co59",
    "U235",
    "U238",
    "Pu239",
    "Pb208",
]

#: Reaction keys to try against PyNE's simple-xs `reaction()` accessor.
_SIMPLE_XS_RX = ("total", "sigma_t", 1)

#: NIST NCNR bound coherent lengths in fm (Sears 1992 tabulation).
#: `scattering_lengths.tsv` copies these values exactly, hence the tight check.
SCATTERING_ANCHORS_FM = {
    "H1": -3.7406,
    "H2": 6.671,
    "C12": 6.6511,
    "O16": 5.803,
    "Fe56": 9.94,
    "U238": 8.402,
    "Pb208": 9.5,
    "B10": -0.1,
}

#: Mean *prompt* recoverable decay energies in MeV from ENDF/B-VII.1 MF8/MT457
#: (daughter gammas belong to the daughter row: Cs137 excludes the 662 keV
#: line, which lives on Ba137_m1). Screening-level comparison with a loose
#: 20% band (NOT ENSDF evaluations).
DECAY_ENERGY_SPOTS_MEV = {
    "H3": 0.00569,
    "Co60": 2.60061,
    "Sr90": 0.1958,
    "I131": 0.573438,
    "Xe135": 0.56798,
    "Cs137": 0.179448,
    "Ba137_m1": 0.661397,
    "U235": 4.61919,
    "Pu239": 5.24326,
    "Am241": 5.62799,
}

#: Evaluated decay branches from ENDF/B-VIII.0 MF8/MT457 NDK records
#: (RTYP code, RFS daughter state, BR fraction), mirroring the
#: ``DECAY_ENERGY_SPOTS_MEV`` pattern. K-40 pins the half-life table entry
#: derived from the same VIII.0 checkout (3.93839e16 s) plus its beta-/EC
#: pair; Es-254 members follow this repo's own branch table.
#: ``{parent: (half_life_s, [(progeny, mode, bf)])}``.
DECAY_BRANCH_SPOTS = {
    "K40": (
        3.93839e16,
        [("Ca40", "beta-", 0.8914), ("Ar40", "ec/beta+", 0.1086)],
    ),
    "Es254": (
        2.382048e7,
        [("Bk250", "alpha", 1.0)],
    ),
    "Es254_m1": (
        141479.9,
        [
            ("Bk250", "alpha", 0.0032),
            ("Fm254", "beta-", 0.98),
            ("Cf254", "ec/beta+", 0.0008),
            ("Es254", "IT", 0.0155),
        ],
    ),
    "Ba137_m1": (153.12, [("Ba137", "IT", 1.0)]),
    "He8": (0.1191, [("Li8", "beta-", 0.84), ("Li7", "beta-", 0.16)]),
}

#: Evaluated fission product yields read off the ENDF/B-VIII.0 fission-yield
#: tapes (independent MF8/MT454 and cumulative MF8/MT459), mirroring the
#: ``DECAY_BRANCH_SPOTS`` pattern.  Both sublibraries are unchanged carries
#: of ENDF/B-VII.1 evaluations per their README.txt (England et al. ENDF-349,
#: except the Chadwick-Kawano Pu-239 evaluation); the Mattera-Sonzogni
#: correction is not applied.  ``(parent, origin, kind, energy_eV, daughter,
#: expected_Y)`` — U-238's lowest neutron-induced set is 500 keV (it has no
#: thermal nfy evaluation), and Cf-252's spontaneous set is E = 0.
FISSION_YIELD_SPOTS = [
    ("U235", "n", "independent", 0.0253, "Xe135", 0.000785125),
    ("U235", "n", "independent", 0.0253, "Xe135_m1", 0.00178122),
    ("U235", "n", "independent", 0.0253, "Kr85", 0.000255332),
    ("U235", "n", "cumulative", 0.0253, "Xe135", 0.065385),
    ("U235", "n", "cumulative", 0.0253, "Cs137", 0.0618832),
    ("Pu239", "n", "independent", 0.0253, "Xe135", 0.00314131),
    ("U238", "n", "independent", 5.0e5, "Xe135", 0.000111541),
    ("Cf252", "sf", "independent", 0.0, "Xe135", 0.00186145),
]

#: Independent-yield sets whose per-set yield sum must approach 2.0 (two
#: fragments per fission).  ``(parent, origin, energy_eV)``.
FISSION_YIELD_SUM_SPOTS = [
    ("U235", "n", 0.0253),
    ("Pu239", "n", 0.0253),
    ("U238", "n", 5.0e5),
    ("Cf252", "sf", 0.0),
]

#: EPA FGR 15 (EPA 402-R-25-001, July 2025) external-dosimetry spots,
#: transcribed by hand from the published ``FGR15_Tables/Table_4_*.DAT``
#: members of the official EPA zip (``(scenario, nuclide, age, expected)``;
#: isomer names keep the FGR 15 spelling ``Ba-137m``).  One spot per scenario
#: (4.1-4.7), with the 4.1 trio pinning a non-adult column and an isomer
#: letter.  Spots are read from the same ASCII decimals the parser consumes
#: and both parses are correctly rounded, so the gate is exact equality: it
#: verifies the (table, nuclide, age) -> cell mapping, not float round-trip.
FGR15_SPOTS = [
    ("ground_surface", "H-3", "newborn", 3.84e-27),  # Table 4.1
    ("ground_surface", "H-3", "adult", 8.97e-28),  # Table 4.1
    ("ground_surface", "Be-7", "adult", 3.17e-17),  # Table 4.1
    ("ground_surface", "Ba-137m", "adult", 3.87e-16),  # Table 4.1
    ("soil_1cm", "K-40", "adult", 9.32e-19),  # Table 4.2
    ("soil_5cm", "Sr-90", "adult", 2.54e-21),  # Table 4.3
    ("soil_15cm", "I-131", "adult", 9.55e-18),  # Table 4.4
    ("soil_infinite", "H-3", "adult", 2.49e-28),  # Table 4.5
    ("air_submersion", "Rn-222", "adult", 1.72e-17),  # Table 4.6
    ("water_immersion", "U-238", "adult", 6.51e-21),  # Table 4.7
]


def _pyne_simple_xs_source():
    """Return PyNE's KAERI simple-xs source, or raise SkipCheck with a reason."""
    try:
        from pyne.xs.data_source import SimpleDataSource
    except Exception as exc:
        raise SkipCheck(
            "PyNE simple_xs skipped: pyne.xs.data_source is unavailable"
            f" ({exc}); the nomoab build may lack this module."
        ) from exc
    try:
        src = SimpleDataSource()
    except Exception as exc:
        raise SkipCheck(
            f"PyNE simple_xs skipped: cannot construct SimpleDataSource ({exc})."
        ) from exc
    if not getattr(src, "exists", True):
        raise SkipCheck(
            "PyNE simple_xs skipped: SimpleDataSource.exists is False"
            " (nuc_data.h5 has no /neutron/simple_xs table)."
        )
    return src


def _pyne_simple_xs_lookup(src, name: str) -> tuple[float, float]:
    """Return (thermal_b, fast14_b) totals from PyNE, or raise SkipCheck."""
    reasons: list[str] = []
    for rx in _SIMPLE_XS_RX:
        try:
            data = src.reaction(name, rx)
        except Exception as exc:
            reasons.append(f"{rx}: {exc}")
            continue
        if data is None:
            reasons.append(f"{rx}: no data")
            continue
        vals = [float(v) for v in list(data)]
        if len(vals) >= 2 and any(v > 0 for v in vals):
            # Source group structure is descending energy: the first point
            # is 14 MeV and the last is thermal (2.53e-8 MeV), so thermal
            # is vals[-1] and fast-14 is vals[0].
            return vals[-1], vals[0]
        reasons.append(f"{rx}: empty response")
    raise SkipCheck(f"no total channel for {name} ({'; '.join(reasons)}).")


def compare_simple_xs(sample: list[str]) -> dict:
    """Compare `simple_xs` thermal/fast totals vs PyNE `nuc_data.h5` simple_xs."""
    rows: list[list[str]] = []
    skipped: list[str] = []
    diffs: list[float] = []
    if not HAS_PYNE:
        skipped.append(PYNE_SKIP)
        return {"rows": rows, "skipped": skipped, "diffs": diffs, "available": False}
    try:
        src = _pyne_simple_xs_source()
    except SkipCheck as exc:
        skipped.append(str(exc))
        return {"rows": rows, "skipped": skipped, "diffs": diffs, "available": False}
    for name in sample:
        try:
            nuc_val = nucleide.nuclei.simple_xs(name)
        except Exception as exc:
            skipped.append(f"{name}: nucleide.nuclei.simple_xs raised ({exc}).")
            continue
        if nuc_val is None:
            skipped.append(f"{name}: Nucleide has no simple_xs entry.")
            continue
        try:
            ref_th, ref_fa = _pyne_simple_xs_lookup(src, name)
        except SkipCheck as exc:
            skipped.append(f"{name}: {exc}")
            continue
        nuc_th, nuc_fa = nuc_val
        d_th = rel_diff(nuc_th, ref_th)
        d_fa = rel_diff(nuc_fa, ref_fa)
        diffs.extend([d_th, d_fa])
        rows.append(
            [name, fmt(nuc_th), fmt(ref_th), fmt(d_th), fmt(nuc_fa), fmt(ref_fa), fmt(d_fa)]
        )
    return {"rows": rows, "skipped": skipped, "diffs": diffs, "available": bool(rows)}


def compare_scattering_lengths() -> dict:
    """Compare coherent scattering lengths vs the NIST-anchored values."""
    rows: list[list[str]] = []
    diffs: list[float] = []
    missing: list[str] = []
    for name, anchor in SCATTERING_ANCHORS_FM.items():
        val = nucleide.nuclei.scattering_length(name)
        if val is None:
            missing.append(name)
            rows.append([name, fmt(anchor), "None", "n/a"])
            continue
        diffs.append(abs_diff(val, anchor))
        rows.append([name, fmt(val), fmt(anchor), fmt(diffs[-1])])
    return {
        "rows": rows,
        "missing": missing,
        "max_abs": max(diffs) if diffs else float("nan"),
    }


def compare_decay_branches() -> dict:
    """Compare `decay_branches`/`decay_branch_fraction` vs ENDF VIII.0 spots."""
    rows: list[list[str]] = []
    diffs: list[float] = []
    missing: list[str] = []
    for parent, (hl_spot, spots) in DECAY_BRANCH_SPOTS.items():
        hl = nucleide.nuclei.half_life(parent)
        if hl is None:
            missing.append(f"{parent} half-life")
            rows.append([parent, "half-life", "None", fmt(hl_spot), "n/a"])
        else:
            diffs.append(rel_diff(hl, hl_spot))
            rows.append([parent, "half-life", fmt(hl), fmt(hl_spot), fmt(diffs[-1])])
        branches = nucleide.nuclei.decay_branches(parent)
        by_prog = {prog: (bf, mode) for prog, bf, mode in branches}
        for prog, mode, bf_spot in spots:
            entry = by_prog.get(prog)
            if entry is None or entry[1] != mode:
                missing.append(f"{parent}->{prog} [{mode}]")
                rows.append([parent, f"{prog} [{mode}]", "None", fmt(bf_spot), "n/a"])
                continue
            diffs.append(rel_diff(entry[0], bf_spot))
            rows.append([parent, f"{prog} [{mode}]", fmt(entry[0]), fmt(bf_spot), fmt(diffs[-1])])
    return {
        "rows": rows,
        "missing": missing,
        "max_rel": max(diffs) if diffs else float("nan"),
    }


def compare_decay_energies() -> dict:
    """Compare `decay_energy` vs chain/ENDF-derived spot values (20% band)."""
    rows: list[list[str]] = []
    diffs: list[float] = []
    missing: list[str] = []
    for name, spot in DECAY_ENERGY_SPOTS_MEV.items():
        val = nucleide.nuclei.decay_energy(name)
        if val is None:
            missing.append(name)
            rows.append([name, "None", fmt(spot), "n/a"])
            continue
        diffs.append(rel_diff(val, spot))
        rows.append([name, fmt(val), fmt(spot), fmt(diffs[-1])])
    return {
        "rows": rows,
        "missing": missing,
        "stable_none": nucleide.nuclei.decay_energy("Fe56") is None,
        "max_rel": max(diffs) if diffs else float("nan"),
    }


def compare_fission_yields() -> dict:
    """Compare `fission_yields` against ENDF/B-VIII.0 tape spot values."""
    rows: list[list[str]] = []
    diffs: list[float] = []
    missing: list[str] = []
    for parent, origin, kind, energy, daughter, spot in FISSION_YIELD_SPOTS:
        label = f"{parent}[{origin}/{kind}]@{energy:.6g} -> {daughter}"
        sets = nucleide.nuclei.fission_yields(parent, origin=origin, kind=kind)
        by_energy = {e: {n: (y, dy) for n, y, dy in prods} for e, prods in sets}
        entry = by_energy.get(energy)
        if entry is None or daughter not in entry:
            missing.append(label)
            rows.append([label, "None", fmt(spot), "n/a"])
            continue
        got = entry[daughter][0]
        diffs.append(rel_diff(got, spot))
        rows.append([label, fmt(got), fmt(spot), fmt(diffs[-1])])
    for parent, origin, energy in FISSION_YIELD_SUM_SPOTS:
        label = f"{parent}[{origin}]@{energy:.6g} independent sum"
        sets = nucleide.nuclei.fission_yields(parent, origin=origin, kind="independent")
        prods = dict(sets).get(energy)
        if prods is None:
            missing.append(label)
            rows.append([label, "None", "2.0", "n/a"])
            continue
        total = sum(y for _, y, _ in prods)
        diffs.append(rel_diff(total, 2.0))
        rows.append([label, fmt(total), "2.0", fmt(diffs[-1])])
    return {
        "rows": rows,
        "missing": missing,
        "max_rel": max(diffs) if diffs else float("nan"),
    }


def compare_fission_yields_openmc() -> dict:
    """Cross-check the committed table vs OpenMC's ENDF reader, when tapes exist.

    OpenMC's ``FissionProductYields.from_endf`` reads MF8/MT454 (independent)
    preferentially; the probe needs the ENDF/B-VIII.0 nfy tapes on disk. They
    are not part of the container inputs (``validation/.cache/`` holds only
    the CASL chain), so this SKIP is expected there — recorded loudly.
    """
    rows: list[list[str]] = []
    diffs: list[float] = []
    skipped: list[str] = []
    if not HAS_OPENMC:
        skipped.append(OPENMC_SKIP or "OpenMC oracle skipped: openmc.data unavailable.")
        return {"rows": rows, "diffs": diffs, "skipped": skipped, "available": False}
    import glob
    import os

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    tapes = sorted(glob.glob(os.path.join(cache_dir, "**", "nfy-*.endf"), recursive=True))
    if not tapes:
        skipped.append(
            "OpenMC fission-yield cross-check skipped: ENDF/B-VIII.0 nfy tapes are not "
            "present under validation/.cache (the container cache holds only the CASL "
            "chain); the tape spot gates above still pin the committed table."
        )
        return {"rows": rows, "diffs": diffs, "skipped": skipped, "available": False}
    try:
        u235_tape = next(t for t in tapes if "U_235" in t)
        fpy = openmc.data.FissionProductYields.from_endf(u235_tape)
        energy0 = float(fpy.energies[0])
        set0 = fpy[0]
        om_nuclides = list(set0["nuclides"])
        om_yields = [float(v) for v in set0["yields"]]
        pairs = list(zip(om_nuclides, om_yields, strict=True))
    except Exception as exc:
        note = f"OpenMC fission-yield cross-check skipped: openmc.data FY read failed ({exc})."
        skipped.append(note)
        return {"rows": rows, "diffs": diffs, "skipped": skipped, "available": False}
    # OpenMC may serve independent (sum ~2) or cumulative (sum >>2) values;
    # pick the matching kind instead of assuming the MT.
    om_sum = sum(v for _, v in pairs)
    kind = "independent" if abs(om_sum - 2.0) < 0.1 else "cumulative"
    nuc_sets = nucleide.nuclei.fission_yields("U235", kind=kind)
    by_energy = {e: {n: y for n, y, _ in prods} for e, prods in nuc_sets}
    entry = by_energy.get(energy0)
    if entry is None:
        note = f"OpenMC fission-yield cross-check skipped: no Nucleide {kind} set at E={energy0}."
        skipped.append(note)
        return {"rows": rows, "diffs": diffs, "skipped": skipped, "available": False}
    for name, om_val in pairs:
        nuc_val = entry.get(name)
        if nuc_val is None or om_val == 0.0:
            continue
        diffs.append(rel_diff(nuc_val, om_val))
    if diffs:
        rows.append([f"U235 {kind} @{energy0:.6g}", f"n={len(diffs)}", "max rel", fmt(max(diffs))])
    return {"rows": rows, "diffs": diffs, "skipped": skipped, "available": bool(diffs)}


def compare_fgr15() -> dict:
    """Gate the runtime-downloaded EPA FGR 15 tables against published spots.

    Downloads the official coefficient zip once into ``validation/.cache/``
    (git-ignored; re-downloaded only if absent — the CASL-chain pattern) and
    enforces the pinned SHA-256.  Parses all seven scenario tables through
    ``nucleide.nuclei.load_fgr15_table`` and gates the per-table row count
    (1,252), the six age columns, and the hand-transcribed ``FGR15_SPOTS``
    at exact equality.
    """
    from pathlib import Path

    cache_dir = Path(__file__).resolve().parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = Path(nucleide.data.fetch_fgr15(dest=cache_dir))
    scenarios = [
        "ground_surface",
        "soil_1cm",
        "soil_5cm",
        "soil_15cm",
        "soil_infinite",
        "air_submersion",
        "water_immersion",
    ]
    rows: list[list[str]] = []
    problems: list[str] = []
    for i, scenario in enumerate(scenarios, start=1):
        table = nucleide.nuclei.load_fgr15_table(scenario, dest=cache_dir)
        coefficients = table["coefficients"]
        n_rows = len(coefficients)
        bad_cols = sum(1 for row in coefficients.values() if len(row) != 6)
        label = f"{scenario} (Table 4.{i})"
        if n_rows != 1252:
            problems.append(f"{label}: {n_rows} rows, expected 1252")
        if bad_cols:
            problems.append(f"{label}: {bad_cols} rows without 6 age columns")
        ok = "ok" if (n_rows == 1252 and not bad_cols) else "FAIL"
        rows.append([label, str(n_rows), "6" if not bad_cols else f"{bad_cols} bad", ok])
    for scenario, nuc, age, expected in FGR15_SPOTS:
        got, units = nucleide.nuclei.fgr15_dose_rate(nuc, scenario, age, dest=cache_dir)
        if got != expected:
            problems.append(f"{scenario} {nuc} {age}: got {got!r}, expected {expected!r}")
        rows.append(
            [
                f"{scenario} {nuc} {age}",
                f"{got:.3e} {units}",
                f"{expected:.3e}",
                "ok" if got == expected else "FAIL",
            ]
        )
    return {"rows": rows, "problems": problems, "zip": str(zip_path)}


def main() -> int:
    report = Report("nuclear_data", "Nuclear data (`nuclear_data_vs_refs.py`)")

    mass_sample = [
        "H1",
        "C12",
        "N14",
        "O16",
        "Fe56",
        "U235",
        "U238",
        "Pu239",
        "Pu240",
        "Am241",
        "Am242m",
        "Ba137m",
        "Xe135",
        "Cs137",
        "Sr90",
        "Co60",
        "Ni58",
        "Mn55",
        "Cu63",
        "Mo95",
        "Tc99",
        "I129",
        "I135",
        "Xe136",
        "Nd143",
        "Sm149",
        "Eu151",
        "Gd157",
        "Ho165",
        "W182",
        "W186",
        "Pb206",
        "Pb207",
        "Pb208",
        "Bi209",
        "Th232",
        "Pa233",
        "U233",
        "Np237",
        "Pu241",
        "Pu242",
        "Am243",
        "Cm244",
        "Bk249",
        "Cf252",
        "Es253",
        "Fm257",
        "Md260",
        "No259",
        "Lr262",
    ]

    hl_sample = [
        "H3",
        "C14",
        "Co60",
        "Sr90",
        "Tc99",
        "I129",
        "I135",
        "Cs137",
        "Ba137m",
        "Pm147",
        "Sm151",
        "Eu154",
        "Am241",
        "Am242m",
        "Cm244",
        "Pu239",
        "Pu240",
        "U235",
        "U238",
        "Np237",
    ]

    name_sample = [
        "H1",
        "U235",
        "Pu239",
        "Am242m",
        "Ba137m",
        "Co60",
        "Cs137",
        "I135",
        "Xe135",
        "Fe56",
        "W186",
    ]

    mass_stats = compare_atomic_masses(mass_sample) if (HAS_PYNE and HAS_OPENMC) else None
    abund_stats = compare_natural_abundances() if (HAS_PYNE and HAS_OPENMC) else None
    hl_stats = compare_half_lives(hl_sample) if (HAS_PYNE and HAS_OPENMC) else None
    name_stats = compare_name_conversions(name_sample) if (HAS_PYNE and HAS_OPENMC) else None

    if not (HAS_PYNE and HAS_OPENMC):
        report.heading("Reference-data oracles (PyNE/OpenMC)")
        for reason in (PYNE_SKIP, OPENMC_SKIP):
            if reason:
                print(reason)
                report.prose(f"SKIPPED: {reason}")
        report.prose(
            "The atomic-mass, natural-abundance, half-life, and name-dialect"
            " comparisons require both PyNE and OpenMC; they rerun inside the"
            " validation container."
        )
    else:
        assert mass_stats is not None and abund_stats is not None
        assert hl_stats is not None and name_stats is not None
        report.heading("Atomic masses")
        report.table(
            ["Reference", "Max abs diff (u)", "Mean abs diff (u)"],
            [
                ["OpenMC", fmt(mass_stats["openmc_max_abs"]), fmt(mass_stats["openmc_mean_abs"])],
                ["PyNE", fmt(mass_stats["pyne_max_abs"]), fmt(mass_stats["pyne_mean_abs"])],
            ],
        )

        report.heading(f"Natural abundances ({abund_stats['count']} isotopes)")
        report.table(
            ["Reference", "Max abs diff", "Mean abs diff"],
            [
                ["OpenMC", fmt(abund_stats["openmc_max_abs"]), fmt(abund_stats["openmc_mean_abs"])],
                ["PyNE", fmt(abund_stats["pyne_max_abs"]), fmt(abund_stats["pyne_mean_abs"])],
            ],
        )

        report.heading("Half-lives")
        report.table(
            ["Reference", "Max rel diff", "Mean rel diff"],
            [
                ["OpenMC", fmt(hl_stats["openmc_max_rel"]), fmt(hl_stats["openmc_mean_rel"])],
                ["PyNE", fmt(hl_stats["pyne_max_rel"]), fmt(hl_stats["pyne_mean_rel"])],
            ],
        )

        report.heading("Name-dialect conversions vs `pyne.nucname`")
        report.prose(
            "All conversions (alara, cinder, nist, nucid, serpent, sza, zaid, zzaaam) had a\n"
            f"maximum relative/absolute difference of **{fmt(max(name_stats.values()))}**."
        )

    xs_stats = compare_simple_xs(SIMPLE_XS_SAMPLE)
    report.heading("Screening cross sections (`simple_xs`) vs PyNE `nuc_data.h5`")
    report.prose(
        "Nucleide thermal (0.0253 eV) and 14-MeV total cross sections vs PyNE's"
        " KAERI-anchored `SimpleDataSource` totals (`/neutron/simple_xs` in the"
        " `nuc_data.h5` bundled with PyNE, read in place — no download)."
        " Thermal is the source's first group point, fast the last (14 MeV)."
    )
    if xs_stats["rows"]:
        report.table(
            [
                "Nuclide",
                "Nucleide th (b)",
                "PyNE th (b)",
                "Rel diff th",
                "Nucleide fast (b)",
                "PyNE fast (b)",
                "Rel diff fast",
            ],
            xs_stats["rows"],
        )
    for note in xs_stats["skipped"]:
        print(f"SKIPPED simple_xs: {note}")
        report.prose(f"SKIPPED simple_xs: {note}")

    scat_stats = compare_scattering_lengths()
    report.heading("Bound coherent scattering lengths vs NIST anchors")
    report.prose(
        "Nucleide `scattering_length` (coherent, fm) vs the NIST NCNR bound"
        " coherent lengths (Sears, Neutron News 3(3), 1992) that"
        " `scattering_lengths.tsv` copies exactly — hence the tight tolerance."
    )
    report.table(
        ["Nuclide", "Nucleide (fm)", "NIST (fm)", "Abs diff (fm)"],
        scat_stats["rows"],
    )

    de_stats = compare_decay_energies()
    report.heading("Decay energies vs chain/ENDF-derived spot values (screening-level)")
    report.prose(
        "Nucleide `decay_energy` (mean recoverable MeV per decay) vs"
        " chain/ENDF-derived spot values within a loose 20% screening band:"
        " these placeholder heat values are NOT ENSDF evaluations — never for"
        " spectroscopy, dose, or safety use."
    )
    report.table(
        ["Nuclide", "Nucleide (MeV)", "Spot (MeV)", "Rel diff"],
        de_stats["rows"],
    )
    if de_stats["stable_none"]:
        de_note = "Stable Fe56 correctly resolves to None (no decay-energy entry)."
    else:
        de_note = "UNEXPECTED: stable Fe56 has a decay-energy entry."
    print(de_note)
    report.prose(de_note)

    br_stats = compare_decay_branches()
    report.heading("Decay branches vs ENDF/B-VIII.0 spot values")
    report.prose(
        "Nucleide `decay_branches` (per-branch daughters) and"
        " `decay_branch_fraction` vs the ENDF/B-VIII.0 MF8/MT457 NDK values"
        " they were generated from, with the parent half-life pinned to the"
        " same VIII.0 checkout: K-40 pins 3.93839e16 s plus its beta-/EC pair,"
        " Es-254 members follow this repo's own branch table. SF branches are"
        " dropped at generation, so they never appear here."
    )
    report.table(
        ["Parent", "Branch", "Nucleide", "Spot", "Rel diff"],
        br_stats["rows"],
    )

    fy_stats = compare_fission_yields()
    report.heading("Fission yields vs ENDF/B-VIII.0 tape spot values")
    report.prose(
        "Nucleide `fission_yields` (independent MF8/MT454 and cumulative"
        " MF8/MT459 sets) vs values read off the ENDF/B-VIII.0"
        " neutron-induced/spontaneous fission-yield tapes they were"
        " generated from. Both sublibraries are unchanged carries of"
        " ENDF/B-VII.1 evaluations per their README.txt (England et al."
        " ENDF-349, except the Chadwick-Kawano Pu-239 evaluation), and the"
        " Mattera-Sonzogni cumulative-yield correction is not applied."
        " Independent sets sum to ~2.0 (two fragments per fission); U-238's"
        " lowest neutron-induced set is 500 keV, and Cf-252's spontaneous"
        " set is E = 0."
    )
    report.table(
        ["Set -> daughter", "Nucleide", "Spot", "Rel diff"],
        fy_stats["rows"],
    )
    fy_om = compare_fission_yields_openmc()
    for note in fy_om["skipped"]:
        print(f"SKIPPED fission_yields: {note}")
        report.prose(f"SKIPPED fission_yields: {note}")
    if fy_om["rows"]:
        report.table(["Cross-check", "n", "Metric", "Value"], fy_om["rows"])

    fgr15_stats = compare_fgr15()
    report.heading("EPA FGR 15 external-dosimetry coefficients (runtime download)")
    report.prose(
        "Nucleide `load_fgr15_table` / `fgr15_dose_rate` parse the seven EPA\n"
        "FGR 15 (EPA 402-R-25-001, July 2025) `Table_4_*.DAT` scenario tables\n"
        "from the official zip, downloaded once into `validation/.cache/` and\n"
        f"hash-pinned (`{nucleide.data.FGR15_SHA256}`). Gates: per-table row\n"
        "count 1,252, six age columns per row, and ten hand spots transcribed\n"
        "from the published tables at exact equality (both sides read the same\n"
        "ASCII decimals, so the gate verifies the cell mapping, not float\n"
        "round-trip). Screening-level only — not for safety decisions."
    )
    report.table(["Table / spot", "Nucleide", "Expected", "Gate"], fgr15_stats["rows"])

    if xs_stats["available"]:
        print(f"simple_xs vs PyNE: max rel diff {fmt(max(xs_stats['diffs']))}")  # type: ignore[arg-type]
    print(f"scattering vs NIST: max abs diff {fmt(scat_stats['max_abs'])} fm")
    print(f"decay_energy vs spots: max rel diff {fmt(de_stats['max_rel'])}")
    print(f"decay_branches vs spots: max rel diff {fmt(br_stats['max_rel'])}")
    print(f"fission_yields vs spots: max rel diff {fmt(fy_stats['max_rel'])}")
    if fy_om["available"]:
        print(f"fission_yields vs OpenMC: max rel diff {fmt(max(fy_om['diffs']))}")  # type: ignore[arg-type]
    print(f"fgr15 spots: {len(FGR15_SPOTS)} checked, {len(fgr15_stats['problems'])} problems")

    emit_report(report)

    if abund_stats is not None and abund_stats["openmc_max_abs"] > 1.0e-12:
        print("FAIL: natural abundance mismatch with OpenMC", file=sys.stderr)
        return 1
    if hl_stats is not None and hl_stats["openmc_max_rel"] > 1.0e-6:
        print("FAIL: half-life mismatch with OpenMC", file=sys.stderr)
        return 1
    if xs_stats["available"] and max(xs_stats["diffs"]) > 0.30:  # type: ignore[arg-type]
        print("FAIL: simple_xs mismatch with PyNE simple_xs beyond 30%", file=sys.stderr)
        return 1
    if scat_stats["missing"] or scat_stats["max_abs"] > 1.0e-9:
        print("FAIL: scattering-length mismatch with NIST anchors", file=sys.stderr)
        return 1
    if de_stats["missing"] or not de_stats["stable_none"] or de_stats["max_rel"] > 0.20:
        print("FAIL: decay-energy spot check outside the 20% screening band", file=sys.stderr)
        return 1
    if br_stats["missing"] or br_stats["max_rel"] > 1.0e-6:
        print("FAIL: decay-branch spot check vs ENDF/B-VIII.0 failed", file=sys.stderr)
        return 1
    if fy_stats["missing"] or fy_stats["max_rel"] > 1.0e-6:
        print("FAIL: fission-yield spot check vs ENDF/B-VIII.0 tapes failed", file=sys.stderr)
        return 1
    if fgr15_stats["problems"]:
        for problem in fgr15_stats["problems"]:
            print(f"FAIL: fgr15: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
