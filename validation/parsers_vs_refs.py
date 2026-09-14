"""Cross-validate Nucleide's file parsers against independent oracle readers.

Oracles (run inside the validation container):
- Serpent `*.m` files: serpentTools (pip-installed; see Containerfile).
- MCNP files: `pyne.mcnp` (Xsdir, SurfSrc, PtracReader; Wwinp/Meshtal need
  PyMOAB, which the nomoab PyNE build lacks — those probes skip loudly).
- FLUKA: `pyne.fluka.Usrbin` reads only binary USRBIN and needs PyMOAB; no
  working oracle exists for our ASCII `.lis` fixtures, so FLUKA is skipped.

Inputs are our own committed fixtures under `fixtures/`; no third-party files.
Every skip is printed and recorded in the report prose.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from common import Report, fmt, rel_diff

import nucleide

REPO_ROOT = Path(__file__).resolve().parent.parent
SERPENT_DIR = REPO_ROOT / "fixtures" / "serpent"
MCNP_DIR = REPO_ROOT / "fixtures" / "mcnp"
FLUKA_DIR = REPO_ROOT / "fixtures" / "fluka"
CSG_DIR = MCNP_DIR / "inp"

# Scoped CSG fixtures: GO decks translate, reject decks raise ValueError.
CSG_GO_FIXTURES = [
    "deck_csg_sphere_box.txt",
    "deck_csg_rpp.txt",
    "deck_csg_rcc.txt",
    "deck_csg_complement.txt",
    "deck_csg_universe_fill.txt",
    "deck_csg_universe_data.txt",
    "deck_csg_lattice_rect.txt",
]
CSG_REJECT_FIXTURES = ["deck_csg_complement_reject.txt"]

# Worst relative difference over every compared numeric field.
WORST = 0.0


def _track(diff: float) -> float:
    global WORST
    WORST = max(WORST, diff)
    return diff


def _note(text: str) -> str:
    """Loud skip note: printed, and wrapped so the rendered Markdown lints."""
    note = textwrap.fill(text, width=100)
    print(note)
    return note


def arr_max_rel_diff(a, b) -> tuple[float, int]:
    """Max elementwise relative difference between two array-likes."""
    va = np.asarray(a, dtype=float).reshape(-1)
    vb = np.asarray(b, dtype=float).reshape(-1)
    if va.size != vb.size:
        raise ValueError(f"length mismatch: {va.size} vs {vb.size}")
    return max((_track(rel_diff(x, y)) for x, y in zip(va, vb, strict=True)), default=0.0), int(
        va.size
    )


def serpent_kind(path: Path) -> str | None:
    """Infer the Serpent file kind from its suffix."""
    name = path.name
    for suffix, kind in (("_res.m", "res"), ("_dep.m", "dep"), ("_det.m", "det")):
        if name.endswith(suffix):
            return kind
    return None


def compare_serpent_dep(path: Path, st_reader) -> list[list[str]]:
    """Compare depletion-file fields shared by Nucleide and serpentTools."""
    nuc = nucleide.serpent.read_serpent(str(path), "dep")
    rows: list[list[str]] = []

    # Scalar/vector metadata: ZAI, DAYS, BU.
    md = st_reader.metadata
    d, n = arr_max_rel_diff(nuc["ZAI"], md["zai"])
    rows.append(["ZAI", str(n), fmt(d)])
    d, n = arr_max_rel_diff(nuc["DAYS"], md["days"])
    rows.append(["DAYS", str(n), fmt(d)])
    d, n = arr_max_rel_diff(nuc["BU"], md["burnup"])
    rows.append(["BU (burnup)", str(n), fmt(d)])
    names_nuc = [str(x).strip() for x in nuc["NAMES"]]
    names_st = [str(x).strip() for x in md["names"]]
    n_mismatch = sum(1 for a, b in zip(names_nuc, names_st, strict=False) if a != b)
    n_mismatch += abs(len(names_nuc) - len(names_st))
    rows.append(["NAMES", str(len(names_nuc)), f"{n_mismatch} mismatches"])
    if n_mismatch:
        _track(1.0)

    # Per-material arrays present in both readers.
    fields = {"ADENS": "adens", "MDENS": "mdens", "VOLUME": "volume"}
    for mat_name, st_mat in sorted(st_reader.materials.items()):
        prefix = f"MAT_{mat_name}_" if mat_name != "total" else "TOT_"
        for nuc_key, st_key in sorted(fields.items()):
            full_key = prefix + nuc_key
            if full_key not in nuc or st_key not in st_mat.data:
                continue
            d, n = arr_max_rel_diff(nuc[full_key], st_mat.data[st_key])
            rows.append([full_key, str(n), fmt(d)])
    return rows


def compare_serpent_det(path: Path, st_reader) -> list[list[str]]:
    """Compare detector tallies, errors and grids shared by both readers."""
    nuc = nucleide.serpent.read_serpent(str(path), "det")
    rows: list[list[str]] = []
    for det_name, st_det in sorted(st_reader.detectors.items()):
        base = f"DET{det_name}"
        if base not in nuc:
            continue
        bins = np.asarray(nuc[base], dtype=float)
        d, n = arr_max_rel_diff(bins[:, 11], st_det.tallies)
        rows.append([f"{base} tallies", str(n), fmt(d)])
        d, n = arr_max_rel_diff(bins[:, 12], st_det.errors)
        rows.append([f"{base} rel errors", str(n), fmt(d)])
        for grid, st_grid in sorted(st_det.grids.items()):
            key = base + grid
            if key not in nuc:
                continue
            d, n = arr_max_rel_diff(nuc[key], st_grid)
            rows.append([f"{key} grid", str(n), fmt(d)])
    return rows


def serpent_section(report: Report) -> None:
    """Serpent fixtures vs serpentTools."""
    report.heading("Serpent vs serpentTools")
    try:
        import serpentTools
    except ImportError:
        note = (
            "SKIPPED: serpentTools is not installed in this environment; the Serpent "
            "oracle comparison did not run."
        )
        report.prose(_note(note))
        return

    st_reader_kind = {"res": "results", "dep": "dep", "det": "det"}
    report.prose(
        f"serpentTools {serpentTools.__version__} readers run on `fixtures/serpent/*.m`."
        "\nCompared fields — dep: `ZAI`, `DAYS`, `BU`/`burnup`, `NAMES`, per-material"
        " `ADENS`,\n`MDENS`, `VOLUME`; det: per-detector tally and relative-error columns plus"
        " shared\nbin grids (`E`, `T`, `X`, `Y`)."
    )
    rows: list[list[str]] = []
    skips: list[str] = []
    for path in sorted(SERPENT_DIR.glob("*.m")):
        kind = serpent_kind(path)
        if kind is None:
            continue
        try:
            reader = serpentTools.read(str(path), st_reader_kind[kind])
        except Exception as exc:
            note = (
                f"SKIPPED {path.name}: serpentTools {serpentTools.__version__} cannot read it "
                f"({type(exc).__name__}: {exc})"
            )
            skips.append(_note(note))
            continue
        if kind == "dep":
            for field, n, diff in compare_serpent_dep(path, reader):
                rows.append([path.name, field, n, diff])
        elif kind == "det":
            for field, n, diff in compare_serpent_det(path, reader):
                rows.append([path.name, field, n, diff])
        else:
            # res: nucleide side parses fine; compare nothing without an oracle
            # (serpentTools ResultsReader postprocessing fails on these files,
            # caught above).
            pass
    if rows:
        report.table(["File", "Field", "Values compared", "Max rel diff"], rows)
    for note in skips:
        report.prose(note)


def compare_xsdir() -> tuple[list[list[str]], list[str]]:
    """Compare xsdir table entries against pyne.mcnp.Xsdir."""
    try:
        from pyne import mcnp
    except ImportError as exc:
        return [], [_note(f"SKIPPED xsdir: PyNE oracle unavailable ({exc})")]

    path = MCNP_DIR / "xsdir" / "dummy_xsdir"
    nuc = nucleide.mcnp.read_xsdir(str(path))
    ref = mcnp.Xsdir(str(path))
    rows: list[list[str]] = []
    rows.append(["table count", str(len(nuc.tables)), fmt(abs(len(nuc.tables) - len(ref.tables)))])
    nuc_tables = {t.name: t for t in nuc.tables}
    for rt in ref.tables:
        nt = nuc_tables.get(rt.name)
        if nt is None:
            rows.append([rt.name, "—", "MISSING in Nucleide"])
            _track(1.0)
            continue
        str_ok = nt.filename == rt.filename and nt.ptable == rt.ptable
        int_ok = (
            nt.filetype == rt.filetype
            and nt.address == rt.address
            and nt.tablelength == rt.tablelength
        )
        d, _ = arr_max_rel_diff([nt.awr, nt.temperature or 0.0], [rt.awr, rt.temperature or 0.0])
        status = fmt(d) if (str_ok and int_ok) else "field mismatch"
        if not (str_ok and int_ok):
            _track(1.0)
        rows.append([f"{rt.name} (awr, temperature, name/dir/ints)", "2 + 5", status])
    return rows, []


def compare_surfsrc() -> tuple[list[list[str]], list[str]]:
    """Compare SSW surface-source files against pyne.mcnp.SurfSrc."""
    try:
        from pyne import mcnp
    except ImportError as exc:
        return [], [_note(f"SKIPPED ssw: PyNE oracle unavailable ({exc})")]

    rows: list[list[str]] = []
    skips: list[str] = []
    fields = ["nps", "wgt", "erg", "tme", "x", "y", "z", "u", "v", "w", "cs"]
    for path in sorted(MCNP_DIR.glob("ssw/*.w")):
        try:
            ref = mcnp.SurfSrc(str(path), "rb")
            ref.read_header()
            ref.read_tracklist()
        except Exception as exc:
            note = f"SKIPPED {path.name}: PyNE SurfSrc failed ({type(exc).__name__}: {exc})"
            skips.append(_note(note))
            continue
        nuc = nucleide.mcnp.read_ssw(str(path))
        header_ok = (
            nuc.kod.strip() == ref.kod.strip()
            and nuc.ver.strip() == ref.ver.strip()
            and nuc.np1 == ref.np1
            and nuc.nrss == ref.nrss
            and abs(nuc.ncrd) == abs(ref.ncrd)
            and nuc.njsw == ref.njsw
            and nuc.niss == ref.niss
        )
        nuc_tracks = nuc.tracks()
        worst = 0.0
        n_vals = 0
        for field in fields:
            a = [t[field] for t in nuc_tracks]
            b = [getattr(t, field) for t in ref.tracklist]
            d, n = arr_max_rel_diff(a, b)
            worst = max(worst, d)
            n_vals += n
        status = fmt(worst) if header_ok else "header mismatch"
        if not header_ok:
            _track(1.0)
        rows.append([f"{path.name} ({len(nuc_tracks)} tracks)", str(n_vals), status])
    return rows, skips


def compare_ptrac() -> tuple[list[list[str]], list[str]]:
    """Compare PTRAC headers against pyne.mcnp.PtracReader.

    PyNE ships no `PtracFile` class; `PtracReader` is its low-level binary
    helper, so only the header scalars it exposes (problem title, per-record
    variable counts) are compared.
    """
    try:
        from pyne.mcnp import PtracReader
    except ImportError as exc:
        return [], [_note(f"SKIPPED ptrac: PyNE oracle unavailable ({exc})")]

    rows: list[list[str]] = []
    skips: list[str] = []
    for path in sorted(MCNP_DIR.glob("ptrac/*.ptrac")):
        try:
            ref = PtracReader(str(path))
        except Exception as exc:
            note = f"SKIPPED {path.name}: PyNE PtracReader failed ({type(exc).__name__}: {exc})"
            skips.append(_note(note))
            continue
        nuc = nucleide.mcnp.read_ptrac(str(path))
        ref_title = ref.problem_title
        if isinstance(ref_title, bytes):
            ref_title = ref_title.decode(errors="replace")
        title_ok = nuc.problem_title.strip() == ref_title.strip()
        nums_ok = dict(nuc.variable_nums) == dict(ref.variable_nums)
        n_bad = (not title_ok) + (not nums_ok)
        if n_bad:
            _track(1.0)
        rows.append([path.name, "problem_title + variable_nums", "OK" if not n_bad else "MISMATCH"])
    return rows, skips


def compare_mctal() -> tuple[list[list[str]], list[str]]:
    """Compare MCTAL headers/cycles against `pyne.mcnp.Mctal`.

    PyNE's reader (`Mctal().read(path)`) parses the header, the declared
    tally numbers, and the `kcode` cycles only — tally bodies are skipped
    without advancing past them, so body-bearing files have no PyNE oracle
    at all. Comparison therefore runs only on the kcode-only fixtures;
    every body-bearing file (standard, mesh, tfc/variant) is a loud SKIP
    whose contents are covered instead by the synthetic closed-form
    fixtures (`crates/mcnp-io` tests plus `tests/test_mcnp_io.py`).
    """
    try:
        from pyne.mcnp import Mctal as PyneMctal
    except ImportError as exc:
        return [], [_note(f"SKIPPED mctal: PyNE oracle unavailable ({exc})")]

    rows: list[list[str]] = []
    skips: list[str] = []
    kcode_only = {"synthetic_kcode5.mctal", "synthetic_kcode19.mctal"}
    series = ["k_col", "k_abs", "k_path", "prompt_life_col", "prompt_life_path"]
    for path in sorted((MCNP_DIR / "mctal").glob("*.mctal")):
        if path.name not in kcode_only:
            skips.append(
                _note(
                    f"SKIPPED {path.name}: PyNE Mctal skips tally bodies without "
                    "advancing past them, so body-bearing files have no PyNE "
                    "oracle; covered by synthetic closed-form fixtures instead."
                )
            )
            continue
        try:
            ref = PyneMctal()
            ref.read(str(path))
        except Exception as exc:
            skips.append(
                _note(f"SKIPPED {path.name}: PyNE Mctal failed ({type(exc).__name__}: {exc})")
            )
            continue
        nuc = nucleide.mcnp.read_mctal(str(path))
        header_ok = (
            nuc.code_name == ref.code_name
            and nuc.comment == ref.comment
            and nuc.n_histories == ref.n_histories
            and nuc.n_cycles == ref.n_cycles
            and nuc.n_inactive == ref.n_inactive
        )
        if not header_ok:
            _track(1.0)
        rows.append([path.name, "header scalars", "5", "OK" if header_ok else "MISMATCH"])
        for field in series:
            d, n = arr_max_rel_diff(getattr(nuc, field), getattr(ref, field))
            rows.append([path.name, field, str(n), fmt(d)])
        if ref.avg_k_col:
            got = [(r["avg_k_col"], r["avg_k_col_stdev"]) for r in nuc.averages]
            d, n = arr_max_rel_diff(
                [v for pair in got for v in pair], [v for pair in ref.avg_k_col for v in pair]
            )
            rows.append([path.name, "avg_k_col pairs", str(n), fmt(d)])
    return rows, skips


def compare_endl() -> tuple[list[list[str]], list[str]]:
    """Compare ENDL tables against `pyne.endl.Library` (EEDL/EPDL scope)."""
    try:
        from pyne.endl import Library as PyneEndl
    except ImportError as exc:
        return [], [_note(f"SKIPPED endl: PyNE endl oracle unavailable ({exc})")]
    import numpy as np_mod

    path = REPO_ROOT / "fixtures" / "endl" / "synthetic_eedl.txt"
    try:
        ref = PyneEndl(str(path))
    except Exception as exc:
        return [], [_note(f"SKIPPED endl: PyNE Library failed ({type(exc).__name__}: {exc})")]
    nuc = nucleide.mcnp.read_endl(str(path))
    rows: list[list[str]] = []
    skips: list[str] = []
    ref_nucs = sorted(int(k) for k in ref.structure)
    if sorted(nuc.nuclides()) != ref_nucs:
        _track(1.0)
        rows.append(["nuclides", str(len(ref_nucs)), "MISMATCH"])
    else:
        rows.append(["nuclides", str(len(ref_nucs)), fmt(0.0)])
    for key in [(820000000, 9, 10, 0, None, None), (820000000, 9, 82, 21, None, None)]:
        nuc_id, p_in, rdesc, rprop, x1, p_out = key
        try:
            expected = np_mod.asarray(
                ref.get_rx(nuc_id, p_in, rdesc, rprop, x1=x1, p_out=p_out), dtype=float
            )
        except Exception as exc:
            skips.append(_note(f"SKIPPED endl {key}: PyNE get_rx failed ({exc})"))
            continue
        got = nuc.get_rx(nuc_id, p_in, rdesc, rprop, x1=x1, p_out=p_out)
        d, n = arr_max_rel_diff(np_mod.asarray(got, dtype=float), expected)
        rows.append([f"get_rx{nuc_id, p_in, rdesc, rprop}", str(n), fmt(d)])
    return rows, skips


def endl_section(report: Report) -> None:
    """ENDL fixtures vs `pyne.endl`."""
    report.heading("ENDL vs PyNE")
    report.prose(
        "PyNE `pyne.endl.Library` (EEDL/EPDL scope) runs on the committed"
        "\n`fixtures/endl/synthetic_eedl.txt` file. Compared fields — nucleus-id"
        "\nset plus `get_rx` arrays for the integrated (`rdesc=10`, `rprop=0`)"
        "\nand spectra (`rdesc=82`, `rprop=21`) tables."
    )
    rows, skips = compare_endl()
    if rows:
        report.table(["Item", "Values compared", "Max rel diff / status"], rows)
    for note in skips:
        report.prose(note)


def mcnp_section(report: Report) -> None:
    """MCNP fixtures vs pyne.mcnp."""
    report.heading("MCNP vs PyNE")
    report.prose(
        "PyNE 0.7.5 (`nomoab` build) `pyne.mcnp` readers run on the committed"
        "\n`fixtures/mcnp/` files. Compared fields — xsdir: per-table `name`, `awr`,"
        "\n`filename`, `filetype`, `address`, `tablelength`, `temperature`, `ptable`;"
        "\nssw: header (`kod`, `ver`, `np1`, `nrss`, `ncrd`, `njsw`, `niss`) and per-track"
        "\n`nps`, `wgt`, `erg`, `tme`, `x`, `y`, `z`, `u`, `v`, `w`, `cs` payloads;"
        "\nptrac: problem title and per-record variable counts (PyNE has no `PtracFile`"
        "\nclass; its low-level `PtracReader` exposes only headers);"
        "\nmctal: header scalars plus per-cycle keff/lifetime series on the"
        "\nkcode-only fixtures (PyNE `Mctal` skips tally bodies, so body-bearing"
        "\nfiles — standard, mesh, tfc/variant — are loud skips covered by"
        "\nsynthetic closed-form fixtures instead)."
    )

    rows: list[list[str]] = []
    xsdir_rows, xsdir_skips = compare_xsdir()
    for name, fileds, status in xsdir_rows:
        rows.append(["xsdir", name, fileds, status])
    ssw_rows, ssw_skips = compare_surfsrc()
    for name, n_vals, status in ssw_rows:
        rows.append(["ssw", name, n_vals, status])
    ptrac_rows, ptrac_skips = compare_ptrac()
    for name, what, status in ptrac_rows:
        rows.append(["ptrac", name, what, status])
    mctal_rows, mctal_skips = compare_mctal()
    for name, field, n_vals, status in mctal_rows:
        rows.append(["mctal", f"{name} {field}", n_vals, status])
    report.table(["Format", "Item", "Values compared", "Max rel diff / status"], rows)

    skips = xsdir_skips + ssw_skips + ptrac_skips + mctal_skips
    for what, reason in [
        ("wwinp", "`pyne.mcnp.Wwinp` requires PyMOAB to build its mesh (nomoab build)"),
        ("meshtal", "`pyne.mcnp.Meshtal` requires PyMOAB (nomoab build)"),
    ]:
        for path in sorted(MCNP_DIR.glob(f"{what}/*")):
            note = f"SKIPPED {path.name}: {reason}"
            skips.append(_note(note))
    for note in skips:
        report.prose(note)


def fluka_section(report: Report) -> None:
    """FLUKA fixtures: no working oracle."""
    report.heading("FLUKA vs PyNE")
    note = (
        "SKIPPED: no working FLUKA oracle exists in this environment. `pyne.fluka.Usrbin`"
        " requires PyMOAB (absent in the nomoab build) and reads only binary USRBIN output,"
        " while Nucleide's committed fixtures are ASCII `.lis` files"
        f" ({', '.join(p.name for p in sorted(FLUKA_DIR.glob('*.lis')))})."
    )
    report.prose(_note(note))


def csg_section(report: Report) -> None:
    """MCNP CSG translation fixtures vs OpenMC (scoped v3: + rectangular lattices)."""
    report.heading("CSG translation vs OpenMC")
    report.prose(
        "The `nucleide.mcnp.parse_csg_to_openmc` facade translates the committed"
        "\n`fixtures/mcnp/inp/deck_csg_*.txt` decks to OpenMC `geometry.xml` (scoped"
        "\nv3: surfaces, cells, nested universes, rectangular `LAT=1` lattices, and a"
        "\nmaterial stub; a filled cell carries `fill` instead of `material`)."
        "\nStructural probes — surface and cell counts, region ids referencing defined"
        "\nsurfaces, boundary attributes, material/fill stubs — always run, as do"
        "\nlattice cross-references (dimension/count products, universe-id references,"
        "\nfill linkage) for decks with `<lattice>` blocks. When `openmc` is importable"
        "\n(validation container), the `Region.from_expression` cross-check runs, every"
        "\nGO deck's `geometry.xml` is fully loaded by OpenMC (`Geometry.from_xml` with"
        "\nstub materials for the emitted material numbers), and lattice decks are"
        "\nverified geometrically against the source deck: each element's universe id"
        "\nmatches the deck `FILL` matrix entry at the same MCNP `(i, j, k)` indices"
        "\n(`k, j, i` i-fastest vs OpenMC row-major `z` up/`y` down/`x` up), the"
        "\nelement centroid lies inside the universe's cells as positioned by the"
        "\ndeck surfaces, and `pitch * dimension` spans the lattice cell's RPP bounds."
        "\nReject decks must raise `ValueError`."
    )
    rows, skips = compare_csg()
    if rows:
        report.table(["Deck", "Probe", "Values compared", "Max rel diff / status"], rows)
    for note in skips:
        report.prose(note)


def _serpent_probe(text: str) -> tuple[str, str]:
    """Structural probe over translated Serpent `surf`/`cell` cards.

    Returns (values compared, status). Checks card counts, that region
    surface refs name defined surfaces, `#n` refs name defined cells, filled
    cells carry no material entry, and every `lat` card is cuboidal (type
    11) with its element count matching the universe table and each listed
    universe assigned to some cell.
    """
    import math
    import re

    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("%")]
    surfs = [ln.split() for ln in lines if ln.startswith("surf ")]
    cells = [ln.split() for ln in lines if ln.startswith("cell ")]
    known_surfs = {t[1] for t in surfs}
    known_cells = {t[1] for t in cells}
    known_universes = {t[2] for t in cells}
    bad = 0
    n_refs = 0
    for toks in cells:
        if len(toks) > 4 and toks[3] == "fill":
            bad += not toks[4].isdigit()
            rest = toks[5:]
        else:
            bad += not (
                len(toks) > 3 and (toks[3] == "void" or re.fullmatch(r"m\d+", toks[3]) is not None)
            )
            rest = toks[4:]
        for tok in rest:
            for num in re.findall(r"-?\d+", tok):
                n_refs += 1
                num = num.lstrip("-")
                if tok.startswith("#"):
                    bad += num not in known_cells
                else:
                    bad += num not in known_surfs
    n_lat = 0
    for toks in [ln.split() for ln in lines if ln.startswith("lat ")]:
        n_lat += 1
        # `lat id 11 cx cy cz nx ny nz px py pz u...` (cuboidal type 11).
        if len(toks) < 12 or toks[2] != "11":
            bad += 1
            continue
        try:
            counts = [int(v) for v in toks[6:9]]
        except ValueError:
            bad += 1
            continue
        tail = toks[12:]
        n_refs += len(tail) + 1
        bad += math.prod(counts) != len(tail)
        bad += any(not u.isdigit() or u not in known_universes for u in tail)
        bad += sum(t[3] == "fill" and len(t) > 4 and t[4] == toks[1] for t in cells) != 1
    if bad:
        _track(1.0)
    compared = f"{len(surfs)} surfs, {len(cells)} cells, {n_refs} refs"
    if n_lat:
        compared += f", {n_lat} lattices"
    return (compared, "OK" if not bad else f"{bad} MISMATCHES")


def _phits_probe(text: str) -> tuple[str, str]:
    """Structural probe over translated PHITS `[Surface]`/`[Cell]` sections.

    Returns (values compared, status). Checks section presence, that region
    surface refs name defined surfaces, `#n` refs name defined cells, void/
    outer-void cells omit the density field, and every `LAT=1` cell carries
    a matrix `FILL` whose element count matches its ranges and whose
    universe list references `U=` universes defined by other cells.
    """
    import math
    import re

    section = ""
    surfs: list[list[str]] = []
    cells: list[list[str]] = []
    ok_sections = "[ Surface ]" in text and "[ Cell ]" in text
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped or stripped.startswith("$"):
            continue
        if stripped.startswith("["):
            section = stripped
            continue
        if section == "[ Surface ]":
            surfs.append(stripped.split())
        elif section == "[ Cell ]":
            cells.append(stripped.split())
    known_surfs = {t[0].lstrip("*") for t in surfs}
    known_cells = {t[0] for t in cells}
    known_universes = {t.split("=", 1)[1] for c in cells for t in c if t.startswith("U=")}
    bad = 0 if ok_sections else 1
    n_refs = 0
    n_lat = 0
    for toks in cells:
        # Cell params (`U=`, `FILL=`, `LAT=`, ...) trail the region; a
        # lattice `FILL` value spans three range tokens plus the universe
        # list, so the first `=`-bearing token marks the region end.
        pidx = next((i for i, t in enumerate(toks) if "=" in t), len(toks))
        body = toks[:pidx]
        params = toks[pidx:]
        if len(body) < 3:
            bad += 1
            continue
        try:
            mat = int(body[1])
        except ValueError:
            bad += 1
            continue
        if mat in (0, -1):
            region = body[2:]
            bad += any("=" in t for t in region)
        else:
            try:
                float(body[2])
            except ValueError:
                bad += 1
                continue
            region = body[3:]
        for tok in region:
            for num in re.findall(r"-?\d+", tok):
                n_refs += 1
                num = num.lstrip("-")
                if tok.startswith("#"):
                    bad += num not in known_cells
                else:
                    bad += num not in known_surfs
        lat = next((t for t in params if t.startswith("LAT=")), None)
        if lat is None:
            continue
        n_lat += 1
        bad += lat != "LAT=1"
        fidx = next((i for i, t in enumerate(params) if t.startswith("FILL=")), None)
        if fidx is None or fidx + 3 > len(params):
            bad += 1
            continue
        # `FILL=i1:i2 j1:j2 k1:k2`: the first range is glued to `FILL=`,
        # the next two are bare range tokens, then the universe list.
        try:
            spans = [params[fidx].split("=", 1)[1].split(":")] + [
                r.split(":") for r in params[fidx + 1 : fidx + 3]
            ]
            counts = [int(hi) - int(lo) + 1 for lo, hi in spans]
        except ValueError:
            bad += 1
            continue
        tail = params[fidx + 3 :]
        n_refs += len(tail) + 1
        bad += math.prod(counts) != len(tail)
        bad += any(":" in t for t in tail)
        bad += any(not u.isdigit() or u not in known_universes for u in tail)
    if bad:
        _track(1.0)
    compared = f"{len(surfs)} surfs, {len(cells)} cells, {n_refs} refs"
    if n_lat:
        compared += f", {n_lat} lattices"
    return (compared, "OK" if not bad else f"{bad} MISMATCHES")


def _openmc_lattice_probe(root) -> tuple[str, str]:
    """Universe-id cross-reference and count checks over `<lattice>` blocks.

    `Region.from_expression` below is region-only, so these checks cover the
    lattice block itself: the dimension product matches the universe-list
    length, `lower_left`/`pitch` carry three components, the lattice id
    collides with no cell or surface id, exactly one cell fills it, and
    every listed universe is assigned to some cell.
    """
    import math

    cells = root.findall("cell")
    known_ids = {s.get("id") for s in root.findall("surface")} | {c.get("id") for c in cells}
    defined_universes = {c.get("universe", "0") for c in cells}
    bad = 0
    n_vals = 0
    lattices = root.findall("lattice")
    for lat in lattices:
        lat_id = lat.get("id", "")
        dims = lat.findtext("dimension", "").split()
        unis = lat.findtext("universes", "").split()
        try:
            n_elements = math.prod(int(d) for d in dims)
        except ValueError:
            n_elements = -1
        n_vals += len(dims) + len(unis) + 4
        bad += n_elements != len(unis)
        bad += len(lat.findtext("lower_left", "").split()) != 3
        bad += len(lat.findtext("pitch", "").split()) != 3
        bad += lat_id in known_ids
        bad += sum(u not in defined_universes for u in unis)
        bad += sum(c.get("fill") == lat_id for c in cells) != 1
    if bad:
        _track(1.0)
    return (
        f"{len(lattices)} lattices, {n_vals} values",
        "OK" if not bad else f"{bad} MISMATCHES",
    )


def compare_csg() -> tuple[list[list[str]], list[str]]:
    """Translate the CSG fixtures and probe the resulting geometry XML."""
    import re
    import xml.etree.ElementTree as ET

    rows: list[list[str]] = []
    skips: list[str] = []
    parsed: dict[str, ET.Element] = {}
    for name in CSG_GO_FIXTURES:
        path = CSG_DIR / name
        try:
            xml, _ = nucleide.mcnp.read_csg_to_openmc(str(path))
        except Exception as exc:
            _track(1.0)
            rows.append([name, "translate", "—", f"ERROR: {exc}"])
            continue
        try:
            root = ET.fromstring(xml)
        except Exception as exc:
            _track(1.0)
            rows.append([name, "xml well-formed", "—", f"ERROR: {exc}"])
            continue
        parsed[name] = root
        surfs = root.findall("surface")
        cells = root.findall("cell")
        known = {s.get("id") for s in surfs}
        n_refs = 0
        bad = 0
        for cell in cells:
            region = cell.get("region", "")
            for tok in re.findall(r"-?\d+", region):
                n_refs += 1
                bad += tok.lstrip("-") not in known
            mat = cell.get("material", "")
            if cell.get("fill") is not None:
                bad += mat != "" or not cell.get("fill", "").isdigit()
            else:
                bad += not (mat == "void" or mat.isdigit())
        for surf in surfs:
            boundary = surf.get("boundary", "transmission")
            if boundary not in ("transmission", "reflective", "periodic"):
                bad += 1
            if boundary == "periodic" and surf.get("periodic_surface_id") not in known:
                bad += 1
        if bad:
            _track(1.0)
        rows.append(
            [
                name,
                "structure",
                f"{len(surfs)} surfs, {len(cells)} cells, {n_refs} refs",
                "OK" if not bad else f"{bad} MISMATCHES",
            ]
        )
        if root.findall("lattice"):
            rows.append([name, "lattice cross-reference", *_openmc_lattice_probe(root)])
        try:
            serpent_text, _ = nucleide.mcnp.read_csg_to_serpent(str(path))
        except Exception as exc:
            _track(1.0)
            rows.append([name, "serpent structure", "—", f"ERROR: {exc}"])
            continue
        rows.append([name, "serpent structure", *_serpent_probe(serpent_text)])
        try:
            phits_text, _ = nucleide.mcnp.read_csg_to_phits(str(path))
        except Exception as exc:
            _track(1.0)
            rows.append([name, "phits structure", "—", f"ERROR: {exc}"])
            continue
        rows.append([name, "phits structure", *_phits_probe(phits_text)])
    for name in CSG_REJECT_FIXTURES:
        try:
            nucleide.mcnp.read_csg_to_openmc(str(CSG_DIR / name))
        except ValueError:
            rows.append([name, "reject", "1", "OK (ValueError)"])
        except Exception as exc:
            _track(1.0)
            rows.append([name, "reject", "1", f"WRONG ERROR: {exc}"])
        else:
            _track(1.0)
            rows.append([name, "reject", "1", "MISSING ERROR"])
    try:
        import openmc
    except ImportError as exc:
        skips.append(
            _note(
                "SKIPPED CSG OpenMC cross-check (region parsing, geometry load,"
                f" lattice geometry): oracle unavailable ({exc})"
            )
        )
        return rows, skips
    for name, root in parsed.items():
        try:
            surfaces = {
                int(s.get("id", "0")): _openmc_surface(openmc, s) for s in root.findall("surface")
            }
            cells = root.findall("cell")
            for cell in cells:
                region = openmc.Region.from_expression(cell.get("region", ""), surfaces)
                _ = str(region)
            rows.append([name, "openmc regions", str(len(cells)), "OK"])
        except Exception as exc:
            _track(1.0)
            rows.append([name, "openmc regions", "—", f"ERROR: {exc}"])
    loaded: dict[str, Any] = {}
    for name, root in parsed.items():
        try:
            loaded[name] = _openmc_load_geometry(openmc, root)
        except Exception as exc:
            _track(1.0)
            rows.append([name, "openmc geometry load", "—", f"ERROR: {exc}"])
            continue
        n_cells = len(loaded[name].get_all_cells())
        n_universes = len(loaded[name].get_all_universes())
        what = f"{n_cells} cells, {n_universes} universes"
        rows.append([name, "openmc geometry load", what, "OK"])
    for name, root in parsed.items():
        if not root.findall("lattice") or name not in loaded:
            continue
        try:
            deck = nucleide.mcnp.read_deck(str(CSG_DIR / name))
            compared, status = _openmc_lattice_geometry_probe(openmc, loaded[name], deck)
        except Exception as exc:
            _track(1.0)
            rows.append([name, "openmc lattice geometry", "—", f"ERROR: {exc}"])
            continue
        rows.append([name, "openmc lattice geometry", compared, status])
    return rows, skips


def _openmc_surface(openmc: Any, elem: Any) -> Any:
    """Build an `openmc.Surface` from one translated `<surface>` element."""
    stype = elem.get("type", "")
    coeffs = [float(v) for v in elem.get("coeffs", "").split()]
    if stype == "x-plane":
        return openmc.XPlane(x0=coeffs[0])
    if stype == "y-plane":
        return openmc.YPlane(y0=coeffs[0])
    if stype == "z-plane":
        return openmc.ZPlane(z0=coeffs[0])
    if stype == "sphere":
        return openmc.Sphere(x0=coeffs[0], y0=coeffs[1], z0=coeffs[2], r=coeffs[3])
    if stype == "x-cylinder":
        return openmc.XCylinder(y0=coeffs[0], z0=coeffs[1], r=coeffs[2])
    if stype == "y-cylinder":
        return openmc.YCylinder(x0=coeffs[0], z0=coeffs[1], r=coeffs[2])
    if stype == "z-cylinder":
        return openmc.ZCylinder(x0=coeffs[0], y0=coeffs[1], r=coeffs[2])
    raise ValueError(f"CSG probe cannot build OpenMC surface type {stype!r}")


def _openmc_load_geometry(openmc: Any, root: Any) -> Any:
    """Load one translated `<geometry>` element into a real `openmc.Geometry`.

    OpenMC 0.16.0 resolves cell `material` attributes against an
    `openmc.Materials` collection, so stub materials are supplied for exactly
    the MCNP material numbers the emitter wrote (void cells need none).
    """
    import tempfile
    import xml.etree.ElementTree as ET

    mat_ids = sorted(
        {
            int(cell.get("material"))
            for cell in root.findall("cell")
            if (cell.get("material") or "").isdigit()
        }
    )
    materials = openmc.Materials([openmc.Material(material_id=m) for m in mat_ids])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "geometry.xml"
        decl = b'<?xml version="1.0" encoding="utf-8"?>\n'
        path.write_bytes(decl + ET.tostring(root))
        return openmc.Geometry.from_xml(str(path), materials=materials)


def _deck_lattice_bounds(deck: Any, cell_num: int) -> list[float]:
    """Lattice-cell `[xmin, xmax, ymin, ymax, zmin, zmax]` from the parsed deck.

    Mirrors the translator's two supported boundings: an interior `RPP`
    (`-N`) or an intersection of six axis-plane half-spaces (the `+`/bare side
    gives the minimum, the `-` side the maximum, one pair per axis).
    """
    import math

    surfs = {int(s["num"]): s for s in deck.surfs}
    cell = next(c for c in deck.cells if int(c["num"]) == cell_num)
    toks = cell["geom"].split()
    if len(toks) == 1 and toks[0].startswith("-"):
        rpp = surfs.get(int(toks[0][1:]))
        if rpp is not None and rpp["kind"].upper() == "RPP":
            coeffs = [float(v) for v in rpp["coeffs"].split()]
            if len(coeffs) == 6 and all(coeffs[2 * a] < coeffs[2 * a + 1] for a in range(3)):
                return coeffs
    axis_of = {"PX": 0, "X": 0, "PY": 1, "Y": 1, "PZ": 2, "Z": 2}
    lo = [math.nan] * 3
    hi = [math.nan] * 3
    if len(toks) == 6:
        for tok in toks:
            sign = -1 if tok.startswith("-") else 1
            surf = surfs.get(int(tok.lstrip("+-")))
            if surf is None:
                break
            axis = axis_of.get(surf["kind"].upper())
            if axis is None:
                break
            slot = lo if sign > 0 else hi
            if not math.isnan(slot[axis]):
                raise ValueError(f"cell {cell_num} bounds repeat one axis side")
            slot[axis] = float(surf["coeffs"].split()[0])
    if any(math.isnan(v) for v in lo + hi) or any(lo[a] >= hi[a] for a in range(3)):
        raise ValueError(f"cell {cell_num} bounds are not an ordered axis-aligned box")
    return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]


def _openmc_lattice_geometry_probe(openmc: Any, geometry: Any, deck: Any) -> tuple[str, str]:
    """Geometric verification of a loaded `<lattice>` against the source deck.

    The deck's matrix `FILL` lists element universes in MCNP `k, j, i` order
    (`i` fastest); the OpenMC `<universes>` list is row-major with `z`
    ascending, `y` descending (top row first), and `x` ascending. For every
    lattice element this checks, through the loaded OpenMC geometry, that:

    - the element's universe id equals the deck matrix entry for the same
      MCNP indices (natural OpenMC element `(ix, iy, iz)` corresponds to
      matrix indices `(imin + ix, jmin + iy, kmin + iz)`);
    - the element centroid lies inside exactly one cell of that universe
      (geometrically pinning `lower_left`, `pitch`, and the index order), and
      `RectLattice.find_element` maps the centroid back to the same index; and
    - the universe contains exactly the cells the deck assigns to it (`u=k`).

    Also checks the fill linkage, that `pitch * dimension` spans the lattice
    cell bounds per axis with `lower_left` at the bounds minimum, and that
    every lattice universe resolves to a deck `u=k` universe (OpenMC
    auto-creates empty universes for unknown ids, so a bogus id would surface
    as an empty cell set).

    Returns (values compared, status).
    """
    import math

    bad = 0
    n_vals = 0
    n_elements = 0
    deck_universes = {
        int(u["number"]): sorted(int(c) for c in u["cells"].split()) for u in deck.universes
    }
    cells_by_id = geometry.get_all_cells()
    for fill in deck.fills:
        if fill["kind"] != "matrix":
            continue
        lat_cell = cells_by_id.get(int(fill["cell"]))
        n_vals += 1
        lat = lat_cell.fill if lat_cell is not None else None
        if not isinstance(lat, openmc.RectLattice):
            _track(1.0)
            return (
                "1 fill",
                f"MISMATCH (cell {fill['cell']} is not lattice-filled in the loaded geometry)",
            )
        mins = [int(v) for v in fill["min_index"].split()]
        maxs = [int(v) for v in fill["max_index"].split()]
        counts = [hi - lo + 1 for lo, hi in zip(mins, maxs, strict=True)]
        nx, ny, nz = counts
        matrix = [int(v) for v in fill["universes"].split()]
        bounds = _deck_lattice_bounds(deck, int(fill["cell"]))
        n_elements += nx * ny * nz
        for a in range(3):
            n_vals += 2
            bad += not math.isclose(
                lat.lower_left[a], bounds[2 * a], rel_tol=1.0e-12, abs_tol=1.0e-12
            )
            bad += not math.isclose(
                lat.pitch[a] * lat.shape[a],
                bounds[2 * a + 1] - bounds[2 * a],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
        for uni in lat.universes.ravel():
            n_vals += 1
            bad += sorted(c.id for c in uni.cells.values()) != deck_universes.get(uni.id)
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    expected = matrix[(iz * ny + iy) * nx + ix]
                    uni = lat.universes[lat.get_universe_index((ix, iy, iz))]
                    centroid = tuple(
                        lat.lower_left[a] + (idx + 0.5) * lat.pitch[a]
                        for a, idx in enumerate((ix, iy, iz))
                    )
                    hits = [c.id for c in uni.cells.values() if c.region and centroid in c.region]
                    found = lat.find_element(centroid)[0]
                    n_vals += 3
                    bad += uni.id != expected
                    bad += len(hits) != 1
                    bad += tuple(found) != (ix, iy, iz)
    if bad:
        _track(1.0)
    return (
        f"{n_elements} elements, {n_vals} values",
        "OK" if not bad else f"{bad} MISMATCHES",
    )


def main() -> int:
    report = Report("parsers", "Parser cross-validation (`parsers_vs_refs.py`)")
    report.prose(
        "Nucleide's readers are cross-checked against independent oracle readers on the"
        "\nsame committed fixture files. Skipped comparisons (missing or incapable oracle)"
        "\nare listed explicitly below; none are silent."
    )
    serpent_section(report)
    mcnp_section(report)
    csg_section(report)
    endl_section(report)
    fluka_section(report)
    report.emit()

    if WORST > 1.0e-6:
        print(f"FAIL: parser oracle mismatch up to {WORST:.3e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
