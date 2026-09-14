"""EPA FGR 15 external-dosimetry coefficients (synthetic data, fully offline).

The published EPA zip is never touched here: tests build a synthetic zip in
the exact ``Table_4_*.DAT`` layout (CRLF, element separators, a BOM case) and
monkeypatch ``nucleide.data.FGR15_SHA256`` to its digest — ``fetch_fgr15``
enforces the pinned hash on every call, so the override is how a synthetic
"release" is pinned.
"""

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import pytest

import nucleide
import nucleide.data

MEMBER_41 = "FGR15_Tables/Table_4_1.DAT"
MEMBER_46 = "FGR15_Tables/Table_4_6.DAT"

Row = tuple[str, str, list[float]]

#: (element separator, nuclide, six coefficients) — hand-built fake values.
ROWS_41: list[Row] = [
    ("Hydrogen", "H-3", [1.00e-27, 2.00e-27, 3.00e-27, 4.00e-27, 5.00e-27, 6.00e-27]),
    ("Antimony", "Sb-124n", [1.10e-27, 1.20e-27, 1.30e-27, 1.40e-27, 1.50e-27, 1.60e-27]),
    ("Barium", "Ba-137m", [2.10e-27, 2.20e-27, 2.30e-27, 2.40e-27, 2.50e-27, 2.60e-27]),
]
ROWS_46: list[Row] = [
    ("Hydrogen", "H-3", [7.00e-28, 8.00e-28, 9.00e-28, 1.00e-27, 1.10e-27, 1.20e-27]),
    ("Carbon", "C-14", [3.00e-28, 3.10e-28, 3.20e-28, 3.30e-28, 3.40e-28, 3.50e-28]),
    ("Caesium", "Cs-137", [4.00e-28, 4.10e-28, 4.20e-28, 4.30e-28, 4.40e-28, 4.50e-28]),
]


def table_text(table_no: str, title: str, units: str, rows: list[Row]) -> str:
    """Build one synthetic ``Table_4_*.DAT`` member in the exact EPA layout."""
    lines = [
        f"Table {table_no}. {title}",
        "",
        f"          ------- Reference Person Coefficients ({units}) --------",
        " Nuclide  Newborn   1-yr-old  5-yr-old 10-yr-old 15-yr-old   Adult",
        "-" * 69,
    ]
    for i, (element, name, coeffs) in enumerate(rows):
        # The first element separator carries a BOM, as in the real Table 4.5.
        prefix = "\ufeff" if i == 0 else ""
        lines.append(f"{prefix}{element}")
        lines.append(f" {name}  " + "  ".join(f"{c:.2e}" for c in coeffs))
    lines.append("-" * 69)
    return "\r\n".join(lines) + "\r\n"


def make_zip(
    path: Path,
    rows41: list[Row] | None = None,
    table_no_41: str = "4.1",
) -> None:
    """Write a synthetic two-table FGR 15 zip (tables 4.1 and 4.6)."""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            MEMBER_41,
            table_text(
                table_no_41,
                "Synthetic Ground Surface (3 mm)",
                "Sv m2/Bq s",
                ROWS_41 if rows41 is None else rows41,
            ),
        )
        zf.writestr(
            MEMBER_46,
            table_text("4.6", "Synthetic Air Submersion", "Sv m3 per Bq s", ROWS_46),
        )


def make_pinned_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build the synthetic zip and pin it as the expected FGR 15 download."""
    zip_path = tmp_path / "fgr15_synthetic.zip"
    make_zip(zip_path)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    monkeypatch.setattr(nucleide.data, "FGR15_SHA256", digest)
    return zip_path


def fetch_kwargs(tmp_path: Path, zip_path: Path) -> dict[str, Any]:
    return {"dest": tmp_path / "cache", "url": zip_path.as_uri()}


class TestFetch:
    def test_default_cache_dir_under_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = make_pinned_zip(tmp_path, monkeypatch)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        got = nucleide.data.fetch_fgr15(url=zip_path.as_uri())
        assert got == str(tmp_path / ".cache" / "nucleide" / "fgr15_synthetic.zip")

    def test_download_then_cache_hit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        zip_path = make_pinned_zip(tmp_path, monkeypatch)
        got = nucleide.data.fetch_fgr15(**fetch_kwargs(tmp_path, zip_path))
        assert got == str(tmp_path / "cache" / "fgr15_synthetic.zip")
        # A verified cache is reused without touching the network: the second
        # call's URL is unreachable and must never be opened.
        again = nucleide.data.fetch_fgr15(
            dest=tmp_path / "cache", url="http://127.0.0.1:59999/fgr15_synthetic.zip"
        )
        assert again == got

    def test_hash_mismatch_names_hashes_and_cleans_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = make_pinned_zip(tmp_path, monkeypatch)
        monkeypatch.setattr(nucleide.data, "FGR15_SHA256", "0" * 64)
        with pytest.raises(RuntimeError, match="sha256 mismatch.*expected"):
            nucleide.data.fetch_fgr15(**fetch_kwargs(tmp_path, zip_path))
        assert not (tmp_path / "cache" / "fgr15_synthetic.zip").exists()

    def test_corrupt_cache_is_a_loud_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = make_pinned_zip(tmp_path, monkeypatch)
        dest = tmp_path / "cache"
        dest.mkdir()
        (dest / "fgr15_synthetic.zip").write_bytes(b"corrupt")
        with pytest.raises(RuntimeError, match="sha256 mismatch"):
            nucleide.data.fetch_fgr15(dest=dest, url=zip_path.as_uri())

    def test_network_failure_without_cache(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="failed to download.*No usable cached copy"):
            nucleide.data.fetch_fgr15(
                dest=tmp_path / "cache",
                url="http://127.0.0.1:59999/fgr15_synthetic.zip",
            )


class TestParseBinding:
    def test_synthetic_table_round_trip(self) -> None:
        text = table_text("4.1", "Synthetic Ground Surface (3 mm)", "Sv m2/Bq s", ROWS_41)
        table = nucleide.nuclei.parse_fgr15_table(text, 3)
        assert table["scenario"] == "ground_surface"
        assert table["units"] == "Sv m2/Bq s"
        assert set(table["coefficients"]) == {"H-3", "Sb-124n", "Ba-137m"}
        assert table["coefficients"]["Sb-124n"] == [
            1.10e-27,
            1.20e-27,
            1.30e-27,
            1.40e-27,
            1.50e-27,
            1.60e-27,
        ]

    def test_row_count_gate(self) -> None:
        text = table_text("4.1", "Synthetic Ground Surface (3 mm)", "Sv m2/Bq s", ROWS_41[:2])
        with pytest.raises(ValueError, match="expected 3"):
            nucleide.nuclei.parse_fgr15_table(text, 3)

    def test_malformed_row_is_loud(self) -> None:
        text = table_text("4.1", "Synthetic Ground Surface (3 mm)", "Sv m2/Bq s", ROWS_41)
        broken = text.replace("2.00e-27", "not-a-float")
        with pytest.raises(ValueError, match="malformed data row"):
            nucleide.nuclei.parse_fgr15_table(broken, 3)

    def test_age_index(self) -> None:
        assert nucleide.nuclei.fgr15_age_index("newborn") == 0
        assert nucleide.nuclei.fgr15_age_index("15-yr-old") == 4
        assert nucleide.nuclei.fgr15_age_index("adult") == 5
        with pytest.raises(ValueError, match="unknown FGR 15 age"):
            nucleide.nuclei.fgr15_age_index("middle-aged")


class TestFacade:
    def test_dose_rate_lookup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = fetch_kwargs(tmp_path, make_pinned_zip(tmp_path, monkeypatch))
        value, units = nucleide.nuclei.fgr15_dose_rate(
            "H-3", "ground_surface", "adult", expected_rows=3, **kwargs
        )
        assert value == 6.00e-27
        assert units == "Sv m2/Bq s"

        # Isomer letters and element-case tolerance; age alias.
        value, _ = nucleide.nuclei.fgr15_dose_rate(
            "SB-124N", "ground_surface", "1-yr-old", expected_rows=3, **kwargs
        )
        assert value == 1.20e-27
        # Table-number scenario alias; volume-table units wording.
        value, units = nucleide.nuclei.fgr15_dose_rate(
            "H-3", "4.6", "newborn", expected_rows=3, **kwargs
        )
        assert value == 7.00e-28
        assert units == "Sv m3 per Bq s"
        # Cached parse is shared: same zip, same scenario.
        table = nucleide.nuclei.load_fgr15_table("ground_surface", expected_rows=3, **kwargs)
        assert len(table["coefficients"]) == 3
        assert all(len(row) == 6 for row in table["coefficients"].values())

    def test_unknown_inputs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        kwargs = fetch_kwargs(tmp_path, make_pinned_zip(tmp_path, monkeypatch))
        with pytest.raises(ValueError, match="unknown FGR 15 scenario"):
            nucleide.nuclei.fgr15_dose_rate("H-3", "moon", "adult", expected_rows=3, **kwargs)
        with pytest.raises(ValueError, match="unknown FGR 15 age"):
            nucleide.nuclei.fgr15_dose_rate("H-3", "ground_surface", "2", expected_rows=3, **kwargs)
        with pytest.raises(ValueError, match="has no FGR 15 row"):
            nucleide.nuclei.fgr15_dose_rate(
                "U-235", "ground_surface", "adult", expected_rows=3, **kwargs
            )
        with pytest.raises(ValueError, match="unknown FGR 15 scenario"):
            nucleide.nuclei.load_fgr15_table("4.8", **kwargs)

    def test_zip_without_member(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        zip_path = tmp_path / "fgr15_synthetic.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("FGR15_Tables/Table_4_2.DAT", "junk")
        monkeypatch.setattr(
            nucleide.data, "FGR15_SHA256", hashlib.sha256(zip_path.read_bytes()).hexdigest()
        )
        with pytest.raises(ValueError, match="has no FGR15_Tables/Table_4_1.DAT member"):
            nucleide.nuclei.load_fgr15_table(
                "ground_surface", dest=tmp_path / "cache", url=zip_path.as_uri()
            )

    def test_scenario_declare_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Member named Table_4_1.DAT but titled "Table 4.2." — loud, not silent.
        zip_path = tmp_path / "fgr15_synthetic.zip"
        make_zip(zip_path, table_no_41="4.2")
        monkeypatch.setattr(
            nucleide.data, "FGR15_SHA256", hashlib.sha256(zip_path.read_bytes()).hexdigest()
        )
        with pytest.raises(ValueError, match="declares scenario 'soil_1cm'"):
            nucleide.nuclei.load_fgr15_table(
                "ground_surface",
                expected_rows=3,
                dest=tmp_path / "cache",
                url=zip_path.as_uri(),
            )

    def test_short_table_fails_row_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        zip_path = tmp_path / "fgr15_synthetic.zip"
        make_zip(zip_path, rows41=ROWS_41[:2])
        monkeypatch.setattr(
            nucleide.data, "FGR15_SHA256", hashlib.sha256(zip_path.read_bytes()).hexdigest()
        )
        with pytest.raises(ValueError, match="expected 1252"):
            nucleide.nuclei.load_fgr15_table(
                "ground_surface", dest=tmp_path / "cache", url=zip_path.as_uri()
            )
