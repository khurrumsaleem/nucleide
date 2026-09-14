"""Nuclide identifiers, nuclear data, and reaction names (backed by the `nucleide-nuclei` crate)."""

import zipfile
from pathlib import Path
from typing import Any

from nucleide._internal import (
    Nuclide,
    Particle,
    armi_to_nucid,
    atomic_mass,
    decay_branch_fraction,
    decay_branches,
    decay_constant,
    decay_energy,
    dose_f1,
    dose_factor,
    dose_lung_model,
    fgr15_age_index,
    fission_yield,
    fission_yields,
    from_zaid,
    half_life,
    mcc3_to_nucid,
    natural_abundance,
    normalize_nuclide,
    nucid_to_armi,
    parse_fgr15_table,
    particle_is_heavy_ion,
    particle_is_hydrogen,
    particle_is_valid,
    particle_is_valid_pdc,
    q_value_alpha,
    q_value_capture,
    rxname_child,
    rxname_doc,
    rxname_id,
    rxname_id_from_nucdelta,
    rxname_label,
    rxname_mt,
    rxname_name,
    rxname_parent,
    rxname_reaction,
    scattering_length,
    simple_xs,
)
from nucleide.data import fetch_fgr15

#: EPA FGR 15 scenario keys mapped to their zip member names, in table order.
_FGR15_TABLES = {
    "ground_surface": "FGR15_Tables/Table_4_1.DAT",
    "soil_1cm": "FGR15_Tables/Table_4_2.DAT",
    "soil_5cm": "FGR15_Tables/Table_4_3.DAT",
    "soil_15cm": "FGR15_Tables/Table_4_4.DAT",
    "soil_infinite": "FGR15_Tables/Table_4_5.DAT",
    "air_submersion": "FGR15_Tables/Table_4_6.DAT",
    "water_immersion": "FGR15_Tables/Table_4_7.DAT",
}
_FGR15_TABLE_NUMBERS = {f"4.{i}": name for i, name in enumerate(_FGR15_TABLES, start=1)}
_FGR15_TABLE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}


def _fgr15_scenario_key(scenario: str) -> str:
    """Normalize a scenario key: snake name or EPA table number."""
    key = scenario.strip().lower().replace("-", "_").replace(" ", "_")
    key = key.removeprefix("table_")
    return _FGR15_TABLE_NUMBERS.get(key, key)


def _fgr15_name_key(nuclide: str) -> str:
    """Case-canonical FGR 15 name spelling (``sb-124N`` -> ``Sb-124n``)."""
    symbol, _, rest = nuclide.partition("-")
    return f"{symbol[:1].upper()}{symbol[1:].lower()}-{rest.lower()}"


def load_fgr15_table(
    scenario: str,
    *,
    expected_rows: int = 1252,
    dest: str | Path | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    """Parse one EPA FGR 15 scenario table from the cached coefficient zip.

    ``scenario`` is one of ``ground_surface``, ``soil_1cm``, ``soil_5cm``,
    ``soil_15cm``, ``soil_infinite``, ``air_submersion``, ``water_immersion``
    (or the EPA table numbers ``4.1``–``4.7``). The zip is fetched with
    `nucleide.data.fetch_fgr15` (``dest``/``url`` pass through; verified
    copies are cached per zip path). Returns ``{"scenario": str, "units":
    str, "coefficients": {name: [6 floats]}}`` with coefficient lists in
    canonical age order (newborn, 1-yr, 5-yr, 10-yr, 15-yr, adult) and FGR 15
    name spellings (``H-3``, ``Ba-137m``, ``Sb-124n``). ``expected_rows`` is
    the exact nuclide-row count the table must hold (1,252 in the published
    EPA tables; the gate is loud). Raises ``ValueError`` for unknown
    scenarios and when the cached zip is missing, hash-mismatched, or
    structurally invalid.
    """
    key = _fgr15_scenario_key(scenario)
    member = _FGR15_TABLES.get(key)
    if member is None:
        supported = ", ".join(_FGR15_TABLES)
        raise ValueError(f"unknown FGR 15 scenario {scenario!r} (supported: {supported})")
    zip_path = fetch_fgr15(dest=dest, url=url)
    cache_key = (zip_path, key)
    if cache_key not in _FGR15_TABLE_CACHE:
        with zipfile.ZipFile(zip_path) as archive:
            try:
                text = archive.read(member).decode("utf-8")
            except KeyError as exc:
                raise ValueError(
                    f"{zip_path} has no {member} member; is it the FGR 15 data zip?"
                ) from exc
        table = parse_fgr15_table(text, expected_rows)
        if table["scenario"] != key:
            raise ValueError(f"{member} declares scenario {table['scenario']!r}, expected {key!r}")
        _FGR15_TABLE_CACHE[cache_key] = table
    return dict(_FGR15_TABLE_CACHE[cache_key])


def fgr15_dose_rate(
    nuclide: str,
    scenario: str,
    age: str,
    *,
    expected_rows: int = 1252,
    dest: str | Path | None = None,
    url: str | None = None,
) -> tuple[float, str]:
    """Effective dose (rate) coefficient for external exposure, from EPA FGR 15.

    Returns ``(coefficient, units)`` where ``units`` is the table's own
    header string (``Sv m2/Bq s`` for the surface table, ``Sv m3 per Bq s``
    for the soil-volume and immersion tables). ``nuclide`` uses FGR 15
    spelling (``Cs-137``, ``Ba-137m``, ``Sb-124n``; element-case tolerant);
    ``age`` is one of ``newborn``, ``1``, ``5``, ``10``, ``15``, ``adult``
    (common variants like ``1-yr-old`` accepted); ``scenario`` is any key
    accepted by `load_fgr15_table`. Raises ``ValueError`` for unknown
    scenario/age and when the nuclide has no row in the scenario table.

    FGR 15 is *external exposure only*. The composition-level
    `nucleide.material.dose_per_g` (HNF-5636/PyNE ingestion/inhalation/air/
    soil factors) stays the screening default for material dose; the two are
    complementary and neither silently takes precedence — compare units
    before combining. Screening-level only — not for safety decisions.
    """
    table = load_fgr15_table(scenario, expected_rows=expected_rows, dest=dest, url=url)
    index = fgr15_age_index(age)
    coefficients = table["coefficients"]
    row = coefficients.get(nuclide)
    if row is None:
        row = coefficients.get(_fgr15_name_key(nuclide))
    if row is None:
        raise ValueError(f"nuclide {nuclide!r} has no FGR 15 row in the {table['scenario']} table")
    return row[index], table["units"]


__all__ = [
    "Nuclide",
    "Particle",
    "from_zaid",
    "armi_to_nucid",
    "nucid_to_armi",
    "mcc3_to_nucid",
    "normalize_nuclide",
    "atomic_mass",
    "natural_abundance",
    "half_life",
    "decay_constant",
    "q_value_capture",
    "q_value_alpha",
    "rxname_id",
    "rxname_name",
    "rxname_mt",
    "rxname_label",
    "rxname_doc",
    "rxname_reaction",
    "rxname_id_from_nucdelta",
    "rxname_child",
    "rxname_parent",
    "particle_is_valid",
    "particle_is_valid_pdc",
    "particle_is_hydrogen",
    "particle_is_heavy_ion",
    "simple_xs",
    "scattering_length",
    "decay_energy",
    "decay_branches",
    "decay_branch_fraction",
    "fission_yields",
    "fission_yield",
    "dose_factor",
    "dose_f1",
    "dose_lung_model",
    "parse_fgr15_table",
    "fgr15_age_index",
    "load_fgr15_table",
    "fgr15_dose_rate",
]
