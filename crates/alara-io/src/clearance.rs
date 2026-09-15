//! Clearance / waste-classification analytics over parsed inventories.
//!
//! Pure arithmetic on activation-code inventories: the **clearance index**
//!
//! ```text
//! CI = sum_i A_i / CL_i
//! ```
//!
//! over per-nuclide activities `A_i` and clearance levels `CL_i`, plus the
//! **sum-of-fractions rule** (IAEA Safety Standards Series No. RS-G-1.7 §5,
//! referenced by designation only): a material satisfies the clearance
//! screening criterion when `sum_i A_i / CL_i <= 1` (boundary included). The
//! methodology follows Sublet et al., *Nuclear Data Sheets* **139** (2017) 77
//! (FISPACT-II radiological indices); further indices from that paper stay
//! recorded, not implemented.
//!
//! This module is **screening arithmetic, not a compliance decision**: real
//! clearance requires the governing regulatory table, material bookkeeping,
//! and the national transposition of the underlying directive — see
//! [`ClearanceTable::eu_annex_vii`] for the provenance and limits of the
//! vendored default.
//!
//! Unit discipline is the caller's: `A_i` and `CL_i` must carry the same
//! basis. The vendored [`ClearanceTable::eu_annex_vii`] table is an
//! *activity-concentration* table (Bq/g), so inventories compared against it
//! must be massic activities (Bq/g), not total activities (Bq). Nuclide keys
//! are canonical [`NuclideId`]s resolved through the shared dialect
//! machinery — no second naming convention is introduced here.
//!
//! Inventories normally come from parsed [`crate::output::ResponseFrame`]
//! rows (see [`inventory_from_frame`]); any other source can supply the
//! same `&[(NuclideId, f64)]` shape directly.

use std::collections::BTreeMap;

use nucleide_nuclei::NuclideId;

use crate::error::{Error, Result};
use crate::output::{ResponseFrame, ResponseVar};

/// Vendored EU 2013/59/Euratom Annex VII Table A transcription (Bq/g).
///
/// Transcribed from the official legal text (EUR-Lex CELEX:32013L0059,
/// OJ L 13, 17.1.2014, Annex VII Table A, "Activity concentrations and
/// activities per unit mass of radionuclides ... for solid materials"),
/// accessed 2026-09-15. EU legal text is reusable with attribution under
/// Commission Implementing Decision 2011/833/EU. Columns: `GNDS name`,
/// `limit Bq/g`. Plain published facts (nuclide, number); never vendor IAEA
/// tables (RS-G-1.7, GSG-17) — reference them by designation only.
const EU_ANNEX_VII_A_TSV: &str = include_str!("data/eu_annex_vii_a.tsv");

/// Per-nuclide clearance-level table used by [`clearance_index`] and
/// [`sum_of_fractions`].
///
/// Keys are canonical [`NuclideId`]s; values are the clearance levels in the
/// unit basis the caller is working in (Bq total, Bq/g, ...). Build one with
/// [`ClearanceTable::insert`] or take the vendored
/// [`ClearanceTable::eu_annex_vii`] default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClearanceTable {
    limits: BTreeMap<NuclideId, f64>,
}

impl ClearanceTable {
    /// Empty caller-supplied table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace one nuclide's clearance level.
    ///
    /// `limit` must be finite and strictly positive. Errors:
    /// [`Error::BadClearanceValue`] on `limit <= 0.0` or non-finite input.
    pub fn insert(&mut self, nuclide: NuclideId, limit: f64) -> Result<()> {
        if !limit.is_finite() || limit <= 0.0 {
            return Err(Error::BadClearanceValue {
                nuclide: nuclide.to_name(),
                msg: format!("clearance limit must be finite and > 0, got {limit}"),
            });
        }
        self.limits.insert(nuclide, limit);
        Ok(())
    }

    /// Clearance level for `nuclide`, or `None` when the table carries none.
    pub fn get(&self, nuclide: NuclideId) -> Option<f64> {
        self.limits.get(&nuclide).copied()
    }

    /// Number of table entries.
    pub fn len(&self) -> usize {
        self.limits.len()
    }

    /// True when the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.limits.is_empty()
    }

    /// Iterate entries in canonical nucid order.
    pub fn iter(&self) -> impl Iterator<Item = (NuclideId, f64)> + '_ {
        self.limits.iter().map(|(nuc, limit)| (*nuc, *limit))
    }

    /// Vendored default: EU 2013/59/Euratom Annex VII Table A, activity
    /// concentrations for solid material in Bq/g.
    ///
    /// Provenance: official legal text transcribed 2026-09-15 from EUR-Lex
    /// CELEX:32013L0059 (Council Directive 2013/59/Euratom of 5 December
    /// 2013, OJ L 13, 17.1.2014, p. 1), Annex VII Table A, consolidated
    /// table version; reusable with attribution per Commission Implementing
    /// Decision (EU) 2011/833/EU. The transcription is a plain-facts table
    /// (`name`, `Bq/g`); see the module docs for the unit-basis contract.
    ///
    /// The table is embedded at build time and parsed lazily on first call.
    /// The transcription is committed data (like the `nuclei` static tables)
    /// pinned by the `eu_table_row_count` test, so this constructor treats a
    /// corrupt transcription as unreachable-in-practice and panics with the
    /// offending line; fallible callers use [`Self::try_eu_annex_vii`].
    pub fn eu_annex_vii() -> Self {
        static TABLE: std::sync::OnceLock<ClearanceTable> = std::sync::OnceLock::new();
        TABLE
            .get_or_init(|| {
                Self::try_eu_annex_vii().unwrap_or_else(|e| panic!("eu_annex_vii_a.tsv: {e}"))
            })
            .clone()
    }

    /// Fallible parse of the embedded EU Annex VII Table A transcription.
    ///
    /// Same data as [`Self::eu_annex_vii`] without the lazy cache: every
    /// malformed line (missing tab, unknown nuclide, bad limit) is a loud
    /// [`Error::Parse`] carrying the 1-based TSV line number.
    pub fn try_eu_annex_vii() -> Result<Self> {
        let mut table = ClearanceTable::new();
        for (index, line) in EU_ANNEX_VII_A_TSV.lines().enumerate() {
            let line_no = index + 1;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let (name, value_text) = trimmed.split_once('\t').ok_or_else(|| Error::Parse {
                line: line_no,
                msg: format!("expected `name<TAB>Bq/g`, found `{trimmed}`"),
            })?;
            let nuclide = NuclideId::from_name(name.trim()).map_err(|_| Error::Parse {
                line: line_no,
                msg: format!("unknown nuclide `{name}`"),
            })?;
            let limit: f64 = value_text.trim().parse().map_err(|_| Error::Parse {
                line: line_no,
                msg: format!("bad limit `{value_text}`"),
            })?;
            table.insert(nuclide, limit).map_err(|e| Error::Parse {
                line: line_no,
                msg: format!("{e}"),
            })?;
        }
        Ok(table)
    }
}

/// Sum-of-fractions screening outcome (RS-G-1.7 §5, referenced by designation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClearanceClass {
    /// `sum_i A_i / CL_i <= 1`: the inventory satisfies the sum-of-fractions
    /// screening criterion. The boundary `== 1` lands here ("does not
    /// exceed"), matching the exemption/clearance wording.
    Satisfied,
    /// `sum_i A_i / CL_i > 1`: the inventory does not satisfy the criterion.
    Exceeded,
}

impl std::fmt::Display for ClearanceClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Satisfied => "satisfied",
            Self::Exceeded => "exceeded",
        })
    }
}

/// Result of [`sum_of_fractions`]: the fraction sum, its screening class,
/// and the single dominant contributor (largest per-nuclide fraction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SumOfFractions {
    /// `sum_i A_i / CL_i` over the inventory.
    pub sum: f64,
    /// Screening classification of `sum`.
    pub class: ClearanceClass,
    /// Largest per-nuclide fraction `A_i / CL_i`.
    pub max_fraction: f64,
    /// Nuclide carrying [`Self::max_fraction`]; `None` for an empty inventory
    /// (where [`Self::max_fraction`] is `0.0`).
    pub max_nuclide: Option<NuclideId>,
}

/// Clearance index `CI = sum_i A_i / CL_i` over an inventory.
///
/// `inventory` holds `(nuclide, activity)` pairs; `table` carries the
/// clearance levels in the same unit basis (see the module contract). Every
/// inventory nuclide must have a table entry and every activity must be
/// finite and `>= 0` — violations are loud [`Error`]s, never silent skips.
/// Duplicate nuclides are the caller's problem to avoid; they sum like any
/// other pair.
pub fn clearance_index(inventory: &[(NuclideId, f64)], table: &ClearanceTable) -> Result<f64> {
    Ok(fraction_sum(inventory, table)?.sum)
}

/// Sum-of-fractions screening: the fraction sum, its class (`<= 1` passes,
/// boundary included), and the dominant nuclide.
///
/// Same input contract as [`clearance_index`].
pub fn sum_of_fractions(
    inventory: &[(NuclideId, f64)],
    table: &ClearanceTable,
) -> Result<SumOfFractions> {
    fraction_sum(inventory, table)
}

/// Shared accumulation for [`clearance_index`] / [`sum_of_fractions`].
fn fraction_sum(inventory: &[(NuclideId, f64)], table: &ClearanceTable) -> Result<SumOfFractions> {
    let mut sum = 0.0f64;
    let mut max_fraction = 0.0f64;
    let mut max_nuclide = None;
    for (nuclide, activity) in inventory {
        if !activity.is_finite() || *activity < 0.0 {
            return Err(Error::BadClearanceValue {
                nuclide: nuclide.to_name(),
                msg: format!("activity must be finite and >= 0, got {activity}"),
            });
        }
        let limit = table
            .get(*nuclide)
            .ok_or_else(|| Error::MissingClearanceLimit {
                nuclide: nuclide.to_name(),
                table_len: table.len(),
            })?;
        let fraction = activity / limit;
        sum += fraction;
        if fraction > max_fraction {
            max_fraction = fraction;
            max_nuclide = Some(*nuclide);
        }
    }
    // Finite inputs can still overflow the sum (huge activity over a tiny
    // limit): an infinite total is a loud error, never `Ok(inf)`.
    if !sum.is_finite() {
        return Err(Error::BadClearanceValue {
            nuclide: "total".to_string(),
            msg: format!(
                "fraction sum overflowed to {sum} over {} entries",
                inventory.len()
            ),
        });
    }
    // `inventory` empty: no dominant nuclide (`max_fraction` is 0.0).
    let class = if sum <= 1.0 {
        ClearanceClass::Satisfied
    } else {
        ClearanceClass::Exceeded
    };
    Ok(SumOfFractions {
        sum,
        class,
        max_fraction,
        max_nuclide,
    })
}

/// Extract `(nuclide, activity)` inventory pairs from one cooling time of a
/// parsed [`ResponseFrame`].
///
/// Keeps [`ResponseVar::SpecificActivity`] rows at `time_s` and drops `total`
/// aggregate rows. Rows whose nuclide name does not resolve through the
/// shared dialect machinery are a loud [`Error::Parse`] (parser-validated
/// frames cannot produce them; hand-built frames can).
///
/// Unit note: the returned activities are in the frame's own activity unit
/// (ALARA `Bq/cm3` vs FISPACT-II `Bq`, see `var_unit`); the caller must pick
/// a [`ClearanceTable`] with the matching basis.
pub fn inventory_from_frame(frame: &ResponseFrame, time_s: f64) -> Result<Vec<(NuclideId, f64)>> {
    let mut pairs = Vec::new();
    for row in &frame.rows {
        if row.variable != ResponseVar::SpecificActivity || row.time_s != time_s {
            continue;
        }
        if row.is_total() {
            continue;
        }
        let nuclide = row.nuclide_id().map_err(|e| Error::Parse {
            line: 0,
            msg: format!("frame row nuclide `{}` does not resolve: {e}", row.nuclide),
        })?;
        pairs.push((nuclide, row.value));
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built three-nuclide table (synthetic, round numbers).
    fn toy_table() -> ClearanceTable {
        let mut table = ClearanceTable::new();
        table
            .insert(NuclideId::from_name("Co60").unwrap(), 10.0)
            .unwrap();
        table
            .insert(NuclideId::from_name("H3").unwrap(), 5.0)
            .unwrap();
        table
            .insert(NuclideId::from_name("Fe55").unwrap(), 2.0)
            .unwrap();
        table
    }

    #[test]
    fn hand_computed_ci_vectors_at_exact_equality() {
        let table = toy_table();
        // Single nuclide at its own limit: CI == 1 exactly.
        let inv = [(NuclideId::from_name("Co60").unwrap(), 10.0)];
        assert_eq!(clearance_index(&inv, &table).unwrap(), 1.0);
        // Two nuclides at half limit each: 0.5 + 0.5 == 1 exactly.
        let inv = [
            (NuclideId::from_name("Co60").unwrap(), 5.0),
            (NuclideId::from_name("H3").unwrap(), 2.5),
        ];
        assert_eq!(clearance_index(&inv, &table).unwrap(), 1.0);
        // Mixed vector: 10/10 + 5/5 + 4/2 == 4 exactly.
        let inv = [
            (NuclideId::from_name("Co60").unwrap(), 10.0),
            (NuclideId::from_name("H3").unwrap(), 5.0),
            (NuclideId::from_name("Fe55").unwrap(), 4.0),
        ];
        assert_eq!(clearance_index(&inv, &table).unwrap(), 4.0);
        // Zero activity contributes nothing.
        let inv = [(NuclideId::from_name("Fe55").unwrap(), 0.0)];
        assert_eq!(clearance_index(&inv, &table).unwrap(), 0.0);
        // Empty inventory: CI == 0.
        assert_eq!(clearance_index(&[], &table).unwrap(), 0.0);
    }

    #[test]
    fn sum_of_fractions_boundary_probes_both_sides() {
        let table = toy_table();
        // Exactly == 1 on the boundary: satisfied (boundary included).
        let inv = [(NuclideId::from_name("Co60").unwrap(), 10.0)];
        let out = sum_of_fractions(&inv, &table).unwrap();
        assert_eq!(out.sum, 1.0);
        assert_eq!(out.class, ClearanceClass::Satisfied);
        assert_eq!(out.max_fraction, 1.0);
        assert_eq!(out.max_nuclide, Some(NuclideId::from_name("Co60").unwrap()));

        // One ulp above the boundary: exceeded.
        let inv = [(NuclideId::from_name("Co60").unwrap(), 10.000000000000002)];
        let out = sum_of_fractions(&inv, &table).unwrap();
        assert!(out.sum > 1.0);
        assert_eq!(out.class, ClearanceClass::Exceeded);

        // One ulp below the boundary: satisfied.
        let inv = [(NuclideId::from_name("Co60").unwrap(), 9.999999999999998)];
        let out = sum_of_fractions(&inv, &table).unwrap();
        assert!(out.sum < 1.0);
        assert_eq!(out.class, ClearanceClass::Satisfied);

        // Fraction accumulation both sides: 0.4 + 0.5 = 0.9 vs 0.6 + 0.5 = 1.1.
        let below = [
            (NuclideId::from_name("Co60").unwrap(), 4.0),
            (NuclideId::from_name("H3").unwrap(), 2.5),
        ];
        assert_eq!(
            sum_of_fractions(&below, &table).unwrap().class,
            ClearanceClass::Satisfied
        );
        let above = [
            (NuclideId::from_name("Co60").unwrap(), 6.0),
            (NuclideId::from_name("H3").unwrap(), 2.5),
        ];
        let out = sum_of_fractions(&above, &table).unwrap();
        assert_eq!(out.class, ClearanceClass::Exceeded);
        assert_eq!(out.max_nuclide, Some(NuclideId::from_name("Co60").unwrap()));
        assert!(out.max_fraction > out.sum - out.max_fraction);
    }

    #[test]
    fn eu_table_loads_with_pinned_entry_count() {
        // Pins the committed transcription: any edit to the TSV must update
        // this count deliberately, which keeps `eu_annex_vii()` honest about
        // its unreachable-in-practice panic.
        let table = ClearanceTable::try_eu_annex_vii().unwrap();
        assert_eq!(table.len(), 260);
        assert_eq!(ClearanceTable::eu_annex_vii().len(), 260);
        // Spot check: Co-60 Table A value is 0.1 Bq/g.
        let co60 = table.get(NuclideId::from_name("Co60").unwrap()).unwrap();
        assert_eq!(co60, 0.1);
    }

    #[test]
    fn overflowing_fraction_sum_is_a_loud_error() {
        // Finite inputs whose sum overflows: loud, never Ok(inf).
        // f64::MAX over a 1e-308 limit overflows the single fraction to inf.
        let mut table = ClearanceTable::new();
        table
            .insert(NuclideId::from_name("Co60").unwrap(), 1e-308)
            .unwrap();
        let inv = [(NuclideId::from_name("Co60").unwrap(), f64::MAX)];
        match clearance_index(&inv, &table) {
            Err(Error::BadClearanceValue { nuclide, msg }) => {
                assert_eq!(nuclide, "total");
                assert!(msg.contains("overflowed"), "msg was `{msg}`");
            }
            other => panic!("expected overflow error, got {other:?}"),
        }
    }

    #[test]
    fn missing_limit_and_bad_values_are_loud_errors() {
        let table = toy_table();
        let unknown = [(NuclideId::from_name("Mn54").unwrap(), 1.0)];
        match clearance_index(&unknown, &table) {
            Err(Error::MissingClearanceLimit { nuclide, table_len }) => {
                assert_eq!(nuclide, "Mn54");
                assert_eq!(table_len, 3);
            }
            other => panic!("expected MissingClearanceLimit, got {other:?}"),
        }
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let inv = [(NuclideId::from_name("Co60").unwrap(), bad)];
            assert!(
                matches!(
                    clearance_index(&inv, &table),
                    Err(Error::BadClearanceValue { .. })
                ),
                "activity {bad} should fail"
            );
        }
        let mut bad_table = ClearanceTable::new();
        for bad in [0.0, -2.0, f64::NAN] {
            assert!(
                matches!(
                    bad_table.insert(NuclideId::from_name("Co60").unwrap(), bad),
                    Err(Error::BadClearanceValue { .. })
                ),
                "limit {bad} should fail"
            );
        }
        assert!(bad_table.is_empty());
        bad_table
            .insert(NuclideId::from_name("Co60").unwrap(), 1.0)
            .unwrap();
        assert_eq!(bad_table.len(), 1);
    }

    #[test]
    fn frame_inventory_extracts_specific_activity_at_one_time() {
        let text = "*** Specific Activity [Bq/cm3] ***\n\
                     Interval #1 (Zone: inner) :\n\
                     isotope  t_1/2(s)   shutdown      1 d\n\
                     =====\n\
                     co-60 \t1.6636e+08  4.0000e+01  2.0000e+01\n\
                     h-3 \t3.8881e+08   1.0000e+01  5.0000e+00\n\
                     =====\n                     total   0           5.0000e+01  2.5000e+01\n";
        let frame = ResponseFrame::parse(text, "r").unwrap();
        let pairs = inventory_from_frame(&frame, 86_400.0).unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], (NuclideId::from_name("Co60").unwrap(), 20.0));
        assert_eq!(pairs[1], (NuclideId::from_name("H3").unwrap(), 5.0));
        // Shutdown time: other values.
        let pairs = inventory_from_frame(&frame, 0.0).unwrap();
        assert_eq!(pairs[0].1, 40.0);
        // A time with no rows yields an empty (valid) inventory.
        assert!(inventory_from_frame(&frame, -1.0).unwrap().is_empty());
    }

    #[test]
    fn eu_annex_vii_default_table_loads_and_covers_classic_clearance_nuclides() {
        let table = ClearanceTable::eu_annex_vii();
        assert!(!table.is_empty());
        // Spot values transcribed from the directive (Bq/g, solid materials):
        // H-3 -> 100, C-14 -> 1, Co-60 -> 0.1, Cs-137 -> 0.1, Sr-90 -> 1,
        // Pu-239 -> 0.1; Part 2: U-238 -> 1, Th-232 -> 1, K-40 -> 10.
        for (name, expected) in [
            ("H3", 100.0),
            ("C14", 1.0),
            ("Co60", 0.1),
            ("Cs137", 0.1),
            ("Sr90", 1.0),
            ("Pu239", 0.1),
            ("U238", 1.0),
            ("Th232", 1.0),
            ("K40", 10.0),
        ] {
            let nuc = NuclideId::from_name(name).unwrap();
            assert_eq!(
                table.get(nuc),
                Some(expected),
                "{name} limit should be {expected} Bq/g"
            );
        }
        // All entries positive; canonical-keyed.
        assert!(table.iter().all(|(_, limit)| limit > 0.0));
        assert_eq!(table, ClearanceTable::eu_annex_vii());
    }
}
