//! FISPACT-II clearance-block reader: the wide inventory table printed with
//! the `HAZARDS` + `CLEAR` keywords.
//!
//! Grammar source: real FISPACT-II main-output inventory sections with the
//! clearance index column, as shipped in the `fispact/workshops` reference
//! outputs (Apache-2.0), e.g.
//! `2020/exercises/files/basic/compressxs/ref/inventorywithcompress_ref.out`,
//! controlled by the `CLEAR` keyword (see `inventorywithcompress_ref.i`).
//! That on-disk sample is the citable grammar this reader mirrors; the
//! synthetic fixture `fixtures/fispact/clearance.out` exercises every branch.
//!
//! Recognized layout:
//! - Time-step header: a line containing `TIME INTERVAL <n>` followed by
//!   either `TIME IS <value> <unit>` (irradiation) or
//!   `COOLING TIME IS <value> <unit> [...]` (cooling). The *first*
//!   `<value> <unit>` pair after the `IS` tag gives the step time in
//!   seconds (`SECS` primary form in the reference outputs; `s m h d w y`
//!   names also accepted, `y = 365 d` matching the crate convention).
//! - Two-line column header: a line starting with `NUCLIDE` that carries the
//!   `CLEARANCE` and `HALF LIFE` columns (the wide hazards layout with
//!   `ATOMS GRAMS Bq b-Energy a-Energy g-Energy DOSE RATE INGESTION
//!   INHALATION CLEARANCE Bq/A2 HALF LIFE`), then one units line (`kW ...`)
//!   that is skipped.
//! - Data rows: `<symbol> <mass>[isomer] [flags] <11 floats> <float|Stable>`.
//!   Flags are `#` (stable), `>` (present before irradiation), `&` (gamma
//!   spectrum approximately calculated), `?` (convergence not reached), kept
//!   verbatim. The 11 floats are ATOMS, GRAMS, Bq, b-Energy, a-Energy,
//!   g-Energy, DOSE RATE, INGESTION, INHALATION, CLEARANCE INDEX, Bq/A2
//!   ratio; the last column is the half-life in seconds or `Stable`
//!   (stored as `-1.0`, matching the ALARA output convention).
//! - The table ends at `TOTAL NUMBER OF NUCLIDES PRINTED IN INVENTORY` (or
//!   any line that is not a data row).
//!
//! Everything else in the file (preamble prose, totals blocks, dominant-
//! nuclides tables, spectra) is skipped. Nuclide names are normalized to the
//! shared dialect spelling (`V 55` -> `V-55`, `Rb 90m` -> `Rb-90m`) through
//! the `nuclei` dialect machinery — no second naming convention.

use std::path::Path;

use nucleide_nuclei::dialects::normalize_nuclide_name;

use crate::error::{Error, Result};

/// One nuclide row of a parsed clearance-bearing inventory table.
#[derive(Debug, Clone, PartialEq)]
pub struct ClearanceRow {
    /// 1-based `TIME INTERVAL` number of the source step.
    pub interval: i64,
    /// Step time in seconds (first `<value> <unit>` after the `IS` tag).
    pub time_s: f64,
    /// Verbatim step-time tag payload, e.g. `TIME IS 0.0000E+00 SECS` or
    /// `COOLING TIME IS 8.6400E+04 SECS OR 1.0000E+00 DAYS`.
    pub time_label: String,
    /// True for `COOLING TIME IS` steps, false for `TIME IS` irradiation steps.
    pub cooling: bool,
    /// Nuclide name in the shared dialect spelling (`Co-60`, `Rb-86m`).
    pub nuclide: String,
    /// Verbatim flag characters (`#`, `>`, `&`, `?`); empty when none.
    pub flags: String,
    /// Activity column in becquerel.
    pub activity_bq: f64,
    /// FISPACT-II clearance index column for this nuclide (Bq divided by the
    /// code's internal clearance library value — code-side, not a table).
    pub clearance_index: f64,
    /// Half-life in seconds; `-1.0` marks `Stable` rows.
    pub half_life_s: f64,
}

impl ClearanceRow {
    /// Resolve [`Self::nuclide`] to a canonical [`nucleide_nuclei::NuclideId`].
    pub fn nuclide_id(
        &self,
    ) -> std::result::Result<nucleide_nuclei::NuclideId, nucleide_nuclei::dialects::DialectError>
    {
        normalize_nuclide_name(&self.nuclide)
    }

    /// True when the row carries the `#` (stable) flag or a `Stable`
    /// half-life spelling.
    pub fn is_stable(&self) -> bool {
        self.half_life_s == -1.0
    }
}

/// All clearance-table rows parsed from one FISPACT-II output listing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClearanceScan {
    /// Rows in file order, one per (time step, nuclide).
    pub rows: Vec<ClearanceRow>,
}

impl ClearanceScan {
    /// Keep rows of one time step selected by seconds (exact float match,
    /// as produced by this parser's unit conversion).
    pub fn at_time(&self, time_s: f64) -> Self {
        Self {
            rows: self
                .rows
                .iter()
                .filter(|row| row.time_s == time_s)
                .cloned()
                .collect(),
        }
    }

    /// Sum of the per-nuclide clearance-index column (FISPACT-II's own
    /// `A_i / CL_i` values summed over the scan).
    pub fn total_clearance_index(&self) -> f64 {
        self.rows.iter().map(|row| row.clearance_index).sum()
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// True when no rows were parsed.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Parse FISPACT-II clearance-block text (the wide `CLEAR`-keyword inventory
/// table) into a [`ClearanceScan`].
///
/// Blank input is an error; every malformed time header or data row inside a
/// recognized clearance table is a loud 1-based [`Error::Parse`]. Files with
/// no clearance table at all yield an empty scan (callers decide whether that
/// is an error).
pub fn parse_clearance(text: &str) -> Result<ClearanceScan> {
    if text.trim().is_empty() {
        return Err(Error::Parse {
            line: 1,
            msg: "empty FISPACT-II output".to_string(),
        });
    }
    let mut scan = ClearanceScan::default();
    let mut step: Option<(i64, f64, String, bool)> = None;
    let mut in_table = false;
    let mut skip_units = false;

    for (index, raw) in text.lines().enumerate() {
        let line_no = index + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            continue;
        }
        if is_time_header(line) {
            step = Some(parse_time_header(line, line_no)?);
            in_table = false;
            skip_units = false;
            continue;
        }
        if is_column_header(line) {
            in_table = step.is_some();
            skip_units = in_table;
            continue;
        }
        if skip_units {
            // The units line (`kW ... seconds`) directly follows the column header.
            skip_units = false;
            continue;
        }
        if !in_table {
            continue;
        }
        if looks_like_data_row(line) {
            let Some(current) = step.clone() else {
                continue;
            };
            scan.rows.push(parse_data_row(line, current, line_no)?);
        } else {
            in_table = false;
        }
    }
    Ok(scan)
}

/// Read and parse a FISPACT-II clearance block from disk.
pub fn from_file(path: impl AsRef<Path>) -> Result<ClearanceScan> {
    let text = std::fs::read_to_string(path.as_ref())?;
    parse_clearance(&text)
}

/// True when `line` opens a `* * * TIME INTERVAL n * * *` header.
fn is_time_header(line: &str) -> bool {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    tokens.contains(&"INTERVAL") && tokens.contains(&"TIME")
}

/// Parse the TIME INTERVAL header into `(interval, time_s, label, cooling)`.
fn parse_time_header(line: &str, line_no: usize) -> Result<(i64, f64, String, bool)> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let interval_pos = tokens
        .iter()
        .position(|t| *t == "INTERVAL")
        .ok_or_else(|| Error::Parse {
            line: line_no,
            msg: format!("expected `TIME INTERVAL n`, found `{line}`"),
        })?;
    let interval: i64 = tokens
        .get(interval_pos + 1)
        .and_then(|t| t.parse().ok())
        .ok_or_else(|| Error::Parse {
            line: line_no,
            msg: format!("expected interval number after `TIME INTERVAL`, found `{line}`"),
        })?;
    // The step-time tag is the first `IS` after `INTERVAL`; later `IS`
    // tokens (`ELAPSED TIME IS`, `FLUX AMP IS`) are not step times.
    let is_pos = tokens[interval_pos..]
        .iter()
        .position(|t| *t == "IS")
        .map(|pos| interval_pos + pos)
        .ok_or_else(|| Error::Parse {
            line: line_no,
            msg: format!("expected `TIME IS` / `COOLING TIME IS` tag, found `{line}`"),
        })?;
    let cooling = tokens[interval_pos..is_pos].contains(&"COOLING");
    let (value_text, unit) = tokens
        .get(is_pos + 1)
        .and_then(|v| tokens.get(is_pos + 2).map(|u| (*v, *u)))
        .ok_or_else(|| Error::Parse {
            line: line_no,
            msg: format!("expected `<value> <unit>` after `IS` tag, found `{line}`"),
        })?;
    let value: f64 = value_text.parse().map_err(|_| Error::Parse {
        line: line_no,
        msg: format!("expected step time, found `{value_text}`"),
    })?;
    let factor = time_unit_to_seconds(unit).ok_or_else(|| Error::Parse {
        line: line_no,
        msg: format!("unknown step-time unit `{unit}`"),
    })?;
    let tag_start = if cooling {
        // `COOLING TIME IS ...`
        is_pos.saturating_sub(2)
    } else {
        // `TIME IS ...`
        is_pos.saturating_sub(1)
    };
    // Keep the full tag payload, including `OR <value> <unit>` alternative
    // spellings, up to the next `* * *` marker or `ELAPSED` continuation.
    let mut end = is_pos + 2;
    while end + 2 < tokens.len() && tokens[end + 1] == "OR" {
        end += 3;
    }
    let label = tokens[tag_start..=end].join(" ");
    Ok((interval, value * factor, label, cooling))
}

/// True for the wide inventory column header carrying the clearance column.
fn is_column_header(line: &str) -> bool {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    tokens.first() == Some(&"NUCLIDE") && tokens.contains(&"CLEARANCE") && tokens.contains(&"HALF")
}

/// True when `line` looks like a nuclide data-row attempt (element symbol +
/// mass-like token), as opposed to structural prose.
fn looks_like_data_row(line: &str) -> bool {
    let mut tokens = line.split_whitespace();
    let Some(sym) = tokens.next() else {
        return false;
    };
    let Some(mass) = tokens.next() else {
        return false;
    };
    let symbol_ok = sym.len() <= 2 && sym.chars().all(|c| c.is_ascii_alphabetic());
    let mass_ok = {
        let mut chars = mass.chars();
        let digits = chars.by_ref().take_while(|c| c.is_ascii_digit()).count();
        digits > 0 && chars.all(|c| c.is_ascii_alphabetic())
    };
    symbol_ok && mass_ok
}

/// True for flag tokens made only of `#>?&` characters.
fn is_flags_token(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|c| matches!(c, '#' | '>' | '?' | '&'))
}

/// Parse one data row into a [`ClearanceRow`].
fn parse_data_row(
    line: &str,
    (interval, time_s, time_label, cooling): (i64, f64, String, bool),
    line_no: usize,
) -> Result<ClearanceRow> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let sym = tokens[0];
    let mass = tokens[1];
    let mut flags = String::new();
    let mut idx = 2;
    while idx < tokens.len() && is_flags_token(tokens[idx]) {
        flags.push_str(tokens[idx]);
        idx += 1;
    }
    let values_end = idx + 11;
    if tokens.len() != values_end + 1 {
        return Err(Error::Parse {
            line: line_no,
            msg: format!("expected `nuclide flags? 11 floats halflife|Stable`, found `{line}`"),
        });
    }
    let mut floats = [0.0f64; 11];
    for (slot, token) in floats.iter_mut().zip(&tokens[idx..values_end]) {
        *slot = token.parse().map_err(|_| Error::Parse {
            line: line_no,
            msg: format!("expected float, found `{token}`"),
        })?;
    }
    let half_life_s = match tokens[values_end] {
        "Stable" => -1.0,
        text => text.parse().map_err(|_| Error::Parse {
            line: line_no,
            msg: format!("expected half-life or `Stable`, found `{text}`"),
        })?,
    };
    // Normalize the two-token nuclide spelling through the shared dialect
    // machinery (validates element and mass); isomer letter rides on `mass`.
    let nuclide = format!("{sym}-{mass}");
    normalize_nuclide_name(&nuclide).map_err(|_| Error::Parse {
        line: line_no,
        msg: format!("unknown nuclide `{nuclide}`"),
    })?;
    Ok(ClearanceRow {
        interval,
        time_s,
        time_label,
        cooling,
        nuclide,
        flags,
        activity_bq: floats[2],
        clearance_index: floats[9],
        half_life_s,
    })
}

/// Conversion factor to seconds for a FISPACT step-time unit.
///
/// Single letters `s m h d w y` plus common long names (case-insensitive),
/// with `w = 7 d` and `y = 365 d` (matching the crate's inventory parser).
fn time_unit_to_seconds(unit: &str) -> Option<f64> {
    match unit.trim().to_ascii_lowercase().as_str() {
        "s" | "sec" | "secs" | "second" | "seconds" => Some(1.0),
        "m" | "min" | "mins" | "minute" | "minutes" => Some(60.0),
        "h" | "hr" | "hrs" | "hour" | "hours" => Some(3_600.0),
        "d" | "day" | "days" => Some(86_400.0),
        "w" | "week" | "weeks" => Some(604_800.0),
        "y" | "yr" | "yrs" | "year" | "years" => Some(31_536_000.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nucleide_nuclei::NuclideId;

    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/fispact/clearance.out"
    );

    #[test]
    fn rejects_empty() {
        assert!(parse_clearance("  \n").is_err());
    }

    #[test]
    fn parses_fixture_rows_with_flags_isomers_and_stable() {
        let scan = from_file(FIXTURE).unwrap();
        // 2 steps x (4 + 3) rows.
        assert_eq!(scan.len(), 7);
        assert!(!scan.is_empty());

        // Irradiation step.
        let step1 = scan.at_time(0.0);
        assert_eq!(step1.len(), 4);
        assert!(step1.rows.iter().all(|row| !row.cooling));
        assert!(step1.rows.iter().all(|row| row.interval == 1));
        assert_eq!(step1.rows[0].nuclide, "V-55");
        assert_eq!(step1.rows[0].flags, ">");
        assert!(step1.rows[0].is_stable());
        assert_eq!(step1.rows[0].half_life_s, -1.0);
        assert_eq!(step1.rows[0].activity_bq, 0.0);
        assert_eq!(step1.rows[0].time_label, "TIME IS 0.0000E+00 SECS");

        let co60 = &step1.rows[2];
        assert_eq!(co60.nuclide, "Co-60");
        assert_eq!(co60.activity_bq, 4.0e6);
        assert_eq!(co60.clearance_index, 1.0e6);
        assert_eq!(co60.half_life_s, 1.6636e8);
        assert_eq!(
            co60.nuclide_id().unwrap(),
            NuclideId::from_name("Co60").unwrap()
        );

        let rb86m = &step1.rows[3];
        assert_eq!(rb86m.nuclide, "Rb-86m");
        assert_eq!(rb86m.flags, "&");
        assert_eq!(rb86m.clearance_index, 4.0e9);
        assert_eq!(
            rb86m.nuclide_id().unwrap().state(),
            1,
            "isomer letter rides the mass token"
        );

        // Multi-flag verbatim order preserved.
        assert_eq!(step1.rows[1].flags, "#>");

        // Cooling step: 365-day-independent SECS primary value, cooling flag.
        let step2 = scan.at_time(86_400.0);
        assert_eq!(step2.len(), 3);
        assert!(step2.rows.iter().all(|row| row.cooling));
        assert!(step2.rows.iter().all(|row| row.interval == 2));
        assert_eq!(
            step2.rows[1].time_label,
            "COOLING TIME IS 8.6400E+04 SECS OR 1.0000E+00 DAYS"
        );
        assert_eq!(step2.rows[1].activity_bq, 3.8e6);

        // Hand-computed column totals across both steps.
        assert_eq!(scan.total_clearance_index(), 1.0e6 + 4.0e9 + 9.5e5 + 2.0e7);
    }

    #[test]
    fn no_clearance_table_yields_empty_scan() {
        let scan = parse_clearance("FISPACT-II RUN\nSOME PROSE LINE\n").unwrap();
        assert!(scan.is_empty());
    }

    #[test]
    fn malformed_rows_and_headers_are_loud() {
        let header = "1 * * * TIME INTERVAL 1 * * * TIME IS 1.0 SECS\n\
                      \n  NUCLIDE ATOMS CLEARANCE HALF LIFE\n kW seconds\n";
        // Bad float in the 11-float block.
        let bad =
            format!("{header}Co 60 > 1.0 1.0 notafloat 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n");
        match parse_clearance(&bad) {
            Err(Error::Parse { line, msg }) => {
                assert_eq!(line, 5);
                assert!(msg.contains("notafloat"), "msg was `{msg}`");
            }
            other => panic!("expected parse error, got {other:?}"),
        }
        // Missing half-life column.
        let short = format!("{header}Co 60 > 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n");
        assert!(matches!(parse_clearance(&short), Err(Error::Parse { .. })));
        // Unknown nuclide.
        let badnuc = format!("{header}Xx 99 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n");
        match parse_clearance(&badnuc) {
            Err(Error::Parse { line, msg }) => {
                assert_eq!(line, 5);
                assert!(msg.contains("Xx-99"), "msg was `{msg}`");
            }
            other => panic!("expected parse error, got {other:?}"),
        }
        // Malformed time header: no unit after the IS tag.
        let badtime = "1 * * * TIME INTERVAL 1 * * * TIME IS\n";
        assert!(matches!(
            parse_clearance(badtime),
            Err(Error::Parse { line: 1, .. })
        ));
        // Unknown time unit.
        let badunit = "1 * * * TIME INTERVAL 1 * * * TIME IS 1.0 fortnights\n";
        match parse_clearance(badunit) {
            Err(Error::Parse { line: 1, msg }) => {
                assert!(msg.contains("fortnights"), "msg was `{msg}`");
            }
            other => panic!("expected parse error, got {other:?}"),
        }
    }

    #[test]
    fn time_units_convert_with_documented_factors() {
        for (tag, expected) in [
            ("TIME IS 2.0 SECS", 2.0),
            ("TIME IS 2.0 MINS", 120.0),
            ("TIME IS 2.0 HOURS", 7_200.0),
            ("COOLING TIME IS 2.0 DAYS", 172_800.0),
            ("COOLING TIME IS 2.0 WEEKS", 1_209_600.0),
            ("COOLING TIME IS 2.0 YEARS", 63_072_000.0),
        ] {
            let text = format!(
                "1 * * * TIME INTERVAL 1 * * * {tag}\n\
                 \n  NUCLIDE ATOMS GRAMS Bq b a g DOSE ING INH CLEARANCE Bq/A2 HALF LIFE\n\
                 \n kW kW kW Sv/hr DOSE(Sv) DOSE(Sv) INDEX Ratio seconds\n\
                 \nFe 56 # 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 2.0 1.0 1.0\n"
            );
            let scan = parse_clearance(&text).unwrap();
            assert_eq!(scan.len(), 1, "tag `{tag}`");
            assert_eq!(scan.rows[0].time_s, expected, "tag `{tag}`");
            assert_eq!(scan.rows[0].clearance_index, 2.0, "tag `{tag}`");
        }
        // Irradiation flag off, cooling flag on.
        let text = "1 * * * TIME INTERVAL 9 * * * COOLING TIME IS 3.0 DAYS\n\
                    \n  NUCLIDE ATOMS GRAMS Bq b a g DOSE ING INH CLEARANCE Bq/A2 HALF LIFE\n\
                    \n kW kW kW Sv/hr DOSE(Sv) DOSE(Sv) INDEX Ratio seconds\n\
                    \nFe 56 # 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 2.0 1.0 1.0\n\
                    \n0 TOTAL NUMBER OF NUCLIDES PRINTED IN INVENTORY = 1\n";
        let scan = parse_clearance(text).unwrap();
        assert_eq!(scan.len(), 1);
        assert_eq!(scan.rows[0].time_s, 259_200.0);
        assert!(scan.rows[0].cooling);
    }
}
