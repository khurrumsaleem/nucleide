//! EPA Federal Guidance Report No. 15 external-dosimetry coefficients.
//!
//! [FGR 15] (EPA 402-R-25-001, July 2025) publishes effective dose
//! coefficients for external exposure pathways: ground-surface sources at
//! several soil depths, air submersion, and water immersion, per reference
//! person age group (newborn, 1-yr, 5-yr, 10-yr, 15-yr, adult). This module
//! parses the `FGR15_Tables/Table_4_*.DAT` members of the EPA distribution
//! zip into typed, nucid-keyed tables with nuclide lookup.
//!
//! # Distribution contract
//!
//! Nothing from the EPA file is vendored into this repository: the zip is
//! fetched at runtime by the Python layer (`nucleide.data.fetch_fgr15`),
//! hash-pinned against the published SHA-256, and member text is passed here
//! for parsing. Tests use synthetic hand-built tables only.
//!
//! # Table layout
//!
//! Each `Table_4_N.DAT` member is CRLF-terminated ASCII:
//!
//! ```text
//! Table 4.1. Effective Dose Rate Coefficients for Ground Surface (3 mm)
//! <blank>
//!           ------- Reference Person Coefficients (Sv m2/Bq s) --------
//!  Nuclide  Newborn   1-yr-old  5-yr-old 10-yr-old 15-yr-old   Adult
//! ---------------------------------------------------------------------
//! Hydrogen
//!  H-3       3.84e-27  3.37e-27  2.73e-27  2.56e-27  1.01e-27  8.97e-28
//! ...
//! ---------------------------------------------------------------------
//! ```
//!
//! - Line 1 titles the table as `Table 4.N.`; `N` selects the [`Fgr15Scenario`].
//! - The units string is recorded verbatim from the parenthesized group of
//!   the `Reference Person Coefficients (...)` header line (`Sv m2/Bq s` for
//!   the surface table, `Sv m3 per Bq s` for the volume/immersion tables —
//!   wording varies, so it is never hardcoded).
//! - Element-name separator lines (e.g. `Hydrogen`) carry no data and are
//!   structural; some tables prefix the first one with a BOM/zero-width
//!   character, which parsing strips.
//! - A data row is a nuclide name (`H-3`, `Cs-137`, `Ba-137m`, `Sb-124n` —
//!   element-mass plus an optional lowercase isomer letter following the
//!   repo-wide `mnopqrstuvxyz` state convention) followed by exactly six
//!   finite floats in [`Fgr15Age::ALL`] order. Anything else in the row
//!   region is an error; rows are never silently skipped, duplicates are
//!   rejected, and the parsed row total must equal the caller's
//!   `expected_rows`.
//!
//! Screening-level only — not for safety decisions.
//!
//! [FGR 15]: https://www.epa.gov/radiation/federal-guidance-report-no-15-external-exposure-radionuclides-air-water-and-soil

use std::collections::BTreeMap;
use std::fmt;

use crate::dialects::{isomer_letter, isomer_state};
use crate::{element_z, NuclideId, ELEMENTS};

/// Result alias for FGR 15 parsing.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors from parsing an EPA FGR 15 `Table_4_*.DAT` member.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The first line is not a `Table 4.N.` title heading.
    BadTitle(String),
    /// The title's table number does not identify a known FGR 15 scenario.
    UnknownScenario(String),
    /// No `Reference Person Coefficients (...)` units header line was found.
    MissingUnits,
    /// The column header line is not the expected seven-column heading.
    MalformedHeader(String),
    /// The dashed separator after the column header is missing.
    MissingSeparator,
    /// A line in the row region is neither structural nor a data row.
    UnexpectedLine {
        /// 1-based source line number.
        line: usize,
        /// The offending line text.
        text: String,
    },
    /// A data-shaped row has the wrong column count, an unparseable or
    /// non-finite coefficient, or an invalid nuclide name.
    MalformedRow {
        /// 1-based source line number.
        line: usize,
        /// The offending line text.
        text: String,
    },
    /// The same nuclide appears twice in one table.
    DuplicateRow {
        /// 1-based source line number.
        line: usize,
        /// The duplicated nuclide name.
        name: String,
    },
    /// The parsed row total differs from the caller's expectation.
    RowCount {
        /// The table being parsed.
        scenario: Fgr15Scenario,
        /// Expected row count.
        expected: usize,
        /// Parsed row count.
        found: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadTitle(t) => write!(f, "first line is not a `Table 4.N.` title: `{t}`"),
            Self::UnknownScenario(t) => write!(f, "unknown FGR 15 scenario in title: `{t}`"),
            Self::MissingUnits => write!(f, "missing `Reference Person Coefficients (...)` header"),
            Self::MalformedHeader(h) => write!(f, "malformed column header: `{h}`"),
            Self::MissingSeparator => write!(f, "missing dashed separator after column header"),
            Self::UnexpectedLine { line, text } => {
                write!(f, "unexpected content at line {line}: `{text}`")
            }
            Self::MalformedRow { line, text } => {
                write!(f, "malformed data row at line {line}: `{text}`")
            }
            Self::DuplicateRow { line, name } => {
                write!(f, "duplicate nuclide row `{name}` at line {line}")
            }
            Self::RowCount {
                scenario,
                expected,
                found,
            } => write!(
                f,
                "table {} ({}) holds {found} rows, expected {expected}",
                scenario.table_number(),
                scenario.as_str()
            ),
        }
    }
}

impl std::error::Error for Error {}

/// FGR 15 exposure scenario: the seven `Table 4.1`–`Table 4.7` coefficient
/// tables (EPA 402-R-25-001 §4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Fgr15Scenario {
    /// Table 4.1 — effective dose rate coefficients for ground surface (3 mm).
    GroundSurface,
    /// Table 4.2 — effective dose coefficients for soil to 1 cm depth.
    Soil1cm,
    /// Table 4.3 — effective dose coefficients for soil to 5 cm depth.
    Soil5cm,
    /// Table 4.4 — effective dose coefficients for soil to 15 cm depth.
    Soil15cm,
    /// Table 4.5 — effective dose coefficients for soil to infinite depth.
    SoilInfinite,
    /// Table 4.6 — effective dose rate coefficients for air submersion.
    AirSubmersion,
    /// Table 4.7 — effective dose rate coefficients for water immersion.
    WaterImmersion,
}

impl Fgr15Scenario {
    /// Canonical snake-case name (`ground_surface`, `soil_1cm`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GroundSurface => "ground_surface",
            Self::Soil1cm => "soil_1cm",
            Self::Soil5cm => "soil_5cm",
            Self::Soil15cm => "soil_15cm",
            Self::SoilInfinite => "soil_infinite",
            Self::AirSubmersion => "air_submersion",
            Self::WaterImmersion => "water_immersion",
        }
    }

    /// EPA table number (`4.1`, `4.2`, ...), as spelled in the member title.
    pub fn table_number(self) -> &'static str {
        match self {
            Self::GroundSurface => "4.1",
            Self::Soil1cm => "4.2",
            Self::Soil5cm => "4.3",
            Self::Soil15cm => "4.4",
            Self::SoilInfinite => "4.5",
            Self::AirSubmersion => "4.6",
            Self::WaterImmersion => "4.7",
        }
    }

    /// EPA scenario title as printed in the table heading.
    pub fn title(self) -> &'static str {
        match self {
            Self::GroundSurface => "Ground Surface (3 mm)",
            Self::Soil1cm => "Soil to 1 cm Depth",
            Self::Soil5cm => "Soil to 5 cm Depth",
            Self::Soil15cm => "Soil to 15 cm Depth",
            Self::SoilInfinite => "Soil to Infinite Depth",
            Self::AirSubmersion => "Air Submersion",
            Self::WaterImmersion => "Water Immersion",
        }
    }

    /// All seven scenarios in table order.
    pub const ALL: [Self; 7] = [
        Self::GroundSurface,
        Self::Soil1cm,
        Self::Soil5cm,
        Self::Soil15cm,
        Self::SoilInfinite,
        Self::AirSubmersion,
        Self::WaterImmersion,
    ];

    /// Parse a scenario key: the EPA table number (`4.1`–`4.7`) or the
    /// canonical snake-case name (case-insensitive; `-` accepted for `_`).
    pub fn parse(s: &str) -> Option<Self> {
        let key = s.trim().to_ascii_lowercase().replace('-', "_");
        match key.as_str() {
            "ground_surface" => Some(Self::GroundSurface),
            "soil_1cm" => Some(Self::Soil1cm),
            "soil_5cm" => Some(Self::Soil5cm),
            "soil_15cm" => Some(Self::Soil15cm),
            "soil_infinite" => Some(Self::SoilInfinite),
            "air_submersion" => Some(Self::AirSubmersion),
            "water_immersion" => Some(Self::WaterImmersion),
            _ => match key.trim_start_matches("table ").trim() {
                "4.1" => Some(Self::GroundSurface),
                "4.2" => Some(Self::Soil1cm),
                "4.3" => Some(Self::Soil5cm),
                "4.4" => Some(Self::Soil15cm),
                "4.5" => Some(Self::SoilInfinite),
                "4.6" => Some(Self::AirSubmersion),
                "4.7" => Some(Self::WaterImmersion),
                _ => None,
            },
        }
    }

    fn from_table_number(n: u32) -> Option<Self> {
        match n {
            1 => Some(Self::GroundSurface),
            2 => Some(Self::Soil1cm),
            3 => Some(Self::Soil5cm),
            4 => Some(Self::Soil15cm),
            5 => Some(Self::SoilInfinite),
            6 => Some(Self::AirSubmersion),
            7 => Some(Self::WaterImmersion),
            _ => None,
        }
    }
}

/// Reference-person age group: the six coefficient columns of every FGR 15
/// table, in canonical column order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Fgr15Age {
    /// Newborn.
    Newborn,
    /// 1-year-old.
    Year1,
    /// 5-year-old.
    Year5,
    /// 10-year-old.
    Year10,
    /// 15-year-old.
    Year15,
    /// Adult (reference person).
    Adult,
}

impl Fgr15Age {
    /// All six age groups in canonical column order.
    pub const ALL: [Self; 6] = [
        Self::Newborn,
        Self::Year1,
        Self::Year5,
        Self::Year10,
        Self::Year15,
        Self::Adult,
    ];

    /// Canonical short name (`newborn`, `1-yr`, `5-yr`, `10-yr`, `15-yr`,
    /// `adult`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Newborn => "newborn",
            Self::Year1 => "1-yr",
            Self::Year5 => "5-yr",
            Self::Year10 => "10-yr",
            Self::Year15 => "15-yr",
            Self::Adult => "adult",
        }
    }

    /// Column index of this age group in a coefficient row (0–5).
    pub fn index(self) -> usize {
        match self {
            Self::Newborn => 0,
            Self::Year1 => 1,
            Self::Year5 => 2,
            Self::Year10 => 3,
            Self::Year15 => 4,
            Self::Adult => 5,
        }
    }

    /// Parse an age-group key: `newborn`/`adult`, a bare year (`1`, `5`,
    /// `10`, `15`), or a spelled variant (`1yr`, `1-yr`, `1-yr-old`, ...;
    /// case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        let key = s.trim().to_ascii_lowercase();
        match key.as_str() {
            "newborn" => return Some(Self::Newborn),
            "adult" => return Some(Self::Adult),
            _ => {}
        }
        let digits = key.len() - key.trim_start_matches(|c: char| c.is_ascii_digit()).len();
        if digits == 0 {
            return None;
        }
        let rest = &key[digits..];
        const SUFFIXES: [&str; 7] = ["", "-yr", "-y", "-yr-old", "-y-old", "yr", "y"];
        if !SUFFIXES.contains(&rest) {
            return None;
        }
        match &key[..digits] {
            "1" => Some(Self::Year1),
            "5" => Some(Self::Year5),
            "10" => Some(Self::Year10),
            "15" => Some(Self::Year15),
            _ => None,
        }
    }
}

/// One parsed FGR 15 coefficient table.
#[derive(Debug, Clone, PartialEq)]
pub struct Fgr15Table {
    scenario: Fgr15Scenario,
    units: String,
    rows: BTreeMap<u32, [f64; 6]>,
}

impl Fgr15Table {
    /// The table's exposure scenario.
    pub fn scenario(&self) -> Fgr15Scenario {
        self.scenario
    }

    /// Coefficient units, recorded verbatim from the table header
    /// (e.g. `Sv m2/Bq s` or `Sv m3 per Bq s`).
    pub fn units(&self) -> &str {
        &self.units
    }

    /// Number of nuclide rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table has no rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Coefficient for `id` at `age`, or `None` when the nuclide has no row.
    pub fn coefficient(&self, id: NuclideId, age: Fgr15Age) -> Option<f64> {
        self.rows.get(&id.nucid()).map(|row| row[age.index()])
    }

    /// Coefficient for an FGR 15 nuclide name (see [`parse_nuclide_name`]),
    /// or `None` when the name is invalid or the nuclide has no row.
    pub fn coefficient_by_name(&self, name: &str, age: Fgr15Age) -> Option<f64> {
        let id = parse_nuclide_name(name)?;
        self.coefficient(id, age)
    }

    /// Iterate `(nucid, coefficients)` pairs in nucid order; coefficient
    /// slices follow [`Fgr15Age::ALL`] column order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &[f64; 6])> + '_ {
        self.rows.iter().map(|(nucid, row)| (*nucid, row))
    }
}

/// Whether `token` has the surface shape of an FGR 15 nuclide name:
/// 1–2 ASCII letters, `-`, one or more digits, optional single letter.
/// Component validation happens in [`parse_nuclide_name`].
fn has_name_shape(token: &str) -> bool {
    let Some((symbol, rest)) = token.split_once('-') else {
        return false;
    };
    if symbol.is_empty() || symbol.len() > 2 || !symbol.chars().all(|c| c.is_ascii_alphabetic()) {
        return false;
    }
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    if digits == 0 {
        return false;
    }
    let suffix = &rest[digits..];
    suffix.is_empty()
        || (suffix.len() == 1
            && suffix
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic()))
}

/// Parse an FGR 15 nuclide name (`H-3`, `Cs-137`, `Ba-137m`, `Sb-124n`) into
/// a [`NuclideId`]: element symbol, `-`, mass number, and an optional
/// isomer designator letter (`m` → state 1, `n` → state 2, ... following
/// the repo-wide `mnopqrstuvxyz` convention, case-insensitive). Returns
/// `None` for any other shape or invalid components.
pub fn parse_nuclide_name(name: &str) -> Option<NuclideId> {
    let (symbol, rest) = name.split_once('-')?;
    if symbol.is_empty() || symbol.len() > 2 || !symbol.chars().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let z = element_z(symbol).or_else(|| {
        let mut chars = symbol.chars();
        let canonical = match chars.next() {
            Some(first) => {
                let mut s = first.to_uppercase().collect::<String>();
                s.push_str(&chars.as_str().to_lowercase());
                s
            }
            None => return None,
        };
        element_z(&canonical)
    })?;
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let (a_str, suffix) = rest.split_at(digits);
    if a_str.is_empty() {
        return None;
    }
    let a: u32 = a_str.parse().ok()?;
    let state = if suffix.is_empty() {
        0
    } else {
        let mut letters = suffix.chars();
        let (letter, extra) = (letters.next()?, letters.next());
        if extra.is_some() {
            return None;
        }
        isomer_state(letter)?
    };
    NuclideId::new(z, a, state).ok()
}

/// FGR 15 name spelling for a nuclide (`H-3`, `Ba-137m`, `Sb-124n`); the
/// isomer letter follows the `mnopqrstuvxyz` state convention.
pub fn name_of(id: NuclideId) -> String {
    let mut name = String::new();
    name.push_str(ELEMENTS[id.z() as usize]);
    name.push('-');
    name.push_str(&id.a().to_string());
    if let Some(letter) = isomer_letter(id.state()) {
        name.push(letter);
    }
    name
}

/// Strip invisible formatting characters the EPA file sprinkles into
/// otherwise-structural lines (UTF-8 BOM before the first element name,
/// zero-width spaces).
fn clean(line: &str) -> String {
    line.chars()
        .filter(|&c| c != '\u{feff}' && c != '\u{200b}')
        .collect()
}

/// Parse the `Table 4.N.` title line, returning the scenario.
fn parse_title(line: &str) -> Result<Fgr15Scenario> {
    let title = line.trim();
    let Some(rest) = title.strip_prefix("Table 4.") else {
        return Err(Error::BadTitle(title.to_string()));
    };
    let digits = rest
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(rest.len());
    let (n_str, tail) = rest.split_at(digits);
    let n: u32 = n_str
        .parse()
        .map_err(|_| Error::BadTitle(title.to_string()))?;
    if !tail.starts_with('.') {
        return Err(Error::BadTitle(title.to_string()));
    }
    Fgr15Scenario::from_table_number(n).ok_or_else(|| Error::UnknownScenario(title.to_string()))
}

/// Extract the parenthesized units group from the
/// `Reference Person Coefficients (...)` header line.
fn extract_units(line: &str) -> Result<String> {
    let open = line.find('(').ok_or(Error::MissingUnits)?;
    let close = line.rfind(')').ok_or(Error::MissingUnits)?;
    if close <= open + 1 {
        return Err(Error::MissingUnits);
    }
    Ok(line[open + 1..close].to_string())
}

/// Expected column header tokens (the six [`Fgr15Age::ALL`] headings).
const COLUMN_HEADER: [&str; 7] = [
    "Nuclide",
    "Newborn",
    "1-yr-old",
    "5-yr-old",
    "10-yr-old",
    "15-yr-old",
    "Adult",
];

/// Parse one FGR 15 `Table_4_*.DAT` member.
///
/// `expected_rows` is the exact nuclide-row count the table must hold
/// (1,252 for the published EPA tables); a mismatch is a loud
/// [`Error::RowCount`]. Row-region content is never silently skipped:
/// element-name separators, dashed separators, and blank lines are
/// structural; anything else must be a well-formed six-column data row.
pub fn parse_table(text: &str, expected_rows: usize) -> Result<Fgr15Table> {
    let mut lines = text.lines().enumerate();

    let (_, title) = lines.next().ok_or_else(|| Error::BadTitle(String::new()))?;
    let scenario = parse_title(&clean(title))?;

    // Between the title and the units header only blank lines are legal.
    let units = loop {
        match lines.next() {
            Some((lineno, raw)) => {
                let line = clean(raw);
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if trimmed.contains("Reference Person Coefficients") {
                    break extract_units(trimmed)?;
                }
                return Err(Error::UnexpectedLine {
                    line: lineno + 1,
                    text: trimmed.to_string(),
                });
            }
            None => return Err(Error::MissingUnits),
        }
    };

    let header = match lines.next() {
        Some((_, raw)) => clean(raw),
        None => return Err(Error::MalformedHeader(String::new())),
    };
    let columns: Vec<&str> = header.split_whitespace().collect();
    if columns != COLUMN_HEADER {
        return Err(Error::MalformedHeader(header.trim().to_string()));
    }

    let separator = match lines.next() {
        Some((_, raw)) => clean(raw),
        None => return Err(Error::MissingSeparator),
    };
    let sep = separator.trim();
    if sep.len() < 10 || !sep.chars().all(|c| c == '-') {
        return Err(Error::MissingSeparator);
    }

    let mut rows: BTreeMap<u32, [f64; 6]> = BTreeMap::new();
    for (lineno, raw) in lines {
        let line = clean(raw);
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.chars().all(|c| c == '-') {
            continue; // blank or structural dashed separator
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() == 1 && is_element_separator(tokens[0]) {
            continue;
        }
        let Some(id) = parse_nuclide_name(tokens[0]) else {
            // Name-shaped but invalid components (bad element, A < Z,
            // A > 999, unknown isomer letter): the row region promised a
            // data row, so this is a malformed row, not foreign content.
            return Err(if has_name_shape(tokens[0]) {
                Error::MalformedRow {
                    line: lineno + 1,
                    text: trimmed.to_string(),
                }
            } else {
                Error::UnexpectedLine {
                    line: lineno + 1,
                    text: trimmed.to_string(),
                }
            });
        };
        if tokens.len() != 7 {
            return Err(Error::MalformedRow {
                line: lineno + 1,
                text: trimmed.to_string(),
            });
        }
        let mut row = [0.0; 6];
        for (i, token) in tokens[1..].iter().enumerate() {
            let value: f64 = token.parse().map_err(|_| Error::MalformedRow {
                line: lineno + 1,
                text: trimmed.to_string(),
            })?;
            if !value.is_finite() {
                return Err(Error::MalformedRow {
                    line: lineno + 1,
                    text: trimmed.to_string(),
                });
            }
            row[i] = value;
        }
        if rows.insert(id.nucid(), row).is_some() {
            return Err(Error::DuplicateRow {
                line: lineno + 1,
                name: tokens[0].to_string(),
            });
        }
    }

    if rows.len() != expected_rows {
        return Err(Error::RowCount {
            scenario,
            expected: expected_rows,
            found: rows.len(),
        });
    }
    Ok(Fgr15Table {
        scenario,
        units,
        rows,
    })
}

/// Element-name separator lines are single capitalized alphabetic words
/// (`Hydrogen`, `Beryllium`).
fn is_element_separator(token: &str) -> bool {
    let mut chars = token.chars();
    match chars.next() {
        Some(first) if first.is_ascii_uppercase() => chars.all(|c| c.is_ascii_alphabetic()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic hand-built Table 4.1-layout text: CRLF endings, element
    /// separators (one with a BOM), six fake coefficient columns, trailing
    /// dashed line. No EPA content.
    const SYNTHETIC: &str = concat!(
        "Table 4.1. Synthetic Ground Surface (3 mm)\r\n",
        "\r\n",
        "          ------- Reference Person Coefficients (Sv m2/Bq s) --------\r\n",
        " Nuclide  Newborn   1-yr-old  5-yr-old 10-yr-old 15-yr-old   Adult\r\n",
        "---------------------------------------------------------------------\r\n",
        "\u{feff}Hydrogen\r\n",
        " H-3       1.00e-01  2.00e-01  3.00e-01  4.00e-01  5.00e-01  6.00e-01\r\n",
        "Beryllium\r\n",
        " Be-7      1.10e-01  1.20e-01  1.30e-01  1.40e-01  1.50e-01  1.60e-01\r\n",
        "Antimony\r\n",
        " Sb-124n   2.10e-01  2.20e-01  2.30e-01  2.40e-01  2.50e-01  2.60e-01\r\n",
        "Barium\r\n",
        " Ba-137m   3.10e-01  3.20e-01  3.30e-01  3.40e-01  3.50e-01  3.60e-01\r\n",
        "Caesium\r\n",
        " Cs-137    4.10e-01  4.20e-01  4.30e-01  4.40e-01  4.50e-01  4.60e-01\r\n",
        "---------------------------------------------------------------------\r\n",
    );

    #[test]
    fn parses_synthetic_table_exact_layout() {
        let table = parse_table(SYNTHETIC, 5).unwrap();
        assert_eq!(table.scenario(), Fgr15Scenario::GroundSurface);
        assert_eq!(table.units(), "Sv m2/Bq s");
        assert_eq!(table.len(), 5);
        assert!(!table.is_empty());

        let h3 = parse_nuclide_name("H-3").unwrap();
        assert_eq!(table.coefficient(h3, Fgr15Age::Newborn), Some(0.10));
        assert_eq!(table.coefficient(h3, Fgr15Age::Adult), Some(0.60));
        assert_eq!(
            table.coefficient_by_name("Cs-137", Fgr15Age::Year10),
            Some(0.44)
        );
        // Isomer letters: m -> state 1, n -> state 2.
        let sb124n = parse_nuclide_name("Sb-124n").unwrap();
        assert_eq!(sb124n.state(), 2);
        assert_eq!(table.coefficient(sb124n, Fgr15Age::Year5), Some(0.23));
        assert_eq!(name_of(sb124n), "Sb-124n");
        assert_eq!(name_of(h3), "H-3");
        assert_eq!(name_of(parse_nuclide_name("Ba-137m").unwrap()), "Ba-137m");
        // Unknown nuclide resolves to None.
        let u235 = parse_nuclide_name("U-235").unwrap();
        assert_eq!(table.coefficient(u235, Fgr15Age::Adult), None);
        assert_eq!(
            table.coefficient_by_name("Not-A-Nuclide", Fgr15Age::Adult),
            None
        );
    }

    #[test]
    fn row_region_accepts_non_finite_rejected_and_misc_structure() {
        // The synthetic layout above parses with LF-only endings too.
        let lf_only = SYNTHETIC.replace("\r\n", "\n");
        assert_eq!(parse_table(&lf_only, 5).unwrap().len(), 5);
    }

    #[test]
    fn row_count_mismatch_is_loud() {
        let short = SYNTHETIC.replace(
            " Cs-137    4.10e-01  4.20e-01  4.30e-01  4.40e-01  4.50e-01  4.60e-01\r\n",
            "",
        );
        match parse_table(&short, 5).unwrap_err() {
            Error::RowCount {
                scenario,
                expected,
                found,
            } => {
                assert_eq!(scenario, Fgr15Scenario::GroundSurface);
                assert_eq!(expected, 5);
                assert_eq!(found, 4);
            }
            other => panic!("{other:?}"),
        }
        // Wrong expectation on an intact table errors the same way.
        match parse_table(SYNTHETIC, 1252).unwrap_err() {
            Error::RowCount {
                expected, found, ..
            } => {
                assert_eq!(expected, 1252);
                assert_eq!(found, 5);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn malformed_rows_error() {
        let bad_float = SYNTHETIC.replace("2.00e-01", "not-a-float");
        assert!(matches!(
            parse_table(&bad_float, 5).unwrap_err(),
            Error::MalformedRow { .. }
        ));
        let short_row = SYNTHETIC.replace(
            " H-3       1.00e-01  2.00e-01  3.00e-01  4.00e-01  5.00e-01  6.00e-01\r\n",
            " H-3       1.00e-01  2.00e-01\r\n",
        );
        assert!(matches!(
            parse_table(&short_row, 5).unwrap_err(),
            Error::MalformedRow { .. }
        ));
        let non_finite = SYNTHETIC.replace("3.00e-01", "inf");
        assert!(matches!(
            parse_table(&non_finite, 5).unwrap_err(),
            Error::MalformedRow { .. }
        ));
        let bad_mass = SYNTHETIC.replace(" H-3 ", " H-9999 ");
        assert!(matches!(
            parse_table(&bad_mass, 5).unwrap_err(),
            Error::MalformedRow { .. }
        ));
    }

    #[test]
    fn duplicate_row_errors() {
        let dup = SYNTHETIC.replace("Beryllium\r\n", "Beryllium\r\n Be-7      9.00e-01  9.00e-01  9.00e-01  9.00e-01  9.00e-01  9.00e-01\r\n");
        match parse_table(&dup, 5).unwrap_err() {
            Error::DuplicateRow { name, .. } => assert_eq!(name, "Be-7"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn unexpected_line_errors() {
        let junk = SYNTHETIC.replace("Beryllium\r\n", "total garbage here\r\n");
        assert!(matches!(
            parse_table(&junk, 5).unwrap_err(),
            Error::UnexpectedLine { .. }
        ));
        let lowercase = SYNTHETIC.replace("Beryllium\r\n", "beryllium\r\n");
        assert!(matches!(
            parse_table(&lowercase, 5).unwrap_err(),
            Error::UnexpectedLine { .. }
        ));
    }

    #[test]
    fn structural_errors() {
        match parse_table("", 0).unwrap_err() {
            Error::BadTitle(t) => assert!(t.is_empty()),
            other => panic!("{other:?}"),
        }
        // Title plus the blank line after it, then end of file: no units header.
        let truncated: String = SYNTHETIC.lines().take(2).collect::<Vec<_>>().join("\n");
        assert_eq!(parse_table(&truncated, 5).unwrap_err(), Error::MissingUnits);
        let bad_header = SYNTHETIC.replace(" Nuclide  Newborn", " Nuclide  Infant");
        assert!(matches!(
            parse_table(&bad_header, 5).unwrap_err(),
            Error::MalformedHeader(_)
        ));
        let no_sep = SYNTHETIC.replace(
            "---------------------------------------------------------------------\r\n",
            "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\r\n",
        );
        assert_eq!(
            parse_table(&no_sep, 5).unwrap_err(),
            Error::MissingSeparator
        );
        let bad_title = SYNTHETIC.replacen("Table 4.1.", "Table 4.8.", 1);
        assert!(matches!(
            parse_table(&bad_title, 5).unwrap_err(),
            Error::UnknownScenario(_)
        ));
        let not_a_title = SYNTHETIC.replacen("Table 4.1.", "Table 9.9.", 1);
        assert!(matches!(
            parse_table(&not_a_title, 5).unwrap_err(),
            Error::BadTitle(_)
        ));
    }

    #[test]
    fn scenario_keys_round_trip() {
        for scenario in Fgr15Scenario::ALL {
            assert_eq!(Fgr15Scenario::parse(scenario.as_str()), Some(scenario));
            assert_eq!(
                Fgr15Scenario::parse(scenario.table_number()),
                Some(scenario)
            );
            // Retitling the synthetic table selects the matching scenario.
            let text = SYNTHETIC.replacen(
                "Table 4.1.",
                &format!("Table {}.", scenario.table_number()),
                1,
            );
            let table = parse_table(&text, 5).unwrap();
            assert_eq!(table.scenario(), scenario);
            assert_eq!(table.units(), "Sv m2/Bq s");
        }
        assert_eq!(
            Fgr15Scenario::parse("SOIL-15CM"),
            Some(Fgr15Scenario::Soil15cm)
        );
        assert_eq!(
            Fgr15Scenario::parse("table 4.7"),
            Some(Fgr15Scenario::WaterImmersion)
        );
        assert_eq!(Fgr15Scenario::parse("4.8"), None);
        assert_eq!(Fgr15Scenario::parse("moon"), None);
    }

    #[test]
    fn age_keys_parse() {
        for (i, age) in Fgr15Age::ALL.iter().enumerate() {
            assert_eq!(age.index(), i);
            assert_eq!(Fgr15Age::parse(age.as_str()), Some(*age));
        }
        assert_eq!(Fgr15Age::parse("1"), Some(Fgr15Age::Year1));
        assert_eq!(Fgr15Age::parse("10-YR-OLD"), Some(Fgr15Age::Year10));
        assert_eq!(Fgr15Age::parse("5yr"), Some(Fgr15Age::Year5));
        assert_eq!(Fgr15Age::parse(" 15-yr "), Some(Fgr15Age::Year15));
        assert_eq!(Fgr15Age::parse("Adult"), Some(Fgr15Age::Adult));
        assert_eq!(Fgr15Age::parse("2"), None);
        assert_eq!(Fgr15Age::parse("newborns"), None);
        assert_eq!(Fgr15Age::parse("1 yr old"), None);
    }

    #[test]
    fn nuclide_name_spellings() {
        assert_eq!(parse_nuclide_name("H-3").unwrap().nucid(), 10_030_000);
        assert_eq!(parse_nuclide_name("cs-137").unwrap().state(), 0);
        assert_eq!(parse_nuclide_name("SB-124N").unwrap().state(), 2);
        assert!(parse_nuclide_name("H3").is_none());
        assert!(parse_nuclide_name("H-").is_none());
        assert!(parse_nuclide_name("-3").is_none());
        assert!(parse_nuclide_name("Xx-3").is_none());
        assert!(parse_nuclide_name("U-2").is_none()); // A < Z
        assert!(parse_nuclide_name("H-3mn").is_none());
        assert!(parse_nuclide_name("H-3w").is_none()); // 'w' is not a designator
    }
}
