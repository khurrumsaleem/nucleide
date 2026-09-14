//! Emission drift report — the `emit`-crate pattern applied to source cards.
//!
//! Each emission row records how much of the source's total emission
//! strength survives the translation onto a card, with the same
//! `rel_drift = (in - out) / in` convention as the `nucleide-emit` mass-drift
//! report. For source cards the "mass" is emission probability: a
//! monoenergetic line accounts 1.0
//! (drift 0), while a tabulated Gaussian drops the tails beyond the
//! `±width_sigma` window (drift = `1 - coverage`). `reparsed` marks rows
//! machine-verified by feeding the emitted text back through that code's
//! reader (MCNP `SDEF` only — the workspace has no Serpent source reader, so
//! Serpent rows are analytic by design, exactly like the emit crate's
//! Serpent/FLUKA/PARTISN material rows).

use std::fmt;

/// One quantity checked during emission.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftRow {
    /// What was checked (e.g. `"emission probability"`, `"spatial distribution"`).
    pub quantity: String,
    /// Fraction of the incoming strength the card accounts.
    pub accounted: f64,
    /// `1 - accounted` relative to unit incoming strength.
    pub rel_drift: f64,
    /// True when the emitted text round-tripped through that code's reader.
    pub reparsed: bool,
    /// Loud note: approximations, truncation, unverified semantics.
    pub note: String,
}

impl DriftRow {
    /// New row from an accounted fraction; `rel_drift` is `1 - accounted`.
    pub fn new(
        quantity: impl Into<String>,
        accounted: f64,
        reparsed: bool,
        note: impl Into<String>,
    ) -> Self {
        Self {
            quantity: quantity.into(),
            accounted,
            rel_drift: 1.0 - accounted,
            reparsed,
            note: note.into(),
        }
    }
}

/// Per-emission drift report.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriftReport {
    /// Rows in emission order.
    pub rows: Vec<DriftRow>,
}

impl DriftReport {
    /// Empty report.
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    /// Append a row.
    pub fn push(&mut self, row: DriftRow) {
        self.rows.push(row);
    }

    /// Worst `rel_drift` across rows (0 for an empty report).
    pub fn worst_rel_drift(&self) -> f64 {
        self.rows
            .iter()
            .map(|r| r.rel_drift.abs())
            .fold(0.0, f64::max)
    }
}

impl fmt::Display for DriftReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, row) in self.rows.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(
                f,
                "{}: accounted={:.6e} rel_drift={:.3e} reparsed={} — {}",
                row.quantity, row.accounted, row.rel_drift, row.reparsed, row.note
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rel_drift_is_one_minus_accounted() {
        let row = DriftRow::new("emission probability", 0.999_937, true, "note");
        assert!((row.rel_drift - (1.0 - 0.999_937)).abs() < 1e-15);
        let report = DriftReport {
            rows: vec![row.clone()],
        };
        assert!((report.worst_rel_drift() - row.rel_drift).abs() < 1e-15);
        assert_eq!(DriftReport::new().worst_rel_drift(), 0.0);
    }

    #[test]
    fn display_mentions_every_row() {
        let report = DriftReport {
            rows: vec![
                DriftRow::new("emission probability", 1.0, true, "monoenergetic"),
                DriftRow::new("spatial distribution", 1.0, false, "analytic by design"),
            ],
        };
        let text = report.to_string();
        assert!(text.contains("emission probability"));
        assert!(text.contains("spatial distribution"));
        assert!(text.contains("reparsed=false"));
    }
}
