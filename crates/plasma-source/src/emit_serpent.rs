//! Serpent `src` source-card emission.
//!
//! Serpent defines sources with one or more `src <id> <key> <value>` lines
//! that accumulate parameters for the same id, plus `SIn`/`SPn` distribution
//! pairs in the Serpent input-manual sense (`SI` sample points, `SP`
//! probabilities; when `SP` carries one entry fewer than `SI` the pair is a
//! histogram over the `SI` boundaries). This module renders:
//!
//! ```text
//! src 1 pos 0 0 25
//! src 1 rad d1
//! src 1 erg d2
//! src 1 wgt 1
//! SI1 300 300
//! SP1 1
//! SI2 <e1> <e2> ...
//! SP2 <p1> <p2> ...
//! ```
//!
//! - a point source emits `pos x y z` only;
//! - a ring adds `rad d1` with `SI1 R R` / `SP1 1` — a single zero-width
//!   histogram bin, the Serpent spelling of a delta ring at radius `R`;
//! - the spectrum is `erg <e>` for a monoenergetic line or `erg dn` with the
//!   tabulated Gaussian on `SIn`/`SPn` (bin centers and normal-CDF
//!   probabilities, as in the MCNP direction).
//!
//! The workspace has no Serpent source reader, so these rows are analytic by
//! design in the drift report (the `nucleide-emit` Serpent/FLUKA/PARTISN
//! posture): surface syntax follows the public Serpent input manual, and the
//! sampling semantics are Serpent's.

use crate::emit_sdef::drift_report;
use crate::parametric::{emission_histograms, ParametricPlasmaConfig};
use crate::{EmittedCard, PlasmaSourceConfig, Result, SourceModel};

/// Half-width of the tabulated spectrum window, in sigma.
const TABLE_WIDTH_SIGMA: f64 = 4.0;

/// Format like C++ `std::ostream <<` default formatting (`%g` with precision
/// 6, trailing zeros stripped) — the same number shape the MCNP `SDEF`
/// dialect renders, so both cards quote identical values.
fn fmt_g6(x: f64) -> String {
    debug_assert!(x.is_finite());
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    const P: i32 = 6;
    let sci = format!("{:.*e}", (P - 1) as usize, x);
    let (mant, exp) = sci.split_once('e').expect("scientific format has e");
    let exp: i32 = exp.parse().expect("scientific exponent is an integer");
    let mant = trim_zeros(mant);
    if !(-4..P).contains(&exp) {
        format!("{mant}e{exp:+03}")
    } else {
        let decimals = (P - 1 - exp).max(0) as usize;
        trim_zeros(&format!("{x:.decimals$}"))
    }
}

/// Strip trailing fractional zeros (`"1.40000"` -> `"1.4"`); integers pass.
fn trim_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        t.strip_suffix('.').unwrap_or(t).to_string()
    } else {
        s.to_string()
    }
}

/// Render a fusion source as Serpent `src` card lines plus drift report.
///
/// `n_bins` sets the Gaussian tabulation bin count (values `< 2` fall back
/// to the 21-bin default).
pub fn emit_serpent(config: &PlasmaSourceConfig, n_bins: usize) -> Result<EmittedCard> {
    config.validate()?;
    let spectrum = config.spectrum()?;
    let table = spectrum.tabulate(if n_bins >= 2 { n_bins } else { 21 }, TABLE_WIDTH_SIGMA)?;

    // Distribution numbering mirrors the MCNP direction: the ring radial
    // delta takes d1, so the spectrum starts at d2 for rings, d1 for points.
    let spectrum_number = match config.model {
        SourceModel::Ring(_) => 2,
        SourceModel::Point(_) => 1,
    };

    let mut lines: Vec<String> = Vec::new();
    match config.model {
        SourceModel::Point(p) => {
            lines.push(format!(
                "src 1 pos {} {} {}",
                fmt_g6(p.x_cm),
                fmt_g6(p.y_cm),
                fmt_g6(p.z_cm)
            ));
        }
        SourceModel::Ring(r) => {
            lines.push(format!("src 1 pos 0 0 {}", fmt_g6(r.height_cm)));
            lines.push("src 1 rad d1".to_string());
        }
    }
    if spectrum.is_mono() {
        lines.push(format!("src 1 erg {}", fmt_g6(table.energies_mev[0])));
    } else {
        lines.push(format!("src 1 erg d{spectrum_number}"));
    }
    lines.push(format!("src 1 wgt {}", fmt_g6(config.weight)));
    if let SourceModel::Ring(r) = config.model {
        lines.push(format!(
            "SI1 {} {}",
            fmt_g6(r.radius_cm),
            fmt_g6(r.radius_cm)
        ));
        lines.push("SP1 1".to_string());
    }
    if !spectrum.is_mono() {
        let energies: Vec<String> = table.energies_mev.iter().map(|&e| fmt_g6(e)).collect();
        let probs: Vec<String> = table.probabilities.iter().map(|&p| fmt_g6(p)).collect();
        lines.push(format!("SI{spectrum_number} {}", energies.join(" ")));
        lines.push(format!("SP{spectrum_number} {}", probs.join(" ")));
    }

    let text = lines.join("\n");
    let drift = drift_report(&spectrum, &table, false);
    Ok(EmittedCard { text, drift })
}

/// Render a parametric plasma source as Serpent `src` card lines plus drift
/// report.
///
/// The card carries the same three marginals as the MCNP direction —
/// `rad d1` (birth minor-radius profile), `ext d2` (birth Z profile), and
/// `erg d3` (global marginal energy spectrum); see
/// [`emit_sdef_parametric`](crate::emit_sdef::emit_sdef_parametric) for the
/// product-form caveat. Drift rows are analytic by design (no Serpent
/// source reader in the workspace).
pub fn emit_serpent_parametric(
    config: &ParametricPlasmaConfig,
    n_bins: usize,
) -> Result<EmittedCard> {
    config.validate()?;
    let bins = if n_bins >= 2 { n_bins } else { 21 };
    let hist = emission_histograms(config, bins)?;

    let mut lines: Vec<String> = Vec::new();
    lines.push("src 1 pos 0 0 0".to_string());
    lines.push("src 1 rad d1".to_string());
    lines.push("src 1 ext d2".to_string());
    lines.push("src 1 erg d3".to_string());
    lines.push(format!("src 1 wgt {}", fmt_g6(config.weight)));
    for (number, dist) in [(1, &hist.radial), (2, &hist.vertical), (3, &hist.energy)] {
        let centers: Vec<String> = dist.centers.iter().map(|&c| fmt_g6(c)).collect();
        let masses: Vec<String> = dist.masses.iter().map(|&m| fmt_g6(m)).collect();
        lines.push(format!("SI{number} {}", centers.join(" ")));
        lines.push(format!("SP{number} {}", masses.join(" ")));
    }
    let text = lines.join("\n");

    let mut drift = crate::report::DriftReport::new();
    drift.push(crate::report::DriftRow::new(
        "emission probability",
        hist.energy_coverage,
        false,
        "marginal energy spectrum tabulated; tail mass dropped, local T_i \
         correlation with position not representable on the card",
    ));
    drift.push(crate::report::DriftRow::new(
        "spatial marginals",
        1.0,
        false,
        "radial and vertical birth-profile marginals preserved as discrete histograms",
    ));
    drift.push(crate::report::DriftRow::new(
        "joint correlation",
        1.0 - hist.joint_correlation,
        false,
        "product-form card: half the L1 distance between the true (r, z) birth \
         joint and the product of its marginals is lost (analytic by design)",
    ));
    Ok(EmittedCard { text, drift })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FusionReaction;

    #[test]
    fn point_mono_card_golden() {
        let config =
            PlasmaSourceConfig::point(1.0, 2.0, 3.0, FusionReaction::Dd, 0.0).with_weight(0.5);
        let card = emit_serpent(&config, 21).unwrap();
        assert_eq!(
            card.text,
            "src 1 pos 1 2 3\nsrc 1 erg 2.4495\nsrc 1 wgt 0.5"
        );
        assert_eq!(card.drift.rows[0].rel_drift, 0.0);
        assert!(!card.drift.rows[0].reparsed, "analytic by design");
    }

    #[test]
    fn point_gaussian_card_golden() {
        let config = PlasmaSourceConfig::point(0.0, 0.0, 0.0, FusionReaction::Dt, 20.0);
        let card = emit_serpent(&config, 4).unwrap();
        assert!(card
            .text
            .starts_with("src 1 pos 0 0 0\nsrc 1 erg d1\nsrc 1 wgt 1\nSI1 "));
        assert!(card.text.contains("\nSP1 "));
        let table = config.spectrum().unwrap().tabulate(4, 4.0).unwrap();
        assert!(card.text.contains(&format!(
            "SI1 {}",
            table
                .energies_mev
                .iter()
                .map(|&e| fmt_g6(e))
                .collect::<Vec<_>>()
                .join(" ")
        )));
        assert!((card.drift.rows[0].rel_drift - (1.0 - table.coverage)).abs() < 1e-12);
    }

    #[test]
    fn ring_card_uses_single_bin_histogram_delta() {
        let config = PlasmaSourceConfig::ring(300.0, 25.0, FusionReaction::Dt, 0.0);
        let card = emit_serpent(&config, 21).unwrap();
        assert_eq!(
            card.text,
            "src 1 pos 0 0 25\nsrc 1 rad d1\nsrc 1 erg 14.021\nsrc 1 wgt 1\nSI1 300 300\nSP1 1"
        );
    }

    #[test]
    fn ring_gaussian_numbers_distributions_without_collision() {
        let config = PlasmaSourceConfig::ring(300.0, 0.0, FusionReaction::Dt, 20.0);
        let card = emit_serpent(&config, 8).unwrap();
        let lines: Vec<&str> = card.text.lines().collect();
        assert_eq!(lines[0], "src 1 pos 0 0 0");
        assert_eq!(lines[1], "src 1 rad d1");
        assert_eq!(lines[2], "src 1 erg d2");
        assert_eq!(lines[4], "SI1 300 300");
        assert_eq!(lines[5], "SP1 1");
        assert!(lines[6].starts_with("SI2 "));
        assert!(lines[7].starts_with("SP2 "));
        assert_eq!(lines.len(), 8);
    }

    #[test]
    fn invalid_configs_are_loud() {
        let bad = PlasmaSourceConfig::point(0.0, 0.0, 0.0, FusionReaction::Dt, -1.0);
        assert!(emit_serpent(&bad, 21).is_err());
    }

    #[test]
    fn parametric_card_carries_three_marginals() {
        let config = crate::parametric::tests::iter_h_mode();
        let card = emit_serpent_parametric(&config, 12).unwrap();
        let lines: Vec<&str> = card.text.lines().collect();
        assert_eq!(lines[0], "src 1 pos 0 0 0");
        assert_eq!(lines[1], "src 1 rad d1");
        assert_eq!(lines[2], "src 1 ext d2");
        assert_eq!(lines[3], "src 1 erg d3");
        assert_eq!(lines[4], "src 1 wgt 1");
        for number in 1..=3 {
            assert!(lines[3 + 2 * number].starts_with(&format!("SI{number} ")));
            assert!(lines[4 + 2 * number].starts_with(&format!("SP{number} ")));
        }
        assert_eq!(lines.len(), 11);
        assert_eq!(card.drift.rows.len(), 3);
        assert!(!card.drift.rows[0].reparsed, "analytic by design");
        let sp3: Vec<f64> = lines[10]
            .trim_start_matches("SP3 ")
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect();
        let sum: f64 = sp3.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "energy masses {sum}");
    }
}
