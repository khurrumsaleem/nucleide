//! Parameter-sweep expansion for multi-case R2S studies (WATTS-class).
//!
//! [`SweepAxis`] declares one named parameter with a finite value list;
//! [`expand_sweep`] forms the deterministic cartesian product of all axes
//! as [`SweepCase`] bundles (`name` joins `axis=value` pairs in axis
//! order); [`assemble_results`] collects one caller-supplied scalar per
//! case into a comparison table, rejecting missing, duplicate, or unknown
//! cases loudly. Pure data plumbing: no file I/O, no template engine, no
//! code execution — rendering inputs and running transport stay with the
//! caller, mirroring how [`crate::workflow`] builds workflows without
//! solving anything.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// One named sweep axis with a finite value list.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepAxis {
    /// Parameter name (non-empty, unique within a sweep).
    pub name: String,
    /// Axis values (non-empty, all finite).
    pub values: Vec<f64>,
}

impl SweepAxis {
    /// Validate a sweep axis (non-empty name, non-empty finite values).
    pub fn new(name: &str, values: Vec<f64>) -> Result<Self> {
        if name.trim().is_empty() {
            return Err(Error::Invalid(
                "sweep axis names an empty parameter".to_string(),
            ));
        }
        if values.is_empty() {
            return Err(Error::Invalid(format!("sweep axis `{name}` has no values")));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(format!(
                "sweep axis `{name}` has non-finite values"
            )));
        }
        Ok(Self {
            name: name.to_string(),
            values,
        })
    }
}

/// One expanded sweep case: parameter assignment plus derived name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepCase {
    /// Case name (`axis=value` pairs joined with `,`, in axis order).
    pub name: String,
    /// Parameter assignment in axis order.
    pub params: Vec<(String, f64)>,
}

impl SweepCase {
    /// Parameter map for template rendering by the caller.
    pub fn mapping(&self) -> BTreeMap<String, f64> {
        self.params.iter().cloned().collect()
    }
}

/// Expand sweep axes to the cartesian product of cases (axis order, then
/// value order; deterministic). Duplicate axis names are an [`Error`].
pub fn expand_sweep(axes: &[SweepAxis]) -> Result<Vec<SweepCase>> {
    let mut seen = BTreeSet::new();
    for axis in axes {
        if axis.name.trim().is_empty() {
            return Err(Error::Invalid(
                "sweep axis names an empty parameter".to_string(),
            ));
        }
        if axis.values.is_empty() || axis.values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(format!(
                "sweep axis `{}` has no (finite) values",
                axis.name
            )));
        }
        if !seen.insert(axis.name.clone()) {
            return Err(Error::Invalid(format!(
                "duplicate sweep axis `{}`",
                axis.name
            )));
        }
    }
    let mut cases = vec![SweepCase {
        name: String::new(),
        params: Vec::new(),
    }];
    for axis in axes {
        let mut next = Vec::with_capacity(cases.len() * axis.values.len());
        for case in &cases {
            for value in &axis.values {
                let mut params = case.params.clone();
                params.push((axis.name.clone(), *value));
                let name = params
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(",");
                next.push(SweepCase { name, params });
            }
        }
        cases = next;
    }
    Ok(cases)
}

/// Collect one scalar result per sweep case into a comparison table.
///
/// Every expanded case must appear exactly once: missing, duplicate, or
/// unknown case names are [`Error`]s, never silent drops.
pub fn assemble_results(
    cases: &[SweepCase],
    results: &[(String, f64)],
) -> Result<BTreeMap<String, f64>> {
    let mut table = BTreeMap::new();
    for (name, value) in results {
        if !value.is_finite() {
            return Err(Error::Invalid(format!(
                "result for case `{name}` is non-finite"
            )));
        }
        if table.insert(name.clone(), *value).is_some() {
            return Err(Error::Invalid(format!(
                "duplicate result for case `{name}`"
            )));
        }
    }
    for case in cases {
        if !table.contains_key(&case.name) {
            return Err(Error::Invalid(format!(
                "missing result for case `{}`",
                case.name
            )));
        }
    }
    if table.len() != cases.len() {
        let unknown: Vec<&String> = table
            .keys()
            .filter(|k| !cases.iter().any(|c| &c.name == *k))
            .collect();
        return Err(Error::Invalid(format!(
            "unknown result case(s): {unknown:?}"
        )));
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Vec<SweepAxis> {
        vec![
            SweepAxis::new("flux_scale", vec![0.5, 1.0]).unwrap(),
            SweepAxis::new("cooling_s", vec![0.0, 3600.0, 7200.0]).unwrap(),
        ]
    }

    #[test]
    fn axis_rejects_bad_inputs() {
        assert!(SweepAxis::new("", vec![1.0]).is_err());
        assert!(SweepAxis::new("x", vec![]).is_err());
        assert!(SweepAxis::new("x", vec![f64::NAN]).is_err());
        assert!(SweepAxis::new("x", vec![f64::INFINITY]).is_err());
    }

    #[test]
    fn expansion_is_deterministic_cartesian() {
        let cases = expand_sweep(&axes()).unwrap();
        assert_eq!(cases.len(), 6);
        assert_eq!(cases[0].name, "flux_scale=0.5,cooling_s=0");
        assert_eq!(cases[5].name, "flux_scale=1,cooling_s=7200");
        assert_eq!(
            cases[2].mapping()["cooling_s"],
            7200.0,
            "{:?}",
            cases[2].params
        );
        // Repeat expansion is identical.
        assert_eq!(expand_sweep(&axes()).unwrap(), cases);
    }

    #[test]
    fn duplicate_axes_rejected() {
        let dup = vec![
            SweepAxis::new("x", vec![1.0]).unwrap(),
            SweepAxis::new("x", vec![2.0]).unwrap(),
        ];
        assert!(expand_sweep(&dup).is_err());
    }

    #[test]
    fn assembly_requires_exact_coverage() {
        let cases = expand_sweep(&axes()).unwrap();
        let full: Vec<(String, f64)> = cases
            .iter()
            .enumerate()
            .map(|(i, c)| (c.name.clone(), i as f64))
            .collect();
        let table = assemble_results(&cases, &full).unwrap();
        assert_eq!(table.len(), 6);
        assert_eq!(table["flux_scale=0.5,cooling_s=0"], 0.0);
        assert!(assemble_results(&cases, &full[..5]).is_err());
        let mut dup = full.clone();
        dup.push(dup[0].clone());
        assert!(assemble_results(&cases, &dup).is_err());
        let mut unknown = full;
        unknown.push(("nope=1".to_string(), 0.0));
        assert!(assemble_results(&cases, &unknown).is_err());
        assert!(assemble_results(
            &cases,
            &[("flux_scale=0.5,cooling_s=0".to_string(), f64::NAN)]
        )
        .is_err());
    }
}
