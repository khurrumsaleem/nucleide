---
title: UQ Sampling
sidebar:
  order: 13
---

Nucleide UQ-lite draws seeded multivariate-normal samples over
caller-supplied covariance blocks — the SANDY role (seeded MVN draws over
caller blocks) with no vendored covariance stores. Engine note: SANDY
itself factorises with SVD behind NumPy's PCG64 stream, while this kernel
uses Cholesky-with-eigen-clip behind ChaCha8 — same target distribution,
so draws are not interchangeable, but the moment estimators agree exactly
(the automated validation harness checks `sample_mean`/`sample_cov`
against `Samples.get_mean`/`get_cov` to 1e-9). This tutorial covers the Python API; the
kernel lives in the `linalg` crate (`sample` + `decay` modules).

## Sample a covariance block

`sample_mvn` draws `n` samples `x ~ N(mean, cov)` reproducibly from `seed`:
identical inputs always yield identical samples. The block below is the
synthetic 2x2 sample from `fixtures/uq/cov_2x2.json` (variances 0.25/0.16,
covariance 0.10 — round numbers, no evaluated data):

```python
from nucleide.uq import check_convergence, sample_cov, sample_mean, sample_mvn

mean, cov = [1.0, 2.0], [[0.25, 0.10], [0.10, 0.16]]
out = sample_mvn(mean, cov, 2000, 20260913)
print(out["method"])  # "cholesky" (positive-definite path)
print(sample_mean(out["samples"]))
print(sample_cov(out["samples"]))
```

`method` names the factorisation path: `"cholesky"` for positive-definite
blocks, `"eigen_clip"` when a positive-semidefinite but singular block
falls back to eigen-clipping (then `min_eigen`/`max_eigen` carry the
unclipped extremes). It is reported, never silent.

## Sample with Latin hypercube

`sample_lhs` draws `n` stratified samples `x ~ N(mean, cov)` reproducibly
from `seed`: per dimension, one jittered draw per stratum
(`u = (perm[i] + w) / n`, with a Fisher–Yates permutation `perm` of `0..n-1`
and `w ~ U(0,1)`, both from the seeded `StdRng`), mapped through the
hand-rolled `inv_normal_cdf` to standard normals, then the shared
Cholesky/eigen-clip factor path with `x = μ + Bz`. The block below is the
synthetic 2x2 sample from `fixtures/uq/lhs_2x2.json` (mean `[1, 2]`, variances
0.25/0.16, covariance 0.10 — round numbers, no evaluated data; `n = 5000`,
`k = 5`, seed `20260916`):

```python
from nucleide.uq import sample_lhs

mean, cov = [1.0, 2.0], [[0.25, 0.10], [0.10, 0.16]]
out = sample_lhs(mean, cov, 5000, 20260916)
print(out["method"])  # "cholesky" (positive-definite path)
```

Return shape matches `sample_mvn` (`samples`, `method`, `min_eigen`/
`max_eigen`), and identical inputs always yield identical samples. The
theory page's separate U6/U7 correctness checks (checks G1/G2 there)
require **stratification-exactness** (each dimension hits each of the `n`
strata exactly once at the pinned seed) plus an **LHS-valid moment bound**
(sample mean/covariance within `k` IID standard errors as an *upper* bound
— the IID `k`-SE null is wrong for stratified draws, whose variance is
smaller by construction, never an equality null).
LHS is a draw mode, not a perturbation convention
(`perturb_energies(..., "lhs")` stays an error).

## Check convergence

`check_convergence` recomputes the sample mean and unbiased sample
covariance (`1/(n-1)`, SANDY `get_cov` style) and compares them against the
inputs that generated the draws. Tolerances are caller-supplied:

```python
rep = check_convergence(mean, cov, out["samples"], 0.05, 0.05)
print(rep["passed"], rep["mean_err_max"], rep["cov_err_fro"])
```

## Perturb decay data

`perturb_branches` applies relative deltas to one parent's kept branch
fractions and renormalises to preserve the incoming `1 - BR(SF)` deficit —
the evaluated store drops spontaneous-fission branches, so the synthetic
sample vector in `fixtures/uq/decay_perturb.json` sums to 0.90 and the
output sums to 0.90 too. `perturb_energies` perturbs per-nuclide energies with no sum
constraint (`"relative"` or `"absolute"`, negatives clamped to zero):

```python
from nucleide.uq import perturb_branches, perturb_energies

print(perturb_branches([0.5, 0.3, 0.1], [0.1, -0.2, 0.0]))
print(perturb_energies([0.5, 1.5], [0.2, -0.1], "relative"))
```

`perturb_fission_yields` perturbs a caller-supplied yield block with the
same deficit discipline (`raw = base * (1 + rel)`, negatives clamped to
zero, rescaled to the incoming block sum — independent blocks sum to 2,
cumulative blocks higher; zero-base rows stay zero). The caller supplies
the block and builds `rel` (e.g. `dY/Y` sigmas, or correlated draws from
`sample_mvn`); set selection (e.g. the lowest-energy independent set)
lives with the caller. There are no vendored covariance, branch, or yield
stores anywhere in this path:

```python
from nucleide.uq import perturb_fission_yields

print(perturb_fission_yields([0.9, 0.7, 0.4], [0.1, -0.2, 0.0]))
```

## See also

- `crates/linalg/src/sample.rs` and `crates/linalg/src/decay.rs`
  for the Rust API.
- `tests/test_uq.py` for worked examples.
- [Cross-code validation results](https://github.com/nukehub-dev/nucleide/blob/main/validation/results.md)
  for the SANDY moment cross-check, reproduced by the automated
  [validation harness](../../development/validation.md).
