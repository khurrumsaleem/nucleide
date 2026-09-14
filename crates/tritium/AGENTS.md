# `crates/tritium` AGENTS.md

## Purpose

1D tritium diffusion-trapping kernel: Fickian mobile transport (T1) coupled
to N extrinsic McNabb–Foster trap species (T2) on a slab with
caller-supplied temperature, plus the Dirichlet/Sieverts/Henry/zero-flux
surface taxonomy. Pure 1D finite-volume PDE; no multi-D, no FEM, no heat
solve, no property tables.

## Ownership

Owns `crates/tritium/src/` (`params.rs`, `bc.rs`, `solve.rs`, `error.rs`),
the Python surface (`nucleide.tritium`, `tritium_*` in `_internal`),
`tests/test_tritium.py`, the `tritiumBreakthrough` WASM facade, the
`TritiumBreakthrough` demo, and `docs/tutorials/interactive/tritium.mdx`.
Replays (never rewrites) `fixtures/tritium/`; shares `linalg::tridiag`
with no other consumer yet.

## Local Contracts

- Equation set (T1–T2) is pinned: mobile balance, McNabb–Foster trap
  kinetics with the Langmuir equilibrium, Arrhenius caller-data stance,
  surface taxonomy. Derivations live in `docs/theory/tritium.mdx`; code
  comments cite equation labels, never external paths.
- Boundary taxonomy: `dirichlet` (default surface), `sieverts`
  (`c = K_S sqrt(p)`), `henry` (linear variant), `zero_flux` (symmetry /
  impermeable wall); `recombination` (`J = K_r c²`) is closed in the
  steady state by the exact face-response construction (G5) and in the
  transient by the per-step face Newton sharing the affine face-response
  construction (G6a–G6e) — never a silent pass.
- Discretization rule: cell-centred finite volume with implicit
  theta-stepping (Crank–Nicolson default, backward Euler on request);
  diffusion implicit, traps via the exact per-cell backward-Euler map with
  Picard coupling to rtol/atol. Method changes re-run the G1–G4 gates.
- Synthetic fixtures only: hand-built params plus closed-form values with
  recorded provenance; never evaluated-library data. `fixtures/tritium/`
  is read-only for this crate (no new fixture files; analytic oracles
  already exist).
- Oracle tolerances are gates, not claims: 1e-12 algebraic (G1/G3a/G3b
  isotherm/G3c/G4/G5a steady, G5b recovery at 1e-6), 1e-6 transient
  breakthrough curve (G2), 2% time-lag
  intercept, roundoff mass conservation. G6 has no algebraic oracle
  (asymptotic + self-convergence, plus the G6e independent cross-check):
  G6a steady-asymptote at 1e-6 relative on fluxes and flux balance at
  t = 60·t_lag with the G5a
  linear profile at 1e-9, G6b K_r→∞ flux recovery at 1e-6 and K_r→0
  zero-flux profile at 1e-5 absolute, G6c discrete mass balance at 1e-12
  relative against the reported K_r cf² faces, G6d dt-halving order
  bands 1.5–2.5 (Crank–Nicolson) and 0.7–1.3 (backward Euler) on two
  successive halvings in the resolved band (t = 1200 s, dt ≤ 3 s), G6e
  mid-transient cross-check at t = 0.5/1/2·t_lag against an independent
  method-of-lines solver (node-centred central FD + explicit RK4,
  ghost-node recombination end) at 1.5e-4 relative on the mobile-profile
  max-norm (6e-4 trapped profile, outlet flux 5e-3 relative with a
  1e-3·J_ss floor), trap-free and trap-coupled.
- Out of scope (do not expand here): multi-D/FEM, heat coupling,
  plasma-facing implantation models, TBR coupling, FESTIM-file I/O, any
  dolfinx linkage, vendored D/K_S tables (caller-supplied only).
- WASM/tutorial surface (owned): `tritiumBreakthrough` in `bindings/wasm`
  (trap-free permeation solve returning the `J/J_ss` series plus `t_lag`
  and `J_ss`; thin facade, fixed 200-cell grid), the
  `TritiumBreakthrough` demo in
  `website/src/components/interactive/TritiumBreakthrough.tsx`, and the
  `docs/tutorials/interactive/tritium.mdx` page. Demo presets stay
  synthetic (`fixtures/tritium/` values only).

## Work Guidance

- New analyses add a module plus Python/tests/docs in the same change
  (bindings thin, no logic).
- Keep the layering: this crate depends on `nucleide-linalg` (tridiag)
  only; bindings depend on it, never the reverse.
- No `validation/*_vs_*.py` script: no external oracle exists (analytic-gate
  replay only, per the theory validation section). A FESTIM
  cross-check, if ever attempted, stays a loud-SKIP probe and never adds
  FESTIM/FEniCS to the validation container.

## Verification

- `cargo test -p nucleide-tritium` (includes fixture replay).
- `pytest tests/test_tritium.py` after `maturin develop`.

## Child NAD Index

None.
