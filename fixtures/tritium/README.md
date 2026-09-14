# fixtures/tritium/ — synthetic 1D tritium-transport analytic-gate oracles

Hand-built synthetic data only — no laboratory, evaluated-library, or
code-output values. Params use round numbers chosen for this repo; expected
values are closed-form consequences of the T1–T2/G1–G4 equation set pinned in
the theory page, computed once in float64 with the provenance recorded in
each file. No property tables are vendored (caller-supplied Arrhenius stance).

- `g1_steady.json` — G1 Dirichlet steady state (`D = 1e-9 m²/s`,
  `L = 1e-3 m`, `c0 = 1.0`, `cL = 0.0 mol/m³`): linear profile at five
  nodes, `J_ss = 1e-6 mol/m²/s`, inventory `5e-4 mol/m²`.
- `g2_timelag.json` — G2 permeation transient on the same slab
  (`t_lag = L²/6D = 166.67 s`): normalized outlet flux
  `J(L,t)/J_ss = 1 + 2Σ(−1)ⁿexp(−Dn²π²t/L²)` at seven log-spaced times,
  series truncated at term `< 1e-18` (solver replay tolerance `1e-6`).
- `g3a_oriani.json` — G3a low-occupancy single-trap limit
  (`Kc = 1e-3 ≪ 1`): Oriani retardation `R = 1 + KN = 4`,
  `D_eff = D/4 = 2.5e-10 m²/s`.
- `g3b_saturated.json` — G3b high-occupancy limit (`Kc = 10 ≫ 1`):
  trapped profile `c_t = N·Kc/(1+Kc)` at five nodes plus the exact
  trapped inventory `2L(1 − ln 2)`.
- `g3c_irreversible.json` — G3c irreversible trap (`p → 0`) at fixed
  mobile `c`: `c_t(t) = N(1 − e^{−kct})` at seven times.
- `g4_sieverts.json` — G4 Sieverts steady state (`K_S = 2.0`,
  `p1 = 16 Pa`, `p2 = 1 Pa`): `c0 = 8.0`, `cL = 2.0 mol/m³`, linear
  profile, `J = 6e-6 mol/m²/s`, permeability `Φ = D·K_S`.
