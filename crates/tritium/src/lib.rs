//! 1D tritium diffusion-trapping kernel (T1–T2) with analytic permeation gates.
//!
//! A 1D slab `0 ≤ x ≤ L` holds mobile tritium `c_m(x, t)` plus trapped
//! populations `c_{t,j}(x, t)` for `N` extrinsic trap species. Tritium moves
//! by Fickian diffusion with a caller-supplied diffusivity `D(T)`, parks in
//! traps with temperature-dependent McNabb–Foster rates, and enters or
//! leaves through surface laws (fixed concentration, Sieverts/Henry
//! equilibrium, recombination flux, or symmetry). Temperature is a
//! caller-supplied steady profile; there is no heat solve in v1.
//!
//! ```text
//! ∂c_m/∂t = ∂/∂x(D(T) ∂c_m/∂x) − Σⱼ ∂c_{t,j}/∂t + S(x)          (T1)
//! ∂c_{t,j}/∂t = kⱼ(T) c_m (Nⱼ − c_{t,j}) − pⱼ(T) c_{t,j}          (T2)
//! ```
//!
//! Every coefficient is caller data, constant or Arrhenius in the
//! caller-supplied temperature (T-Arr); the kernel ships no material
//! property tables. Trap site densities `N_j` are static in v1 (extrinsic
//! traps only — intrinsic trap evolution is out of scope).
//!
//! Modules: [`params`] (transport/trap data), [`bc`] (surface-law
//! taxonomy), [`mod@solve`] (finite-volume theta stepper, steady solve,
//! and closed-form gate helpers), [`error`] (error type).
//!
//! Gate-to-test mapping (fixtures in `fixtures/tritium/`, replayed in
//! `fixture_tests`): G1/G4 steady linear profiles and G3a/G3b/G3c trap
//! limits as algebraic assertions (`1e-12`), G2 as a transient solve
//! (`1e-6` on the breakthrough curve plus the `L²/6D` time lag), with
//! positivity and mass-balance invariant checks.

#![warn(missing_docs)]

pub mod bc;
pub mod error;
pub mod params;
pub mod solve;

pub use bc::Boundary;
pub use error::{Error, Result};
pub use params::{arrhenius, TransportParams, TrapSpec, GAS_CONSTANT};
pub use solve::{
    breakthrough_ratio, effective_diffusivity, equilibrium_trapped, irreversible_fill,
    recombination_rate_arrhenius, sieverts_concentration, solve, steady_state, time_lag,
    InitialState, Solution, SolverOptions, SteadyState, Theta, TimeGrid,
};

#[cfg(test)]
mod fixture_tests;
