//! Solver parameters: Voellmy rheology and numerical configuration.

use crate::FlowError;

/// Voellmy-Salm friction parameters (spec §2.2, RAMMS formulation).
///
/// Basal resistance per unit mass, applied opposite to the velocity
/// `u = (u, v)` with `s = |u|`:
///
/// ```text
/// S_f/(ρh) = [ μ·g·cosθ + g·s²/(ξ·h) ] · (u/s, v/s)    if s > 0
/// ```
#[derive(Clone, Debug)]
pub struct VoellmyParams {
    /// Coulomb friction coefficient μ (dimensionless). Physically meaningful
    /// range 0.05–0.4; values down to 0 are accepted for validation cases
    /// (frictionless dam breaks, lake at rest). Default 0.15.
    pub mu: f32,
    /// Turbulent friction coefficient ξ in m/s². Recommended range 100–1000;
    /// `f32::INFINITY` disables the turbulent term. Default 200.
    pub xi: f32,
    /// Stopping threshold in m/s: after friction, cells slower than this are
    /// brought to rest (spec §2.2 detention criterion). Default 0.01.
    pub v_stop: f32,
}

impl Default for VoellmyParams {
    fn default() -> Self {
        Self {
            mu: 0.15,
            xi: 200.0,
            v_stop: 0.01,
        }
    }
}

impl VoellmyParams {
    /// Validate parameter ranges (spec §4: out-of-range params are an error).
    pub(crate) fn validate(&self) -> Result<(), FlowError> {
        if !(self.mu.is_finite() && (0.0..=2.0).contains(&self.mu)) {
            return Err(FlowError::InvalidParam {
                name: "mu",
                value: f64::from(self.mu),
                constraint: "0 <= mu <= 2 and finite (recommended 0.05-0.4)",
            });
        }
        // NaN is rejected here; +inf is explicitly allowed (no turbulent drag).
        if self.xi.is_nan() || self.xi <= 0.0 {
            return Err(FlowError::InvalidParam {
                name: "xi",
                value: f64::from(self.xi),
                constraint: "xi > 0 (recommended 100-1000; +inf disables the term)",
            });
        }
        if !(self.v_stop.is_finite() && self.v_stop >= 0.0) {
            return Err(FlowError::InvalidParam {
                name: "v_stop",
                value: f64::from(self.v_stop),
                constraint: "v_stop >= 0 and finite",
            });
        }
        Ok(())
    }
}

/// Numerical configuration of the solver (spec §3, §4).
#[derive(Clone, Debug)]
pub struct SolverConfig {
    /// CFL number for the adaptive time step (spec §3.7). Must lie in
    /// `(0, 0.5]`. Default 0.45.
    pub cfl: f32,
    /// Wet/dry threshold in metres (spec §3.5): cells with `h < h_dry` carry
    /// no momentum and are excluded from flux computation unless a wet
    /// neighbour can flood them. Default `1e-3`.
    pub h_dry: f32,
    /// Maximum number of CFL substeps a single [`crate::Simulation::step`]
    /// call may take before erroring out (guards pathological calls).
    /// Default 200.
    pub max_substeps: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            cfl: 0.45,
            h_dry: 1e-3,
            max_substeps: 200,
        }
    }
}

impl SolverConfig {
    /// Validate configuration ranges (spec §4).
    pub(crate) fn validate(&self) -> Result<(), FlowError> {
        if !(self.cfl.is_finite() && self.cfl > 0.0 && self.cfl <= 0.5) {
            return Err(FlowError::InvalidParam {
                name: "cfl",
                value: f64::from(self.cfl),
                constraint: "0 < cfl <= 0.5",
            });
        }
        if !(self.h_dry.is_finite() && self.h_dry > 0.0) {
            return Err(FlowError::InvalidParam {
                name: "h_dry",
                value: f64::from(self.h_dry),
                constraint: "h_dry > 0 and finite",
            });
        }
        if self.max_substeps == 0 {
            return Err(FlowError::InvalidParam {
                name: "max_substeps",
                value: 0.0,
                constraint: "max_substeps >= 1",
            });
        }
        Ok(())
    }
}
