//! Voellmy friction, point-implicit (spec §3.6).
//!
//! Operator splitting: the explicit hyperbolic step yields a provisional
//! state `(h*, u*, v*)`; friction is then applied in velocity, Coulomb first,
//! turbulent second — this order is normative and avoids the stiffness of the
//! `s²/h` term at small depths without subcycling:
//!
//! ```text
//! a_c = μ g cosθ Δt
//!   |u*| ≤ a_c → u = 0            (Coulomb detention)
//!   else       → u = u*(1 − a_c/|u*|)
//! u = u / (1 + Δt·g·|u*| / (ξ·h*))   (turbulent, using the provisional |u*|)
//! ```
//!
//! After both factors, cells slower than `v_stop` are brought to rest — the
//! physical runout mechanism (flow stops where tanθ < μ, test T5).

use crate::G;
use crate::grid::SimGrid;
use crate::params::{SolverConfig, VoellmyParams};
use crate::state::FlowState;

/// Apply point-implicit Voellmy friction in place to the provisional state.
pub(crate) fn apply(
    state: &mut FlowState,
    grid: &SimGrid,
    params: &VoellmyParams,
    config: &SolverConfig,
    dt: f64,
) {
    let mu = f64::from(params.mu);
    let xi = f64::from(params.xi);
    let v_stop = f64::from(params.v_stop);
    let h_dry = f64::from(config.h_dry);

    for i in 0..state.h.len() {
        if grid.solid_at(i) {
            continue;
        }
        let h = f64::from(state.h[i]);
        if h < h_dry {
            continue;
        }
        let mut u = f64::from(state.hu[i]) / h;
        let mut v = f64::from(state.hv[i]) / h;
        let s_star = (u * u + v * v).sqrt();
        if s_star == 0.0 {
            continue;
        }

        // Coulomb (normative first):
        let a_c = mu * G * grid.cos_theta_at(i) * dt;
        if s_star <= a_c {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            continue;
        }
        let coulomb = 1.0 - a_c / s_star;
        u *= coulomb;
        v *= coulomb;

        // Turbulent (normative second, denominator uses the provisional |u*|):
        let denom = 1.0 + dt * G * s_star / (xi * h);
        u /= denom;
        v /= denom;

        if (u * u + v * v).sqrt() < v_stop {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
        } else {
            state.hu[i] = (u * h) as f32;
            state.hv[i] = (v * h) as f32;
        }
    }
}
