//! Diagnostics: mass accounting and arrival times (spec §4).

use crate::grid::SimGrid;
use crate::state::FlowState;

/// Total flow volume Σ h·A in m³, accumulated in f64 in fixed cell order
/// (deterministic, spec §4).
pub(crate) fn total_mass(state: &FlowState, grid: &SimGrid) -> f64 {
    let area = grid.cellsize() * grid.cellsize();
    let mut sum = 0.0f64;
    for &h in &state.h {
        sum += f64::from(h);
    }
    sum * area
}

/// Record `t` as the arrival time of every newly wetted cell
/// (first time with `h > h_dry`; cells never wetted stay NaN).
pub(crate) fn update_arrivals(arrival: &mut [f32], state: &FlowState, h_dry: f32, t: f64) {
    for (a, &h) in arrival.iter_mut().zip(&state.h) {
        if a.is_nan() && h > h_dry {
            *a = t as f32;
        }
    }
}
