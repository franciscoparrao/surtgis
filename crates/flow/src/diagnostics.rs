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

/// Deterministic parallel sum of a per-cell field: f64 row sums computed
/// in parallel, combined serially in row order — bitwise independent of
/// thread count (needed because the mass-budget invariant runs every
/// substep, spec v1.1 §2.4.3).
pub(crate) fn det_sum(v: &[f32], cols: usize) -> f64 {
    use rayon::prelude::*;
    v.par_chunks(cols)
        .map(|row| {
            let mut s = 0.0f64;
            for &x in row {
                s += f64::from(x);
            }
            s
        })
        .collect::<Vec<_>>()
        .into_iter()
        .sum()
}

/// Record `t` as the arrival time of every newly wetted cell
/// (first time with `h > h_dry`; cells never wetted stay NaN).
/// Per-cell writes only: deterministic under any parallel partition.
pub(crate) fn update_arrivals(arrival: &mut [f32], state: &FlowState, h_dry: f32, t: f64) {
    use rayon::prelude::*;
    arrival
        .par_iter_mut()
        .zip_eq(state.h.par_iter())
        .for_each(|(a, &h)| {
            if a.is_nan() && h > h_dry {
                *a = t as f32;
            }
        });
}
