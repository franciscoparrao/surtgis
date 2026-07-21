//! Bed entrainment (spec v1.1 §2): erosion law, four hard caps, staged
//! commit.
//!
//! Law (§2.1), evaluated on the POST-friction provisional state (§2.3):
//!
//! ```text
//! ė = K·s·h                       if s ≥ v_entr_min and e < e_max
//! Δe = min(ė·Δt, rate_max·Δt, e_max − e, f_max·h)    (all four MUST)
//! ```
//!
//! Incorporation (§2.2): `h += Δe` with momentum untouched — the bed
//! material starts at rest, so the flow velocity dilutes (`u = hu/h`),
//! which is stabilising. `z` and `e` are NOT mutated here: `Δe` is staged
//! into a scratch buffer and committed by the stepper only after the
//! NaN and mass-budget checks pass, so a failed substep leaves `h`, `z`
//! and `e` mutually consistent (same freeze contract as `Diverged`).

use rayon::prelude::*;

use crate::params::EntrainmentParams;
use crate::state::FlowState;

/// Per-simulation entrainment state (spec v1.1 §4). Adds the two arrays
/// the amended §6 memory budget accounts for (`e_max`, `e`); the Δe
/// staging reuses the stepper's `alpha` scratch.
pub(crate) struct Entrainment {
    pub params: EntrainmentParams,
    /// Maximum erodible depth per cell [m] (user input, 0 = non-erodible).
    pub e_max: Vec<f32>,
    /// Cumulative eroded depth per cell [m].
    pub e: Vec<f32>,
    /// Σ `e_max`·A, in m³ (erodible budget, spec §2.4.3).
    pub budget_total: f64,
    /// Flow volume at activation time (= release volume, since activation
    /// is only allowed before the first step), in m³.
    pub release_volume: f64,
    /// Running Σ Δe·A over the whole run, in f64 (deterministic row-order
    /// accumulation), in m³.
    pub eroded_volume: f64,
}

/// Stage the erosion increments for one substep into `de` (one value per
/// cell; only cells with `Δe > 0` are written — the buffer must arrive
/// zeroed) and apply the mass incorporation to `h`. Purely per-cell,
/// parallel over row bands, deterministic (T11).
pub(crate) fn stage(
    state: &mut FlowState,
    ent: &Entrainment,
    de: &mut [f32],
    cols: usize,
    h_dry: f64,
    dt: f64,
) {
    let k = f64::from(ent.params.k);
    let rate_cap = f64::from(ent.params.rate_max) * dt;
    let v_min = f64::from(ent.params.v_entr_min);
    let f_max = f64::from(ent.params.f_max);
    // Disjoint field borrows: h mutable, momenta read-only.
    let hu = &state.hu;
    let hv = &state.hv;

    state
        .h
        .par_chunks_mut(cols)
        .zip_eq(de.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(r, (h_row, de_row))| {
            let row = r * cols;
            for c in 0..cols {
                let h = f64::from(h_row[c]);
                if h < h_dry {
                    continue;
                }
                let i = row + c;
                let e = f64::from(ent.e[i]);
                let e_max = f64::from(ent.e_max[i]);
                if e >= e_max {
                    continue;
                }
                let u = f64::from(hu[i]) / h;
                let v = f64::from(hv[i]) / h;
                let s = (u * u + v * v).sqrt();
                if s < v_min {
                    continue;
                }
                let d = (k * s * h * dt).min(rate_cap).min(e_max - e).min(f_max * h);
                if d > 0.0 {
                    de_row[c] = d as f32;
                    h_row[c] = (h + d) as f32;
                }
            }
        });
}
