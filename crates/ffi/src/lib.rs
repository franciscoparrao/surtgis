//! Stable C ABI for the `surtgis-flow` debris-flow solver (spec §5).
//!
//! Contract highlights (see `include/surtgis_flow.h`, the authoritative
//! header shipped to the `GeodeoSim` Unreal plugin):
//!
//! - Every entry point is wrapped in `catch_unwind`: a panic NEVER crosses
//!   the FFI boundary — it degrades to `SF_ERR_INTERNAL` (spec §5).
//! - `sf_sim` is opaque; it is `Send` but NOT `Sync` — the caller drives it
//!   from a single simulation thread.
//! - All input buffers are row-major, row 0 = north, length `w*h`,
//!   `NoData = NaN`. The solver copies them: the caller keeps ownership.
//! - If the state develops NaN/Inf, `sf_step` returns `SF_ERR_DIVERGED` and
//!   the state stays frozen at the last valid substep for inspection.
//!
//! ABI versioning: bump [`SF_ABI_VERSION`] on ANY signature change.

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    // FFI layer: i32<->usize and f32<->f64 conversions are validated at the
    // boundary before use.
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    // (h, u, v, w) are the C API's own parameter names (spec §5 header);
    // renaming them here would desynchronise the two sides of the contract.
    clippy::many_single_char_names
)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{FlowError, Simulation, SolverConfig, VoellmyParams};

/// Success.
pub const SF_OK: i32 = 0;
/// A pointer was null or a numeric argument was out of range.
pub const SF_ERR_INVALID_ARG: i32 = 1;
/// The provided raster does not match the simulation grid.
pub const SF_ERR_GRID_MISMATCH: i32 = 2;
/// NaN/Inf detected in the state; the simulation is frozen at the last
/// valid substep.
pub const SF_ERR_DIVERGED: i32 = 3;
/// Internal error (including a caught panic).
pub const SF_ERR_INTERNAL: i32 = 4;

/// Current ABI version. MUST be bumped on any signature change (spec §5).
pub const SF_ABI_VERSION: i32 = 1;

/// Wet/dry threshold used to derive velocities in [`sf_read_state`];
/// matches the solver default the FFI constructor uses.
const H_DRY: f32 = 1e-3;

/// Opaque simulation handle behind the C `sf_sim` type.
pub struct SfSim {
    sim: Simulation,
}

fn status_of(e: &FlowError) -> i32 {
    match e {
        FlowError::GridMismatch { .. } | FlowError::TransformMismatch => SF_ERR_GRID_MISMATCH,
        FlowError::Diverged { .. } => SF_ERR_DIVERGED,
        // Parameter/geometry/dt violations (incl. MaxSubstepsExceeded) are
        // caller errors.
        _ => SF_ERR_INVALID_ARG,
    }
}

/// Run `f` with panics converted to `SF_ERR_INTERNAL` (spec §5: MUST NOT
/// panic across the boundary).
fn guarded(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(SF_ERR_INTERNAL)
}

/// Build a georeferenced raster from a raw row-major buffer (row 0 = north).
fn raster_from(buf: &[f32], rows: usize, cols: usize, cellsize: f64) -> Option<Raster<f32>> {
    let mut r = Raster::from_vec(buf.to_vec(), rows, cols).ok()?;
    r.set_transform(GeoTransform::new(
        0.0,
        rows as f64 * cellsize,
        cellsize,
        -cellsize,
    ));
    Some(r)
}

/// Create a simulation from raw DEM and release buffers.
///
/// `dem`/`release`: row-major, row 0 = north, length `w*h`, `NoData` = NaN.
/// The buffers are copied; the caller keeps ownership. On success `*out`
/// holds the new handle (destroy with [`sf_destroy`]); on failure `*out` is
/// set to null.
///
/// # Safety
///
/// `dem` and `release` must point to `w*h` readable `float`s and `out` to a
/// writable pointer slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_create(
    dem: *const f32,
    release: *const f32,
    w: i32,
    h: i32,
    cellsize: f32,
    mu: f32,
    xi: f32,
    out: *mut *mut SfSim,
) -> i32 {
    guarded(|| {
        if out.is_null() {
            return SF_ERR_INVALID_ARG;
        }
        unsafe { out.write(std::ptr::null_mut()) };
        if dem.is_null()
            || release.is_null()
            || w <= 0
            || h <= 0
            || !(cellsize.is_finite() && cellsize > 0.0)
        {
            return SF_ERR_INVALID_ARG;
        }
        let (rows, cols) = (h as usize, w as usize);
        let n = rows * cols;
        let dem_buf = unsafe { std::slice::from_raw_parts(dem, n) };
        let rel_buf = unsafe { std::slice::from_raw_parts(release, n) };
        let cs = f64::from(cellsize);
        let (Some(dem_r), Some(rel_r)) = (
            raster_from(dem_buf, rows, cols, cs),
            raster_from(rel_buf, rows, cols, cs),
        ) else {
            return SF_ERR_INTERNAL;
        };
        let params = VoellmyParams {
            mu,
            xi,
            ..VoellmyParams::default()
        };
        let config = SolverConfig {
            // The render loop calls sf_step with small dt; the guard only
            // cuts truly pathological calls.
            max_substeps: 10_000_000,
            ..SolverConfig::default()
        };
        match Simulation::new(&dem_r, &rel_r, params, config) {
            Ok(sim) => {
                unsafe { out.write(Box::into_raw(Box::new(SfSim { sim }))) };
                SF_OK
            }
            Err(e) => status_of(&e),
        }
    })
}

/// Replace the Voellmy parameters mid-run.
///
/// # Safety
///
/// `sim` must be a live handle from [`sf_create`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_set_params(sim: *mut SfSim, mu: f32, xi: f32, v_stop: f32) -> i32 {
    guarded(|| {
        let Some(s) = (unsafe { sim.as_mut() }) else {
            return SF_ERR_INVALID_ARG;
        };
        match s.sim.set_params(VoellmyParams { mu, xi, v_stop }) {
            Ok(()) => SF_OK,
            Err(e) => status_of(&e),
        }
    })
}

/// Replace the DEM mid-run (live mitigation barriers). Same layout and
/// length as at creation; re-derives the slope cosines.
///
/// # Safety
///
/// `sim` must be a live handle; `dem` must point to `w*h` readable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_update_dem(sim: *mut SfSim, dem: *const f32) -> i32 {
    guarded(|| {
        let Some(s) = (unsafe { sim.as_mut() }) else {
            return SF_ERR_INVALID_ARG;
        };
        if dem.is_null() {
            return SF_ERR_INVALID_ARG;
        }
        let grid = s.sim.grid();
        let (rows, cols, cs) = (grid.rows(), grid.cols(), grid.cellsize());
        let buf = unsafe { std::slice::from_raw_parts(dem, rows * cols) };
        let Some(dem_r) = raster_from(buf, rows, cols, cs) else {
            return SF_ERR_INTERNAL;
        };
        match s.sim.update_dem(&dem_r) {
            Ok(()) => SF_OK,
            Err(e) => status_of(&e),
        }
    })
}

/// Advance `dt` seconds of physical time (CFL-substepped internally).
/// `substeps_out` may be null.
///
/// On `SF_ERR_DIVERGED` the state stays frozen at the last valid substep.
///
/// # Safety
///
/// `sim` must be a live handle; `substeps_out`, if non-null, must be
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_step(sim: *mut SfSim, dt: f32, substeps_out: *mut u32) -> i32 {
    guarded(|| {
        let Some(s) = (unsafe { sim.as_mut() }) else {
            return SF_ERR_INVALID_ARG;
        };
        match s.sim.step(dt) {
            Ok(n) => {
                if !substeps_out.is_null() {
                    unsafe { substeps_out.write(n) };
                }
                SF_OK
            }
            Err(e) => status_of(&e),
        }
    })
}

/// Copy the state into caller-owned buffers of `w*h` floats each. Any
/// pointer may be null to skip that field. `u`/`v` receive velocities in
/// m/s (not momenta), zero below the wet threshold.
///
/// # Safety
///
/// `sim` must be a live handle; non-null buffers must hold `w*h` writable
/// floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_read_state(
    sim: *const SfSim,
    h: *mut f32,
    u: *mut f32,
    v: *mut f32,
) -> i32 {
    guarded(|| {
        let Some(s) = (unsafe { sim.as_ref() }) else {
            return SF_ERR_INVALID_ARG;
        };
        let state = s.sim.state();
        let n = state.h.len();
        if !h.is_null() {
            unsafe { std::slice::from_raw_parts_mut(h, n) }.copy_from_slice(&state.h);
        }
        if !u.is_null() || !v.is_null() {
            for i in 0..n {
                let hh = state.h[i];
                let (ui, vi) = if hh >= H_DRY {
                    (state.hu[i] / hh, state.hv[i] / hh)
                } else {
                    (0.0, 0.0)
                };
                if !u.is_null() {
                    unsafe { u.add(i).write(ui) };
                }
                if !v.is_null() {
                    unsafe { v.add(i).write(vi) };
                }
            }
        }
        SF_OK
    })
}

/// Copy the arrival-time raster (seconds; NaN where the flow never arrived)
/// into a caller-owned buffer of `w*h` floats.
///
/// # Safety
///
/// `sim` must be a live handle; `t_arrival` must hold `w*h` writable floats.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_read_arrival(sim: *const SfSim, t_arrival: *mut f32) -> i32 {
    guarded(|| {
        let Some(s) = (unsafe { sim.as_ref() }) else {
            return SF_ERR_INVALID_ARG;
        };
        if t_arrival.is_null() {
            return SF_ERR_INVALID_ARG;
        }
        let a = s.sim.arrival_times();
        unsafe { std::slice::from_raw_parts_mut(t_arrival, a.len()) }.copy_from_slice(a);
        SF_OK
    })
}

/// Accumulated physical time in seconds (NaN for a null handle).
///
/// # Safety
///
/// `sim` must be a live handle or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_time(sim: *const SfSim) -> f64 {
    unsafe { sim.as_ref() }.map_or(f64::NAN, |s| s.sim.time())
}

/// Total flow volume Σ h·A in m³ (NaN for a null handle).
///
/// # Safety
///
/// `sim` must be a live handle or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_total_mass(sim: *const SfSim) -> f64 {
    catch_unwind(AssertUnwindSafe(|| {
        unsafe { sim.as_ref() }.map_or(f64::NAN, |s| s.sim.total_mass())
    }))
    .unwrap_or(f64::NAN)
}

/// Destroy a handle. Null is a no-op.
///
/// # Safety
///
/// `sim` must be a handle from [`sf_create`] not yet destroyed, or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sf_destroy(sim: *mut SfSim) {
    if !sim.is_null() {
        drop(unsafe { Box::from_raw(sim) });
    }
}

/// ABI version of this library (compare against `SF_ABI_VERSION` in the
/// header at load time).
#[unsafe(no_mangle)]
pub extern "C" fn sf_abi_version() -> i32 {
    SF_ABI_VERSION
}
