//! 2D debris-flow solver: shallow water equations with Voellmy-Salm rheology.
//!
//! Implements the normative spec `surtgis-flow` v1.0 (2026-07-18). Governing
//! equations (spec §2.1), in conserved variables `U = (h, hu, hv)` over a fixed
//! bed `z(x, y)`:
//!
//! ```text
//! ∂h/∂t  + ∂(hu)/∂x + ∂(hv)/∂y = 0
//! ∂(hu)/∂t + ∂(hu² + ½gh²)/∂x + ∂(huv)/∂y = −g h ∂z/∂x − S_fx
//! ∂(hv)/∂t + ∂(huv)/∂x + ∂(hv² + ½gh²)/∂y = −g h ∂z/∂y − S_fy
//! ```
//!
//! Numerical scheme (spec §3, all choices normative): first-order Godunov
//! finite volumes on the DEM grid, HLL fluxes with Einfeldt wave-speed
//! estimates, hydrostatic reconstruction of Audusse et al. (2004) for the bed
//! source term (well-balanced), positivity-preserving wet/dry handling, and
//! point-implicit Voellmy friction (Coulomb → turbulent) via operator
//! splitting.
//!
//! Entry point: [`Simulation`].
//!
//! ```no_run
//! use surtgis_core::Raster;
//! use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};
//!
//! # fn run(dem: Raster<f32>, release: Raster<f32>) -> Result<(), surtgis_flow::FlowError> {
//! let mut sim = Simulation::new(&dem, &release,
//!                               VoellmyParams::default(), SolverConfig::default())?;
//! sim.step(2.0)?; // advance 2 s of physical time (internally CFL-substepped)
//! let h = &sim.state().h;
//! # Ok(()) }
//! ```

#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(
    // Numeric kernel: f64<->f32 and usize<->index conversions are pervasive
    // and bounds-checked by construction.
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    // Grid dimensions are raster sizes, far below isize::MAX.
    clippy::cast_possible_wrap,
    // Exact float comparisons are intentional: exact-zero semantics (wet/dry,
    // wall fluxes, rotation terms) and bitwise determinism checks (spec T7).
    clippy::float_cmp,
    // (h, hu, hv, u, v, s, z...) is the standard shallow-water notation; renaming
    // to appease the lint would hurt traceability to the spec equations.
    clippy::similar_names,
    clippy::many_single_char_names
)]

mod diagnostics;
mod entrainment;
mod flux;
mod friction;
mod grid;
mod params;
mod state;
mod stepper;

pub use grid::SimGrid;
pub use params::{EntrainmentParams, SolverConfig, VoellmyParams};
pub use state::FlowState;
pub use stepper::Simulation;

/// Standard gravity, in m/s². Spec §2.1: this exact value MUST be used.
pub(crate) const G: f64 = 9.806_65;

/// Errors returned by the `surtgis-flow` public API.
///
/// The public API never panics (spec §4); every failure mode is reported
/// through this enum.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FlowError {
    /// The DEM raster is empty (zero rows or columns).
    #[error("DEM raster is empty")]
    EmptyGrid,

    /// The DEM has rotation terms in its geotransform; the solver requires an
    /// axis-aligned, north-up grid.
    #[error(
        "rotated grids are not supported (row_rotation={row_rotation}, col_rotation={col_rotation})"
    )]
    RotatedGrid {
        /// Rotation about the X axis from the geotransform.
        row_rotation: f64,
        /// Rotation about the Y axis from the geotransform.
        col_rotation: f64,
    },

    /// The DEM cells are not square (spec §3.1: MUST reject).
    #[error("non-square cells: pixel_width={pixel_width}, pixel_height={pixel_height}")]
    NonSquareCells {
        /// Pixel width from the geotransform.
        pixel_width: f64,
        /// Pixel height from the geotransform (sign included).
        pixel_height: f64,
    },

    /// The release raster does not share the DEM's dimensions.
    #[error(
        "grid mismatch: DEM is {expected_rows}x{expected_cols}, other raster is {got_rows}x{got_cols}"
    )]
    GridMismatch {
        /// DEM rows.
        expected_rows: usize,
        /// DEM columns.
        expected_cols: usize,
        /// Rows of the offending raster.
        got_rows: usize,
        /// Columns of the offending raster.
        got_cols: usize,
    },

    /// The release raster shares dimensions but not the DEM's geotransform.
    #[error("geotransform mismatch between DEM and release raster")]
    TransformMismatch,

    /// A parameter is outside its valid range (spec §4).
    #[error("invalid parameter {name}={value}: must satisfy {constraint}")]
    InvalidParam {
        /// Parameter name.
        name: &'static str,
        /// Offending value.
        value: f64,
        /// Human-readable constraint that was violated.
        constraint: &'static str,
    },

    /// The release raster contains a negative thickness.
    #[error("negative release thickness {value} at (row={row}, col={col})")]
    NegativeRelease {
        /// Row of the offending cell.
        row: usize,
        /// Column of the offending cell.
        col: usize,
        /// Offending value in metres.
        value: f32,
    },

    /// The state developed NaN/Inf. The simulation is frozen at the last
    /// valid step for inspection (spec §5: MUST NOT abort).
    #[error("state diverged (NaN/Inf) at t={time} s; state frozen at last valid step")]
    Diverged {
        /// Physical time of the last valid state, in seconds.
        time: f64,
    },

    /// `step` needed more than `max_substeps` CFL substeps to cover the
    /// requested interval (spec §4: cuts pathological calls short).
    #[error("exceeded max_substeps={max} after advancing {advanced} of {requested} s")]
    MaxSubstepsExceeded {
        /// Configured substep limit.
        max: u32,
        /// Physical time actually advanced within this `step` call, in seconds.
        advanced: f64,
        /// Physical time requested, in seconds.
        requested: f64,
    },

    /// The global mass budget invariant was violated with entrainment
    /// active (spec v1.1 §2.4.3): flow volume exceeded release plus eroded
    /// volume, or eroded volume exceeded the erodible budget. The state is
    /// frozen at the last valid substep. This is the hard anti-runaway
    /// guard — it MUST never trip under the per-cell/rate caps; tripping
    /// indicates a solver bug, and it fails loudly instead of growing
    /// silently.
    #[error(
        "mass budget violated at t={time} s: flow volume {flow_volume} m³ vs budget {budget} m³; state frozen at last valid substep"
    )]
    MassBudgetViolated {
        /// Physical time of the last valid state, in seconds.
        time: f64,
        /// Flow volume at the failed check, in m³.
        flow_volume: f64,
        /// Violated bound (release + eroded, or erodible budget), in m³.
        budget: f64,
    },

    /// `set_erodible` was called after the simulation already stepped
    /// (spec v1.1 §4: entrainment activates only before the first step so
    /// runs stay reproducible from their inputs).
    #[error("set_erodible must be called before the first step (t={time} s)")]
    AlreadyStepped {
        /// Physical time already simulated, in seconds.
        time: f64,
    },
}
