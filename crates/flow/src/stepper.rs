//! Main solver loop: CFL substepping, flux application, positivity limiting.
//!
//! Each substep is two passes over the grid (spec §6): (1) face fluxes and
//! per-cell accumulation, (2) friction and clamps. Interior faces are
//! recomputed from each adjacent cell rather than stored — the flux function
//! is deterministic, so both evaluations agree bitwise, no face arrays are
//! needed (memory budget §6), and the loop parallelises without
//! synchronisation.

use rayon::prelude::*;
use surtgis_core::Raster;

use crate::diagnostics;
use crate::flux::{self, NumFlux, Side};
use crate::friction;
use crate::grid::SimGrid;
use crate::params::{SolverConfig, VoellmyParams};
use crate::state::FlowState;
use crate::{FlowError, G};

/// A running debris-flow simulation (spec §4).
///
/// Owns the domain, the conserved state and the diagnostics. Construct with
/// [`Simulation::new`], advance with [`Simulation::step`].
pub struct Simulation {
    grid: SimGrid,
    state: FlowState,
    /// Scratch state written by each substep; swapped in on success so a
    /// diverged substep leaves `state` frozen at the last valid step.
    scratch: FlowState,
    /// Per-cell positivity limiter α ∈ (0, 1] recomputed every substep.
    alpha: Vec<f32>,
    arrival: Vec<f32>,
    time: f64,
    params: VoellmyParams,
    config: SolverConfig,
}

/// Which of a cell's four faces is being evaluated.
#[derive(Clone, Copy)]
enum Face {
    East,
    West,
    North,
    South,
}

impl Face {
    const ALL: [Face; 4] = [Face::East, Face::West, Face::North, Face::South];

    /// `true` if the cell sits on the left (lower-coordinate) side of the
    /// face in the canonical flux orientation (+x for E/W, +y for N/S; row 0
    /// is north, so the south neighbour is the left side of a N/S face).
    fn cell_is_left(self) -> bool {
        matches!(self, Face::East | Face::North)
    }

    fn is_x(self) -> bool {
        matches!(self, Face::East | Face::West)
    }
}

/// Flux across one face of a cell, with the neighbour cell index (if the
/// other side is a real cell; `None` for domain-edge ghosts and walls).
struct CellFace {
    flux: NumFlux,
    cell_is_left: bool,
    is_x: bool,
    neighbor: Option<usize>,
}

impl Simulation {
    /// Create a simulation from a DEM and a release-thickness raster on the
    /// same grid, with zero initial velocity (spec §2.3).
    ///
    /// # Errors
    ///
    /// Mismatched grids, non-square cells, rotated grids, out-of-range
    /// parameters or negative release thicknesses (spec §4).
    pub fn new(
        dem: &Raster<f32>,
        release: &Raster<f32>,
        params: VoellmyParams,
        config: SolverConfig,
    ) -> Result<Simulation, FlowError> {
        params.validate()?;
        config.validate()?;
        let grid = SimGrid::from_raster(dem)?;
        grid.check_compatible(release)?;

        let n = grid.len();
        let cols = grid.cols();
        let mut state = FlowState::zeros(n);
        for (i, &v) in release.data().iter().enumerate() {
            // NoData / non-finite release cells contribute no material.
            if release.is_nodata(v) || !v.is_finite() {
                continue;
            }
            if v < 0.0 {
                return Err(FlowError::NegativeRelease {
                    row: i / cols,
                    col: i % cols,
                    value: v,
                });
            }
            if !grid.solid_at(i) {
                state.h[i] = v;
            }
        }

        let mut arrival = vec![f32::NAN; n];
        diagnostics::update_arrivals(&mut arrival, &state, config.h_dry, 0.0);

        Ok(Simulation {
            scratch: state.clone(),
            alpha: vec![1.0; n],
            grid,
            state,
            arrival,
            time: 0.0,
            params,
            config,
        })
    }

    /// Advance exactly `dt` seconds of physical time, subcycling internally
    /// at the CFL-stable step (spec §3.7). Returns the number of substeps.
    ///
    /// # Errors
    ///
    /// [`FlowError::Diverged`] if the state develops NaN/Inf (the state stays
    /// frozen at the last valid substep) and
    /// [`FlowError::MaxSubstepsExceeded`] if more than
    /// `config.max_substeps` substeps would be needed.
    pub fn step(&mut self, dt: f32) -> Result<u32, FlowError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(FlowError::InvalidParam {
                name: "dt",
                value: f64::from(dt),
                constraint: "dt >= 0 and finite",
            });
        }
        let requested = f64::from(dt);
        let mut remaining = requested;
        // Absorb FP residue so `remaining` cannot asymptote to 0 over many
        // tiny trailing substeps.
        let eps = requested * 1e-12;
        let mut substeps: u32 = 0;
        while remaining > eps {
            if substeps >= self.config.max_substeps {
                return Err(FlowError::MaxSubstepsExceeded {
                    max: self.config.max_substeps,
                    advanced: requested - remaining,
                    requested,
                });
            }
            let dt_sub = self.stable_dt().map_or(remaining, |d| d.min(remaining));
            self.substep(dt_sub)?;
            substeps += 1;
            self.time += dt_sub;
            remaining -= dt_sub;
            diagnostics::update_arrivals(
                &mut self.arrival,
                &self.state,
                self.config.h_dry,
                self.time,
            );
        }
        Ok(substeps)
    }

    /// Current conserved state.
    #[must_use]
    pub fn state(&self) -> &FlowState {
        &self.state
    }

    /// Simulation domain (DEM geometry, solid mask).
    #[must_use]
    pub fn grid(&self) -> &SimGrid {
        &self.grid
    }

    /// Accumulated physical time in seconds.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Total flow volume Σ h·A in m³, accumulated in f64 (spec §4).
    #[must_use]
    pub fn total_mass(&self) -> f64 {
        diagnostics::total_mass(&self.state, &self.grid)
    }

    /// Per-cell first arrival time (first `t` with `h > h_dry`), NaN where
    /// the flow never arrived. Row-major, row 0 = north.
    #[must_use]
    pub fn arrival_times(&self) -> &[f32] {
        &self.arrival
    }

    /// Replace the DEM mid-run (live mitigation barriers, spec §4).
    /// Re-derives the slope cosines; flow thickness on cells that became
    /// `NoData` is discarded.
    ///
    /// # Errors
    ///
    /// [`FlowError::GridMismatch`] / [`FlowError::TransformMismatch`] if the
    /// new DEM is not on the simulation grid.
    pub fn update_dem(&mut self, dem: &Raster<f32>) -> Result<(), FlowError> {
        self.grid.check_compatible(dem)?;
        self.grid.replace_dem(dem);
        for i in 0..self.grid.len() {
            if self.grid.solid_at(i) {
                self.state.h[i] = 0.0;
                self.state.hu[i] = 0.0;
                self.state.hv[i] = 0.0;
            }
        }
        Ok(())
    }

    /// CFL-stable time step over wet cells (spec §3.7), `None` if nothing is
    /// wet (the caller then jumps over the remaining interval).
    /// The reduction is a plain `max`, which is exact in floating point:
    /// the result is independent of the parallel partition and thread count,
    /// so the CFL step is deterministic by construction (spec §4, T7).
    fn stable_dt(&self) -> Option<f64> {
        let h_dry = f64::from(self.config.h_dry);
        let cols = self.grid.cols();
        let s_max = (0..self.grid.rows())
            .into_par_iter()
            .map(|r| {
                let mut row_max = 0.0f64;
                for c in 0..cols {
                    let i = r * cols + c;
                    let h = f64::from(self.state.h[i]);
                    if h < h_dry || self.grid.solid_at(i) {
                        continue;
                    }
                    let u = f64::from(self.state.hu[i]) / h;
                    let v = f64::from(self.state.hv[i]) / h;
                    let c_wave = (G * h).sqrt();
                    row_max = row_max.max(u.abs().max(v.abs()) + c_wave);
                }
                row_max
            })
            .reduce(|| 0.0f64, f64::max);
        (s_max > 0.0).then(|| f64::from(self.config.cfl) * self.grid.cellsize() / s_max)
    }

    /// One explicit substep of size `dt`: hyperbolic update with positivity
    /// limiting, then point-implicit friction, then wet/dry cleanup.
    fn substep(&mut self, dt: f64) -> Result<(), FlowError> {
        let lambda = dt / self.grid.cellsize();

        let mu = f64::from(self.params.mu);
        compute_alpha(
            &self.grid,
            &self.state,
            &self.config,
            mu,
            dt,
            &mut self.alpha,
        );
        self.scratch.copy_from(&self.state);
        apply_fluxes(
            &self.grid,
            &self.state,
            &self.config,
            mu,
            &self.alpha,
            lambda,
            &mut self.scratch,
        );
        friction::apply(
            &mut self.scratch,
            &self.grid,
            &self.params,
            &self.config,
            dt,
        );

        // Wet/dry cleanup (spec §3.5): clamp FP dust to exact 0 and strip
        // momentum from cells below the dry threshold.
        let h_dry = self.config.h_dry;
        self.scratch
            .h
            .par_iter_mut()
            .zip_eq(self.scratch.hu.par_iter_mut())
            .zip_eq(self.scratch.hv.par_iter_mut())
            .for_each(|((h, hu), hv)| {
                if *h < 0.0 {
                    *h = 0.0;
                }
                if *h < h_dry {
                    *hu = 0.0;
                    *hv = 0.0;
                }
            });

        if self.scratch.has_non_finite() {
            // Spec §5: freeze at the last valid step, never abort.
            return Err(FlowError::Diverged { time: self.time });
        }
        std::mem::swap(&mut self.state, &mut self.scratch);
        Ok(())
    }
}

/// Velocity components of a cell, zero below the dry threshold (dry cells
/// carry no momentum, spec §3.5).
#[inline]
fn cell_velocity(state: &FlowState, i: usize, h_dry: f64) -> (f64, f64, f64) {
    let h = f64::from(state.h[i]);
    if h >= h_dry {
        (h, f64::from(state.hu[i]) / h, f64::from(state.hv[i]) / h)
    } else {
        (h, 0.0, 0.0)
    }
}

/// One side of a face for cell `i`, rotated so `un` is the face-normal
/// velocity (`u` for x-faces, `v` for y-faces).
#[inline]
fn cell_side(grid: &SimGrid, state: &FlowState, i: usize, is_x: bool, h_dry: f64) -> Side {
    let (h, u, v) = cell_velocity(state, i, h_dry);
    let (un, ut) = if is_x { (u, v) } else { (v, u) };
    Side {
        h,
        un,
        ut,
        z: grid.z_at(i),
    }
}

/// Evaluate the flux across one face of cell (r, c).
///
/// Returns `None` for inactive faces (both sides below the dry threshold —
/// dry cells exchange nothing unless a wet neighbour floods them, spec §3.5).
/// Domain edges use a transmissive mirror ghost (free outflow), `NoData`
/// neighbours a reflective ghost with negated normal velocity (spec §2.3).
fn cell_face(
    grid: &SimGrid,
    state: &FlowState,
    config: &SolverConfig,
    mu: f64,
    r: usize,
    c: usize,
    face: Face,
) -> Option<CellFace> {
    let h_dry = f64::from(config.h_dry);
    let cols = grid.cols();
    let rows = grid.rows();
    let i = r * cols + c;
    let is_x = face.is_x();
    let cell_is_left = face.cell_is_left();

    let neighbor: Option<usize> = match face {
        Face::East => (c + 1 < cols).then(|| i + 1),
        Face::West => (c > 0).then(|| i - 1),
        Face::North => (r > 0).then(|| i - cols),
        Face::South => (r + 1 < rows).then(|| i + cols),
    };

    let mine = cell_side(grid, state, i, is_x, h_dry);
    let (other_side, other_idx) = match neighbor {
        Some(j) if !grid.solid_at(j) => (cell_side(grid, state, j, is_x, h_dry), Some(j)),
        Some(_) => {
            // Reflective wall: mirror with negated normal velocity.
            (
                Side {
                    un: -mine.un,
                    ..mine
                },
                None,
            )
        }
        // Transmissive domain edge: zero-gradient mirror ghost.
        None => (mine, None),
    };

    if mine.h < h_dry && other_side.h < h_dry {
        return None;
    }

    // Static-yield detention (Coulomb closure of spec §2.2, required by T5):
    // between two cells at rest, the HLL mass flux carries a diffusive term
    // ~ -c·Δη/2 that makes frozen deposits creep indefinitely even though
    // their momentum is zeroed every step. When both sides are exactly at
    // rest and the free-surface gradient is below the Coulomb yield μ·cosθ,
    // the *mass* flux is suppressed — the pressure (normal-momentum) fluxes
    // are kept, so the well-balanced Audusse property is untouched and the
    // residual driving force is absorbed by the Coulomb detention in the
    // friction step (a sub-yield kick is ≤ a_c by construction). With μ = 0
    // the threshold is 0 and water levels out exactly as before.
    let static_below_yield =
        mine.un == 0.0 && mine.ut == 0.0 && other_side.un == 0.0 && other_side.ut == 0.0 && {
            let d_eta = ((other_side.h + other_side.z) - (mine.h + mine.z)).abs();
            let cos_other =
                other_idx.map_or_else(|| grid.cos_theta_at(i), |j| grid.cos_theta_at(j));
            let cos_avg = 0.5 * (grid.cos_theta_at(i) + cos_other);
            d_eta <= mu * cos_avg * grid.cellsize()
        };

    let (l, rr) = if cell_is_left {
        (mine, other_side)
    } else {
        (other_side, mine)
    };
    let mut flux = flux::face_flux(l, rr);
    if static_below_yield {
        // Suppress only the diffusive mass exchange (and the mass-carried
        // transverse momentum); identical on both sides of the face, so mass
        // conservation is preserved exactly.
        flux.mass = 0.0;
        flux.mom_t = 0.0;
    }
    Some(CellFace {
        flux,
        cell_is_left,
        is_x,
        neighbor: other_idx,
    })
}

/// Pass 1: per-cell positivity limiter α = min(1, h·Δx / (Δt·Σ outgoing mass
/// flux)) — scaling every outgoing flux by the donor's α guarantees h ≥ 0
/// (spec §3.5, positivity preserving).
fn compute_alpha(
    grid: &SimGrid,
    state: &FlowState,
    config: &SolverConfig,
    mu: f64,
    dt: f64,
    alpha: &mut [f32],
) {
    let dx = grid.cellsize();
    let cols = grid.cols();
    alpha
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, alpha_row)| {
            for (c, a_out) in alpha_row.iter_mut().enumerate() {
                let i = r * cols + c;
                *a_out = 1.0;
                let h = f64::from(state.h[i]);
                if grid.solid_at(i) || h <= 0.0 {
                    continue;
                }
                let mut outflow = 0.0f64;
                for face in Face::ALL {
                    if let Some(cf) = cell_face(grid, state, config, mu, r, c, face) {
                        let out = if cf.cell_is_left {
                            cf.flux.mass
                        } else {
                            -cf.flux.mass
                        };
                        if out > 0.0 {
                            outflow += out;
                        }
                    }
                }
                if outflow > 0.0 {
                    let a = (h * dx) / (dt * outflow);
                    if a < 1.0 {
                        *a_out = a as f32;
                    }
                }
            }
        });
}

/// Pass 2: accumulate the α-limited face fluxes into the scratch state.
/// Each face's α is the donor cell's (the side mass flows out of); ghost
/// donors (inflow through a transmissive edge) are unlimited.
/// Parallelisation (spec §6): row bands via `par_chunks_mut`; each cell
/// accumulates its own four face fluxes in a fixed order, so no thread ever
/// writes outside its band and faces on band boundaries are simply
/// recomputed by both sides (bit-identical — the flux function is pure).
/// Cell results are therefore independent of the partition and thread
/// count: determinism (T7) holds by construction.
#[allow(clippy::too_many_arguments)]
fn apply_fluxes(
    grid: &SimGrid,
    state: &FlowState,
    config: &SolverConfig,
    mu: f64,
    alpha: &[f32],
    lambda: f64,
    scratch: &mut FlowState,
) {
    let cols = grid.cols();
    scratch
        .h
        .par_chunks_mut(cols)
        .zip_eq(scratch.hu.par_chunks_mut(cols))
        .zip_eq(scratch.hv.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(r, ((h_row, hu_row), hv_row))| {
            for c in 0..cols {
                let i = r * cols + c;
                if grid.solid_at(i) {
                    continue;
                }
                let mut dh = 0.0f64;
                let mut dhu = 0.0f64;
                let mut dhv = 0.0f64;
                for face in Face::ALL {
                    let Some(cf) = cell_face(grid, state, config, mu, r, c, face) else {
                        continue;
                    };
                    let donor = if (cf.flux.mass >= 0.0) == cf.cell_is_left {
                        Some(i)
                    } else {
                        cf.neighbor
                    };
                    let a = donor.map_or(1.0, |d| f64::from(alpha[d]));
                    // Cell on the left of the face: the face normal points out
                    // of the cell, so the flux leaves (−λF); on the right it
                    // enters.
                    let sign = if cf.cell_is_left { -1.0 } else { 1.0 };
                    let mom_n = if cf.cell_is_left {
                        cf.flux.mom_l
                    } else {
                        cf.flux.mom_r
                    };
                    let s = sign * lambda * a;
                    dh += s * cf.flux.mass;
                    if cf.is_x {
                        dhu += s * mom_n;
                        dhv += s * cf.flux.mom_t;
                    } else {
                        dhv += s * mom_n;
                        dhu += s * cf.flux.mom_t;
                    }
                }
                h_row[c] = (f64::from(state.h[i]) + dh) as f32;
                hu_row[c] = (f64::from(state.hu[i]) + dhu) as f32;
                hv_row[c] = (f64::from(state.hv[i]) + dhv) as f32;
            }
        });
}
