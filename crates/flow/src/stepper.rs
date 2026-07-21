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
use crate::entrainment::{self, Entrainment};
use crate::flux::{self, NumFlux, Side};
use crate::friction;
use crate::grid::SimGrid;
use crate::params::{EntrainmentParams, SolverConfig, VoellmyParams};
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
    /// Bed-entrainment state (spec v1.1 §2); `None` = exact v1.0 behaviour.
    entrainment: Option<Entrainment>,
    /// `true` once any substep ran — gates `set_erodible` (spec v1.1 §4).
    has_stepped: bool,
}

/// Rows per parallel band (spec §6). Faces interior to a band are evaluated
/// once per pass; the band's boundary face-rows are recomputed by the
/// adjacent band — recomputing is cheaper than synchronising, and the flux
/// function is pure, so both evaluations are bit-identical. 16 rows amortise
/// the duplicated boundary work to ~12% extra flux evaluations.
const BAND_ROWS: usize = 16;

/// Flux across a face plus the real cell index on each side (`None` when
/// that side is a domain-edge ghost or a `NoData` wall).
struct EvaluatedFace {
    flux: NumFlux,
    l_donor: Option<usize>,
    r_donor: Option<usize>,
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
            entrainment: None,
            has_stepped: false,
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

    /// Replace the Voellmy parameters mid-run (spec §5 `sf_set_params`;
    /// interactive tuning from the Unreal side).
    ///
    /// # Errors
    ///
    /// [`FlowError::InvalidParam`] if a parameter is out of range — the
    /// previous parameters stay in effect.
    pub fn set_params(&mut self, params: VoellmyParams) -> Result<(), FlowError> {
        params.validate()?;
        self.params = params;
        Ok(())
    }

    /// Activate bed entrainment (spec v1.1 §2, §4). `e_max` is the maximum
    /// erodible depth per cell in metres, on the DEM grid (`NoData`/NaN and
    /// solid cells count as 0). Callable only before the first step so runs
    /// stay reproducible from their inputs; calling again before stepping
    /// replaces the previous activation.
    ///
    /// # Errors
    ///
    /// [`FlowError::AlreadyStepped`] after the first step, grid/transform
    /// mismatches, out-of-range parameters, or negative `e_max` values.
    pub fn set_erodible(
        &mut self,
        e_max: &Raster<f32>,
        params: EntrainmentParams,
    ) -> Result<(), FlowError> {
        params.validate()?;
        if self.has_stepped {
            return Err(FlowError::AlreadyStepped { time: self.time });
        }
        self.grid.check_compatible(e_max)?;
        let n = self.grid.len();
        let mut emax = vec![0.0f32; n];
        for (i, &v) in e_max.data().iter().enumerate() {
            if e_max.is_nodata(v) || !v.is_finite() {
                continue; // treated as non-erodible
            }
            if v < 0.0 {
                return Err(FlowError::InvalidParam {
                    name: "e_max",
                    value: f64::from(v),
                    constraint: "erodible depths must be >= 0",
                });
            }
            if !self.grid.solid_at(i) {
                emax[i] = v;
            }
        }
        let area = self.grid.cellsize() * self.grid.cellsize();
        let mut budget = 0.0f64;
        for &v in &emax {
            budget += f64::from(v);
        }
        self.entrainment = Some(Entrainment {
            params,
            e_max: emax,
            e: vec![0.0; n],
            budget_total: budget * area,
            release_volume: self.total_mass(),
            eroded_volume: 0.0,
        });
        Ok(())
    }

    /// Cumulative eroded depth per cell in metres, row-major. Empty slice
    /// until [`Simulation::set_erodible`] activates entrainment (deviation
    /// from the spec-v1.1 wording "0 sin entrainment": an empty slice makes
    /// the inactive state explicit instead of allocating a zero grid).
    #[must_use]
    pub fn eroded_depth(&self) -> &[f32] {
        self.entrainment.as_ref().map_or(&[], |ent| &ent.e)
    }

    /// Total eroded volume Σ Δe·A in m³, f64 accumulated deterministically
    /// (0 without entrainment).
    #[must_use]
    pub fn total_eroded(&self) -> f64 {
        self.entrainment
            .as_ref()
            .map_or(0.0, |ent| ent.eroded_volume)
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

        // Entrainment (spec v1.1 §2.3: after friction, before cleanup).
        // Δe is STAGED into the alpha buffer (free after apply_fluxes,
        // rezeroed here) and committed to e/z only after the NaN and budget
        // checks pass, so a failed substep freezes a consistent state.
        let entrainment_active = self.entrainment.is_some();
        if let Some(ent) = &self.entrainment {
            self.alpha.par_iter_mut().for_each(|a| *a = 0.0);
            entrainment::stage(
                &mut self.scratch,
                ent,
                &mut self.alpha,
                self.grid.cols(),
                f64::from(self.config.h_dry),
                dt,
            );
        }

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

        // Mass-budget invariant + commit (spec v1.1 §2.4.3): checked every
        // substep, BEFORE the swap — a violation freezes the last valid
        // state exactly like Diverged. Under the per-cell/rate caps it can
        // only trip on a solver bug: it fails loudly instead of letting
        // volume run away (the r.avaflow 631M m³ failure mode).
        if entrainment_active {
            let cols = self.grid.cols();
            let area = self.grid.cellsize() * self.grid.cellsize();
            let d_eroded = diagnostics::det_sum(&self.alpha, cols) * area;
            let v_flow = diagnostics::det_sum(&self.scratch.h, cols) * area;
            let ent = self.entrainment.as_mut().expect("checked active");
            let eroded_new = ent.eroded_volume + d_eroded;
            let tol = 1e-4 * (ent.release_volume + ent.budget_total).max(1.0);
            if v_flow > ent.release_volume + eroded_new + tol {
                return Err(FlowError::MassBudgetViolated {
                    time: self.time,
                    flow_volume: v_flow,
                    budget: ent.release_volume + eroded_new,
                });
            }
            if eroded_new > ent.budget_total + tol {
                return Err(FlowError::MassBudgetViolated {
                    time: self.time,
                    flow_volume: v_flow,
                    budget: ent.budget_total,
                });
            }
            ent.eroded_volume = eroded_new;
            ent.e
                .par_iter_mut()
                .zip_eq(self.alpha.par_iter())
                .for_each(|(e, &d)| {
                    if d > 0.0 {
                        *e += d;
                    }
                });
            self.grid.erode(&self.alpha);
        }

        std::mem::swap(&mut self.state, &mut self.scratch);
        self.has_stepped = true;
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

/// Evaluate the flux across the face between cell slots `l` and `r`
/// (canonical orientation: `l` is the lower-coordinate side — west for
/// x-faces, south for y-faces; row 0 is north, so the south neighbour is the
/// left side of a y-face). `None` in a slot means off-grid; a solid slot
/// acts as a wall.
///
/// Returns `None` for inactive faces: both sides below the dry threshold
/// (spec §3.5 — dry cells exchange nothing unless a wet neighbour floods
/// them) or no real side at all. Domain edges use a transmissive mirror
/// ghost (free outflow), `NoData` neighbours a reflective ghost with negated
/// normal velocity (spec §2.3).
fn face_between(
    grid: &SimGrid,
    state: &FlowState,
    config: &SolverConfig,
    mu: f64,
    l: Option<usize>,
    r: Option<usize>,
    is_x: bool,
) -> Option<EvaluatedFace> {
    let h_dry = f64::from(config.h_dry);
    let l_real = l.filter(|&i| !grid.solid_at(i));
    let r_real = r.filter(|&i| !grid.solid_at(i));

    // Cheap dry-dry early-out before building any Side: ghosts and walls
    // mirror the real side's depth, so the face is inactive iff every real
    // side is below the dry threshold. This is what keeps the face sweep
    // O(wet) — dry regions cost two loads per face.
    let l_wet = l_real.is_some_and(|i| f64::from(state.h[i]) >= h_dry);
    let r_wet = r_real.is_some_and(|i| f64::from(state.h[i]) >= h_dry);
    if !l_wet && !r_wet {
        return None;
    }

    let (ls, rs) = match (l_real, r_real) {
        (Some(li), Some(ri)) => (
            cell_side(grid, state, li, is_x, h_dry),
            cell_side(grid, state, ri, is_x, h_dry),
        ),
        (Some(li), None) => {
            let s = cell_side(grid, state, li, is_x, h_dry);
            // Wall if the slot exists but is solid; transmissive ghost if
            // off-grid.
            let ghost = if r.is_some() {
                Side { un: -s.un, ..s }
            } else {
                s
            };
            (s, ghost)
        }
        (None, Some(ri)) => {
            let s = cell_side(grid, state, ri, is_x, h_dry);
            let ghost = if l.is_some() {
                Side { un: -s.un, ..s }
            } else {
                s
            };
            (ghost, s)
        }
        (None, None) => return None,
    };

    if ls.h < h_dry && rs.h < h_dry {
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
    let static_below_yield = ls.un == 0.0 && ls.ut == 0.0 && rs.un == 0.0 && rs.ut == 0.0 && {
        let d_eta = ((rs.h + rs.z) - (ls.h + ls.z)).abs();
        let cos_l = l_real.or(r_real).map_or(1.0, |i| grid.cos_theta_at(i));
        let cos_r = r_real.or(l_real).map_or(1.0, |i| grid.cos_theta_at(i));
        d_eta <= mu * 0.5 * (cos_l + cos_r) * grid.cellsize()
    };

    let mut flux = flux::face_flux(ls, rs);
    if static_below_yield {
        // Suppress only the diffusive mass exchange (and the mass-carried
        // transverse momentum); identical on both sides of the face, so mass
        // conservation is preserved exactly.
        flux.mass = 0.0;
        flux.mom_t = 0.0;
    }
    Some(EvaluatedFace {
        flux,
        l_donor: l_real,
        r_donor: r_real,
    })
}

/// Pass 1: per-cell positivity limiter α = min(1, h·Δx / (Δt·Σ outgoing mass
/// flux)) — scaling every outgoing flux by the donor's α guarantees h ≥ 0
/// (spec §3.5, positivity preserving).
///
/// Face sweep per band row: the row's cols+1 x-faces, then the y-faces on
/// its north edge; the band's south boundary face-row is evaluated once more
/// to credit the last row. Every cell's outflow accumulates in the fixed
/// order W, E, N, S regardless of banding or thread count (T7).
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
    let rows = grid.rows();
    alpha
        .par_chunks_mut(cols * BAND_ROWS)
        .enumerate()
        .for_each(|(band, alpha_band)| {
            let r0 = band * BAND_ROWS;
            let band_rows = alpha_band.len() / cols;
            let mut outflow = vec![0.0f64; alpha_band.len()];
            for lr in 0..band_rows {
                let row = (r0 + lr) * cols;
                let base = lr * cols;
                // x-faces: face f sits between cell f-1 (west) and cell f.
                for f in 0..=cols {
                    let l = (f > 0).then(|| row + f - 1);
                    let rr = (f < cols).then(|| row + f);
                    let Some(ef) = face_between(grid, state, config, mu, l, rr, true) else {
                        continue;
                    };
                    let m = ef.flux.mass;
                    if m > 0.0
                        && let Some(d) = ef.l_donor
                    {
                        outflow[base + (d - row)] += m;
                    } else if m < 0.0
                        && let Some(d) = ef.r_donor
                    {
                        outflow[base + (d - row)] += -m;
                    }
                }
                // y-faces on the row's north edge: L = this row (south side),
                // R = the row to the north. The northern row's southward
                // outflow is credited only when it belongs to this band.
                for c in 0..cols {
                    let l = Some(row + c);
                    let rr = (r0 + lr > 0).then(|| row - cols + c);
                    let Some(ef) = face_between(grid, state, config, mu, l, rr, false) else {
                        continue;
                    };
                    let m = ef.flux.mass;
                    if m > 0.0 {
                        if ef.l_donor.is_some() {
                            outflow[base + c] += m;
                        }
                    } else if m < 0.0 && lr > 0 && ef.r_donor.is_some() {
                        outflow[base - cols + c] += -m;
                    }
                }
            }
            // South boundary of the band: credit the last row's southward
            // outflow (face against row r_last+1, or the domain's south edge).
            let r_last = r0 + band_rows - 1;
            let row = r_last * cols;
            let base = (band_rows - 1) * cols;
            for c in 0..cols {
                let l = (r_last + 1 < rows).then(|| row + cols + c);
                let rr = Some(row + c);
                let Some(ef) = face_between(grid, state, config, mu, l, rr, false) else {
                    continue;
                };
                let m = ef.flux.mass;
                if m < 0.0 && ef.r_donor.is_some() {
                    outflow[base + c] += -m;
                }
            }
            for (k, a_out) in alpha_band.iter_mut().enumerate() {
                *a_out = 1.0;
                let i = r0 * cols + k;
                let h = f64::from(state.h[i]);
                if grid.solid_at(i) || h <= 0.0 {
                    continue;
                }
                if outflow[k] > 0.0 {
                    let a = (h * dx) / (dt * outflow[k]);
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
///
/// Same band face-sweep as [`compute_alpha`]: faces interior to the band are
/// evaluated once and applied to both adjacent cells; boundary face-rows are
/// recomputed by the neighbouring band. Contributions accumulate per cell in
/// the fixed order W, E, N, S into f64 band buffers, then land in f32 in one
/// store — bitwise independent of banding and thread count (T7).
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
    let rows = grid.rows();
    scratch
        .h
        .par_chunks_mut(cols * BAND_ROWS)
        .zip_eq(scratch.hu.par_chunks_mut(cols * BAND_ROWS))
        .zip_eq(scratch.hv.par_chunks_mut(cols * BAND_ROWS))
        .enumerate()
        .for_each(|(band, ((h_band, hu_band), hv_band))| {
            let r0 = band * BAND_ROWS;
            let band_rows = h_band.len() / cols;
            let n = h_band.len();
            let mut dh = vec![0.0f64; n];
            let mut dhu = vec![0.0f64; n];
            let mut dhv = vec![0.0f64; n];

            let donor_alpha = |ef: &EvaluatedFace| -> f64 {
                let d = if ef.flux.mass >= 0.0 {
                    ef.l_donor
                } else {
                    ef.r_donor
                };
                d.map_or(1.0, |i| f64::from(alpha[i]))
            };

            for lr in 0..band_rows {
                let row = (r0 + lr) * cols;
                let base = lr * cols;
                for f in 0..=cols {
                    let l = (f > 0).then(|| row + f - 1);
                    let rr = (f < cols).then(|| row + f);
                    let Some(ef) = face_between(grid, state, config, mu, l, rr, true) else {
                        continue;
                    };
                    let s = lambda * donor_alpha(&ef);
                    if let Some(d) = ef.l_donor {
                        let k = base + (d - row);
                        dh[k] -= s * ef.flux.mass;
                        dhu[k] -= s * ef.flux.mom_l;
                        dhv[k] -= s * ef.flux.mom_t;
                    }
                    if let Some(d) = ef.r_donor {
                        let k = base + (d - row);
                        dh[k] += s * ef.flux.mass;
                        dhu[k] += s * ef.flux.mom_r;
                        dhv[k] += s * ef.flux.mom_t;
                    }
                }
                for c in 0..cols {
                    let l = Some(row + c);
                    let rr = (r0 + lr > 0).then(|| row - cols + c);
                    let Some(ef) = face_between(grid, state, config, mu, l, rr, false) else {
                        continue;
                    };
                    let s = lambda * donor_alpha(&ef);
                    // This row is the south (L) side: +y is outward, so the
                    // flux leaves; normal momentum is hv, transverse is hu.
                    if ef.l_donor.is_some() {
                        let k = base + c;
                        dh[k] -= s * ef.flux.mass;
                        dhv[k] -= s * ef.flux.mom_l;
                        dhu[k] -= s * ef.flux.mom_t;
                    }
                    if lr > 0 && ef.r_donor.is_some() {
                        let k = base - cols + c;
                        dh[k] += s * ef.flux.mass;
                        dhv[k] += s * ef.flux.mom_r;
                        dhu[k] += s * ef.flux.mom_t;
                    }
                }
            }
            // South boundary of the band: apply only the r-side (last row)
            // contribution of its south face.
            let r_last = r0 + band_rows - 1;
            let row = r_last * cols;
            let base = (band_rows - 1) * cols;
            for c in 0..cols {
                let l = (r_last + 1 < rows).then(|| row + cols + c);
                let rr = Some(row + c);
                let Some(ef) = face_between(grid, state, config, mu, l, rr, false) else {
                    continue;
                };
                let s = lambda * donor_alpha(&ef);
                if ef.r_donor.is_some() {
                    let k = base + c;
                    dh[k] += s * ef.flux.mass;
                    dhv[k] += s * ef.flux.mom_r;
                    dhu[k] += s * ef.flux.mom_t;
                }
            }

            for k in 0..n {
                let i = r0 * cols + k;
                h_band[k] = (f64::from(state.h[i]) + dh[k]) as f32;
                hu_band[k] = (f64::from(state.hu[i]) + dhu[k]) as f32;
                hv_band[k] = (f64::from(state.hv[i]) + dhv[k]) as f32;
            }
        });
}
