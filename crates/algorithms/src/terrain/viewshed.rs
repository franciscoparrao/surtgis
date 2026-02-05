//! Viewshed analysis
//!
//! Determines which cells are visible from one or more observer points.
//!
//! Two algorithms are provided:
//! - **Bresenham ray tracing** ([`viewshed`]): parallelizable, traces rays to
//!   perimeter cells. Good on multi-core systems for small-to-medium DEMs.
//! - **XDraw** ([`viewshed_xdraw`]): processes each cell exactly once using
//!   interpolated reference planes. 2–5× faster on large DEMs with single-threaded
//!   execution.
//!
//! Reference:
//! Franklin, W.R. & Ray, C. (1994). Higher isn't necessarily better:
//! visibility algorithms and experiments. GIS/LIS.
//! Wang, J. et al. (2000). Approximating viewsheds on gridded DEMs.
//! Cauchi-Saunders, A. (2015). Comprehensive analysis of viewshed algorithms.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};
use std::f64::consts::PI;

/// Parameters for viewshed analysis
#[derive(Debug, Clone)]
pub struct ViewshedParams {
    /// Observer row position
    pub observer_row: usize,
    /// Observer column position
    pub observer_col: usize,
    /// Observer height above ground (meters, default 1.7)
    pub observer_height: f64,
    /// Target height above ground (meters, default 0.0)
    pub target_height: f64,
    /// Maximum radius in cells (0 = unlimited)
    pub max_radius: usize,
}

impl Default for ViewshedParams {
    fn default() -> Self {
        Self {
            observer_row: 0,
            observer_col: 0,
            observer_height: 1.7,
            target_height: 0.0,
            max_radius: 0,
        }
    }
}

/// Compute viewshed from a single observer point
///
/// Uses Bresenham-like line-of-sight algorithm along rays from the
/// observer to each cell on the perimeter, marking visible cells.
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Observer position and height parameters
///
/// # Returns
/// Raster<u8> where 1 = visible, 0 = not visible
pub fn viewshed(dem: &Raster<f64>, params: ViewshedParams) -> Result<Raster<u8>> {
    let (rows, cols) = dem.shape();

    if params.observer_row >= rows || params.observer_col >= cols {
        return Err(Error::IndexOutOfBounds {
            row: params.observer_row,
            col: params.observer_col,
            rows,
            cols,
        });
    }

    let cell_size = dem.cell_size();
    let obs_z = unsafe { dem.get_unchecked(params.observer_row, params.observer_col) }
        + params.observer_height;

    if obs_z.is_nan() {
        return Err(Error::Algorithm("Observer is on NaN cell".into()));
    }

    let max_r = if params.max_radius > 0 {
        params.max_radius
    } else {
        rows.max(cols)
    };

    // Generate target points on the perimeter of the search area
    let obs_r = params.observer_row as isize;
    let obs_c = params.observer_col as isize;

    // Collect perimeter targets
    let mut targets: Vec<(isize, isize)> = Vec::new();
    let r = max_r as isize;

    // Top and bottom rows
    for c in (obs_c - r)..=(obs_c + r) {
        targets.push((obs_r - r, c));
        targets.push((obs_r + r, c));
    }
    // Left and right columns (excluding corners)
    for row in (obs_r - r + 1)..=(obs_r + r - 1) {
        targets.push((row, obs_c - r));
        targets.push((row, obs_c + r));
    }

    // Process rays in parallel - each ray traces from observer to perimeter target
    let visibility_maps: Vec<Vec<(usize, usize)>> = targets
        .into_par_iter()
        .map(|(tr, tc)| {
            trace_ray(dem, obs_r, obs_c, obs_z, tr, tc,
                      params.target_height, cell_size, rows, cols)
        })
        .collect();

    // Merge visibility results
    let mut output_data = Array2::<u8>::zeros((rows, cols));
    output_data[(params.observer_row, params.observer_col)] = 1;

    for visible_cells in &visibility_maps {
        for &(r, c) in visible_cells {
            output_data[(r, c)] = 1;
        }
    }

    let mut output = dem.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = output_data;

    Ok(output)
}

/// Trace a ray from observer to target, returning visible cells
#[allow(clippy::too_many_arguments)]
fn trace_ray(
    dem: &Raster<f64>,
    obs_r: isize, obs_c: isize, obs_z: f64,
    target_r: isize, target_c: isize,
    target_height: f64,
    cell_size: f64,
    rows: usize, cols: usize,
) -> Vec<(usize, usize)> {
    let mut visible = Vec::new();
    let mut max_angle = f64::NEG_INFINITY;

    // Bresenham-style stepping
    let dr = target_r - obs_r;
    let dc = target_c - obs_c;
    let steps = dr.unsigned_abs().max(dc.unsigned_abs());

    if steps == 0 {
        return visible;
    }

    let step_r = dr as f64 / steps as f64;
    let step_c = dc as f64 / steps as f64;

    for s in 1..=steps {
        let cr = (obs_r as f64 + step_r * s as f64).round() as isize;
        let cc = (obs_c as f64 + step_c * s as f64).round() as isize;

        if cr < 0 || cc < 0 || (cr as usize) >= rows || (cc as usize) >= cols {
            break;
        }

        let r = cr as usize;
        let c = cc as usize;

        let z = unsafe { dem.get_unchecked(r, c) };
        if z.is_nan() {
            break;
        }

        let drow = (r as f64 - obs_r as f64) * cell_size;
        let dcol = (c as f64 - obs_c as f64) * cell_size;
        let dist = (drow * drow + dcol * dcol).sqrt();

        if dist < f64::EPSILON {
            continue;
        }

        let target_z = z + target_height;
        let angle = (target_z - obs_z) / dist;

        if angle >= max_angle {
            visible.push((r, c));
            max_angle = angle;
        }
    }

    visible
}

/// Compute viewshed using the XDraw algorithm.
///
/// XDraw processes cells in order of increasing distance from the observer.
/// For each cell, visibility is determined by interpolating a reference plane
/// from two previously-processed "parent" cells that bracket the line of sight.
///
/// **Advantages over ray tracing**:
/// - Each cell is visited exactly once (O(n log n) total vs O(n√n) for rays)
/// - Better cache locality
/// - More consistent results (no sampling artifacts from discrete rays)
///
/// **Trade-offs**:
/// - Not parallelizable (cells depend on parents processed earlier)
/// - Approximate (interpolation can miss narrow features)
///
/// # Arguments
/// * `dem` — Input DEM
/// * `params` — Observer position and height parameters (same as [`viewshed`])
///
/// # Returns
/// Raster<u8> where 1 = visible, 0 = not visible
pub fn viewshed_xdraw(dem: &Raster<f64>, params: ViewshedParams) -> Result<Raster<u8>> {
    let (rows, cols) = dem.shape();

    if params.observer_row >= rows || params.observer_col >= cols {
        return Err(Error::IndexOutOfBounds {
            row: params.observer_row,
            col: params.observer_col,
            rows,
            cols,
        });
    }

    let cell_size = dem.cell_size();
    let obs_r = params.observer_row as isize;
    let obs_c = params.observer_col as isize;
    let obs_z = unsafe { dem.get_unchecked(params.observer_row, params.observer_col) }
        + params.observer_height;

    if obs_z.is_nan() {
        return Err(Error::Algorithm("Observer is on NaN cell".into()));
    }

    let max_r = if params.max_radius > 0 {
        params.max_radius
    } else {
        rows.max(cols)
    };

    let mut visible = Array2::<u8>::zeros((rows, cols));
    // Reference plane: stores the max slope angle seen along the LOS to each cell
    let mut reference = Array2::from_elem((rows, cols), f64::NEG_INFINITY);

    visible[(params.observer_row, params.observer_col)] = 1;

    // Collect all candidate cells with their squared distance
    let mut cells: Vec<(usize, usize, u64)> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let dr = r as isize - obs_r;
            let dc = c as isize - obs_c;
            let dist_sq = (dr * dr + dc * dc) as u64;
            if dist_sq == 0 {
                continue;
            }
            let dist = (dist_sq as f64).sqrt();
            if dist <= max_r as f64 {
                cells.push((r, c, dist_sq));
            }
        }
    }

    // Sort by distance (integer key avoids float comparison issues)
    cells.sort_unstable_by_key(|&(_, _, d)| d);

    // Process cells in distance order
    for &(r, c, _) in &cells {
        let z = unsafe { dem.get_unchecked(r, c) };
        if z.is_nan() {
            continue;
        }

        let dr = r as isize - obs_r;
        let dc = c as isize - obs_c;
        let dist = ((dr * dr + dc * dc) as f64).sqrt() * cell_size;

        if dist < f64::EPSILON {
            continue;
        }

        let slope_angle = (z + params.target_height - obs_z) / dist;

        // Interpolate reference from the two parent cells
        let ref_interp = xdraw_interpolate_ref(
            &reference, obs_r, obs_c, dr, dc, rows, cols,
        );

        if slope_angle >= ref_interp {
            visible[(r, c)] = 1;
        }
        reference[(r, c)] = slope_angle.max(ref_interp);
    }

    let mut output = dem.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = visible;

    Ok(output)
}

/// Interpolate reference plane value from two parent cells.
///
/// For a cell at offset (dr, dc) from observer, finds the two cells
/// one step closer along the primary axis that bracket the line of sight,
/// and linearly interpolates the reference value.
fn xdraw_interpolate_ref(
    reference: &Array2<f64>,
    obs_r: isize, obs_c: isize,
    dr: isize, dc: isize,
    rows: usize, cols: usize,
) -> f64 {
    let abs_dr = dr.unsigned_abs();
    let abs_dc = dc.unsigned_abs();
    let sign_r = dr.signum();
    let sign_c = dc.signum();

    let r = (obs_r + dr) as usize;
    let c = (obs_c + dc) as usize;

    if abs_dr >= abs_dc {
        // Primary axis is row direction
        // Cell A: one step back along primary (same secondary)
        let ar = r as isize - sign_r;
        let ac = c as isize;
        // Cell B: one step back along primary AND one step back along secondary
        let br = r as isize - sign_r;
        let bc = if sign_c != 0 { c as isize - sign_c } else { ac };

        let f = if abs_dr > 0 { abs_dc as f64 / abs_dr as f64 } else { 0.0 };

        let ref_a = safe_ref(reference, ar, ac, rows, cols);
        let ref_b = if sign_c != 0 {
            safe_ref(reference, br, bc, rows, cols)
        } else {
            ref_a
        };

        // Avoid NaN from infinity * 0.0 in IEEE 754
        if f <= 0.0 { ref_a } else if f >= 1.0 { ref_b } else { ref_a * (1.0 - f) + ref_b * f }
    } else {
        // Primary axis is column direction
        // Cell A: one step back along primary (same secondary)
        let ar = r as isize;
        let ac = c as isize - sign_c;
        // Cell B: one step back along primary AND one step back along secondary
        let br = if sign_r != 0 { r as isize - sign_r } else { ar };
        let bc = c as isize - sign_c;

        let f = if abs_dc > 0 { abs_dr as f64 / abs_dc as f64 } else { 0.0 };

        let ref_a = safe_ref(reference, ar, ac, rows, cols);
        let ref_b = if sign_r != 0 {
            safe_ref(reference, br, bc, rows, cols)
        } else {
            ref_a
        };

        // Avoid NaN from infinity * 0.0 in IEEE 754
        if f <= 0.0 { ref_a } else if f >= 1.0 { ref_b } else { ref_a * (1.0 - f) + ref_b * f }
    }
}

/// Safely read reference value, returning NEG_INFINITY for out-of-bounds cells.
#[inline]
fn safe_ref(reference: &Array2<f64>, r: isize, c: isize, rows: usize, cols: usize) -> f64 {
    if r >= 0 && (r as usize) < rows && c >= 0 && (c as usize) < cols {
        reference[(r as usize, c as usize)]
    } else {
        f64::NEG_INFINITY
    }
}

/// Compute viewshed from multiple observer points
///
/// Returns cumulative visibility: each cell's value indicates how many
/// observers can see it.
pub fn viewshed_multiple(
    dem: &Raster<f64>,
    observers: &[(usize, usize)],
    observer_height: f64,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let mut cumulative = Raster::filled(rows, cols, 0.0_f64);
    cumulative.set_transform(*dem.transform());

    for &(obs_row, obs_col) in observers {
        let params = ViewshedParams {
            observer_row: obs_row,
            observer_col: obs_col,
            observer_height,
            target_height: 0.0,
            max_radius: 0,
        };
        let vs = viewshed(dem, params)?;
        for row in 0..rows {
            for col in 0..cols {
                let v = unsafe { vs.get_unchecked(row, col) };
                if v > 0 {
                    let current = unsafe { cumulative.get_unchecked(row, col) };
                    cumulative.set(row, col, current + 1.0).unwrap();
                }
            }
        }
    }

    Ok(cumulative)
}

/// Parameters for probabilistic viewshed
#[derive(Debug, Clone)]
pub struct ProbabilisticViewshedParams {
    /// Observer row position
    pub observer_row: usize,
    /// Observer column position
    pub observer_col: usize,
    /// Observer height above ground (meters, default 1.7)
    pub observer_height: f64,
    /// Target height above ground (meters, default 0.0)
    pub target_height: f64,
    /// Maximum radius in cells (0 = unlimited)
    pub max_radius: usize,
    /// DEM RMSE (vertical uncertainty in meters)
    pub dem_rmse: f64,
    /// Number of Monte Carlo realizations (default 100)
    pub n_realizations: usize,
    /// Random seed for reproducibility (default: 42)
    pub seed: u64,
}

impl Default for ProbabilisticViewshedParams {
    fn default() -> Self {
        Self {
            observer_row: 0,
            observer_col: 0,
            observer_height: 1.7,
            target_height: 0.0,
            max_radius: 0,
            dem_rmse: 1.0,
            n_realizations: 100,
            seed: 42,
        }
    }
}

/// Simple linear congruential generator for deterministic pseudo-random numbers.
/// Returns values in [0, 1).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        // Multiplier and increment from Knuth (MMIX)
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a standard normal random variable using Box-Muller transform
    fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u1 = u1.max(1e-15); // Avoid log(0)
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// Compute probabilistic viewshed using Monte Carlo simulation.
///
/// Following Fisher (1993), propagates DEM elevation uncertainty through
/// viewshed computation by running N realizations with perturbed elevations.
/// The output is a probability map (0.0 to 1.0) indicating the likelihood
/// of each cell being visible.
///
/// # Algorithm
///
/// 1. For each realization i = 1..N:
///    a. Generate perturbed DEM: z'(r,c) = z(r,c) + ε where ε ~ N(0, rmse²)
///    b. Compute viewshed on perturbed DEM
///    c. Accumulate visibility count
/// 2. Probability = count / N
///
/// # Arguments
/// * `dem` — Input DEM raster
/// * `params` — Probabilistic viewshed parameters
///
/// # Returns
/// Raster<f64> with visibility probabilities in [0.0, 1.0]
pub fn viewshed_probabilistic(
    dem: &Raster<f64>,
    params: ProbabilisticViewshedParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();

    if params.observer_row >= rows || params.observer_col >= cols {
        return Err(Error::IndexOutOfBounds {
            row: params.observer_row,
            col: params.observer_col,
            rows,
            cols,
        });
    }

    if params.n_realizations == 0 {
        return Err(Error::Algorithm("n_realizations must be > 0".into()));
    }

    let n = params.n_realizations;
    let rmse = params.dem_rmse;

    // Accumulate visibility counts across realizations
    let mut count_data = Array2::<f64>::zeros((rows, cols));

    let mut rng = Lcg::new(params.seed);

    for _ in 0..n {
        // Generate perturbed DEM
        let mut perturbed = Raster::new(rows, cols);
        perturbed.set_transform(*dem.transform());
        {
            let orig = dem.data();
            let pert = perturbed.data_mut();
            for r in 0..rows {
                for c in 0..cols {
                    let z = orig[[r, c]];
                    if z.is_nan() {
                        pert[[r, c]] = f64::NAN;
                    } else {
                        pert[[r, c]] = z + rng.next_normal() * rmse;
                    }
                }
            }
        }

        // Run deterministic viewshed on perturbed DEM
        let vs_params = ViewshedParams {
            observer_row: params.observer_row,
            observer_col: params.observer_col,
            observer_height: params.observer_height,
            target_height: params.target_height,
            max_radius: params.max_radius,
        };

        let vs = viewshed(&perturbed, vs_params)?;

        // Accumulate
        for r in 0..rows {
            for c in 0..cols {
                if unsafe { vs.get_unchecked(r, c) } > 0 {
                    count_data[[r, c]] += 1.0;
                }
            }
        }
    }

    // Convert counts to probabilities
    let inv_n = 1.0 / n as f64;
    for r in 0..rows {
        for c in 0..cols {
            count_data[[r, c]] *= inv_n;
        }
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = count_data;

    Ok(output)
}

/// Parameters for observer optimization
#[derive(Debug, Clone)]
pub struct ObserverOptimizationParams {
    /// Observer height above ground (meters, default 1.7)
    pub observer_height: f64,
    /// Target height above ground (meters, default 0.0)
    pub target_height: f64,
    /// Maximum viewshed radius in cells (0 = unlimited)
    pub max_radius: usize,
    /// Maximum number of observers to select (default 5)
    pub max_observers: usize,
}

impl Default for ObserverOptimizationParams {
    fn default() -> Self {
        Self {
            observer_height: 1.7,
            target_height: 0.0,
            max_radius: 0,
            max_observers: 5,
        }
    }
}

/// Result of observer optimization
#[derive(Debug)]
pub struct ObserverOptimizationResult {
    /// Selected observer positions (row, col) in order of selection
    pub observers: Vec<(usize, usize)>,
    /// Cumulative coverage fraction [0,1] after adding each observer
    pub coverage: Vec<f64>,
    /// Final combined viewshed (count of how many observers see each cell)
    pub combined_viewshed: Raster<f64>,
}

/// Select optimal observer locations for maximum visibility coverage.
///
/// Uses a greedy MCLP (Maximum Coverage Location Problem) algorithm:
/// 1. Compute viewshed for each candidate location
/// 2. Select the candidate with maximum uncovered visibility
/// 3. Update covered set
/// 4. Repeat until max_observers reached or 100% coverage
///
/// # Arguments
/// * `dem` — Input DEM raster
/// * `candidates` — List of candidate observer positions (row, col)
/// * `params` — Optimization parameters
///
/// # Returns
/// `ObserverOptimizationResult` with selected observers and coverage
pub fn observer_optimization(
    dem: &Raster<f64>,
    candidates: &[(usize, usize)],
    params: ObserverOptimizationParams,
) -> Result<ObserverOptimizationResult> {
    let (rows, cols) = dem.shape();

    if candidates.is_empty() {
        return Err(Error::Algorithm("No candidate locations provided".into()));
    }

    // Count total valid cells for coverage computation
    let mut total_valid = 0_usize;
    for r in 0..rows {
        for c in 0..cols {
            let z = unsafe { dem.get_unchecked(r, c) };
            if !z.is_nan() {
                total_valid += 1;
            }
        }
    }

    if total_valid == 0 {
        return Err(Error::Algorithm("DEM has no valid cells".into()));
    }

    // Precompute viewsheds for all candidates
    let viewsheds: Vec<Option<Raster<u8>>> = candidates.iter()
        .map(|&(obs_r, obs_c)| {
            if obs_r >= rows || obs_c >= cols {
                return None;
            }
            let vs_params = ViewshedParams {
                observer_row: obs_r,
                observer_col: obs_c,
                observer_height: params.observer_height,
                target_height: params.target_height,
                max_radius: params.max_radius,
            };
            viewshed(dem, vs_params).ok()
        })
        .collect();

    // Greedy selection
    let mut covered = Array2::<bool>::from_elem((rows, cols), false);
    let mut selected: Vec<usize> = Vec::new();
    let mut coverage_history: Vec<f64> = Vec::new();
    let mut combined_data = Array2::<f64>::zeros((rows, cols));

    let max_sel = params.max_observers.min(candidates.len());

    for _ in 0..max_sel {
        // Find candidate with maximum marginal coverage gain
        let mut best_idx = None;
        let mut best_gain = 0_usize;

        for (ci, vs_opt) in viewsheds.iter().enumerate() {
            if selected.contains(&ci) {
                continue;
            }
            let vs = match vs_opt {
                Some(v) => v,
                None => continue,
            };

            let mut gain = 0_usize;
            for r in 0..rows {
                for c in 0..cols {
                    if !covered[[r, c]] && unsafe { vs.get_unchecked(r, c) } > 0 {
                        gain += 1;
                    }
                }
            }

            if gain > best_gain {
                best_gain = gain;
                best_idx = Some(ci);
            }
        }

        let ci = match best_idx {
            Some(i) if best_gain > 0 => i,
            _ => break, // No more gain possible
        };

        // Select this candidate
        selected.push(ci);

        // Update covered set
        if let Some(vs) = &viewsheds[ci] {
            for r in 0..rows {
                for c in 0..cols {
                    if unsafe { vs.get_unchecked(r, c) } > 0 {
                        covered[[r, c]] = true;
                        combined_data[[r, c]] += 1.0;
                    }
                }
            }
        }

        // Compute coverage fraction
        let covered_count = covered.iter().filter(|&&v| v).count();
        coverage_history.push(covered_count as f64 / total_valid as f64);
    }

    let mut combined = dem.with_same_meta::<f64>(rows, cols);
    combined.set_nodata(Some(f64::NAN));
    *combined.data_mut() = combined_data;

    let observers: Vec<(usize, usize)> = selected.iter().map(|&i| candidates[i]).collect();

    Ok(ObserverOptimizationResult {
        observers,
        coverage: coverage_history,
        combined_viewshed: combined,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_viewshed_flat() {
        // Flat terrain: everything should be visible
        let mut dem = Raster::filled(20, 20, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = viewshed(&dem, ViewshedParams {
            observer_row: 10,
            observer_col: 10,
            observer_height: 1.7,
            ..Default::default()
        }).unwrap();

        // Center and nearby cells should be visible
        assert_eq!(result.get(10, 10).unwrap(), 1);
        assert_eq!(result.get(10, 15).unwrap(), 1);
        assert_eq!(result.get(5, 10).unwrap(), 1);
    }

    #[test]
    fn test_viewshed_blocked() {
        // Wall between observer and target
        let mut dem = Raster::filled(20, 20, 0.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        // Create a wall at column 10
        for row in 0..20 {
            dem.set(row, 10, 1000.0).unwrap();
        }

        let result = viewshed(&dem, ViewshedParams {
            observer_row: 10,
            observer_col: 5,
            observer_height: 1.7,
            ..Default::default()
        }).unwrap();

        // Cells behind the wall should not be visible
        assert_eq!(result.get(10, 15).unwrap(), 0, "Cell behind wall should be hidden");
        // Cells before the wall should be visible
        assert_eq!(result.get(10, 8).unwrap(), 1, "Cell before wall should be visible");
    }

    #[test]
    fn test_viewshed_observer_visible() {
        let mut dem = Raster::filled(10, 10, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = viewshed(&dem, ViewshedParams {
            observer_row: 5,
            observer_col: 5,
            observer_height: 0.0,
            ..Default::default()
        }).unwrap();
        assert_eq!(result.get(5, 5).unwrap(), 1, "Observer should always be visible");
    }

    #[test]
    fn test_viewshed_invalid_observer() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        let result = viewshed(&dem, ViewshedParams {
            observer_row: 20,
            observer_col: 5,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // XDraw tests

    #[test]
    fn test_xdraw_flat_all_visible() {
        let mut dem = Raster::filled(20, 20, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = viewshed_xdraw(&dem, ViewshedParams {
            observer_row: 10,
            observer_col: 10,
            observer_height: 1.7,
            ..Default::default()
        }).unwrap();

        assert_eq!(result.get(10, 10).unwrap(), 1, "Observer should be visible");
        assert_eq!(result.get(10, 15).unwrap(), 1, "Flat terrain should be visible");
        assert_eq!(result.get(5, 10).unwrap(), 1, "Flat terrain should be visible");
    }

    #[test]
    fn test_xdraw_wall_blocks() {
        let mut dem = Raster::filled(20, 20, 0.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        // Wall at column 10
        for row in 0..20 {
            dem.set(row, 10, 1000.0).unwrap();
        }

        let result = viewshed_xdraw(&dem, ViewshedParams {
            observer_row: 10,
            observer_col: 5,
            observer_height: 1.7,
            ..Default::default()
        }).unwrap();

        assert_eq!(result.get(10, 15).unwrap(), 0, "XDraw: cell behind wall should be hidden");
        assert_eq!(result.get(10, 8).unwrap(), 1, "XDraw: cell before wall should be visible");
    }

    #[test]
    fn test_xdraw_observer_always_visible() {
        let mut dem = Raster::filled(10, 10, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = viewshed_xdraw(&dem, ViewshedParams {
            observer_row: 5,
            observer_col: 5,
            observer_height: 0.0,
            ..Default::default()
        }).unwrap();

        assert_eq!(result.get(5, 5).unwrap(), 1, "Observer always visible");
    }

    #[test]
    fn test_xdraw_all_adjacent_visible_on_flat() {
        // On flat terrain with zero observer height, all cells should be visible
        let mut dem = Raster::filled(15, 15, 50.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));

        let result = viewshed_xdraw(&dem, ViewshedParams {
            observer_row: 7,
            observer_col: 7,
            observer_height: 0.0,
            target_height: 0.0,
            max_radius: 6,
        }).unwrap();

        // All cells within max_radius should be visible on perfectly flat terrain
        let mut missed: Vec<(usize, usize, f64)> = Vec::new();
        for r in 0..15 {
            for c in 0..15 {
                let dr = r as f64 - 7.0;
                let dc = c as f64 - 7.0;
                let dist = (dr * dr + dc * dc).sqrt();
                if dist <= 6.0 && dist > 0.0 {
                    if result.get(r, c).unwrap() != 1 {
                        missed.push((r, c, dist));
                    }
                }
            }
        }

        assert!(
            missed.is_empty(),
            "Flat terrain: cells not visible (should all be visible): {:?}",
            missed
        );
    }

    #[test]
    fn test_xdraw_max_radius_limits() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let result = viewshed_xdraw(&dem, ViewshedParams {
            observer_row: 10,
            observer_col: 10,
            observer_height: 0.0,
            target_height: 0.0,
            max_radius: 3,
        }).unwrap();

        // Cell at distance 3 should be visible
        assert_eq!(result.get(7, 10).unwrap(), 1, "Cell at distance 3 should be visible");
        // Cell at distance > 3 should not be (outside search radius)
        assert_eq!(result.get(6, 10).unwrap(), 0, "Cell beyond max_radius should be 0");
    }

    #[test]
    fn test_xdraw_invalid_observer() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(viewshed_xdraw(&dem, ViewshedParams {
            observer_row: 20,
            observer_col: 5,
            ..Default::default()
        }).is_err());
    }

    #[test]
    fn test_probabilistic_flat_terrain() {
        // On flat terrain, probability should be ~1.0 everywhere visible
        let mut dem = Raster::filled(15, 15, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));

        let result = viewshed_probabilistic(&dem, ProbabilisticViewshedParams {
            observer_row: 7,
            observer_col: 7,
            observer_height: 10.0,
            dem_rmse: 0.1, // Small uncertainty relative to observer height
            n_realizations: 50,
            ..Default::default()
        }).unwrap();

        // Cells near observer should have probability ~1.0
        let p = result.get(7, 8).unwrap();
        assert!(
            p > 0.9,
            "Near observer on flat terrain should be ~1.0, got {}",
            p
        );
    }

    #[test]
    fn test_probabilistic_range() {
        let mut dem = Raster::filled(15, 15, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));

        let result = viewshed_probabilistic(&dem, ProbabilisticViewshedParams {
            observer_row: 7,
            observer_col: 7,
            dem_rmse: 1.0,
            n_realizations: 30,
            ..Default::default()
        }).unwrap();

        // All probabilities should be in [0, 1]
        for r in 0..15 {
            for c in 0..15 {
                let p = result.get(r, c).unwrap();
                assert!(
                    p >= 0.0 && p <= 1.0,
                    "Probability should be in [0,1], got {} at ({},{})",
                    p, r, c
                );
            }
        }
    }

    #[test]
    fn test_probabilistic_high_uncertainty_reduces_visibility() {
        // Wall with marginal visibility — high uncertainty should make it fuzzy
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        // Create a wall that just barely blocks line of sight
        for c in 0..21 {
            dem.set(10, c, 102.0).unwrap(); // 2m wall
        }

        // Low uncertainty: result should be binary-ish
        let low_u = viewshed_probabilistic(&dem, ProbabilisticViewshedParams {
            observer_row: 5,
            observer_col: 10,
            observer_height: 1.7,
            dem_rmse: 0.01, // Tiny uncertainty
            n_realizations: 50,
            seed: 42,
            ..Default::default()
        }).unwrap();

        // High uncertainty: result should be more varied
        let high_u = viewshed_probabilistic(&dem, ProbabilisticViewshedParams {
            observer_row: 5,
            observer_col: 10,
            observer_height: 1.7,
            dem_rmse: 3.0, // Large uncertainty
            n_realizations: 50,
            seed: 42,
            ..Default::default()
        }).unwrap();

        // Behind wall, high uncertainty should show higher probability
        // (some realizations will lower the wall)
        let p_low = low_u.get(15, 10).unwrap();
        let p_high = high_u.get(15, 10).unwrap();

        assert!(
            p_high >= p_low,
            "Higher uncertainty should increase visibility behind wall: low={:.2}, high={:.2}",
            p_low, p_high
        );
    }

    #[test]
    fn test_probabilistic_deterministic_seed() {
        let mut dem = Raster::filled(11, 11, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for c in 0..11 { dem.set(5, c, 102.0).unwrap(); }

        let params = ProbabilisticViewshedParams {
            observer_row: 2,
            observer_col: 5,
            dem_rmse: 2.0,
            n_realizations: 30,
            seed: 123,
            ..Default::default()
        };

        let r1 = viewshed_probabilistic(&dem, params.clone()).unwrap();
        let r2 = viewshed_probabilistic(&dem, params).unwrap();

        // Same seed → same results
        let p1 = r1.get(8, 5).unwrap();
        let p2 = r2.get(8, 5).unwrap();
        assert!(
            (p1 - p2).abs() < 1e-10,
            "Same seed should give identical results: {} vs {}",
            p1, p2
        );
    }

    #[test]
    fn test_observer_optimization_basic() {
        let mut dem = Raster::filled(15, 15, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));

        let candidates = vec![(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)];

        let result = observer_optimization(&dem, &candidates, ObserverOptimizationParams {
            max_observers: 3,
            ..Default::default()
        }).unwrap();

        assert!(!result.observers.is_empty());
        assert!(result.observers.len() <= 3);
        // Coverage should increase with each observer
        for i in 1..result.coverage.len() {
            assert!(
                result.coverage[i] >= result.coverage[i - 1],
                "Coverage should be non-decreasing"
            );
        }
    }

    #[test]
    fn test_observer_optimization_coverage_increases() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let candidates = vec![(5, 5), (5, 15), (15, 5), (15, 15)];

        let result = observer_optimization(&dem, &candidates, ObserverOptimizationParams {
            max_observers: 4,
            ..Default::default()
        }).unwrap();

        // With 4 corners on flat terrain, should get high coverage
        assert!(result.coverage.last().unwrap() > &0.5,
            "4 observers on flat terrain should cover >50%");
    }

    #[test]
    fn test_observer_optimization_empty_candidates() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(observer_optimization(&dem, &[], ObserverOptimizationParams::default()).is_err());
    }

    #[test]
    fn test_observer_optimization_picks_best() {
        // Create terrain where center observer has clear advantage
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        // Add tall walls near edges to limit corner visibility
        for i in 0..21 {
            dem.set(2, i, 200.0).unwrap();
            dem.set(18, i, 200.0).unwrap();
            dem.set(i, 2, 200.0).unwrap();
            dem.set(i, 18, 200.0).unwrap();
        }

        let candidates = vec![(0, 0), (10, 10)];

        let result = observer_optimization(&dem, &candidates, ObserverOptimizationParams {
            max_observers: 1,
            ..Default::default()
        }).unwrap();

        // Center observer should be selected (sees through the interior)
        assert_eq!(result.observers[0], (10, 10),
            "Center observer should be selected (less blocked)");
    }
}
