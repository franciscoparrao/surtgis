//! Per-basin concavity-index (θ) estimation via χ scatter minimisation.
//!
//! The "concavity index" θ in the slope-area scaling S = ks · A^(-θ) is
//! usually fixed at the reference 0.45 for inter-basin comparison
//! (`ksn`), but the actual θ that best linearises a given basin's
//! elevation profile against χ is a measurement in its own right.
//! Deviations from 0.45 can flag transient response or non-uniform
//! erodibility (see Mudd et al. 2018; Gallen & Wegmann 2017).
//!
//! ## Method (spec §4.4, after Perron & Royden 2013)
//!
//! For each basin defined by the input `basins` raster:
//!
//! 1. Collect the basin's stream cells and elevations.
//! 2. Identify the basin outlet — the basin stream cell whose
//!    downstream flow exits the basin (or is a graph terminal).
//! 3. For each candidate θ in `theta_range`:
//!    a. Compute χ for every basin stream cell via BFS upstream from
//!       the basin outlet (χ at the outlet = 0, accumulates upstream
//!       with the same Riemann sum used in [`super::chi`]).
//!    b. Fit elevation ~ χ with ordinary least squares.
//!    c. Record the residual RMSE.
//! 4. The θ that minimises RMSE is `theta_opt` for the basin.
//! 5. Bootstrap (default n=200): resample (cell-index, χ_at_θ, z) pairs
//!    with replacement, repeat the grid search, collect the bootstrapped
//!    θ_opts. The 2.5 / 97.5 percentiles give the 95 % CI.
//!
//! ## Determinism
//!
//! Bootstrap resampling uses a seeded `DefaultHasher` (the same pattern
//! as [`crate::imagery`] / `extract-patches`) so results are reproducible
//! without pulling the `rand` crate.

use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};

use surtgis_core::{Raster, Result};

use super::stream_traversal::{StreamGraph, StreamGraphError, build_stream_graph};

/// Parameters for [`concavity_index`].
#[derive(Debug, Clone)]
pub struct ConcavityParams {
    /// Inclusive search range for θ. Default `(0.1, 0.9)`.
    pub theta_range: (f64, f64),
    /// Step between candidate θ values. Default 0.05.
    pub theta_step: f64,
    /// Number of bootstrap iterations for the 95 % CI. Default 200.
    /// Set to 0 to disable bootstrap (CI returned as `(theta_opt, theta_opt)`).
    pub bootstrap_n: usize,
    /// Cell size in metres (caller-supplied — see the CRS heuristic in
    /// the CLI handler for the standard auto-detection rule).
    pub cell_size_m: f64,
    /// Reproducibility seed for the bootstrap resampler. Default 42.
    pub seed: u64,
    /// Minimum number of basin stream cells required to attempt an
    /// estimate. Below this, the basin is skipped. Default 30.
    pub min_basin_cells: usize,
}

impl Default for ConcavityParams {
    fn default() -> Self {
        Self {
            theta_range: (0.1, 0.9),
            theta_step: 0.05,
            bootstrap_n: 200,
            cell_size_m: 30.0,
            seed: 42,
            min_basin_cells: 30,
        }
    }
}

/// One row of the concavity-index output table.
#[derive(Debug, Clone)]
pub struct ConcavityResult {
    pub basin_id: i32,
    pub theta_opt: f64,
    /// `(low, high)` 95 % CI from the bootstrap. Equal to `(theta_opt,
    /// theta_opt)` when bootstrap is disabled.
    pub theta_ci: (f64, f64),
    pub n_cells: usize,
    /// Residual RMSE of the elevation~χ regression at `theta_opt`.
    pub rmse: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ConcavityError {
    #[error(transparent)]
    Graph(#[from] StreamGraphError),
    #[error("raster shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),
    #[error("ConcavityParams.cell_size_m must be > 0 (got {0})")]
    NonPositiveCellSize(f64),
    #[error("ConcavityParams.theta_range invalid: low={0} high={1}")]
    BadThetaRange(f64, f64),
    #[error("ConcavityParams.theta_step must be > 0 (got {0})")]
    NonPositiveStep(f64),
}

/// Estimate the optimal concavity index θ for every basin in `basins`.
///
/// Returns one [`ConcavityResult`] per basin that passes the
/// `min_basin_cells` check. Basins with too few stream cells, or with
/// degenerate elevation distributions (zero variance), are silently
/// skipped — `result.len()` may be smaller than the number of unique
/// `basin_id`s in the input.
#[allow(clippy::too_many_arguments)]
pub fn concavity_index(
    stream: &Raster<u8>,
    flow_dir: &Raster<u8>,
    flow_acc: &Raster<f64>,
    dem: &Raster<f64>,
    basins: &Raster<i32>,
    params: ConcavityParams,
) -> Result<Vec<ConcavityResult>> {
    if params.cell_size_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            ConcavityError::NonPositiveCellSize(params.cell_size_m).to_string(),
        ));
    }
    let (lo, hi) = params.theta_range;
    if !(lo > 0.0 && hi > lo) {
        return Err(surtgis_core::Error::Other(
            ConcavityError::BadThetaRange(lo, hi).to_string(),
        ));
    }
    if params.theta_step <= 0.0 {
        return Err(surtgis_core::Error::Other(
            ConcavityError::NonPositiveStep(params.theta_step).to_string(),
        ));
    }
    let s_shape = stream.shape();
    for other in [
        flow_dir.shape(),
        flow_acc.shape(),
        dem.shape(),
        basins.shape(),
    ] {
        if other != s_shape {
            return Err(surtgis_core::Error::Other(
                ConcavityError::ShapeMismatch(other, s_shape).to_string(),
            ));
        }
    }

    let graph = build_stream_graph(stream, flow_dir)
        .map_err(|e| surtgis_core::Error::Other(e.to_string()))?;

    // θ grid: inclusive endpoints, step interval.
    let mut theta_grid: Vec<f64> = Vec::new();
    let mut t = lo;
    while t <= hi + 1e-12 {
        theta_grid.push((t * 1e4).round() / 1e4); // mild rounding for cleanliness
        t += params.theta_step;
    }
    let n_theta = theta_grid.len();

    // ── 1. Group stream nodes by basin id. ────────────────────────────
    //
    // basins[r, c] of 0 means "no basin"; skip those cells.
    let mut basin_nodes: std::collections::HashMap<i32, Vec<usize>> =
        std::collections::HashMap::new();
    for (idx, &(r, c)) in graph.stream_cells.iter().enumerate() {
        let bid = basins.get(r, c).unwrap_or(0);
        if bid == 0 {
            continue;
        }
        basin_nodes.entry(bid).or_default().push(idx);
    }

    let mut out: Vec<ConcavityResult> = Vec::with_capacity(basin_nodes.len());

    // Sort basin ids for deterministic output ordering.
    let mut sorted_bids: Vec<i32> = basin_nodes.keys().copied().collect();
    sorted_bids.sort();

    for bid in sorted_bids {
        let nodes = &basin_nodes[&bid];
        if nodes.len() < params.min_basin_cells {
            continue;
        }
        let basin_set: HashSet<usize> = nodes.iter().copied().collect();

        // Find the basin's outlet(s): basin-stream nodes whose downstream
        // link is either None (graph outlet) OR points outside the basin
        // set (channel exits the basin).
        let outlets: Vec<usize> = nodes
            .iter()
            .copied()
            .filter(|&i| match graph.downstream_link[i] {
                None => true,
                Some(d) => !basin_set.contains(&d),
            })
            .collect();
        if outlets.is_empty() {
            continue;
        }

        // Collect basin elevations once.
        let mut elevations: Vec<f64> = Vec::with_capacity(nodes.len());
        let mut valid_node_indices: Vec<usize> = Vec::with_capacity(nodes.len());
        for &i in nodes {
            let (r, c) = graph.stream_cells[i];
            let z = dem.get(r, c).unwrap_or(f64::NAN);
            if z.is_finite() {
                elevations.push(z);
                valid_node_indices.push(i);
            }
        }
        if valid_node_indices.len() < params.min_basin_cells {
            continue;
        }
        let z_var = variance(&elevations);
        if z_var < 1e-6 {
            // Constant elevation → fit RMSE is identically 0 across all θ;
            // estimator is meaningless. Skip.
            continue;
        }

        // ── 2. χ_per_theta[t_idx][cell_pos_in_valid] ─────────────────
        let mut chi_per_theta: Vec<Vec<f64>> = Vec::with_capacity(n_theta);
        for &theta in &theta_grid {
            let chi_map = compute_basin_chi(
                &graph,
                &basin_set,
                &outlets,
                flow_acc,
                theta,
                params.cell_size_m,
            );
            // chi_map: node_idx → χ. Build a vector aligned to valid_node_indices.
            let mut chi_vec: Vec<f64> = Vec::with_capacity(valid_node_indices.len());
            for &i in &valid_node_indices {
                chi_vec.push(*chi_map.get(&i).unwrap_or(&f64::NAN));
            }
            chi_per_theta.push(chi_vec);
        }

        // ── 3. Grid-search θ_opt on the full sample. ───────────────────
        let full_idx: Vec<usize> = (0..valid_node_indices.len()).collect();
        let (theta_opt, rmse_opt) =
            grid_search_theta(&chi_per_theta, &elevations, &theta_grid, &full_idx);

        // ── 4. Bootstrap CI. ────────────────────────────────────────────
        let theta_ci = if params.bootstrap_n == 0 {
            (theta_opt, theta_opt)
        } else {
            let n = valid_node_indices.len();
            let mut samples: Vec<f64> = Vec::with_capacity(params.bootstrap_n);
            for boot in 0..params.bootstrap_n {
                // Deterministic resample-with-replacement: hash(seed, basin, boot, k)
                // → index in 0..n.
                let resampled_idx: Vec<usize> = (0..n)
                    .map(|k| {
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        params.seed.hash(&mut h);
                        bid.hash(&mut h);
                        boot.hash(&mut h);
                        k.hash(&mut h);
                        (h.finish() as usize) % n
                    })
                    .collect();
                let (t_b, _) =
                    grid_search_theta(&chi_per_theta, &elevations, &theta_grid, &resampled_idx);
                samples.push(t_b);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let pct = |p: f64| -> f64 {
                let idx =
                    ((p * (samples.len() as f64 - 1.0)).round() as usize).min(samples.len() - 1);
                samples[idx]
            };
            (pct(0.025), pct(0.975))
        };

        out.push(ConcavityResult {
            basin_id: bid,
            theta_opt,
            theta_ci,
            n_cells: valid_node_indices.len(),
            rmse: rmse_opt,
        });
    }

    Ok(out)
}

/// Compute χ for every cell in `basin_set` by BFS upstream from the
/// basin outlets. Returns a map node_idx → χ.
///
/// Mirrors the recurrence in [`super::chi::chi_transform`] but restricted
/// to the basin's nodes and parameterised by the per-iteration θ. The
/// chi origin is the outlet (χ_outlet = 0); χ accumulates upstream with
/// Δx · (A₀ / A(u))^θ where A₀ is fixed at 1·10⁶ m² (only the absolute
/// scale changes with A₀, and the RMSE minimisation is invariant under
/// affine rescaling of χ).
fn compute_basin_chi(
    graph: &StreamGraph,
    basin_set: &HashSet<usize>,
    outlets: &[usize],
    flow_acc: &Raster<f64>,
    theta: f64,
    cell_size_m: f64,
) -> std::collections::HashMap<usize, f64> {
    const A0_M2: f64 = 1.0e6;
    let cell_diag = cell_size_m * std::f64::consts::SQRT_2;
    let cell_area_m2 = cell_size_m * cell_size_m;

    let mut chi: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut queue: VecDeque<usize> = VecDeque::with_capacity(outlets.len());
    for &o in outlets {
        chi.insert(o, 0.0);
        queue.push_back(o);
    }
    while let Some(i) = queue.pop_front() {
        let (ir, ic) = graph.stream_cells[i];
        let chi_i = chi.get(&i).copied().unwrap_or(f64::NAN);
        for &u in &graph.upstream_links[i] {
            if !basin_set.contains(&u) {
                continue; // upstream cell outside this basin
            }
            let (ur, uc) = graph.stream_cells[u];
            let dr = (ur as isize - ir as isize).abs();
            let dc = (uc as isize - ic as isize).abs();
            let dx = if dr + dc == 2 { cell_diag } else { cell_size_m };
            let a_cells = flow_acc.get(ur, uc).unwrap_or(f64::NAN);
            let a_m2 = a_cells * cell_area_m2;
            let chi_u = if chi_i.is_finite() && a_m2 > 0.0 && a_m2.is_finite() {
                chi_i + dx * (A0_M2 / a_m2).powf(theta)
            } else {
                f64::NAN
            };
            chi.insert(u, chi_u);
            queue.push_back(u);
        }
    }
    chi
}

/// Compute the RMSE of OLS regression `z ~ chi`. Returns f64::INFINITY
/// when the regression is undefined (zero variance in χ on the subset).
fn ols_rmse(chi: &[f64], z: &[f64], indices: &[usize]) -> f64 {
    let n = indices.len();
    if n == 0 {
        return f64::INFINITY;
    }
    let mut sum_x = 0.0;
    let mut sum_z = 0.0;
    let mut n_valid = 0.0;
    for &i in indices {
        if !chi[i].is_finite() || !z[i].is_finite() {
            continue;
        }
        sum_x += chi[i];
        sum_z += z[i];
        n_valid += 1.0;
    }
    if n_valid < 3.0 {
        return f64::INFINITY;
    }
    let mean_x = sum_x / n_valid;
    let mean_z = sum_z / n_valid;
    let mut s_xx = 0.0;
    let mut s_xz = 0.0;
    for &i in indices {
        if !chi[i].is_finite() || !z[i].is_finite() {
            continue;
        }
        let dx = chi[i] - mean_x;
        let dz = z[i] - mean_z;
        s_xx += dx * dx;
        s_xz += dx * dz;
    }
    if s_xx < 1e-12 {
        return f64::INFINITY;
    }
    let slope = s_xz / s_xx;
    let intercept = mean_z - slope * mean_x;
    let mut ss_res = 0.0;
    for &i in indices {
        if !chi[i].is_finite() || !z[i].is_finite() {
            continue;
        }
        let pred = slope * chi[i] + intercept;
        let r = z[i] - pred;
        ss_res += r * r;
    }
    (ss_res / n_valid).sqrt()
}

fn grid_search_theta(
    chi_per_theta: &[Vec<f64>],
    z: &[f64],
    theta_grid: &[f64],
    indices: &[usize],
) -> (f64, f64) {
    let mut best_t = theta_grid[0];
    let mut best_rmse = f64::INFINITY;
    for (ti, &theta) in theta_grid.iter().enumerate() {
        let rmse = ols_rmse(&chi_per_theta[ti], z, indices);
        if rmse < best_rmse {
            best_rmse = rmse;
            best_t = theta;
        }
    }
    (best_t, best_rmse)
}

fn variance(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len() as f64;
    let mean: f64 = v.iter().sum::<f64>() / n;
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::raster::Raster;

    fn raster_u8(data: Vec<Vec<u8>>) -> Raster<u8> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<u8> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        Raster::from_array(arr)
    }
    fn raster_f64(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        Raster::from_array(arr)
    }
    fn raster_i32(data: Vec<Vec<i32>>) -> Raster<i32> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<i32> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        Raster::from_array(arr)
    }

    /// Build a single linear channel of length `n` where drainage area
    /// scales as A(c) = (c_from_outlet + 1) in cell counts and elevation
    /// follows the steady-state z = K · χ(θ_true) + base. The
    /// concavity_index estimator should recover θ_true.
    fn synthetic_basin(
        n: usize,
        theta_true: f64,
        k: f64,
        base: f64,
        cell: f64,
    ) -> (
        Raster<u8>,
        Raster<u8>,
        Raster<f64>,
        Raster<f64>,
        Raster<i32>,
    ) {
        // Layout: single row, channel flowing east, outlet at the east edge.
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]); // all east; col n-1 exits raster
        // Drainage area: leftmost (headwater) = 1 cell, rightmost (outlet) = n.
        let acc_vals: Vec<f64> = (1..=n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc_vals.clone()]);
        // Compute χ at each cell using θ_true, then z = base + k · χ.
        // χ at outlet = 0; χ at cell c (counting from outlet, c=0 at outlet)
        // = Σ_{c'=1..=c} cell · (A0 / A(c'))^θ_true
        // where A(c') = n - c' (the count of cells downstream + 1 from cell c').
        // Wait — we built acc so col j has A = j+1 (col 0 = headwater, A=1;
        // col n-1 = outlet, A=n). χ at col j is computed walking upstream
        // from col n-1; the cell at col j is `n-1-j` steps upstream of outlet.
        const A0: f64 = 1.0e6;
        let cell_area_m2 = cell * cell;
        let mut chi_at = vec![0.0f64; n];
        for jj in 1..n {
            // walking upstream: cell to compute is col (n-1-jj); previous was col (n-1-jj+1)
            let col_u = n - 1 - jj;
            let col_d = col_u + 1;
            let a_u_m2 = (col_u + 1) as f64 * cell_area_m2;
            chi_at[col_u] = chi_at[col_d] + cell * (A0 / a_u_m2).powf(theta_true);
        }
        let z_vec: Vec<f64> = chi_at.iter().map(|x| base + k * x).collect();
        let dem = raster_f64(vec![z_vec]);
        // One basin covering the entire channel.
        let basins = raster_i32(vec![vec![1; n]]);

        (stream, flow_dir, flow_acc, dem, basins)
    }

    /// Spec §7.2 headline: synthetic profile with θ_true=0.45,
    /// estimator must recover θ_opt in [0.40, 0.50] given ≥ 200 cells.
    #[test]
    fn synthetic_steady_state_recovers_theta() {
        let (stream, flow_dir, flow_acc, dem, basins) =
            synthetic_basin(300, 0.45, 1.0, 100.0, 30.0);

        let params = ConcavityParams {
            theta_range: (0.10, 0.90),
            theta_step: 0.05,
            bootstrap_n: 0, // skip CI for speed; tested separately
            cell_size_m: 30.0,
            seed: 42,
            min_basin_cells: 30,
        };
        let results = concavity_index(&stream, &flow_dir, &flow_acc, &dem, &basins, params)
            .expect("concavity ok");
        assert_eq!(results.len(), 1, "exactly one basin");
        let r = &results[0];
        assert!(
            (r.theta_opt - 0.45).abs() < 0.06,
            "theta_opt {} should be within 0.06 of 0.45",
            r.theta_opt
        );
        assert!(
            r.rmse < 1e-6,
            "RMSE on a perfectly linear profile should be ~0, got {}",
            r.rmse
        );
    }

    /// Bootstrap CI must (a) be non-degenerate (low < high or at least
    /// low == high == theta_opt for very tight problems) and (b) bracket
    /// θ_opt itself.
    #[test]
    fn bootstrap_ci_brackets_theta_opt() {
        let (stream, flow_dir, flow_acc, dem, basins) =
            synthetic_basin(200, 0.45, 1.0, 100.0, 30.0);

        let params = ConcavityParams {
            theta_range: (0.10, 0.90),
            theta_step: 0.05,
            bootstrap_n: 50, // smaller for test speed
            cell_size_m: 30.0,
            seed: 42,
            min_basin_cells: 30,
        };
        let results =
            concavity_index(&stream, &flow_dir, &flow_acc, &dem, &basins, params).expect("ok");
        let r = &results[0];
        // For a perfectly linear synthetic profile, every bootstrap sample
        // picks (almost) the same θ; CI is tight but well-defined.
        assert!(
            r.theta_ci.0 <= r.theta_opt && r.theta_opt <= r.theta_ci.1,
            "θ_opt {} should fall inside CI [{}, {}]",
            r.theta_opt,
            r.theta_ci.0,
            r.theta_ci.1,
        );
        assert!(
            r.theta_ci.0 >= 0.1 && r.theta_ci.1 <= 0.9,
            "CI must stay inside the searched θ range"
        );
    }

    /// Basins with too few stream cells must be silently skipped.
    #[test]
    fn small_basin_is_skipped() {
        // 10-cell channel, but require min_basin_cells = 30 → should skip.
        let (stream, flow_dir, flow_acc, dem, basins) = synthetic_basin(10, 0.45, 1.0, 100.0, 30.0);
        let params = ConcavityParams {
            min_basin_cells: 30,
            bootstrap_n: 0,
            ..ConcavityParams::default()
        };
        let results =
            concavity_index(&stream, &flow_dir, &flow_acc, &dem, &basins, params).expect("ok");
        assert!(
            results.is_empty(),
            "10-cell basin should be skipped, got {:?}",
            results
        );
    }

    /// Two independent basins must yield two independent estimates.
    #[test]
    fn two_basins_yield_two_independent_estimates() {
        // Layout: row 0 is basin 1, row 2 is basin 2. Both same θ_true.
        let n = 200;
        let stream = raster_u8(vec![vec![1u8; n], vec![0u8; n], vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n], vec![0u8; n], vec![1u8; n]]);
        let acc_row: Vec<f64> = (1..=n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc_row.clone(), vec![0.0; n], acc_row]);
        // Compute z for each row independently, both with θ_true = 0.45.
        let (theta_true, k, base, cell) = (0.45_f64, 1.0_f64, 100.0_f64, 30.0_f64);
        const A0: f64 = 1.0e6;
        let cell_area_m2 = cell * cell;
        let mut chi_at = vec![0.0f64; n];
        for jj in 1..n {
            let col_u = n - 1 - jj;
            let a_u_m2 = (col_u + 1) as f64 * cell_area_m2;
            chi_at[col_u] = chi_at[col_u + 1] + cell * (A0 / a_u_m2).powf(theta_true);
        }
        let z_row: Vec<f64> = chi_at.iter().map(|x| base + k * x).collect();
        let dem = raster_f64(vec![z_row.clone(), vec![0.0; n], z_row]);
        let basins = raster_i32(vec![vec![1; n], vec![0; n], vec![2; n]]);

        let params = ConcavityParams {
            bootstrap_n: 0,
            min_basin_cells: 30,
            ..ConcavityParams::default()
        };
        let results =
            concavity_index(&stream, &flow_dir, &flow_acc, &dem, &basins, params).expect("ok");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].basin_id, 1);
        assert_eq!(results[1].basin_id, 2);
        for r in &results {
            assert!(
                (r.theta_opt - 0.45).abs() < 0.06,
                "basin {} θ {} not close to 0.45",
                r.basin_id,
                r.theta_opt
            );
        }
    }
}
