//! Divide-migration metrics (Whipple, Forte et al. 2017; Willett et al. 2014).
//!
//! Quantifies asymmetry across watershed divides. Two adjacent basins
//! are in equilibrium when their headwaters sit at the same χ; departures
//! flag basins that are "winning" or "losing" area through divide
//! migration. Combined with the Gilbert metric (relief / elevation
//! difference across the divide), this is the standard topographic
//! diagnostic for active divide migration.
//!
//! ## References
//!
//! - Willett, S.D., McCoy, S.W., Perron, J.T., Goren, L. & Chen, C.-Y.
//!   (2014). *Dynamic reorganization of river basins.*
//!   Science 343, 1248765. <https://doi.org/10.1126/science.1248765>
//! - Whipple, K.X., Forte, A.M., DiBiase, R.A., Gasparini, N.M. &
//!   Ouimet, W.B. (2017). *Timescales of landscape response to divide
//!   migration.* JGR-Earth Surface 122, 248–273.
//!   <https://doi.org/10.1002/2016JF003973>
//!
//! ## Algorithm (spec §4.5)
//!
//! 1. Scan every cell. A cell at `(r, c)` is a **divide cell** if any of
//!    its 4-neighbours `(r', c')` has a different (nonzero) basin id.
//!    The pair `((r, c), (r', c'))` is one **divide pair** between
//!    `basins[r,c]` and `basins[r',c']`.
//! 2. Group divide pairs by the sorted basin id pair
//!    `(min(a, b), max(a, b))`. Each group is one divide between two
//!    adjacent basins.
//! 3. For each group, compute:
//!    - **Δχ** at every pair → median is the divide's χ asymmetry.
//!      Positive ⇒ side `basin_a` (the smaller id) has higher χ ⇒
//!      "losing" to side `basin_b`.
//!    - **Δelev** = `z(side_a) − z(side_b)`. Median is the Gilbert
//!      elevation asymmetry.
//!    - **Δrelief** = `relief(side_a) − relief(side_b)` where `relief`
//!      is local max − min over a 3×3 window centred on each cell.
//!      Median across pairs is the Gilbert relief asymmetry.
//! 4. Build a LineString geometry per group by greedy nearest-neighbour
//!    traversal of the cell-pair midpoints; filter groups whose
//!    cumulative polyline length is below `min_divide_length_m`.
//!
//! ## v1 simplifications
//!
//! - 4-connected neighbourhood for divide detection (not 8). Diagonal
//!   neighbours are also adjacent in raster topology but produce more
//!   "checkerboard" divide pairs; 4-connected gives cleaner polylines.
//! - Polyline ordering is greedy nearest-neighbour — adequate for the
//!   typical linear divide, suboptimal for branched divides. Documented.
//! - "Side A" of the divide is always the cell whose basin id is the
//!   *smaller* of the two basin ids in the sorted pair. This keeps the
//!   sign of Δχ / Δelev / Δrelief consistent across rerun s.

use std::collections::HashMap;

use surtgis_core::{Raster, Result};

/// Parameters for [`divide_migration`].
#[derive(Debug, Clone)]
pub struct DivideMigrationParams {
    /// Cell size in metres. Caller-supplied; the CLI handler reads it
    /// from the raster transform under the standard CRS heuristic.
    pub cell_size_m: f64,
    /// Minimum cumulative polyline length (m) for a divide to be
    /// reported. Default 500 m — short divides are rarely robust.
    pub min_divide_length_m: f64,
}

impl Default for DivideMigrationParams {
    fn default() -> Self {
        Self {
            cell_size_m: 30.0,
            min_divide_length_m: 500.0,
        }
    }
}

/// One divide between two adjacent basins.
#[derive(Debug, Clone)]
pub struct DivideSegment {
    /// Polyline of `(x, y)` midpoint coordinates in source-CRS units,
    /// ordered by greedy nearest-neighbour traversal of divide pairs.
    pub coordinates: Vec<(f64, f64)>,
    /// Basin id with the smaller numeric value of the pair.
    pub basin_a: i32,
    /// Basin id with the larger numeric value of the pair.
    pub basin_b: i32,
    /// Median `χ(side_a) − χ(side_b)`. NaN when no χ raster supplied.
    pub median_chi_diff: f64,
    /// Median `z(side_a) − z(side_b)` (Gilbert elevation asymmetry).
    pub median_elev_diff: f64,
    /// Median local-relief difference `relief(side_a) − relief(side_b)`.
    pub median_relief_diff: f64,
    /// Number of cell pairs contributing to the medians.
    pub n_pairs: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum DivideMigrationError {
    #[error("raster shape mismatch: {0:?} vs {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),
    #[error("DivideMigrationParams.cell_size_m must be > 0 (got {0})")]
    NonPositiveCellSize(f64),
    #[error("DivideMigrationParams.min_divide_length_m must be >= 0 (got {0})")]
    NegativeMinLength(f64),
}

/// Compute divide-migration metrics for every adjacent-basin pair.
///
/// `chi` is optional: when `None`, the `median_chi_diff` field of every
/// segment is NaN; the elevation / relief asymmetries are still
/// computed.
pub fn divide_migration(
    basins: &Raster<i32>,
    dem: &Raster<f64>,
    chi: Option<&Raster<f64>>,
    flow_acc: &Raster<f64>,
    params: DivideMigrationParams,
) -> Result<Vec<DivideSegment>> {
    if params.cell_size_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            DivideMigrationError::NonPositiveCellSize(params.cell_size_m).to_string(),
        ));
    }
    if params.min_divide_length_m < 0.0 {
        return Err(surtgis_core::Error::Other(
            DivideMigrationError::NegativeMinLength(params.min_divide_length_m).to_string(),
        ));
    }
    let b_shape = basins.shape();
    for other in [dem.shape(), flow_acc.shape()] {
        if other != b_shape {
            return Err(surtgis_core::Error::Other(
                DivideMigrationError::ShapeMismatch(other, b_shape).to_string(),
            ));
        }
    }
    if let Some(c) = chi {
        if c.shape() != b_shape {
            return Err(surtgis_core::Error::Other(
                DivideMigrationError::ShapeMismatch(c.shape(), b_shape).to_string(),
            ));
        }
    }
    let (rows, cols) = b_shape;
    let cell = params.cell_size_m;
    let _ = flow_acc; // currently unused; kept in signature per spec for future area-weighted statistics

    // ── 1. Scan for divide pairs (4-connected). ───────────────────────
    //
    // For each pair of 4-adjacent cells with distinct nonzero basin ids,
    // emit a `(basin_min, basin_max, cell_a, cell_b)` tuple. To avoid
    // double counting we only scan east + south neighbours per cell.
    type PairKey = (i32, i32);
    let mut groups: HashMap<PairKey, Vec<DividePair>> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let bid = basins.get(r, c).unwrap_or(0);
            if bid == 0 {
                continue;
            }
            // East neighbour
            if c + 1 < cols {
                let bid_e = basins.get(r, c + 1).unwrap_or(0);
                if bid_e != 0 && bid_e != bid {
                    record_pair(&mut groups, basins, dem, chi, rows, cols, (r, c), (r, c + 1));
                }
            }
            // South neighbour
            if r + 1 < rows {
                let bid_s = basins.get(r + 1, c).unwrap_or(0);
                if bid_s != 0 && bid_s != bid {
                    record_pair(&mut groups, basins, dem, chi, rows, cols, (r, c), (r + 1, c));
                }
            }
        }
    }

    // ── 2. Per-group: medians + polyline + length filter. ─────────────
    let mut out: Vec<DivideSegment> = Vec::with_capacity(groups.len());
    let mut keys: Vec<PairKey> = groups.keys().copied().collect();
    keys.sort();

    for key in keys {
        let pairs = &groups[&key];
        if pairs.is_empty() {
            continue;
        }
        // Per-cell midpoint geometry (x, y) in raster CRS — cell-centre
        // midpoint between the two cells of each divide pair.
        let mut midpoints: Vec<(f64, f64)> = Vec::with_capacity(pairs.len());
        for p in pairs {
            let (xa, ya) = cell_centre(basins, p.row_a, p.col_a);
            let (xb, yb) = cell_centre(basins, p.row_b, p.col_b);
            midpoints.push(((xa + xb) * 0.5, (ya + yb) * 0.5));
        }
        let coords = greedy_polyline(&midpoints);
        let total_len = polyline_length(&coords);
        if total_len < params.min_divide_length_m {
            continue;
        }

        let median_chi = median_of(&pairs.iter().map(|p| p.chi_diff).collect::<Vec<_>>());
        let median_elev = median_of(&pairs.iter().map(|p| p.elev_diff).collect::<Vec<_>>());
        let median_relief = median_of(&pairs.iter().map(|p| p.relief_diff).collect::<Vec<_>>());

        out.push(DivideSegment {
            coordinates: coords,
            basin_a: key.0,
            basin_b: key.1,
            median_chi_diff: median_chi,
            median_elev_diff: median_elev,
            median_relief_diff: median_relief,
            n_pairs: pairs.len(),
        });
        let _ = cell; // not used in this function — `polyline_length` uses Cartesian distance directly
    }

    Ok(out)
}

#[derive(Debug, Clone)]
struct DividePair {
    /// "Side A" = the cell whose basin id matches the *smaller* of the
    /// two ids in the sorted pair.
    row_a: usize,
    col_a: usize,
    row_b: usize,
    col_b: usize,
    elev_diff: f64,
    relief_diff: f64,
    chi_diff: f64,
}

fn record_pair(
    groups: &mut HashMap<(i32, i32), Vec<DividePair>>,
    basins: &Raster<i32>,
    dem: &Raster<f64>,
    chi: Option<&Raster<f64>>,
    rows: usize,
    cols: usize,
    (r1, c1): (usize, usize),
    (r2, c2): (usize, usize),
) {
    let b1 = basins.get(r1, c1).unwrap_or(0);
    let b2 = basins.get(r2, c2).unwrap_or(0);
    if b1 == 0 || b2 == 0 || b1 == b2 {
        return;
    }
    let key = if b1 < b2 { (b1, b2) } else { (b2, b1) };
    // "Side A" is the cell whose basin == key.0 (the smaller id).
    let (sa_r, sa_c, sb_r, sb_c) = if b1 == key.0 {
        (r1, c1, r2, c2)
    } else {
        (r2, c2, r1, c1)
    };
    let z_a = dem.get(sa_r, sa_c).unwrap_or(f64::NAN);
    let z_b = dem.get(sb_r, sb_c).unwrap_or(f64::NAN);
    let chi_a = chi.map(|c| c.get(sa_r, sa_c).unwrap_or(f64::NAN));
    let chi_b = chi.map(|c| c.get(sb_r, sb_c).unwrap_or(f64::NAN));
    let chi_diff = match (chi_a, chi_b) {
        (Some(a), Some(b)) if a.is_finite() && b.is_finite() => a - b,
        _ => f64::NAN,
    };
    let r_a = local_relief(dem, sa_r, sa_c, rows, cols);
    let r_b = local_relief(dem, sb_r, sb_c, rows, cols);
    groups.entry(key).or_default().push(DividePair {
        row_a: sa_r,
        col_a: sa_c,
        row_b: sb_r,
        col_b: sb_c,
        elev_diff: z_a - z_b,
        relief_diff: r_a - r_b,
        chi_diff,
    });
}

/// Local relief = max(z) - min(z) over a 3×3 window centred on (r, c).
/// Cells partially outside the raster shrink the window; NaN samples
/// are excluded from the min/max.
fn local_relief(dem: &Raster<f64>, r: usize, c: usize, rows: usize, cols: usize) -> f64 {
    let r0 = r.saturating_sub(1);
    let c0 = c.saturating_sub(1);
    let r1 = (r + 1).min(rows - 1);
    let c1 = (c + 1).min(cols - 1);
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for rr in r0..=r1 {
        for cc in c0..=c1 {
            let v = dem.get(rr, cc).unwrap_or(f64::NAN);
            if v.is_finite() {
                if v < mn { mn = v; }
                if v > mx { mx = v; }
            }
        }
    }
    if mn.is_finite() && mx.is_finite() {
        mx - mn
    } else {
        f64::NAN
    }
}

fn cell_centre(raster: &Raster<i32>, r: usize, c: usize) -> (f64, f64) {
    let gt = raster.transform();
    let (x0, y0) = raster.pixel_to_geo(c, r);
    (x0 + 0.5 * gt.pixel_width, y0 + 0.5 * gt.pixel_height)
}

/// Greedy nearest-neighbour polyline ordering. Starts at points[0],
/// repeatedly appends the nearest unvisited point. Adequate for simple
/// linear divides; suboptimal for divides with branches (those would
/// require an Eulerian path solve we leave for v2).
fn greedy_polyline(points: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![points[0]];
    }
    let mut visited = vec![false; n];
    let mut order: Vec<(f64, f64)> = Vec::with_capacity(n);
    let mut cur = 0usize;
    visited[cur] = true;
    order.push(points[cur]);
    for _ in 1..n {
        let mut best = usize::MAX;
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            if visited[j] { continue; }
            let dx = points[j].0 - points[cur].0;
            let dy = points[j].1 - points[cur].1;
            let d = dx * dx + dy * dy;
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        if best == usize::MAX { break; }
        visited[best] = true;
        order.push(points[best]);
        cur = best;
    }
    order
}

fn polyline_length(coords: &[(f64, f64)]) -> f64 {
    let mut total = 0.0;
    for w in coords.windows(2) {
        let dx = w[1].0 - w[0].0;
        let dy = w[1].1 - w[0].1;
        total += (dx * dx + dy * dy).sqrt();
    }
    total
}

fn median_of(v: &[f64]) -> f64 {
    let mut finite: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = finite.len();
    if n % 2 == 0 {
        (finite[n / 2 - 1] + finite[n / 2]) * 0.5
    } else {
        finite[n / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::raster::Raster;

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

    /// Spec §7.2 headline test: two adjacent basins with mirror-image
    /// DEMs (same elevation profile) and identical χ → median Δχ across
    /// the divide must be ≈ 0.
    #[test]
    fn symmetric_basins_yield_zero_chi_diff() {
        // 5×10 grid. Left half = basin 1, right half = basin 2.
        // Identical elevation on both sides.
        let n_cols = 10;
        let n_rows = 5;
        let mut basins_data = vec![vec![0i32; n_cols]; n_rows];
        let mut dem_data = vec![vec![0f64; n_cols]; n_rows];
        let mut chi_data = vec![vec![0f64; n_cols]; n_rows];
        for r in 0..n_rows {
            for c in 0..n_cols {
                basins_data[r][c] = if c < n_cols / 2 { 1 } else { 2 };
                dem_data[r][c] = (n_rows - 1 - r) as f64 * 10.0; // elevation independent of column
                chi_data[r][c] = (n_rows - 1 - r) as f64 * 5.0;
            }
        }
        let basins = raster_i32(basins_data);
        let dem = raster_f64(dem_data);
        let chi = raster_f64(chi_data);
        let flow_acc = raster_f64(vec![vec![1.0f64; n_cols]; n_rows]);

        let params = DivideMigrationParams {
            cell_size_m: 30.0,
            min_divide_length_m: 0.0,
        };
        let result =
            divide_migration(&basins, &dem, Some(&chi), &flow_acc, params).expect("ok");
        assert_eq!(result.len(), 1);
        let d = &result[0];
        assert_eq!((d.basin_a, d.basin_b), (1, 2));
        assert!(d.median_chi_diff.abs() < 1e-9,
            "symmetric basins should give |Δχ| ≈ 0, got {}", d.median_chi_diff);
        assert!(d.median_elev_diff.abs() < 1e-9,
            "symmetric basins should give |Δelev| ≈ 0, got {}", d.median_elev_diff);
    }

    /// Asymmetric basins: basin 1 has higher elevations than basin 2 →
    /// median Δelev must be positive (side_a = basin 1 = smaller id).
    #[test]
    fn asymmetric_basins_yield_consistent_sign() {
        let n_cols = 10;
        let n_rows = 5;
        let mut basins_data = vec![vec![0i32; n_cols]; n_rows];
        let mut dem_data = vec![vec![0f64; n_cols]; n_rows];
        for r in 0..n_rows {
            for c in 0..n_cols {
                basins_data[r][c] = if c < n_cols / 2 { 1 } else { 2 };
                // Basin 1: high (100 m). Basin 2: low (50 m).
                dem_data[r][c] = if c < n_cols / 2 { 100.0 } else { 50.0 };
            }
        }
        let basins = raster_i32(basins_data);
        let dem = raster_f64(dem_data);
        let flow_acc = raster_f64(vec![vec![1.0f64; n_cols]; n_rows]);

        let params = DivideMigrationParams {
            cell_size_m: 30.0,
            min_divide_length_m: 0.0,
        };
        let result =
            divide_migration(&basins, &dem, None, &flow_acc, params).expect("ok");
        assert_eq!(result.len(), 1);
        let d = &result[0];
        assert_eq!((d.basin_a, d.basin_b), (1, 2));
        assert!((d.median_elev_diff - 50.0).abs() < 1e-9,
            "expected median Δelev = +50, got {}", d.median_elev_diff);
        assert!(d.median_chi_diff.is_nan(), "chi=None must give NaN chi_diff");
    }

    /// min_divide_length_m filters out short divides.
    #[test]
    fn short_divide_below_length_filter_is_dropped() {
        // 3×3 grid: two basins with only 1 cell-pair divide → polyline
        // length ≈ 0 (one midpoint). Should be filtered when min > 0.
        let basins = raster_i32(vec![
            vec![1, 1, 2],
            vec![1, 0, 0],
            vec![0, 0, 0],
        ]);
        let dem = raster_f64(vec![vec![1.0f64; 3]; 3]);
        let flow_acc = raster_f64(vec![vec![1.0f64; 3]; 3]);
        let params = DivideMigrationParams {
            cell_size_m: 30.0,
            min_divide_length_m: 100.0, // requires >= 100 m of divide
        };
        let result = divide_migration(&basins, &dem, None, &flow_acc, params).expect("ok");
        assert!(result.is_empty(), "1-cell divide should be filtered, got {:?}", result);
    }

    /// Three basins → up to 3 divide pairs (1-2, 1-3, 2-3). Pairing
    /// invariant: smaller id is always `basin_a`.
    #[test]
    fn three_basins_yield_correct_pairing() {
        // 3×9 layout, three side-by-side basins.
        let basins = raster_i32(vec![
            vec![1, 1, 1, 2, 2, 2, 3, 3, 3],
            vec![1, 1, 1, 2, 2, 2, 3, 3, 3],
            vec![1, 1, 1, 2, 2, 2, 3, 3, 3],
        ]);
        let dem = raster_f64(vec![vec![10.0; 9]; 3]);
        let flow_acc = raster_f64(vec![vec![1.0; 9]; 3]);
        let params = DivideMigrationParams {
            cell_size_m: 30.0,
            min_divide_length_m: 0.0,
        };
        let result = divide_migration(&basins, &dem, None, &flow_acc, params).expect("ok");
        // Only TWO divides: 1↔2 and 2↔3 (no 1↔3 because they don't touch).
        let pairs: Vec<(i32, i32)> = result.iter().map(|d| (d.basin_a, d.basin_b)).collect();
        assert_eq!(pairs.len(), 2, "expected 2 divides, got {:?}", pairs);
        assert!(pairs.contains(&(1, 2)) && pairs.contains(&(2, 3)));
        for d in &result {
            assert!(d.basin_a < d.basin_b, "basin_a must be the smaller id");
        }
    }

    /// chi=None path: chi_diff per segment = NaN, elev/relief still computed.
    #[test]
    fn chi_none_produces_nan_chi_diff_only() {
        let basins = raster_i32(vec![
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 1, 1, 2, 2, 2],
            vec![1, 1, 1, 2, 2, 2],
        ]);
        let dem = raster_f64(vec![vec![10.0; 6]; 4]);
        let flow_acc = raster_f64(vec![vec![1.0; 6]; 4]);
        let result = divide_migration(
            &basins,
            &dem,
            None,
            &flow_acc,
            DivideMigrationParams {
                cell_size_m: 30.0,
                min_divide_length_m: 0.0,
            },
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        let d = &result[0];
        assert!(d.median_chi_diff.is_nan(), "chi_diff must be NaN with chi=None");
        assert!(d.median_elev_diff.is_finite(), "elev_diff must remain finite");
    }
}
