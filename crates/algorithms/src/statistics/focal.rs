//! Focal (moving window) statistics
//!
//! Computes statistics within a moving window centered on each cell.
//! Supports: Mean, StdDev, Min, Max, Range, Sum, Count, Median, Percentile.

use super::focal_fast::{circular_moment_stat, hgw_square_2d, huang_focal, square_moment_stat};
use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Available focal statistics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocalStatistic {
    /// Arithmetic mean
    Mean,
    /// Standard deviation (population)
    StdDev,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Range (max - min)
    Range,
    /// Sum of values
    Sum,
    /// Count of valid (non-NaN) values
    Count,
    /// Median value
    Median,
    /// Percentile (0-100)
    Percentile(f64),
    /// Majority (mode) — most frequent value
    Majority,
}

/// Parameters for focal statistics
#[derive(Debug, Clone)]
pub struct FocalParams {
    /// Window radius (actual window size = 2*radius + 1)
    pub radius: usize,
    /// Statistic to compute
    pub statistic: FocalStatistic,
    /// Whether to use circular window (default: false = square)
    pub circular: bool,
}

impl Default for FocalParams {
    fn default() -> Self {
        Self {
            radius: 1,
            statistic: FocalStatistic::Mean,
            circular: false,
        }
    }
}

/// Compute focal statistics on a raster
///
/// Applies a moving window of the specified radius and computes the
/// requested statistic for all valid cells within the window.
///
/// Mean/StdDev/Sum/Count use a summed-area table (square window) or a
/// per-row prefix table (circular window): O(1)/O(radius) per cell.
/// Min/Max/Range use the van Herk-Gil-Werman separable sliding filter for
/// a square window (O(1) amortized per cell); a circular window falls
/// back to a direct scan over the circular offsets, without the
/// per-cell `Vec` allocation of the original brute-force path. Median
/// and Percentile use Huang's sliding histogram (O(radius) amortized per
/// cell, both window shapes). Majority is unchanged (out of scope for
/// this optimization pass).
///
/// # Arguments
/// * `raster` - Input raster
/// * `params` - Focal parameters (radius, statistic, circular)
///
/// # Returns
/// Raster with the computed statistic at each cell
pub fn focal_statistics(raster: &Raster<f64>, params: FocalParams) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Focal radius must be > 0".into()));
    }

    if let FocalStatistic::Percentile(p) = params.statistic
        && !(0.0..=100.0).contains(&p)
    {
        return Err(Error::Algorithm(
            "Percentile must be between 0 and 100".into(),
        ));
    }

    let (rows, cols) = raster.shape();
    let radius = params.radius;

    let data: Array2<f64> = match params.statistic {
        FocalStatistic::Majority => focal_bruteforce_array(raster, &params)?,

        FocalStatistic::Mean
        | FocalStatistic::StdDev
        | FocalStatistic::Sum
        | FocalStatistic::Count => {
            let stat_kind: u8 = match params.statistic {
                FocalStatistic::Mean => 0,
                FocalStatistic::StdDev => 1,
                FocalStatistic::Sum => 2,
                FocalStatistic::Count => 3,
                _ => unreachable!(),
            };
            if params.circular {
                circular_moment_stat(raster, radius, stat_kind)
            } else {
                square_moment_stat(raster, radius, stat_kind)
            }
        }

        FocalStatistic::Min | FocalStatistic::Max | FocalStatistic::Range => {
            if params.circular {
                focal_circular_extreme(raster, &params)
            } else {
                match params.statistic {
                    FocalStatistic::Min => hgw_square_2d(raster, radius, true),
                    FocalStatistic::Max => hgw_square_2d(raster, radius, false),
                    FocalStatistic::Range => {
                        let min_arr = hgw_square_2d(raster, radius, true);
                        let max_arr = hgw_square_2d(raster, radius, false);
                        Array2::from_shape_fn((rows, cols), |(r, c)| {
                            let mn = min_arr[[r, c]];
                            let mx = max_arr[[r, c]];
                            if mn.is_nan() || mx.is_nan() {
                                f64::NAN
                            } else {
                                mx - mn
                            }
                        })
                    }
                    _ => unreachable!(),
                }
            }
        }

        FocalStatistic::Median => huang_focal(raster, radius, params.circular, None),
        FocalStatistic::Percentile(p) => huang_focal(raster, radius, params.circular, Some(p)),
    };

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = data;

    Ok(output)
}

/// Direct circular Min/Max/Range scan without a fast path (circular
/// windows are not separable, so van Herk-Gil-Werman does not apply).
/// Still avoids the per-cell `Vec` allocation of the original brute
/// force by reducing min/max inline over the precomputed offsets.
fn focal_circular_extreme(raster: &Raster<f64>, params: &FocalParams) -> Array2<f64> {
    let (rows, cols) = raster.shape();
    let r = params.radius as isize;
    let r_sq = (params.radius * params.radius) as isize;
    let mut offsets = Vec::new();
    for dr in -r..=r {
        for dc in -r..=r {
            if dr * dr + dc * dc <= r_sq {
                offsets.push((dr, dc));
            }
        }
    }

    par_map_rows(rows, cols, |row, out_row| {
        for (col, out) in out_row.iter_mut().enumerate() {
            let mut min_v = f64::INFINITY;
            let mut max_v = f64::NEG_INFINITY;
            let mut any = false;

            for &(dr, dc) in &offsets {
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                    let valid = !raster
                        .is_nodata_at(nr as usize, nc as usize)
                        .unwrap_or(true);
                    if valid {
                        let v = unsafe { raster.get_unchecked(nr as usize, nc as usize) };
                        if v < min_v {
                            min_v = v;
                        }
                        if v > max_v {
                            max_v = v;
                        }
                        any = true;
                    }
                }
            }

            *out = if !any {
                f64::NAN
            } else {
                match params.statistic {
                    FocalStatistic::Min => min_v,
                    FocalStatistic::Max => max_v,
                    FocalStatistic::Range => max_v - min_v,
                    _ => unreachable!(),
                }
            };
        }
    })
}

/// Original brute-force focal statistics: for every cell, collect the
/// valid neighbor values in the window into a `Vec` and dispatch to
/// [`compute_statistic`]. O(k^2) per cell. Kept as the production path
/// for `Majority` (out of scope for the fast-path work) and reused by
/// tests as the reference implementation the fast paths are checked
/// against.
fn focal_bruteforce_array(raster: &Raster<f64>, params: &FocalParams) -> Result<Array2<f64>> {
    let (rows, cols) = raster.shape();
    let r = params.radius as isize;

    // Precompute circular mask offsets if needed
    let offsets: Vec<(isize, isize)> = if params.circular {
        let r_sq = (params.radius * params.radius) as isize;
        let mut offs = Vec::new();
        for dr in -r..=r {
            for dc in -r..=r {
                if dr * dr + dc * dc <= r_sq {
                    offs.push((dr, dc));
                }
            }
        }
        offs
    } else {
        let mut offs = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
        for dr in -r..=r {
            for dc in -r..=r {
                offs.push((dr, dc));
            }
        }
        offs
    };

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                // Collect valid neighbor values
                let mut values: Vec<f64> = Vec::with_capacity(offsets.len());

                for &(dr, dc) in &offsets {
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;

                    if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                        let valid = !raster
                            .is_nodata_at(nr as usize, nc as usize)
                            .unwrap_or(true);
                        if valid {
                            let v = unsafe { raster.get_unchecked(nr as usize, nc as usize) };
                            values.push(v);
                        }
                    }
                }

                if values.is_empty() {
                    continue;
                }

                *row_data_col = compute_statistic(&mut values, &params.statistic);
            }

            row_data
        })
        .collect();

    Array2::from_shape_vec((rows, cols), output_data).map_err(|e| Error::Other(e.to_string()))
}

fn compute_statistic(values: &mut [f64], stat: &FocalStatistic) -> f64 {
    let n = values.len() as f64;

    match stat {
        FocalStatistic::Mean => values.iter().sum::<f64>() / n,
        FocalStatistic::StdDev => {
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            var.sqrt()
        }
        FocalStatistic::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
        FocalStatistic::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        FocalStatistic::Range => {
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            max - min
        }
        FocalStatistic::Sum => values.iter().sum::<f64>(),
        FocalStatistic::Count => n,
        FocalStatistic::Median => {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            }
        }
        FocalStatistic::Percentile(p) => {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (p / 100.0 * (values.len() - 1) as f64).round() as usize;
            values[idx.min(values.len() - 1)]
        }
        FocalStatistic::Majority => {
            // Round values to avoid floating-point noise in class rasters.
            // Uses a simple counting approach: sort, then find longest run.
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut best_val = values[0];
            let mut best_count = 1usize;
            let mut cur_val = values[0];
            let mut cur_count = 1usize;
            for v in &values[1..] {
                // Treat values within 1e-9 as equal (class rasters are typically integers)
                if (*v - cur_val).abs() < 1e-9 {
                    cur_count += 1;
                } else {
                    if cur_count > best_count {
                        best_count = cur_count;
                        best_val = cur_val;
                    }
                    cur_val = *v;
                    cur_count = 1;
                }
            }
            if cur_count > best_count {
                best_val = cur_val;
            }
            best_val
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn uniform_raster(size: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(size, size, value);
        r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        r
    }

    fn gradient_raster(size: usize) -> Raster<f64> {
        let mut r = Raster::new(size, size);
        r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        for row in 0..size {
            for col in 0..size {
                r.set(row, col, (row * size + col) as f64).unwrap();
            }
        }
        r
    }

    #[test]
    fn test_focal_mean_uniform() {
        let r = uniform_raster(10, 5.0);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Mean,
                circular: false,
            },
        )
        .unwrap();
        let v = result.get(5, 5).unwrap();
        assert!(
            (v - 5.0).abs() < 1e-10,
            "Mean of uniform should be 5.0, got {}",
            v
        );
    }

    #[test]
    fn test_focal_min_max() {
        let r = gradient_raster(10);
        let min_result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Min,
                circular: false,
            },
        )
        .unwrap();
        let max_result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Max,
                circular: false,
            },
        )
        .unwrap();

        let min_v = min_result.get(5, 5).unwrap();
        let max_v = max_result.get(5, 5).unwrap();
        // Cell (5,5) = 55, neighbors span (4,4)=44 to (6,6)=66
        assert!((min_v - 44.0).abs() < 1e-10);
        assert!((max_v - 66.0).abs() < 1e-10);
    }

    #[test]
    fn test_focal_std_uniform() {
        let r = uniform_raster(10, 5.0);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::StdDev,
                circular: false,
            },
        )
        .unwrap();
        let v = result.get(5, 5).unwrap();
        assert!(v.abs() < 1e-10, "StdDev of uniform should be 0, got {}", v);
    }

    #[test]
    fn test_focal_range() {
        let r = gradient_raster(10);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Range,
                circular: false,
            },
        )
        .unwrap();
        let v = result.get(5, 5).unwrap();
        assert!((v - 22.0).abs() < 1e-10, "Range should be 22, got {}", v);
    }

    #[test]
    fn test_focal_sum() {
        let r = uniform_raster(10, 1.0);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Sum,
                circular: false,
            },
        )
        .unwrap();
        // Interior cell: 3x3 = 9 cells
        let v = result.get(5, 5).unwrap();
        assert!((v - 9.0).abs() < 1e-10, "Sum should be 9, got {}", v);
    }

    #[test]
    fn test_focal_count() {
        let r = uniform_raster(10, 1.0);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Count,
                circular: false,
            },
        )
        .unwrap();
        let v = result.get(5, 5).unwrap();
        assert!((v - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_focal_median() {
        let r = gradient_raster(10);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Median,
                circular: false,
            },
        )
        .unwrap();
        // Median of 3x3 window around (5,5)=55 should be 55
        let v = result.get(5, 5).unwrap();
        assert!((v - 55.0).abs() < 1e-10, "Median should be 55, got {}", v);
    }

    #[test]
    fn test_focal_circular() {
        let r = uniform_raster(10, 1.0);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 2,
                statistic: FocalStatistic::Count,
                circular: true,
            },
        )
        .unwrap();
        // Circular window r=2: offsets with dr²+dc² <= 4
        // (0,0),(±1,0),(0,±1),(±1,±1),(±2,0),(0,±2) = 13 cells
        let v = result.get(5, 5).unwrap();
        assert!(
            (v - 13.0).abs() < 1e-10,
            "Circular r=2 should have 13 cells, got {}",
            v
        );
    }

    #[test]
    fn test_focal_radius_zero_error() {
        let r = uniform_raster(5, 1.0);
        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 0,
                statistic: FocalStatistic::Mean,
                circular: false,
            },
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_focal_majority() {
        // Create a raster with a dominant class in the center
        let mut r = Raster::new(5, 5);
        r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                r.set(row, col, 1.0).unwrap(); // All class 1
            }
        }
        // Set center and neighbors to class 2 (majority in 3x3 window around center)
        r.set(1, 1, 2.0).unwrap();
        r.set(1, 2, 2.0).unwrap();
        r.set(1, 3, 2.0).unwrap();
        r.set(2, 1, 2.0).unwrap();
        r.set(2, 2, 2.0).unwrap();
        r.set(2, 3, 2.0).unwrap();
        r.set(3, 1, 2.0).unwrap();
        r.set(3, 2, 2.0).unwrap();
        r.set(3, 3, 2.0).unwrap();

        let result = focal_statistics(
            &r,
            FocalParams {
                radius: 1,
                statistic: FocalStatistic::Majority,
                circular: false,
            },
        )
        .unwrap();

        let v = result.get(2, 2).unwrap();
        assert!((v - 2.0).abs() < 1e-10, "Majority should be 2.0, got {}", v);
    }

    // -----------------------------------------------------------------
    // Fast-path vs. brute-force parity tests (P4 optimization sprint)
    // -----------------------------------------------------------------

    /// Tiny deterministic PRNG (xorshift64) — no external `rand` dependency.
    fn xorshift64(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    /// Build a `size x size` raster of pseudo-random values in `[0, 100)`,
    /// with a deterministic fraction of cells set to NaN.
    fn random_raster_with_nan(size: usize, seed: u64, nan_prob: f64) -> Raster<f64> {
        let mut state = seed | 1; // xorshift needs a nonzero seed
        let mut r = Raster::new(size, size);
        r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        r.set_nodata(Some(f64::NAN));
        for row in 0..size {
            for col in 0..size {
                let u1 = (xorshift64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                let u2 = (xorshift64(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
                let value = if u2 < nan_prob { f64::NAN } else { u1 * 100.0 };
                r.set(row, col, value).unwrap();
            }
        }
        r
    }

    /// Assert two values are within `rel_tol` of each other (relative to
    /// the brute-force reference magnitude), plus a small absolute floor
    /// `abs_tol`, or both NaN.
    ///
    /// The absolute floor matters specifically for StdDev: the
    /// summed-area-table variance formula (`E[(x-ref)^2] - E[x-ref]^2`)
    /// can leave a ~1e-12-scale floating-point residual even after
    /// ref-centering, and `sqrt()` amplifies that near zero (variance
    /// ~1e-12 -> stddev ~1e-6) whenever the true variance is exactly or
    /// nearly 0 (e.g. a single-cell window). That is expected numerical
    /// behaviour, not a correctness bug.
    fn assert_close_or_nan(
        fast: f64,
        brute: f64,
        rel_tol: f64,
        abs_tol: f64,
        ctx: impl std::fmt::Display,
    ) {
        if fast.is_nan() && brute.is_nan() {
            return;
        }
        assert!(
            !fast.is_nan() && !brute.is_nan(),
            "NaN mismatch ({ctx}): fast={fast}, brute={brute}"
        );
        let diff = (fast - brute).abs();
        let scale = brute.abs().max(1.0);
        assert!(
            diff <= rel_tol * scale + abs_tol,
            "{ctx}: fast={fast}, brute={brute}, diff={diff}, rel_tol={rel_tol}, abs_tol={abs_tol}"
        );
    }

    /// Compare `focal_statistics` (fast path) against the brute-force
    /// reference for every cell of `raster`, for the given params.
    fn compare_fast_vs_bruteforce(raster: &Raster<f64>, params: FocalParams, rel_tol: f64) {
        let fast = focal_statistics(raster, params.clone()).unwrap();
        let brute = focal_bruteforce_array(raster, &params).unwrap();
        // See `assert_close_or_nan` doc comment: StdDev needs a small
        // absolute floor because sqrt() amplifies near-zero-variance
        // floating point residuals from the summed-area-table formula.
        let abs_tol = if matches!(params.statistic, FocalStatistic::StdDev) {
            1e-4
        } else {
            1e-9
        };
        let (rows, cols) = raster.shape();
        for row in 0..rows {
            for col in 0..cols {
                let f = fast.get(row, col).unwrap();
                let b = *brute.get((row, col)).unwrap();
                assert_close_or_nan(
                    f,
                    b,
                    rel_tol,
                    abs_tol,
                    format!(
                        "stat={:?} circular={} radius={} at ({row},{col})",
                        params.statistic, params.circular, params.radius
                    ),
                );
            }
        }
    }

    fn all_fast_path_statistics() -> Vec<FocalStatistic> {
        vec![
            FocalStatistic::Mean,
            FocalStatistic::StdDev,
            FocalStatistic::Min,
            FocalStatistic::Max,
            FocalStatistic::Range,
            FocalStatistic::Sum,
            FocalStatistic::Count,
            FocalStatistic::Median,
            FocalStatistic::Percentile(25.0),
            FocalStatistic::Percentile(90.0),
        ]
    }

    #[test]
    fn test_fast_path_matches_bruteforce_random_50x50() {
        let r = random_raster_with_nan(50, 0xC0FFEE, 0.05);

        for &radius in &[3usize, 8usize] {
            for circular in [false, true] {
                for stat in all_fast_path_statistics() {
                    let rel_tol = match stat {
                        FocalStatistic::Median | FocalStatistic::Percentile(_) => 1e-2,
                        _ => 1e-6,
                    };
                    compare_fast_vs_bruteforce(
                        &r,
                        FocalParams {
                            radius,
                            statistic: stat,
                            circular,
                        },
                        rel_tol,
                    );
                }
            }
        }
    }

    #[test]
    fn test_fast_path_matches_bruteforce_edge_clipping() {
        // 10x10 raster, radius=8: every window is clipped against the
        // raster bounds for most cells, exercising the boundary handling
        // of the SAT / HGW / Huang fast paths.
        let r = random_raster_with_nan(10, 0xBADC0DE, 0.1);

        for circular in [false, true] {
            for stat in all_fast_path_statistics() {
                let rel_tol = match stat {
                    FocalStatistic::Median | FocalStatistic::Percentile(_) => 1e-2,
                    _ => 1e-6,
                };
                compare_fast_vs_bruteforce(
                    &r,
                    FocalParams {
                        radius: 8,
                        statistic: stat,
                        circular,
                    },
                    rel_tol,
                );
            }
        }
    }

    #[test]
    fn test_fast_path_all_nan_band_preserves_count_zero_nan() {
        // A raster with a full NaN stripe across the middle rows: cells
        // whose entire window falls inside the stripe must be NaN, for
        // both the fast path and the brute-force reference.
        let size = 20;
        let mut r = random_raster_with_nan(size, 0x5EED, 0.0);
        for row in 8..12 {
            for col in 0..size {
                r.set(row, col, f64::NAN).unwrap();
            }
        }

        for circular in [false, true] {
            for stat in all_fast_path_statistics() {
                let rel_tol = match stat {
                    FocalStatistic::Median | FocalStatistic::Percentile(_) => 1e-2,
                    _ => 1e-6,
                };
                compare_fast_vs_bruteforce(
                    &r,
                    FocalParams {
                        radius: 2,
                        statistic: stat,
                        circular,
                    },
                    rel_tol,
                );

                // Explicitly confirm the fast path itself: a cell dead
                // center in the NaN stripe has an entirely-invalid
                // window at radius=1 and must read back as NaN.
                let fast = focal_statistics(
                    &r,
                    FocalParams {
                        radius: 1,
                        statistic: stat,
                        circular,
                    },
                )
                .unwrap();
                let v = fast.get(9, 10).unwrap();
                assert!(
                    v.is_nan(),
                    "expected NaN in all-NaN window for {:?} circular={}, got {}",
                    stat,
                    circular,
                    v
                );
            }
        }
    }

    /// Informal timing comparison, brute force vs. fast path, radius=10 on
    /// a 500x500 raster, for Mean and Min. Not a correctness test — run
    /// with `cargo test -p surtgis-algorithms --release -- --ignored
    /// --nocapture bench_focal_informal`.
    #[test]
    #[ignore]
    fn bench_focal_informal() {
        use std::time::Instant;

        let r = random_raster_with_nan(500, 0x51DE, 0.02);
        let radius = 10;

        for (name, stat) in [("Mean", FocalStatistic::Mean), ("Min", FocalStatistic::Min)] {
            let params = FocalParams {
                radius,
                statistic: stat,
                circular: false,
            };

            let t0 = Instant::now();
            let _ = focal_bruteforce_array(&r, &params).unwrap();
            let brute_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let _ = focal_statistics(&r, params).unwrap();
            let fast_ms = t1.elapsed().as_secs_f64() * 1000.0;

            println!(
                "[bench] {name} radius={radius} 500x500: bruteforce={brute_ms:.2}ms fast={fast_ms:.2}ms speedup={:.1}x",
                brute_ms / fast_ms.max(0.001)
            );
        }
    }
}
