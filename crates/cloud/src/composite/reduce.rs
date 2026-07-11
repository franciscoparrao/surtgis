//! Per-pixel compositing of scene strips: median reduce, coverage-ordered
//! first-valid fill, and iterative neighbor-mean gap fill.
//!
//! These three stages are the scientific core of the STAC composite — the
//! exact per-pixel semantics users get in the output GeoTIFF:
//!
//! 1. [`median_composite`]: per pixel, the median of the finite values
//!    across scenes (mean of the two middle values for even counts).
//! 2. [`fill_from_scenes_by_coverage`]: pixels with no finite value in any
//!    scene at stage 1 (all masked/cloudy) take the first finite value in
//!    coverage order — scenes with more valid pixels are consulted first.
//! 3. [`fill_gaps_neighbor_mean`]: remaining gaps grow inward from their
//!    borders, each pass filling pixels that have ≥ 2 finite 3×3 neighbors
//!    with their mean (Jacobi-style: each pass reads the previous pass's
//!    snapshot).
//!
//! [`composite_scene_strips`] chains the three exactly as the CLI composite
//! always has.

use ndarray::Array2;

/// Scene indices paired with their finite-pixel counts, sorted descending
/// by count — the consultation order for [`fill_from_scenes_by_coverage`].
pub fn coverage_order(scene_strips: &[Array2<f64>]) -> Vec<(usize, usize)> {
    let mut coverage: Vec<(usize, usize)> = scene_strips
        .iter()
        .enumerate()
        .map(|(i, s)| (i, s.iter().filter(|v| v.is_finite()).count()))
        .collect();
    coverage.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    coverage
}

/// Per-pixel median across scene strips.
///
/// Pixels with no finite value in any scene stay NaN. For an even number
/// of finite values the median is the mean of the two middle values.
/// Scene strips smaller than `rows × cols` contribute only where they have
/// data.
pub fn median_composite(scene_strips: &[Array2<f64>], rows: usize, cols: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    let n = scene_strips.len();
    for r in 0..rows {
        for c in 0..cols {
            let mut values: Vec<f64> = Vec::with_capacity(n);
            for strip in scene_strips {
                if r < strip.nrows() && c < strip.ncols() {
                    let v = strip[[r, c]];
                    if v.is_finite() {
                        values.push(v);
                    }
                }
            }
            if !values.is_empty() {
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = values.len() / 2;
                out[[r, c]] = if values.len().is_multiple_of(2) {
                    (values[mid - 1] + values[mid]) / 2.0
                } else {
                    values[mid]
                };
            }
        }
    }
    out
}

/// Fill non-finite pixels of `out` with the first finite value found in
/// `coverage` order (see [`coverage_order`]); stops early once no gaps
/// remain.
pub fn fill_from_scenes_by_coverage(
    out: &mut Array2<f64>,
    scene_strips: &[Array2<f64>],
    coverage: &[(usize, usize)],
) {
    let (rows, cols) = out.dim();
    for &(scene_idx, _) in coverage {
        let strip = &scene_strips[scene_idx];
        for r in 0..rows {
            for c in 0..cols {
                if !out[[r, c]].is_finite() && r < strip.nrows() && c < strip.ncols() {
                    let v = strip[[r, c]];
                    if v.is_finite() {
                        out[[r, c]] = v;
                    }
                }
            }
        }
        let remaining = out.iter().filter(|v| !v.is_finite()).count();
        if remaining == 0 {
            break;
        }
    }
}

/// Iteratively fill remaining non-finite pixels with the mean of their
/// finite 3×3 neighbors (a pixel fills only when it has ≥ 2 finite
/// neighbors), up to `max_passes` passes or until a pass fills nothing.
/// Each pass reads a snapshot of the previous state, so fills don't cascade
/// within a pass. Returns the number of non-finite pixels remaining.
///
/// A fully-NaN buffer is left untouched (there is nothing to grow from).
pub fn fill_gaps_neighbor_mean(out: &mut Array2<f64>, max_passes: usize) -> usize {
    let (rows, cols) = out.dim();
    let mut nan_remaining = out.iter().filter(|v| !v.is_finite()).count();
    if nan_remaining == 0 || nan_remaining == out.len() {
        return nan_remaining;
    }
    let mut prev_buf = out.clone();
    for _pass in 0..max_passes {
        prev_buf.assign(out);
        let mut filled = 0usize;
        for r in 0..rows {
            for c in 0..cols {
                if prev_buf[[r, c]].is_finite() {
                    continue;
                }
                let mut sum = 0.0;
                let mut cnt = 0u32;
                for dr in -1i32..=1 {
                    for dc in -1i32..=1 {
                        let nr = r as i32 + dr;
                        let nc = c as i32 + dc;
                        if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                            let v = prev_buf[[nr as usize, nc as usize]];
                            if v.is_finite() {
                                sum += v;
                                cnt += 1;
                            }
                        }
                    }
                }
                if cnt >= 2 {
                    out[[r, c]] = sum / cnt as f64;
                    filled += 1;
                }
            }
        }
        if filled == 0 {
            break;
        }
        nan_remaining = out.iter().filter(|v| !v.is_finite()).count();
        if nan_remaining == 0 {
            break;
        }
    }
    nan_remaining
}

/// Number of gap-fill passes the composite runs (historical CLI value).
pub const GAP_FILL_PASSES: usize = 20;

/// Full compositing pipeline for one band strip, exactly as the CLI
/// composite performs it: median across scenes, coverage-ordered
/// first-valid fill, then up to [`GAP_FILL_PASSES`] neighbor-mean passes.
/// An empty scene list yields an all-NaN strip.
pub fn composite_scene_strips(
    scene_strips: &[Array2<f64>],
    rows: usize,
    cols: usize,
) -> Array2<f64> {
    if scene_strips.is_empty() {
        return Array2::from_elem((rows, cols), f64::NAN);
    }
    let coverage = coverage_order(scene_strips);
    let mut out = median_composite(scene_strips, rows, cols);
    let nan_before = out.iter().filter(|v| !v.is_finite()).count();
    if nan_before > 0 {
        fill_from_scenes_by_coverage(&mut out, scene_strips, &coverage);
    }
    fill_gaps_neighbor_mean(&mut out, GAP_FILL_PASSES);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strip(rows: usize, cols: usize, vals: &[f64]) -> Array2<f64> {
        Array2::from_shape_vec((rows, cols), vals.to_vec()).unwrap()
    }

    #[test]
    fn median_odd_and_even_counts() {
        let a = strip(1, 2, &[1.0, 10.0]);
        let b = strip(1, 2, &[3.0, 20.0]);
        let c = strip(1, 2, &[2.0, f64::NAN]);
        let out = median_composite(&[a, b, c], 1, 2);
        // Pixel 0: {1,3,2} → median 2. Pixel 1: {10,20} → (10+20)/2.
        assert_eq!(out[[0, 0]], 2.0);
        assert_eq!(out[[0, 1]], 15.0);
    }

    #[test]
    fn median_ignores_nan_and_keeps_gap_when_all_nan() {
        let a = strip(1, 2, &[f64::NAN, 5.0]);
        let b = strip(1, 2, &[f64::NAN, f64::NAN]);
        let out = median_composite(&[a, b], 1, 2);
        assert!(!out[[0, 0]].is_finite(), "all-NaN pixel stays NaN");
        assert_eq!(out[[0, 1]], 5.0);
    }

    #[test]
    fn median_handles_smaller_scene_strips() {
        // One scene covers only the first row; the composite grid is 2×1.
        let small = strip(1, 1, &[7.0]);
        let out = median_composite(&[small], 2, 1);
        assert_eq!(out[[0, 0]], 7.0);
        assert!(!out[[1, 0]].is_finite());
    }

    #[test]
    fn coverage_fill_prefers_scene_with_most_valid_pixels() {
        let sparse = strip(1, 3, &[1.0, f64::NAN, f64::NAN]);
        let dense = strip(1, 3, &[2.0, 2.0, f64::NAN]);
        let strips = [sparse, dense];
        let coverage = coverage_order(&strips);
        assert_eq!(coverage[0].0, 1, "dense scene consulted first");

        let mut out = strip(1, 3, &[f64::NAN, f64::NAN, f64::NAN]);
        fill_from_scenes_by_coverage(&mut out, &strips, &coverage);
        assert_eq!(out[[0, 0]], 2.0, "dense scene wins pixel 0");
        assert_eq!(out[[0, 1]], 2.0);
        assert!(!out[[0, 2]].is_finite(), "no scene has pixel 2");
    }

    #[test]
    fn neighbor_fill_requires_two_finite_neighbors() {
        // A lone NaN surrounded by values fills; a NaN with one finite
        // neighbor does not.
        let mut ring = strip(3, 3, &[1.0, 1.0, 1.0, 1.0, f64::NAN, 1.0, 1.0, 1.0, 1.0]);
        let remaining = fill_gaps_neighbor_mean(&mut ring, 20);
        assert_eq!(remaining, 0);
        assert_eq!(ring[[1, 1]], 1.0);

        let mut lonely = strip(1, 3, &[5.0, f64::NAN, f64::NAN]);
        // pixel 1 has one finite neighbor (pixel 0) → cnt 1 < 2 → no fill.
        let remaining = fill_gaps_neighbor_mean(&mut lonely, 20);
        assert_eq!(remaining, 2);
        assert!(!lonely[[0, 1]].is_finite());
    }

    #[test]
    fn neighbor_fill_is_jacobi_not_gauss_seidel() {
        // 1×5 with values on both ends: within one pass, only pixels
        // adjacent to ≥2 finite SNAPSHOT neighbors fill. Ends: (0) 4.0,
        // (4) 8.0. Pixel 1 has neighbors {4.0, NaN} → 1 finite → no fill in
        // pass 1. Nothing fills in pass 1 → loop stops with 3 gaps.
        let mut line = strip(1, 5, &[4.0, f64::NAN, f64::NAN, f64::NAN, 8.0]);
        let remaining = fill_gaps_neighbor_mean(&mut line, 20);
        assert_eq!(
            remaining, 3,
            "no pixel has 2 finite neighbors in the snapshot"
        );
    }

    #[test]
    fn neighbor_fill_leaves_all_nan_untouched() {
        let mut empty = strip(2, 2, &[f64::NAN; 4]);
        let remaining = fill_gaps_neighbor_mean(&mut empty, 20);
        assert_eq!(remaining, 4);
    }

    #[test]
    fn full_pipeline_empty_scenes_is_all_nan() {
        let out = composite_scene_strips(&[], 2, 3);
        assert!(out.iter().all(|v| !v.is_finite()));
        assert_eq!(out.dim(), (2, 3));
    }

    #[test]
    fn full_pipeline_median_then_fills() {
        // 2×2: pixel (0,0) has values in both scenes (median), (0,1) only
        // in scene B (coverage fill), (1,0)/(1,1) in no scene but adjacent
        // to filled pixels (neighbor fill needs ≥2 finite neighbors).
        let a = strip(2, 2, &[10.0, f64::NAN, f64::NAN, f64::NAN]);
        let b = strip(2, 2, &[20.0, 30.0, f64::NAN, f64::NAN]);
        let out = composite_scene_strips(&[a, b], 2, 2);
        assert_eq!(out[[0, 0]], 15.0, "median of {{10, 20}}");
        assert_eq!(out[[0, 1]], 30.0, "coverage fill from scene B");
        // (1,0) sees finite {15, 30} in the snapshot → mean 22.5.
        assert_eq!(out[[1, 0]], 22.5);
        assert_eq!(out[[1, 1]], 22.5);
    }
}
