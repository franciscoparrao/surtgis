//! Swath profile — statistics envelope along a baseline corridor.
//!
//! Densifies a polyline baseline to a uniform spacing, casts a
//! perpendicular at every densified vertex, samples the raster at
//! cross-corridor steps within `±half_width_m`, and emits per-bin
//! min / max / mean / median / p25 / p75 / n_samples. This is the
//! standard TopoToolbox swath product used to summarise topographic
//! variation along a ridge, a fault trace, or a basin trunk — the
//! data shape behind the "envelope" plots that pair an elevation
//! profile with its spatial spread.
//!
//! ## Choice of baseline
//!
//! The baseline is **caller-supplied** as a polyline in the raster
//! CRS. This keeps the algorithm agnostic to where the baseline came
//! from — it could be:
//!   - The cell-centre coordinates of a [`super::LongProfile`]
//!     (`profile.nodes.iter().map(|n| n.coord).collect()`).
//!   - A fault trace digitised by the user.
//!   - A ridge axis from a hand-picked transect.
//!
//! Tying the algorithm to one specific source (e.g. only stream
//! networks) would limit it to the channel-analysis use case; the
//! general "sample a raster along a corridor" pattern is too useful
//! to lock down.
//!
//! ## Algorithm
//!
//! 1. **Densify** the baseline to `step_along_m` spacing — produces a
//!    uniform along-track sample grid.
//! 2. **Per densified vertex**: compute the local tangent (forward
//!    difference; reuses the previous tangent at the final point),
//!    rotate +90° to get the right-hand perpendicular, and sample the
//!    raster at `vertex + k · step_cross_m · perp` for `k` in
//!    `[-K, K]` where `K = floor(half_width_m / step_cross_m)`.
//! 3. **Filter NaN**, accumulate valid samples per vertex, compute
//!    stats. The bin's `distance_along_m` is `i · step_along_m` (the
//!    densification spacing is uniform by construction).
//!
//! Sample lookup uses **nearest-neighbour** raster access. Bilinear
//! interpolation is a polish item; for typical paper-figure resolutions
//! (DEM ≤ 30 m pixels, corridor width ≥ several pixels) NN matches the
//! visual resolution and is faster.

use surtgis_core::{Raster, Result};

/// Parameters for [`swath_profile`].
#[derive(Debug, Clone)]
pub struct SwathParams {
    /// Perpendicular reach in metres on each side of the baseline.
    /// Total corridor width is `2 * half_width_m`.
    pub half_width_m: f64,
    /// Along-track densification spacing in metres. The output has one
    /// bin per densified vertex.
    pub step_along_m: f64,
    /// Cross-track sample spacing in metres. Each bin collects
    /// `2 * floor(half_width_m / step_cross_m) + 1` samples (minus the
    /// ones that fall outside the raster or are NaN).
    pub step_cross_m: f64,
}

impl Default for SwathParams {
    fn default() -> Self {
        Self {
            half_width_m: 500.0,
            step_along_m: 30.0,
            step_cross_m: 30.0,
        }
    }
}

/// Per-bin statistics for the swath profile.
#[derive(Debug, Clone, Copy)]
pub struct SwathStats {
    /// Cumulative along-track distance from the baseline's first point, in metres.
    pub distance_along_m: f64,
    /// Minimum sampled value in the bin.
    pub min: f64,
    /// Maximum sampled value in the bin.
    pub max: f64,
    /// Mean of the sampled values in the bin.
    pub mean: f64,
    /// Median of the sampled values in the bin.
    pub median: f64,
    /// 25th percentile of the sampled values in the bin.
    pub p25: f64,
    /// 75th percentile of the sampled values in the bin.
    pub p75: f64,
    /// Count of valid (non-NaN, in-bounds) samples that contributed to the stats.
    pub n_samples: usize,
}

/// One full swath profile.
#[derive(Debug, Clone)]
pub struct SwathProfile {
    /// The densified baseline vertices that were used as sample anchors.
    /// `bins[i]` corresponds to `densified_baseline[i]`.
    pub densified_baseline: Vec<(f64, f64)>,
    /// Per-bin statistics, ordered from baseline start to end.
    pub bins: Vec<SwathStats>,
}

/// Errors specific to [`swath_profile`].
#[derive(Debug, thiserror::Error)]
pub enum SwathError {
    /// The baseline had fewer than two points.
    #[error("baseline must have at least 2 points (got {0})")]
    DegenerateBaseline(usize),
    /// `half_width_m` was not strictly positive.
    #[error("SwathParams.half_width_m must be > 0 (got {0})")]
    NonPositiveHalfWidth(f64),
    /// `step_along_m` was not strictly positive.
    #[error("SwathParams.step_along_m must be > 0 (got {0})")]
    NonPositiveStepAlong(f64),
    /// `step_cross_m` was not strictly positive.
    #[error("SwathParams.step_cross_m must be > 0 (got {0})")]
    NonPositiveStepCross(f64),
    /// All baseline vertices coincided, giving zero total length.
    #[error("baseline has zero total length (all vertices coincident)")]
    ZeroLengthBaseline,
}

/// Compute the swath profile of `raster` along `baseline`.
///
/// Returns a [`SwathProfile`] with one bin per densified vertex. Each
/// bin holds the cross-corridor statistics of the raster at that
/// along-track position. NaN / out-of-bounds samples are excluded from
/// the stats; a bin with zero valid samples gets all-NaN stats and
/// `n_samples == 0`.
pub fn swath_profile(
    raster: &Raster<f64>,
    baseline: &[(f64, f64)],
    params: SwathParams,
) -> Result<SwathProfile> {
    if baseline.len() < 2 {
        return Err(surtgis_core::Error::Other(
            SwathError::DegenerateBaseline(baseline.len()).to_string(),
        ));
    }
    if params.half_width_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            SwathError::NonPositiveHalfWidth(params.half_width_m).to_string(),
        ));
    }
    if params.step_along_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            SwathError::NonPositiveStepAlong(params.step_along_m).to_string(),
        ));
    }
    if params.step_cross_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            SwathError::NonPositiveStepCross(params.step_cross_m).to_string(),
        ));
    }

    // ── 1. Densify the baseline to step_along_m spacing. ────────────────
    let densified = densify(baseline, params.step_along_m)?;

    // ── 2. Per densified vertex, sample the perpendicular corridor. ─────
    let (rows, cols) = raster.shape();
    let k_max = (params.half_width_m / params.step_cross_m).floor() as i64;

    let mut bins = Vec::with_capacity(densified.len());
    let mut samples_scratch: Vec<f64> = Vec::with_capacity((2 * k_max + 1) as usize);

    for (i, &(x, y)) in densified.iter().enumerate() {
        // Local tangent: forward difference to next vertex; at the last
        // point, reuse the tangent from the previous step. With a
        // uniformly densified baseline this is exact (length per step is
        // exactly step_along_m).
        let (tx, ty) = if i + 1 < densified.len() {
            let (nx, ny) = densified[i + 1];
            (nx - x, ny - y)
        } else {
            let (px, py) = densified[i - 1];
            (x - px, y - py)
        };
        let tlen = (tx * tx + ty * ty).sqrt();
        let (tx, ty) = if tlen > 0.0 {
            (tx / tlen, ty / tlen)
        } else {
            // Should be unreachable after densify(); falls back to
            // sampling at the vertex itself.
            (1.0, 0.0)
        };
        // Right-hand perpendicular: (tx, ty) → (-ty, tx). The sign is
        // arbitrary as long as we walk symmetrically around the
        // baseline — distance bins are computed from the absolute
        // value of `k`.
        let (px, py) = (-ty, tx);

        samples_scratch.clear();
        for k in -k_max..=k_max {
            let offset = (k as f64) * params.step_cross_m;
            let sx = x + offset * px;
            let sy = y + offset * py;
            let (col_f, row_f) = raster.geo_to_pixel(sx, sy);
            // Nearest-neighbour with explicit bounds check (avoid
            // wrapping or silently sampling the edge).
            if !col_f.is_finite() || !row_f.is_finite() {
                continue;
            }
            let col = col_f.floor() as i64;
            let row = row_f.floor() as i64;
            if row < 0 || row >= rows as i64 || col < 0 || col >= cols as i64 {
                continue;
            }
            let v = raster.get(row as usize, col as usize).unwrap_or(f64::NAN);
            if v.is_finite() {
                samples_scratch.push(v);
            }
        }

        let stats = compute_stats(i as f64 * params.step_along_m, &mut samples_scratch);
        bins.push(stats);
    }

    Ok(SwathProfile {
        densified_baseline: densified,
        bins,
    })
}

/// Sample the baseline at uniform `step_m` spacing.
///
/// Walks each polyline segment, emitting points at every multiple of
/// `step_m` of cumulative arc length. Always includes the first
/// vertex; includes the last vertex only if it falls exactly on a
/// step. Returns `Err(ZeroLengthBaseline)` if the total length is zero.
fn densify(baseline: &[(f64, f64)], step_m: f64) -> Result<Vec<(f64, f64)>> {
    let total_len = baseline
        .windows(2)
        .map(|w| {
            let (dx, dy) = (w[1].0 - w[0].0, w[1].1 - w[0].1);
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>();
    if total_len <= 0.0 {
        return Err(surtgis_core::Error::Other(
            SwathError::ZeroLengthBaseline.to_string(),
        ));
    }

    let n_steps = (total_len / step_m).floor() as usize + 1;
    let mut out = Vec::with_capacity(n_steps);
    out.push(baseline[0]);

    let mut target = step_m;
    let mut accum = 0.0;
    for w in baseline.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        let dx = x1 - x0;
        let dy = y1 - y0;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len <= 0.0 {
            continue;
        }
        let seg_dirx = dx / seg_len;
        let seg_diry = dy / seg_len;
        // While the next sample target falls inside this segment.
        while target <= accum + seg_len {
            let t = target - accum;
            out.push((x0 + t * seg_dirx, y0 + t * seg_diry));
            target += step_m;
        }
        accum += seg_len;
    }

    Ok(out)
}

/// Order-statistics on the (mutable) sample buffer. The buffer is
/// sorted in place; callers don't depend on the post-call ordering.
fn compute_stats(distance_along_m: f64, samples: &mut [f64]) -> SwathStats {
    let n_samples = samples.len();
    if n_samples == 0 {
        return SwathStats {
            distance_along_m,
            min: f64::NAN,
            max: f64::NAN,
            mean: f64::NAN,
            median: f64::NAN,
            p25: f64::NAN,
            p75: f64::NAN,
            n_samples: 0,
        };
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let sum: f64 = samples.iter().sum();
    let pct = |p: f64| -> f64 {
        let idx = ((p * (n_samples as f64 - 1.0)).round() as usize).min(n_samples - 1);
        samples[idx]
    };
    SwathStats {
        distance_along_m,
        min: samples[0],
        max: samples[n_samples - 1],
        mean: sum / n_samples as f64,
        median: pct(0.50),
        p25: pct(0.25),
        p75: pct(0.75),
        n_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::raster::{GeoTransform, Raster};

    /// 100×100 raster with pixel size 1 (so geographic coordinates
    /// equal pixel coordinates, modulo the half-pixel offset for the
    /// upper-left origin convention used by `pixel_to_geo`).
    fn unit_raster<F: Fn(usize, usize) -> f64>(values: F) -> Raster<f64> {
        let rows = 100;
        let cols = 100;
        let mut data: Vec<f64> = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(values(r, c));
            }
        }
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut r = Raster::from_array(arr);
        // Origin (0, 0) at upper-left, pixel 1×1. `pixel_height` is
        // negative — y decreases going down rows, matching GeoTIFF
        // convention.
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r
    }

    #[test]
    fn densify_emits_uniform_spacing_along_a_straight_baseline() {
        let baseline = vec![(0.0, 0.0), (10.0, 0.0)];
        let dens = densify(&baseline, 1.0).unwrap();
        // Total length 10, step 1 → samples at 0, 1, …, 10 → 11 points.
        assert_eq!(dens.len(), 11);
        for (i, p) in dens.iter().enumerate() {
            assert!((p.0 - i as f64).abs() < 1e-9);
            assert!(p.1.abs() < 1e-9);
        }
    }

    #[test]
    fn densify_handles_corners_with_cumulative_arc_length() {
        // L-shape: (0,0) → (10,0) → (10,10). Total length 20.
        let baseline = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)];
        let dens = densify(&baseline, 5.0).unwrap();
        // Expected vertices at arc lengths 0, 5, 10, 15, 20.
        assert_eq!(dens.len(), 5);
        assert!((dens[0].0 - 0.0).abs() < 1e-9 && (dens[0].1 - 0.0).abs() < 1e-9);
        assert!((dens[1].0 - 5.0).abs() < 1e-9 && (dens[1].1 - 0.0).abs() < 1e-9);
        // At arc length 10 we are AT the corner.
        assert!((dens[2].0 - 10.0).abs() < 1e-9 && (dens[2].1 - 0.0).abs() < 1e-9);
        // Arc length 15 is 5 into the vertical leg.
        assert!((dens[3].0 - 10.0).abs() < 1e-9 && (dens[3].1 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn swath_on_uniform_raster_gives_constant_stats() {
        // Flat raster of 42.0 everywhere → every bin should report
        // min = max = mean = median = 42.
        let r = unit_raster(|_, _| 42.0);
        let baseline = vec![(2.5, -2.5), (97.5, -2.5)]; // y is negative because pixel_height is -1
        let params = SwathParams {
            half_width_m: 5.0,
            step_along_m: 1.0,
            step_cross_m: 1.0,
        };
        let sw = swath_profile(&r, &baseline, params).unwrap();
        for s in &sw.bins {
            assert!(s.n_samples > 0, "expected samples at every bin");
            assert!((s.min - 42.0).abs() < 1e-9);
            assert!((s.max - 42.0).abs() < 1e-9);
            assert!((s.median - 42.0).abs() < 1e-9);
        }
    }

    #[test]
    fn swath_picks_up_cross_track_variation() {
        // Raster: elevation = column index (0..99). Baseline runs
        // east at y=10 from x=20 to x=80. Perpendicular is the
        // y-axis; sampling at ±5 cells perpendicular means each bin
        // sees ONLY values at the same column as the baseline. Stats
        // should reflect zero cross-track spread (min == max).
        let r = unit_raster(|_, c| c as f64);
        let baseline = vec![(20.5, -10.5), (80.5, -10.5)];
        let params = SwathParams {
            half_width_m: 5.0,
            step_along_m: 1.0,
            step_cross_m: 1.0,
        };
        let sw = swath_profile(&r, &baseline, params).unwrap();
        for (i, s) in sw.bins.iter().enumerate() {
            // The baseline x at bin i is 20.5 + i (densified
            // uniformly).
            let expected_col = (20.5 + i as f64).floor();
            assert!(
                (s.min - expected_col).abs() < 1e-9,
                "bin {i}: min = {}, expected {}",
                s.min,
                expected_col
            );
            assert!(
                (s.max - expected_col).abs() < 1e-9,
                "bin {i}: max = {}, expected {}",
                s.max,
                expected_col
            );
        }
    }

    #[test]
    fn swath_perpendicular_to_gradient_gives_real_spread() {
        // Elevation = column index. Baseline runs NORTH (perpendicular
        // to the gradient). Sampling at ±5 cells perpendicular sweeps
        // 11 column values centred on the baseline x. Each bin should
        // report a spread of (max - min) == 10 across the column range.
        let r = unit_raster(|_, c| c as f64);
        let baseline = vec![(50.5, -10.5), (50.5, -80.5)];
        let params = SwathParams {
            half_width_m: 5.0,
            step_along_m: 1.0,
            step_cross_m: 1.0,
        };
        let sw = swath_profile(&r, &baseline, params).unwrap();
        for s in &sw.bins {
            assert!(s.n_samples >= 10);
            assert!(
                (s.max - s.min - 10.0).abs() < 1e-9,
                "expected max-min == 10, got {} ({} - {})",
                s.max - s.min,
                s.max,
                s.min
            );
            // Median should sit at the baseline column ≈ 50.
            assert!(
                (s.median - 50.0).abs() < 1.0,
                "expected median ≈ 50, got {}",
                s.median
            );
        }
    }

    #[test]
    fn swath_validates_input() {
        let r = unit_raster(|_, _| 0.0);
        let baseline_short = vec![(0.0, 0.0)];
        assert!(swath_profile(&r, &baseline_short, SwathParams::default()).is_err());

        let baseline_ok = vec![(0.0, 0.0), (10.0, 0.0)];
        let bad = SwathParams {
            half_width_m: 0.0,
            ..Default::default()
        };
        assert!(swath_profile(&r, &baseline_ok, bad).is_err());
    }
}
