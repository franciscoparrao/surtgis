//! Normalised channel steepness index (`ksn`) per Wobus et al. (2006).
//!
//! `ksn` is a per-cell proxy for the ratio U/K (uplift rate divided by
//! erodibility): zones of high `ksn` are either uplifting fast, or
//! resistant to erosion, or both. It is the workhorse metric for
//! geomorphology-from-topography studies and the standard product paired
//! with χ for tectonic interpretation.
//!
//! ## Reference
//!
//! Wobus, C., Whipple, K.X., Kirby, E., Snyder, N., Johnson, J.,
//! Spyropolou, K., Crosby, B. & Sheehan, D. (2006). "Tectonics from
//! topography." GSA Special Paper 398, 55–74.
//! <https://doi.org/10.1130/2006.2398(04)>
//!
//! ## Algorithm (spec §4.2)
//!
//! Three stages:
//!
//! 1. **Per-cell channel slope.** For each stream cell `c` with a
//!    downstream link `d`, S(c) = (z(c) − z(d)) / Δx where Δx is the
//!    centre-to-centre distance along the channel (= cell_size for a
//!    cardinal D8 step, = cell_size·√2 for a diagonal step). Outlet
//!    cells (no downstream) get NaN.
//!
//! 2. **Raw ksn per cell.** ksn_raw(c) = S(c) · A(c)^θref where A(c) is
//!    drainage area in m².
//!
//! 3. **Smoothing along the channel.** For each cell, mean `ksn_raw`
//!    over the cells lying within ±`segment_length_m`/2 of `c` measured
//!    along the network (NOT 2-D distance). At confluences during the
//!    upstream walk we follow the largest-area tributary (the "main
//!    stem"), so the window is always a single linear chain. Without
//!    smoothing the raster is dominated by per-pixel slope noise.
//!
//! ## Implementation notes
//!
//! - **Drainage-area threshold.** Cells whose A is below
//!   [`KsnParams::min_drainage_area_m2`] (default 1 km² = 1·10⁶ m²) are
//!   skipped — spec §8 pitfall #3: short tributaries give unstable `ksn`.
//! - **Channel slope is NOT the 2-D terrain slope.** Geomorphology cares
//!   about along-channel elevation change per metre of flow. The two
//!   coincide only at boundaries; in general they diverge by a factor of
//!   `cos(aspect)`. Using 2-D slope here is a textbook beginner mistake.
//! - **NaN propagation.** Missing DEM, missing accumulation, or
//!   sub-threshold area at any cell in the smoothing window contributes
//!   NaN, which excludes the cell from the mean but does not invalidate
//!   the cell being smoothed; the centre cell still gets a value if any
//!   window neighbour is valid.

use std::collections::VecDeque;

use ndarray::Array2;
use surtgis_core::{Raster, Result};

use super::stream_traversal::{StreamGraph, StreamGraphError, build_stream_graph};

/// Parameters for [`channel_steepness`].
#[derive(Debug, Clone)]
pub struct KsnParams {
    /// Reference concavity. Default 0.45 (bedrock channels, steady state).
    pub theta_ref: f64,
    /// Smoothing window length in metres, measured along the channel.
    /// Default 500 m — the literature standard (Wobus 2006, §3).
    pub segment_length_m: f64,
    /// Pixel size in metres. Caller-supplied because the function does
    /// not introspect the raster transform.
    pub cell_size_m: f64,
    /// Minimum drainage area in m² for a cell to contribute to ksn.
    /// Default 1 km². Cells below this give numerically unstable values.
    pub min_drainage_area_m2: f64,
    /// Number of bootstrap iterations for the per-segment 95 % CI.
    /// Default 200 (matches `concavity_index`). Set to 0 to disable;
    /// the CI is then `(ksn_mean, ksn_mean)`.
    pub bootstrap_n: usize,
    /// Reproducibility seed for the bootstrap resampler. Default 42.
    pub seed: u64,
}

impl Default for KsnParams {
    fn default() -> Self {
        Self {
            theta_ref: 0.45,
            segment_length_m: 500.0,
            cell_size_m: 30.0,
            min_drainage_area_m2: 1.0e6,
            bootstrap_n: 200,
            seed: 42,
        }
    }
}

/// One continuous stream segment between two graph terminals (outlet or
/// confluence). Coordinates are in the source raster's CRS.
#[derive(Debug, Clone)]
pub struct KsnSegment {
    /// Ordered list of `(x, y)` cell-centre coordinates, downstream → upstream.
    pub coordinates: Vec<(f64, f64)>,
    /// Mean `ksn` over the segment cells (NaNs excluded).
    pub ksn_mean: f64,
    /// 95 % CI on `ksn_mean` from a deterministic bootstrap over the
    /// segment's per-cell `ksn` values. When `bootstrap_n == 0` in
    /// the params, this equals `(ksn_mean, ksn_mean)`. Standard
    /// percentile bootstrap; matches the pattern in
    /// `concavity_index`.
    pub ksn_ci: (f64, f64),
    /// Number of cells contributing to the mean.
    pub n_cells: usize,
}

/// Combined raster + optional vector segments output.
#[derive(Debug)]
pub struct KsnResult {
    /// Per-cell `ksn` raster.
    pub ksn_raster: Raster<f64>,
    /// Vector segments with aggregated `ksn`, when segment output was requested.
    pub segments: Option<Vec<KsnSegment>>,
}

/// Errors specific to ksn computation.
#[derive(Debug, thiserror::Error)]
pub enum KsnError {
    /// Building the stream graph failed.
    #[error(transparent)]
    Graph(#[from] StreamGraphError),

    /// Two input rasters had incompatible shapes.
    #[error("input raster shapes disagree: {0:?} vs {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),

    /// `cell_size_m` was not strictly positive.
    #[error("KsnParams.cell_size_m must be > 0 (got {0})")]
    NonPositiveCellSize(f64),

    /// `segment_length_m` was not strictly positive.
    #[error("KsnParams.segment_length_m must be > 0 (got {0})")]
    NonPositiveSegmentLength(f64),
}

/// Compute normalised channel steepness `ksn`.
///
/// Returns a [`KsnResult`] holding the per-cell ksn raster (NaN at
/// non-stream cells, outlets, and cells below the area threshold).
/// When `emit_segments` is true, `segments` carries the vector
/// representation: one [`KsnSegment`] per maximal single-channel run
/// between graph terminals.
#[allow(clippy::too_many_arguments)]
pub fn channel_steepness(
    stream: &Raster<u8>,
    flow_dir: &Raster<u8>,
    flow_acc: &Raster<f64>,
    dem: &Raster<f64>,
    params: KsnParams,
    emit_segments: bool,
) -> Result<KsnResult> {
    if params.cell_size_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            KsnError::NonPositiveCellSize(params.cell_size_m).to_string(),
        ));
    }
    if params.segment_length_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            KsnError::NonPositiveSegmentLength(params.segment_length_m).to_string(),
        ));
    }
    let s_shape = stream.shape();
    for (other, label) in [
        (flow_dir.shape(), "flow_dir"),
        (flow_acc.shape(), "flow_acc"),
        (dem.shape(), "dem"),
    ] {
        if other != s_shape {
            let _ = label;
            return Err(surtgis_core::Error::Other(
                KsnError::ShapeMismatch(other, s_shape).to_string(),
            ));
        }
    }
    let (rows, cols) = s_shape;

    let graph = build_stream_graph(stream, flow_dir)
        .map_err(|e| surtgis_core::Error::Other(e.to_string()))?;

    let cell = params.cell_size_m;
    let cell_diag = cell * std::f64::consts::SQRT_2;
    let cell_area_m2 = cell * cell;
    let half_window_m = params.segment_length_m * 0.5;

    // ── Stage 1+2: per-cell channel slope and raw ksn. ─────────────────
    //
    // Each stream cell contributes one slope (and therefore one raw ksn)
    // computed from its downstream neighbour. Outlet nodes get NaN — they
    // have no downstream cell against which to take an elevation
    // difference.
    let n = graph.len();
    let mut ksn_raw = vec![f64::NAN; n];
    for i in 0..n {
        let Some(d) = graph.downstream_link[i] else {
            continue; // outlet → NaN
        };
        let (ir, ic) = graph.stream_cells[i];
        let (dr_idx, dc_idx) = graph.stream_cells[d];
        let z_i = dem.get(ir, ic).unwrap_or(f64::NAN);
        let z_d = dem.get(dr_idx, dc_idx).unwrap_or(f64::NAN);
        if !z_i.is_finite() || !z_d.is_finite() {
            continue;
        }
        let dr = (dr_idx as isize - ir as isize).abs();
        let dc = (dc_idx as isize - ic as isize).abs();
        let dx = if dr + dc == 2 { cell_diag } else { cell };
        // Channel slope: positive when z drops downstream (the normal
        // case). Clamp negative slopes (DEM artefacts) to 0 — Wobus 2006
        // also documents this.
        let s_local = ((z_i - z_d) / dx).max(0.0);

        // flow_acc counts *upstream* cells only (a headwater has acc=0);
        // the cell's own contributing area must be included, so the
        // drainage area is (acc + 1) cells, not acc cells.
        let a_cells = flow_acc.get(ir, ic).unwrap_or(f64::NAN);
        let a_m2 = (a_cells + 1.0) * cell_area_m2;
        if !a_m2.is_finite() || a_m2 < params.min_drainage_area_m2 {
            continue;
        }
        ksn_raw[i] = s_local * a_m2.powf(params.theta_ref);
    }

    // ── Stage 3: smoothing along the channel. ──────────────────────────
    //
    // For each cell, walk upstream (main-stem only) and downstream
    // collecting cells whose channel-following distance from the centre
    // is ≤ half_window_m. The smoothed ksn is the mean of the raw ksn
    // values in that window.
    let mut ksn_smoothed = vec![f64::NAN; n];
    for i in 0..n {
        // Collect window contributors and aggregate the mean inline.
        let mut sum = 0.0_f64;
        let mut count = 0_usize;

        // Centre cell itself (no distance cost).
        if ksn_raw[i].is_finite() {
            sum += ksn_raw[i];
            count += 1;
        }

        // Walk downstream from i.
        let mut cur = i;
        let (mut cur_r, mut cur_c) = graph.stream_cells[cur];
        let mut acc_dist = 0.0_f64;
        while let Some(d) = graph.downstream_link[cur] {
            let (dr_idx, dc_idx) = graph.stream_cells[d];
            let step = step_distance(cur_r, cur_c, dr_idx, dc_idx, cell, cell_diag);
            acc_dist += step;
            if acc_dist > half_window_m {
                break;
            }
            if ksn_raw[d].is_finite() {
                sum += ksn_raw[d];
                count += 1;
            }
            cur = d;
            cur_r = dr_idx;
            cur_c = dc_idx;
        }

        // Walk upstream from i. At confluences pick the largest-area
        // tributary (main stem). Headwaters stop the walk.
        let (mut cur_r, mut cur_c) = graph.stream_cells[i];
        let mut cur = i;
        let mut acc_dist = 0.0_f64;
        loop {
            let ups = &graph.upstream_links[cur];
            if ups.is_empty() {
                break; // headwater
            }
            let next = if ups.len() == 1 {
                ups[0]
            } else {
                pick_largest_area(ups, &graph, flow_acc)
            };
            let (ur, uc) = graph.stream_cells[next];
            let step = step_distance(cur_r, cur_c, ur, uc, cell, cell_diag);
            acc_dist += step;
            if acc_dist > half_window_m {
                break;
            }
            if ksn_raw[next].is_finite() {
                sum += ksn_raw[next];
                count += 1;
            }
            cur = next;
            cur_r = ur;
            cur_c = uc;
        }

        if count > 0 {
            ksn_smoothed[i] = sum / count as f64;
        }
    }

    // ── Materialise as a raster preserving stream's geo metadata. ─────
    let mut data = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    for (i, &(r, c)) in graph.stream_cells.iter().enumerate() {
        data[(r, c)] = ksn_smoothed[i];
    }
    let mut ksn_raster = stream.with_same_meta::<f64>(rows, cols);
    ksn_raster.set_nodata(Some(f64::NAN));
    *ksn_raster.data_mut() = data;

    // ── Optional vector segments. ─────────────────────────────────────
    let segments = if emit_segments {
        Some(extract_segments(
            &graph,
            &ksn_smoothed,
            stream,
            flow_acc,
            params.bootstrap_n,
            params.seed,
        ))
    } else {
        None
    };

    Ok(KsnResult {
        ksn_raster,
        segments,
    })
}

fn step_distance(r0: usize, c0: usize, r1: usize, c1: usize, cardinal: f64, diagonal: f64) -> f64 {
    let dr = (r1 as isize - r0 as isize).abs();
    let dc = (c1 as isize - c0 as isize).abs();
    if dr + dc == 2 { diagonal } else { cardinal }
}

fn pick_largest_area(candidates: &[usize], graph: &StreamGraph, flow_acc: &Raster<f64>) -> usize {
    let mut best = candidates[0];
    let mut best_a = -1.0_f64;
    for &c in candidates {
        let (r, col) = graph.stream_cells[c];
        let a = flow_acc.get(r, col).unwrap_or(f64::NEG_INFINITY);
        if a.is_finite() && a > best_a {
            best_a = a;
            best = c;
        }
    }
    best
}

/// Walk the stream graph and extract one segment per maximal single-channel
/// run between two graph terminals (outlets and confluences). Coordinates
/// are computed from the raster transform via `pixel_to_geo` and ordered
/// downstream → upstream.
///
/// Terminal definition:
///   - outlet  = `downstream_link.is_none()`
///   - confluence (downstream end) = `upstream_links.len() >= 2`
///   - confluence (upstream end)   = node whose downstream is a confluence
///
/// For each downstream terminal that's an outlet OR confluence, walk
/// upstream along the single-stem path until the next confluence or
/// headwater. Each such walk is one [`KsnSegment`].
fn extract_segments(
    graph: &StreamGraph,
    ksn: &[f64],
    stream: &Raster<u8>,
    flow_acc: &Raster<f64>,
    bootstrap_n: usize,
    seed: u64,
) -> Vec<KsnSegment> {
    use std::hash::{Hash, Hasher};

    let mut segments = Vec::new();
    let mut starts: VecDeque<usize> = VecDeque::new();

    // Each segment originates either from an outlet OR from an upstream
    // tributary of a confluence. Iterating outlets first yields the
    // downstream-most segment of each subnetwork; iterating each
    // confluence's tributaries yields the rest.
    for i in 0..graph.len() {
        if graph.is_outlet[i] {
            starts.push_back(i);
        } else if graph.upstream_links[i].len() >= 2 {
            // Each tributary at this confluence will start a new segment.
            for &trib in &graph.upstream_links[i] {
                starts.push_back(trib);
            }
        }
    }

    for (segment_id, start) in starts.into_iter().enumerate() {
        // Walk upstream from start, single-stem, until the next confluence
        // (multiple upstream links) or headwater (no upstream).
        let mut cells: Vec<usize> = vec![start];
        let mut cur = start;
        loop {
            let ups = &graph.upstream_links[cur];
            if ups.is_empty() {
                break;
            }
            if ups.len() >= 2 {
                // Reached a confluence — this segment ends here. The
                // confluence cell itself is NOT included; the upstream
                // tributaries are separate segments seeded above.
                break;
            }
            let next = ups[0];
            // Defensive: avoid extending into another segment via main-stem
            // jumps; if next has >=2 ups, the iteration above handles it.
            cells.push(next);
            cur = next;
        }

        // Convert cell indices → (x, y) raster-CRS coordinates AND
        // collect the per-cell finite ksn values used for bootstrap.
        let mut coords: Vec<(f64, f64)> = Vec::with_capacity(cells.len());
        let mut values: Vec<f64> = Vec::with_capacity(cells.len());
        for &i in &cells {
            let (r, c) = graph.stream_cells[i];
            // Cell centre — `pixel_to_geo` already returns the pixel
            // centre (col + 0.5, row + 0.5 internally), so no further
            // offset is applied here.
            let (x, y) = stream.pixel_to_geo(c, r);
            coords.push((x, y));
            if ksn[i].is_finite() {
                values.push(ksn[i]);
            }
        }
        let count = values.len();
        let ksn_mean = if count > 0 {
            values.iter().sum::<f64>() / count as f64
        } else {
            f64::NAN
        };

        // Deterministic bootstrap on the per-cell ksn values. Hash
        // (seed, segment_id, boot_k, cell_k) for the resampling
        // index — same recipe as `concavity_index` so results are
        // reproducible across the suite.
        let ksn_ci = if bootstrap_n == 0 || count == 0 {
            (ksn_mean, ksn_mean)
        } else {
            let n = count;
            let mut samples: Vec<f64> = Vec::with_capacity(bootstrap_n);
            for boot in 0..bootstrap_n {
                let mut sum_b = 0.0;
                for k in 0..n {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut h);
                    segment_id.hash(&mut h);
                    boot.hash(&mut h);
                    k.hash(&mut h);
                    let idx = (h.finish() as usize) % n;
                    sum_b += values[idx];
                }
                samples.push(sum_b / n as f64);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let pct = |p: f64| -> f64 {
                let idx =
                    ((p * (samples.len() as f64 - 1.0)).round() as usize).min(samples.len() - 1);
                samples[idx]
            };
            (pct(0.025), pct(0.975))
        };

        let _ = flow_acc; // currently unused; reserved for future per-segment area-weighted mean
        segments.push(KsnSegment {
            coordinates: coords,
            ksn_mean,
            ksn_ci,
            n_cells: count,
        });
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Spec §7.2 headline test for ksn: two channels with the same
    /// drainage area but different slopes must give ksn_ratio ≈
    /// slope_ratio. We build two parallel 6-cell channels in the same
    /// raster, each with constant flow_acc = 5000 (upstream cell count;
    /// physical area = (flow_acc + 1) * cell² ≈ 4.5 km² for cell=30 m,
    /// well above the 1 km² threshold). Slope is controlled by the DEM:
    /// channel A drops 1 m per cell, channel B drops 3 m per cell. With
    /// smoothing window 60 m and cell 30 m, each cell averages itself +
    /// 1 upstream + 1 downstream.
    #[test]
    fn equal_area_different_slope_gives_ratio_ksn_equal_ratio_slope() {
        let cell = 30.0;
        let n = 6;
        // Two horizontal channels at rows 0 and 2, both flowing east.
        let stream = raster_u8(vec![
            vec![1; n], // channel A
            vec![0; n],
            vec![1; n], // channel B
        ]);
        // flow_dir: east everywhere on the channels; non-channel cells 0.
        let flow_dir = raster_u8(vec![vec![1; n], vec![0; n], vec![1; n]]);
        // Both channels: constant flow_acc = 5000 (upstream cell count).
        let flow_acc = raster_f64(vec![vec![5000.0; n], vec![0.0; n], vec![5000.0; n]]);
        // DEM: channel A drops 1 m/cell, channel B drops 3 m/cell.
        let dem_a: Vec<f64> = (0..n).map(|c| (n - 1 - c) as f64 * 1.0).collect();
        let dem_b: Vec<f64> = (0..n).map(|c| (n - 1 - c) as f64 * 3.0).collect();
        let dem = raster_f64(vec![dem_a, vec![0.0; n], dem_b]);

        let params = KsnParams {
            theta_ref: 0.45,
            segment_length_m: 60.0, // window = 60 m = 2 cells radius
            cell_size_m: cell,
            min_drainage_area_m2: 1.0e6,
            bootstrap_n: 0,
            seed: 42,
        };
        let result = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, false).unwrap();

        // Pick an interior cell in each channel (col 2 of 6 — has full
        // window on both sides without hitting outlet at col 5).
        let ksn_a = result.ksn_raster.get(0, 2).unwrap();
        let ksn_b = result.ksn_raster.get(2, 2).unwrap();
        assert!(
            ksn_a.is_finite() && ksn_b.is_finite(),
            "ksn must be finite on interior cells; got A={}, B={}",
            ksn_a,
            ksn_b,
        );

        let ratio = ksn_b / ksn_a;
        // Slopes differ by factor 3, areas equal → ksn ratio must be 3.
        assert!(
            (ratio - 3.0).abs() < 0.05,
            "ksn_B / ksn_A should ≈ slope_B / slope_A = 3.0; got {}",
            ratio,
        );
    }

    /// Outlet cells get NaN ksn because there is no downstream cell
    /// against which to take an elevation difference.
    #[test]
    fn outlet_cell_yields_nan_when_no_window_neighbours_valid() {
        // Single 2-cell channel; rightmost cell is the outlet.
        let stream = raster_u8(vec![vec![1, 1]]);
        let flow_dir = raster_u8(vec![vec![1, 1]]); // both east; right cell exits raster
        let flow_acc = raster_f64(vec![vec![5000.0, 5000.0]]);
        let dem = raster_f64(vec![vec![10.0, 5.0]]);

        // Window 30 m (= 1 cell) — outlet's window is just the outlet
        // itself, whose raw ksn is NaN → smoothed must also be NaN.
        let params = KsnParams {
            theta_ref: 0.45,
            segment_length_m: 1.0, // half-window 0.5 m: no neighbours reach in
            cell_size_m: 30.0,
            min_drainage_area_m2: 1.0e6,
            bootstrap_n: 0,
            seed: 42,
        };
        let result = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, false).unwrap();

        let outlet = result.ksn_raster.get(0, 1).unwrap();
        assert!(outlet.is_nan(), "outlet ksn should be NaN, got {}", outlet);
    }

    /// Drainage-area threshold: cells below `min_drainage_area_m2` must
    /// not contribute their own raw ksn to the smoothed mean. Build a
    /// 5-cell channel where the headwater two cells fall below the
    /// threshold (small A) and verify those cells' smoothed ksn drops to
    /// NaN when no in-window neighbour qualifies.
    #[test]
    fn below_threshold_cells_excluded_from_window() {
        let cell = 30.0;
        let cell_area = cell * cell;
        let threshold_cells = 5.0; // 5 cells → ~4500 m²
        let stream = raster_u8(vec![vec![1, 1, 1, 1, 1]]);
        let flow_dir = raster_u8(vec![vec![1, 1, 1, 1, 1]]);
        // flow_acc (upstream cell count, per flow_accumulation()'s
        // headwater=0 convention) increases west→east: 0, 1, 2, 3, 4.
        // Physical area = (flow_acc + 1) cells = 1, 2, 3, 4, 5. With a
        // threshold of 5 cells, only the rightmost cell qualifies.
        let flow_acc = raster_f64(vec![vec![0.0, 1.0, 2.0, 3.0, 4.0]]);
        let dem = raster_f64(vec![vec![10.0, 8.0, 6.0, 4.0, 2.0]]);

        let params = KsnParams {
            theta_ref: 0.45,
            segment_length_m: 1.0,
            cell_size_m: cell,
            min_drainage_area_m2: threshold_cells * cell_area,
            bootstrap_n: 0,
            seed: 42,
        };
        let result = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, false).unwrap();

        // Far-left cells: below threshold + window does not reach the
        // qualifying rightmost cell → smoothed NaN.
        for c in 0..3 {
            let v = result.ksn_raster.get(0, c).unwrap();
            assert!(
                v.is_nan(),
                "cell {} should be NaN (below threshold), got {}",
                c,
                v
            );
        }
    }

    /// emit_segments=true yields exactly one segment for a simple Y
    /// (two tributaries + one main stem to outlet = 3 segments, since the
    /// confluence triggers two tributary starts plus the outlet start
    /// that walks up to the confluence).
    #[test]
    fn y_confluence_produces_three_segments() {
        let stream = raster_u8(vec![
            vec![1, 1, 1, 0, 0],
            vec![0, 0, 1, 1, 1],
            vec![1, 1, 1, 0, 0],
        ]);
        let flow_dir = raster_u8(vec![
            vec![1, 1, 7, 0, 0],
            vec![0, 0, 1, 1, 1],
            vec![1, 1, 3, 0, 0],
        ]);
        let flow_acc = raster_f64(vec![
            vec![5000.0, 5000.0, 5000.0, 0.0, 0.0],
            vec![0.0, 0.0, 10000.0, 10000.0, 10000.0],
            vec![5000.0, 5000.0, 5000.0, 0.0, 0.0],
        ]);
        let dem = raster_f64(vec![
            vec![20.0, 18.0, 16.0, 0.0, 0.0],
            vec![0.0, 0.0, 14.0, 12.0, 10.0],
            vec![20.0, 18.0, 16.0, 0.0, 0.0],
        ]);

        let params = KsnParams {
            theta_ref: 0.45,
            segment_length_m: 30.0,
            cell_size_m: 30.0,
            min_drainage_area_m2: 1.0e6,
            bootstrap_n: 0,
            seed: 42,
        };
        let result = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, true).unwrap();
        let segs = result.segments.expect("emit_segments=true");
        // 1 outlet-rooted segment (main stem) + 2 confluence-tributary
        // segments = 3 total.
        assert_eq!(
            segs.len(),
            3,
            "expected 3 segments for a Y network, got {}",
            segs.len()
        );
    }

    /// Bootstrap CI brackets the segment mean with the configured
    /// percentiles, and is reproducible across runs with the same seed.
    #[test]
    fn ksn_bootstrap_ci_brackets_the_mean_and_is_seed_stable() {
        // Single straight 8-cell channel, varying slope so the per-cell
        // ksn values have real spread for the bootstrap to chew on.
        let n = 8;
        let cell = 30.0;
        let a_cells = 5000.0;
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        let flow_acc = raster_f64(vec![vec![a_cells; n]]);
        // Heterogeneous elevation drops: alternating 1 m and 3 m steps
        // produce ksn values with non-trivial variance.
        let drops = [3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0];
        let mut elev = vec![0.0; n];
        for (i, d) in drops.iter().enumerate() {
            elev[n - 2 - i] = elev[n - 1 - i] + d;
        }
        let dem = raster_f64(vec![elev]);

        let mut params = KsnParams {
            theta_ref: 0.45,
            segment_length_m: 1.0,
            cell_size_m: cell,
            min_drainage_area_m2: 1.0e6,
            bootstrap_n: 200,
            seed: 42,
        };

        let r1 =
            channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params.clone(), true).unwrap();
        let r2 =
            channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params.clone(), true).unwrap();
        let s1 = r1.segments.unwrap();
        let s2 = r2.segments.unwrap();
        assert_eq!(s1.len(), s2.len());
        for (a, b) in s1.iter().zip(s2.iter()) {
            assert_eq!(a.ksn_ci, b.ksn_ci, "same seed must give identical CI");
            assert!(
                a.ksn_ci.0 <= a.ksn_mean + 1e-9 && a.ksn_mean <= a.ksn_ci.1 + 1e-9,
                "CI must bracket the mean: ksn_mean={}, ci=({}, {})",
                a.ksn_mean,
                a.ksn_ci.0,
                a.ksn_ci.1
            );
        }

        // Bumping the seed must change at least one CI (the resampled
        // indices differ).
        params.seed = 1729;
        let r3 = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, true).unwrap();
        let s3 = r3.segments.unwrap();
        let any_changed = s1.iter().zip(s3.iter()).any(|(a, b)| {
            (a.ksn_ci.0 - b.ksn_ci.0).abs() > 1e-12 || (a.ksn_ci.1 - b.ksn_ci.1).abs() > 1e-12
        });
        assert!(
            any_changed,
            "different seeds should change at least one CI; got identical CIs"
        );
    }

    /// `bootstrap_n = 0` disables the CI; it must equal `(mean, mean)`.
    #[test]
    fn ksn_bootstrap_disabled_means_ci_equals_mean() {
        let n = 6;
        let cell = 30.0;
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        let flow_acc = raster_f64(vec![vec![5000.0; n]]);
        let dem = raster_f64(vec![vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]]);
        let params = KsnParams {
            theta_ref: 0.45,
            segment_length_m: 1.0,
            cell_size_m: cell,
            min_drainage_area_m2: 1.0e6,
            bootstrap_n: 0,
            seed: 42,
        };
        let r = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, true).unwrap();
        let segs = r.segments.unwrap();
        for s in &segs {
            if s.ksn_mean.is_finite() {
                assert_eq!(s.ksn_ci.0, s.ksn_mean);
                assert_eq!(s.ksn_ci.1, s.ksn_mean);
            }
        }
    }

    /// Regression test for the ½-pixel offset bug (CR-1): segment
    /// coordinates must equal `pixel_to_geo` exactly, with no extra
    /// half-cell added on top. `pixel_to_geo` already returns the pixel
    /// *centre* (col + 0.5, row + 0.5 internally).
    #[test]
    fn segment_coord_matches_pixel_to_geo_exactly_no_extra_half_pixel() {
        use surtgis_core::GeoTransform;

        let n = 2;
        let mut stream = raster_u8(vec![vec![1u8; n]]);
        stream.set_transform(GeoTransform::new(0.0, 0.0, 10.0, -10.0));
        let mut flow_dir = raster_u8(vec![vec![1u8, 0]]); // col0 → east (outlet at col1)
        flow_dir.set_transform(GeoTransform::new(0.0, 0.0, 10.0, -10.0));
        let mut flow_acc = raster_f64(vec![vec![5000.0; n]]);
        flow_acc.set_transform(GeoTransform::new(0.0, 0.0, 10.0, -10.0));
        let mut dem = raster_f64(vec![vec![5.0, 0.0]]);
        dem.set_transform(GeoTransform::new(0.0, 0.0, 10.0, -10.0));

        let params = KsnParams {
            theta_ref: 0.45,
            segment_length_m: 1.0,
            cell_size_m: 10.0,
            min_drainage_area_m2: 1.0e6,
            bootstrap_n: 0,
            seed: 42,
        };
        let result = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, true).unwrap();
        let segs = result.segments.expect("emit_segments=true");
        let first_coord = segs[0].coordinates[0];

        // Known coordinate: with origin (0,0) and pixel_size 10,
        // pixel_to_geo(0, 0) must be (5.0, -5.0), not (10.0, -10.0).
        let expected_00 = stream.pixel_to_geo(0, 0);
        assert_eq!(expected_00, (5.0, -5.0));

        // The outlet segment starts at the outlet cell (0, 1).
        let expected = stream.pixel_to_geo(1, 0);
        assert_eq!(
            first_coord, expected,
            "segment coordinate must match pixel_to_geo exactly"
        );
    }
}
