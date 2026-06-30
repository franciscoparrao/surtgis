//! Knickpoint detection on river long profiles.
//!
//! A knickpoint is a sharp break in the slope of a longitudinal river
//! profile. Knickpoints can be **transient** — wave-form responses to a
//! tectonic perturbation propagating upstream — or **stationary** —
//! anchored to a lithologic contrast. Detecting and classifying them is
//! one of the core operations in tectonic geomorphology.
//!
//! ## Method (spec §4.3)
//!
//! We follow the reproducible recipe of Neely, DiBiase et al. (2017):
//!
//! 1. Build the `(χ, z)` profile for every continuous stream segment
//!    between two graph terminals (outlet, confluence, or headwater).
//! 2. Total-Variation Denoise the elevation series along the profile
//!    using Condat's (2013) exact O(N) direct algorithm.
//! 3. Compute the discrete second derivative `d²z/dχ²` over the denoised
//!    profile using a non-uniform-grid 3-point finite-difference stencil.
//! 4. Flag a cell as a knickpoint candidate when `|d²z/dχ²|` exceeds the
//!    user threshold AND the elevation magnitude across a small window
//!    around the cell exceeds `min_magnitude_m`.
//! 5. Classify polarity:
//!    - `d²z/dχ² > 0` → concave (decreasing slope downstream) — usually
//!      lithology-pinned.
//!    - `d²z/dχ² < 0` → convex (increasing slope downstream) — usually
//!      transient / tectonic.
//! 6. Exclude cells within `confluence_buffer_cells` (default 5) of the
//!    segment's ends, per spec §8 pitfall #9: confluences and outlets
//!    induce spurious curvature.
//!
//! ## References
//!
//! - Neely, A.B., DiBiase, R.A., Corbett, L.B., Bierman, P.R. & Caffee,
//!   M.W. (2017). *Bedrock fracture density controls on hillslope
//!   erodibility.* EPSL 469, 157–168.
//!   <https://doi.org/10.1016/j.epsl.2017.04.008>
//! - Condat, L. (2013). *A direct algorithm for 1-D total variation
//!   denoising.* IEEE SP Letters 20(11), 1054–1057.
//!   <https://doi.org/10.1109/LSP.2013.2278339>
//!
//! ## TVD crate decision
//!
//! Spec §10.2 recommended the external crate `tvr`. That crate is not
//! published to crates.io as of 2026-05; rather than adopt an unmaintained
//! or non-existent dependency, the 1-D TVD inline implementation below
//! (Condat 2013, ~80 LOC) is exact and dependency-free.

use surtgis_core::{Raster, Result};

use super::chi::{ChiParams, chi_transform};
use super::stream_traversal::{StreamGraph, StreamGraphError, build_stream_graph};

/// Parameters for knickpoint detection.
#[derive(Debug, Clone)]
pub struct KnickpointParams {
    /// Reference concavity used to compute χ. Default 0.45.
    pub theta_ref: f64,
    /// TVD regularization parameter. Larger = stronger smoothing.
    /// Default 0.5 — tuned for 10–30 m DEMs in Neely et al. 2017.
    pub tvd_lambda: f64,
    /// Threshold on `|d²z/dχ²|` for a cell to be a knickpoint candidate.
    /// Units: 1 / metre (z in m, χ in m). Default 1.0 — tunes with the
    /// chosen `a_0_m2` (we use 1·10⁶ m² internally).
    pub curvature_threshold: f64,
    /// Minimum elevation magnitude across the knickpoint window for the
    /// candidate to be kept. Filters out tiny perturbations. Default 10 m.
    pub min_magnitude_m: f64,
    /// Pixel size in metres (must be > 0).
    pub cell_size_m: f64,
    /// Number of segment-end cells to ignore as candidates. Defaults to 5
    /// (spec §8 pitfall #9: confluences and outlets induce spurious
    /// curvature).
    pub confluence_buffer_cells: usize,
}

impl Default for KnickpointParams {
    fn default() -> Self {
        Self {
            theta_ref: 0.45,
            tvd_lambda: 0.5,
            curvature_threshold: 1.0,
            min_magnitude_m: 10.0,
            cell_size_m: 30.0,
            confluence_buffer_cells: 5,
        }
    }
}

/// Sign of curvature at a knickpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnickpointPolarity {
    /// `d²z/dχ² > 0`: slope decreases downstream. Typical interpretation:
    /// lithologic contrast where a softer bed sits below a resistant one.
    Concave,
    /// `d²z/dχ² < 0`: slope increases downstream. Typical interpretation:
    /// transient response to a tectonic pulse propagating upstream.
    Convex,
}

/// One detected knickpoint.
#[derive(Debug, Clone)]
pub struct Knickpoint {
    /// Row of the knickpoint cell in the source raster.
    pub row: usize,
    /// Column of the knickpoint cell in the source raster.
    pub col: usize,
    /// Denoised elevation at the cell (metres).
    pub elevation_m: f64,
    /// Elevation magnitude across the knickpoint window (metres).
    pub magnitude_m: f64,
    /// χ value at the cell (metres) — useful for downstream plotting.
    pub chi: f64,
    /// Sign convention; see [`KnickpointPolarity`].
    pub polarity: KnickpointPolarity,
}

/// Errors specific to knickpoint detection.
#[derive(Debug, thiserror::Error)]
pub enum KnickpointError {
    /// Building the stream graph failed.
    #[error(transparent)]
    Graph(#[from] StreamGraphError),
    /// Two input rasters had incompatible shapes.
    #[error("input raster shapes disagree: {0:?} vs {1:?}")]
    ShapeMismatch((usize, usize), (usize, usize)),
    /// `cell_size_m` was not strictly positive.
    #[error("KnickpointParams.cell_size_m must be > 0 (got {0})")]
    NonPositiveCellSize(f64),
    /// The total-variation denoising weight `tvd_lambda` was not strictly positive.
    #[error("KnickpointParams.tvd_lambda must be > 0 (got {0})")]
    NonPositiveLambda(f64),
}

/// Detect knickpoints along every continuous stream segment.
///
/// Returns a flat `Vec<Knickpoint>` — one entry per detection. Multiple
/// knickpoints on the same segment are returned independently; callers
/// who want spatial aggregation should cluster after the fact.
#[allow(clippy::too_many_arguments)]
pub fn knickpoint_detection(
    stream: &Raster<u8>,
    flow_dir: &Raster<u8>,
    flow_acc: &Raster<f64>,
    dem: &Raster<f64>,
    params: KnickpointParams,
) -> Result<Vec<Knickpoint>> {
    if params.cell_size_m <= 0.0 {
        return Err(surtgis_core::Error::Other(
            KnickpointError::NonPositiveCellSize(params.cell_size_m).to_string(),
        ));
    }
    if params.tvd_lambda <= 0.0 {
        return Err(surtgis_core::Error::Other(
            KnickpointError::NonPositiveLambda(params.tvd_lambda).to_string(),
        ));
    }
    let s_shape = stream.shape();
    for other in [flow_dir.shape(), flow_acc.shape(), dem.shape()] {
        if other != s_shape {
            return Err(surtgis_core::Error::Other(
                KnickpointError::ShapeMismatch(other, s_shape).to_string(),
            ));
        }
    }

    let graph = build_stream_graph(stream, flow_dir)
        .map_err(|e| surtgis_core::Error::Other(e.to_string()))?;

    // Compute χ once. Within a segment we read χ from this raster — the
    // numerical value of a_0_m2 only sets the absolute χ scale; curvature
    // d²z/dχ² scales by 1/a_0_m2^(2θ), which the user implicitly tunes
    // via `curvature_threshold`. Hardcoding 1·10⁶ m² gives reasonable
    // threshold values around 1.0 for 10–30 m DEMs.
    let chi_raster = chi_transform(
        stream,
        flow_dir,
        flow_acc,
        ChiParams {
            theta_ref: params.theta_ref,
            a_0_m2: 1.0e6,
            cell_size_m: params.cell_size_m,
            base_outlets: None,
        },
    )?;

    // Iterate over every continuous segment in downstream → upstream
    // order (this is the order Condat's TVD expects on a 1-D signal).
    let segments = extract_segments(&graph);
    let mut out: Vec<Knickpoint> = Vec::new();

    for seg_cells in segments {
        // Read z and χ along the segment in downstream → upstream order.
        let n = seg_cells.len();
        if n < 2 * params.confluence_buffer_cells + 3 {
            // Too short to host a meaningful knickpoint after edge buffer.
            continue;
        }
        let mut z: Vec<f64> = Vec::with_capacity(n);
        let mut chi: Vec<f64> = Vec::with_capacity(n);
        for &node in &seg_cells {
            let (r, c) = graph.stream_cells[node];
            let zi = dem.get(r, c).unwrap_or(f64::NAN);
            let ci = chi_raster.get(r, c).unwrap_or(f64::NAN);
            z.push(zi);
            chi.push(ci);
        }

        // Skip the whole segment if any sample is non-finite — we don't
        // want NaNs leaking into TVD or curvature.
        if z.iter().any(|v| !v.is_finite()) || chi.iter().any(|v| !v.is_finite()) {
            continue;
        }

        // Denoise z.
        let z_smooth = tvd_condat_1d(&z, params.tvd_lambda);

        // Curvature at interior cells with non-uniform-spacing 3-point
        // stencil. d²z/dχ² ≈ 2 · (z[i-1]·(χ[i+1]-χ[i]) - z[i]·(χ[i+1]-χ[i-1])
        //                           + z[i+1]·(χ[i]-χ[i-1])) /
        //                          ((χ[i+1]-χ[i])·(χ[i]-χ[i-1])·(χ[i+1]-χ[i-1]))
        let buf = params.confluence_buffer_cells;
        let i_start = buf.max(1);
        let i_end = n.saturating_sub(buf.max(1));
        if i_end <= i_start + 1 {
            continue;
        }
        for i in i_start..i_end {
            let dchi_down = chi[i] - chi[i - 1];
            let dchi_up = chi[i + 1] - chi[i];
            let dchi_full = chi[i + 1] - chi[i - 1];
            if dchi_down <= 0.0 || dchi_up <= 0.0 || dchi_full <= 0.0 {
                // χ should monotonically increase upstream; any non-monotonic
                // step indicates a flow-dir / chi inconsistency. Skip rather
                // than divide-by-zero.
                continue;
            }
            let num = 2.0
                * (z_smooth[i - 1] * dchi_up - z_smooth[i] * dchi_full
                    + z_smooth[i + 1] * dchi_down);
            let den = dchi_up * dchi_down * dchi_full;
            let curvature = num / den;
            if !curvature.is_finite() || curvature.abs() < params.curvature_threshold {
                continue;
            }
            // Magnitude: elevation drop across a ±2-cell window. Bounded
            // by segment edges. Use the denoised elevations.
            let w = 2usize;
            let lo = i.saturating_sub(w);
            let hi = (i + w).min(n - 1);
            let mag = (z_smooth[lo] - z_smooth[hi]).abs();
            if mag < params.min_magnitude_m {
                continue;
            }
            let polarity = if curvature > 0.0 {
                KnickpointPolarity::Concave
            } else {
                KnickpointPolarity::Convex
            };
            let (r, c) = graph.stream_cells[seg_cells[i]];
            out.push(Knickpoint {
                row: r,
                col: c,
                elevation_m: z_smooth[i],
                magnitude_m: mag,
                chi: chi[i],
                polarity,
            });
        }
    }

    Ok(out)
}

/// Extract every maximal single-channel segment in downstream → upstream
/// order. Each segment is a `Vec<usize>` of [`StreamGraph`] node
/// indices; the first element is the most downstream node of the
/// segment, the last is the most upstream.
///
/// Same topology recipe as `channel_steepness::extract_segments` but
/// returns node indices instead of geometry. Kept local to this module
/// so the two stay independently maintainable (their needs may diverge
/// in later sprints).
fn extract_segments(graph: &StreamGraph) -> Vec<Vec<usize>> {
    let mut segments: Vec<Vec<usize>> = Vec::new();
    let mut starts: Vec<usize> = Vec::new();
    for i in 0..graph.len() {
        if graph.is_outlet[i] {
            starts.push(i);
        } else if graph.upstream_links[i].len() >= 2 {
            for &trib in &graph.upstream_links[i] {
                starts.push(trib);
            }
        }
    }
    for start in starts {
        let mut cells = vec![start];
        let mut cur = start;
        loop {
            let ups = &graph.upstream_links[cur];
            if ups.is_empty() || ups.len() >= 2 {
                break;
            }
            let next = ups[0];
            cells.push(next);
            cur = next;
        }
        segments.push(cells);
    }
    segments
}

/// Condat's direct 1-D Total-Variation Denoising algorithm (2013).
///
/// Solves the proximal operator of the discrete TV norm:
/// argmin_y { ½ Σᵢ (yᵢ-xᵢ)² + λ Σᵢ |y_{i+1}-yᵢ| }.
///
/// Exact, O(N) in the typical case, no iteration tolerances or step
/// sizes to tune. The full reference algorithm is ~80 LOC and fits
/// comfortably inline; depending on an external crate for this would be
/// overkill (cf. note at the top of the module).
fn tvd_condat_1d(x: &[f64], lambda: f64) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![x[0]];
    }
    let mut y = vec![0.0f64; n];

    let mut k: usize = 0;
    let mut k0: usize = 0;
    let mut k_minus: usize = 0;
    let mut k_plus: usize = 0;
    let mut v_min = x[0] - lambda;
    let mut v_max = x[0] + lambda;
    let mut u_min = lambda;
    let mut u_max = -lambda;

    loop {
        if k == n - 1 {
            if u_min < 0.0 {
                // descending step: emit v_min over [k0..=k_minus], restart
                for j in k0..=k_minus {
                    y[j] = v_min;
                }
                k = k_minus + 1;
                k0 = k;
                k_minus = k;
                k_plus = k;
                v_min = x[k];
                v_max = x[k] + 2.0 * lambda;
                u_min = lambda;
                u_max = -lambda;
            } else if u_max > 0.0 {
                // ascending step: emit v_max over [k0..=k_plus], restart
                for j in k0..=k_plus {
                    y[j] = v_max;
                }
                k = k_plus + 1;
                k0 = k;
                k_minus = k;
                k_plus = k;
                v_min = x[k] - 2.0 * lambda;
                v_max = x[k];
                u_min = lambda;
                u_max = -lambda;
            } else {
                // optimal level for the trailing run
                let level = v_min + u_min / ((k - k0 + 1) as f64);
                for j in k0..n {
                    y[j] = level;
                }
                return y;
            }
            // After restart above, we need to keep iterating. Skip to
            // bottom of loop unconditionally rather than the increment.
            continue;
        }

        k += 1;
        u_min += x[k] - v_min;
        u_max += x[k] - v_max;

        if u_min < -lambda {
            // commit v_min run, slide window
            for j in k0..=k_minus {
                y[j] = v_min;
            }
            k = k_minus + 1;
            k0 = k;
            k_minus = k;
            k_plus = k;
            v_min = x[k];
            v_max = x[k] + 2.0 * lambda;
            u_min = lambda;
            u_max = -lambda;
        } else if u_max > lambda {
            for j in k0..=k_plus {
                y[j] = v_max;
            }
            k = k_plus + 1;
            k0 = k;
            k_minus = k;
            k_plus = k;
            v_min = x[k] - 2.0 * lambda;
            v_max = x[k];
            u_min = lambda;
            u_max = -lambda;
        } else {
            if u_min >= lambda {
                k_minus = k;
                v_min += (u_min - lambda) / ((k_minus - k0 + 1) as f64);
                u_min = lambda;
            }
            if u_max <= -lambda {
                k_plus = k;
                v_max += (u_max + lambda) / ((k_plus - k0 + 1) as f64);
                u_max = -lambda;
            }
        }
    }
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

    /// Condat TVD smoke: a single sharp step should remain a step (the
    /// optimal TVD output preserves the step location); a noisy flat
    /// signal should collapse to its mean.
    #[test]
    fn tvd_collapses_noise_around_step() {
        let mut x = vec![0.0f64; 20];
        for i in 10..20 {
            x[i] = 1.0;
        }
        // Add small noise to both sides of the step.
        for i in 0..x.len() {
            x[i] += if i % 2 == 0 { 0.05 } else { -0.05 };
        }
        let y = tvd_condat_1d(&x, 0.2);
        // After denoising, left half should be approximately constant
        // (< noise amplitude) and right half should be approximately
        // constant ~1.0 - noise.
        let mean_left: f64 = y[..10].iter().sum::<f64>() / 10.0;
        let mean_right: f64 = y[10..].iter().sum::<f64>() / 10.0;
        for v in &y[..10] {
            assert!(
                (v - mean_left).abs() < 0.01,
                "left should be flat after TVD"
            );
        }
        for v in &y[10..] {
            assert!(
                (v - mean_right).abs() < 0.01,
                "right should be flat after TVD"
            );
        }
        assert!((mean_right - mean_left) > 0.5, "TVD must preserve the step");
    }

    /// Spec §7.2 headline test for knickpoints: build a synthetic profile
    /// with a sharp elevation drop at a known cell, run the detector,
    /// confirm the detected knickpoint lands within ±2 cells.
    #[test]
    fn synthetic_knickpoint_localises_within_two_cells() {
        let n = 30;
        // Straight east-flowing channel at row 0.
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        // Drainage area increases west→east. Headwater A=1 cell, outlet A=n.
        let acc: Vec<f64> = (1..=n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc]);
        // DEM: gentle slope on each side of a sharp 30 m drop at col 15.
        // The cell at col 15 is the "downstream" side of the knickpoint;
        // upstream is col 16. Channel flows EAST, so downstream means
        // INCREASING col index. χ INCREASES upstream → DECREASING col.
        let knick_col = 15;
        let drop_m = 30.0;
        let z: Vec<f64> = (0..n)
            .map(|c| {
                // Far-east cells (c >= knick_col) are LOW; far-west cells are HIGH.
                // Add a constant drop at the knickpoint.
                if c >= knick_col {
                    (n - 1 - c) as f64 * 0.5
                } else {
                    (n - 1 - c) as f64 * 0.5 + drop_m
                }
            })
            .collect();
        let dem = raster_f64(vec![z]);

        let params = KnickpointParams {
            theta_ref: 0.45,
            tvd_lambda: 0.3,
            curvature_threshold: 0.0001,
            min_magnitude_m: 5.0,
            cell_size_m: 30.0,
            confluence_buffer_cells: 3,
        };
        let knicks = knickpoint_detection(&stream, &flow_dir, &flow_acc, &dem, params).unwrap();

        assert!(
            !knicks.is_empty(),
            "expected at least one knickpoint, got none"
        );
        // The drop site should appear at col 15 or 16 (the boundary).
        // Allow ±2-cell tolerance per spec §7.2.
        let detected_cols: Vec<usize> = knicks.iter().map(|k| k.col).collect();
        let close = detected_cols
            .iter()
            .any(|&c| (c as i64 - knick_col as i64).abs() <= 2);
        assert!(
            close,
            "no detected knickpoint within ±2 cells of col {}: got {:?}",
            knick_col, detected_cols,
        );
    }

    /// A smooth profile (no breaks) must NOT produce knickpoint detections
    /// — the false-positive guard from spec §7.2.
    #[test]
    fn smooth_profile_no_false_positives() {
        let n = 30;
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        let acc: Vec<f64> = (1..=n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc]);
        // Pure linear profile in z, no breaks.
        let z: Vec<f64> = (0..n).map(|c| (n - 1 - c) as f64 * 1.0).collect();
        let dem = raster_f64(vec![z]);

        let params = KnickpointParams {
            theta_ref: 0.45,
            tvd_lambda: 0.5,
            curvature_threshold: 0.01,
            min_magnitude_m: 5.0,
            cell_size_m: 30.0,
            confluence_buffer_cells: 3,
        };
        let knicks = knickpoint_detection(&stream, &flow_dir, &flow_acc, &dem, params).unwrap();
        assert!(
            knicks.is_empty(),
            "smooth profile produced {} false-positive knickpoints",
            knicks.len()
        );
    }

    /// Confluence buffer must exclude detections near segment ends.
    /// Build a profile with a knickpoint sitting close to the upstream
    /// end and confirm a large buffer cancels the detection.
    #[test]
    fn confluence_buffer_excludes_edge_detections() {
        let n = 12;
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        let acc: Vec<f64> = (1..=n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc]);
        // Knickpoint at col 2 (very close to headwater at col 0).
        // With confluence_buffer_cells = 5, this should be excluded.
        let z: Vec<f64> = (0..n)
            .map(|c| {
                if c >= 2 {
                    (n - 1 - c) as f64 * 0.5
                } else {
                    (n - 1 - c) as f64 * 0.5 + 30.0
                }
            })
            .collect();
        let dem = raster_f64(vec![z]);

        let params = KnickpointParams {
            theta_ref: 0.45,
            tvd_lambda: 0.3,
            curvature_threshold: 0.0001,
            min_magnitude_m: 5.0,
            cell_size_m: 30.0,
            confluence_buffer_cells: 5,
        };
        let knicks = knickpoint_detection(&stream, &flow_dir, &flow_acc, &dem, params).unwrap();
        // With a confluence buffer of 5 in a 12-cell segment, candidate
        // window is cells [5..7) — only 2 cells. The drop at col 2 is
        // outside this window so it must NOT be reported.
        for k in &knicks {
            assert!(
                k.col >= 5 && k.col < 7,
                "knickpoint at col {} should have been excluded by buffer",
                k.col
            );
        }
    }

    /// Polarity classification: at least one polarity must be Convex on a
    /// profile whose channel slope sharpens downstream (curvature d²z/dχ²
    /// is negative). We use the same general construction as the
    /// localisation test, with a single big elevation drop, since that
    /// drop biases curvature negative when crossed in the downstream→
    /// upstream walk (z rises sharply at the kink moving upstream, then
    /// flattens; dz/dχ is large at low χ and small at high χ, so
    /// d²z/dχ² < 0 = Convex).
    #[test]
    fn polarity_assignment_includes_convex_on_sharp_downstream_drop() {
        let n = 30;
        let stream = raster_u8(vec![vec![1u8; n]]);
        let flow_dir = raster_u8(vec![vec![1u8; n]]);
        let acc: Vec<f64> = (1..=n as i32).map(|i| i as f64).collect();
        let flow_acc = raster_f64(vec![acc]);
        let knick_col = 15;
        let drop_m = 80.0; // big drop so TVD can't smooth it out
        let z: Vec<f64> = (0..n)
            .map(|c| {
                if c >= knick_col {
                    (n - 1 - c) as f64 * 0.5
                } else {
                    (n - 1 - c) as f64 * 0.5 + drop_m
                }
            })
            .collect();
        let dem = raster_f64(vec![z]);

        let params = KnickpointParams {
            theta_ref: 0.45,
            tvd_lambda: 0.1, // light smoothing — preserve the kink
            curvature_threshold: 0.0001,
            min_magnitude_m: 5.0,
            cell_size_m: 30.0,
            confluence_buffer_cells: 3,
        };
        let knicks = knickpoint_detection(&stream, &flow_dir, &flow_acc, &dem, params).unwrap();
        assert!(!knicks.is_empty(), "no knickpoint detected");
        assert!(
            knicks
                .iter()
                .any(|k| k.polarity == KnickpointPolarity::Convex),
            "expected at least one Convex knickpoint on a sharp downstream drop, \
             got polarities: {:?}",
            knicks.iter().map(|k| k.polarity).collect::<Vec<_>>()
        );
    }
}
