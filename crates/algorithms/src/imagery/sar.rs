//! SAR (Synthetic Aperture Radar) water and flood mapping primitives.
//!
//! A minimal, dependency-free SAR toolkit aimed at Sentinel-1-style dual-pol
//! flood mapping. It deliberately reuses the existing imagery and temporal
//! machinery for the heavy lifting:
//!
//! - **Multi-temporal compositing** (min / median backscatter to suppress
//!   transient bright returns) → use [`crate::temporal::temporal_min`] /
//!   [`crate::imagery::median_composite`].
//! - **Amplitude change detection** (flood = backscatter drop vs. a dry
//!   reference) → use [`crate::imagery::raster_difference`] or
//!   [`crate::imagery::ir_mad`] on the backscatter bands.
//!
//! This module adds the SAR-specific pieces that were missing:
//!
//! - [`linear_to_db`] / [`db_to_linear`] — backscatter unit conversion, needed
//!   because water-detection thresholds are conventionally expressed in dB.
//! - [`dual_pol_water_index`] — the normalized co-/cross-pol difference
//!   `(VV - VH) / (VV + VH)`, which is elevated over smooth open water.
//! - [`sar_water_mask`] — threshold a backscatter (or index) raster into a
//!   binary water/flood mask.
//! - [`lee_filter`] — the classic Lee (1980) adaptive speckle filter, applied
//!   before thresholding to suppress multiplicative speckle while preserving
//!   edges.
//! - [`refined_lee_filter`] — the edge-aligned *refined* Lee (1981): estimates
//!   the local statistics from the half-window on the homogeneous side of the
//!   dominant edge, so it preserves edges and linear features better than the
//!   square-window classic Lee.

use crate::imagery::normalized_difference;
use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// u8 mask value for non-water cells.
pub const SAR_NON_WATER: u8 = 0;
/// u8 mask value for water / flooded cells.
pub const SAR_WATER: u8 = 1;
/// u8 mask value for nodata cells.
pub const SAR_NODATA: u8 = 255;

#[inline]
fn is_nodata(v: f64, nodata: Option<f64>) -> bool {
    v.is_nan() || nodata.map(|nd| v == nd).unwrap_or(false)
}

/// Map a single-band f64 raster elementwise (row-parallel), returning a new
/// f64 raster whose nodata is `NaN`.
fn map_f64<F>(src: &Raster<f64>, f: F) -> Result<Raster<f64>>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let (rows, cols) = src.shape();
    let nodata = src.nodata();
    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let v = unsafe { src.get_unchecked(row, col) };
                if is_nodata(v, nodata) {
                    continue;
                }
                *out = f(v);
            }
            row_data
        })
        .collect();

    let mut output = src.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

/// Convert linear-power backscatter (σ⁰, γ⁰ …) to decibels: `10·log₁₀(x)`.
///
/// Non-positive or nodata cells become `NaN`. Already-calibrated products from
/// Planetary Computer (`sentinel-1-rtc`) are in linear power and should be
/// converted with this before applying dB thresholds.
pub fn linear_to_db(backscatter: &Raster<f64>) -> Result<Raster<f64>> {
    map_f64(backscatter, |x| {
        if x <= 0.0 { f64::NAN } else { 10.0 * x.log10() }
    })
}

/// Convert decibel backscatter back to linear power: `10^(x/10)`.
pub fn db_to_linear(backscatter_db: &Raster<f64>) -> Result<Raster<f64>> {
    map_f64(backscatter_db, |x| 10.0_f64.powf(x / 10.0))
}

/// Dual-polarisation water index: `(co_pol - cross_pol) / (co_pol + cross_pol)`.
///
/// For Sentinel-1 pass `co_pol = VV`, `cross_pol = VH` (linear power). Smooth
/// open water depolarises weakly, so VH collapses faster than VV and the index
/// rises toward +1 over water while staying low over rough land. Use
/// [`sar_water_mask`] with a positive threshold and `water_below = false` to
/// extract water from this index.
///
/// This is the SAR analogue of the optical normalized-difference water indices
/// and shares their numerics via [`normalized_difference`].
pub fn dual_pol_water_index(co_pol: &Raster<f64>, cross_pol: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(co_pol, cross_pol)
}

/// Threshold a backscatter or index raster into a binary water/flood mask.
///
/// # Arguments
/// * `raster` - input backscatter (e.g. VV in dB) or a water index.
/// * `threshold` - decision boundary.
/// * `water_below` - if `true`, cells `< threshold` are water (the usual case
///   for backscatter: open water is specular and returns little energy); if
///   `false`, cells `> threshold` are water (the case for a water *index* that
///   is high over water).
///
/// Returns a `u8` raster using [`SAR_WATER`], [`SAR_NON_WATER`] and
/// [`SAR_NODATA`] (with the raster nodata set to [`SAR_NODATA`]). `NaN` /
/// nodata input cells map to [`SAR_NODATA`].
pub fn sar_water_mask(
    raster: &Raster<f64>,
    threshold: f64,
    water_below: bool,
) -> Result<Raster<u8>> {
    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();
    let data: Vec<u8> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![SAR_NODATA; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let v = unsafe { raster.get_unchecked(row, col) };
                if is_nodata(v, nodata) {
                    continue;
                }
                let is_water = if water_below {
                    v < threshold
                } else {
                    v > threshold
                };
                *out = if is_water { SAR_WATER } else { SAR_NON_WATER };
            }
            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(SAR_NODATA));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

/// Lee speckle filter (Lee 1980) for SAR amplitude/intensity imagery.
///
/// An adaptive MMSE filter for multiplicative speckle. In each local window it
/// computes the mean `μ` and variance `σ²`, and blends the centre pixel toward
/// the local mean by a weight that depends on how heterogeneous the window is:
///
/// ```text
/// Cu² = 1 / ENL                  (squared coeff. of variation of pure speckle)
/// Ci² = σ² / μ²                  (squared local coeff. of variation)
/// W   = max(0, 1 − Cu² / Ci²)
/// out = μ + W · (centre − μ)
/// ```
///
/// In homogeneous regions (`Ci ≤ Cu`) the weight is 0 and the output is the
/// local mean — full speckle smoothing. Near edges and bright targets
/// (`Ci ≫ Cu`) the weight approaches 1 and the centre pixel is preserved, so
/// edges and point scatterers are not blurred.
///
/// Operates on linear-power (intensity) backscatter. `looks` is the equivalent
/// number of looks (ENL) of the product — use 1.0 for single-look data; higher
/// ENL (multi-look / RTC products) yields a weaker speckle model and less
/// smoothing. This is the classic Lee filter; see [`refined_lee_filter`] for the
/// edge-aligned refined Lee (1981).
///
/// # Arguments
/// * `image` - single-band SAR backscatter (linear power).
/// * `window_size` - odd window side length (e.g. 3, 5, 7).
/// * `looks` - equivalent number of looks (ENL), must be positive.
///
/// # References
/// - Lee, J.-S. (1980). Digital image enhancement and noise filtering by use of
///   local statistics. *IEEE TPAMI* 2(2), 165–168.
pub fn lee_filter(image: &Raster<f64>, window_size: usize, looks: f64) -> Result<Raster<f64>> {
    if window_size < 3 || window_size % 2 == 0 {
        return Err(Error::Other(
            "window_size must be an odd number >= 3".into(),
        ));
    }
    if looks <= 0.0 {
        return Err(Error::Other("looks (ENL) must be positive".into()));
    }

    let (rows, cols) = image.shape();
    let nodata = image.nodata();
    let radius = window_size / 2;
    // Squared coefficient of variation of fully-developed speckle.
    let cu2 = 1.0 / looks;

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { image.get_unchecked(row, col) };
                if is_nodata(center, nodata) {
                    continue;
                }

                let r0 = row.saturating_sub(radius);
                let r1 = (row + radius).min(rows - 1);
                let c0 = col.saturating_sub(radius);
                let c1 = (col + radius).min(cols - 1);

                let mut n = 0usize;
                let mut sum = 0.0;
                let mut sumsq = 0.0;
                for wr in r0..=r1 {
                    for wc in c0..=c1 {
                        let v = unsafe { image.get_unchecked(wr, wc) };
                        if is_nodata(v, nodata) {
                            continue;
                        }
                        n += 1;
                        sum += v;
                        sumsq += v * v;
                    }
                }

                let mean = sum / n as f64;
                // Degenerate window: too few valid samples or zero mean -> mean.
                if n < 2 || mean == 0.0 {
                    *out = mean;
                    continue;
                }

                let var = (sumsq / n as f64 - mean * mean).max(0.0);
                let ci2 = var / (mean * mean);
                let w = if ci2 <= cu2 { 0.0 } else { 1.0 - cu2 / ci2 };
                *out = mean + w * (center - mean);
            }
            row_data
        })
        .collect();

    let mut output = image.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

/// Refined Lee speckle filter (Lee 1981) for SAR backscatter.
///
/// An edge-aware refinement of [`lee_filter`]. The classic Lee filter estimates
/// the local statistics from the full square window, so it blurs across edges;
/// the refined Lee instead estimates them from an **edge-aligned half-window**,
/// preserving edges and linear features while still smoothing speckle in
/// homogeneous areas.
///
/// For each pixel it (1) finds the dominant edge orientation among four
/// directions (horizontal, vertical, and the two diagonals) from the
/// across-edge mean contrast, (2) keeps the half of the window on the
/// *homogeneous* side of that edge — the side whose mean is closer to the centre
/// pixel — and (3) applies the same MMSE estimator as [`lee_filter`]
/// (`out = μ + W·(centre − μ)`, `W = max(0, 1 − Cu²/Ci²)`, `Cu² = 1/ENL`) using
/// only that half-window's mean and variance. Where no edge is detected the full
/// window is used (it reduces to the classic Lee filter).
///
/// Operates on linear-power backscatter. `window_size` is the odd side length
/// (classically 7); `looks` is the equivalent number of looks (ENL).
///
/// # References
/// - Lee, J.-S. (1981). Refined filtering of image noise using local statistics.
///   *Computer Graphics and Image Processing* 15(4), 380–389.
pub fn refined_lee_filter(
    image: &Raster<f64>,
    window_size: usize,
    looks: f64,
) -> Result<Raster<f64>> {
    if window_size < 3 || window_size % 2 == 0 {
        return Err(Error::Other(
            "window_size must be an odd number >= 3".into(),
        ));
    }
    if looks <= 0.0 {
        return Err(Error::Other("looks (ENL) must be positive".into()));
    }

    let (rows, cols) = image.shape();
    let nodata = image.nodata();
    let radius = window_size / 2;
    let cu2 = 1.0 / looks;

    // Signed across-edge projection for each of the four edge orientations.
    let proj = |d: usize, di: isize, dj: isize| -> isize {
        match d {
            0 => di,      // horizontal edge / N-S contrast
            1 => dj,      // vertical edge / E-W contrast
            2 => di + dj, // diagonal
            _ => di - dj, // anti-diagonal
        }
    };

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { image.get_unchecked(row, col) };
                if is_nodata(center, nodata) {
                    continue;
                }
                let r0 = row.saturating_sub(radius);
                let r1 = (row + radius).min(rows - 1);
                let c0 = col.saturating_sub(radius);
                let c1 = (col + radius).min(cols - 1);

                // Pass 1: per-direction means of the two half-windows.
                let mut s_neg = [0.0_f64; 4];
                let mut n_neg = [0usize; 4];
                let mut s_pos = [0.0_f64; 4];
                let mut n_pos = [0usize; 4];
                for wr in r0..=r1 {
                    for wc in c0..=c1 {
                        let v = unsafe { image.get_unchecked(wr, wc) };
                        if is_nodata(v, nodata) {
                            continue;
                        }
                        let di = wr as isize - row as isize;
                        let dj = wc as isize - col as isize;
                        for d in 0..4 {
                            let p = proj(d, di, dj);
                            if p < 0 {
                                s_neg[d] += v;
                                n_neg[d] += 1;
                            } else if p > 0 {
                                s_pos[d] += v;
                                n_pos[d] += 1;
                            }
                        }
                    }
                }

                // Dominant edge = max across-edge mean contrast; keep the side
                // whose mean is closer to the centre (the homogeneous side).
                let mut best_d = 0usize;
                let mut best_g = -1.0_f64;
                let mut keep_pos = true;
                for d in 0..4 {
                    if n_neg[d] == 0 || n_pos[d] == 0 {
                        continue;
                    }
                    let mn = s_neg[d] / n_neg[d] as f64;
                    let mp = s_pos[d] / n_pos[d] as f64;
                    let g = (mp - mn).abs();
                    if g > best_g {
                        best_g = g;
                        best_d = d;
                        keep_pos = (center - mp).abs() <= (center - mn).abs();
                    }
                }

                // Pass 2: mean/variance over the kept half-window (or the full
                // window if no edge orientation was resolvable).
                let mut n = 0usize;
                let mut sum = 0.0;
                let mut sumsq = 0.0;
                for wr in r0..=r1 {
                    for wc in c0..=c1 {
                        let v = unsafe { image.get_unchecked(wr, wc) };
                        if is_nodata(v, nodata) {
                            continue;
                        }
                        let keep = if best_g < 0.0 {
                            true
                        } else {
                            let p = proj(
                                best_d,
                                wr as isize - row as isize,
                                wc as isize - col as isize,
                            );
                            p == 0 || (keep_pos && p > 0) || (!keep_pos && p < 0)
                        };
                        if keep {
                            n += 1;
                            sum += v;
                            sumsq += v * v;
                        }
                    }
                }

                let mean = sum / n as f64;
                if n < 2 || mean == 0.0 {
                    *out = mean;
                    continue;
                }
                let var = (sumsq / n as f64 - mean * mean).max(0.0);
                let ci2 = var / (mean * mean);
                let w = if ci2 <= cu2 { 0.0 } else { 1.0 - cu2 / ci2 };
                *out = mean + w * (center - mean);
            }
            row_data
        })
        .collect();

    let mut output = image.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn r(data: Vec<f64>, rows: usize, cols: usize) -> Raster<f64> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut x = Raster::from_array(arr);
        x.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        x
    }

    #[test]
    fn test_linear_db_roundtrip() {
        let lin = r(vec![1.0, 0.1, 0.01, 10.0], 2, 2);
        let db = linear_to_db(&lin).unwrap();
        // 10*log10(1)=0, 0.1->-10, 0.01->-20, 10->10
        assert!((db.get(0, 0).unwrap() - 0.0).abs() < 1e-9);
        assert!((db.get(0, 1).unwrap() + 10.0).abs() < 1e-9);
        assert!((db.get(1, 0).unwrap() + 20.0).abs() < 1e-9);
        assert!((db.get(1, 1).unwrap() - 10.0).abs() < 1e-9);
        // round-trip
        let back = db_to_linear(&db).unwrap();
        for (a, b) in lin.data().iter().zip(back.data().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_linear_to_db_nonpositive_is_nan() {
        let lin = r(vec![0.0, -1.0, 0.5, f64::NAN], 2, 2);
        let db = linear_to_db(&lin).unwrap();
        assert!(db.get(0, 0).unwrap().is_nan());
        assert!(db.get(0, 1).unwrap().is_nan());
        assert!(!db.get(1, 0).unwrap().is_nan());
        assert!(db.get(1, 1).unwrap().is_nan());
    }

    #[test]
    fn test_dual_pol_index_water_high() {
        // Water: VV moderate, VH near zero -> index near +1.
        // Land: VV ~ VH -> index near 0.
        let vv = r(vec![0.1, 0.2, 0.2, 0.2], 2, 2);
        let vh = r(vec![0.001, 0.18, 0.2, 0.19], 2, 2);
        let idx = dual_pol_water_index(&vv, &vh).unwrap();
        assert!(idx.get(0, 0).unwrap() > 0.9, "water cell should be near +1");
        assert!(idx.get(1, 0).unwrap().abs() < 1e-6, "equal pol -> 0");
    }

    #[test]
    fn test_sar_water_mask_below() {
        // Backscatter in dB: water is low (< -17 dB typical for VV).
        let vv_db = r(vec![-20.0, -5.0, -18.0, -3.0], 2, 2);
        let mask = sar_water_mask(&vv_db, -17.0, true).unwrap();
        assert_eq!(mask.get(0, 0).unwrap(), SAR_WATER); // -20 < -17
        assert_eq!(mask.get(0, 1).unwrap(), SAR_NON_WATER); // -5
        assert_eq!(mask.get(1, 0).unwrap(), SAR_WATER); // -18
        assert_eq!(mask.get(1, 1).unwrap(), SAR_NON_WATER); // -3
    }

    #[test]
    fn test_sar_water_mask_above_and_nodata() {
        // Index mode: water is HIGH; include a NaN cell.
        let idx = r(vec![0.8, 0.1, f64::NAN, 0.6], 2, 2);
        let mask = sar_water_mask(&idx, 0.5, false).unwrap();
        assert_eq!(mask.get(0, 0).unwrap(), SAR_WATER);
        assert_eq!(mask.get(0, 1).unwrap(), SAR_NON_WATER);
        assert_eq!(mask.get(1, 0).unwrap(), SAR_NODATA);
        assert_eq!(mask.get(1, 1).unwrap(), SAR_WATER);
        assert_eq!(mask.nodata(), Some(SAR_NODATA));
    }

    #[test]
    fn test_lee_filter_homogeneous_returns_mean() {
        // A uniform field has Ci = 0 <= Cu, so every output equals the local
        // mean = the constant value: the filter must be a no-op (within fp).
        let img = r(vec![5.0; 25], 5, 5);
        let out = lee_filter(&img, 3, 1.0).unwrap();
        for v in out.data().iter() {
            assert!((v - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_lee_filter_preserves_strong_point_target() {
        // A single very bright pixel in a dark field: Ci >> Cu at the centre,
        // so W -> ~1 and the bright value is largely preserved (not smoothed to
        // the local mean).
        let mut data = vec![1.0; 25];
        data[12] = 1000.0; // centre of 5x5
        let img = r(data, 5, 5);
        let out = lee_filter(&img, 3, 1.0).unwrap();
        let center = out.get(2, 2).unwrap();
        let local_mean = (1000.0 + 8.0 * 1.0) / 9.0; // ~112
        // The target must be largely preserved: far above the local mean and
        // much closer to the original peak than to the mean.
        assert!(
            center > 800.0,
            "strong target should be preserved, got {center} (mean ~{local_mean})"
        );
        assert!(
            (1000.0 - center) < (center - local_mean),
            "output {center} should be closer to the peak than to the mean {local_mean}"
        );
    }

    #[test]
    fn test_lee_filter_smooths_speckle() {
        // Noisy-ish window: filtered centre should move toward the local mean
        // (variance of the output block is lower than the input block).
        let data = vec![
            2.0, 8.0, 3.0, 9.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0, 1.0,
            7.0, 2.0, 8.0, 3.0, 9.0, 1.0, 7.0, 2.0,
        ];
        let img = r(data.clone(), 5, 5);
        let out = lee_filter(&img, 5, 4.0).unwrap();
        let in_var = {
            let m = data.iter().sum::<f64>() / 25.0;
            data.iter().map(|v| (v - m).powi(2)).sum::<f64>() / 25.0
        };
        let ov: Vec<f64> = out.data().iter().copied().collect();
        let out_var = {
            let m = ov.iter().sum::<f64>() / 25.0;
            ov.iter().map(|v| (v - m).powi(2)).sum::<f64>() / 25.0
        };
        assert!(
            out_var < in_var,
            "filter should reduce variance: {out_var} !< {in_var}"
        );
    }

    #[test]
    fn test_lee_filter_nodata_and_validation() {
        let img = r(vec![1.0, 2.0, f64::NAN, 4.0], 2, 2);
        let out = lee_filter(&img, 3, 1.0).unwrap();
        assert!(out.get(1, 0).unwrap().is_nan()); // nodata center stays nodata
        assert!(!out.get(0, 0).unwrap().is_nan());

        assert!(lee_filter(&img, 2, 1.0).is_err()); // even window
        assert!(lee_filter(&img, 1, 1.0).is_err()); // too small
        assert!(lee_filter(&img, 3, 0.0).is_err()); // bad ENL
    }

    #[test]
    fn test_refined_lee_homogeneous_unchanged() {
        let img = r(vec![5.0; 49], 7, 7);
        let out = refined_lee_filter(&img, 7, 1.0).unwrap();
        for v in out.data().iter() {
            assert!((v - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_refined_lee_preserves_edge_better_than_classic() {
        // Vertical step: left half = 1, right half = 100. At a pixel on the
        // dark (left) side next to the edge, the refined Lee keeps the dark
        // half-window, so its output stays near 1 — closer to the homogeneous
        // side than the classic Lee, which averages across the bright edge.
        let n = 9;
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = if j < n / 2 { 1.0 } else { 100.0 };
            }
        }
        let img = r(data, n, n);
        let refined = refined_lee_filter(&img, 7, 4.0).unwrap();
        let classic = lee_filter(&img, 7, 4.0).unwrap();
        // pixel just left of the edge (col 3, edge between 3 and 4)
        let (i, j) = (4, 3);
        let rf = refined.get(i, j).unwrap();
        let cl = classic.get(i, j).unwrap();
        assert!(
            (rf - 1.0).abs() < (cl - 1.0).abs(),
            "refined Lee should preserve the dark side: refined={rf} classic={cl}"
        );
        assert!(
            rf < 30.0,
            "refined Lee should stay near the dark side, got {rf}"
        );
    }

    #[test]
    fn test_refined_lee_validation() {
        let img = r(vec![1.0; 9], 3, 3);
        assert!(refined_lee_filter(&img, 2, 1.0).is_err());
        assert!(refined_lee_filter(&img, 1, 1.0).is_err());
        assert!(refined_lee_filter(&img, 3, 0.0).is_err());
    }
}
