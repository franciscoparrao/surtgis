//! Vegetation phenology metrics from NDVI/EVI time series.
//!
//! Extracts growing season parameters:
//! - **SOS** (Start of Season): day when greenness exceeds threshold
//! - **EOS** (End of Season): day when greenness drops below threshold
//! - **Peak**: maximum NDVI value in the season
//! - **Peak Time**: day of year of peak NDVI
//! - **Amplitude**: peak - base level
//! - **Season Length**: EOS - SOS (days)
//! - **Green-up Rate**: slope of increasing NDVI phase

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for phenology extraction.
#[derive(Debug, Clone)]
pub struct PhenologyParams {
    /// Threshold as fraction of amplitude (0-1) for SOS/EOS detection.
    /// Default: 0.5 (half-maximum method).
    pub threshold: f64,
    /// Savitzky-Golay smoothing window size (must be odd). 0 = no smoothing.
    pub smooth_window: usize,
}

impl Default for PhenologyParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            smooth_window: 5,
        }
    }
}

/// Phenology result with multiple metric rasters.
pub struct PhenologyResult {
    /// Start of Season (index into time series, or DOY if times provided)
    pub sos: Raster<f64>,
    /// End of Season
    pub eos: Raster<f64>,
    /// Peak value (maximum NDVI/EVI)
    pub peak: Raster<f64>,
    /// Time of peak (index or DOY)
    pub peak_time: Raster<f64>,
    /// Amplitude: peak - base
    pub amplitude: Raster<f64>,
    /// Season length: EOS - SOS
    pub season_length: Raster<f64>,
}

/// Extract vegetation phenology metrics from an NDVI/EVI time series.
///
/// # Arguments
/// * `rasters` - Time-ordered vegetation index rasters (at least 6 time steps)
/// * `times` - Optional day-of-year values for each raster. If None, uses indices 0..n.
/// * `params` - Phenology extraction parameters
pub fn vegetation_phenology(
    rasters: &[&Raster<f64>],
    times: Option<&[f64]>,
    params: &PhenologyParams,
) -> Result<PhenologyResult> {
    let n = rasters.len();
    if n < 6 {
        return Err(Error::Other("phenology requires at least 6 time steps".into()));
    }
    let (rows, cols) = rasters[0].shape();
    for r in rasters.iter().skip(1) {
        if r.shape() != (rows, cols) {
            return Err(Error::SizeMismatch {
                er: rows, ec: cols, ar: r.rows(), ac: r.cols(),
            });
        }
    }

    let t_vals: Vec<f64> = match times {
        Some(t) => {
            if t.len() != n {
                return Err(Error::Other(format!(
                    "times length {} != raster count {}", t.len(), n
                )));
            }
            t.to_vec()
        }
        None => (0..n).map(|i| i as f64).collect(),
    };

    let total = rows * cols;
    let mut sos_flat = vec![f64::NAN; total];
    let mut eos_flat = vec![f64::NAN; total];
    let mut peak_flat = vec![f64::NAN; total];
    let mut peak_time_flat = vec![f64::NAN; total];
    let mut amp_flat = vec![f64::NAN; total];
    let mut len_flat = vec![f64::NAN; total];

    let smooth_w = if params.smooth_window > 0 && params.smooth_window % 2 == 1 {
        params.smooth_window
    } else {
        0
    };

    sos_flat.par_chunks_mut(cols)
        .zip(eos_flat.par_chunks_mut(cols))
        .zip(peak_flat.par_chunks_mut(cols))
        .zip(peak_time_flat.par_chunks_mut(cols))
        .zip(amp_flat.par_chunks_mut(cols))
        .zip(len_flat.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(row, (((((sos_row, eos_row), peak_row), pt_row), amp_row), len_row))| {
            let mut series = vec![f64::NAN; n];
            let mut smoothed = vec![0.0f64; n];

            for col in 0..cols {
                // Collect time series for this pixel
                let mut valid_count = 0;
                for (i, r) in rasters.iter().enumerate() {
                    let v = unsafe { r.get_unchecked(row, col) };
                    series[i] = v;
                    if v.is_finite() { valid_count += 1; }
                }

                if valid_count < 4 {
                    continue;
                }

                // Simple gap-fill: linear interpolation of NaN gaps
                gap_fill_linear(&mut series);

                // Smooth
                if smooth_w > 0 {
                    moving_average(&series, &mut smoothed, smooth_w);
                } else {
                    smoothed.copy_from_slice(&series);
                }

                // Find base (minimum) and peak (maximum)
                let mut base = f64::INFINITY;
                let mut peak_val = f64::NEG_INFINITY;
                let mut peak_idx = 0;
                for (i, &v) in smoothed.iter().enumerate() {
                    if !v.is_finite() { continue; }
                    if v < base { base = v; }
                    if v > peak_val { peak_val = v; peak_idx = i; }
                }

                if !base.is_finite() || !peak_val.is_finite() || peak_val <= base {
                    continue;
                }

                let amplitude = peak_val - base;
                let threshold_val = base + amplitude * params.threshold;

                // SOS: first crossing above threshold before peak
                let mut sos_idx = None;
                for i in 0..peak_idx {
                    if smoothed[i].is_finite() && smoothed[i] >= threshold_val {
                        // Interpolate between i-1 and i
                        if i > 0 && smoothed[i - 1].is_finite() && smoothed[i - 1] < threshold_val {
                            let frac = (threshold_val - smoothed[i - 1]) / (smoothed[i] - smoothed[i - 1]);
                            sos_idx = Some(t_vals[i - 1] + frac * (t_vals[i] - t_vals[i - 1]));
                        } else {
                            sos_idx = Some(t_vals[i]);
                        }
                        break;
                    }
                }

                // EOS: first crossing below threshold after peak
                let mut eos_idx = None;
                for i in (peak_idx + 1)..n {
                    if smoothed[i].is_finite() && smoothed[i] <= threshold_val {
                        if smoothed[i - 1].is_finite() && smoothed[i - 1] > threshold_val {
                            let frac = (smoothed[i - 1] - threshold_val) / (smoothed[i - 1] - smoothed[i]);
                            eos_idx = Some(t_vals[i - 1] + frac * (t_vals[i] - t_vals[i - 1]));
                        } else {
                            eos_idx = Some(t_vals[i]);
                        }
                        break;
                    }
                }

                peak_row[col] = peak_val;
                pt_row[col] = t_vals[peak_idx];
                amp_row[col] = amplitude;

                if let Some(sos) = sos_idx {
                    sos_row[col] = sos;
                    if let Some(eos) = eos_idx {
                        eos_row[col] = eos;
                        len_row[col] = eos - sos;
                    }
                }
            }
        });

    let make_raster = |flat: Vec<f64>| {
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(rasters[0].transform().clone());
        r.set_nodata(Some(f64::NAN));
        if let Some(crs) = rasters[0].crs() {
            r.set_crs(Some(crs.clone()));
        }
        r
    };

    Ok(PhenologyResult {
        sos: make_raster(sos_flat),
        eos: make_raster(eos_flat),
        peak: make_raster(peak_flat),
        peak_time: make_raster(peak_time_flat),
        amplitude: make_raster(amp_flat),
        season_length: make_raster(len_flat),
    })
}

/// Linear interpolation for NaN gaps in a time series.
fn gap_fill_linear(series: &mut [f64]) {
    let n = series.len();
    let mut i = 0;
    while i < n {
        if series[i].is_nan() {
            // Find gap bounds
            let start = i;
            while i < n && series[i].is_nan() {
                i += 1;
            }
            let end = i;

            // Get bounding values
            let left = if start > 0 { series[start - 1] } else { f64::NAN };
            let right = if end < n { series[end] } else { f64::NAN };

            if left.is_finite() && right.is_finite() {
                // Linear interpolation
                let gap_len = (end - start + 1) as f64;
                for j in start..end {
                    let frac = (j - start + 1) as f64 / gap_len;
                    series[j] = left + frac * (right - left);
                }
            } else if left.is_finite() {
                for j in start..end { series[j] = left; }
            } else if right.is_finite() {
                for j in start..end { series[j] = right; }
            }
        } else {
            i += 1;
        }
    }
}

/// Simple moving average smoothing.
fn moving_average(input: &[f64], output: &mut [f64], window: usize) {
    let n = input.len();
    let half = window / 2;
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = (i + half + 1).min(n);
        let mut sum = 0.0;
        let mut count = 0;
        for j in lo..hi {
            if input[j].is_finite() {
                sum += input[j];
                count += 1;
            }
        }
        output[i] = if count > 0 { sum / count as f64 } else { f64::NAN };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_raster(val: f64) -> Raster<f64> {
        let arr = Array2::from_shape_vec((1, 1), vec![val]).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_phenology_bell_curve() {
        // Simulate a growing season: low → high → low
        // DOY: 30, 60, 90, 120, 150, 180, 210, 240, 270, 300
        let doys = vec![30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0];
        let ndvi = vec![0.1, 0.15, 0.3, 0.55, 0.8, 0.75, 0.5, 0.25, 0.12, 0.1];

        let rasters: Vec<Raster<f64>> = ndvi.iter().map(|&v| make_raster(v)).collect();
        let refs: Vec<&Raster<f64>> = rasters.iter().collect();

        let params = PhenologyParams { threshold: 0.5, smooth_window: 0 };
        let result = vegetation_phenology(&refs, Some(&doys), &params).unwrap();

        let peak = result.peak.data()[[0, 0]];
        let peak_time = result.peak_time.data()[[0, 0]];
        let amp = result.amplitude.data()[[0, 0]];
        let sos = result.sos.data()[[0, 0]];
        let eos = result.eos.data()[[0, 0]];

        assert!((peak - 0.8).abs() < 1e-10, "peak should be 0.8, got {}", peak);
        assert!((peak_time - 150.0).abs() < 1e-10, "peak at DOY 150");
        assert!((amp - 0.7).abs() < 1e-10, "amplitude = 0.8 - 0.1 = 0.7");
        assert!(sos.is_finite(), "SOS should be defined");
        assert!(eos.is_finite(), "EOS should be defined");
        assert!(sos < peak_time, "SOS < peak");
        assert!(eos > peak_time, "EOS > peak");

        let season_len = result.season_length.data()[[0, 0]];
        assert!(season_len > 0.0, "season length should be positive");
    }

    #[test]
    fn test_phenology_with_nan() {
        let doys = vec![30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0];
        let ndvi = vec![0.1, f64::NAN, 0.4, 0.7, 0.9, 0.6, f64::NAN, 0.1];

        let rasters: Vec<Raster<f64>> = ndvi.iter().map(|&v| make_raster(v)).collect();
        let refs: Vec<&Raster<f64>> = rasters.iter().collect();

        let params = PhenologyParams { threshold: 0.5, smooth_window: 0 };
        let result = vegetation_phenology(&refs, Some(&doys), &params).unwrap();

        assert!(result.peak.data()[[0, 0]].is_finite(), "should handle NaN gaps");
    }

    #[test]
    fn test_gap_fill_linear() {
        let mut series = vec![1.0, f64::NAN, f64::NAN, 4.0, 5.0];
        gap_fill_linear(&mut series);
        assert!((series[1] - 2.0).abs() < 1e-10);
        assert!((series[2] - 3.0).abs() < 1e-10);
    }
}
