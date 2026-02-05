//! Multi-Resolution Valley Bottom Flatness (MRVBF)
//!
//! Gallant & Dowling (2003): Identifies valley bottoms at multiple scales.
//! Also computes MRRTF (Multi-Resolution Ridge Top Flatness).
//!
//! The algorithm iteratively smooths the DEM at increasing scales and
//! classifies cells based on slope and elevation percentile thresholds.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for MRVBF
#[derive(Debug, Clone)]
pub struct MrvbfParams {
    /// Initial slope threshold in percent (default 16%)
    pub initial_slope_threshold: f64,
    /// Slope reduction factor per step (default 2.0)
    pub slope_reduction: f64,
    /// Number of resolution steps (default 3)
    pub steps: usize,
    /// Shape parameter for flatness transformation (default 4.0)
    pub shape: f64,
}

impl Default for MrvbfParams {
    fn default() -> Self {
        Self {
            initial_slope_threshold: 16.0,
            slope_reduction: 2.0,
            steps: 3,
            shape: 4.0,
        }
    }
}

/// Compute MRVBF and MRRTF
///
/// Returns (MRVBF, MRRTF) rasters.
///
/// MRVBF values: 0 = not valley bottom, increasing values = larger/flatter valleys.
/// MRRTF values: 0 = not ridge top, increasing values = larger/flatter ridges.
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - MRVBF parameters
pub fn mrvbf(dem: &Raster<f64>, params: MrvbfParams) -> Result<(Raster<f64>, Raster<f64>)> {
    if params.steps == 0 {
        return Err(Error::Algorithm("Steps must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();

    let mut vbf = Array2::from_elem((rows, cols), 0.0_f64);
    let mut rtf = Array2::from_elem((rows, cols), 0.0_f64);

    let mut current_dem = dem.data().clone();

    for step in 0..params.steps {
        let threshold = params.initial_slope_threshold / params.slope_reduction.powi(step as i32);

        // 1. Compute slope at current resolution
        let slope_pct = compute_slope_percent(&current_dem, rows, cols, cell_size);

        // 2. Compute elevation percentile (local lowness/highness)
        let radius = (3 * (1 << step)).min(rows.min(cols) / 4);
        let (low_pctl, high_pctl) = compute_percentiles(&current_dem, rows, cols, radius);

        // 3. Flatness classification
        let step_vbf: Vec<f64> = (0..rows)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![0.0; cols];
                for col in 0..cols {
                    let slp = slope_pct[(row, col)];
                    let pctl = low_pctl[(row, col)];

                    if slp.is_nan() || pctl.is_nan() {
                        continue;
                    }

                    // Flatness from slope: 1/(1 + (slope/threshold)^shape)
                    let flatness = 1.0 / (1.0 + (slp / threshold).powf(params.shape));

                    // Lowness: transforms percentile so that low cells (valleys) score high
                    let lowness = 1.0 - pctl;

                    // Valley bottom flatness = min(flatness, lowness)
                    row_data[col] = flatness.min(lowness);
                }
                row_data
            })
            .collect();

        let step_rtf: Vec<f64> = (0..rows)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![0.0; cols];
                for col in 0..cols {
                    let slp = slope_pct[(row, col)];
                    let pctl = high_pctl[(row, col)];

                    if slp.is_nan() || pctl.is_nan() {
                        continue;
                    }

                    let flatness = 1.0 / (1.0 + (slp / threshold).powf(params.shape));
                    let highness = pctl;

                    row_data[col] = flatness.min(highness);
                }
                row_data
            })
            .collect();

        // 4. Combine with previous steps
        let scale_weight = (step + 1) as f64;
        for row in 0..rows {
            for col in 0..cols {
                let v = step_vbf[row * cols + col];
                if v > 0.5 {
                    vbf[(row, col)] = vbf[(row, col)].max(scale_weight - 0.5 + v);
                }
                let r = step_rtf[row * cols + col];
                if r > 0.5 {
                    rtf[(row, col)] = rtf[(row, col)].max(scale_weight - 0.5 + r);
                }
            }
        }

        // 5. Smooth DEM for next step (3×3 mean filter, applied multiple times)
        current_dem = smooth_dem(&current_dem, rows, cols);
    }

    let mut vbf_raster = dem.with_same_meta::<f64>(rows, cols);
    vbf_raster.set_nodata(Some(f64::NAN));
    *vbf_raster.data_mut() = vbf;

    let mut rtf_raster = dem.with_same_meta::<f64>(rows, cols);
    rtf_raster.set_nodata(Some(f64::NAN));
    *rtf_raster.data_mut() = rtf;

    Ok((vbf_raster, rtf_raster))
}

fn compute_slope_percent(data: &Array2<f64>, rows: usize, cols: usize, cs: f64) -> Array2<f64> {
    let mut slope = Array2::from_elem((rows, cols), f64::NAN);
    let eight_cs = 8.0 * cs;

    for row in 1..rows - 1 {
        for col in 1..cols - 1 {
            let a = data[(row - 1, col - 1)];
            let b = data[(row - 1, col)];
            let c = data[(row - 1, col + 1)];
            let d = data[(row, col - 1)];
            let f = data[(row, col + 1)];
            let g = data[(row + 1, col - 1)];
            let h = data[(row + 1, col)];
            let i = data[(row + 1, col + 1)];

            if [a, b, c, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                continue;
            }

            let dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_cs;
            let dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / eight_cs;

            slope[(row, col)] = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt() * 100.0;
        }
    }
    slope
}

fn compute_percentiles(
    data: &Array2<f64>,
    rows: usize, cols: usize,
    radius: usize,
) -> (Array2<f64>, Array2<f64>) {
    let mut low = Array2::from_elem((rows, cols), f64::NAN);
    let mut high = Array2::from_elem((rows, cols), f64::NAN);
    let r = radius as isize;

    for row in 0..rows {
        for col in 0..cols {
            let z0 = data[(row, col)];
            if z0.is_nan() { continue; }

            let mut lower = 0;
            let mut total = 0;

            for dr in -r..=r {
                for dc in -r..=r {
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;
                    if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                        continue;
                    }
                    let z = data[(nr as usize, nc as usize)];
                    if z.is_nan() { continue; }
                    total += 1;
                    if z < z0 { lower += 1; }
                }
            }

            if total > 0 {
                let pct = lower as f64 / total as f64;
                low[(row, col)] = pct;
                high[(row, col)] = pct;
            }
        }
    }

    (low, high)
}

fn smooth_dem(data: &Array2<f64>, rows: usize, cols: usize) -> Array2<f64> {
    let mut smoothed = data.clone();

    for row in 1..rows - 1 {
        for col in 1..cols - 1 {
            let mut sum = 0.0;
            let mut count = 0;
            for dr in -1_isize..=1 {
                for dc in -1_isize..=1 {
                    let v = data[((row as isize + dr) as usize, (col as isize + dc) as usize)];
                    if !v.is_nan() {
                        sum += v;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                smoothed[(row, col)] = sum / count as f64;
            }
        }
    }

    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_mrvbf_valley() {
        let mut dem = Raster::new(30, 30);
        dem.set_transform(GeoTransform::new(0.0, 30.0, 10.0, -10.0));

        // Create a V-shaped valley
        for row in 0..30 {
            for col in 0..30 {
                let dist_from_center = (col as f64 - 15.0).abs();
                dem.set(row, col, dist_from_center * 10.0).unwrap();
            }
        }

        let (vbf, _rtf) = mrvbf(&dem, MrvbfParams::default()).unwrap();
        let valley_bottom = vbf.get(15, 15).unwrap();
        let slope_side = vbf.get(15, 25).unwrap();

        assert!(
            valley_bottom > slope_side,
            "Valley bottom should have higher MRVBF: {} vs {}",
            valley_bottom, slope_side
        );
    }

    #[test]
    fn test_mrvbf_ridge() {
        let mut dem = Raster::new(30, 30);
        dem.set_transform(GeoTransform::new(0.0, 30.0, 10.0, -10.0));

        // Create an inverted V (ridge)
        for row in 0..30 {
            for col in 0..30 {
                let dist_from_center = (col as f64 - 15.0).abs();
                dem.set(row, col, 100.0 - dist_from_center * 10.0).unwrap();
            }
        }

        let (_vbf, rtf) = mrvbf(&dem, MrvbfParams::default()).unwrap();
        let ridge_top = rtf.get(15, 15).unwrap();
        let slope_side = rtf.get(15, 25).unwrap();

        assert!(
            ridge_top > slope_side,
            "Ridge top should have higher MRRTF: {} vs {}",
            ridge_top, slope_side
        );
    }

    #[test]
    fn test_mrvbf_steps_zero() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(mrvbf(&dem, MrvbfParams { steps: 0, ..Default::default() }).is_err());
    }
}
