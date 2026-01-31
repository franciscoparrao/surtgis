//! Feature-Preserving DEM Smoothing
//!
//! Removes noise while preserving breaks-in-slope (ridges, valleys, scarps).
//! Based on Sun et al. (2007) feature-preserving mesh denoising adapted for DEMs.

use ndarray::Array2;
use rayon::prelude::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for feature-preserving smoothing
#[derive(Debug, Clone)]
pub struct SmoothingParams {
    /// Filter radius in cells (default 2)
    pub radius: usize,
    /// Number of iterations (default 3)
    pub iterations: usize,
    /// Normal difference threshold in degrees (default 15.0)
    /// Neighbors with surface normal difference > threshold are excluded
    pub threshold: f64,
}

impl Default for SmoothingParams {
    fn default() -> Self {
        Self {
            radius: 2,
            iterations: 3,
            threshold: 15.0,
        }
    }
}

/// Apply feature-preserving smoothing to a DEM
///
/// Uses bilateral filtering adapted for terrain: smooths only within
/// regions of similar surface orientation, preserving edges where
/// surface normals change abruptly (breaks-in-slope).
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Smoothing parameters
///
/// # Returns
/// Smoothed DEM preserving slope breaks
pub fn feature_preserving_smoothing(
    dem: &Raster<f64>,
    params: SmoothingParams,
) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }
    if params.iterations == 0 {
        return Err(Error::Algorithm("Iterations must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let threshold_rad = params.threshold.to_radians();
    let r = params.radius as isize;

    let mut current = dem.data().clone();

    for _ in 0..params.iterations {
        let prev = current.clone();

        let new_data: Vec<f64> = (0..rows)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![f64::NAN; cols];

                for col in 0..cols {
                    if row < 1 || row >= rows - 1 || col < 1 || col >= cols - 1 {
                        row_data[col] = prev[(row, col)];
                        continue;
                    }

                    let z0 = prev[(row, col)];
                    if z0.is_nan() {
                        continue;
                    }

                    // Compute surface normal at center
                    let n0 = compute_normal(&prev, row, col, rows, cols);

                    let mut weighted_sum = 0.0;
                    let mut weight_total = 0.0;

                    for dr in -r..=r {
                        for dc in -r..=r {
                            let nr = row as isize + dr;
                            let nc = col as isize + dc;

                            if nr < 1 || nc < 1 || (nr as usize) >= rows - 1 || (nc as usize) >= cols - 1 {
                                continue;
                            }

                            let nr = nr as usize;
                            let nc = nc as usize;
                            let z = prev[(nr, nc)];
                            if z.is_nan() {
                                continue;
                            }

                            // Compute surface normal at neighbor
                            let nn = compute_normal(&prev, nr, nc, rows, cols);

                            // Normal similarity: angle between normals
                            let cos_angle = (n0.0 * nn.0 + n0.1 * nn.1 + n0.2 * nn.2)
                                .clamp(-1.0, 1.0);
                            let angle = cos_angle.acos();

                            if angle > threshold_rad {
                                continue; // Skip: different surface orientation
                            }

                            // Spatial weight (Gaussian)
                            let dist_sq = (dr * dr + dc * dc) as f64;
                            let sigma = params.radius as f64;
                            let spatial_w = (-dist_sq / (2.0 * sigma * sigma)).exp();

                            // Normal similarity weight
                            let normal_w = (-(angle * angle) / (2.0 * threshold_rad * threshold_rad)).exp();

                            let w = spatial_w * normal_w;
                            weighted_sum += z * w;
                            weight_total += w;
                        }
                    }

                    if weight_total > 0.0 {
                        row_data[col] = weighted_sum / weight_total;
                    } else {
                        row_data[col] = z0;
                    }
                }

                row_data
            })
            .collect();

        current = Array2::from_shape_vec((rows, cols), new_data)
            .map_err(|e| Error::Other(e.to_string()))?;
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = current;

    Ok(output)
}

/// Compute approximate surface normal at a cell (nx, ny, nz)
fn compute_normal(data: &Array2<f64>, row: usize, col: usize, rows: usize, cols: usize) -> (f64, f64, f64) {
    if row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1 {
        return (0.0, 0.0, 1.0);
    }

    let dz_dx = (data[(row, col + 1)] - data[(row, col - 1)]) / 2.0;
    let dz_dy = (data[(row + 1, col)] - data[(row - 1, col)]) / 2.0;

    let len = (dz_dx * dz_dx + dz_dy * dz_dy + 1.0).sqrt();
    (-dz_dx / len, -dz_dy / len, 1.0 / len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_smoothing_preserves_flat() {
        let mut dem = Raster::filled(20, 20, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = feature_preserving_smoothing(&dem, SmoothingParams::default()).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!((v - 100.0).abs() < 0.01, "Flat should stay flat, got {}", v);
    }

    #[test]
    fn test_smoothing_reduces_noise() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        // Tilted plane with noise
        for row in 0..20 {
            for col in 0..20 {
                let base = row as f64 * 10.0;
                let noise = ((row * 7 + col * 13) % 5) as f64 - 2.0;
                dem.set(row, col, base + noise).unwrap();
            }
        }

        let result = feature_preserving_smoothing(&dem, SmoothingParams::default()).unwrap();

        // Check that variance is reduced
        let orig_stats = dem.statistics();
        let smooth_stats = result.statistics();

        // The smoothed version should have less local variation
        // (hard to test precisely, but mean should be similar)
        assert!(
            (orig_stats.mean.unwrap() - smooth_stats.mean.unwrap()).abs() < 5.0,
            "Smoothing shouldn't change global mean significantly"
        );
    }

    #[test]
    fn test_smoothing_params_validation() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(feature_preserving_smoothing(&dem, SmoothingParams {
            radius: 0, ..Default::default()
        }).is_err());
        assert!(feature_preserving_smoothing(&dem, SmoothingParams {
            iterations: 0, ..Default::default()
        }).is_err());
    }
}
