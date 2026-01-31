//! Sky View Factor (SVF)
//!
//! Fraction of the sky hemisphere visible from each cell.
//! Values range from 0 (completely enclosed) to 1 (flat, open terrain).
//! Based on horizon angle computation in multiple directions.

use ndarray::Array2;
use rayon::prelude::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Sky View Factor computation
#[derive(Debug, Clone)]
pub struct SvfParams {
    /// Search radius in cells (default 10)
    pub radius: usize,
    /// Number of azimuth directions (default 16)
    pub directions: usize,
}

impl Default for SvfParams {
    fn default() -> Self {
        Self {
            radius: 10,
            directions: 16,
        }
    }
}

/// Compute Sky View Factor
///
/// SVF = 1 - (1/n) × Σ sin²(γᵢ)
/// where γᵢ is the maximum horizon angle in direction i.
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Search radius and number of directions
///
/// # Returns
/// Raster<f64> with SVF values [0, 1]
pub fn sky_view_factor(dem: &Raster<f64>, params: SvfParams) -> Result<Raster<f64>> {
    if params.radius == 0 || params.directions == 0 {
        return Err(Error::Algorithm("Radius and directions must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let n_dirs = params.directions;

    // Precompute direction vectors
    let dir_vectors: Vec<(f64, f64)> = (0..n_dirs)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_dirs as f64;
            (angle.sin(), angle.cos()) // (dc_step, dr_step) in continuous space
        })
        .collect();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                let mut sin_sq_sum = 0.0;

                for &(dc_step, dr_step) in &dir_vectors {
                    let max_angle = compute_horizon_angle(
                        dem, row, col, z0, dr_step, dc_step,
                        params.radius, cell_size, rows, cols,
                    );
                    let sin_angle = max_angle.sin();
                    sin_sq_sum += sin_angle * sin_angle;
                }

                row_data[col] = 1.0 - sin_sq_sum / n_dirs as f64;
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Compute maximum horizon angle in a given direction
pub(crate) fn compute_horizon_angle(
    dem: &Raster<f64>,
    row: usize, col: usize, z0: f64,
    dr_step: f64, dc_step: f64,
    radius: usize, cell_size: f64,
    rows: usize, cols: usize,
) -> f64 {
    let mut max_angle = 0.0_f64;

    for step in 1..=radius {
        let fr = row as f64 + dr_step * step as f64;
        let fc = col as f64 + dc_step * step as f64;

        let nr = fr.round() as isize;
        let nc = fc.round() as isize;

        if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
            break;
        }

        let z = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
        if z.is_nan() {
            break;
        }

        let dist = ((fr - row as f64).powi(2) + (fc - col as f64).powi(2)).sqrt() * cell_size;
        if dist < f64::EPSILON {
            continue;
        }

        let angle = ((z - z0) / dist).atan();
        if angle > max_angle {
            max_angle = angle;
        }
    }

    max_angle.max(0.0) // SVF only considers positive (above-horizon) angles
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_svf_flat() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let result = sky_view_factor(&dem, SvfParams::default()).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!((v - 1.0).abs() < 0.01, "Flat terrain SVF should be ~1.0, got {}", v);
    }

    #[test]
    fn test_svf_pit() {
        // Deep pit → low SVF
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 10.0).unwrap();
            }
        }

        let result = sky_view_factor(&dem, SvfParams::default()).unwrap();
        let center = result.get(10, 10).unwrap();
        let edge = result.get(0, 0).unwrap();
        assert!(center < edge, "Pit center should have lower SVF than edge");
        assert!(center < 0.8, "Pit center SVF should be low, got {}", center);
    }

    #[test]
    fn test_svf_range() {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                dem.set(row, col, (row as f64 * 5.0) + ((col * 3 + row * 7) % 10) as f64).unwrap();
            }
        }

        let result = sky_view_factor(&dem, SvfParams::default()).unwrap();
        for row in 1..20 {
            for col in 1..20 {
                let v = result.get(row, col).unwrap();
                if !v.is_nan() {
                    assert!(v >= 0.0 && v <= 1.0, "SVF must be [0,1], got {} at ({},{})", v, row, col);
                }
            }
        }
    }

    #[test]
    fn test_svf_invalid_params() {
        let dem = Raster::filled(5, 5, 100.0_f64);
        assert!(sky_view_factor(&dem, SvfParams { radius: 0, directions: 16 }).is_err());
        assert!(sky_view_factor(&dem, SvfParams { radius: 5, directions: 0 }).is_err());
    }
}
