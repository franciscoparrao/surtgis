//! Terrain Openness (Yokoyama et al. 2002)
//!
//! Positive openness: mean of zenith angles to horizon in all directions.
//! Negative openness: mean of nadir angles in all directions.
//! Both expressed in degrees (0-180).

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for openness computation
#[derive(Debug, Clone)]
pub struct OpennessParams {
    /// Search radius in cells (default 10)
    pub radius: usize,
    /// Number of azimuth directions (default 8)
    pub directions: usize,
}

impl Default for OpennessParams {
    fn default() -> Self {
        Self {
            radius: 10,
            directions: 8,
        }
    }
}

/// Compute positive terrain openness
///
/// For each cell, the positive openness is the mean of (90° - max_horizon_angle)
/// over all directions. A flat open area has openness ≈ 90°.
///
/// # Returns
/// Raster with openness values in degrees
pub fn positive_openness(dem: &Raster<f64>, params: OpennessParams) -> Result<Raster<f64>> {
    compute_openness(dem, params, true)
}

/// Compute negative terrain openness
///
/// Negative openness uses the maximum angle looking DOWN in each direction.
///
/// # Returns
/// Raster with negative openness values in degrees
pub fn negative_openness(dem: &Raster<f64>, params: OpennessParams) -> Result<Raster<f64>> {
    compute_openness(dem, params, false)
}

fn compute_openness(
    dem: &Raster<f64>,
    params: OpennessParams,
    positive: bool,
) -> Result<Raster<f64>> {
    if params.radius == 0 || params.directions == 0 {
        return Err(Error::Algorithm("Radius and directions must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let n_dirs = params.directions;

    let dir_vectors: Vec<(f64, f64)> = (0..n_dirs)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n_dirs as f64;
            (angle.sin(), angle.cos())
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

                let mut angle_sum = 0.0;

                for &(dc_step, dr_step) in &dir_vectors {
                    let angle = if positive {
                        compute_positive_angle(dem, row, col, z0, dr_step, dc_step,
                                               params.radius, cell_size, rows, cols)
                    } else {
                        compute_negative_angle(dem, row, col, z0, dr_step, dc_step,
                                               params.radius, cell_size, rows, cols)
                    };
                    angle_sum += angle;
                }

                row_data[col] = angle_sum / n_dirs as f64;
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

/// Positive openness angle for one direction: 90° - max_up_angle
fn compute_positive_angle(
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
        if z.is_nan() { break; }

        let dist = ((fr - row as f64).powi(2) + (fc - col as f64).powi(2)).sqrt() * cell_size;
        if dist < f64::EPSILON { continue; }

        let angle = ((z - z0) / dist).atan();
        if angle > max_angle {
            max_angle = angle;
        }
    }

    (90.0_f64.to_radians() - max_angle.max(0.0)).to_degrees()
}

/// Negative openness angle for one direction: 90° - max_down_angle
fn compute_negative_angle(
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
        if z.is_nan() { break; }

        let dist = ((fr - row as f64).powi(2) + (fc - col as f64).powi(2)).sqrt() * cell_size;
        if dist < f64::EPSILON { continue; }

        let angle = ((z0 - z) / dist).atan(); // Looking DOWN
        if angle > max_angle {
            max_angle = angle;
        }
    }

    (90.0_f64.to_radians() - max_angle.max(0.0)).to_degrees()
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_positive_openness_flat() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let result = positive_openness(&dem, OpennessParams::default()).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!((v - 90.0).abs() < 1.0, "Flat openness should be ~90°, got {}", v);
    }

    #[test]
    fn test_positive_openness_pit() {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 10.0).unwrap();
            }
        }

        let result = positive_openness(&dem, OpennessParams::default()).unwrap();
        let center = result.get(10, 10).unwrap();
        assert!(center < 80.0, "Pit should have lower positive openness, got {}", center);
    }

    #[test]
    fn test_negative_openness_peak() {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, 100.0 - (dx * dx + dy * dy).sqrt() * 5.0).unwrap();
            }
        }

        let result = negative_openness(&dem, OpennessParams::default()).unwrap();
        let center = result.get(10, 10).unwrap();
        assert!(center < 80.0, "Peak should have lower negative openness, got {}", center);
    }
}
