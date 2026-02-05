//! Wind Exposure (Topex)
//!
//! Measures exposure or shelter from wind based on topography.
//! Topex = sum of angles to horizon in specified directions.
//! Based on the Forestry Commission (UK) topographic exposure method.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for wind exposure
#[derive(Debug, Clone)]
pub struct WindExposureParams {
    /// Search radius in cells (default 30, ~300m at 10m resolution)
    pub radius: usize,
    /// Number of azimuth directions (default 8)
    pub directions: usize,
    /// Prevailing wind direction in degrees from north (None = all directions)
    pub wind_direction: Option<f64>,
    /// Angular window around wind direction in degrees (default 45)
    pub wind_window: f64,
}

impl Default for WindExposureParams {
    fn default() -> Self {
        Self {
            radius: 30,
            directions: 8,
            wind_direction: None,
            wind_window: 45.0,
        }
    }
}

/// Compute wind exposure (Topex) index
///
/// For each cell, measures the sum of horizon angles in specified directions.
/// Positive values indicate exposed locations; negative values indicate sheltered.
///
/// The Topex value is: Σ(horizon_angle) over directions
/// where horizon_angle > 0 means higher terrain (shelter)
/// and horizon_angle < 0 means the horizon is below horizontal (exposure).
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Search parameters
///
/// # Returns
/// Raster with Topex values in degrees. Negative = exposed, positive = sheltered.
pub fn wind_exposure(dem: &Raster<f64>, params: WindExposureParams) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let n_dirs = params.directions;

    // Build direction vectors, optionally filtered by wind direction
    let dir_vectors: Vec<(f64, f64, f64)> = (0..n_dirs)
        .filter_map(|i| {
            let azimuth_deg = 360.0 * i as f64 / n_dirs as f64;
            let azimuth_rad = azimuth_deg.to_radians();

            if let Some(wind_dir) = params.wind_direction {
                let mut diff = (azimuth_deg - wind_dir).abs();
                if diff > 180.0 { diff = 360.0 - diff; }
                if diff > params.wind_window {
                    return None;
                }
            }

            // Azimuth: 0=N, 90=E, etc.
            let dc = azimuth_rad.sin();  // East component
            let dr = -azimuth_rad.cos(); // North component (negative row = north)
            Some((dr, dc, azimuth_deg))
        })
        .collect();

    if dir_vectors.is_empty() {
        return Err(Error::Algorithm("No directions selected for wind exposure".into()));
    }

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                let mut topex_sum = 0.0;

                for &(dr_step, dc_step, _) in &dir_vectors {
                    let mut max_angle = f64::NEG_INFINITY;

                    for step in 1..=params.radius {
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
                        if dist < f64::EPSILON { continue; }

                        let angle = ((z - z0) / dist).atan().to_degrees();
                        if angle > max_angle {
                            max_angle = angle;
                        }
                    }

                    if max_angle > f64::NEG_INFINITY {
                        topex_sum += max_angle;
                    }
                }

                *row_data_col = topex_sum;
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

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_wind_exposure_flat() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let result = wind_exposure(&dem, WindExposureParams::default()).unwrap();
        let v = result.get(10, 10).unwrap();
        // Flat terrain: all horizon angles = 0 → topex = 0
        assert!(v.abs() < 0.1, "Flat terrain topex should be ~0, got {}", v);
    }

    #[test]
    fn test_wind_exposure_valley() {
        // Valley: center lower than surroundings → sheltered (positive topex)
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, (dx * dx + dy * dy).sqrt() * 5.0).unwrap();
            }
        }

        let result = wind_exposure(&dem, WindExposureParams::default()).unwrap();
        let center = result.get(10, 10).unwrap();
        assert!(center > 0.0, "Valley center should be sheltered (positive), got {}", center);
    }

    #[test]
    fn test_wind_exposure_peak() {
        // Peak: center higher → exposed (negative/zero topex)
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, 100.0 - (dx * dx + dy * dy).sqrt() * 5.0).unwrap();
            }
        }

        let result = wind_exposure(&dem, WindExposureParams::default()).unwrap();
        let center = result.get(10, 10).unwrap();
        assert!(center <= 0.1, "Peak should be exposed (≤0 topex), got {}", center);
    }

    #[test]
    fn test_wind_direction_filter() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let result = wind_exposure(&dem, WindExposureParams {
            wind_direction: Some(270.0), // West wind
            wind_window: 45.0,
            ..Default::default()
        }).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!(!v.is_nan());
    }
}
