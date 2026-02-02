//! Hillshade (shaded relief) calculation
//!
//! Creates a shaded relief visualization from a DEM based on
//! illumination angle and direction.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};
use std::f64::consts::PI;

/// Parameters for hillshade calculation
#[derive(Debug, Clone)]
pub struct HillshadeParams {
    /// Sun azimuth in degrees (0 = North, clockwise)
    pub azimuth: f64,
    /// Sun altitude in degrees above horizon (0-90)
    pub altitude: f64,
    /// Z-factor for vertical exaggeration
    pub z_factor: f64,
    /// Output range: false = 0-255, true = 0.0-1.0
    pub normalized: bool,
}

impl Default for HillshadeParams {
    fn default() -> Self {
        Self {
            azimuth: 315.0,   // NW illumination (standard)
            altitude: 45.0,   // 45° above horizon
            z_factor: 1.0,
            normalized: false,
        }
    }
}

/// Hillshade algorithm
#[derive(Debug, Clone, Default)]
pub struct Hillshade;

impl Algorithm for Hillshade {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = HillshadeParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Hillshade"
    }

    fn description(&self) -> &'static str {
        "Calculate shaded relief from a DEM"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        hillshade(&input, params)
    }
}

/// Calculate hillshade from a DEM
///
/// Uses the standard algorithm based on slope, aspect, and illumination geometry.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Hillshade parameters (azimuth, altitude, z-factor)
///
/// # Returns
/// Raster with hillshade values (0-255 or 0.0-1.0)
pub fn hillshade(dem: &Raster<f64>, params: HillshadeParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size() * params.z_factor;
    let nodata = dem.nodata();

    // Pre-compute illumination angles in radians
    let azimuth_rad = (360.0 - params.azimuth + 90.0).to_radians();
    let zenith_rad = (90.0 - params.altitude).to_radians();
    let cos_zenith = zenith_rad.cos();
    let sin_zenith = zenith_rad.sin();

    let eight_cell_size = 8.0 * cell_size;

    // Process rows in parallel
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![0.0; cols];

            for col in 0..cols {
                // Get center value
                let e = unsafe { dem.get_unchecked(row, col) };
                if e.is_nan() || (nodata.is_some() && (e - nodata.unwrap()).abs() < f64::EPSILON) {
                    row_data[col] = 0.0;
                    continue;
                }

                // Skip edges
                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    row_data[col] = 0.0;
                    continue;
                }

                // Get 3x3 neighborhood
                let a = unsafe { dem.get_unchecked(row - 1, col - 1) };
                let b = unsafe { dem.get_unchecked(row - 1, col) };
                let c = unsafe { dem.get_unchecked(row - 1, col + 1) };
                let d = unsafe { dem.get_unchecked(row, col - 1) };
                let f = unsafe { dem.get_unchecked(row, col + 1) };
                let g = unsafe { dem.get_unchecked(row + 1, col - 1) };
                let h = unsafe { dem.get_unchecked(row + 1, col) };
                let i = unsafe { dem.get_unchecked(row + 1, col + 1) };

                // Check for nodata
                if [a, b, c, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    row_data[col] = 0.0;
                    continue;
                }

                // Horn's method for gradients
                let dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_cell_size;
                let dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / eight_cell_size;

                // Calculate slope and aspect
                let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();

                let aspect_rad = if dz_dx.abs() < 1e-10 && dz_dy.abs() < 1e-10 {
                    0.0 // Flat
                } else {
                    let aspect = (-dz_dy).atan2(-dz_dx);
                    if aspect < 0.0 {
                        2.0 * PI + aspect
                    } else {
                        aspect
                    }
                };

                // Hillshade formula
                // shade = cos(zenith) * cos(slope) + sin(zenith) * sin(slope) * cos(azimuth - aspect)
                let shade = cos_zenith * slope_rad.cos()
                    + sin_zenith * slope_rad.sin() * (azimuth_rad - aspect_rad).cos();

                // Clamp to [0, 1]
                let shade_clamped = shade.max(0.0).min(1.0);

                row_data[col] = if params.normalized {
                    shade_clamped
                } else {
                    (shade_clamped * 255.0).round()
                };
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(0.0));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dem() -> Raster<f64> {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64 * 10.0).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_hillshade_range() {
        let dem = create_test_dem();
        let result = hillshade(&dem, HillshadeParams::default()).unwrap();

        // All values should be in [0, 255]
        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                assert!(
                    val >= 0.0 && val <= 255.0,
                    "Hillshade value {} out of range at ({}, {})",
                    val,
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_hillshade_flat() {
        let mut dem: Raster<f64> = Raster::filled(10, 10, 100.0);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let params = HillshadeParams {
            altitude: 45.0,
            ..Default::default()
        };

        let result = hillshade(&dem, params).unwrap();
        let val = result.get(5, 5).unwrap();

        // Flat surface at 45° altitude should have shade ≈ cos(45°) ≈ 0.707 → ~180
        assert!(
            (val - 180.0).abs() < 20.0,
            "Expected ~180 for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_hillshade_normalized() {
        let dem = create_test_dem();
        let params = HillshadeParams {
            normalized: true,
            ..Default::default()
        };

        let result = hillshade(&dem, params).unwrap();

        // All values should be in [0, 1]
        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Normalized hillshade {} out of range",
                    val
                );
            }
        }
    }
}
