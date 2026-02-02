//! Multidirectional hillshade
//!
//! Computes a weighted blend of hillshade from multiple illumination azimuths,
//! reducing the directional bias inherent in single-azimuth hillshade.
//!
//! Uses the USGS method (Mark 1992) with 6 azimuth directions and
//! aspect-dependent weighting so that each pixel is primarily shaded from
//! the direction most oblique to its local slope.
//!
//! Reference: Mark, R.K. (1992) "A multidirectional, oblique-weighted,
//! shaded-relief image of the Island of Hawaii" (USGS Open-File Report 92-422)

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};
use std::f64::consts::PI;

/// Parameters for multidirectional hillshade
#[derive(Debug, Clone)]
pub struct MultiHillshadeParams {
    /// Sun altitude in degrees above horizon (0-90)
    pub altitude: f64,
    /// Z-factor for vertical exaggeration
    pub z_factor: f64,
    /// Output range: false = 0-255, true = 0.0-1.0
    pub normalized: bool,
}

impl Default for MultiHillshadeParams {
    fn default() -> Self {
        Self {
            altitude: 45.0,
            z_factor: 1.0,
            normalized: false,
        }
    }
}

/// Calculate multidirectional hillshade
///
/// Blends hillshade from 6 azimuths (0°, 60°, 120°, 180°, 240°, 300°)
/// using aspect-dependent weights. Each pixel is shaded primarily from
/// the direction most oblique to its local slope direction.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Multidirectional hillshade parameters
///
/// # Returns
/// Raster with hillshade values (0-255 or 0.0-1.0)
pub fn multidirectional_hillshade(
    dem: &Raster<f64>,
    params: MultiHillshadeParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size() * params.z_factor;
    let nodata = dem.nodata();

    let zenith_rad = (90.0 - params.altitude).to_radians();
    let cos_zenith = zenith_rad.cos();
    let sin_zenith = zenith_rad.sin();
    let eight_cs = 8.0 * cell_size;

    // 6 equally-spaced azimuths converted to the mathematical convention
    let azimuths_rad: Vec<f64> = (0..6)
        .map(|i| (360.0 - (i as f64 * 60.0) + 90.0).to_radians())
        .collect();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![0.0; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let e = unsafe { dem.get_unchecked(row, col) };
                if e.is_nan() || nodata.is_some_and(|nd| (e - nd).abs() < f64::EPSILON) {
                    continue;
                }

                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    continue;
                }

                let a = unsafe { dem.get_unchecked(row - 1, col - 1) };
                let b = unsafe { dem.get_unchecked(row - 1, col) };
                let c = unsafe { dem.get_unchecked(row - 1, col + 1) };
                let d = unsafe { dem.get_unchecked(row, col - 1) };
                let f = unsafe { dem.get_unchecked(row, col + 1) };
                let g = unsafe { dem.get_unchecked(row + 1, col - 1) };
                let h = unsafe { dem.get_unchecked(row + 1, col) };
                let i = unsafe { dem.get_unchecked(row + 1, col + 1) };

                if [a, b, c, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    continue;
                }

                // Horn's method gradients
                let dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_cs;
                let dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / eight_cs;

                let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
                let cos_slope = slope_rad.cos();
                let sin_slope = slope_rad.sin();

                let aspect_rad = if dz_dx.abs() < 1e-10 && dz_dy.abs() < 1e-10 {
                    0.0
                } else {
                    let asp = (-dz_dy).atan2(-dz_dx);
                    if asp < 0.0 { 2.0 * PI + asp } else { asp }
                };

                // Compute weighted blend of hillshade from each azimuth
                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;

                for az in &azimuths_rad {
                    let shade = cos_zenith * cos_slope
                        + sin_zenith * sin_slope * (az - aspect_rad).cos();
                    let shade = shade.max(0.0);

                    // Weight: highest when azimuth is perpendicular to aspect
                    // w = 1 + cos²(azimuth - aspect + π/2)
                    let angle_diff = az - aspect_rad + PI / 2.0;
                    let w = 1.0 + angle_diff.cos().powi(2);

                    weighted_sum += shade * w;
                    weight_total += w;
                }

                let shade_val = if weight_total > 0.0 {
                    (weighted_sum / weight_total).min(1.0)
                } else {
                    0.0
                };

                *row_data_col = if params.normalized {
                    shade_val
                } else {
                    (shade_val * 255.0).round()
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
    use surtgis_core::GeoTransform;

    fn test_dem() -> Raster<f64> {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64 * 10.0).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_multihillshade_range() {
        let dem = test_dem();
        let result = multidirectional_hillshade(&dem, MultiHillshadeParams::default()).unwrap();

        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                assert!(
                    val >= 0.0 && val <= 255.0,
                    "Value {} out of range at ({}, {})",
                    val,
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_multihillshade_flat() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = multidirectional_hillshade(&dem, MultiHillshadeParams::default()).unwrap();
        let val = result.get(5, 5).unwrap();

        // Flat surface at 45° altitude → shade ≈ cos(45°) ≈ 0.707 → ~180
        assert!(
            (val - 180.0).abs() < 20.0,
            "Expected ~180 for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_multihillshade_less_directional_bias() {
        // Multi-hillshade should produce more uniform shading than single
        let dem = test_dem();
        let multi = multidirectional_hillshade(
            &dem,
            MultiHillshadeParams {
                normalized: true,
                ..Default::default()
            },
        )
        .unwrap();

        // Interior values should all be > 0 (no completely dark faces)
        let mut all_positive = true;
        for row in 1..9 {
            for col in 1..9 {
                let val = multi.get(row, col).unwrap();
                if val <= 0.0 {
                    all_positive = false;
                }
            }
        }
        assert!(all_positive, "Multi-hillshade should avoid fully dark pixels");
    }
}
