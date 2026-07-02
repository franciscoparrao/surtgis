//! Hillshade (shaded relief) calculation
//!
//! Creates a shaded relief visualization from a DEM based on
//! illumination angle and direction.

use crate::maybe_rayon::*;
use ndarray::Array2;
use std::f64::consts::PI;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

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
            azimuth: 315.0, // NW illumination (standard)
            altitude: 45.0, // 45° above horizon
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

    let data = dem.data();

    // Process rows in parallel
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                // Skip edges first
                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    *row_data_col = f64::NAN;
                    continue;
                }

                // Get center value
                let e = data[[row, col]];
                if e.is_nan() || (nodata.is_some() && (e - nodata.unwrap()).abs() < f64::EPSILON) {
                    *row_data_col = f64::NAN;
                    continue;
                }

                // Get 3x3 neighborhood
                let a = data[[row - 1, col - 1]];
                let b = data[[row - 1, col]];
                let c = data[[row - 1, col + 1]];
                let d = data[[row, col - 1]];
                let f = data[[row, col + 1]];
                let g = data[[row + 1, col - 1]];
                let h = data[[row + 1, col]];
                let i = data[[row + 1, col + 1]];

                // Check for nodata
                if [a, b, c, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    *row_data_col = f64::NAN;
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
                    // Downslope direction in the (east, north) frame is
                    // (-dz_dx, +dz_dy): dz_dy is computed south-minus-north,
                    // so it is already the northward component of downslope.
                    let aspect = dz_dy.atan2(-dz_dx);
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
                let shade_clamped = shade.clamp(0.0, 1.0);

                *row_data_col = if params.normalized {
                    shade_clamped
                } else {
                    (shade_clamped * 255.0).round()
                };
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    // NaN, not 0.0: a shade of 0 is a valid value (full shadow).
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming hillshade calculator implementing `WindowAlgorithm`.
///
/// Processes a DEM strip-by-strip with bounded memory.
/// Uses the same Horn (1981) 3×3 method as `hillshade()`.
#[derive(Debug, Clone)]
pub struct HillshadeStreaming {
    /// Sun azimuth in degrees (0 = North, clockwise)
    pub azimuth: f64,
    /// Sun altitude in degrees above horizon (0-90)
    pub altitude: f64,
    /// Z-factor for vertical exaggeration
    pub z_factor: f64,
}

impl Default for HillshadeStreaming {
    fn default() -> Self {
        Self {
            azimuth: 315.0,
            altitude: 45.0,
            z_factor: 1.0,
        }
    }
}

impl surtgis_core::WindowAlgorithm for HillshadeStreaming {
    fn kernel_radius(&self) -> usize {
        1 // 3×3 kernel
    }

    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        _cell_size_y: f64,
    ) {
        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        // Pre-compute illumination angles (same convention as hillshade())
        let azimuth_rad = (360.0 - self.azimuth + 90.0).to_radians();
        let zenith_rad = (90.0 - self.altitude).to_radians();
        let cos_zenith = zenith_rad.cos();
        let sin_zenith = zenith_rad.sin();

        let cell_size = cell_size_x * self.z_factor;
        let eight_cs = 8.0 * cell_size;

        for r in 0..out_rows {
            let ir = r + radius; // input row corresponding to output row r
            if ir == 0 || ir >= in_rows - 1 {
                // Edge row — fill with NaN
                for c in 0..cols {
                    output[[r, c]] = f64::NAN;
                }
                continue;
            }

            for c in 0..cols {
                if c == 0 || c >= cols - 1 {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let e = input[[ir, c]];
                if e.is_nan() || nodata.map_or(false, |nd| (e - nd).abs() < f64::EPSILON) {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let a = input[[ir - 1, c - 1]];
                let b = input[[ir - 1, c]];
                let cv = input[[ir - 1, c + 1]];
                let d = input[[ir, c - 1]];
                let f = input[[ir, c + 1]];
                let g = input[[ir + 1, c - 1]];
                let h = input[[ir + 1, c]];
                let i = input[[ir + 1, c + 1]];

                if [a, b, cv, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                // Horn's method for gradients
                let dz_dx = ((cv + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_cs;
                let dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + cv)) / eight_cs;

                // Calculate slope and aspect
                let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();

                let aspect_rad = if dz_dx.abs() < 1e-10 && dz_dy.abs() < 1e-10 {
                    0.0 // Flat
                } else {
                    // Downslope in (east, north): dz_dy is south-minus-north,
                    // already the northward component (see hillshade()).
                    let asp = dz_dy.atan2(-dz_dx);
                    if asp < 0.0 { 2.0 * PI + asp } else { asp }
                };

                // Hillshade formula
                let shade = cos_zenith * slope_rad.cos()
                    + sin_zenith * slope_rad.sin() * (azimuth_rad - aspect_rad).cos();

                output[[r, c]] = (shade.clamp(0.0, 1.0) * 255.0).round();
            }
        }
    }
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

        // All valid values should be in [0, 255]; edges are NaN (nodata)
        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                if row == 0 || row == result.rows() - 1 || col == 0 || col == result.cols() - 1 {
                    assert!(val.is_nan(), "Edge should be NaN at ({}, {})", row, col);
                    continue;
                }
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

    /// Regression test for the N–S mirrored illumination bug.
    ///
    /// A uniform north-facing slope (elevation increasing southward) lit
    /// from the NW (az=315, alt=45) must be bright. Verified against
    /// `gdaldem hillshade`: expected 218 for slope tan = 0.5; the mirrored
    /// formula produced 104. The south-facing slope must be darker.
    #[test]
    fn test_hillshade_directional_north_vs_south() {
        let n = 20;
        let cell = 10.0;
        // North-facing: z increases with row (rows go south in a north-up raster)
        let mut north = Raster::new(n, n);
        north.set_transform(surtgis_core::GeoTransform::new(
            0.0,
            n as f64 * cell,
            cell,
            -cell,
        ));
        // South-facing: z decreases with row
        let mut south = Raster::new(n, n);
        south.set_transform(surtgis_core::GeoTransform::new(
            0.0,
            n as f64 * cell,
            cell,
            -cell,
        ));
        for row in 0..n {
            for col in 0..n {
                north.set(row, col, row as f64 * cell * 0.5).unwrap();
                south
                    .set(row, col, (n - 1 - row) as f64 * cell * 0.5)
                    .unwrap();
            }
        }

        let params = HillshadeParams::default(); // az 315, alt 45
        let hn = hillshade(&north, params.clone()).unwrap();
        let hs = hillshade(&south, params).unwrap();

        let vn = hn.get(n / 2, n / 2).unwrap();
        let vs = hs.get(n / 2, n / 2).unwrap();

        // GDAL gives 218 for the north-facing slope with these parameters
        assert!(
            (vn - 218.0).abs() <= 1.0,
            "North-facing slope with NW sun should be ~218 (GDAL), got {}",
            vn
        );
        // North-facing must be brighter than south-facing under NW sun
        assert!(
            vn > vs + 50.0,
            "North-facing ({}) must be brighter than south-facing ({}) under NW sun",
            vn,
            vs
        );
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

        // All valid values should be in [0, 1]; edges are NaN
        for row in 0..result.rows() {
            for col in 0..result.cols() {
                let val = result.get(row, col).unwrap();
                if val.is_nan() {
                    continue;
                }
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Normalized hillshade {} out of range",
                    val
                );
            }
        }
    }
}
