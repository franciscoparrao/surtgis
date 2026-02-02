//! Slope calculation from DEMs
//!
//! Calculates the rate of change of elevation using the Horn (1981) method,
//! which uses a 3x3 neighborhood to compute partial derivatives.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// Units for slope output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlopeUnits {
    /// Degrees (0-90)
    #[default]
    Degrees,
    /// Percent (0-infinity, typically 0-100+)
    Percent,
    /// Radians (0-π/2)
    Radians,
}

/// Parameters for slope calculation
#[derive(Debug, Clone)]
pub struct SlopeParams {
    /// Output units
    pub units: SlopeUnits,
    /// Z-factor for unit conversion (default 1.0)
    /// Use ~111320 for lat/lon DEMs with meters elevation
    pub z_factor: f64,
}

impl Default for SlopeParams {
    fn default() -> Self {
        Self {
            units: SlopeUnits::Degrees,
            z_factor: 1.0,
        }
    }
}

/// Slope algorithm
#[derive(Debug, Clone, Default)]
pub struct Slope;

impl Algorithm for Slope {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = SlopeParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Slope"
    }

    fn description(&self) -> &'static str {
        "Calculate slope (rate of change of elevation) from a DEM using Horn's method"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        slope(&input, params)
    }
}

/// Calculate slope from a DEM
///
/// Uses Horn's (1981) method with a 3x3 neighborhood:
/// ```text
/// a b c
/// d e f
/// g h i
/// ```
///
/// dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cellsize)
/// dz/dy = ((g + 2h + i) - (a + 2b + c)) / (8 * cellsize)
/// slope = atan(sqrt(dz/dx² + dz/dy²))
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Slope calculation parameters
///
/// # Returns
/// Raster with slope values in the specified units
pub fn slope(dem: &Raster<f64>, params: SlopeParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size() * params.z_factor;
    let nodata = dem.nodata();

    // Pre-compute constants
    let eight_cell_size = 8.0 * cell_size;

    // Process rows in parallel
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                // Get center value
                let e = unsafe { dem.get_unchecked(row, col) };
                if e.is_nan() || (nodata.is_some() && (e - nodata.unwrap()).abs() < f64::EPSILON) {
                    continue;
                }

                // Skip edges (need full 3x3 neighborhood)
                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    row_data[col] = f64::NAN;
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

                // Check for nodata in neighborhood
                if [a, b, c, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    row_data[col] = f64::NAN;
                    continue;
                }

                // Horn's method
                let dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / eight_cell_size;
                let dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / eight_cell_size;

                let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();

                row_data[col] = match params.units {
                    SlopeUnits::Degrees => slope_rad.to_degrees(),
                    SlopeUnits::Percent => slope_rad.tan() * 100.0,
                    SlopeUnits::Radians => slope_rad,
                };
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

    fn create_test_dem() -> Raster<f64> {
        // Create a simple tilted plane: z = x + y
        let mut dem = Raster::new(10, 10);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_slope_flat() {
        let mut dem: Raster<f64> = Raster::filled(10, 10, 100.0);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = slope(&dem, SlopeParams::default()).unwrap();

        // Interior cells should have zero slope
        let val = result.get(5, 5).unwrap();
        assert!(
            val.abs() < 0.001,
            "Expected ~0 slope for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_slope_tilted() {
        let dem = create_test_dem();
        let result = slope(&dem, SlopeParams::default()).unwrap();

        // All interior cells should have the same slope (constant gradient)
        let val1 = result.get(3, 3).unwrap();
        let val2 = result.get(5, 5).unwrap();

        assert!(
            (val1 - val2).abs() < 0.001,
            "Expected uniform slope, got {} vs {}",
            val1,
            val2
        );
    }

    #[test]
    fn test_slope_units() {
        let dem = create_test_dem();

        let deg = slope(&dem, SlopeParams { units: SlopeUnits::Degrees, z_factor: 1.0 }).unwrap();
        let rad = slope(&dem, SlopeParams { units: SlopeUnits::Radians, z_factor: 1.0 }).unwrap();
        let pct = slope(&dem, SlopeParams { units: SlopeUnits::Percent, z_factor: 1.0 }).unwrap();

        let deg_val = deg.get(5, 5).unwrap();
        let rad_val = rad.get(5, 5).unwrap();
        let pct_val = pct.get(5, 5).unwrap();

        // Verify unit conversions
        assert!(
            (deg_val - rad_val.to_degrees()).abs() < 0.001,
            "Degree/radian mismatch"
        );
        assert!(
            (pct_val - rad_val.tan() * 100.0).abs() < 0.001,
            "Percent mismatch"
        );
    }
}
