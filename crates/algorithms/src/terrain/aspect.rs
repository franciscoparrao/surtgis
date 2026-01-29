//! Aspect calculation from DEMs
//!
//! Calculates the direction of the steepest slope using the Horn (1981) method.

use ndarray::Array2;
use rayon::prelude::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};
use std::f64::consts::PI;

/// Output format for aspect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AspectOutput {
    /// Degrees (0-360, 0=North, clockwise)
    #[default]
    Degrees,
    /// Radians (0-2π)
    Radians,
    /// Compass direction (N, NE, E, SE, S, SW, W, NW) as 1-8
    Compass,
}

/// Aspect algorithm
#[derive(Debug, Clone, Default)]
pub struct Aspect;

impl Algorithm for Aspect {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = AspectOutput;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Aspect"
    }

    fn description(&self) -> &'static str {
        "Calculate aspect (direction of steepest descent) from a DEM"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        aspect(&input, params)
    }
}

/// Calculate aspect from a DEM
///
/// Uses Horn's (1981) method. Aspect is measured clockwise from north:
/// - 0° (or 360°) = North
/// - 90° = East
/// - 180° = South
/// - 270° = West
///
/// Flat areas (slope = 0) are assigned -1 (nodata).
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `output_format` - Format for output values
///
/// # Returns
/// Raster with aspect values
pub fn aspect(dem: &Raster<f64>, output_format: AspectOutput) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Threshold for considering a surface flat
    const FLAT_THRESHOLD: f64 = 1e-10;

    // Process rows in parallel
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![-1.0; cols];

            for col in 0..cols {
                // Get center value
                let e = unsafe { dem.get_unchecked(row, col) };
                if e.is_nan() || (nodata.is_some() && (e - nodata.unwrap()).abs() < f64::EPSILON) {
                    row_data[col] = -1.0;
                    continue;
                }

                // Skip edges
                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    row_data[col] = -1.0;
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
                    row_data[col] = -1.0;
                    continue;
                }

                // Horn's method for gradients
                let dz_dx = (c + 2.0 * f + i) - (a + 2.0 * d + g);
                let dz_dy = (g + 2.0 * h + i) - (a + 2.0 * b + c);

                // Check for flat area
                if dz_dx.abs() < FLAT_THRESHOLD && dz_dy.abs() < FLAT_THRESHOLD {
                    row_data[col] = -1.0;
                    continue;
                }

                // Compute aspect as compass bearing (0=North, clockwise)
                //
                // In pixel space:
                //   dz_dx > 0 → elevation increases eastward
                //   dz_dy > 0 → elevation increases with row (= southward in north-up image)
                //
                // Descent direction in geographic (east, north) space:
                //   east component  = -dz_dx
                //   north component = dz_dy  (inverted because pixel Y opposes geo Y)
                //
                // Compass bearing = atan2(east, north)
                let aspect_north = (-dz_dx).atan2(dz_dy);
                let aspect_north = if aspect_north < 0.0 {
                    aspect_north + 2.0 * PI
                } else {
                    aspect_north
                };

                row_data[col] = match output_format {
                    AspectOutput::Degrees => aspect_north.to_degrees(),
                    AspectOutput::Radians => aspect_north,
                    AspectOutput::Compass => {
                        // Convert to 8-direction compass (1-8)
                        let deg = aspect_north.to_degrees();
                        if deg < 22.5 || deg >= 337.5 {
                            1.0 // N
                        } else if deg < 67.5 {
                            2.0 // NE
                        } else if deg < 112.5 {
                            3.0 // E
                        } else if deg < 157.5 {
                            4.0 // SE
                        } else if deg < 202.5 {
                            5.0 // S
                        } else if deg < 247.5 {
                            6.0 // SW
                        } else if deg < 292.5 {
                            7.0 // W
                        } else {
                            8.0 // NW
                        }
                    }
                };
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(-1.0));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_north_slope() {
        // Create a DEM that slopes down to the north
        let mut dem: Raster<f64> = Raster::new(10, 10);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                // Higher in south (high row), lower in north (low row)
                dem.set(row, col, row as f64).unwrap();
            }
        }

        let result = aspect(&dem, AspectOutput::Degrees).unwrap();
        let val = result.get(5, 5).unwrap();

        // Should be close to 0° (North)
        assert!(
            val < 10.0 || val > 350.0,
            "Expected aspect ~0° (North), got {}°",
            val
        );
    }

    #[test]
    fn test_aspect_east_slope() {
        // Create a DEM that slopes down to the east
        let mut dem: Raster<f64> = Raster::new(10, 10);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                // Higher in west (low col), lower in east (high col)
                dem.set(row, col, -(col as f64)).unwrap();
            }
        }

        let result = aspect(&dem, AspectOutput::Degrees).unwrap();
        let val = result.get(5, 5).unwrap();

        // Should be close to 90° (East)
        assert!(
            (val - 90.0).abs() < 10.0,
            "Expected aspect ~90° (East), got {}°",
            val
        );
    }

    #[test]
    fn test_aspect_flat() {
        let mut dem: Raster<f64> = Raster::filled(10, 10, 100.0);
        dem.set_transform(surtgis_core::GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = aspect(&dem, AspectOutput::Degrees).unwrap();
        let val = result.get(5, 5).unwrap();

        // Flat areas should return -1
        assert_eq!(val, -1.0, "Expected -1 for flat area, got {}", val);
    }
}
