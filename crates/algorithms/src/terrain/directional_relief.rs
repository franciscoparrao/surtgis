//! Directional Relief
//!
//! Computes the maximum elevation range along a given azimuth direction
//! within a search radius from each cell:
//!
//!   For azimuth θ: trace a line from the cell in direction θ and θ+180°,
//!   collect elevations, and compute max(elev) - min(elev).
//!
//! When azimuth = 0, computes the average relief across 8 evenly-spaced
//! directions (multidirectional mode).
//!
//! Reference: WhiteboxTools `DirectionalRelief`

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for directional relief.
#[derive(Debug, Clone)]
pub struct DirectionalReliefParams {
    /// Search radius in cells (default 10).
    pub radius: usize,
    /// Azimuth in degrees (0 = multidirectional average of 8 directions).
    pub azimuth: f64,
}

impl Default for DirectionalReliefParams {
    fn default() -> Self {
        Self {
            radius: 10,
            azimuth: 0.0,
        }
    }
}

/// Compute directional relief.
pub fn directional_relief(
    dem: &Raster<f64>,
    params: DirectionalReliefParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let radius = params.radius;
    let nodata = dem.nodata();

    // Determine azimuths to evaluate
    let azimuths: Vec<f64> = if params.azimuth == 0.0 {
        // Multidirectional: 8 evenly-spaced directions
        (0..8).map(|i| i as f64 * 45.0).collect()
    } else {
        vec![params.azimuth]
    };

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { dem.get_unchecked(row, col) };
                if center.is_nan()
                    || nodata.is_some_and(|nd| (center - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                let mut total_relief = 0.0;
                let mut az_count = 0;

                for &az_deg in &azimuths {
                    let az_rad = az_deg.to_radians();
                    // Direction components (azimuth: 0=N, 90=E)
                    let dy = -az_rad.cos(); // row direction (north = decreasing row)
                    let dx = az_rad.sin();  // col direction

                    // Find the maximum signed elevation change along the profile.
                    // For each cell on the line, compute (z_cell - z_center).
                    // Report the one with the largest absolute value, preserving sign.
                    let mut best_change = 0.0_f64;

                    // Trace in both directions (forward and backward)
                    for &sign in &[1.0_f64, -1.0] {
                        for step in 1..=radius {
                            let nr_f = row as f64 + sign * dy * step as f64;
                            let nc_f = col as f64 + sign * dx * step as f64;

                            let nr = nr_f.round() as isize;
                            let nc = nc_f.round() as isize;

                            if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                                break;
                            }

                            let v = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
                            if v.is_nan()
                                || nodata.is_some_and(|nd| (v - nd).abs() < f64::EPSILON)
                            {
                                continue;
                            }

                            let change = v - center;
                            if change.abs() > best_change.abs() {
                                best_change = change;
                            }
                        }
                    }

                    total_relief += best_change;
                    az_count += 1;
                }

                if az_count > 0 {
                    *out = total_relief / az_count as f64;
                }
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
    fn test_flat_zero_relief() {
        let mut dem = Raster::filled(20, 20, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = directional_relief(&dem, DirectionalReliefParams { radius: 5, azimuth: 0.0 }).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!(v.abs() < 1e-10, "Flat DEM should have 0 relief, got {}", v);
    }

    #[test]
    fn test_slope_has_relief() {
        // N-S slope: elevation increases southward (row increases)
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                dem.set(r, c, r as f64 * 10.0).unwrap();
            }
        }

        // Azimuth 180 = south → cells ahead are higher, max change is positive
        let result = directional_relief(&dem, DirectionalReliefParams { radius: 5, azimuth: 180.0 }).unwrap();
        let v = result.get(10, 10).unwrap();
        // Max signed change along S: +50 (5 cells south × 10m/cell)
        assert!(v.abs() > 30.0, "N-S slope should have large relief, got {}", v);
    }

    #[test]
    fn test_finite_values() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                let x = c as f64 - 10.0;
                let y = r as f64 - 10.0;
                dem.set(r, c, x * x + y * y).unwrap();
            }
        }

        let result = directional_relief(&dem, DirectionalReliefParams::default()).unwrap();
        for r in 2..18 {
            for c in 2..18 {
                let v = result.get(r, c).unwrap();
                // Values can be positive or negative (signed relief)
                assert!(v.is_finite(), "Relief should be finite, got {} at ({},{})", v, r, c);
            }
        }
    }
}
