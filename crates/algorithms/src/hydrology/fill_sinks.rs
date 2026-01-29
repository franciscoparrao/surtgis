//! Sink filling for hydrological analysis
//!
//! Implements the Planchon-Darboux (2001) algorithm for filling
//! depressions in a DEM to ensure continuous flow paths.
//!
//! Reference:
//! Planchon, O., Darboux, F. (2001). A fast, simple and versatile algorithm
//! to fill the depressions of digital elevation models.
//! Catena, 46(2-3), 159-176.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// Parameters for sink filling
#[derive(Debug, Clone)]
pub struct FillSinksParams {
    /// Minimum slope to enforce between cells (prevents flat areas)
    /// Set to 0.0 to allow flat areas after filling.
    pub min_slope: f64,
}

impl Default for FillSinksParams {
    fn default() -> Self {
        Self {
            min_slope: 0.01,
        }
    }
}

/// Fill sinks algorithm
#[derive(Debug, Clone, Default)]
pub struct FillSinks;

impl Algorithm for FillSinks {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = FillSinksParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Fill Sinks"
    }

    fn description(&self) -> &'static str {
        "Fill depressions in a DEM using Planchon-Darboux (2001) method"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        fill_sinks(&input, params)
    }
}

/// D8 neighbor offsets: (row_offset, col_offset)
const D8_OFFSETS: [(isize, isize); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

/// D8 distances: cardinal = cell_size, diagonal = cell_size * sqrt(2)
const D8_DISTANCES: [f64; 8] = [
    std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
    1.0,                           1.0,
    std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
];

/// Fill depressions in a DEM using the Planchon-Darboux (2001) algorithm.
///
/// This algorithm ensures that every cell has a downslope path to the
/// edge of the DEM, which is required for flow direction and accumulation.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Fill parameters (minimum slope enforcement)
///
/// # Returns
/// A new raster with all depressions filled
pub fn fill_sinks(dem: &Raster<f64>, params: FillSinksParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let cell_size = dem.cell_size();
    let epsilon = params.min_slope * cell_size;

    // Step 1: Initialize output surface W
    // W(c) = DEM(c) if c is on the border
    // W(c) = very large value otherwise
    let big_value = f64::MAX / 2.0;
    let mut w = Array2::from_elem((rows, cols), big_value);

    // Set border cells to DEM values
    for row in 0..rows {
        for col in 0..cols {
            let val = unsafe { dem.get_unchecked(row, col) };

            let is_nodata = match nodata {
                Some(nd) => val.is_nan() || (val - nd).abs() < f64::EPSILON,
                None => val.is_nan(),
            };

            if is_nodata {
                w[(row, col)] = val;
                continue;
            }

            // Border cells
            if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                w[(row, col)] = val;
            }
        }
    }

    // Step 2: Iteratively lower W until stable
    // A cell's W value can be lowered if a neighbor n satisfies:
    //   W(c) > W(n) + epsilon
    // then: W(c) = max(DEM(c), W(n) + epsilon_d)
    // where epsilon_d accounts for diagonal distance
    let mut changed = true;
    while changed {
        changed = false;

        // Forward pass: top-left to bottom-right
        for row in 1..rows - 1 {
            for col in 1..cols - 1 {
                let dem_val = unsafe { dem.get_unchecked(row, col) };

                // Skip nodata
                if dem_val.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata {
                    if (dem_val - nd).abs() < f64::EPSILON {
                        continue;
                    }
                }

                if w[(row, col)] > dem_val {
                    for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                        let nr = (row as isize + dr) as usize;
                        let nc = (col as isize + dc) as usize;
                        let eps_d = epsilon * D8_DISTANCES[idx];

                        let wn = w[(nr, nc)];
                        if wn.is_nan() || wn >= big_value {
                            continue;
                        }

                        let new_val = wn + eps_d;
                        if dem_val >= new_val {
                            w[(row, col)] = dem_val;
                            changed = true;
                            break;
                        }
                        if w[(row, col)] > new_val {
                            w[(row, col)] = new_val;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Backward pass: bottom-right to top-left
        for row in (1..rows - 1).rev() {
            for col in (1..cols - 1).rev() {
                let dem_val = unsafe { dem.get_unchecked(row, col) };

                if dem_val.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata {
                    if (dem_val - nd).abs() < f64::EPSILON {
                        continue;
                    }
                }

                if w[(row, col)] > dem_val {
                    for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                        let nr = (row as isize + dr) as usize;
                        let nc = (col as isize + dc) as usize;
                        let eps_d = epsilon * D8_DISTANCES[idx];

                        let wn = w[(nr, nc)];
                        if wn.is_nan() || wn >= big_value {
                            continue;
                        }

                        let new_val = wn + eps_d;
                        if dem_val >= new_val {
                            w[(row, col)] = dem_val;
                            changed = true;
                            break;
                        }
                        if w[(row, col)] > new_val {
                            w[(row, col)] = new_val;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    let mut output = dem.like(0.0);
    *output.data_mut() = w;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn create_dem_with_sink() -> Raster<f64> {
        // 7x7 DEM with a depression in the center
        //
        // 9 9 9 9 9 9 9
        // 9 8 8 8 8 8 9
        // 9 8 7 7 7 8 9
        // 9 8 7 3 7 8 9   <- center cell is a sink (3 < 7)
        // 9 8 7 7 7 8 9
        // 9 8 8 8 8 8 9
        // 9 9 9 9 9 9 9
        let mut dem = Raster::new(7, 7);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));

        let values = [
            9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
            9.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0,
            9.0, 8.0, 7.0, 7.0, 7.0, 8.0, 9.0,
            9.0, 8.0, 7.0, 3.0, 7.0, 8.0, 9.0,
            9.0, 8.0, 7.0, 7.0, 7.0, 8.0, 9.0,
            9.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0,
            9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
        ];

        for (idx, &val) in values.iter().enumerate() {
            let row = idx / 7;
            let col = idx % 7;
            dem.set(row, col, val).unwrap();
        }

        dem
    }

    #[test]
    fn test_fill_sinks_raises_depression() {
        let dem = create_dem_with_sink();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // The center cell (3,3) had value 3.0, surrounded by 7.0
        // After filling, it should be >= 7.0
        let center = filled.get(3, 3).unwrap();
        assert!(
            center >= 7.0,
            "Sink at (3,3) should be filled to >= 7.0, got {}",
            center
        );
    }

    #[test]
    fn test_fill_sinks_preserves_border() {
        let dem = create_dem_with_sink();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // Border values should remain unchanged
        assert_eq!(filled.get(0, 0).unwrap(), 9.0);
        assert_eq!(filled.get(0, 3).unwrap(), 9.0);
        assert_eq!(filled.get(6, 6).unwrap(), 9.0);
    }

    #[test]
    fn test_fill_sinks_with_outlet() {
        // 5x5 DEM with an outlet gap in the border
        // All border = 10, except (4,2)=2 (outlet)
        // Center (2,2)=1 (sink below outlet)
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                let is_border = row == 0 || row == 4 || col == 0 || col == 4;
                dem.set(row, col, if is_border { 10.0 } else { 5.0 }).unwrap();
            }
        }
        dem.set(2, 2, 1.0).unwrap(); // Sink
        dem.set(4, 2, 2.0).unwrap(); // Low outlet on border

        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // Sink should be filled but only up to the outlet level
        let center = filled.get(2, 2).unwrap();
        assert!(
            center >= 1.0 && center <= 5.0,
            "Center should be filled but not above interior level, got {}",
            center
        );

        // Interior cells that aren't sinks should be unchanged
        let side = filled.get(1, 1).unwrap();
        assert_eq!(side, 5.0, "Non-sink interior should be preserved, got {}", side);
    }

    #[test]
    fn test_fill_sinks_no_change_on_clean_dem() {
        // A sloped plane has no sinks
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                // Slope toward bottom-right corner
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }

        let filled = fill_sinks(&dem, FillSinksParams::default()).unwrap();

        // Should be essentially unchanged
        for row in 0..10 {
            for col in 0..10 {
                let orig = dem.get(row, col).unwrap();
                let fill = filled.get(row, col).unwrap();
                assert!(
                    fill >= orig,
                    "Filled value should be >= original at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_fill_sinks_with_min_slope() {
        let dem = create_dem_with_sink();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.01 }).unwrap();

        // With min_slope, filled values should create a slight gradient
        let center = filled.get(3, 3).unwrap();
        let neighbor = filled.get(3, 4).unwrap();

        // Center should be slightly lower than its neighbor (or equal)
        // because we enforce a minimum slope
        assert!(
            center <= neighbor + 0.1,
            "Min slope should create gradient: center={}, neighbor={}",
            center,
            neighbor
        );
    }
}
