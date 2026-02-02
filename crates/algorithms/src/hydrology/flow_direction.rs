//! D8 flow direction algorithm
//!
//! Calculates the direction of flow from each cell to its steepest
//! downslope neighbor using the D8 (deterministic eight-node) method.
//!
//! Flow direction encoding:
//! ```text
//!   4  3  2
//!   5  0  1
//!   6  7  8
//! ```
//! 0 = pit/flat (no outflow), 1-8 = direction to steepest neighbor

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// D8 neighbor offsets: (row_offset, col_offset)
/// Indexed to match the direction encoding (1=E, 2=NE, ..., 8=SE)
const D8_OFFSETS: [(isize, isize); 8] = [
    (0, 1),   // 1: E
    (-1, 1),  // 2: NE
    (-1, 0),  // 3: N
    (-1, -1), // 4: NW
    (0, -1),  // 5: W
    (1, -1),  // 6: SW
    (1, 0),   // 7: S
    (1, 1),   // 8: SE
];

/// Distance factors for each D8 direction
const D8_DIST: [f64; 8] = [
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
    1.0, std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
];

/// Flow direction algorithm (D8)
#[derive(Debug, Clone, Default)]
pub struct FlowDirection;

impl Algorithm for FlowDirection {
    type Input = Raster<f64>;
    type Output = Raster<u8>;
    type Params = ();
    type Error = Error;

    fn name(&self) -> &'static str {
        "Flow Direction (D8)"
    }

    fn description(&self) -> &'static str {
        "Calculate D8 flow direction from a filled DEM"
    }

    fn execute(&self, input: Self::Input, _params: Self::Params) -> Result<Self::Output> {
        flow_direction(&input)
    }
}

/// Calculate D8 flow direction from a DEM.
///
/// The input DEM should ideally be hydrologically conditioned (sinks filled)
/// for meaningful results.
///
/// # Direction Encoding
/// ```text
///   4  3  2
///   5  0  1
///   6  7  8
/// ```
/// - `0` = pit or flat (no downslope neighbor)
/// - `1`-`8` = direction to the steepest downslope neighbor
///
/// # Arguments
/// * `dem` - Input DEM (ideally filled)
///
/// # Returns
/// Raster<u8> with flow direction codes
pub fn flow_direction(dem: &Raster<f64>) -> Result<Raster<u8>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let cell_size = dem.cell_size();

    let output_data: Vec<u8> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![0u8; cols];

            for col in 0..cols {
                let center = unsafe { dem.get_unchecked(row, col) };

                // Skip nodata
                if center.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata {
                    if (center - nd).abs() < f64::EPSILON {
                        continue;
                    }
                }

                let mut max_drop = 0.0_f64;
                let mut best_dir: u8 = 0;

                for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;

                    if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                        continue;
                    }

                    let neighbor = unsafe { dem.get_unchecked(nr as usize, nc as usize) };

                    if neighbor.is_nan() {
                        continue;
                    }
                    if let Some(nd) = nodata {
                        if (neighbor - nd).abs() < f64::EPSILON {
                            continue;
                        }
                    }

                    // Drop = (center - neighbor) / distance
                    let distance = D8_DIST[idx] * cell_size;
                    let drop = (center - neighbor) / distance;

                    if drop > max_drop {
                        max_drop = drop;
                        best_dir = (idx + 1) as u8; // Direction codes are 1-indexed
                    }
                }

                row_data[col] = best_dir;
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_flow_direction_slope_east() {
        // DEM slopes down to the east: elevation = -col
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - col) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let center = fdir.get(2, 2).unwrap();

        // Should flow East (direction 1)
        assert_eq!(center, 1, "Expected flow direction E (1), got {}", center);
    }

    #[test]
    fn test_flow_direction_slope_south() {
        // DEM slopes down to the south: elevation = -row
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let center = fdir.get(2, 2).unwrap();

        // Should flow South (direction 7)
        assert_eq!(center, 7, "Expected flow direction S (7), got {}", center);
    }

    #[test]
    fn test_flow_direction_pit() {
        // Central pit: center is lowest
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, 10.0).unwrap();
            }
        }
        dem.set(2, 2, 1.0).unwrap(); // Pit

        let fdir = flow_direction(&dem).unwrap();
        let center = fdir.get(2, 2).unwrap();

        // Center is a pit, no downslope neighbor → direction 0
        assert_eq!(center, 0, "Expected pit (0), got {}", center);
    }

    #[test]
    fn test_flow_direction_diagonal() {
        // DEM slopes down to the SE: elevation = -(row + col)
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (10 - row - col) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let center = fdir.get(2, 2).unwrap();

        // Should flow SE (direction 8)
        assert_eq!(center, 8, "Expected flow direction SE (8), got {}", center);
    }
}
