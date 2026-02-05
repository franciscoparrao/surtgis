//! Watershed delineation algorithm
//!
//! Delineates watersheds (drainage basins) from a D8 flow direction raster.
//! Supports two modes:
//! - From pour points: delineate specific catchments
//! - All basins: label all independent drainage basins

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};
use std::collections::VecDeque;

/// D8 neighbor offsets matching flow direction encoding (1=E, 2=NE, ..., 8=SE)
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

/// Get the opposite direction index for D8
fn opposite_dir(dir: u8) -> u8 {
    if dir == 0 { return 0; }
    ((dir - 1 + 4) % 8) + 1
}

/// Parameters for watershed delineation
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct WatershedParams {
    /// Pour points as (row, col) coordinates.
    /// If empty, all independent basins are delineated.
    pub pour_points: Vec<(usize, usize)>,
}


/// Watershed delineation algorithm
#[derive(Debug, Clone, Default)]
pub struct Watershed;

impl Algorithm for Watershed {
    type Input = Raster<u8>;
    type Output = Raster<i32>;
    type Params = WatershedParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Watershed"
    }

    fn description(&self) -> &'static str {
        "Delineate watersheds from D8 flow direction"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        watershed(&input, params)
    }
}

/// Delineate watersheds from a D8 flow direction raster.
///
/// # Modes
///
/// - **Pour points provided**: Each pour point defines a basin. All cells
///   upstream of each pour point are labeled with the basin ID (1-indexed).
///
/// - **No pour points**: All outlet cells (cells that flow off the grid or
///   into pits) become basin seeds, and the entire grid is partitioned.
///
/// # Arguments
/// * `flow_dir` - D8 flow direction raster
/// * `params` - Watershed parameters (pour points)
///
/// # Returns
/// Raster<i32> with basin labels (0 = unassigned/nodata)
pub fn watershed(flow_dir: &Raster<u8>, params: WatershedParams) -> Result<Raster<i32>> {
    let (rows, cols) = flow_dir.shape();
    let mut basins = Array2::<i32>::zeros((rows, cols));
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    if params.pour_points.is_empty() {
        // Mode: delineate all independent basins
        // Find all outlet cells (flow off grid or into pits)
        let mut basin_id: i32 = 0;

        for row in 0..rows {
            for col in 0..cols {
                let dir = unsafe { flow_dir.get_unchecked(row, col) };

                let is_outlet = if dir == 0 {
                    // Pit or nodata
                    true
                } else {
                    let dir_idx = (dir - 1) as usize;
                    if dir_idx >= D8_OFFSETS.len() {
                        true
                    } else {
                        let (dr, dc) = D8_OFFSETS[dir_idx];
                        let nr = row as isize + dr;
                        let nc = col as isize + dc;
                        // Flows off grid
                        nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize
                    }
                };

                if is_outlet {
                    basin_id += 1;
                    basins[(row, col)] = basin_id;
                    queue.push_back((row, col));
                }
            }
        }
    } else {
        // Mode: delineate from specific pour points
        for (id, &(row, col)) in params.pour_points.iter().enumerate() {
            if row < rows && col < cols {
                let basin_id = (id + 1) as i32;
                basins[(row, col)] = basin_id;
                queue.push_back((row, col));
            }
        }
    }

    // Trace upstream from each seed cell using BFS
    // A cell flows INTO (row, col) if its flow direction points to (row, col)
    while let Some((row, col)) = queue.pop_front() {
        let basin_id = basins[(row, col)];

        // Check all 8 neighbors
        for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
            let nr = row as isize + dr;
            let nc = col as isize + dc;

            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                continue;
            }

            let nr = nr as usize;
            let nc = nc as usize;

            // Skip if already assigned
            if basins[(nr, nc)] != 0 {
                continue;
            }

            // Check if this neighbor flows INTO (row, col)
            let neighbor_dir = unsafe { flow_dir.get_unchecked(nr, nc) };

            if neighbor_dir == 0 {
                continue;
            }

            // The neighbor at direction idx is at offset D8_OFFSETS[idx] from (row,col).
            // For it to flow into (row,col), its flow direction must be the opposite.
            let expected_dir = opposite_dir((idx + 1) as u8);

            if neighbor_dir == expected_dir {
                basins[(nr, nc)] = basin_id;
                queue.push_back((nr, nc));
            }
        }
    }

    let mut output = flow_dir.with_same_meta::<i32>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = basins;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrology::flow_direction::flow_direction;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_watershed_single_basin() {
        // 5x5 DEM sloping south - should be one basin
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let basins = watershed(&fdir, WatershedParams::default()).unwrap();

        // All cells should be assigned to some basin
        for row in 0..5 {
            for col in 0..5 {
                let basin = basins.get(row, col).unwrap();
                assert!(basin > 0, "Cell ({}, {}) should be in a basin", row, col);
            }
        }
    }

    #[test]
    fn test_watershed_two_basins() {
        // DEM with a ridge in the middle: left half flows left, right half flows right
        let mut dem = Raster::new(5, 7);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..7 {
                // Ridge at col=3, slopes away on both sides
                let dist_from_ridge = (col as f64 - 3.0).abs();
                dem.set(row, col, 10.0 - dist_from_ridge).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let basins = watershed(&fdir, WatershedParams::default()).unwrap();

        // Left and right sides should have different basin IDs
        let left = basins.get(2, 0).unwrap();
        let right = basins.get(2, 6).unwrap();

        assert!(left > 0 && right > 0, "Both sides should have basins");
        assert_ne!(
            left, right,
            "Left ({}) and right ({}) should be different basins",
            left, right
        );
    }

    #[test]
    fn test_watershed_from_pour_point() {
        // 5x5 DEM sloping south
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let basins = watershed(
            &fdir,
            WatershedParams {
                pour_points: vec![(4, 2)], // Bottom center
            },
        )
        .unwrap();

        // The pour point should be in basin 1
        let pp = basins.get(4, 2).unwrap();
        assert_eq!(pp, 1, "Pour point should be basin 1");

        // Cells upstream should also be basin 1
        let upstream = basins.get(2, 2).unwrap();
        assert_eq!(upstream, 1, "Upstream cell should be in same basin");
    }

    #[test]
    fn test_opposite_direction() {
        assert_eq!(opposite_dir(1), 5); // E → W
        assert_eq!(opposite_dir(3), 7); // N → S
        assert_eq!(opposite_dir(5), 1); // W → E
        assert_eq!(opposite_dir(7), 3); // S → N
        assert_eq!(opposite_dir(2), 6); // NE → SW
        assert_eq!(opposite_dir(8), 4); // SE → NW
    }
}
