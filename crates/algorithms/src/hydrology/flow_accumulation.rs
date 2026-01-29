//! Flow accumulation algorithm
//!
//! Calculates the number of upstream cells flowing into each cell
//! based on D8 flow direction. This represents the upstream
//! contributing area (in cell counts).

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

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

/// Flow accumulation algorithm
#[derive(Debug, Clone, Default)]
pub struct FlowAccumulation;

impl Algorithm for FlowAccumulation {
    type Input = Raster<u8>;
    type Output = Raster<f64>;
    type Params = ();
    type Error = Error;

    fn name(&self) -> &'static str {
        "Flow Accumulation"
    }

    fn description(&self) -> &'static str {
        "Calculate upstream contributing area from D8 flow direction"
    }

    fn execute(&self, input: Self::Input, _params: Self::Params) -> Result<Self::Output> {
        flow_accumulation(&input)
    }
}

/// Calculate flow accumulation from a D8 flow direction raster.
///
/// Each cell receives a count of all upstream cells that flow into it.
/// Headwater cells (no upstream neighbors) have accumulation = 0.
///
/// # Algorithm
/// 1. Count incoming flows for each cell (in-degree)
/// 2. Start from cells with in-degree 0 (headwaters)
/// 3. Propagate downstream, accumulating counts
///
/// # Arguments
/// * `flow_dir` - D8 flow direction raster (output from `flow_direction`)
///
/// # Returns
/// Raster<f64> with flow accumulation values
pub fn flow_accumulation(flow_dir: &Raster<u8>) -> Result<Raster<f64>> {
    let (rows, cols) = flow_dir.shape();

    // Step 1: Build in-degree count (how many cells flow INTO each cell)
    let mut in_degree = Array2::<u32>::zeros((rows, cols));

    for row in 0..rows {
        for col in 0..cols {
            let dir = unsafe { flow_dir.get_unchecked(row, col) };

            if dir == 0 {
                continue; // Pit or nodata
            }

            let dir_idx = (dir - 1) as usize;
            if dir_idx >= D8_OFFSETS.len() {
                continue;
            }

            let (dr, dc) = D8_OFFSETS[dir_idx];
            let nr = row as isize + dr;
            let nc = col as isize + dc;

            if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                in_degree[(nr as usize, nc as usize)] += 1;
            }
        }
    }

    // Step 2: Initialize queue with headwater cells (in-degree = 0)
    // Direction 0 means pit/flat/outlet, NOT nodata. All cells participate.
    let mut queue: Vec<(usize, usize)> = Vec::new();
    let mut accumulation = Array2::<f64>::zeros((rows, cols));

    for row in 0..rows {
        for col in 0..cols {
            if in_degree[(row, col)] == 0 {
                queue.push((row, col));
            }
        }
    }

    // Step 3: Process queue (topological sort)
    // Each cell passes its accumulation + 1 to its downstream cell
    while let Some((row, col)) = queue.pop() {
        let dir = unsafe { flow_dir.get_unchecked(row, col) };

        if dir == 0 {
            continue; // Pit
        }

        let dir_idx = (dir - 1) as usize;
        if dir_idx >= D8_OFFSETS.len() {
            continue;
        }

        let (dr, dc) = D8_OFFSETS[dir_idx];
        let nr = row as isize + dr;
        let nc = col as isize + dc;

        if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
            continue;
        }

        let nr = nr as usize;
        let nc = nc as usize;

        // Pass accumulation downstream: this cell + all its upstream
        accumulation[(nr, nc)] += accumulation[(row, col)] + 1.0;

        // Decrease in-degree of downstream cell
        in_degree[(nr, nc)] = in_degree[(nr, nc)].saturating_sub(1);

        // If all upstream cells have been processed, add to queue
        if in_degree[(nr, nc)] == 0 {
            queue.push((nr, nc));
        }
    }

    let mut output = flow_dir.with_same_meta::<f64>(rows, cols);
    *output.data_mut() = accumulation;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrology::flow_direction::flow_direction;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_flow_accumulation_linear() {
        // 1x5 strip sloping east: all flow goes E
        // Cell 0 → Cell 1 → Cell 2 → Cell 3 → Cell 4
        // Acc:  0      1      2       3       4
        let mut dem = Raster::new(1, 5);
        dem.set_transform(GeoTransform::new(0.0, 1.0, 1.0, -1.0));

        for col in 0..5 {
            dem.set(0, col, (5 - col) as f64).unwrap();
        }

        let fdir = flow_direction(&dem).unwrap();
        let acc = flow_accumulation(&fdir).unwrap();

        assert_eq!(acc.get(0, 0).unwrap(), 0.0); // Headwater
        assert_eq!(acc.get(0, 1).unwrap(), 1.0);
        assert_eq!(acc.get(0, 2).unwrap(), 2.0);
        assert_eq!(acc.get(0, 3).unwrap(), 3.0);
        assert_eq!(acc.get(0, 4).unwrap(), 4.0); // Outlet
    }

    #[test]
    fn test_flow_accumulation_convergent() {
        // 3x3 DEM with center lowest - all flow converges to center
        //  5 5 5
        //  5 1 5
        //  5 5 5
        let mut dem = Raster::new(3, 3);
        dem.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        for row in 0..3 {
            for col in 0..3 {
                dem.set(row, col, 5.0).unwrap();
            }
        }
        dem.set(1, 1, 1.0).unwrap();

        let fdir = flow_direction(&dem).unwrap();
        let acc = flow_accumulation(&fdir).unwrap();

        // Center should receive flow from all 8 neighbors
        let center = acc.get(1, 1).unwrap();
        assert_eq!(
            center, 8.0,
            "Center should accumulate all 8 neighbors, got {}",
            center
        );
    }

    #[test]
    fn test_flow_accumulation_plane() {
        // 5x5 plane sloping south: each row accumulates from rows above
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let acc = flow_accumulation(&fdir).unwrap();

        // Top row cells should have accumulation = 0
        for col in 0..5 {
            assert_eq!(
                acc.get(0, col).unwrap(),
                0.0,
                "Top row should have 0 accumulation"
            );
        }

        // Bottom row should have highest accumulation
        let bottom_center = acc.get(4, 2).unwrap();
        assert!(
            bottom_center >= 4.0,
            "Bottom center should have high accumulation, got {}",
            bottom_center
        );
    }
}
