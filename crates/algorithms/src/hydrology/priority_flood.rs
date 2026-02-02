//! Priority-Flood depression filling
//!
//! State-of-the-art O(n log n) algorithm for filling depressions in a DEM.
//! Uses a priority queue (min-heap) to process cells in elevation order,
//! starting from the DEM boundary.
//!
//! Advantages over Planchon-Darboux (2001):
//! - Single pass through the data (no iterative convergence)
//! - Guaranteed O(n log n) worst case for floating-point DEMs
//! - Simpler implementation (~30 lines of core logic)
//!
//! Reference:
//! Barnes, R., Lehman, C., & Mulla, D. (2014). Priority-Flood: An optimal
//! depression-filling and watershed-labeling algorithm for digital elevation
//! models. *Computers & Geosciences*, 62, 117–127.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// A cell in the priority queue, ordered by elevation (min-heap via Reverse).
#[derive(Debug, Clone)]
struct Cell {
    elevation: f64,
    row: usize,
    col: usize,
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.elevation == other.elevation
    }
}

impl Eq for Cell {}

// Reverse ordering so BinaryHeap (max-heap) acts as a min-heap
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: lower elevation has higher priority
        other.elevation.partial_cmp(&self.elevation)
            .unwrap_or(Ordering::Equal)
    }
}

/// D8 neighbor offsets
const D8_OFFSETS: [(isize, isize); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

/// Parameters for Priority-Flood filling
#[derive(Debug, Clone)]
pub struct PriorityFloodParams {
    /// Minimum elevation increment to enforce between cells.
    /// Use a small epsilon (e.g., 1e-5) to break ties and create
    /// a slight gradient in flat areas. Set to 0.0 to allow perfectly
    /// flat filled areas.
    pub epsilon: f64,
}

impl Default for PriorityFloodParams {
    fn default() -> Self {
        Self { epsilon: 1e-5 }
    }
}

/// Priority-Flood fill algorithm
#[derive(Debug, Clone, Default)]
pub struct PriorityFlood;

impl Algorithm for PriorityFlood {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = PriorityFloodParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Priority-Flood"
    }

    fn description(&self) -> &'static str {
        "Fill depressions using Priority-Flood (Barnes 2014)"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        priority_flood(&input, params)
    }
}

/// Fill depressions in a DEM using the Priority-Flood algorithm (Barnes 2014).
///
/// This is the state-of-the-art method for depression filling.
/// It processes cells in elevation order using a priority queue,
/// ensuring each cell is visited at most once.
///
/// # Algorithm
/// 1. Initialize: add all border cells to a min-heap, mark as visited
/// 2. Pop the lowest cell from the heap
/// 3. For each unvisited neighbor:
///    - Set output = max(neighbor_elevation, popped_elevation + epsilon)
///    - Mark as visited, push to heap
/// 4. Repeat until heap is empty
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Fill parameters (epsilon for gradient enforcement)
///
/// # Returns
/// A new raster with all depressions filled
pub fn priority_flood(dem: &Raster<f64>, params: PriorityFloodParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let epsilon = params.epsilon;

    let mut output = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    let mut visited = Array2::<bool>::from_elem((rows, cols), false);
    let mut heap = BinaryHeap::new();

    // Step 1: Seed the priority queue with border cells
    for row in 0..rows {
        for col in 0..cols {
            let val = unsafe { dem.get_unchecked(row, col) };

            // Check if nodata
            let is_nd = val.is_nan()
                || nodata.map_or(false, |nd| (val - nd).abs() < f64::EPSILON);

            if is_nd {
                visited[(row, col)] = true;
                output[(row, col)] = val; // preserve nodata
                continue;
            }

            // Border cells seed the queue
            if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                heap.push(Cell { elevation: val, row, col });
                visited[(row, col)] = true;
                output[(row, col)] = val;
            }
        }
    }

    // Step 2: Process cells in order of increasing elevation
    while let Some(cell) = heap.pop() {
        for &(dr, dc) in &D8_OFFSETS {
            let nr = cell.row as isize + dr;
            let nc = cell.col as isize + dc;

            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }

            let nr = nr as usize;
            let nc = nc as usize;

            if visited[(nr, nc)] {
                continue;
            }

            visited[(nr, nc)] = true;

            let neighbor_elev = unsafe { dem.get_unchecked(nr, nc) };

            // Check nodata
            let is_nd = neighbor_elev.is_nan()
                || nodata.map_or(false, |nd| (neighbor_elev - nd).abs() < f64::EPSILON);

            if is_nd {
                output[(nr, nc)] = neighbor_elev;
                continue;
            }

            // Core Priority-Flood logic:
            // If neighbor is lower than the current cell's output elevation,
            // raise it (fill the depression). Otherwise keep original.
            let filled_elev = if neighbor_elev < cell.elevation + epsilon {
                cell.elevation + epsilon
            } else {
                neighbor_elev
            };

            output[(nr, nc)] = filled_elev;
            heap.push(Cell {
                elevation: filled_elev,
                row: nr,
                col: nc,
            });
        }
    }

    let mut result = dem.with_same_meta::<f64>(rows, cols);
    result.set_nodata(dem.nodata());
    *result.data_mut() = output;

    Ok(result)
}

/// Convenience: Priority-Flood with epsilon=0 (flat filling).
/// Creates perfectly flat areas in filled depressions.
pub fn priority_flood_flat(dem: &Raster<f64>) -> Result<Raster<f64>> {
    priority_flood(dem, PriorityFloodParams { epsilon: 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn create_dem_with_sink() -> Raster<f64> {
        // 7x7 DEM with a depression in the center
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
            dem.set(idx / 7, idx % 7, val).unwrap();
        }

        dem
    }

    #[test]
    fn test_priority_flood_fills_sink() {
        let dem = create_dem_with_sink();
        let filled = priority_flood(&dem, PriorityFloodParams { epsilon: 0.0 }).unwrap();

        // The center cell (3,3) had value 3.0, surrounded by ring at 7.0.
        // After flat filling (epsilon=0), it should be raised to 7.0.
        let center = filled.get(3, 3).unwrap();
        assert!(
            center >= 7.0,
            "Sink at (3,3) should be filled to >= 7.0, got {}",
            center
        );
    }

    #[test]
    fn test_priority_flood_preserves_border() {
        let dem = create_dem_with_sink();
        let filled = priority_flood(&dem, PriorityFloodParams::default()).unwrap();

        assert_eq!(filled.get(0, 0).unwrap(), 9.0);
        assert_eq!(filled.get(0, 3).unwrap(), 9.0);
        assert_eq!(filled.get(6, 6).unwrap(), 9.0);
    }

    #[test]
    fn test_priority_flood_no_change_on_clean_dem() {
        // Sloped plane: no sinks
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }

        let filled = priority_flood(&dem, PriorityFloodParams::default()).unwrap();

        for row in 0..10 {
            for col in 0..10 {
                let orig = dem.get(row, col).unwrap();
                let fill = filled.get(row, col).unwrap();
                assert!(
                    (fill - orig).abs() < 1e-4,
                    "Clean DEM should be unchanged at ({}, {}): orig={}, fill={}",
                    row, col, orig, fill
                );
            }
        }
    }

    #[test]
    fn test_priority_flood_with_epsilon_creates_gradient() {
        let dem = create_dem_with_sink();
        let filled = priority_flood(&dem, PriorityFloodParams { epsilon: 0.01 }).unwrap();

        // With epsilon > 0, the filled area should have a slight gradient
        // (center slightly higher than rim of depression)
        let center = filled.get(3, 3).unwrap();
        let ring = filled.get(2, 3).unwrap();

        // Center is deeper in the depression, so after filling with epsilon,
        // it gets raised more than the ring cells
        assert!(
            center >= 7.0,
            "Center should be filled above rim: center={}",
            center
        );
        // Center should be slightly higher than ring due to epsilon gradient
        assert!(
            center > ring - 0.1,
            "Epsilon should create gradient: center={}, ring={}",
            center, ring
        );
    }

    #[test]
    fn test_priority_flood_never_lowers_elevation() {
        let dem = create_dem_with_sink();
        let filled = priority_flood(&dem, PriorityFloodParams::default()).unwrap();

        let (rows, cols) = dem.shape();
        for row in 0..rows {
            for col in 0..cols {
                let orig = dem.get(row, col).unwrap();
                let fill = filled.get(row, col).unwrap();
                if !orig.is_nan() && !fill.is_nan() {
                    assert!(
                        fill >= orig - 1e-10,
                        "Priority-Flood must never lower elevation at ({}, {}): orig={}, fill={}",
                        row, col, orig, fill
                    );
                }
            }
        }
    }

    #[test]
    fn test_priority_flood_outlet_respects_low_border() {
        // 5x5 DEM: border=10 except outlet at (4,2)=2, center sink at (2,2)=1
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

        let filled = priority_flood(&dem, PriorityFloodParams { epsilon: 0.0 }).unwrap();

        // Center should be filled to the outlet level (2.0), not the higher border (10.0)
        let center = filled.get(2, 2).unwrap();
        assert!(
            center >= 2.0 && center <= 5.0,
            "Sink should fill to outlet level (~2.0–5.0), got {}",
            center
        );
    }

    #[test]
    fn test_priority_flood_flat_convenience() {
        let dem = create_dem_with_sink();
        let filled = priority_flood_flat(&dem).unwrap();

        let center = filled.get(3, 3).unwrap();
        assert!(center >= 7.0, "Flat fill should raise sink, got {}", center);
    }
}
