//! Breach depressions for hydrological conditioning
//!
//! Removes depressions by "breaching" — carving channels from pit cells
//! toward lower terrain, rather than filling them. Breaching preserves
//! more of the original DEM surface than filling.
//!
//! Algorithm (least-cost breaching):
//! 1. Identify all pit cells (cells with no downslope D8 neighbor)
//! 2. For each pit, use Dijkstra's algorithm to find the least-cost
//!    path to a cell that is lower than the pit
//! 3. Carve the path by lowering cells along it to create a gradient
//!
//! Reference:
//! Lindsay, J.B. (2016). Efficient hybrid breaching-filling sink removal
//! methods for flow path enforcement in digital elevation models.
//! *Hydrological Processes*, 30(6), 846–857.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::Result;

/// D8 neighbor offsets
const D8_OFFSETS: [(isize, isize); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

/// D8 distance factors
const D8_DIST: [f64; 8] = [
    std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
    1.0,                           1.0,
    std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
];

/// Parameters for breach depressions
#[derive(Debug, Clone)]
pub struct BreachParams {
    /// Maximum breach depth (meters). Depressions deeper than this
    /// will be partially filled after breaching.
    /// Default: f64::MAX (no limit)
    pub max_depth: f64,

    /// Maximum breach length (cells). Longer breach paths are truncated
    /// and the remaining depression is filled.
    /// Default: usize::MAX (no limit)
    pub max_length: usize,

    /// Whether to fill remaining depressions after breaching.
    /// If true (default), any depressions that couldn't be fully
    /// breached are filled to create a hydrologically complete surface.
    pub fill_remaining: bool,
}

impl Default for BreachParams {
    fn default() -> Self {
        Self {
            max_depth: f64::MAX,
            max_length: usize::MAX,
            fill_remaining: true,
        }
    }
}

/// Cell for Dijkstra's priority queue
#[derive(Debug, Clone)]
struct BreachCell {
    cost: f64,
    row: usize,
    col: usize,
}

impl PartialEq for BreachCell {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for BreachCell {}

impl PartialOrd for BreachCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BreachCell {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower cost = higher priority
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

/// Breach depressions in a DEM.
///
/// Carves channels from pit cells toward lower terrain using
/// least-cost paths. Preserves more of the original DEM than filling.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Breach parameters
///
/// # Returns
/// A new raster with depressions breached (and optionally filled)
pub fn breach_depressions(dem: &Raster<f64>, params: BreachParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Copy DEM to output
    let mut output = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    for row in 0..rows {
        for col in 0..cols {
            output[(row, col)] = unsafe { dem.get_unchecked(row, col) };
        }
    }

    // Find all pit cells (not on border, with no downslope D8 neighbor)
    let mut pits: Vec<(usize, usize)> = Vec::new();

    for row in 1..rows - 1 {
        for col in 1..cols - 1 {
            let z = output[(row, col)];
            if z.is_nan() || nodata.is_some_and(|nd| (z - nd).abs() < f64::EPSILON) {
                continue;
            }

            let mut is_pit = true;
            for &(dr, dc) in &D8_OFFSETS {
                let nr = (row as isize + dr) as usize;
                let nc = (col as isize + dc) as usize;
                let nz = output[(nr, nc)];
                if nz.is_nan() {
                    continue;
                }
                if nz < z {
                    is_pit = false;
                    break;
                }
            }

            if is_pit {
                pits.push((row, col));
            }
        }
    }

    // Breach each pit using Dijkstra least-cost path
    for &(pit_row, pit_col) in &pits {
        let pit_elev = output[(pit_row, pit_col)];

        if pit_elev.is_nan() {
            continue;
        }

        // Dijkstra: find least-cost path to a cell lower than the pit
        // or to a border cell
        let total = rows * cols;
        let mut cost = vec![f64::MAX; total];
        let mut prev = vec![usize::MAX; total];
        let mut visited = vec![false; total];

        let start_idx = pit_row * cols + pit_col;
        cost[start_idx] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(BreachCell {
            cost: 0.0,
            row: pit_row,
            col: pit_col,
        });

        let mut target_idx: Option<usize> = None;
        let mut depth_from_start: Vec<usize> = vec![0; total];

        while let Some(cell) = heap.pop() {
            let idx = cell.row * cols + cell.col;

            if visited[idx] {
                continue;
            }
            visited[idx] = true;

            let z = output[(cell.row, cell.col)];

            // Check if this is a valid target:
            // (1) lower than the pit, or (2) on the border
            if idx != start_idx {
                let on_border = cell.row == 0 || cell.row == rows - 1
                    || cell.col == 0 || cell.col == cols - 1;

                if on_border || z < pit_elev {
                    target_idx = Some(idx);
                    break;
                }
            }

            // Check breach length limit
            if depth_from_start[idx] >= params.max_length {
                continue;
            }

            // Expand neighbors
            for (d_idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                let nr = cell.row as isize + dr;
                let nc = cell.col as isize + dc;

                if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                    continue;
                }

                let nr = nr as usize;
                let nc = nc as usize;
                let n_idx = nr * cols + nc;

                if visited[n_idx] {
                    continue;
                }

                let nz = output[(nr, nc)];
                if nz.is_nan() || nodata.is_some_and(|nd| (nz - nd).abs() < f64::EPSILON) {
                    continue;
                }

                // Cost = amount we'd need to carve, weighted by distance
                let carve = if nz > pit_elev { nz - pit_elev } else { 0.0 };
                let dist = D8_DIST[d_idx];
                let edge_cost = carve * dist;
                let new_cost = cell.cost + edge_cost;

                if new_cost < cost[n_idx] {
                    cost[n_idx] = new_cost;
                    prev[n_idx] = idx;
                    depth_from_start[n_idx] = depth_from_start[idx] + 1;
                    heap.push(BreachCell {
                        cost: new_cost,
                        row: nr,
                        col: nc,
                    });
                }
            }
        }

        // Trace back and carve the path
        if let Some(target) = target_idx {
            let mut path: Vec<usize> = Vec::new();
            let mut trace = target;
            while trace != start_idx && trace != usize::MAX {
                path.push(trace);
                trace = prev[trace];
            }
            path.reverse();

            // Carve: enforce decreasing elevation along path from pit to target
            let target_elev = output[(target / cols, target % cols)];
            let n_steps = path.len();

            if n_steps > 0 {
                for (step, &idx) in path.iter().enumerate() {
                    let r = idx / cols;
                    let c = idx % cols;

                    // Linear interpolation between pit elevation and target elevation
                    let frac = (step + 1) as f64 / (n_steps + 1) as f64;
                    let target_z = pit_elev + frac * (target_elev - pit_elev);

                    // Only lower, never raise
                    if output[(r, c)] > target_z {
                        let depth = output[(r, c)] - target_z;
                        if depth <= params.max_depth {
                            output[(r, c)] = target_z;
                        }
                    }
                }
            }
        }
    }

    // Optionally fill remaining depressions using Priority-Flood
    if params.fill_remaining {
        let eps = 1e-5;
        let mut pf_visited = Array2::<bool>::from_elem((rows, cols), false);
        let mut pf_heap = BinaryHeap::new();

        // Seed border cells and nodata
        for row in 0..rows {
            for col in 0..cols {
                let val = output[(row, col)];
                let is_nd = val.is_nan()
                    || nodata.is_some_and(|nd| (val - nd).abs() < f64::EPSILON);

                if is_nd {
                    pf_visited[(row, col)] = true;
                    continue;
                }

                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    pf_heap.push(BreachCell { cost: val, row, col });
                    pf_visited[(row, col)] = true;
                }
            }
        }

        // Process in elevation order
        while let Some(cell) = pf_heap.pop() {
            for &(dr, dc) in &D8_OFFSETS {
                let nr = cell.row as isize + dr;
                let nc = cell.col as isize + dc;
                if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;

                if pf_visited[(nr, nc)] {
                    continue;
                }
                pf_visited[(nr, nc)] = true;

                let nz = output[(nr, nc)];
                if nz.is_nan() {
                    continue;
                }

                let filled = if nz < cell.cost + eps {
                    cell.cost + eps
                } else {
                    nz
                };

                output[(nr, nc)] = filled;
                pf_heap.push(BreachCell { cost: filled, row: nr, col: nc });
            }
        }
    }

    let mut result = dem.with_same_meta::<f64>(rows, cols);
    result.set_nodata(dem.nodata());
    *result.data_mut() = output;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn create_dem_with_sink() -> Raster<f64> {
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
    fn test_breach_removes_depression() {
        let dem = create_dem_with_sink();
        let result = breach_depressions(&dem, BreachParams::default()).unwrap();

        // After breaching, the center cell should have a path to the border.
        // Verify no interior pits remain (every interior cell has a downslope neighbor)
        for row in 1..6 {
            for col in 1..6 {
                let z = result.get(row, col).unwrap();
                let mut has_lower = false;
                for &(dr, dc) in &D8_OFFSETS {
                    let nr = (row as isize + dr) as usize;
                    let nc = (col as isize + dc) as usize;
                    let nz = result.get(nr, nc).unwrap();
                    if nz < z {
                        has_lower = true;
                        break;
                    }
                }
                // All interior cells should have at least one lower neighbor
                // (or be at the same level as the border)
                assert!(
                    has_lower || z >= 9.0 - 0.01,
                    "Cell ({},{}) z={} should have a downslope path",
                    row, col, z
                );
            }
        }
    }

    #[test]
    fn test_breach_preserves_more_than_fill() {
        let dem = create_dem_with_sink();
        let breached = breach_depressions(&dem, BreachParams::default()).unwrap();

        // Breaching should modify fewer cells than filling
        let (rows, cols) = dem.shape();
        let mut breach_changes = 0;

        for row in 0..rows {
            for col in 0..cols {
                let orig = dem.get(row, col).unwrap();
                let b = breached.get(row, col).unwrap();
                if (orig - b).abs() > 1e-10 {
                    breach_changes += 1;
                }
            }
        }

        // Breaching should change fewer cells than the total interior
        assert!(
            breach_changes < 25,
            "Breaching should modify fewer cells than filling, changed {}",
            breach_changes
        );
    }

    #[test]
    fn test_breach_no_change_on_clean_dem() {
        // Plane with no sinks
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }

        let result = breach_depressions(&dem, BreachParams::default()).unwrap();

        for row in 0..10 {
            for col in 0..10 {
                let orig = dem.get(row, col).unwrap();
                let res = result.get(row, col).unwrap();
                assert!(
                    (orig - res).abs() < 1e-10,
                    "Clean DEM should be unchanged at ({},{}): orig={}, result={}",
                    row, col, orig, res
                );
            }
        }
    }

    #[test]
    fn test_breach_never_raises_elevation() {
        let dem = create_dem_with_sink();
        let result = breach_depressions(&dem, BreachParams {
            fill_remaining: false,
            ..Default::default()
        }).unwrap();

        let (rows, cols) = dem.shape();
        for row in 0..rows {
            for col in 0..cols {
                let orig = dem.get(row, col).unwrap();
                let res = result.get(row, col).unwrap();
                assert!(
                    res <= orig + 1e-10,
                    "Breaching (without fill) should never raise elevation at ({},{}): orig={}, result={}",
                    row, col, orig, res
                );
            }
        }
    }
}
