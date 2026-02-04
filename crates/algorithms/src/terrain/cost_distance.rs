//! Cost-distance analysis
//!
//! Computes the accumulated cost of traveling from source cells across a cost
//! surface using Dijkstra's algorithm with 8-connectivity.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for cost distance
#[derive(Debug, Clone)]
pub struct CostDistanceParams {
    /// Source cell locations as (row, col) pairs.
    /// If empty, cells with value 0 in the cost surface are used as sources.
    pub sources: Vec<(usize, usize)>,
}

impl Default for CostDistanceParams {
    fn default() -> Self {
        Self { sources: Vec::new() }
    }
}

/// State in the priority queue (min-heap via Reverse ordering).
#[derive(Debug, Clone, PartialEq)]
struct State {
    cost: f64,
    row: usize,
    col: usize,
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

/// 8-connected neighbor offsets with their distance multipliers.
const NEIGHBORS: [(isize, isize, f64); 8] = [
    (-1, -1, std::f64::consts::SQRT_2),
    (-1,  0, 1.0),
    (-1,  1, std::f64::consts::SQRT_2),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    ( 1, -1, std::f64::consts::SQRT_2),
    ( 1,  0, 1.0),
    ( 1,  1, std::f64::consts::SQRT_2),
];

/// Compute accumulated cost distance from source cells.
///
/// Uses Dijkstra's algorithm on an 8-connected grid. The cost to traverse
/// between two cells is the average of their cost values multiplied by the
/// cell distance (1.0 for cardinal, sqrt(2) for diagonal neighbors).
///
/// # Arguments
/// * `cost_surface` - Cost raster (higher values = more costly to traverse).
///   NaN cells are impassable barriers.
/// * `params` - Source cell locations. If empty, cells with value 0 are sources.
///
/// # Returns
/// Raster of accumulated minimum cost from the nearest source cell.
/// Source cells have cost 0. Unreachable cells are NaN.
pub fn cost_distance(
    cost_surface: &Raster<f64>,
    params: CostDistanceParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = cost_surface.shape();

    // Initialize output with infinity
    let mut dist = vec![f64::INFINITY; rows * cols];
    let mut heap = BinaryHeap::new();

    // Determine source cells
    let sources = if params.sources.is_empty() {
        // Use cells with value 0 (or very near 0) as sources
        let mut auto_sources = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let v = unsafe { cost_surface.get_unchecked(r, c) };
                if v.abs() < 1e-10 {
                    auto_sources.push((r, c));
                }
            }
        }
        auto_sources
    } else {
        params.sources
    };

    if sources.is_empty() {
        return Err(Error::Algorithm("No source cells found for cost distance".into()));
    }

    // Initialize sources
    for &(r, c) in &sources {
        if r < rows && c < cols {
            dist[r * cols + c] = 0.0;
            heap.push(State { cost: 0.0, row: r, col: c });
        }
    }

    // Dijkstra
    while let Some(State { cost, row, col }) = heap.pop() {
        // Skip if we already found a better path
        if cost > dist[row * cols + col] {
            continue;
        }

        let cost_here = unsafe { cost_surface.get_unchecked(row, col) };

        for &(dr, dc, step_dist) in &NEIGHBORS {
            let nr = row as isize + dr;
            let nc = col as isize + dc;

            if nr < 0 || nc < 0 || nr as usize >= rows || nc as usize >= cols {
                continue;
            }

            let nr = nr as usize;
            let nc = nc as usize;
            let cost_neighbor = unsafe { cost_surface.get_unchecked(nr, nc) };

            if cost_neighbor.is_nan() || cost_neighbor < 0.0 {
                continue; // Impassable
            }

            // Average cost * distance
            let traverse_cost = (cost_here + cost_neighbor) / 2.0 * step_dist;
            let new_cost = cost + traverse_cost;

            if new_cost < dist[nr * cols + nc] {
                dist[nr * cols + nc] = new_cost;
                heap.push(State { cost: new_cost, row: nr, col: nc });
            }
        }
    }

    // Convert infinity to NaN
    for d in &mut dist {
        if d.is_infinite() {
            *d = f64::NAN;
        }
    }

    let mut output = cost_surface.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), dist)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn uniform_cost(rows: usize, cols: usize, val: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, val);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn test_cost_distance_basic() {
        let r = uniform_cost(10, 10, 1.0);
        let result = cost_distance(&r, CostDistanceParams {
            sources: vec![(0, 0)],
        }).unwrap();

        // Source cell should be 0
        let v00 = result.get(0, 0).unwrap();
        assert!((v00).abs() < 1e-10, "Source should be 0, got {}", v00);

        // Adjacent cell (0,1): avg_cost=1.0, distance=1.0 → 1.0
        let v01 = result.get(0, 1).unwrap();
        assert!((v01 - 1.0).abs() < 1e-10, "Adjacent should be 1.0, got {}", v01);

        // Diagonal cell (1,1): avg_cost=1.0, distance=sqrt(2) → sqrt(2)
        let v11 = result.get(1, 1).unwrap();
        assert!((v11 - std::f64::consts::SQRT_2).abs() < 1e-10,
            "Diagonal should be sqrt(2), got {}", v11);
    }

    #[test]
    fn test_cost_distance_barrier() {
        let mut r = uniform_cost(5, 5, 1.0);
        // Create a barrier wall
        for row in 0..5 {
            r.set(row, 2, f64::NAN).unwrap();
        }

        let result = cost_distance(&r, CostDistanceParams {
            sources: vec![(2, 0)],
        }).unwrap();

        // Cell beyond barrier should be unreachable (NaN)
        let v = result.get(2, 4).unwrap();
        assert!(v.is_nan(), "Cell beyond barrier should be NaN, got {}", v);
    }

    #[test]
    fn test_cost_distance_auto_sources() {
        let mut r = uniform_cost(5, 5, 1.0);
        r.set(2, 2, 0.0).unwrap(); // Auto-source

        let result = cost_distance(&r, CostDistanceParams::default()).unwrap();

        let v = result.get(2, 2).unwrap();
        assert!(v.abs() < 1e-10, "Auto-source should be 0, got {}", v);
    }

    #[test]
    fn test_cost_distance_no_sources() {
        let r = uniform_cost(5, 5, 1.0);
        let result = cost_distance(&r, CostDistanceParams::default());
        assert!(result.is_err(), "Should error with no sources");
    }
}
