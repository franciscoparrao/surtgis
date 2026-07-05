//! Maximum Branch Length
//!
//! Computes the longest flow path (in cells or distance) that drains
//! to each cell from upstream, using D8 flow direction.
//!
//! Algorithm:
//! 1. Fill depressions (priority-flood)
//! 2. Compute D8 flow direction
//! 3. Topological sort from headwater cells
//! 4. For each cell: max_branch = max(upstream_branch + distance_to_upstream)
//!
//! Unlike flow accumulation (which sums upstream cells), this takes
//! the maximum upstream path length.
//!
//! Reference: WhiteboxTools `MaxBranchLength`

use crate::hydrology::{PriorityFloodParams, flow_direction, priority_flood};
use ndarray::Array2;
use std::collections::VecDeque;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use crate::hydrology::d8::{D8_DISTANCE as D8_DIST, D8_OFFSETS};

/// Compute maximum branch length (longest upstream flow path).
///
/// Returns distances in cell units (cardinal = 1, diagonal = sqrt(2)).
pub fn max_branch_length(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();

    // Step 1: fill depressions
    let filled = priority_flood(dem, PriorityFloodParams { epsilon: 1e-5 })?;

    // Step 2: compute D8 flow direction
    let fdir = flow_direction(&filled)?;

    // Step 3: compute in-degree for each cell (how many cells flow INTO it)
    let mut in_degree = vec![0u32; rows * cols];
    let idx = |r: usize, c: usize| r * cols + c;

    for r in 0..rows {
        for c in 0..cols {
            let d = unsafe { fdir.get_unchecked(r, c) };
            if d == 0 || d > 8 {
                continue;
            }
            let (dr, dc) = D8_OFFSETS[(d - 1) as usize];
            let nr = r as isize + dr;
            let nc = c as isize + dc;
            if nr >= 0 && nr < rows as isize && nc >= 0 && nc < cols as isize {
                in_degree[idx(nr as usize, nc as usize)] += 1;
            }
        }
    }

    // Step 4: topological sort - start from headwater cells (in_degree = 0)
    let mut max_branch = vec![0.0_f64; rows * cols];
    let mut queue = VecDeque::new();

    for r in 0..rows {
        for c in 0..cols {
            let v = unsafe { dem.get_unchecked(r, c) };
            if dem.is_nodata(v) {
                continue;
            }
            if in_degree[idx(r, c)] == 0 {
                queue.push_back((r, c));
            }
        }
    }

    // Process in topological order
    while let Some((r, c)) = queue.pop_front() {
        let d = unsafe { fdir.get_unchecked(r, c) };
        if d == 0 || d > 8 {
            continue;
        }

        let dist = D8_DIST[(d - 1) as usize];
        let (dr, dc) = D8_OFFSETS[(d - 1) as usize];
        let nr = r as isize + dr;
        let nc = c as isize + dc;

        if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
            continue;
        }
        let nr = nr as usize;
        let nc = nc as usize;

        // Update downstream cell's max branch length
        let new_length = max_branch[idx(r, c)] + dist;
        if new_length > max_branch[idx(nr, nc)] {
            max_branch[idx(nr, nc)] = new_length;
        }

        // Decrement in-degree of downstream cell
        in_degree[idx(nr, nc)] -= 1;
        if in_degree[idx(nr, nc)] == 0 {
            queue.push_back((nr, nc));
        }
    }

    // Build output raster
    let mut output_data = vec![f64::NAN; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let v = unsafe { dem.get_unchecked(r, c) };
            if dem.is_nodata(v) {
                continue;
            }
            output_data[idx(r, c)] = max_branch[idx(r, c)];
        }
    }

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
    fn test_flat_zero_branch() {
        // Flat DEM: all cells are pits → branch length = 0
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = max_branch_length(&dem).unwrap();
        let v = result.get(5, 5).unwrap();
        // After priority flood with epsilon, there's a gradient toward boundary
        // so max_branch should be small
        assert!(
            v < 20.0,
            "Flat DEM should have small branch length, got {}",
            v
        );
    }

    #[test]
    fn test_slope_increasing_downstream() {
        // Linear slope: row 0 highest, row 9 lowest
        // Cells at bottom should have longer max branches
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, (9 - r) as f64 * 10.0).unwrap();
            }
        }

        let result = max_branch_length(&dem).unwrap();
        let top = result.get(0, 5).unwrap();
        let bottom = result.get(9, 5).unwrap();
        assert!(
            bottom > top,
            "Bottom should have longer branch ({}) than top ({})",
            bottom,
            top
        );
    }

    #[test]
    fn test_non_negative() {
        let mut dem = Raster::new(15, 15);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));
        for r in 0..15 {
            for c in 0..15 {
                let x = c as f64 - 7.0;
                let y = r as f64 - 7.0;
                dem.set(r, c, 100.0 - x * x - y * y).unwrap();
            }
        }

        let result = max_branch_length(&dem).unwrap();
        for r in 0..15 {
            for c in 0..15 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(
                        v >= 0.0,
                        "Branch length should be >= 0, got {} at ({},{})",
                        v,
                        r,
                        c
                    );
                }
            }
        }
    }
}
