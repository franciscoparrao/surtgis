//! Heuristic water mask via flat-area connected components.
//!
//! Rayshader's `detect_water()` finds large patches of near-constant
//! elevation and tags them as water. This is a heuristic — it can't
//! tell a real lake from a paved plaza — but for shaded relief it does
//! the right thing most of the time: the only large dead-flat regions
//! in DEMs of natural landscapes are water bodies.
//!
//! Algorithm:
//!   1. For each cell, look at the 8-neighbour z values. If every finite
//!      neighbour is within `flatness_eps` of the cell's own elevation,
//!      mark the cell as "flat candidate".
//!   2. Union-find over the 4-connected flat-candidate cells.
//!   3. Drop components with fewer than `min_area_cells` cells.
//!
//! Output is a `Raster<u8>` mask: `1` = water, `0` = land, nodata cells
//! stay `0` (we don't propagate NaN through a boolean mask).

use ndarray::Array2;
use surtgis_core::raster::Raster;

use crate::{ReliefError, Result};

/// Parameters for [`detect_water`].
#[derive(Debug, Clone)]
pub struct WaterParams {
    /// Minimum number of cells a flat region must contain to count as
    /// water. Smaller regions (e.g. a flat parking lot, a flat roof) are
    /// rejected. Default 200 cells.
    pub min_area_cells: usize,
    /// Maximum elevation difference (in DEM units, usually metres) within
    /// which neighbours are considered "the same elevation". Default 0.5 m
    /// — comfortably above DEM noise floors of ~5 cm.
    pub flatness_eps: f64,
}

impl Default for WaterParams {
    fn default() -> Self {
        Self {
            min_area_cells: 200,
            flatness_eps: 0.5,
        }
    }
}

/// Detect water bodies as large flat connected components.
///
/// Returns a binary `Raster<u8>` (1 = water, 0 = land).
///
/// # Errors
///
/// Returns [`ReliefError::Shape`] if the DEM has zero rows or columns
/// (defensive — `Raster::new` doesn't usually let this happen).
pub fn detect_water(dem: &Raster<f64>, params: &WaterParams) -> Result<Raster<u8>> {
    let (rows, cols) = dem.shape();
    if rows == 0 || cols == 0 {
        return Err(ReliefError::Shape(format!("empty DEM: {rows}x{cols}")));
    }

    // Step 1: per-cell flat-candidate mask. A cell is a candidate iff
    // every finite 8-neighbour is within flatness_eps of its own value.
    let mut flat = vec![false; rows * cols];
    let eps = params.flatness_eps;
    for r in 0..rows {
        for c in 0..cols {
            let z = unsafe { dem.get_unchecked(r, c) };
            if !z.is_finite() {
                continue;
            }
            let mut is_flat = true;
            for dr in -1i32..=1 {
                for dc in -1i32..=1 {
                    if dr == 0 && dc == 0 {
                        continue;
                    }
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                        continue;
                    }
                    let zn = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
                    if !zn.is_finite() {
                        continue;
                    }
                    if (zn - z).abs() > eps {
                        is_flat = false;
                        break;
                    }
                }
                if !is_flat {
                    break;
                }
            }
            if is_flat {
                flat[r * cols + c] = true;
            }
        }
    }

    // Step 2: union-find on 4-connected flat cells. Labels are the
    // root-cell index. We use path compression but not rank balancing —
    // good enough for raster-sized inputs.
    let n = rows * cols;
    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            if !flat[i] {
                continue;
            }
            // Look at up + left only (we sweep top-to-bottom, left-to-right).
            if r > 0 {
                let j = (r - 1) * cols + c;
                if flat[j] {
                    union(&mut parent, i, j);
                }
            }
            if c > 0 {
                let j = r * cols + (c - 1);
                if flat[j] {
                    union(&mut parent, i, j);
                }
            }
        }
    }

    // Step 3: tally component sizes, then mark cells in components
    // whose size >= min_area_cells.
    let mut size = vec![0usize; n];
    for i in 0..n {
        if flat[i] {
            let r = find(&mut parent, i);
            size[r] += 1;
        }
    }

    let mut mask = vec![0u8; n];
    for i in 0..n {
        if !flat[i] {
            continue;
        }
        let r = find(&mut parent, i);
        if size[r] >= params.min_area_cells {
            mask[i] = 1;
        }
    }

    let mut out = Raster::new(rows, cols);
    out.set_transform(*dem.transform());
    *out.data_mut() = Array2::from_shape_vec((rows, cols), mask)
        .map_err(|e| ReliefError::Shape(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_block_is_water_if_above_min_area() {
        // 20x20 DEM, central 10x10 block at z=0, rest is a ramp.
        let mut dem = Raster::new(20, 20);
        for r in 0..20 {
            for c in 0..20 {
                let z = if (5..15).contains(&r) && (5..15).contains(&c) {
                    0.0
                } else {
                    (r + c) as f64
                };
                dem.set(r, c, z).unwrap();
            }
        }
        let mask = detect_water(
            &dem,
            &WaterParams {
                min_area_cells: 50,
                flatness_eps: 0.5,
            },
        )
        .unwrap();
        // Centre of the flat block should be water.
        assert_eq!(mask.get(10, 10).unwrap(), 1);
        // Far corner of the ramp should be land.
        assert_eq!(mask.get(0, 0).unwrap(), 0);
    }

    #[test]
    fn small_flat_block_is_rejected_below_min_area() {
        // 20x20, 3x3 flat block — too small.
        let mut dem = Raster::new(20, 20);
        for r in 0..20 {
            for c in 0..20 {
                let z = if (5..8).contains(&r) && (5..8).contains(&c) {
                    0.0
                } else {
                    (r + c) as f64
                };
                dem.set(r, c, z).unwrap();
            }
        }
        let mask = detect_water(
            &dem,
            &WaterParams {
                min_area_cells: 50,
                flatness_eps: 0.5,
            },
        )
        .unwrap();
        for r in 0..20 {
            for c in 0..20 {
                assert_eq!(mask.get(r, c).unwrap(), 0);
            }
        }
    }

    #[test]
    fn nan_cells_are_not_water() {
        let mut dem = Raster::new(10, 10);
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, f64::NAN).unwrap();
            }
        }
        let mask = detect_water(&dem, &WaterParams::default()).unwrap();
        for v in mask.data().iter() {
            assert_eq!(*v, 0);
        }
    }
}
