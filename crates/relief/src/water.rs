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

/// Compute per-cell water depth from a binary water mask. Returns a
/// `Raster<f32>` where every water cell holds its 8-connected Chebyshev
/// distance (in cells) to the nearest non-water neighbour. Shore cells
/// → 1.0, deepest centres of large lakes → large values.
/// Non-water cells return 0.0.
///
/// Algorithm: multi-source breadth-first search seeded at every
/// non-water cell. Chebyshev metric matches the 8-connectivity used in
/// `detect_water` and produces visually smooth shore-to-centre
/// gradients for round / amoeba-shaped lakes alike.
///
/// Cost: O(N) — every cell visits the queue at most once. Allocates
/// a `VecDeque` sized to the number of land cells plus a `Vec<bool>`
/// visited mask. On a 1000 × 1000 mask with 30% water this is < 5 MB
/// of scratch and runs in milliseconds.
pub fn water_depth(mask: &Raster<u8>) -> Result<Raster<f32>> {
    use std::collections::VecDeque;

    let (rows, cols) = mask.shape();
    if rows == 0 || cols == 0 {
        return Err(ReliefError::Shape(format!("empty mask: {rows}x{cols}")));
    }

    let n = rows * cols;
    let mut depth = vec![0f32; n];
    let mut visited = vec![false; n];
    let mut queue: VecDeque<(i32, i32)> = VecDeque::with_capacity(n / 4);

    // Seed: every non-water cell (including grid boundary) is distance 0.
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            let m = mask.get(r, c).unwrap_or(0);
            if m == 0 {
                visited[i] = true;
                queue.push_back((r as i32, c as i32));
            }
        }
    }

    // The implicit border outside the grid is also "land" — water cells
    // adjacent to it must read distance 1, not infinity. Seed that here
    // by walking the perimeter and bumping any water cell whose
    // off-grid neighbour would otherwise be missed.
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            if mask.get(r, c).unwrap_or(0) != 1 {
                continue;
            }
            if r == 0 || r == rows - 1 || c == 0 || c == cols - 1 {
                depth[i] = 1.0;
                visited[i] = true;
                queue.push_back((r as i32, c as i32));
            }
        }
    }

    // BFS: each pop fans out to 8 neighbours; first visit fixes the
    // Chebyshev distance to `parent + 1`.
    while let Some((r, c)) = queue.pop_front() {
        let i = (r as usize) * cols + c as usize;
        let d = depth[i];
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = r + dr;
                let nc = c + dc;
                if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                    continue;
                }
                let ni = (nr as usize) * cols + nc as usize;
                if visited[ni] {
                    continue;
                }
                visited[ni] = true;
                depth[ni] = d + 1.0;
                queue.push_back((nr, nc));
            }
        }
    }

    let mut out = Raster::new(rows, cols);
    out.set_transform(*mask.transform());
    *out.data_mut() = Array2::from_shape_vec((rows, cols), depth)
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

    #[test]
    fn water_depth_all_land_is_zero() {
        let mut mask = Raster::<u8>::new(8, 8);
        for r in 0..8 {
            for c in 0..8 {
                mask.set(r, c, 0).unwrap();
            }
        }
        let depth = water_depth(&mask).unwrap();
        for v in depth.data().iter() {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn water_depth_one_cell_pond_is_one() {
        let mut mask = Raster::<u8>::new(5, 5);
        for r in 0..5 {
            for c in 0..5 {
                mask.set(r, c, 0).unwrap();
            }
        }
        mask.set(2, 2, 1).unwrap();
        let depth = water_depth(&mask).unwrap();
        assert_eq!(depth.get(2, 2).unwrap(), 1.0);
    }

    #[test]
    fn water_depth_5x5_pond_centre_deeper_than_shore() {
        // 9x9 mask with a 5x5 water square at rows 2..7, cols 2..7.
        // Shore cells (the 5x5 perimeter) are at distance 1 from land;
        // the centre cell (4, 4) is at Chebyshev distance 3 from the
        // nearest land cell at (1, 4) — i.e. depth 3.
        let mut mask = Raster::<u8>::new(9, 9);
        for r in 0..9 {
            for c in 0..9 {
                let v = if (2..7).contains(&r) && (2..7).contains(&c) {
                    1
                } else {
                    0
                };
                mask.set(r, c, v).unwrap();
            }
        }
        let depth = water_depth(&mask).unwrap();
        assert_eq!(depth.get(2, 2).unwrap(), 1.0, "shore corner");
        assert_eq!(depth.get(2, 4).unwrap(), 1.0, "shore mid");
        assert_eq!(depth.get(4, 4).unwrap(), 3.0, "deepest centre");
        assert_eq!(depth.get(0, 0).unwrap(), 0.0, "land");
    }

    #[test]
    fn water_depth_edge_touching_pond_reads_one_at_border() {
        // Pond touches the right edge — the border-handling seed
        // ensures the edge cells still read depth 1 instead of
        // distance-to-the-only-land-cell-on-the-left.
        let mut mask = Raster::<u8>::new(5, 5);
        for r in 0..5 {
            for c in 0..5 {
                let v = if c >= 3 { 1 } else { 0 };
                mask.set(r, c, v).unwrap();
            }
        }
        let depth = water_depth(&mask).unwrap();
        // Far-right water cell: nearest land via grid is at column 2
        // (distance 2). But the implicit off-grid land is at column 5
        // (distance 1). The depth seed at the perimeter handles that.
        assert_eq!(depth.get(2, 4).unwrap(), 1.0);
    }
}
