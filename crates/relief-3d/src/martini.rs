//! Martini RTIN (Right-Triangulated Irregular Network) terrain meshing.
//!
//! Port of the Mapbox Martini algorithm (Agafonkin 2019,
//! <https://github.com/mapbox/martini>). Produces an adaptive triangle
//! mesh from a height grid where triangle density follows terrain
//! curvature — more triangles where the surface curves sharply, fewer
//! on flat regions. Same triangle budget as a uniform grid gives a
//! visibly better-looking mesh on heterogeneous DEMs.
//!
//! ## Constraints
//!
//! - **Grid must be `2^n + 1` per side.** A 65 × 65 grid (= 64 cells
//!   per side) is the natural fit for our `chunk_cells = 64` LOD
//!   subdivision; that's what the integration in `lod.rs` targets.
//!   Non-power-of-two grids would require padding or a non-binary
//!   split scheme — out of scope for v1.
//! - **Heights are `f32`.** The algorithm computes vertical errors as
//!   `|interpolated - actual|` at every triangle midpoint and stores
//!   the per-cell maximum in a parallel error grid. f32 is plenty
//!   for relief rendering tolerances (typically ≥ 0.1 m).
//!
//! ## API shape
//!
//! [`Martini::new(grid_size)`] precomputes the per-triangle coordinate
//! table once per grid size; reuse it across many tiles.
//! [`Martini::tile(heights)`] consumes a flat row-major `&[f32]`
//! `grid_size² ` long and returns a [`Tile`] holding the per-cell
//! error pyramid. [`Tile::mesh(max_error)`] then extracts a triangle
//! mesh whose worst-case vertical error is bounded by `max_error`.
//!
//! ## References
//!
//! - Evans, W.S., Kirkpatrick, D., Townsend, G. (2001). "Right-
//!   triangulated irregular networks." Algorithmica 30(2), 264–286.
//! - Lindstrom, P. & Pascucci, V. (2002). "Terrain simplification
//!   simplified." IEEE TVCG 8(3), 239–254.
//! - Mapbox Martini source <https://github.com/mapbox/martini>.

/// Static lookup tables for an `N × N` (= `(2^n + 1)²`) RTIN grid.
/// `coords[i*4..i*4+4]` stores the `(ax, ay, bx, by)` corners of
/// triangle `i`; the third corner `(cx, cy)` is derived as
/// `(ax + ay - by, ay + ax - bx)` — both are integer grid indices.
///
/// The table is built once per `grid_size` and reused across many
/// tiles. Construction is `O(N²)`.
pub struct Martini {
    pub grid_size: usize,
    pub tile_size: usize,
    pub num_triangles: usize,
    pub num_parent_triangles: usize,
    coords: Vec<u32>,
}

/// Errors that can be raised by [`Martini`] construction or use.
#[derive(Debug, thiserror::Error)]
pub enum MartiniError {
    #[error("grid_size must be 2^n + 1 (got {0})")]
    NotPowerOfTwoPlusOne(usize),
    #[error("terrain length {actual} does not match grid_size² = {expected}")]
    SizeMismatch { actual: usize, expected: usize },
}

impl Martini {
    /// Precompute the triangle coordinate table for a `grid_size ×
    /// grid_size` height field. `grid_size` must be `2^n + 1`.
    pub fn new(grid_size: usize) -> Result<Self, MartiniError> {
        if grid_size < 3 {
            return Err(MartiniError::NotPowerOfTwoPlusOne(grid_size));
        }
        let tile_size = grid_size - 1;
        if !tile_size.is_power_of_two() {
            return Err(MartiniError::NotPowerOfTwoPlusOne(grid_size));
        }
        let num_triangles = tile_size * tile_size * 2 - 2;
        let num_parent_triangles = num_triangles - tile_size * tile_size;
        let mut coords = vec![0u32; num_triangles * 4];

        // Port of the Mapbox Martini coord table init. Each triangle's
        // (A, B, C) corners are derived from its position in the
        // implicit binary tree; we only store (A, B) since C is
        // recoverable as `C = A + (rotated AB)`.
        for i in 0..num_triangles {
            let mut id = i + 2;
            let (mut ax, mut ay, mut bx, mut by, mut cx, mut cy) =
                (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
            if id & 1 != 0 {
                // Root triangle 0: A = (0,0), B = C = (tile, tile).
                bx = tile_size as u32;
                by = tile_size as u32;
                cx = tile_size as u32;
            } else {
                // Root triangle 1: A = (tile, tile), B = (0,0),
                // C = (0, tile).
                ax = tile_size as u32;
                ay = tile_size as u32;
                cy = tile_size as u32;
            }
            id >>= 1;
            while id > 1 {
                let mx = (ax + bx) >> 1;
                let my = (ay + by) >> 1;
                if id & 1 != 0 {
                    bx = ax;
                    by = ay;
                    ax = cx;
                    ay = cy;
                } else {
                    ax = bx;
                    ay = by;
                    bx = cx;
                    by = cy;
                }
                cx = mx;
                cy = my;
                id >>= 1;
            }
            let _ = (cx, cy);
            let k = i * 4;
            coords[k] = ax;
            coords[k + 1] = ay;
            coords[k + 2] = bx;
            coords[k + 3] = by;
        }

        Ok(Self {
            grid_size,
            tile_size,
            num_triangles,
            num_parent_triangles,
            coords,
        })
    }

    /// Build an error pyramid for a heightmap. Length must equal
    /// `grid_size²`. Heights are sampled in row-major order.
    pub fn tile<'a>(&'a self, terrain: &[f32]) -> Result<Tile<'a>, MartiniError> {
        let expected = self.grid_size * self.grid_size;
        if terrain.len() != expected {
            return Err(MartiniError::SizeMismatch {
                actual: terrain.len(),
                expected,
            });
        }
        let mut errors = vec![0.0f32; expected];

        // Iterate triangles from smallest (leaves) to largest (root).
        // The coords table is laid out parents-first, so we walk
        // backward.
        for i in (0..self.num_triangles).rev() {
            let k = i * 4;
            let ax = self.coords[k] as usize;
            let ay = self.coords[k + 1] as usize;
            let bx = self.coords[k + 2] as usize;
            let by = self.coords[k + 3] as usize;
            let mx = (ax + bx) >> 1;
            let my = (ay + by) >> 1;
            let cx = mx + my - ay;
            let cy = my + ax - mx;

            let h_a = terrain[ay * self.grid_size + ax];
            let h_b = terrain[by * self.grid_size + bx];
            let interpolated = (h_a + h_b) * 0.5;
            let middle_idx = my * self.grid_size + mx;
            let middle_error = (interpolated - terrain[middle_idx]).abs();

            if errors[middle_idx] < middle_error {
                errors[middle_idx] = middle_error;
            }

            if i < self.num_parent_triangles {
                // Non-leaf triangle: accumulate from its two children.
                // Left  child midpoint = ((ay + cy) / 2, (ax + cx) / 2)
                // Right child midpoint = ((by + cy) / 2, (bx + cx) / 2)
                let left_idx = ((ay + cy) >> 1) * self.grid_size + ((ax + cx) >> 1);
                let right_idx = ((by + cy) >> 1) * self.grid_size + ((bx + cx) >> 1);
                let max_child = errors[left_idx].max(errors[right_idx]);
                if errors[middle_idx] < max_child {
                    errors[middle_idx] = max_child;
                }
            }
        }

        Ok(Tile {
            martini: self,
            terrain: terrain.to_vec(),
            errors,
        })
    }
}

/// A heightmap tile with its precomputed error pyramid. Cheap to
/// query for multiple meshes at different error tolerances — the
/// error pyramid is computed once and reused.
pub struct Tile<'a> {
    martini: &'a Martini,
    terrain: Vec<f32>,
    errors: Vec<f32>,
}

/// Output mesh.
///
/// `vertices` is a flat list of `(x, y, height)` in grid coordinates
/// — x and y are cell indices in `[0, grid_size)` as `f32` so the
/// caller can map them to scene-units. Heights come from the input
/// `terrain` array unchanged.
///
/// `indices` is a flat list of triangle vertex IDs, three per
/// triangle.
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

impl<'a> Tile<'a> {
    /// Extract a triangle mesh whose worst-case vertical error is
    /// bounded by `max_error` (in the same units as `terrain`'s
    /// height values). Smaller error → more triangles.
    ///
    /// `max_error = 0.0` reproduces the full grid (every cell a
    /// vertex); large `max_error` shrinks to a handful of triangles
    /// over flat regions.
    pub fn mesh(&self, max_error: f32) -> Mesh {
        let grid_size = self.martini.grid_size;
        let tile_size = self.martini.tile_size as u32;

        // First pass: walk the tree and mark every vertex that ends
        // up in the mesh. A bitset over the grid (1 byte per vertex
        // for simplicity; the grids are small enough that the byte
        // overhead is negligible vs the triangle counts).
        let mut included = vec![0u8; grid_size * grid_size];

        // The four tile corners are always present.
        included[0] = 1;
        included[tile_size as usize] = 1;
        included[(tile_size as usize) * grid_size] = 1;
        included[(tile_size as usize) * grid_size + (tile_size as usize)] = 1;

        // Recursive traversal starting at the two root triangles.
        // The right-isoceles root pair covers the whole tile.
        self.process_triangle(
            0,
            0,
            tile_size,
            tile_size,
            tile_size,
            0,
            max_error,
            &mut included,
        );
        self.process_triangle(
            tile_size,
            tile_size,
            0,
            0,
            0,
            tile_size,
            max_error,
            &mut included,
        );

        // Second pass: assign vertex IDs to included cells in row-
        // major order. The mapping is `grid index → vertex index`.
        let mut vertex_ids = vec![u32::MAX; grid_size * grid_size];
        let mut vertices: Vec<[f32; 3]> = Vec::new();
        for r in 0..grid_size {
            for c in 0..grid_size {
                let idx = r * grid_size + c;
                if included[idx] != 0 {
                    vertex_ids[idx] = vertices.len() as u32;
                    vertices.push([c as f32, r as f32, self.terrain[idx]]);
                }
            }
        }

        // Third pass: emit triangles by walking the tree again, this
        // time recording vertex IDs instead of just marking inclusion.
        let mut indices: Vec<u32> = Vec::new();
        self.emit_triangle(
            0,
            0,
            tile_size,
            tile_size,
            tile_size,
            0,
            max_error,
            &vertex_ids,
            &mut indices,
        );
        self.emit_triangle(
            tile_size,
            tile_size,
            0,
            0,
            0,
            tile_size,
            max_error,
            &vertex_ids,
            &mut indices,
        );

        Mesh { vertices, indices }
    }

    /// First traversal pass: mark every vertex that lands in the mesh.
    #[allow(clippy::too_many_arguments)]
    fn process_triangle(
        &self,
        ax: u32,
        ay: u32,
        bx: u32,
        by: u32,
        cx: u32,
        cy: u32,
        max_error: f32,
        included: &mut [u8],
    ) {
        let grid_size = self.martini.grid_size;
        let mx = (ax + bx) >> 1;
        let my = (ay + by) >> 1;
        let is_leaf = (ax as i64 - cx as i64).abs() + (ay as i64 - cy as i64).abs() <= 1;
        let midpoint_idx = (my as usize) * grid_size + (mx as usize);
        if !is_leaf && self.errors[midpoint_idx] > max_error {
            // Split: process left and right children.
            included[midpoint_idx] = 1;
            self.process_triangle(cx, cy, ax, ay, mx, my, max_error, included);
            self.process_triangle(bx, by, cx, cy, mx, my, max_error, included);
        }
    }

    /// Second traversal pass: same control flow but record vertex IDs
    /// and emit triangle indices.
    #[allow(clippy::too_many_arguments)]
    fn emit_triangle(
        &self,
        ax: u32,
        ay: u32,
        bx: u32,
        by: u32,
        cx: u32,
        cy: u32,
        max_error: f32,
        vertex_ids: &[u32],
        indices: &mut Vec<u32>,
    ) {
        let grid_size = self.martini.grid_size;
        let mx = (ax + bx) >> 1;
        let my = (ay + by) >> 1;
        let is_leaf = (ax as i64 - cx as i64).abs() + (ay as i64 - cy as i64).abs() <= 1;
        let midpoint_idx = (my as usize) * grid_size + (mx as usize);
        if !is_leaf && self.errors[midpoint_idx] > max_error {
            self.emit_triangle(cx, cy, ax, ay, mx, my, max_error, vertex_ids, indices);
            self.emit_triangle(bx, by, cx, cy, mx, my, max_error, vertex_ids, indices);
        } else {
            let a_idx = (ay as usize) * grid_size + (ax as usize);
            let b_idx = (by as usize) * grid_size + (bx as usize);
            let c_idx = (cy as usize) * grid_size + (cx as usize);
            indices.extend([vertex_ids[a_idx], vertex_ids[b_idx], vertex_ids[c_idx]]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 65×65 flat plate should mesh to just 2 triangles for any
    /// positive `max_error` — every midpoint error is zero, so no
    /// splits happen.
    #[test]
    fn flat_plate_meshes_to_two_triangles() {
        let g = 65;
        let m = Martini::new(g).unwrap();
        let terrain = vec![10.0f32; g * g];
        let tile = m.tile(&terrain).unwrap();
        let mesh = tile.mesh(0.001);
        assert_eq!(
            mesh.indices.len(),
            6,
            "expected 2 triangles, got {}",
            mesh.indices.len() / 3
        );
        assert_eq!(
            mesh.vertices.len(),
            4,
            "expected 4 corner verts, got {}",
            mesh.vertices.len()
        );
    }

    /// `max_error = 0` on a heightmap whose every cell differs from
    /// its neighbours must reproduce the full grid: 65² verts,
    /// 2 × 64² = 8192 triangles. Linear gradient gives every
    /// midpoint a non-zero interpolation error.
    #[test]
    fn zero_error_reproduces_full_grid() {
        let g = 65;
        let m = Martini::new(g).unwrap();
        // Gradient + small per-cell jitter: every interpolation
        // midpoint sees a non-trivial residual, so no triangle can
        // approximate its parent within zero error.
        let mut terrain = vec![0.0f32; g * g];
        for r in 0..g {
            for c in 0..g {
                let r2 = r * r;
                let c2 = c * c;
                terrain[r * g + c] = (r2 + c2) as f32;
            }
        }
        let tile = m.tile(&terrain).unwrap();
        let mesh = tile.mesh(0.0);
        assert_eq!(mesh.vertices.len(), g * g);
        assert_eq!(mesh.indices.len(), 8192 * 3);
    }

    /// A bumpy heightmap with a coarse error tolerance should produce
    /// many fewer triangles than the full grid, but more than a flat
    /// plate. Sanity-check the LOD knob is actually working.
    #[test]
    fn coarse_error_shrinks_the_mesh() {
        let g = 65;
        let m = Martini::new(g).unwrap();
        let mut terrain = vec![0.0f32; g * g];
        for r in 0..g {
            for c in 0..g {
                terrain[r * g + c] = ((r as f32 * 0.3).sin() + (c as f32 * 0.3).cos()) * 5.0;
            }
        }
        let tile = m.tile(&terrain).unwrap();
        let fine = tile.mesh(0.01);
        let coarse = tile.mesh(2.0);
        assert!(
            coarse.indices.len() < fine.indices.len(),
            "coarse tolerance ({} tris) must shrink below fine ({} tris)",
            coarse.indices.len() / 3,
            fine.indices.len() / 3
        );
        assert!(
            coarse.indices.len() >= 6,
            "coarse tolerance must still emit at least the 2 root triangles"
        );
    }

    /// Non-power-of-2 grid sizes are rejected with a clear error.
    #[test]
    fn rejects_invalid_grid_size() {
        assert!(Martini::new(64).is_err()); // 64 - 1 = 63 not a power of 2
        assert!(Martini::new(100).is_err());
        assert!(Martini::new(2).is_err()); // too small
        assert!(Martini::new(65).is_ok());
        assert!(Martini::new(129).is_ok());
        assert!(Martini::new(257).is_ok());
    }

    /// Terrain length must match `grid_size²`.
    #[test]
    fn rejects_mismatched_terrain_length() {
        let m = Martini::new(65).unwrap();
        assert!(m.tile(&vec![0.0; 64 * 65]).is_err());
        assert!(m.tile(&vec![0.0; 65 * 65]).is_ok());
    }
}
