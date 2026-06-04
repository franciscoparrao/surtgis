//! Grid mesh generation.
//!
//! M1 uses a synthetic flat-ish grid (cosine-displacement test pattern) so
//! the spike can validate wgpu perf at production-workload size (≥1M
//! vertices) without needing a real DEM. M2 replaces this with
//! `from_dem(dem)`.

use crate::Vertex;

/// Build a (rows × cols) grid of vertices spanning `[-extent, +extent]`
/// in X and Z. UV coordinates run 0→1 across the grid in row-major order.
/// Heights come from `height_fn(row, col)` so the spike can plug in a
/// procedural test pattern; M2 swaps it for the real DEM.
///
/// Returns `(vertices, indices)` ready to upload to wgpu buffers.
pub fn grid_mesh<F>(rows: usize, cols: usize, extent: f32, height_fn: F) -> (Vec<Vertex>, Vec<u32>)
where
    F: Fn(usize, usize) -> f32,
{
    let mut verts = Vec::with_capacity(rows * cols);
    let inv_rows = 1.0 / (rows.max(2) - 1) as f32;
    let inv_cols = 1.0 / (cols.max(2) - 1) as f32;
    for r in 0..rows {
        for c in 0..cols {
            let u = c as f32 * inv_cols;
            let v = r as f32 * inv_rows;
            let x = (u * 2.0 - 1.0) * extent;
            let z = (v * 2.0 - 1.0) * extent;
            let y = height_fn(r, c);
            verts.push(Vertex {
                position: [x, y, z],
                uv: [u, v],
            });
        }
    }

    // Two triangles per cell. (rows-1) × (cols-1) cells × 6 indices.
    let mut idx = Vec::with_capacity((rows - 1) * (cols - 1) * 6);
    let cols_u32 = cols as u32;
    for r in 0..(rows - 1) {
        for c in 0..(cols - 1) {
            let tl = (r as u32) * cols_u32 + c as u32;
            let tr = tl + 1;
            let bl = tl + cols_u32;
            let br = bl + 1;
            // CCW winding: tl, bl, tr  ;  tr, bl, br
            idx.extend([tl, bl, tr, tr, bl, br]);
        }
    }

    (verts, idx)
}

/// M1 spike test pattern: gentle cosine bumps so the textured surface
/// has visible relief without needing a DEM. Amplitude small enough
/// that the mesh stays "landscape-like" in the default camera framing.
pub fn cosine_test_heights(rows: usize, cols: usize) -> impl Fn(usize, usize) -> f32 {
    let inv_rows = 1.0 / (rows.max(2) - 1) as f32;
    let inv_cols = 1.0 / (cols.max(2) - 1) as f32;
    move |r: usize, c: usize| -> f32 {
        let u = c as f32 * inv_cols * std::f32::consts::TAU * 4.0;
        let v = r as f32 * inv_rows * std::f32::consts::TAU * 3.0;
        0.15 * (u.cos() + v.cos())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_mesh_dimensions() {
        let (verts, idx) = grid_mesh(4, 5, 1.0, |_, _| 0.0);
        assert_eq!(verts.len(), 20);
        assert_eq!(idx.len(), (4 - 1) * (5 - 1) * 6);
    }

    #[test]
    fn grid_mesh_uv_corners() {
        let (verts, _) = grid_mesh(3, 3, 1.0, |_, _| 0.0);
        assert_eq!(verts[0].uv, [0.0, 0.0]);
        assert_eq!(verts[8].uv, [1.0, 1.0]);
    }

    #[test]
    fn cosine_pattern_is_finite() {
        let f = cosine_test_heights(16, 16);
        for r in 0..16 {
            for c in 0..16 {
                assert!(f(r, c).is_finite());
            }
        }
    }
}
