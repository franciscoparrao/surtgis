//! Grid mesh generation.
//!
//! M1 uses a synthetic flat-ish grid (cosine-displacement test pattern) so
//! the spike can validate wgpu perf at production-workload size (≥1M
//! vertices) without needing a real DEM. M2 replaces this with
//! `from_dem(dem)`.

use crate::Vertex;
use surtgis_core::raster::Raster;

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

/// Build a grid mesh from a real DEM raster.
///
/// Heights are normalised to `[0, vertical_exaggeration]` in scene units,
/// preserving relative relief. NaN cells map to 0 so the mesh stays flat
/// in nodata regions instead of spiking. The XZ extent uses the longer
/// DEM side as `2 * extent` scene units; aspect ratio is preserved on
/// the shorter side.
///
/// `vertical_exaggeration` is the displacement scale of the tallest cell
/// in scene units. A typical mountain DEM looks natural at 0.3–0.6;
/// the slider in `examples/render_dem.rs` lets the user retune live.
pub fn from_dem(dem: &Raster<f64>, vertical_exaggeration: f32) -> (Vec<Vertex>, Vec<u32>) {
    let (rows, cols) = dem.shape();
    let longer = rows.max(cols) as f32;
    let extent_x = cols as f32 / longer;
    let extent_z = rows as f32 / longer;

    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for v in dem.data().iter() {
        if !v.is_finite() {
            continue;
        }
        if *v < z_min {
            z_min = *v;
        }
        if *v > z_max {
            z_max = *v;
        }
    }
    let range = if (z_max - z_min).is_finite() && z_max > z_min {
        z_max - z_min
    } else {
        1.0
    };

    let inv_rows = 1.0 / (rows.max(2) - 1) as f32;
    let inv_cols = 1.0 / (cols.max(2) - 1) as f32;

    let mut verts = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let u = c as f32 * inv_cols;
            let v = r as f32 * inv_rows;
            let x = (u * 2.0 - 1.0) * extent_x;
            let z = (v * 2.0 - 1.0) * extent_z;
            let raw = dem.get(r, c).unwrap_or(z_min);
            let normalised = if raw.is_finite() {
                ((raw - z_min) / range).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            verts.push(Vertex {
                position: [x, normalised * vertical_exaggeration, z],
                uv: [u, v],
            });
        }
    }

    let cols_u32 = cols as u32;
    let mut idx = Vec::with_capacity((rows - 1) * (cols - 1) * 6);
    for r in 0..(rows - 1) {
        for c in 0..(cols - 1) {
            let tl = (r as u32) * cols_u32 + c as u32;
            let tr = tl + 1;
            let bl = tl + cols_u32;
            let br = bl + 1;
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
    fn from_dem_normalises_heights() {
        let mut dem = Raster::<f64>::new(5, 7);
        // Ramp 0..N along columns so the tallest cell sits at the
        // rightmost edge.
        for r in 0..5 {
            for c in 0..7 {
                dem.set(r, c, c as f64 * 100.0).unwrap();
            }
        }
        let zex = 0.5;
        let (verts, _idx) = from_dem(&dem, zex);
        assert_eq!(verts.len(), 35);
        // First-row first-col is z_min → y=0
        assert!((verts[0].position[1] - 0.0).abs() < 1e-6);
        // Last-row last-col is z_max → y=zex
        assert!((verts.last().unwrap().position[1] - zex).abs() < 1e-6);
    }

    #[test]
    fn from_dem_nan_becomes_zero() {
        // Ramp 0,100,200 so the DEM has a real elevation range. NaN at
        // (1,1) should land at y=0 (flat) while finite cells land along
        // the normalised ramp.
        let mut dem = Raster::<f64>::new(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                let v = if r == 1 && c == 1 {
                    f64::NAN
                } else {
                    c as f64 * 100.0
                };
                dem.set(r, c, v).unwrap();
            }
        }
        let (verts, _) = from_dem(&dem, 1.0);
        // The NaN cell is at index 4 — must be flat.
        assert_eq!(verts[4].position[1], 0.0);
        // Top-right is z_max=200 → y=1.0.
        assert!((verts[2].position[1] - 1.0).abs() < 1e-6);
        // Top-left is z_min=0 → y=0.0.
        assert!(verts[0].position[1].abs() < 1e-6);
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
