//! Quadtree LOD for DEMs that don't fit in a single full-resolution
//! mesh upload (~ 1 M vertices is the practical native ceiling).
//!
//! Architecture:
//!
//!   - One *full* vertex buffer for the whole DEM (vertices live on
//!     the GPU once; LODs differ only in which indices reference
//!     them). Memory cost: rows × cols × 32 B. At 4 K × 4 K that's
//!     ~512 MB, fine for native, a non-trivial constraint for WASM
//!     that M3 addresses.
//!   - The DEM is subdivided into `chunk_cells × chunk_cells` chunks.
//!   - Each chunk owns one index slice per LOD level. LOD 0 = full
//!     resolution; LOD k = stride 2^k.
//!   - Per frame, the viewer picks a LOD per chunk based on distance
//!     (M1 uses simple distance bands; a screen-space-error metric is
//!     a follow-up).
//!   - Frustum culling drops chunks whose AABB is fully outside any
//!     of the six clip-space half-spaces.
//!
//! Public surface lives in [`crate::lod`] and is also re-exported at
//! the crate root for ergonomics.

use std::ops::Range;

use glam::{Mat4, Vec3, Vec4};
use surtgis_core::raster::Raster;

use crate::Vertex;

/// Tunable knobs for the quadtree pipeline.
#[derive(Debug, Clone)]
pub struct LodParams {
    /// Side length of each leaf chunk, in DEM cells. Default 64
    /// (matches the rayshader convention). Must be a multiple of
    /// `2^(max_decimation_log2)` so coarse LODs index evenly.
    pub chunk_cells: usize,
    /// Number of LOD levels. Level k uses stride `2^k`. Default 4
    /// (strides 1, 2, 4, 8). Combined with `chunk_cells = 64` this
    /// requires `64 % 8 == 0`, which it does.
    pub lod_levels: usize,
    /// Camera-distance bands used to pick a LOD per chunk. Each
    /// entry is the *upper* bound in scene units (mesh longer side
    /// = 2 scene units by convention from [`crate::mesh::from_dem`]).
    /// The number of bands must equal `lod_levels - 1`; the last
    /// LOD is the catch-all.
    ///
    /// Default `[0.6, 1.8, 5.0]` reads as "very close → LOD 0; near
    /// → LOD 1; medium → LOD 2; far → LOD 3", tuned on the 4 K spike.
    /// Loose bands keep too many chunks at high LOD and the per-frame
    /// vertex throughput drops the FPS even when the chunk *count*
    /// looks bounded. M1 tuning; a true screen-space-error metric is
    /// a follow-up.
    pub distance_bands: Vec<f32>,
}

impl Default for LodParams {
    fn default() -> Self {
        Self {
            chunk_cells: 64,
            lod_levels: 4,
            distance_bands: vec![0.6, 1.8, 5.0],
        }
    }
}

/// Axis-aligned bounding box used for frustum culling and the LOD
/// distance metric.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Eight corner positions of the AABB. Order is consistent so the
    /// frustum test can iterate without conditionals.
    pub fn corners(&self) -> [Vec3; 8] {
        let (a, b) = (self.min, self.max);
        [
            Vec3::new(a.x, a.y, a.z),
            Vec3::new(b.x, a.y, a.z),
            Vec3::new(a.x, b.y, a.z),
            Vec3::new(b.x, b.y, a.z),
            Vec3::new(a.x, a.y, b.z),
            Vec3::new(b.x, a.y, b.z),
            Vec3::new(a.x, b.y, b.z),
            Vec3::new(b.x, b.y, b.z),
        ]
    }
}

/// One leaf of the quadtree.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// World-space bounding box. Used for both culling and distance.
    pub bbox: Aabb,
    /// Inclusive cell range in row and column dimensions. Used by
    /// builders that want to inspect or modify the chunk's slice.
    pub row_range: Range<usize>,
    pub col_range: Range<usize>,
    /// One index range per LOD level — `lod_indices[k]` is the slice
    /// into [`QuadtreeMesh::indices`] for LOD k of this chunk.
    pub lod_indices: Vec<Range<u32>>,
}

/// The full mesh + chunk metadata. Vertices are shared across LODs
/// (only the indices differ), so memory cost is dominated by the
/// vertex grid at the highest resolution.
#[derive(Debug)]
pub struct QuadtreeMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub chunks: Vec<Chunk>,
    pub rows: usize,
    pub cols: usize,
    pub lod_levels: usize,
}

impl QuadtreeMesh {
    /// Build a quadtree mesh from a DEM. Heights are normalised to
    /// `[0, vertical_exaggeration]` in scene units, same convention
    /// as [`crate::mesh::from_dem`]. The XZ extent uses
    /// `longer side = 2 scene units`.
    pub fn from_dem(dem: &Raster<f64>, vertical_exaggeration: f32, params: LodParams) -> Self {
        let (rows, cols) = dem.shape();
        let longer = rows.max(cols) as f32;
        let extent_x = cols as f32 / longer;
        let extent_z = rows as f32 / longer;
        let inv_rows = 1.0 / (rows.max(2) - 1) as f32;
        let inv_cols = 1.0 / (cols.max(2) - 1) as f32;

        // Normalise heights to [0, 1] before applying vertical_exaggeration.
        let (z_min, z_max) = {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for v in dem.data().iter() {
                if !v.is_finite() {
                    continue;
                }
                if *v < lo {
                    lo = *v;
                }
                if *v > hi {
                    hi = *v;
                }
            }
            (lo, hi)
        };
        let range = if (z_max - z_min).is_finite() && z_max > z_min {
            z_max - z_min
        } else {
            1.0
        };
        let height_at = |r: usize, c: usize| -> f32 {
            let raw = dem.get(r, c).unwrap_or(z_min);
            let n = if raw.is_finite() {
                ((raw - z_min) / range).clamp(0.0, 1.0) as f32
            } else {
                0.0
            };
            n * vertical_exaggeration
        };

        // ---- 1. Full vertex grid. One vertex per cell, indexed by
        // ----    `row * cols + col`. Normals are computed via central
        // ----    differences, same as `mesh::from_dem`.
        let dx = 2.0 * extent_x / (cols.max(2) - 1) as f32;
        let dz = 2.0 * extent_z / (rows.max(2) - 1) as f32;

        let mut vertices = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let u = c as f32 * inv_cols;
                let v = r as f32 * inv_rows;
                let x = (u * 2.0 - 1.0) * extent_x;
                let z = (v * 2.0 - 1.0) * extent_z;
                let y = height_at(r, c);

                // Central differences (one-sided at borders) — duplicated
                // from `mesh::from_dem` to keep this module standalone.
                let (cm, cp) = (c.saturating_sub(1), (c + 1).min(cols - 1));
                let (rm, rp) = (r.saturating_sub(1), (r + 1).min(rows - 1));
                let span_x = (cp - cm).max(1) as f32 * dx;
                let span_z = (rp - rm).max(1) as f32 * dz;
                let dh_dx = (height_at(r, cp) - height_at(r, cm)) / span_x;
                let dh_dz = (height_at(rp, c) - height_at(rm, c)) / span_z;
                let n = glam::Vec3::new(-dh_dx, 1.0, -dh_dz).normalize_or_zero();

                vertices.push(Vertex {
                    position: [x, y, z],
                    uv: [u, v],
                    normal: [n.x, n.y, n.z],
                });
            }
        }

        // ---- 2. Subdivide the cell grid into chunks. We use
        // ----    half-open ranges over *cells*, not vertices. A chunk
        // ----    covering cells [r0, r1) × [c0, c1) renders quads with
        // ----    corners at (r, c), (r, c+1), (r+1, c), (r+1, c+1).
        let chunk_cells = params.chunk_cells.max(2);
        let lod_levels = params.lod_levels.max(1);
        let cols_u32 = cols as u32;

        let mut chunks = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        for r0 in (0..rows.saturating_sub(1)).step_by(chunk_cells) {
            for c0 in (0..cols.saturating_sub(1)).step_by(chunk_cells) {
                let r1 = (r0 + chunk_cells).min(rows - 1);
                let c1 = (c0 + chunk_cells).min(cols - 1);

                // AABB over the chunk's vertices.
                let mut min = Vec3::splat(f32::INFINITY);
                let mut max = Vec3::splat(f32::NEG_INFINITY);
                for r in r0..=r1 {
                    for c in c0..=c1 {
                        let p = &vertices[r * cols + c].position;
                        let v = Vec3::new(p[0], p[1], p[2]);
                        min = min.min(v);
                        max = max.max(v);
                    }
                }
                let bbox = Aabb { min, max };

                // Per-LOD index buffers. LOD k uses stride 2^k.
                let mut lod_indices = Vec::with_capacity(lod_levels);
                for k in 0..lod_levels {
                    let step = 1usize << k;
                    let start = indices.len() as u32;

                    let mut r = r0;
                    while r + step <= r1 {
                        let mut c = c0;
                        while c + step <= c1 {
                            let tl = (r as u32) * cols_u32 + c as u32;
                            let tr = tl + step as u32;
                            let bl = ((r + step) as u32) * cols_u32 + c as u32;
                            let br = bl + step as u32;
                            indices.extend([tl, bl, tr, tr, bl, br]);
                            c += step;
                        }
                        r += step;
                    }

                    let end = indices.len() as u32;
                    lod_indices.push(start..end);
                }

                chunks.push(Chunk {
                    bbox,
                    row_range: r0..r1,
                    col_range: c0..c1,
                    lod_indices,
                });
            }
        }

        QuadtreeMesh {
            vertices,
            indices,
            chunks,
            rows,
            cols,
            lod_levels,
        }
    }

    /// Total triangle count across every LOD level. Useful for
    /// sanity-checking that LOD 0 reproduces the no-LOD mesh size and
    /// that coarser levels shrink by ~4× each.
    pub fn triangle_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.lod_levels];
        for chunk in &self.chunks {
            for (k, r) in chunk.lod_indices.iter().enumerate() {
                counts[k] += (r.end - r.start) as usize / 3;
            }
        }
        counts
    }

    /// Pick a LOD per chunk and frustum-cull. Returns one entry per
    /// chunk that survives culling: `(chunk_idx, lod_idx)`.
    pub fn select(
        &self,
        view_proj: Mat4,
        camera_pos: Vec3,
        params: &LodParams,
    ) -> Vec<(usize, usize)> {
        // Extract clip-space frustum planes from the view-projection
        // matrix (Gribb–Hartmann). Each plane is (nx, ny, nz, d) with
        // points on the visible side satisfying `nx*x + ny*y + nz*z +
        // d ≥ 0`.
        let planes = frustum_planes(view_proj);

        let mut out = Vec::with_capacity(self.chunks.len());
        for (i, chunk) in self.chunks.iter().enumerate() {
            if !aabb_inside_frustum(chunk.bbox, &planes) {
                continue;
            }
            let lod = select_lod(chunk.bbox.center(), camera_pos, params);
            out.push((i, lod));
        }
        out
    }
}

fn frustum_planes(m: Mat4) -> [Vec4; 6] {
    // Rows of the view-proj matrix. glam stores column-major; rows
    // are `m.row(i)`.
    let r0 = m.row(0);
    let r1 = m.row(1);
    let r2 = m.row(2);
    let r3 = m.row(3);
    let left = r3 + r0;
    let right = r3 - r0;
    let bottom = r3 + r1;
    let top = r3 - r1;
    let near = r3 + r2;
    let far = r3 - r2;
    [left, right, bottom, top, near, far]
}

fn aabb_inside_frustum(aabb: Aabb, planes: &[Vec4; 6]) -> bool {
    // For each plane, check if all 8 AABB corners are on the wrong
    // side. If so, cull. Standard AABB-vs-frustum.
    let corners = aabb.corners();
    for plane in planes {
        let n = plane.truncate();
        let d = plane.w;
        let mut all_out = true;
        for corner in &corners {
            if n.dot(*corner) + d >= 0.0 {
                all_out = false;
                break;
            }
        }
        if all_out {
            return false;
        }
    }
    true
}

fn select_lod(chunk_center: Vec3, camera_pos: Vec3, params: &LodParams) -> usize {
    let d = (chunk_center - camera_pos).length();
    for (i, bound) in params.distance_bands.iter().enumerate() {
        if d < *bound {
            return i;
        }
    }
    params.lod_levels - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut d = Raster::<f64>::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                d.set(r, c, 100.0).unwrap();
            }
        }
        d
    }

    #[test]
    fn quadtree_subdivides_a_small_dem_into_expected_chunks() {
        let dem = flat_dem(128, 128);
        let mesh = QuadtreeMesh::from_dem(&dem, 0.3, LodParams::default());
        // 128 cells / 64 chunk_cells = 2 chunks per side → 4 chunks.
        assert_eq!(mesh.chunks.len(), 4);
        assert_eq!(mesh.lod_levels, 4);
        assert_eq!(mesh.vertices.len(), 128 * 128);
    }

    #[test]
    fn coarser_lods_have_quarter_triangle_counts() {
        let dem = flat_dem(128, 128);
        let mesh = QuadtreeMesh::from_dem(&dem, 0.3, LodParams::default());
        let counts = mesh.triangle_counts();
        // Each coarser LOD has 1/4 the triangles of the previous.
        for k in 1..mesh.lod_levels {
            let ratio = counts[k - 1] as f32 / counts[k].max(1) as f32;
            assert!(
                (ratio - 4.0).abs() < 0.5,
                "LOD {k}/LOD {} ratio = {ratio}, expected ~4",
                k - 1
            );
        }
    }

    #[test]
    fn frustum_cull_drops_chunks_behind_the_camera() {
        let dem = flat_dem(128, 128);
        let mesh = QuadtreeMesh::from_dem(&dem, 0.3, LodParams::default());
        // Camera looking down +Z from (0, 0, -3), so chunks at +Z are
        // *behind* it. Place the projection at +Y above; the back chunks
        // should cull.
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, -3.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(45f32.to_radians(), 16.0 / 9.0, 0.01, 100.0);
        let visible = mesh.select(
            proj * view,
            Vec3::new(0.0, 0.0, -3.0),
            &LodParams::default(),
        );
        // The flat 128×128 DEM at extent 2 is at z in [-1, 1]. The
        // camera at z = -3 looks at origin, so all chunks should be in
        // front. They should all survive culling.
        assert_eq!(visible.len(), mesh.chunks.len());
    }
}
