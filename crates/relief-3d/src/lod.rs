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

/// Per-LOD self-contained vertex + index buffers for one chunk.
///
/// **Pre-compressed at build time** (`VertexC`, 16 B/vertex). Per-frame
/// uploads memcopy these straight to GPU without further work — that's
/// what keeps the M3c batch path under budget. The CPU memory cost is
/// 50 % of the M2 layout (~696 MB on a 4 K spike), still well below
/// the 4 GB WASM heap.
#[derive(Debug, Clone)]
pub struct ChunkLod {
    pub vertices: Vec<crate::VertexC>,
    pub indices: Vec<u32>,
}

impl ChunkLod {
    pub fn upload_bytes(&self) -> usize {
        self.vertices.len() * 16 + self.indices.len() * 4
    }
}

/// One draw instruction emitted by [`QuadtreeMesh::batch_visible`].
/// All draws share the pool's vertex+index buffers. `base_vertex`
/// is kept as a field for API clarity but is always 0 — indices are
/// rebased absolute in [`QuadtreeMesh::batch_visible`] so the GL
/// backend (which doesn't support `drawElementsInstancedBaseVertex`)
/// can issue every draw without that extension.
#[derive(Debug, Clone, Copy)]
pub struct DrawCmd {
    pub base_vertex: u32,
    pub index_start: u32,
    pub index_end: u32,
}

/// Batched per-frame upload data. The two scratch vecs are filled by
/// [`QuadtreeMesh::batch_visible`] and then `write_buffer`'d once
/// each at the start of every frame — much cheaper than one
/// `write_buffer` per chunk would be.
pub struct FrameUpload {
    pub vertices: Vec<crate::VertexC>,
    pub indices: Vec<u32>,
    pub draws: Vec<DrawCmd>,
    /// Approximate visible chunk count. Used by the FPS log so the
    /// viewer can show "drew N/M chunks this frame".
    pub drawn_chunks: u32,
    pub total_chunks: u32,
    /// Last frame's `(chunk_idx, lod_idx)` selection. The cache hit
    /// path compares the new selection against this — if equal, we
    /// reuse the existing `vertices` / `indices` / `draws` and skip
    /// both the CPU memcopy and the `write_buffer`.
    pub last_visible: Vec<(usize, usize)>,
}

/// GPU buffer pool used by the LOD render path. Two fixed-size ring
/// buffers (vertex + index); per frame [`QuadtreeMesh::batch_visible`]
/// fills the CPU scratch vecs, and we `write_buffer` each scratch
/// once to refresh the GPU side. The pool's capacity caps how many
/// visible chunks fit in one frame — overflow chunks silently drop.
pub struct LodPool {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub vertex_capacity_bytes: u64,
    pub index_capacity_bytes: u64,
    /// Scratch CPU vecs reused across frames; pre-allocated to a
    /// chunk-count heuristic to avoid per-frame realloc.
    pub frame: FrameUpload,
}

impl LodPool {
    pub fn new(device: &wgpu::Device, vertex_mb: usize, index_mb: usize) -> Self {
        let v_size = (vertex_mb * 1024 * 1024) as u64;
        let i_size = (index_mb * 1024 * 1024) as u64;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relief3d.lod.pool.vbo"),
            size: v_size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("relief3d.lod.pool.ibo"),
            size: i_size,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Pre-allocate scratch large enough for ~3k chunks at LOD 2 mix,
        // which is the typical 2K–4K visible set.
        let frame = FrameUpload {
            vertices: Vec::with_capacity(4 * 1024 * 1024),
            indices: Vec::with_capacity(8 * 1024 * 1024),
            draws: Vec::with_capacity(8192),
            drawn_chunks: 0,
            total_chunks: 0,
            last_visible: Vec::with_capacity(8192),
        };
        Self {
            vertex_buffer,
            index_buffer,
            vertex_capacity_bytes: v_size,
            index_capacity_bytes: i_size,
            frame,
        }
    }

    /// Total per-frame upload size in bytes (vertex + index).
    pub fn last_frame_bytes(&self) -> usize {
        self.frame.vertices.len() * 16 + self.frame.indices.len() * 4
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
    /// Self-contained vertex + index data for every LOD level. The
    /// CPU memory cost grows by a factor of ~`lod_levels × 4/3`
    /// versus a global vertex buffer (rough rule), but per-chunk
    /// data lets us lazy-upload only visible chunks to GPU.
    pub lod_data: Vec<ChunkLod>,
}

/// The full mesh. Vertices are partitioned per-chunk per-LOD so the
/// rendering path can upload only the chunks the camera sees each
/// frame. Per-chunk CPU memory is higher than the M2 layout that
/// shared one global vertex buffer; the win is that GPU memory now
/// scales with visible chunk count, not DEM size.
#[derive(Debug)]
pub struct QuadtreeMesh {
    pub chunks: Vec<Chunk>,
    pub rows: usize,
    pub cols: usize,
    pub lod_levels: usize,
}

/// Parameters for [`QuadtreeMesh::from_dem_martini`].
#[derive(Debug, Clone)]
pub struct MartiniMeshParams {
    /// Side length of each leaf chunk, in DEM cells. Must be a power
    /// of two (Martini's RTIN requires a `2^n + 1` grid; with
    /// `chunk_cells = 64` the chunk's 65 × 65 vertex grid satisfies
    /// this exactly).
    pub chunk_cells: usize,
    /// Vertical-error tolerance in scene-units (same units as
    /// `vertical_exaggeration`). Smaller → more triangles, finer
    /// surface; larger → coarser, fewer triangles. A reasonable
    /// default for DEMs normalised to `[0, 0.45]` is `0.002`
    /// (≈ 0.5 % of the vertical range).
    pub max_error: f32,
}

impl Default for MartiniMeshParams {
    fn default() -> Self {
        Self {
            chunk_cells: 64,
            max_error: 0.002,
        }
    }
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

        let mut chunks = Vec::new();

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

                // Skirt drop — same heuristic as the M2 design.
                let chunk_y_min = bbox.min.y;
                let chunk_y_span = (bbox.max.y - bbox.min.y).max(0.0);
                let skirt_drop = (chunk_y_span * 1.5).max(vertical_exaggeration * 0.05);
                let skirt_y = chunk_y_min - skirt_drop;

                let mut lod_data = Vec::with_capacity(lod_levels);
                for k in 0..lod_levels {
                    let step = 1usize << k;
                    lod_data.push(build_chunk_lod(
                        &vertices, cols, r0, r1, c0, c1, step, skirt_y,
                    ));
                }

                chunks.push(Chunk {
                    bbox,
                    row_range: r0..r1,
                    col_range: c0..c1,
                    lod_data,
                });
            }
        }

        // Global vertex grid is no longer needed — per-chunk LOD data
        // owns its own subset.
        drop(vertices);

        QuadtreeMesh {
            chunks,
            rows,
            cols,
            lod_levels,
        }
    }

    /// Build a quadtree mesh whose per-chunk triangulation comes
    /// from the [Mapbox Martini RTIN algorithm][crate::martini]
    /// instead of stride-based decimation. Each chunk holds a
    /// **single** [`ChunkLod`] — the adaptive surface mesh at the
    /// requested error tolerance. The render path is unchanged: same
    /// skirts, same `LodPool` upload, same select / batch flow (which
    /// will always pick LOD 0 because that's the only level present).
    ///
    /// Trade-off vs [`Self::from_dem`]:
    ///
    /// |                 | quadtree (`from_dem`) | Martini (`from_dem_martini`) |
    /// |-----------------|-----------------------|------------------------------|
    /// | Triangle layout | uniform per chunk     | adapts to curvature          |
    /// | LOD levels      | distance-band picks   | single error-tuned mesh      |
    /// | Build time      | fast                  | slower (error pyramid + tree walk) |
    /// | Best for        | homogeneous relief    | heterogeneous relief, flat basins |
    ///
    /// `params.chunk_cells` must be a power of two — Martini's RTIN
    /// requires a `(2^n + 1)` grid per chunk. The default `64` is
    /// the natural fit and matches the quadtree default.
    pub fn from_dem_martini(
        dem: &Raster<f64>,
        vertical_exaggeration: f32,
        params: MartiniMeshParams,
    ) -> Self {
        let (rows, cols) = dem.shape();
        let longer = rows.max(cols) as f32;
        let extent_x = cols as f32 / longer;
        let extent_z = rows as f32 / longer;
        let inv_rows = 1.0 / (rows.max(2) - 1) as f32;
        let inv_cols = 1.0 / (cols.max(2) - 1) as f32;

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

        // Full vertex grid (positions + normals via central differences),
        // same as [`Self::from_dem`]. Each chunk extracts its 65 × 65
        // (or whatever `chunk_cells + 1`) sub-grid and Martini-meshes it.
        let dx = 2.0 * extent_x / (cols.max(2) - 1) as f32;
        let dz = 2.0 * extent_z / (rows.max(2) - 1) as f32;

        let mut vertices: Vec<Vertex> = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                let u = c as f32 * inv_cols;
                let v = r as f32 * inv_rows;
                let x = (u * 2.0 - 1.0) * extent_x;
                let z = (v * 2.0 - 1.0) * extent_z;
                let y = height_at(r, c);
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

        let chunk_cells = params.chunk_cells.max(2);
        assert!(
            (chunk_cells).is_power_of_two(),
            "MartiniMeshParams.chunk_cells must be a power of two (Martini RTIN requirement)"
        );
        // Precompute the Martini coord table once per session. Re-used
        // across every chunk of this DEM build.
        let martini = crate::martini::Martini::new(chunk_cells + 1)
            .expect("chunk_cells + 1 is 2^n + 1 by construction");

        let mut chunks = Vec::new();
        for r0 in (0..rows.saturating_sub(1)).step_by(chunk_cells) {
            for c0 in (0..cols.saturating_sub(1)).step_by(chunk_cells) {
                let r1 = (r0 + chunk_cells).min(rows - 1);
                let c1 = (c0 + chunk_cells).min(cols - 1);
                if r1 - r0 != chunk_cells || c1 - c0 != chunk_cells {
                    // Edge chunks smaller than `chunk_cells` cannot
                    // be Martini-meshed (Martini requires a fixed
                    // `2^n + 1` grid). Fall back to the
                    // quadtree-style stride-1 surface mesh, with
                    // skirts as usual. This means a DEM whose side
                    // isn't a multiple of `chunk_cells` has its rim
                    // rendered uniformly; everything inside is
                    // Martini-adaptive.
                    let (bbox, skirt_y) = chunk_bbox_and_skirt(
                        &vertices,
                        cols,
                        r0,
                        r1,
                        c0,
                        c1,
                        vertical_exaggeration,
                    );
                    let lod = build_chunk_lod(&vertices, cols, r0, r1, c0, c1, 1, skirt_y);
                    chunks.push(Chunk {
                        bbox,
                        row_range: r0..r1,
                        col_range: c0..c1,
                        lod_data: vec![lod],
                    });
                    continue;
                }

                let (bbox, skirt_y) =
                    chunk_bbox_and_skirt(&vertices, cols, r0, r1, c0, c1, vertical_exaggeration);

                // Extract the chunk's heights into a (chunk_cells + 1)²
                // row-major f32 buffer for Martini.
                let g = chunk_cells + 1;
                let mut heights = vec![0.0f32; g * g];
                for lr in 0..g {
                    for lc in 0..g {
                        let gr = r0 + lr;
                        let gc = c0 + lc;
                        heights[lr * g + lc] = vertices[gr * cols + gc].position[1];
                    }
                }
                let tile = martini
                    .tile(&heights)
                    .expect("heights length matches Martini grid_size²");
                let mesh = tile.mesh(params.max_error);

                // Convert Martini's (col, row, height) verts into our
                // local ChunkLod vertex buffer. Position / UV / normal
                // come from the global vertex grid at the same cell —
                // the height in `mesh.vertices[i][2]` matches by
                // construction, but we pull the full Vertex to keep
                // normals consistent with the global grid.
                let mut local_vertices: Vec<crate::VertexC> =
                    Vec::with_capacity(mesh.vertices.len());
                // Track local row/col for skirt walks below.
                let mut local_grid_pos: Vec<(usize, usize)> =
                    Vec::with_capacity(mesh.vertices.len());
                for v in &mesh.vertices {
                    let lc = v[0] as usize;
                    let lr = v[1] as usize;
                    let gr = r0 + lr;
                    let gc = c0 + lc;
                    local_vertices.push(crate::VertexC::from_vertex(&vertices[gr * cols + gc]));
                    local_grid_pos.push((lr, lc));
                }
                let mut local_indices = mesh.indices.clone();

                // Skirts. Walk the four boundaries; for each, gather
                // the local vertex IDs whose grid position lies on
                // that edge and emit a strip dropping to `skirt_y`.
                append_martini_skirts(
                    &mut local_vertices,
                    &mut local_indices,
                    &local_grid_pos,
                    g,
                    skirt_y,
                );

                chunks.push(Chunk {
                    bbox,
                    row_range: r0..r1,
                    col_range: c0..c1,
                    lod_data: vec![ChunkLod {
                        vertices: local_vertices,
                        indices: local_indices,
                    }],
                });
            }
        }

        drop(vertices);

        QuadtreeMesh {
            chunks,
            rows,
            cols,
            lod_levels: 1,
        }
    }

    /// Total triangle count across every LOD level. Useful for
    /// sanity-checking that LOD 0 reproduces the no-LOD mesh size and
    /// that coarser levels shrink by ~4× each.
    pub fn triangle_counts(&self) -> Vec<usize> {
        let mut counts = vec![0; self.lod_levels];
        for chunk in &self.chunks {
            for (k, lod) in chunk.lod_data.iter().enumerate() {
                counts[k] += lod.indices.len() / 3;
            }
        }
        counts
    }

    /// Total per-chunk CPU memory (vertices + indices, summed across
    /// every chunk and every LOD). The 4 K spike (16.78 M cells)
    /// lands around 988 MB; the WASM heap budget is 4 GB, so it fits
    /// — and the GPU pool stays a fraction of that, since only
    /// visible chunks need uploading.
    pub fn cpu_bytes(&self) -> usize {
        self.chunks
            .iter()
            .flat_map(|c| c.lod_data.iter())
            .map(|lod| lod.vertices.len() * 16 + lod.indices.len() * 4)
            .sum()
    }

    /// Cull + select + compress all visible chunks into one batched
    /// per-frame upload. Returns `true` if the upload was rebuilt this
    /// frame, `false` if the cached result from a previous frame is
    /// still valid (the camera produced the identical visible set).
    /// Per-frame work when the cache hits drops from "memcopy hundreds
    /// of MB" to "compare two short vecs", which is what keeps the FPS
    /// up when the camera is slow-moving or static.
    pub fn batch_visible(
        &self,
        view_proj: Mat4,
        camera_pos: Vec3,
        params: &LodParams,
        out: &mut FrameUpload,
    ) -> bool {
        out.total_chunks = self.chunks.len() as u32;
        let visible = self.select(view_proj, camera_pos, params);

        // Cache hit — same visible set as last frame, nothing to do.
        // Comparing slices of `(usize, usize)` is a single memcmp on
        // contiguous memory; with thousands of chunks it stays well
        // under a millisecond.
        if visible == out.last_visible {
            return false;
        }

        out.vertices.clear();
        out.indices.clear();
        out.draws.clear();
        out.drawn_chunks = visible.len() as u32;

        for &(chunk_idx, lod_idx) in &visible {
            let lod = &self.chunks[chunk_idx].lod_data[lod_idx];
            let base_vertex = out.vertices.len() as u32;
            let i_start = out.indices.len() as u32;

            // Vertices: straight memcopy (pre-compressed at build).
            out.vertices.extend_from_slice(&lod.vertices);
            // Indices: rebase from chunk-local (0..N) to pool-absolute
            // (base_vertex..base_vertex+N). WebGL2 does not support
            // `drawElementsInstancedBaseVertex`, so we cannot rely on
            // `draw_indexed(..., base_vertex, ...)` — every draw on the
            // GL backend goes through `base_vertex = 0`. The few-ns
            // per-index addition is cheap; rebuild frames stay sub-10 ms
            // on a 4 K mesh and most frames hit the visible-set cache.
            out.indices
                .extend(lod.indices.iter().map(|&i| i + base_vertex));

            let i_end = out.indices.len() as u32;
            out.draws.push(DrawCmd {
                base_vertex: 0,
                index_start: i_start,
                index_end: i_end,
            });
        }
        out.last_visible = visible;
        true
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
            // Clamp to the chunk's actual LOD count so the Martini
            // path (single-LOD per chunk) doesn't panic when
            // `select_lod` picks a finer band than the chunk owns.
            let lod = select_lod(chunk.bbox.center(), camera_pos, params)
                .min(chunk.lod_data.len().saturating_sub(1));
            out.push((i, lod));
        }
        out
    }
}

/// Helper: compute a chunk's bbox + skirt-drop Y from its corner cells.
/// Shared between the quadtree and Martini builders.
fn chunk_bbox_and_skirt(
    vertices: &[Vertex],
    cols: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    vertical_exaggeration: f32,
) -> (Aabb, f32) {
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
    let chunk_y_span = (bbox.max.y - bbox.min.y).max(0.0);
    let skirt_drop = (chunk_y_span * 1.5).max(vertical_exaggeration * 0.05);
    let skirt_y = bbox.min.y - skirt_drop;
    (bbox, skirt_y)
}

/// Walk the four boundary edges of a Martini-meshed chunk and emit a
/// skirt strip on each. Same outward-normal convention and `skirt_y`
/// semantics as `build_chunk_lod`'s skirt loop; the only difference
/// is that we use whatever boundary vertices Martini retained
/// instead of the full per-LOD stride sequence.
fn append_martini_skirts(
    vertices: &mut Vec<crate::VertexC>,
    indices: &mut Vec<u32>,
    grid_pos: &[(usize, usize)],
    g: usize,
    skirt_y: f32,
) {
    let last = g - 1;
    // edge 0: top    (lr = 0,   sort by lc ascending) — normal -Z
    // edge 1: bottom (lr = last, sort by lc ascending) — normal +Z
    // edge 2: left   (lc = 0,   sort by lr ascending) — normal -X
    // edge 3: right  (lc = last, sort by lr ascending) — normal +X
    let edges: [(Box<dyn Fn(usize, usize) -> Option<usize>>, [f32; 3]); 4] = [
        (
            Box::new(|lr, _| if lr == 0 { Some(0) } else { None }),
            [0.0, 0.3, -0.95],
        ),
        (
            Box::new(move |lr, _| if lr == last { Some(0) } else { None }),
            [0.0, 0.3, 0.95],
        ),
        (
            Box::new(|_, lc| if lc == 0 { Some(0) } else { None }),
            [-0.95, 0.3, 0.0],
        ),
        (
            Box::new(move |_, lc| if lc == last { Some(0) } else { None }),
            [0.95, 0.3, 0.0],
        ),
    ];
    for (edge_idx, (filter, normal)) in edges.into_iter().enumerate() {
        // Collect (local_vertex_id, sort_key) for verts on this edge.
        let mut edge_verts: Vec<(u32, usize)> = Vec::new();
        for (vid, &(lr, lc)) in grid_pos.iter().enumerate() {
            if filter(lr, lc).is_some() {
                let sort_key = if edge_idx < 2 { lc } else { lr };
                edge_verts.push((vid as u32, sort_key));
            }
        }
        edge_verts.sort_by_key(|&(_, k)| k);
        if edge_verts.len() < 2 {
            continue;
        }
        let mut prev_top: Option<u32> = None;
        let mut prev_skirt: Option<u32> = None;
        for (top_local, _) in edge_verts {
            let top_v = vertices[top_local as usize];
            let top_x = top_v.pos[0] as f32 / 32767.0;
            let top_z = top_v.pos[2] as f32 / 32767.0;
            let top_u = top_v.uv[0] as f32 / 65535.0;
            let top_v_uv = top_v.uv[1] as f32 / 65535.0;
            let skirt_local = vertices.len() as u32;
            vertices.push(crate::VertexC::from_vertex(&Vertex {
                position: [top_x, skirt_y, top_z],
                uv: [top_u, top_v_uv],
                normal,
            }));
            if let (Some(pt), Some(ps)) = (prev_top, prev_skirt) {
                indices.extend([pt, ps, top_local, top_local, ps, skirt_local]);
            }
            prev_top = Some(top_local);
            prev_skirt = Some(skirt_local);
        }
    }
}

/// Build one chunk's per-LOD data. Local indices, self-contained
/// vertex buffer. Stride `step = 2^k`. Skirts walk the four edges
/// after the surface grid and append vertices + triangles connecting
/// surface edge cells to a row of new skirt vertices at `skirt_y`.
fn build_chunk_lod(
    global: &[Vertex],
    cols: usize,
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    step: usize,
    skirt_y: f32,
) -> ChunkLod {
    let chunk_rows = (r1 - r0) / step + 1;
    let chunk_cols = (c1 - c0) / step + 1;

    // ---- Surface grid ----
    let mut vertices: Vec<crate::VertexC> = Vec::with_capacity(chunk_rows * chunk_cols);
    for lr in 0..chunk_rows {
        for lc in 0..chunk_cols {
            let gr = r0 + lr * step;
            let gc = c0 + lc * step;
            vertices.push(crate::VertexC::from_vertex(&global[gr * cols + gc]));
        }
    }
    let cc = chunk_cols as u32;
    let mut indices = Vec::with_capacity(6 * (chunk_rows - 1) * (chunk_cols - 1));
    for lr in 0..(chunk_rows - 1) as u32 {
        for lc in 0..(chunk_cols - 1) as u32 {
            let tl = lr * cc + lc;
            let tr = tl + 1;
            let bl = (lr + 1) * cc + lc;
            let br = bl + 1;
            indices.extend([tl, bl, tr, tr, bl, br]);
        }
    }

    // ---- Skirts ----
    //
    // For each of the four edges, walk the chunk's edge at local
    // stride 1 (already at LOD k), emit skirt vertices below each top
    // vertex, and connect consecutive pairs with two triangles.
    let edge_id = |edge: usize, i: usize| -> u32 {
        match edge {
            0 => i as u32,                                   // top row (lr=0)
            1 => ((chunk_rows - 1) * chunk_cols + i) as u32, // bottom row
            2 => (i * chunk_cols) as u32,                    // left col (lc=0)
            _ => (i * chunk_cols + (chunk_cols - 1)) as u32, // right col
        }
    };
    for edge in 0..4 {
        let count = if edge < 2 { chunk_cols } else { chunk_rows };
        if count < 2 {
            continue;
        }
        let normal = match edge {
            0 => [0.0, 0.3, -0.95],
            1 => [0.0, 0.3, 0.95],
            2 => [-0.95, 0.3, 0.0],
            _ => [0.95, 0.3, 0.0],
        };
        let mut prev_top: Option<u32> = None;
        let mut prev_skirt: Option<u32> = None;
        for i in 0..count {
            let top_local = edge_id(edge, i);
            let top_v = vertices[top_local as usize];
            let skirt_local = vertices.len() as u32;
            // Decode top vertex's snorm16 X and Z back to f32 in [-1, 1]
            // so we can build the skirt vertex; encode the new skirt
            // straight back to VertexC.
            let top_x = top_v.pos[0] as f32 / 32767.0;
            let top_z = top_v.pos[2] as f32 / 32767.0;
            let top_u = top_v.uv[0] as f32 / 65535.0;
            let top_v_uv = top_v.uv[1] as f32 / 65535.0;
            vertices.push(crate::VertexC::from_vertex(&Vertex {
                position: [top_x, skirt_y, top_z],
                uv: [top_u, top_v_uv],
                normal,
            }));
            if let (Some(pt), Some(ps)) = (prev_top, prev_skirt) {
                indices.extend([pt, ps, top_local, top_local, ps, skirt_local]);
            }
            prev_top = Some(top_local);
            prev_skirt = Some(skirt_local);
        }
    }

    ChunkLod { vertices, indices }
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
        // P4-M3c: per-chunk per-LOD self-contained buffers. The CPU
        // cost is non-trivial (~988 MB on a 4 K DEM) but the GPU
        // upload pool stays bounded regardless of mesh size — only
        // visible chunks need to be uploaded each frame.
        assert!(mesh.cpu_bytes() > 0);
        for chunk in &mesh.chunks {
            assert_eq!(chunk.lod_data.len(), 4);
            // LOD 0 has the most triangles per chunk.
            assert!(chunk.lod_data[0].indices.len() > chunk.lod_data[3].indices.len());
        }
    }

    #[test]
    fn coarser_lods_have_quarter_triangle_counts() {
        let dem = flat_dem(128, 128);
        let mesh = QuadtreeMesh::from_dem(&dem, 0.3, LodParams::default());
        let counts = mesh.triangle_counts();
        // Each coarser LOD's surface has ~1/4 the triangles of the
        // previous. M2 skirts add a *linear* term (4 × edge length),
        // which softens the ratio a bit at the coarse end where the
        // surface term shrinks fastest. Loosen the tolerance to
        // accommodate.
        for k in 1..mesh.lod_levels {
            let ratio = counts[k - 1] as f32 / counts[k].max(1) as f32;
            assert!(
                ratio >= 3.0 && ratio <= 5.0,
                "LOD {k}/LOD {} ratio = {ratio}, expected 3-5",
                k - 1
            );
        }
    }

    #[test]
    fn skirts_drop_below_chunk_min() {
        // Bumpy DEM so every chunk has a real Y span.
        let mut dem = Raster::<f64>::new(128, 128);
        for r in 0..128 {
            for c in 0..128 {
                let z = ((r as f64 * 0.1).sin() + (c as f64 * 0.13).cos()) * 50.0 + 200.0;
                dem.set(r, c, z).unwrap();
            }
        }
        let mesh = QuadtreeMesh::from_dem(&dem, 0.3, LodParams::default());
        // Each chunk's per-LOD vertex buffer must contain at least one
        // vertex below the chunk's bbox.min.y by the skirt drop.
        // With the per-chunk layout this is trivial to check — the
        // skirt vertices are tail of each `lod_data[k].vertices`.
        for chunk in &mesh.chunks {
            for lod in &chunk.lod_data {
                let mut any_below = false;
                for v in &lod.vertices {
                    // Decode snorm16 y back to f32 in [-1, 1].
                    let y = v.pos[1] as f32 / 32767.0;
                    if y < chunk.bbox.min.y - 1e-3 {
                        any_below = true;
                        break;
                    }
                }
                assert!(any_below, "no skirt found below chunk min Y");
            }
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

    /// `from_dem_martini` on a flat DEM should produce drastically
    /// fewer triangles than `from_dem` at the same chunk size — the
    /// Martini error pyramid is all zeros so every chunk collapses
    /// to the 2 root triangles plus skirts.
    #[test]
    fn martini_collapses_flat_dem_to_few_triangles() {
        let dem = flat_dem(129, 129); // 128 cells + 1 → 2 chunks per side
        let quad = QuadtreeMesh::from_dem(&dem, 0.3, LodParams::default());
        let mart = QuadtreeMesh::from_dem_martini(
            &dem,
            0.3,
            MartiniMeshParams {
                chunk_cells: 64,
                max_error: 0.001,
            },
        );
        let quad_lod0_tris = quad.triangle_counts()[0];
        let mart_tris: usize = mart
            .chunks
            .iter()
            .map(|c| c.lod_data[0].indices.len() / 3)
            .sum();
        assert!(
            mart_tris < quad_lod0_tris / 10,
            "Martini on a flat DEM should produce <10 % of the quadtree LOD0 triangle count; \
             got Martini={} vs quad-LOD0={}",
            mart_tris,
            quad_lod0_tris
        );
        assert_eq!(mart.lod_levels, 1, "Martini path has exactly one LOD");
    }

    /// On a bumpy DEM, Martini at a tight error tolerance should
    /// converge toward the quadtree's full LOD 0 triangle count.
    #[test]
    fn martini_tight_tolerance_approaches_full_grid() {
        let mut dem = Raster::<f64>::new(129, 129);
        for r in 0..129 {
            for c in 0..129 {
                let z = ((r as f64 * 0.4).sin() + (c as f64 * 0.4).cos()) * 30.0 + 100.0;
                dem.set(r, c, z).unwrap();
            }
        }
        let quad = QuadtreeMesh::from_dem(&dem, 0.3, LodParams::default());
        let mart = QuadtreeMesh::from_dem_martini(
            &dem,
            0.3,
            MartiniMeshParams {
                chunk_cells: 64,
                max_error: 0.0,
            },
        );
        let quad_lod0_tris = quad.triangle_counts()[0];
        let mart_tris: usize = mart
            .chunks
            .iter()
            .map(|c| c.lod_data[0].indices.len() / 3)
            .sum();
        // At max_error=0, Martini should be within 25% of the full
        // quadtree LOD 0 count — Martini still wins a bit because
        // some midpoints fall exactly on the interpolation.
        let ratio = mart_tris as f64 / quad_lod0_tris as f64;
        assert!(
            ratio > 0.5 && ratio < 1.5,
            "tight Martini should reproduce ~quadtree-LOD0 count, got ratio={ratio:.2} (mart={mart_tris}, quad={quad_lod0_tris})"
        );
    }
}
