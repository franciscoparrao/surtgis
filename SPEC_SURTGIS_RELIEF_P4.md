# SPEC: `surtgis-relief-3d` P4 — adaptive LOD for production-scale DEMs

**Status:** Proposed
**Target version:** 0.14.0
**Author of spec:** handoff doc — implements the only remaining named
rayshader gap from `SPEC_SURTGIS_RELIEF_P3.md` §6 ("LOD adaptativo
for DEMs ≥ 2K side, P4 sprint candidate"). Path-tracing stays out of
scope, as in P3.

> **Methodological reminder.** The 2D spec's §12.6 documented a class
> of failure: spike configurations that under-estimate the production
> workload by 5–8×, which then poisons every downstream milestone.
> P2's M1 spike answered this with a 1 M-vertex render. P3 hit it
> again on M1 haze (the spec case is grazing-light, not overhead).
> **P4 raises the bar again.** The M1 spike must render a **synthetic
> 4 K × 4 K DEM (16 M cells)** in the native viewer at ≥ 30 FPS
> sustained. `dem_filled.tif` (363 K vertices) already renders at
> 60 FPS without any LOD work; if the spike measures that DEM, every
> milestone after will inherit the false confidence that "no LOD is
> needed". The spike case has to break the no-LOD path first.

---

## 0. Why now

After v0.13.1, `surtgis-relief-3d` covers the rayshader feature
surface on every named axis except path-tracing and LOD. The LOD gap
is not aesthetic — it's a hard ceiling:

| DEM side | Cells | Mesh vertex memory | GPU constraint |
|---|---|---|---|
| 600 (current `dem_filled.tif`) | 360 K | 11.5 MB | fine everywhere |
| 1 K | 1 M | 32 MB | fine, 60 FPS native + browser (P2 M1 spike) |
| 2 K | 4 M | 128 MB | fine native, browser WebGL2 budget under stress |
| 4 K | 16 M | 512 MB | **native scrolling**, browser **OOM** |
| 8 K | 64 M | 2 GB | breaks everywhere without LOD |

The 4 K case is where most national / continental DEMs land (e.g. a
60 km × 60 km tile of cop-dem-glo-30 at 30 m resolution = 2 K × 2 K;
a 120 km × 120 km tile = 4 K × 4 K). Today these users either
pre-downsample before feeding our viewer or use rayshader (which
ships a `reduce_matrix_size` helper for exactly this problem).

P4 ships an adaptive LOD pipeline that:

- Renders DEMs up to 4 K × 4 K side at ≥ 30 FPS in the native viewer.
- Renders the same DEMs in the WebGL2 browser viewer without OOM.
- Stays bit-equivalent to the no-LOD path in the limit where the
  whole DEM fits in the screen-space-error budget.

LOD is the *only* remaining "rayshader wins" line that has a
measurable user impact. After P4, the only honest gap is
path-tracing (which stays out of scope; users wanting that figure
already know to reach for rayshader).

---

## 1. Goal

Quadtree-based LOD with skirts. Per-chunk vertex buffers chosen at
runtime from a small set of pre-built decimation levels. Screen-
space error metric drives chunk selection. Frustum culling drops
chunks the camera does not see.

### Scope (P4 — this spec)

| Deliverable | Cost | Milestone |
|---|---|---|
| Quadtree subdivision + per-chunk multi-LOD vertex buffers | 4 days | M1 |
| Screen-space error metric + chunk selection | 2 days | M1 |
| Frustum culling | 1 day | M1 |
| Skirts at chunk boundaries (no visible cracks) | 3 days | M2 |
| WASM browser path (WebGL2, memory budget) | 4 days | M3 |
| CLI headless path with LOD | 2 days | M4 |

Total estimated effort: ~2 weeks focused work. The spec-original
estimate (P3 §2 table) was 1–2 sprints; the upper bound holds if
M2 skirts run into degenerate edge cases.

### Non-goals for P4

- **No CDLOD / continuous LOD.** Discrete LOD levels with skirts is
  the standard technique and is enough; CDLOD would smooth
  transitions further but the vertex-shader complexity is not worth
  the perceptual gain for landscape relief.
- **No GPU-driven LOD / mesh shaders.** WebGL2 cannot support this;
  building a different LOD pipeline for native vs WASM is more code
  than the value justifies.
- **No streaming load** (DEMs that don't fit in RAM). DEMs up to 4 K
  side fit comfortably; 8 K-and-up streaming is a separate sprint if
  ever needed.
- **No virtual texturing** for the relief raster (we keep uploading
  the full RGBA texture). 4 K × 4 K × 4 = 64 MB texture, fine in
  WebGL2.
- **No path-tracing.** As in P3.

---

## 2. What we reuse

| Need | Source |
|---|---|
| Mesh primitives | extend `crates/relief-3d/src/mesh.rs` |
| Camera + view-proj | `OrbitCamera` from P2 |
| Pipeline / shader | extend P3's `relief.wgsl` |
| DEM I/O | `surtgis_core::io::read_geotiff` |
| Relief texture | `surtgis_relief::ReliefBuilder` (unchanged) |

---

## 3. Crate layout

Extensions only. No new crates.

```
crates/relief-3d/
├── src/
│   ├── mesh.rs        # add `from_dem_quadtree(dem, params) -> QuadtreeMesh`
│   ├── lod.rs         # NEW — Quadtree, Chunk, screen-space error, culling
│   ├── pipeline.rs    # build pipeline still works on a Vec<Vertex>
│   ├── native.rs      # iterate chunks each frame, issue per-chunk draw calls
│   ├── web.rs         # mirror
│   ├── headless.rs    # mirror
│   └── lib.rs
└── examples/
    └── spike_lod_4k.rs  # NEW — synthetic 4K DEM, M1 acceptance bar
```

`QuadtreeMesh` owns:

- The full per-cell vertex grid at the highest LOD level.
- Per-chunk view of the vertex grid + an index buffer per LOD level
  (1×, 2×, 4×, 8×).
- A bounding box per chunk for culling and screen-space error.

Each frame:

1. For each chunk, compute screen-space error from camera distance
   and bounding-box screen-space projected size.
2. Pick the lowest LOD level whose error stays under the budget.
3. Frustum-cull chunks outside the view.
4. Issue one draw call per surviving chunk with the chosen index
   buffer.

---

## 4. Public API (extensions, no breakages)

```rust
/// Configuration for the LOD pipeline.
pub struct LodParams {
    /// Chunk size in cells. Default 64.
    pub chunk_cells: usize,
    /// Maximum decimation factor (1 = full res). Default 8.
    pub max_decimation: usize,
    /// Pixel error budget. Cells whose projected error exceeds this
    /// are split to a finer LOD. Default 1.5 px (a touch above one
    /// pixel; below this you hit the depth-buffer noise floor).
    pub pixel_error_budget: f32,
}

impl Default for LodParams { /* ... */ }

/// Build a quadtree mesh from a DEM. The mesh exposes chunks at
/// multiple LOD levels; the viewer picks per chunk each frame.
pub fn from_dem_quadtree(
    dem: &Raster<f64>,
    vertical_exaggeration: f32,
    params: LodParams,
) -> QuadtreeMesh;

/// One chunk's worth of mesh data + its bounding box.
pub struct Chunk {
    pub bbox_world: AABB,
    pub vertex_offset: u32,
    pub vertex_count: u32,
    /// One index slice per LOD level.
    pub lod_indices: SmallVec<[Range<u32>; 4]>,
}

/// Full mesh for the viewer to consume.
pub struct QuadtreeMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub chunks: Vec<Chunk>,
}
```

`native::run_viewer` and the WASM `run_relief3d_canvas` gain
overloads that take a `QuadtreeMesh` instead of `Vec<Vertex>`. The
old `Vec<Vertex>` path stays for backwards compat and for small
DEMs where LOD is unnecessary overhead.

---

## 5. Milestones

### M1 — Quadtree spike (4 K × 4 K @ ≥ 30 FPS)

**Acceptance bar:** A synthetic 4 K × 4 K DEM renders at ≥ 30 FPS
sustained in the native viewer with the camera orbiting. The DEM is
generated procedurally in `examples/spike_lod_4k.rs` so the test
does not depend on having a real big DEM in the repo. Frustum
culling and chunk selection must be live (i.e. the spike must
actually exercise LOD; rendering 16 M vertices at full resolution
without LOD is what we're trying to escape).

Sub-tasks:

- M1.1 — `crates/relief-3d/src/lod.rs` with `Quadtree`, `Chunk`,
  `AABB`. Quadtree built by recursive subdivision until each leaf is
  ≤ `chunk_cells` square.
- M1.2 — `mesh::from_dem_quadtree` builds per-chunk vertex slices
  and one index buffer per LOD level. Level k uses every 2^k-th cell.
- M1.3 — Screen-space error per chunk: project the chunk's world-
  space AABB to NDC, take its longest screen-space dimension, divide
  by the chunk's cell count. If error > budget, split.
- M1.4 — Frustum culling: drop chunks whose AABB does not intersect
  the camera frustum.
- M1.5 — Native viewer iterates chunks each frame, issues per-chunk
  draw calls.
- M1.6 — `examples/spike_lod_4k.rs` generates a 4 K × 4 K
  procedural DEM (sum of cosines + a bit of noise) and runs the
  viewer. Logs FPS like the existing M1 spike from P2.

### M2 — Skirts (no visible cracks)

**Acceptance bar:** Orbit the camera through any LOD transition zone
on the 4 K DEM; no visible cracks appear. The check is qualitative —
"do I see seams?" — but the implementation is deterministic
(extend each chunk's edge triangles downward by some height under
the lowest neighbouring chunk vertex).

Sub-tasks:

- M2.1 — Per-chunk skirt vertex generation: 4 edge strips, each
  dropping below the chunk's bottom edge.
- M2.2 — Skirt vertices share UV with their parent edge so the
  texture continues seamlessly.
- M2.3 — Skirts render with the same shader; no special pass.

### M3 — WASM browser path

**Acceptance bar:** A 4 K × 4 K DEM loads + renders in Firefox at
≥ 20 FPS without OOM. GPU memory must stay under 256 MB (the
typical browser budget).

Sub-tasks:

- M3.1 — Same `QuadtreeMesh` builder works on wasm32. No new code;
  the cfg-split that's already in `mesh.rs` for rayon vs sequential
  carries forward.
- M3.2 — `web::run_relief3d_canvas` gains a `lod: bool` parameter.
  When true, build the quadtree and run the LOD pipeline.
- M3.3 — `surtgis-demo/relief3d-wgpu.html` exposes a "LOD on/off"
  toggle plus a synthetic 4 K DEM button for the demo.

### M4 — CLI headless path

**Acceptance bar:** `surtgis relief-3d big_dem.tif --output PNG`
where `big_dem.tif` is 4 K × 4 K runs in under 30 s wall-clock and
produces a visually plausible relief PNG. The headless path used to
upload the full mesh; with LOD it uploads chunks adaptively.

Sub-tasks:

- M4.1 — `headless::render_to_rgba` accepts a `QuadtreeMesh`. The
  camera-fixed chunk selection happens once (no animation), so the
  per-chunk draw calls happen in a single pass.
- M4.2 — `surtgis relief-3d --lod` flag (default true for DEMs
  larger than 1 K side).

---

## 6. Risks

- **Skirts can fail in pathological terrain.** Very narrow
  chunks with steep edges may have skirt vertices that the camera
  sees from below. Mitigation: clamp skirt drop distance to the
  chunk's elevation range, not a fixed amount.

- **Screen-space error metric is taste.** Too tight and we draw
  everything at max LOD (no perf win). Too loose and the eye picks
  up the LOD transitions even with skirts. Pick the budget that
  reads as "no visible LOD" at default camera distance on the
  spike DEM; users can override via `LodParams`.

- **WASM memory budget is hostile.** WebGL2 browsers vary from
  128 MB to 512 MB GPU budget. The synthetic 4 K DEM at full LOD
  uploaded would already exhaust the low end. Mitigation: chunk
  upload is lazy — only chunks selected for rendering are sent to
  GPU. The full vertex grid lives in WASM memory (which has a
  separate, more generous budget).

- **The `LodParams::chunk_cells = 64` default is unvalidated.**
  Smaller chunks → more draw calls per frame; larger chunks →
  coarser LOD granularity, more wasted work in each chunk. 64 is
  the rayshader convention; M1 must measure it.

---

## 7. One-paragraph summary for the implementer

P4 extends `surtgis-relief-3d` with a quadtree LOD pipeline so DEMs
that don't fit in a single full-resolution mesh upload still render
fluidly. Build a quadtree per DEM, each leaf chunk gets index
buffers at 1×, 2×, 4×, 8× decimation, and per frame the viewer
picks the lowest LOD whose screen-space projected error stays under
~1.5 px while frustum-culling chunks the camera does not see.
Skirts hide the cracks at LOD transitions. The native, WASM, and
headless paths all consume the same `QuadtreeMesh`; only the wgpu
device differs. **The M1 spike must render a synthetic 4 K × 4 K
DEM at ≥ 30 FPS sustained**, not `dem_filled.tif` — measuring the
small case first is the §12.6 failure mode that has poisoned every
prior spec, and P4 has the most to lose from it.
