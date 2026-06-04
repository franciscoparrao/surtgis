# SPEC: `surtgis-relief-3d` — Native wgpu 3D shaded-relief viewer

**Status:** Proposed
**Target version:** 0.12.0
**Author of spec:** handoff doc — implements P2 of `SPEC_SURTGIS_RELIEF.md`
**Crate name:** `surtgis-relief-3d` (workspace member `crates/relief-3d`)

> **Methodological note** — `SPEC_SURTGIS_RELIEF.md` §12.6 documented a
> spec failure: its spike measured the cheapest configuration
> (3 azimuths × 1 altitude) and underestimated the production workload
> (1 azimuth × 11 altitudes) by 8×. This spec applies that lesson: the
> M1 spike must render the **production-workload mesh size** (≥1M
> vertices, i.e. a ~1K×1K DEM), not just the convenient `dem_filled.tif`
> (637×570 = 363K vertices). If the spike only validates the small
> case, milestones M2–M5 will inherit the same misscoping.

---

## 0. Context

`surtgis-relief` (v0.11.0) ships a 2D shaded-relief composite that beats
rayshader by 6.99× on the standard recipe and runs natively in CLI,
WASM, and Python. The 2D output (RGBA texture) has been validated to
look correct under 3D mesh display via a Three.js + WASM preview at
`surtgis-demo/relief3d.html`.

The preview proves the 2D texture layer is not the weak link. What is
missing for the "single binary 3D viewer in the browser" story is a
**native wgpu pipeline** that replaces Three.js. wgpu compiles to:
- **Native** (desktop) via Vulkan / Metal / DX12 backends
- **Browser** via WebGPU (Chrome, Edge) and falls back to WebGL2

This is the categorical differentiator: rayshader 3D is R + `rgl` =
desktop OpenGL only. surtgis-relief-3d targets both desktop *and*
browser from one codebase.

---

## 1. Goal

Interactive 3D viewer for shaded relief. DEM → triangle mesh, texture
draping from `surtgis-relief::ReliefBuilder`, orbit camera, directional
light. Compiles to native window (winit) and browser canvas (WebGPU).

### Scope (P2 — this spec)

| Deliverable | Effort estimate |
|---|---|
| `surtgis-relief-3d` crate with wgpu pipeline | ~2 weeks |
| Mesh generation: DEM → vertex buffer (one vertex per cell) | M2 |
| Texture draping from `ReliefBuilder.render()` output | M3 |
| Orbit camera + directional light | M3 |
| Native desktop window (winit) | M4 |
| Browser embed in `surtgis-demo/` (WebGPU + WebGL2 fallback) | M4 |
| Headless screenshot (`surtgis relief-3d --output preview.png`) | M5 |

### Non-goals for P2

- **No LOD.** MVP renders full-resolution mesh up to ~1M vertices
  (1K × 1K DEM). Adaptive LOD is a separate sprint (P3 candidate).
- **No advanced shading.** Single directional light, no PBR, no
  reflections, no fog, no atmospheric perspective. The 2D texture
  already bakes shadows + ambient occlusion via `relief_compute` —
  the 3D layer only adds the rendering surface, not extra lighting.
- **No GLB/glTF export.** Useful, but a separate sprint. The viewer is
  the MVP.
- **No water surface effects.** The 2D water mask is the only water
  representation.
- **No native iOS/Android targets.** wgpu supports them, but
  packaging for mobile is out of scope.

---

## 2. What we reuse

The 2D side is fully built. The 3D side reuses:

| 3D need | Source |
|---|---|
| Texture data (RGBA) | `surtgis_relief::ReliefBuilder.render() → RgbaImage` |
| DEM read | `surtgis_core::io::read_geotiff` |
| Mesh geometry | new code in this crate — straightforward grid mesh |
| Camera math | `glam` crate (workspace dep) |
| wgpu setup | `wgpu` crate + boilerplate from wgpu-rs tutorials |

---

## 3. Crate layout

```
crates/relief-3d/
├── Cargo.toml
├── shaders/
│   └── relief.wgsl       # vertex + fragment shader, WGSL source
└── src/
    ├── lib.rs            # ReliefViewer, ViewerConfig, public surface
    ├── mesh.rs           # DEM → vertex buffer
    ├── camera.rs         # orbit camera + matrices
    ├── pipeline.rs       # wgpu device/queue/render-pipeline setup
    ├── texture.rs        # RGBA → wgpu::Texture upload
    ├── native.rs         # winit window wrapper (cfg native)
    └── web.rs            # canvas wrapper (cfg wasm32)
```

### `Cargo.toml`

```toml
[package]
name = "surtgis-relief-3d"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "Native wgpu 3D viewer for SurtGis shaded relief"

[dependencies]
surtgis-core   = { version = "0.11", path = "../core", default-features = false }
surtgis-relief = { version = "0.11", path = "../relief", default-features = false }
wgpu           = "23"
bytemuck       = { version = "1", features = ["derive"] }
glam           = "0.29"
thiserror.workspace = true

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
winit          = "0.30"
pollster       = "0.4"     # block_on for native main loop bootstrap

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen   = "0.2"
web-sys        = { version = "0.3", features = [
  "HtmlCanvasElement", "Window", "Document",
] }
```

**Crate is opt-in** — not pulled by `surtgis-relief` core users. CLI
gets a `relief-3d` feature flag that pulls this in only when the user
asks for the 3D subcommand.

---

## 4. Public API

```rust
/// Configuration for a viewer instance.
pub struct ViewerConfig {
    pub vertical_exaggeration: f32,
    pub camera_distance: f32,
    pub camera_polar_deg: f32,        // 0 = top-down, 90 = side
    pub camera_azimuth_deg: f32,
    pub light_azimuth_deg: f32,
    pub light_altitude_deg: f32,
    pub light_intensity: f32,         // 0.0–1.0
}

impl Default for ViewerConfig { /* sensible mountain-DEM defaults */ }

/// A buildable, runnable 3D relief viewer.
pub struct ReliefViewer<'a> {
    dem: &'a Raster<f64>,
    texture: RgbaImage,
    config: ViewerConfig,
}

impl<'a> ReliefViewer<'a> {
    pub fn new(dem: &'a Raster<f64>, texture: RgbaImage) -> Self;
    pub fn with_config(mut self, cfg: ViewerConfig) -> Self;

    /// Native: open a winit window and run the event loop. Blocks
    /// until the user closes the window.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run_native(self) -> Result<(), ReliefError>;

    /// WASM: attach to an HTMLCanvasElement, run via
    /// requestAnimationFrame.
    #[cfg(target_arch = "wasm32")]
    pub fn run_canvas(self, canvas_id: &str) -> Result<(), ReliefError>;

    /// Render one frame headlessly to RGBA bytes (for screenshot CLI).
    pub fn render_to_rgba(self, width: u32, height: u32) -> Result<Vec<u8>, ReliefError>;
}
```

---

## 5. CLI integration

New subcommand `surtgis relief-3d` (gated `cfg(feature = "relief-3d")`
on the CLI crate):

```
surtgis relief-3d DEM.tif \
    --colormap terrain \
    --shadows --ambient \
    --vertical-exaggeration 1.5 \
    --output preview.png      # headless PNG screenshot (otherwise opens viewer)
```

When `--output` is omitted, opens an interactive native window.

---

## 6. Milestones

| M | Deliverable | Acceptance |
|---|---|---|
| **M1 — wgpu spike** | crates/relief-3d skeleton; render a 1024×1024 textured plane (1M vertices) to a winit window @ ≥60 FPS | **MUST hit 60 FPS at 1M vertices**, not just at 363K. This is the §12.6 lesson application. |
| **M2 — DEM mesh + texture** | Real DEM → mesh with displacement, RGBA texture from `ReliefBuilder` applied, orbit camera works | Renders `dem_filled.tif` interactively |
| **M3 — Lighting + polish** | Directional light, basic normal calc in shader, configurable vertical exaggeration | Looks comparable to Three.js demo |
| **M4 — WASM/browser** | Same code compiles to WebGPU + WebGL2 fallback, embedded in `surtgis-demo/` | Browser viewer works in Chrome (WebGPU) and Firefox (WebGL2) |
| **M5 — Headless + CLI** | `render_to_rgba()` + `surtgis relief-3d --output PNG` subcommand | CI-friendly screenshot generation |

Total estimated effort: ~2 weeks focused work. Spec-original estimate
was 3–4 weeks; the Three.js validation already de-risked the 2D-to-3D
texture path.

---

## 7. Risks

- **WebGPU browser support is uneven.** Chrome/Edge ship it stable;
  Firefox is opt-in (2026-06); Safari is in TP. Fallback to WebGL2 via
  wgpu's `gles` backend covers ~98% of users.
- **wgpu API churn.** wgpu 0.20→23 changed surfaces, command encoders,
  and `RequestAdapterOptions`. Pin to `wgpu = "23"` and update with a
  bump.
- **Memory at full-res mesh.** 1M vertices × (12B position + 8B uv +
  12B normal) ≈ 32 MB GPU. Fine. 4K-side DEM would be ~512 MB and
  needs LOD — out of scope.
- **Native vs WASM event loop divergence.** winit and browser RAF
  have different driving models. The `cfg`-split between
  `native.rs` and `web.rs` is the established pattern; keep the
  shared rendering code in `pipeline.rs`.

---

## 8. One-paragraph summary for the implementer

`surtgis-relief-3d` is a wgpu pipeline that takes a DEM and a
pre-computed RGBA texture from `surtgis-relief` and renders them as a
displaced, textured mesh in a winit window or a browser canvas. Use
the same wgpu device/queue/pipeline code on both targets; cfg-split
only the windowing layer. Build a grid mesh with one vertex per DEM
cell (no LOD in P2). Vertex shader displaces by sampled DEM height;
fragment shader samples the relief texture and modulates by a single
directional light. Camera is an orbit controller around the DEM
centre. M1 spike **must** render a 1024×1024 = 1M-vertex mesh at
60 FPS before claiming the wgpu pipeline is the right approach — this
is the production-workload validation that the 2D spec failed to do.
