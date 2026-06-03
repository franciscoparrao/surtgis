# SPEC: `surtgis-relief` — Rayshader-style relief & 3D rendering

**Status:** P1 shipped (M1–M5 merged on `main` 2026-06-03)
**Target version:** 0.11.0 (MVP 2D) → 0.12.0 (3D web)
**Author of spec:** handoff doc — implemented by the SurtGIS team
**Crate name:** `surtgis-relief` (workspace member `crates/relief`)

> **Implementation reality:** This document is the original handoff spec.
> Several P1 assumptions did not survive contact with the implementation —
> see **§12 Reality check** for what shipped, what was added, and what was
> wrong. Read §12 before §0–§11 if you want the truth before the original
> theory.

---

## 0. Proof of concept (spike, 2026-06-02)

> **Reality:** the spike's "no new terrain math" claim was wrong. See §12.
> It measured 3 azimuths × 1 altitude. Production needed 1 azimuth × 11
> altitudes, where `horizon_angle_map` is 8× too slow without an early
> exit and without shared-azimuth amortisation. Two new ray-march
> primitives were written.

A throwaway spike already validated the central claim of this spec: a
rayshader-style shaded relief composites cleanly from primitives that **already
exist** in `surtgis-algorithms` — no new terrain math. `dem_filled.tif`
(637×570, Andes, EPSG:32719) was rendered by chaining existing CLI commands and
blending the outputs in Python (Python here only stands in for the future
`colormap` PNG encoder + `ReliefBuilder`).

Reference images (committed in `output/`):
- `output/relief_spike_pipeline.png` — the 4 accumulating layers
  (base colormap → +sphere_shade → +ray_shade → +ambient).
- `output/relief_spike_hero.png` — final composite, grazing 18° sun.
- `output/relief_spike_sunangle.png` — same `horizon-angle`, sun 45° vs 18°.

What it confirmed:
- **`ray_shade` = `sun_altitude > horizon_angle_map(dem, az, radius)`.** Units
  line up exactly: horizon output is radians (max 1.527 ≈ 87°), sun 45° =
  0.785 rad. The spec formula works verbatim.
- **Soft shadows for free** by averaging the lit/shadow mask over 3 azimuths
  (300/315/330) — no penumbra code.
- **Sun altitude needs no recompute.** Dropping the sun 45°→18° took shadow
  coverage 1.2%→8.5% by *re-thresholding the same* `horizon-angle` raster —
  confirms §10's note to query `horizon_angle_map` per sun sample and cache.
- Whole composite computed in **~2 s** on 363k cells (release binary):
  hillshade 19 ms, horizon-angle ~350 ms ×3, svf 770 ms.
- **The only real gap is the one this spec adds:** no PNG encoder and no layer
  compositor in-tree. Everything hard already exists.

The spike is the "before" (CLI + Python glue). The crate turns it into one
binary, one command (`surtgis relief … --shadows --soft 8`), plus the WASM path
that rayshader (desktop/R) cannot offer.

---

## 1. Goal

Add a *relief compositing* layer on top of the terrain primitives SurtGIS
already has, producing rayshader-quality shaded-relief images (and, in a
second phase, an interactive 3D terrain viewer that runs in the browser via
WASM/WebGPU).

**The key insight that scopes this work:** ~80% of rayshader's engine already
exists inside `surtgis-algorithms`. This crate is mostly a *compositor*, not a
new set of algorithms. It reuses, it does not reimplement.

### Scope

| Phase | Deliverable | Effort |
|-------|-------------|--------|
| **P1 — MVP (this spec's core)** | 2D relief: `ray_shade`, `ambient_shade`, `sphere_shade`, layer compositing, water layer, PNG output, CLI `surtgis relief`, WASM + Python bindings | ~2 weeks |
| **P2 — 3D web** | DEM → mesh, texture draping, `wgpu`/WebGPU interactive viewer embedded in `surtgis-demo/` | ~3–4 weeks |
| **Out of scope** | Photorealistic path tracer (`rayrender`/`plot_gg` equivalent), depth-of-field, OBJ-with-PBR-materials. Revisit post-publication; if ever needed use `embree-rs`, do not hand-roll. | — |

### Non-goals for P1
- No interactive rendering (P1 is batch image generation).
- ~~No new terrain math — everything composites existing `terrain/` outputs.~~
  Three new primitives shipped: `cast_shadow_ray_mask`, `horizon_tan_map`,
  `detect_water`. See §12.

---

## 2. What we reuse (do NOT reimplement)

> **Reality:** the `ray_shade` and "soft shadows" rows of this table did not
> survive contact with the rayshader benchmark. `horizon_angle_map` is
> correct mathematically but ~8× too slow for the 11-altitude recipe.
> Shipped code in `crates/relief/src/shadow_ray.rs` does its own
> early-exit ray-march (`cast_shadow_ray_mask`) plus a shared-azimuth
> amortisation primitive (`horizon_tan_map`). See §12 for the post-mortem.

All of these already exist in `crates/algorithms/src/terrain/`:

| Rayshader concept | Existing SurtGIS primitive | Signature already in tree |
|---|---|---|
| `ray_shade()` (ray-traced cast shadows) | `horizon_angle_map(dem, azimuth_rad, radius)` → `Raster<f64>` (horizon angle in radians per cell) | `terrain/horizon_angles.rs:227` |
| multi-sample / soft shadows | `horizon_angles(dem, HorizonParams{radius, directions})` → `HorizonAngles` with `.interpolate(row, col, azimuth_rad)` | `terrain/horizon_angles.rs:140` |
| `sphere_shade()` (normal-based shade) | `hillshade(dem, HillshadeParams{azimuth, altitude, z_factor, normalized})` and `multidirectional_hillshade(...)` | `terrain/hillshade.rs:69` |
| `ambient_shade()` (ambient occlusion) | `sky_view_factor.rs`, `openness.rs` | `terrain/sky_view_factor.rs`, `terrain/openness.rs` |
| `height_shade()` (hypsometric tint) | `hypsometric_hillshade.rs` + `colormap` | `terrain/hypsometric_hillshade.rs` |
| sun geometry | `solar_radiation.rs` (sun altitude/azimuth per timestep) | `terrain/solar_radiation.rs` |
| `add_overlay()` colorizing | `raster_to_rgba(raster, &ColormapParams)` → `Vec<u8>` RGBA; `auto_params`; `ColorScheme::{Terrain, Water, Grayscale, …}` | `crates/colormap/src/render.rs:83` |

**The whole P1 algorithm is:** call these, normalize to `[0,1]`, alpha-blend, encode PNG.

### Core derivation — `ray_shade` from `horizon_angle_map`

Rayshader's `ray_shade` marches a ray toward the sun per cell and checks if
terrain occludes it. We already have the occlusion test precomputed:

```
cell is LIT  ⟺  sun_altitude > horizon_angle_map(dem, sun_azimuth, radius)[cell]
cell is in SHADOW otherwise
```

Soft shadows (rayshader's `anglebreaks` penumbra) = average the binary
lit/shadow mask over N sun samples around the nominal sun position. No new
ray-tracing code — just a loop over `horizon_angle_map` calls (or one
`horizon_angles` precompute + `.interpolate`).

---

## 3. Crate layout

```
crates/relief/
├── Cargo.toml
└── src/
    ├── lib.rs          # re-exports, ReliefError
    ├── layer.rs        # RgbaImage, Layer trait, alpha-blend compositor
    ├── ray_shade.rs    # ray_shade() over horizon_angle_map, multi-sun
    ├── ambient.rs      # ambient_shade() wrapping sky_view_factor/openness
    ├── sphere_shade.rs # sphere_shade() wrapping hillshade + colormap
    ├── water.rs        # detect_water() + water layer
    ├── compose.rs      # ReliefBuilder fluent pipeline
    └── encode.rs       # RGBA → PNG (image crate)
```

### `Cargo.toml`

```toml
[package]
name = "surtgis-relief"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "Shaded-relief and hillshade compositing for SurtGis (rayshader-style)"
keywords = ["gis", "hillshade", "relief", "visualization", "terrain"]
categories = ["science::geo", "graphics"]

[dependencies]
surtgis-core      = { version = "0.10", path = "../core", default-features = false }
surtgis-algorithms = { version = "0.10", path = "../algorithms", default-features = false }
surtgis-colormap  = { version = "0.10", path = "../colormap", default-features = false }
num-traits.workspace = true
thiserror.workspace = true

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
image = { version = "0.25", default-features = false, features = ["png"] }
rayon.workspace = true
```

Add `"crates/relief"` to the `members` list in the root `Cargo.toml`.

> **Note on PNG:** `surtgis-colormap` currently returns raw RGBA `Vec<u8>` and
> has no encoder. P1 introduces the first in-tree PNG writer (`encode.rs`).
> For WASM, return raw RGBA and let JS encode (matches existing WASM pattern of
> returning `Vec<u8>`), OR gate the `image` PNG path behind
> `cfg(not(wasm32))`.

---

## 4. Public Rust API (P1)

### 4.1 Image & layer model

```rust
/// An 8-bit RGBA raster, row-major, 4 bytes/pixel. Same memory layout
/// colormap::raster_to_rgba already produces.
pub struct RgbaImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<u8>, // len == width*height*4
}

impl RgbaImage {
    pub fn from_rgba(width: usize, height: usize, pixels: Vec<u8>) -> Self;
    /// Build a single-channel intensity image [0,1] → grayscale RGBA.
    pub fn from_intensity(intensity: &Raster<f64>) -> Self;
    /// Alpha-over composite: `self` is the base, `top` painted on top.
    pub fn over(&mut self, top: &RgbaImage, opacity: f64);
    /// Multiply blend (used to lay shadows over a colored base).
    pub fn multiply(&mut self, top: &RgbaImage, opacity: f64);
}
```

### 4.2 Shading functions — all take `&Raster<f64>`, return intensity `[0,1]`

```rust
pub struct SunSample { pub azimuth_deg: f64, pub altitude_deg: f64 }

pub struct RayShadeParams {
    /// Nominal sun position(s). Multiple = soft shadows.
    pub suns: Vec<SunSample>,
    /// Ray search radius in cells (maps to horizon_angle_map `radius`).
    pub radius: usize,
    /// Vertical exaggeration applied before shadowing.
    pub z_factor: f64,
}
impl Default for RayShadeParams { /* sun az 315 alt 45, radius 100, z 1.0 */ }

/// Ray-traced cast shadows. Internally: for each SunSample, compute a binary
/// lit/shadow mask via `horizon_angle_map`, then average over samples.
/// Returns intensity in [0,1] (1 = fully lit).
pub fn ray_shade(dem: &Raster<f64>, p: &RayShadeParams) -> Result<Raster<f64>, ReliefError>;

/// Ambient occlusion. Thin wrapper over sky_view_factor (or openness),
/// normalized to [0,1].
pub fn ambient_shade(dem: &Raster<f64>, radius: usize) -> Result<Raster<f64>, ReliefError>;

/// Normal-based "sphere" shade. Wrapper over hillshade()/multidirectional,
/// normalized to [0,1].
pub fn sphere_shade(dem: &Raster<f64>, p: &HillshadeParams) -> Result<Raster<f64>, ReliefError>;
```

### 4.3 Water

```rust
pub struct WaterParams { pub min_area_cells: usize, pub flatness_eps: f64 }

/// Heuristic flat-area / constant-elevation water mask (rayshader detect_water
/// analog). Reuse landscape/morphology connected-components if available.
pub fn detect_water(dem: &Raster<f64>, p: &WaterParams) -> Result<Raster<u8>, ReliefError>;
```

### 4.4 Fluent compositor (the rayshader pipeline)

This is the public ergonomics — the part rayshader users actually touch.

```rust
let img = ReliefBuilder::new(&dem)
    .base_colormap(ColorScheme::Terrain)        // colormap::raster_to_rgba
    .add_shade(sphere_shade(&dem, &hs)?, 0.5)   // multiply blend
    .add_shadow(ray_shade(&dem, &rs)?, 0.5)     // ray-traced shadows
    .add_ambient(ambient_shade(&dem, 50)?, 0.3)
    .add_water(detect_water(&dem, &wp)?, ColorScheme::Water)
    .render();                                   // -> RgbaImage

img.save_png("relief.png")?;                     // cfg(not(wasm32))
```

`ReliefBuilder` just accumulates layers and folds them with `over`/`multiply`.
No algorithm logic lives here.

---

## 5. CLI — `surtgis relief`

Follow the existing dispatch pattern in `crates/cli/src/handlers/terrain.rs`
and the subcommand enum in `crates/cli/src/commands.rs`.

```
surtgis relief DEM.tif OUT.png \
    --colormap terrain \
    --sun-azimuth 315 --sun-altitude 45 \
    --shadows           # enable ray_shade layer
    --soft 8            # 8 sun samples for penumbra
    --ambient           # add ambient occlusion
    --water             # detect + tint water
    --z-factor 1.5 \
    --radius 100
```

Add one new module `crates/cli/src/handlers/relief.rs`, register the subcommand,
wire into `commands.rs`. Mirror how `Hillshade`/`HorizonAngle` are wired
(`commands.rs:296`, `commands.rs:589`).

---

## 6. WASM bindings

Mirror the existing `dem_op!` pattern in `crates/wasm/src/lib.rs` (e.g.
`hillshade_compute` at line 79). Return raw RGBA `Vec<u8>` (JS/canvas encodes
the PNG — keeps `image` crate out of the wasm build).

```rust
/// Full relief composite → RGBA bytes (width*height*4). Caller reads
/// dimensions from a companion call or a returned struct.
#[wasm_bindgen]
pub fn relief_compute(
    tiff_bytes: &[u8],
    colormap: &str,
    sun_azimuth: f64,
    sun_altitude: f64,
    shadows: bool,
    ambient: bool,
) -> Result<Vec<u8>, JsValue>;
```

This is what makes the `surtgis-demo/` browser story unique vs rayshader
(which is desktop/rgl-only).

## 7. Python bindings

Mirror `crates/python/src/lib.rs` `#[pyfunction] hillshade_compute` (line 258).
Return a numpy `(H, W, 4) uint8` array.

---

## 8. Testing & validation

- **Unit:** blend math (`over`, `multiply`) against hand-computed pixels;
  `ray_shade` on a synthetic ridge DEM (one side lit, one shadowed) — assert
  the shadowed flank is < lit flank.
- **Golden image:** render `dem_filled.tif` (already in repo root) and the
  `examples/maule_mini` DEM; commit golden PNGs; compare with a tolerance.
- **Cross-check vs rayshader:** render the same DEM in R rayshader
  (`ray_shade` + `sphere_shade`), compare shadow masks structurally (IoU of
  shadow regions > 0.9). Document in `benchmarks/`.
- **Perf:** add a Criterion bench in `crates/relief/benches/` for a 4K² DEM;
  the value proposition is that Rust + Rayon beats rayshader's runtime — measure
  and put the number in the README (consistent with the existing Performance
  section).

---

## 9. Milestones

1. **M1 — crate skeleton + layer model** (`layer.rs`, `encode.rs`): blend +
   PNG out, with a hardcoded grayscale hillshade. Visible PNG end-to-end.
2. **M2 — `ray_shade`** over `horizon_angle_map`, single sun, then multi-sun.
   This is the headline feature.
3. **M3 — `sphere_shade` + `ambient_shade` + colormap base**; `ReliefBuilder`.
4. **M4 — water layer + CLI `surtgis relief`**.
5. **M5 — WASM + Python bindings**, demo page in `surtgis-demo/`.
6. **(P2) M6+ — 3D mesh + wgpu/WebGPU viewer.** Separate spec.

M1–M5 = the ~2-week MVP. Each milestone ends with a runnable example in
`crates/relief/examples/`.

---

## 10. Risks & notes

- **`horizon_angles` memory:** the all-directions struct is
  `8 × directions × rows × cols` bytes (~275 MB for 1000² × 36 dirs, per its own
  doc). For `ray_shade` prefer **`horizon_angle_map` per sun sample** (single
  azimuth, streaming-friendly) over the full `horizon_angles` precompute, unless
  many suns share directions.
  **Reality (post-M2):** `horizon_angle_map` per sun sample is also too slow
  at 11 altitudes (~2.75 s/call × 11 = 30 s). Production uses
  `cast_shadow_ray_mask` per sun (early-exit) or `horizon_tan_map` once
  for the shared-azimuth case. See §12.3.
- **Edition 2024 / no GDAL:** stay consistent — `image` is pure-Rust with the
  `png` feature only; do not pull system libs.
- **Don't widen scope in the current ENVSOFT R1.** This crate ships *after* the
  revision is resolved; it strengthens the "single binary, browser-capable"
  narrative for a follow-up, not for R1.
- **Colormap gap:** `surtgis-colormap` has no PNG encoder today — P1's
  `encode.rs` is the first one. Consider whether it belongs in `colormap`
  instead of `relief` (probably `colormap`, so other crates benefit).

---

## 11. One-paragraph summary for the implementer

> **This paragraph was wrong about "no new terrain math".** See §12 for the
> corrected accounting. Kept here as a historical record of the original
> handoff intent.

`surtgis-relief` is a thin compositor crate. The hard part — ray-traced shadow
computation — is already done by `terrain::horizon_angle_map`; relief just
thresholds it against sun altitude and averages over sun samples for soft
shadows. `sphere_shade` wraps `hillshade`, `ambient_shade` wraps
`sky_view_factor`, the base color wraps `colormap::raster_to_rgba`. Everything
else is alpha-blending RGBA buffers and writing a PNG. No new terrain math.
Build it as workspace member `crates/relief`, expose it through CLI/WASM/Python
exactly like `hillshade` is exposed today, and the browser WASM path gives
SurtGIS a capability rayshader (desktop-only) does not have.

---

## 12. Reality check (post-M5, 2026-06-03)

This section records where the original spec misled and what shipped instead.
It is intentionally honest. Read it before writing the next §-style spec for
another crate — the same traps will lurk there.

### 12.1 What survived intact

The compositor framing (§4 layer model, §4.4 ReliefBuilder), the CLI surface
(§5), and the WASM/Python frontends (§6, §7) shipped close to spec. The
re-use story held for three of the four shading primitives:

| Spec claim | Shipped | Notes |
|---|---|---|
| `sphere_shade` wraps `hillshade` | ✓ 49 LOC in `sphere_shade_impl.rs` | Forces `normalized = true`, otherwise identical |
| `ambient_shade` wraps `sky_view_factor` | ✓ ~30 LOC in `ambient.rs` | Defaults to 16 directions; perf follow-up below |
| base colour wraps `colormap::raster_to_rgba` | ✓ in `compose.rs::base_colormap` | unchanged |
| Layer compositor uses `over` + `multiply` blends | ✓ in `compose.rs` and `surtgis-colormap::RgbaImage` | Only addition: `add_water` paints a u8 mask with a scheme-sampled colour |
| Single binary + WASM + Python frontends | ✓ all three landed in PR #9 (CLI), PR #10 (WASM, Python) | — |

### 12.2 What changed

| Spec said | Reality | Why |
|---|---|---|
| "No new terrain math" (§0, §1, §11) | **Three new primitives shipped in `crates/relief/src/`** | Perf trap, water heuristic — see below |
| PNG encoder in `relief::encode.rs` (§3) | Landed in `surtgis-colormap::encode` | The note at §3 line 147 already flagged this. Done in pre-M1 (PR #7). |
| Layout: `ray_shade.rs` + `sphere_shade.rs` + `ambient.rs` (§3) | `ray_shade_impl.rs`, `shadow_ray.rs`, `sphere_shade_impl.rs`, `ambient.rs`, `water.rs`, `compose.rs` | `shadow_ray.rs` hosts the two new perf primitives; the rest matches |
| `ray_shade` is `sun_altitude > horizon_angle_map(...)` thresholded (§2, §0) | Correct mathematically, **wrong for performance** | See §12.3 |
| Spike timing "~2 s on 363k cells, horizon-angle ~350 ms × 3" (§0) | Spike measured **3 azimuths × 1 altitude**, not the rayshader recipe of 1 azimuth × 11 altitudes. The spec under-estimated the real workload by 4–5×. | The spike's 350 ms × 3 = 1.05 s. A naive port to 11 altitudes would have been 14 s (horizon_angle_map per call) — 8× slower than rayshader. |

### 12.3 Three new primitives in `crates/relief/src/`

The "thin compositor" framing failed at the ray_shade level. Three primitives
had to be written:

1. **`shadow_ray::cast_shadow_ray_mask(dem, az, alt, radius) -> Raster<f64>`**
   (~75 LOC). Early-exit per-cell ray-march. Unlike `horizon_angle_map`,
   which walks the full radius computing the *maximum* horizon angle, this
   stops the moment any occluder is found above the target altitude. With
   the hot-loop tricks (incremental position state in additions, no
   per-step multiplications, `unsafe get_unchecked`), one call takes
   ~180 ms on the 637×570 DEM vs ~2.75 s for `horizon_angle_map(radius=500)`.

2. **`shadow_ray::horizon_tan_map(dem, az, radius) -> Raster<f64>`**
   (~85 LOC). Full-radius march tracking `max_k (z(k) - z0) / dist(k)`
   (the tan of the horizon angle). Same hot-loop optimisations as (1),
   plus pre-computed `inv_dist[k]` so the inner-loop tan is one multiply,
   no division. This is the **amortisation primitive**: when sun samples
   share an azimuth (the rayshader anglebreaks recipe), `ray_shade` calls
   `horizon_tan_map` **once** and each altitude reduces to an O(N)
   thresholding `tan(alt) >= horizon_tan?`. 11 altitudes share work that
   previously cost 11 independent ray marches.

3. **`water::detect_water(dem, WaterParams) -> Raster<u8>`** (~170 LOC).
   Heuristic water mask via flat-area connected components: 8-neighbour
   flatness test + 4-connected union-find + minimum-area filter. The
   spec assumed this could "reuse landscape/morphology connected-components
   if available" (§4.3); the implementation found it cleaner to write a
   focused 170-LOC routine than to thread a generic CC primitive through
   the constraints (flat predicate, NaN handling, area filter, u8 output).

In addition, the auto-detection logic in `ray_shade` (shared-azimuth
fast path vs per-sun fallback) is ~50 LOC of pure compositor code that has
no analog in the spec. It is what makes the WASM and Python frontends emit
a 6.99× rayshader number with no caller-side ceremony.

### 12.4 Perf payoff

The spike said "~2 s composite". The shipped M2 acceptance benchmark, with
the rayshader-equivalent recipe (azimuth 315°, anglebreaks `seq(40, 50, 1)`,
radius `max(rows, cols) = 850`), measures:

- `ray_shade` median: **0.19 s** (10.4× faster than the pre-amortisation
  M2 baseline of 1.98 s, which itself replaced a 14 s naive approach)
- `sphere_shade` median: 0.06 s
- TOTAL median: **0.26 s** = **6.99× rayshader** (rayshader baseline
  measured at 1.80 s in `benchmarks/rayshader_baseline.R`)
- M2 target was ≥ 2×; stretch was ≥ 4×. Both pass.

This number would have been impossible to hit with the spec's
"threshold `horizon_angle_map`" approach. The amortisation insight only
becomes available once you accept that you are writing the primitive
yourself rather than re-using the terrain crate's.

### 12.5 Outstanding perf concern: `ambient_shade`

`ambient_shade` calls `sky_view_factor` with 16 directions × user radius.
On the M3 end-to-end render (`render_relief` example), it dominates the
pipeline at **6.4 s** (vs sphere 0.05 s, ray 0.20 s, compose 0.06 s).
Rayshader's analog is similarly slow, so this does not affect the
competitive perf story — but it is the next target. Likely treatment:
a relief-specific `ambient_tan_map` using the same hot-loop tricks
as `horizon_tan_map`, or an early-exit variant of `sky_view_factor`
upstream in `surtgis-algorithms`.

### 12.6 Lesson for the next handoff spec

The original §0 spike measured the cheapest configuration ("3 azimuths,
1 altitude, sun 18°") and projected linear scaling to the production
configuration ("1 azimuth, 11 altitudes"). That projection was wrong by
8× because `horizon_angle_map` has no early exit and no shared-azimuth
amortisation. **When writing the next spec that claims "we already have
the primitive, just compose it" — measure the production-workload
configuration in the spike, not the cheapest one.** Otherwise the spec
will mis-scope the work by an order of magnitude and the implementer
will rediscover the gap mid-milestone.
