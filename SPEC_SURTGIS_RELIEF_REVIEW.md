# Review of `SPEC_SURTGIS_RELIEF.md` v1

**Reviewer's posture**: read every claim against the actual tree before
recommending. Findings below cite file:line for verification.

**Bottom line**: the spec is grounded. The primitives it reuses all exist
with the signatures it cites. The 4-phase milestone breakdown is sound.
What needs adjustment before implementation begins is mostly mechanical
(workspace-dep declarations, encoder placement, WASM return-shape
language), plus three real gaps the spec under-addresses (streaming-DEM
mode, perf claim that needs anchoring, error-type discipline).

---

## 1. Claims verified (no change needed)

| Spec §2 claim | Verified |
|---|---|
| `horizon_angle_map(dem, az, radius) → Result<Raster<f64>>` | `crates/algorithms/src/terrain/horizon_angles.rs:227` returns `Result<Raster<f64>>`, units in radians (line 226). |
| `horizon_angles(dem, HorizonParams) → HorizonAngles` with `.interpolate()` | `horizon_angles.rs:140`, `impl HorizonAngles` at line 64. |
| `hillshade(dem, HillshadeParams) → Result<Raster<f64>>` | `crates/algorithms/src/terrain/hillshade.rs:69`. |
| `multidirectional_hillshade(...)` | `multidirectional_hillshade.rs:52`. |
| `sky_view_factor(dem, SvfParams) → Result<Raster<f64>>` | `sky_view_factor.rs:41`. |
| `positive_openness` / `negative_openness` | `openness.rs:37`, `:47`. |
| `hypsometric_hillshade(dem, HillshadeParams)` | `hypsometric_hillshade.rs:20`. |
| `solar_radiation` family + `solar_vector(day, hour, lat) → (Vec3, f64)` | `solar_radiation.rs:207, 513, 769, 890, 1053`. |
| `raster_to_rgba(&Raster<T>, &ColormapParams) → Vec<u8>` | `crates/colormap/src/render.rs:83`, generic over `T: RasterElement`. |
| `ColorScheme::{Terrain, Water, Grayscale, …}` | `colormap/src/scheme.rs:40`: Terrain, Divergent, Grayscale, Ndvi, BlueWhiteRed, Geomorphons, Water, Accumulation. |
| `ray_shade = sun_altitude > horizon_angle_map(...)` | Units confirmed: horizon output radians; the spike's 0.785 rad = 45° check holds. |

The "80% already exists" claim is factual. No reimplementation needed
for any algorithm.

---

## 2. Concrete deltas for the spec before implementation

### Δ1. Workspace deps already include `image` — declare as workspace ref

**Spec §3 says**:
```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
image = { version = "0.25", default-features = false, features = ["png"] }
rayon.workspace = true
```

**Actual workspace already declares this** in
`Cargo.toml:90` (under the GUI block). Relief should inherit, not
re-declare:

```toml
[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
image.workspace = true
rayon.workspace = true
```

Same goes for `num-traits` and `thiserror` (already at workspace.deps lines
40 and 26).

### Δ2. WASM signature: `dem_op!` does NOT return RGBA — it writes a GeoTIFF

**Spec §6 says** "Mirror the existing `dem_op!` pattern in
`crates/wasm/src/lib.rs`". This is misleading.

`dem_op!` is defined as (wasm lib.rs around line 47):

```rust
macro_rules! dem_op {
    ($tiff:expr, $body:expr) => {{
        let dem = read_geotiff_from_buffer::<f64>($tiff, None) … ;
        let result = ($body)(&dem) … ;
        write_geotiff_to_buffer(&result, None) …
    }};
}
```

It returns **a GeoTIFF buffer** of a `Raster<f64>` output, not RGBA.
Existing `hillshade_compute` returns hillshade as a GeoTIFF, which the JS
side then re-renders.

For `relief_compute` the spec is correct that the return should be RGBA,
but it cannot just "mirror `dem_op!`". Two clean options:

**Option (a)** — return a `Vec<u8>` of *raw RGBA*, with width/height
returned via a companion call or a wrapper struct:

```rust
#[wasm_bindgen]
pub struct ReliefResult {
    pub width: u32,
    pub height: u32,
    pixels: Vec<u8>,
}

#[wasm_bindgen]
impl ReliefResult {
    #[wasm_bindgen(getter)]
    pub fn pixels(&self) -> Vec<u8> { self.pixels.clone() }
}

#[wasm_bindgen]
pub fn relief_compute(...) -> Result<ReliefResult, JsValue> { … }
```

**Option (b)** — return a PNG-encoded `Vec<u8>` even on WASM, by enabling
the `image` `png` feature on the WASM target. Slightly heavier wasm
binary, simpler JS-side (just `URL.createObjectURL(new Blob([...]))`).

Recommend **(a)** for the headline `relief_compute`. Cheap binary cost,
flexible on JS side. Document this in the spec so the implementer doesn't
spend cycles fighting `dem_op!`.

### Δ3. Encoder placement: move to `colormap` (per spec §10's own footnote)

Spec §10 notes "Consider whether [the PNG encoder] belongs in `colormap`
instead of `relief` (probably `colormap`, so other crates benefit)."

**Recommendation**: lock this in before M1. Add
`crates/colormap/src/encode.rs` with:

```rust
pub fn rgba_to_png_bytes(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, EncodeError>;
pub fn save_png<P: AsRef<Path>>(path: P, width: u32, height: u32, rgba: &[u8]) -> Result<(), EncodeError>;
```

Gate behind `#[cfg(not(target_arch = "wasm32"))]` so the WASM build doesn't
pull `image`. This unblocks: hypsometric_hillshade PNG output, curvature
preview PNGs, fluvial map PNGs — all of which currently end at
`Vec<u8>` RGBA. The relief crate then just `use surtgis_colormap::encode::save_png`.

Cost: one extra commit (encoder in colormap, bump colormap to 0.10.4) before
M1, then M1 is trivially correct.

### Δ4. `RgbaImage` is redundant with the existing RGBA convention

**Spec §4.1** introduces `RgbaImage` with `width`, `height`, `pixels` and
`from_intensity`, `over`, `multiply` methods. Most of this is fine, but:

- `pixels: Vec<u8>` matches the existing `raster_to_rgba` return shape
  exactly. Good.
- `from_intensity(&Raster<f64>) → RgbaImage` is a useful convenience.
- `from_rgba(width, height, pixels)` is just a struct literal — make it
  an inherent constructor only if it enforces invariants
  (`assert_eq!(pixels.len(), 4 * width * height)`); otherwise drop.

**Suggestion**: the type belongs in `colormap` (alongside `raster_to_rgba`)
rather than `relief`. Both `relief` and any future preview-tool would use
it. Plus this colocates the in-memory RGBA type with its encoder
(Δ3 above). Concrete proposal:

- `colormap`: defines `RgbaImage`, `RgbaImage::from_intensity`,
  `RgbaImage::over`, `RgbaImage::multiply`, `RgbaImage::save_png`,
  `raster_to_rgba` (already there).
- `relief`: re-exports `surtgis_colormap::RgbaImage` and adds only the
  *compositor* layer (`ReliefBuilder`, `ray_shade`, `ambient_shade`,
  `sphere_shade`).

This keeps `relief` truly a compositor (per spec §11).

### Δ5. Error type: use `surtgis_core::Error` or its own `ReliefError`?

Spec §3 mentions `ReliefError` in `lib.rs` but never defines it. The rest
of the workspace uses `surtgis_core::Error` (via the
`thiserror`-generated `Error` enum). Two coherent choices:

**Choice A** — `relief` returns `surtgis_core::Result<...>` (free, matches
every other crate).

**Choice B** — `relief` defines its own `ReliefError` with `#[from]`
conversions for `surtgis_core::Error` and `image::ImageError` and
`std::io::Error`. Slightly more typing but the encoder errors (`image`
crate failures) need somewhere to land.

Recommend **B**. Concrete:

```rust
// crates/relief/src/lib.rs
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReliefError {
    #[error("algorithm: {0}")]
    Algorithm(#[from] surtgis_core::Error),
    #[error("image encoding: {0}")]
    Encode(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("input shape mismatch: {0}")]
    Shape(String),
}

pub type Result<T> = std::result::Result<T, ReliefError>;
```

Image-encoding errors land in `Encode`; reuses the workspace's
`thiserror`. Tests stay short.

---

## 3. Gaps the spec under-addresses

### Gap 1. Big-DEM mode (streaming)

The existing `Hillshade` handler at
`crates/cli/src/handlers/terrain.rs:118` is **dual-path**:
- Small DEM → `read_dem` then `hillshade(...)` in-memory.
- Big DEM → `HillshadeStreaming` over a `StripProcessor` (256-row strips).

The spec's `ReliefBuilder` requires multiple passes (hillshade,
sky_view, horizon_angles per sun, blend RGBA, …) and the RGBA compositor
does not have a strip-streaming analog. Two honest options:

- **P1 punts on streaming**. Document this explicitly: "relief is
  in-memory only; for >N-cell DEMs, downsample first." Pick N
  empirically (the spike was 363k cells in ~2 s; 4K² is 16 M cells,
  which would be roughly proportional, so a 4K² is fine — limit notes
  10K² as the rough upper bound on commodity workstations).
- **P1 adds strip-streaming**. Substantially more code; not in the
  M1–M5 estimate. Don't quietly assume it fits in 2 weeks.

Recommend the first. Add a §10 risk to the spec naming "10K² in-memory
ceiling" explicitly.

### Gap 2. Perf claim needs a baseline number to chase

Spec §8 says: "Perf: add a Criterion bench in `crates/relief/benches/`
for a 4K² DEM; the value proposition is that Rust + Rayon beats
rayshader's runtime — measure and put the number in the README."

**Baseline measured 2026-06-02** (script `benchmarks/rayshader_baseline.R`,
data `benchmarks/results/rayshader_baseline.csv`). 5 timed reps + 1
warmup on `dem_filled.tif` (637×570 = 363,090 cells, Andes,
EPSG:32719). Sun azimuth 315°, soft-shadow penumbra over 11
anglebreaks (40°–50°). Rayshader 0.37.3, R 4.3.

| Stage | Median (s) | IQR (s) |
|---|---|---|
| `ray_shade` | **1.24** | [1.22, 1.29] |
| `sphere_shade` | **0.53** | [0.46, 0.58] |
| TOTAL | **1.80** | [1.75, 1.81] |

Variance is tight (TOTAL sd = 0.05 s), so the median is reliable.

**M2 acceptance criterion**: SurtGIS `relief::ray_shade +
relief::sphere_shade` on the same DEM, same 11-sample soft-shadow
budget, must complete in **≤ 0.90 s median** (≥ 2× speedup).
Stretch target: ≤ 0.45 s (≥ 4×).

**Implementation note that bears on this**: the cheap path
(`horizon_angle_map` ~350 ms per call) used 11 times would land at
~3.85 s — *slower* than rayshader. To meet ≥ 2×, M2 must either
use `horizon_angle_map_fast` (Gap 3 below) or amortise the
multi-azimuth pass via the `horizon_angles` precompute. The spec
should say which strategy it picks before M2 starts; "for free over
3 azimuths" (the spike's claim) is a different quality bar than
rayshader's default.

Bench environment for the baseline:
- Same workstation that hosts the spike numbers in spec §0.
- R 4.3.x, rayshader 0.37.3, default thread count.
- Wall-clock via `Sys.time()` deltas, no warm-cache trick beyond
  the discarded warmup rep.

When M2 runs its Criterion bench, run rayshader **and** SurtGIS in
the same session so the comparison cannot be skewed by background
load drift.

### Gap 3. `horizon_angles_fast` exists — prefer it

`horizon_angles.rs:491` defines `horizon_angles_fast` and `:577`
defines `horizon_angle_map_fast`. The spec only cites the non-fast
variants. If `_fast` exists, the soft-shadow multi-azimuth pass
(spec §10's main memory worry) should default to `_fast`. Worth one
sentence in the spec — and a note in M2's acceptance criteria.

---

## 4. Smaller suggestions (drop or keep at author's call)

- **§5 CLI flag names**: `--soft 8` is opaque. Suggest `--soft-shadow-samples 8` or `--shadow-samples 8`. Cheap to make readable.
- **§4.4 fluent API**: `add_shade` (multiply) vs `add_shadow` (over) is currently distinguished only by parameter name. Single method `add_layer(layer, blend_mode, opacity)` with an enum `BlendMode::{Multiply, Over}` reads cleaner and matches the canonical Porter-Duff / Photoshop vocabulary.
- **§6 WASM**: the `colormap: &str` param will explode the wasm export count if you want all 8 schemes. Either accept the strings (and document each), or expose a `ReliefBuilder` constructor on the JS side that takes a `WongScheme` enum.
- **§9 milestones**: M1 (skeleton + layer + encoder) overlaps with Δ3 above. If encoder moves to `colormap`, M1 in `relief` shrinks to just `RgbaImage` re-exports + a single dummy composite test. Estimate becomes more honest.

---

## 5. Recommended ordering before M1 starts

1. **Pre-M1 commit (colormap bump to v0.10.4)**: add `encode.rs` to
   `surtgis-colormap` per Δ3. Bump `colormap` to v0.10.4. Test PNG output.
2. **Pre-M1 commit (workspace deps audit)**: confirm `image` in
   workspace deps; remove duplicate redeclaration from existing
   `crates/gui/Cargo.toml` if it's there (likely).
3. **Pre-M1 baseline measurement**: run rayshader on `dem_filled.tif`,
   record `ray_shade + sphere_shade` wall-clock. This is the M2 target.
4. **M1 (skeleton)**: now trivial. `crates/relief/Cargo.toml` with
   workspace-ref deps, `lib.rs` with `ReliefError` + re-exports.
5. **M2 (`ray_shade`)**: the headline feature. Reuse
   `horizon_angle_map_fast` for soft shadows.
6. **M3–M5**: as spec.

The cost of these three pre-M1 steps is ~half a day; they prevent
half-built dependency-graph regrets later.

---

## 6. Verdict on the spec

- **Technical claims**: solid. Every primitive verified.
- **Scope**: appropriately narrow. Calls out what to NOT do.
- **Risks named**: §10 names memory + scope creep but should add the
  three gaps in §3 above.
- **Ergonomics**: minor naming polish + blend-mode unification would
  read smoother.
- **Path to M1**: half-day of pre-work (Δ3 + workspace deps + rayshader
  baseline) makes M1 trivial.

This is a green-light spec. The pre-M1 work is the only thing standing
between today and a clean implementation start.
