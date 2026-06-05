# SPEC: `surtgis-relief` P3 — polish to rayshader peer

**Status:** Proposed
**Target version:** 0.13.0
**Author of spec:** handoff doc — implements P3 follow-up to P2 (v0.12.0)

> **Methodological reminder.** `SPEC_SURTGIS_RELIEF.md` §12.6 documented a
> spec failure: its M1 spike measured the cheapest configuration (3
> azimuths × 1 altitude) and underestimated production workload by 8×.
> P2's spec applied the lesson by measuring 1M vertices in M1. P3 applies
> it again: every M-acceptance below names the **specific production
> case** the milestone has to look right on — not the easiest case.

---

## 0. Where we are

After v0.12.0 (P2 merged at `0ebc2cf`), `surtgis-relief` + `surtgis-relief-3d`
beat rayshader on 2D perf (6.99×) and on deployment surface (native CLI,
WASM browser, headless screenshot). The honest gaps that remain in
rayshader's favour:

| rayshader gap | What's missing | P3 verdict |
|---|---|---|
| 5 years of palette tuning (imhof1, bw1, …) | Pure data — 8 palettes × 5–10 color stops | **Close** in M1 |
| Atmospheric haze / aerial perspective | ~50–150 LOC fragment shader | **Close** in M1 |
| Water with waterdepth shading | Distance-transform + depth-blend in shader | **Close** in M2 |
| Desktop UX polish (damping, keybindings, GLB) | Ergonomic work, not algorithmic | **Close** in M3 |
| R ecosystem integration | Thin wrapper crate / package | **Close** in M4 |
| LOD adaptativo for DEMs ≥ 2K side | Real engineering (geomipmapping or tile-based) | **P4**, separate sprint |
| Path-tracing (`rayrender` equivalent) | 6–12 months of focused work | **Out of scope** (matches P1 spec, §1.2) |

After P3, the only honest "rayshader wins" line is path-tracing. P4 closes LOD.

---

## 1. Goal

Bring `surtgis-relief` to feature parity with rayshader on the 2D-and-light-3D
fronts that actually matter for landscape figures — Imhof-style palettes,
atmospheric perspective, water depth — and improve desktop UX so the
native viewer feels finished, not prototypical. Add a thin R wrapper so
existing rayshader users can switch with one line of R.

### Scope (P3 — this spec)

| Deliverable | Cost | Milestone |
|---|---|---|
| 8 Imhof-style + extra ColorScheme variants in `surtgis-colormap` | 1 day | M1 |
| Atmospheric haze (depth-based fog mix) in 3D shader + CLI flag | 2 days | M1 |
| Water depth shading via distance transform from shore | 4 days | M2 |
| Desktop UX polish: camera damping, key overlay, GLB export, screenshot key | 3 days | M3 |
| R thin wrapper crate `surtgis-r` (extendr) | 3 days | M4 |

### Non-goals

- **No LOD.** DEMs > 1M vertices still cap rendering quality. P4.
- **No path-tracing.** Use rayshader / `rayrender` for that figure.
- **No physical sky model (Rayleigh+Mie).** A heuristic depth-fog covers
  95% of "looks atmospheric" for landscape figures at far less cost.
- **No water animation / waves / reflections.** Static depth-blended
  water surface only.
- **No mobile (iOS/Android) packaging.**

---

## 2. What we reuse

| P3 need | Source |
|---|---|
| Color stop infrastructure | `surtgis-colormap::ColorScheme` + `multi_stop` |
| 3D pipeline (mesh, shader, camera) | `surtgis-relief-3d::pipeline` |
| Water mask | `surtgis-relief::detect_water` |
| Distance transform | `surtgis-algorithms::morphology` (chamfer DT) |
| Linear depth | computed in vertex shader, passed to fragment as varying |
| R FFI | `extendr-api` crate |

---

## 3. Milestones

### M1 — atmospheric haze + Imhof-style palettes

**Acceptance bar:** A grazing-light DEM render (sun altitude ≤ 15°, low
contrast at distance) with haze ON looks materially different — *and
better* — from haze OFF. A high-altitude bird's-eye render with haze
should still look correct, not flat-washed. Pick the grazing case as
the spec case: if it doesn't work at altitude 10°, it won't work at
altitude 60° either, and milestone passes inherit the bug.

**Sub-tasks:**

- M1.1 — palettes in `surtgis-colormap::scheme.rs`. 8 new variants:
  `Imhof1` (greens → straw → ochre → snow), `Imhof2` (cooler, more
  alpine), `Imhof3` (desert-leaning), `Imhof4` (twilight purples),
  `Bw1` (smooth grayscale), `Bw2` (high-contrast grayscale with mid
  ridge), `DesertDry` (sand → tan → red rock → bone white), `Pastel`
  (gentle rainbow for talks/posters). Each palette = 5–8 stops with
  documented (`#RRGGBB`) values so the curation can be audited.

- M1.2 — fog uniform extension. Add `fog_color : vec4` (xyz colour,
  w density) to `Uniforms`. Vertex shader writes `vs_out.linear_depth
  = -view_pos.z`. Fragment shader: `fog_t = saturate(linear_depth /
  fog_far)`, `final = mix(shaded, fog_color, fog_t * density)`. New
  CLI flag `--haze <density>` (0.0 = off, default 0.0 to preserve
  current output, typical use 0.3–0.6). New keybinding `H` in the
  native viewer toggles haze; `F` / `G` adjust density ±0.05.

- M1.3 — verification: emit two PNGs side-by-side from the M1 spike
  recipe (`dem_filled.tif`, sun azimuth 315°, altitude 12°), one with
  `--haze 0` and one with `--haze 0.55`. The hazed one should reduce
  distant contrast visibly. Commit both as `output/m1_haze_off.png`
  and `output/m1_haze_on.png`.

### M2 — water depth shading

**Acceptance bar:** A DEM with a real water body (lake, river, ocean)
rendered with `--water` shows continuous depth from shore (light blue,
high reflectance) to centre (dark blue, low reflectance) — not a flat
binary blue mask. The spec case is `dem_filled.tif` or a synthetic
lake; if the synthetic case fails the real one will too.

**Sub-tasks:**

- M2.1 — depth raster from a water mask. New function
  `surtgis-relief::water_depth(mask: Raster<u8>) -> Raster<f32>` —
  for each water cell, the Chebyshev (or Euclidean) distance to the
  nearest non-water cell in cells. Returns 0 at shore, increasing
  inland. Use `surtgis-algorithms::morphology` distance transform.

- M2.2 — water depth painting in `ReliefBuilder::add_water`. Replace
  the current flat colour with a depth-to-colour lookup: shallow
  cells use a light scheme value, deep cells use a dark one. Pass
  the depth raster through.

- M2.3 — verification with a synthetic 200×200 lake DEM (centre dipped
  by 10 m to make a fake bathymetry-from-mask scenario). Output PNG
  must show smooth shore→centre gradient.

### M3 — desktop UX polish

**Acceptance bar:** A user who has never seen surtgis-relief-3d before
opens the native viewer, hits `?`, sees all controls, can take a
screenshot with `S`, and rotate smoothly without jitter. The spec case
is "first 60 seconds with the tool" — if those feel rough, no later
polish helps.

**Sub-tasks:**

- M3.1 — camera damping. Mouse-drag rotate/pan/zoom apply a critically-
  damped exponential smoothing (`x_next = lerp(x, target, 1 - exp(-dt
  / tau))`) instead of instant updates. `tau = 80 ms`. Inertia for
  ~200 ms after release.

- M3.2 — key overlay. Press `?` to toggle a corner overlay showing
  every keybinding. Use a simple text drawer (no font crate
  dependency — bitmap font is fine).

- M3.3 — screenshot key. `S` writes the current frame to
  `relief3d-YYYYMMDD-HHMMSS.png` in cwd. Uses the existing
  `headless::render_to_rgba` path repointed at the current camera /
  lighting state.

- M3.4 — GLB export. `E` writes a `.glb` (binary glTF) of the current
  mesh + texture (no animation, single static frame). Pull `gltf` or
  `glam-gltf` crate. Optional: skip if time-boxed too tight.

### M4 — R thin wrapper (`surtgis-r`)

**Acceptance bar:** A rayshader user with R + cargo installed can do:

```r
remotes::install_github("franciscoparrao/surtgis", subdir = "surtgis-r")
library(surtgisr)
relief_image <- surtgis_relief("dem.tif", colormap = "imhof1",
                                shadows = TRUE, ambient = TRUE)
png::writePNG(relief_image, "out.png")
```

…and get a result that visually matches `surtgis relief` CLI output
on the same DEM. The spec case is "rayshader user switches one line of
code"; if the API friction exceeds that, the wrapper isn't pulling its
weight.

**Sub-tasks:**

- M4.1 — `crates/surtgis-r/` workspace member with `extendr-api`.
  Public R functions: `surtgis_relief(dem_path, ...)` and
  `surtgis_relief_3d(dem_path, ...)` (CLI shell-out for the latter
  since extendr + wgpu would balloon the wrapper).

- M4.2 — `DESCRIPTION` + `NAMESPACE` files so `remotes::install_github`
  works. `cargo` is in `Imports:` via a custom `configure` script that
  checks for the toolchain.

- M4.3 — verification: a self-test R script that loads a small DEM,
  calls `surtgis_relief`, asserts the result is an `(H, W, 4)` array.

---

## 4. Risks

- **Fog density is taste, not physics.** Users will inevitably ask "is
  this physically correct?" — answer: no, this is a depth-blend
  heuristic, not Rayleigh+Mie. The README and CLI help must say so up
  front to avoid claim creep.

- **Imhof palettes are subjective.** The curation can be argued.
  Decision rule: pick the palette that reads as "shaded relief" at a
  glance on `dem_filled.tif` with default shading, not the one that
  looks best on a flat colour ramp.

- **R wrapper supply chain.** R users on Windows / macOS without cargo
  installed cannot build the crate. M4 acceptance must include a
  pre-built binary fallback path documented in the README, even if it
  ships in a later release.

- **GLB export pulls a sizable dep.** `gltf` adds ~50 KLOC. If M3.4
  doesn't fit the time-box, ship M3 without it.

---

## 5. One-paragraph summary for the implementer

P3 closes every "rayshader wins" line except path-tracing and LOD.
Two of the four milestones (M1 palettes + M3 UX) are pure data /
ergonomics — fast wins. M1 haze and M2 water depth need shader work
but stay inside the existing wgpu pipeline. M4 opens an
R-distribution channel without rewriting algorithms. The §12.6 lesson
keeps biting: spec cases are the worst-realistic configuration, not the
average. Grazing light for M1, real water body for M2, "first 60s
naive user" for M3, "one-line rayshader → surtgis swap" for M4. If a
sub-task does not have a clear "this is the case it has to handle",
write one before coding.
