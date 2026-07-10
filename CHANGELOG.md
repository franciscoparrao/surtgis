# Changelog

All notable changes to SurtGIS are documented in this file.

Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer 2.0.0](https://semver.org/). The project is still in
the `0.x` series, so minor versions may contain breaking changes; we try to
call them out under a `Breaking` heading when they happen.

## [Unreleased]

## [0.18.0] - 2026-07-10

Scientific-parity release (engine audit R2, "Horizonte v1.0+" package):
D8 flat resolution, geomorphons/solar/viewshed brought to reference-tool
parity, each validated against GRASS/WhiteboxTools. **This is the
1.0-candidate surface**: barring issues found in real-world use, 1.0.0 will
promote this release unchanged once the EMS paper is published.
**Contains breaking changes** — see below.

> ⚠️ **Geomorphons, solar radiation and viewshed results change** relative
> to ≤0.17.0 — in all three cases the old outputs were scientifically
> divergent (ad-hoc classification, unattenuated diffuse, target-inflated
> occlusion), not differently-conventioned. See the `Changed` entries.

### Breaking

- `ViewshedParams` is now `#[non_exhaustive]` with two new fields
  (`earth_curvature`, `refraction_coeff`): construct it via
  `ViewshedParams::default()` and set fields.
- `SolarParams` and `DiffuseModel` are now `#[non_exhaustive]` (pre-1.0
  extensibility policy): construct the params via `SolarParams::default()`
  and set fields.
- `GeomorphonParams` gains `skip` and `flatness_distance` fields (GRASS
  `skip`/`dist`) and is now `#[non_exhaustive]` (pre-1.0 extensibility
  policy): construct it via `GeomorphonParams::default()` and set fields.

### Changed

- **Viewshed: occlusion no longer includes `target_height`, and Earth
  curvature / refraction correction added** (audit divergence D4). The
  occlusion horizon was updated with `z + target_height`, making every
  intermediate cell block the line of sight as if it carried a raised
  target on top — with `target_height > 0`, cells that are plainly
  visible (a 2 m antenna behind a 1 m bump) were reported hidden.
  Visibility now tests the raised target against a horizon built from the
  bare terrain only, in both the Bresenham (`viewshed`) and XDraw
  (`viewshed_xdraw`) algorithms. New opt-in correction `earth_curvature`
  (with `refraction_coeff`, default 1/7 like GRASS r.viewshed; ArcGIS
  uses 0.13) lowers each cell by `(1 − k)·d²/(2R)` — on a flat plain the
  horizon of a 1.7 m observer now falls at the physical ~5 km. CLI:
  `--curvature` / `--refraction` on `viewshed` and `viewshed-xdraw`;
  Python: kwargs `earth_curvature=False`, `refraction_coeff=0.14286`.
- **Solar radiation: diffuse now derives from the attenuated global
  irradiance, plus the E₀ eccentricity correction** (audit divergence D3).
  The diffuse component was computed as a fixed proportion of the
  *top-of-atmosphere* irradiance — unattenuated, independent of
  transmittance/Linke turbidity and inconsistent with the attenuated beam
  (Perez received a TOA-based DNI different from the beam actually used).
  Now `ghi = beam_horizontal / (1 − diffuse_proportion)`,
  `dhi = diffuse_proportion × ghi`, and the DNI seen by Perez is exactly
  the attenuated beam normal. The Spencer (1971) eccentricity correction
  E₀ (±3.3% over the year) is applied to the extraterrestrial irradiance.
  Affects `solar_radiation`, `solar_radiation_shadowed` and both annual
  variants (which delegate). Diffuse (and slightly, total) values drop
  relative to previous releases — the old diffuse was systematically
  overestimated, most strongly at low sun angles.
  `diffuse_proportion` is now validated to `[0, 1)`.
- **Geomorphons rewritten for GRASS `r.geomorphon` parity** (audit
  divergence D2). The previous classification was an ad-hoc heuristic
  (segment counting); it now implements Jasiewicz & Stepinski (2013)
  exactly as the reference implementation does in its default mode
  (`comparison=anglev1`, basic correction): circular distance-based search,
  inner `skip` ring, `dist` flatness-threshold relaxation, the exact
  zenith/nadir ternary rule (tie → 0), the canonical 9×9 form matrix, and
  the nulled border ring of width `skip+1`. Output codes are unchanged
  (1–10, matching the GRASS category values). CLI gains `--skip` and
  `--flat-dist`; Python `geomorphons(...)` gains `skip=0` and
  `flatness_distance=0.0` keyword arguments. Validated cell by cell against
  GRASS (`benchmarks/validate_geomorphons_grass.py`).

### Added

- **Honest `SigV4 not implemented` error** (audit A2-cloud): signing an S3
  request with `AwsAuth` used to emit half-signed headers (no
  `Authorization: AWS4-HMAC-SHA256`), so private buckets answered 403 with
  valid credentials. It now returns an explicit `CloudError::Auth` pointing
  to presigned URLs / VPC endpoints / public HTTPS access. Real SigV4 is
  deferred until there is a concrete private-bucket use case.

- **D8 flow direction now resolves flat surfaces** with the Garbrecht &
  Martz (1997) double-gradient method, in the O(n) formulation of Barnes,
  Lehman & Mulla (2014) — closing audit divergence D6 and matching the
  behaviour of GRASS, WhiteboxTools and TauDEM. Flats (filled depressions,
  lake surfaces, plains in integer-quantized DEMs) previously came out as
  regions of code `0` that silently swallowed the drainage of every
  downstream algorithm (flow accumulation, watershed, stream network, HAND).
  Flats draining to the raster edge or to nodata (e.g. a coastal plain
  against a masked sea) are routed toward it; the boundary-touching cells
  themselves keep code `0` as terminal sinks — SurtGIS directions never
  point off-grid or into nodata. Fully enclosed flats and true pits keep
  code `0` as before. New public API:
  `hydrology::resolve_flats(&dem, &mut directions) -> FlatResolutionStats`
  for composing with custom D8 rasters; `flow_direction()` applies it
  automatically (no signature change — CLI, Python and WASM inherit the
  fix). Validated against WhiteboxTools and GRASS on quantized fractal
  DEMs (`benchmarks/validate_flats_gm.py`): flat coverage 99.8% (WBT:
  99.9%), and SurtGIS↔WBT agreement meets or exceeds the WBT↔GRASS
  reference baseline on direction agreement, log-accumulation correlation
  and stream-network IoU.

### Fixed

- CLI `hydrology fill-sinks --min-slope` default aligned with the library
  default (`0.01` → `1e-5`, audit D7 — the library was fixed in the
  Sprint-1 round but the CLI kept the old value, giving different results
  by the two routes).
- CLI `hydrology flow-direction` help no longer claims ESRI-style
  power-of-two direction codes (`1,2,4,…,128`); SurtGIS writes `0` = no
  outflow, `1`–`8` counter-clockwise from East.

## [0.17.0] - 2026-07-02

Engine-audit release: Sprints 2–4 (performance, structural debt, process
hardening). **Contains breaking changes** — see below.

> ⚠️ **`z_factor` changed to the GDAL/ArcGIS convention.** If you passed a
> non-default `z_factor` (especially the `111320` lat/lon hack), read the
> Breaking section. Geographic DEMs are now corrected automatically.

### Added

- **Nightly fuzzing** (`cargo-fuzz`) of the hand-rolled binary parsers:
  `read_geotiff_from_buffer`, the COG IFD/header parser (both byte
  orders) and tile decompression + TIFF predictors with overflow checks —
  the input of the COG reader arrives from the network. Seed corpus
  committed; crashes upload as CI artifacts.
- **CI hardening**: tests now run on macOS and Windows (core crates);
  the real-DEM integration fixtures download from the `test-fixtures-v1`
  release with sha256 verification, so the integration and GDAL/GRASS
  cross-validation suites actually run in CI (previously the fixture was
  absent and both suites silently never executed); `cargo-deny`
  (advisories/licenses/sources) with a documented ignore list; MSRV
  declared (`rust-version = "1.88"`) and verified by a pinned job.

### Fixed

- **Unbounded allocation in the native TIFF reader (OOM/DoS)** — found by
  the new fuzzer on its first run: a 338-byte file declaring ~63000²
  dimensions forced a single ~32 GB allocation because the reader used
  `Limits::unlimited()`. Decode buffers are now capped at
  `max(1 GiB, source_len · 4096)` — proportional to the input, still far
  above any real DEM. Regression test + corpus seed committed.
- **Cloud async hygiene**: the retry backoff used `std::thread::sleep`
  inside async code (two concurrent retries could freeze all I/O on the
  2-worker blocking runtime) — now `tokio::time::sleep`; `HttpClient`
  retries 429/500/502/503/504 honouring `Retry-After` (previously only
  transport errors retried, and the CLI re-opened the whole COG on 429);
  new structured `CloudError::HttpStatus`; the Zarr V2 metadata fallback
  no longer calls `block_on` inside an async fn (deadlock risk); blocking
  wrappers share one global runtime instead of one per reader.

### Breaking

- **`z_factor` now follows the GDAL/ArcGIS convention** in `slope`,
  `hillshade`, `multi-hillshade` and `curvature` (batch and streaming): it
  scales the **elevations** (`z' = z_factor · z`). The previous semantics
  scaled the *cell size* — the reciprocal — so a user coming from GDAL who
  passed `0.00000898` got slopes inflated by ~10¹⁰. Code that used the
  default `1.0` is unaffected. The legacy `z_factor = 111320` hack for
  lat/lon DEMs must be removed: geographic rasters are now handled
  automatically (below). For curvature the old scheme was additionally
  wrong for second derivatives (the cell size enters squared); elevations
  are now scaled exactly.
- **Geographic rasters get automatic per-row metric cell sizes** in
  `slope`, `aspect`, `hillshade` and `multi-hillshade`. When the raster's
  CRS is geographic (EPSG 4000–4999, `GEOGCS`/`GEOGCRS` WKT, or
  `+proj=longlat`), the gradient uses spheroidal per-row `dx`/`dy`
  (`terrain::spheroidal_grid::cell_dimensions`) — at 53°S one degree of
  longitude is ~40% shorter than at the equator, so the flat-earth
  computation was up to ~30% off in the E–W component. Rasters without a
  CRS keep the raw units (we cannot assume degrees).
- **Rectangular cells (dx ≠ dy) are now honoured** by the same four
  algorithms: each gradient component divides by its own cell size.
  `aspect` previously did not divide by cell size at all (implicit square
  assumption). Square-cell outputs are unchanged.

### Added

- **Multi-raster alignment validation** (`surtgis_core::raster::validate`):
  new `check_aligned` / `check_same_shape` helpers (shape + geotransform
  with relative tolerance + CRS compared only when both sides expose an
  EPSG code, avoiding the EPSG-vs-WKT false-mismatch footgun). Wired into
  every multi-raster entry point: the 15+ spectral indices, `composite`
  (whose pixel-wise fast path previously combined same-shape rasters with
  shifted origins silently), `mad`/`ir_mad`, pansharpening, `spi`/`sti`/
  `twi`. New structured error variants `Error::ShapeMismatch` and
  `Error::Misaligned`.
- **Canonical D8 module** (`hydrology::d8`): single source of truth for
  the D8 offset tables, distances, `encode`/`decode`, `opposite` and
  `downstream` — 19 files previously redefined these with three different
  conventions. The direction encoding written to rasters is unchanged.
  The row-major scan order used by fill/breach tie-breaking is now a
  separate, documented table (`SCAN_OFFSETS`). New integration test pins
  the fill → flow-direction → accumulation pipeline contract.
- `CRS::is_geographic()` heuristic (EPSG 4000–4999 block, `GEOGCS`/
  `GEOGCRS` WKT, `+proj=longlat`).

### Fixed

- **Streaming `normal-vector-deviation` and `spherical-std-dev` ignored
  the cell size** (gradients divided by 8 instead of 8·cellsize): for any
  DEM with cells ≠ 1 unit the streaming outputs were wrong. Found while
  unifying batch/streaming kernels; both now share one kernel with the
  batch (correct) semantics.
- `CRS::identifier()` no longer panics on WKT with multi-byte characters.

### Changed

- **Batch and streaming now share one kernel per algorithm** in the 11
  window algorithms (tpi, tri, dev, diff-from-mean, percent-elev-range,
  circular-variance, convergence, normal-vector-deviation,
  spherical-std-dev, vrm, northness/eastness) — previously each
  `process_chunk` re-implemented its batch kernel verbatim (the N–S
  mirrored hillshade lived in both copies for that reason). A new
  `streaming_batch_parity` test suite demands exact equality across strip
  seams and NaN holes.
- **`process_chunk` is now parallel** in those 11 algorithms: the
  streaming path (the recommended mode for large DEMs) was the only
  sequential one.
- **Hillshade is now transcendental-free per cell.** The classic
  slope/aspect formulation (atan, cos, sin, atan2, cos per cell) collapses
  algebraically to one sqrt and one division —
  `shade = (cosθz + sinθz·(dz_dy·sin az − dz_dx·cos az)) / √(1+|∇z|²)` —
  the same formulation gdaldem uses. Applies to `hillshade`,
  `multi-hillshade` (weights included: `w = 1 + (x·sin az + y·cos az)²/|∇z|²`)
  and both streaming variants. Numerical equivalence with the trigonometric
  form is pinned by tests to 1e-12.
- **`slope` percent output no longer computes `tan(atan(g))`** — it is `g`
  by identity; degrees/radians keep the single `atan`.
- **New `par_map_rows` row harness** (`maybe_rayon`): writes rows directly
  into the final `Array2` via `par_chunks_mut` instead of the
  `flat_map(|row| vec![...]).collect()` pattern (one allocation per row plus
  a fold/reduce merge). `slope`, `aspect`, `hillshade` and
  `multi-hillshade` migrated; their 3×3 inner loops now read through three
  row slices instead of per-access 2-D indexing (stride arithmetic + bounds
  checks), which also unlocks autovectorization.
- **Release profile: `lto = "fat"`** (was `thin`) for cross-crate inlining
  between core and algorithms.
- **Release binaries: new `x86_64-unknown-linux-gnu-v3` artifact** built
  with `-C target-cpu=x86-64-v3` (AVX2+FMA, CPUs ≥ ~2015). The baseline
  x86-64 artifact remains fully portable.

## [0.16.4] - 2026-07-02

Correctness release: Sprint 1 of the July 2026 engine audit. Every fix below
targets the same bug class — incorrect results returned as success.

> ⚠️ **Hillshades produced by SurtGIS ≤ 0.16.3 have inverted North–South
> lighting.** Regenerate any figure or derived product that used them.

### Fixed

- **Hillshade illumination was mirrored North–South** (`hillshade`,
  `multi-hillshade`, `hypsometric-hillshade`). The internal aspect used
  `atan2(-dz_dy, -dz_dx)` instead of `atan2(dz_dy, -dz_dx)`, so a north-facing
  slope was shaded as if it faced south — with the default NW sun (az 315°) the
  result was equivalent to lighting from the SW. Now matches `gdaldem hillshade`
  bit-exactly on synthetic slopes (218 vs the previous 104 for a north-facing
  tan=0.5 slope) and within ±1 DN (rounding) on real terrain. Every hillshade
  produced by previous versions has inverted N–S lighting. New directional
  regression test pins the GDAL value.
- **Priority-Flood lost valid data enclosed by a NaN collar.** Only
  physical-border cells seeded the queue, so the valid interior of a
  reprojected DEM (WGS84→UTM leaves a NaN curtain) was unreachable and came out
  all-NaN. Valid cells adjacent to nodata now seed the queue as drainage
  outlets (Barnes et al. 2014), matching `fill_sinks`. Ported the nodata-curtain
  regression test.
- **`GeoTransform::geo_to_pixel` returned NaN for fine-resolution geographic
  rasters.** The degeneracy check used an absolute determinant threshold
  (`1e-10`); a 1 m pixel in EPSG:4326 gives det ≈ 8e-11, so valid drone-imagery
  transforms were declared degenerate — silently breaking `rasterize` and
  `geo_to_pixel`. The threshold is now relative to the transform's scale.
- **Float nodata matching corrupted small valid values.** `is_nodata` used an
  absolute epsilon (`EPSILON * 100`), so with `nodata = 0.0` any value below
  ~1.2e-5 (NDVI ≈ 0, low reflectances) was classified nodata and rewritten to
  NaN on read — irreversible corruption. Matching is now exact (as GDAL and
  rasterio do), which also fixes large sentinels missed when off by 1 ULP.
- **`read_geotiff` band index differed between backends.** The native backend
  was 0-based while the GDAL feature was 1-based, so `Some(1)` read a different
  band depending on compile flags. Both are now 0-based; the GDAL backend
  converts internally.
- **Zarr reader read the wrong columns for western-hemisphere bboxes on 0–360
  grids (ERA5).** Longitudes were converted to −180..180 *and sorted*, breaking
  the mapping between coordinate positions and physical array indices; any
  Chilean bbox read columns from the wrong hemisphere. The reader now keeps the
  dataset's physical coordinate order and converts the bbox to the dataset
  convention instead (as the NetCDF reader always did). Bboxes crossing the
  0/360 seam are rejected with an explicit error instead of returning wrong
  data. Regression tests cover western, eastern and seam-crossing cases.
- **COG reader returned an all-nodata raster as success for striped TIFFs.**
  A TIFF without `TileOffsets` never fetched anything and every tile was
  silently "skipped". Striped (non-tiled) TIFFs are now rejected at open with a
  clear error, and any non-sparse tile that fails to fetch/decode aborts the
  read instead of returning a partially-empty raster. Sparse COGs (byte count
  0 = intentionally missing tile) still fill with nodata as before.
- **Sentinel nodata round-trip was incoherent for external tools.** Reading a
  file with `GDAL_NODATA = -9999` normalized pixels to NaN but kept the
  sentinel in metadata, so a subsequent write emitted `GDAL_NODATA = -9999`
  over NaN pixels — GDAL/QGIS then treated the NaNs as valid data. Metadata now
  follows the pixels to NaN.
- **`GDAL_NODATA` was written as a BYTE-typed tag** (via `as_bytes()`), which
  SurtGIS's own reader rejected (`InvalidTypeForTag`) — the nodata of
  SurtGIS-written files was invisible to SurtGIS itself. The tag is now written
  as a proper ASCII tag; reading falls back to BYTE for files written by
  ≤ 0.16.3.
- **`hillshade` declared `nodata = 0.0`**, but 0 is a valid shade (full
  shadow) — masking nodata erased real cast shadows. Edges and nodata cells are
  now NaN, consistent with the rest of the terrain algorithms.
- `hypsometric_hillshade` multiplied a 0–255 hillshade by the normalized
  elevation while documenting output in [0, 1]; it now forces the internal
  hillshade to normalized mode, honouring the documented contract.
- Removed leftover debug `eprintln!` lines from `resample_to_grid` that
  polluted stderr of every consumer.

## [0.16.3] - 2026-07-01

### Fixed

- **`rasterize` now preserves the reference CRS.** The command built the mask
  with the reference geotransform but never copied its CRS, so the output
  GeoTIFF was written without one and downstream tools (rasterio/GDAL/QGIS) read
  it as `LOCAL_CS[metre]` — mislabelling geographic (degree) grids as metres and
  breaking alignment, area, and reprojection. `handle_rasterize` now inherits
  `--reference`'s CRS onto the result (as `clip` already did); the GeoTIFF writer
  already encoded geographic CRS via `ModelTypeGeographic` GeoKeys. Reported by a
  glacier-mapping study rasterizing the IPG 2022 inventory onto EPSG:4326
  Sentinel-2 tiles.

### Changed

- **`rasterize` fill is now a scanline algorithm** instead of a per-cell
  point-in-polygon test over each polygon's bounding box. For north-up
  transforms it intersects each row's pixel-centre latitude with every ring edge
  (exterior and holes) and fills the columns between even-odd crossing pairs —
  the same centre-based half-open rule GDAL/`rasterio` use, so real geographic
  polygons rasterize identically (0-px discrepancy). Complexity drops from
  `O(bbox_cells · vertices)` to `O(edges · rows + filled_cells)`; the reporting
  study measured ~1000× on a 6912² grid. Rotated transforms fall back to the
  exact per-cell test.

## [0.16.2] - 2026-06-29

### Added

- **Refined Lee speckle filter** (`imagery::refined_lee_filter`, Lee 1981) — the
  edge-aligned refinement of the classic Lee filter: it estimates the local
  statistics from the half-window on the homogeneous side of the dominant edge
  (chosen among 4 orientations), so it preserves edges and linear features
  instead of blurring across them. CLI `imagery sar-lee --refined`; Python
  `sar_lee_filter(..., refined=True)`. Closes the SAR speckle follow-up from the
  v0.16.0 A3 work. 3 unit tests (incl. edge-preservation vs the classic Lee).

## [0.16.1] - 2026-06-28

LAHARZ hardening from the Nevados de Chillán (Ñuble) field validation: physical
small-volume runout, multi-source seeding, and build-doc fixes. Note: the
`LaharzParams` API changed (`source` → `sources`, new `spread_aspect`); the
type is days old (0.16.0) and the project is pre-1.0.

### Fixed

- **LAHARZ small-volume runout is now physical** (the Ñuble calibration target).
  Two changes, fitted against the Diguillín fixture (documented 2020/2021
  Nevados de Chillán lahars):
  1. The valley-perpendicular cross-section is now **confined to the valley** —
     an arm stops when it drops below the thalweg (an open downslope), not just
     at a wall. This alone removes most of the artefact: a 1e6 m³ lahar's runout
     drops from ~65 km to ~20 km.
  2. A calibrated **lateral-spread** term (`LaharzParams.spread_aspect`, the
     minimum width-to-depth aspect ratio; CLI `--spread-aspect`, Python
     `spread_aspect=`) mantles the valley floor up to `w = sqrt(aspect·A)`
     instead of incising a 1-cell thread. The lahar preset now defaults to
     `aspect = 1000`, bringing 1e6 m³ runout to ~6 km (observed: deposits within
     1–5 km) and 1e7 to ~12 km (observed ~15). The largest volumes (≥1e8) remain
     ~30 % long where the valley confines the spread; spatial IoU against an
     independent HAND-based model stays modest as the two schemes distribute
     inundation differently — runout is the robust cross-comparable quantity.

### Changed

- **LAHARZ now takes multiple sources** (`LaharzParams.source: (usize,usize)` →
  `sources: Vec<(usize,usize)>`; CLI `hydrology laharz --source "r,c;r,c"`;
  Python `laharz_compute(..., sources=[(r,c), ...])`). Each source routes as an
  independent flow and the footprint is their union. Driven by Ñuble validation
  feedback: seeding a multi-drainage edifice at the summit sends the single D8
  path down the wrong drainage. Docs and `--source` help now state explicitly:
  **seed proximal channel cells, not the summit**, and note that small-volume
  runout (a near-thalweg ribbon in confined valleys) is the least trustworthy
  output, pending cross-section/coefficient calibration against documented events.

### Fixed

- **README build notes** for `cargo install --git`: name the package explicitly
  (`cargo install --git <url> surtgis`, since the repo ships two binaries) and
  document the `libhdf5-dev` / `libnetcdf-dev` system dependencies.

## [0.16.0] - 2026-06-28

Multi-hazard release: a full set of geomorphic-hazard algorithms and Python
parity, driven by a multi-hazard susceptibility study in Ñuble, Chile (volcanic
/ lahar, landslide, wildfire, flood around Nevados de Chillán).

### Added

- **LAHARZ inundation** (`hydrology::laharz`) — the Iverson, Schilling &
  Vallance (1998) semi-empirical lahar / debris-flow hazard-zone model. From a
  source it marches down the D8 flow path, filling the valley-perpendicular
  transect to the volume-scaled cross-section `A = c_A·V^(2/3)` at each step and
  stopping when the planimetric area reaches `B = c_B·V^(2/3)`. Flow-type
  presets (lahar / debris-flow / rock-avalanche; Griswold & Iverson 2008). CLI
  `hydrology laharz --flow-dir --source "r,c" --volume --flow-type`; Python
  `laharz_compute`. 4 unit tests incl. an analytic prismatic-V-valley case.
  Completes the A2 lahar request alongside the energy cone and Melton ratio.
- **Fluvial toolkit in the Python bindings** — `chi_compute` and `ksn_compute`
  return per-cell rasters; `knickpoints_compute`, `concavity_compute` and
  `divide_migration_compute` return lists of dicts (with coordinates / basin
  ids / metrics). Closes the last Python-parity gap from the Ñuble requests so a
  full channel-profile analysis can run in one process without shelling out to
  the CLI.
- **Energy-cone inundation** (`hydrology::energy_cone`) — the Malin & Sheridan
  (1982) energy-line model for volcanic mass-flow / lahar runout. From one or
  more sources an energy line descends at angle `φ` (`H/L = tan φ`); a cell is
  reached where the line lies above the ground, and the output is the energy
  height above ground `max(0, E − z)`. CLI `hydrology energy-cone --source
  "r,c" --cone-angle --collapse-height`; Python `energy_cone_compute`. 5 unit
  tests incl. an analytic flat-disc case. Pairs with the Melton ruggedness
  ratio; a valley-confined LAHARZ (Iverson et al. 1998) model remains a
  follow-up.
- **Excess topography** (`terrain::excess_topography`) — the height of terrain
  above a threshold hillslope angle (Blöthe, Korup & Schwanghart 2015), a
  first-rate landslide conditioning factor. Reconstructs the maximal
  slope-limited surface `S ≤ z` with `|∇S| ≤ tan θ` via the fast sweeping
  method (Zhao 2005, as in TopoToolbox's `fsm2d`) and reports `z − S ≥ 0`.
  CLI `terrain excess-topography --threshold <deg>`; Python
  `excess_topography_compute`. Validated against an analytic ramp. 4 unit tests.
- **Lee speckle filter** (`imagery::lee_filter`) — the classic Lee (1980)
  adaptive MMSE filter for SAR multiplicative speckle. Smooths homogeneous
  areas toward the local mean while preserving edges and point targets, with an
  `ENL` (equivalent number of looks) parameter. Completes the SAR water-mapping
  path (run it before `sar_water_mask`). CLI `imagery sar-lee --window-size
  --looks`; Python `sar_lee_filter`. 4 unit tests. The directional *refined*
  Lee remains a possible follow-up.
- **Melton ruggedness ratio** (`hydrology::melton_ruggedness`) — per-basin
  relief energy `(H_max − H_min) / √area`, a first-order debris-flow / lahar
  screening metric (Melton 1965; classes after Wilford et al. 2004). New CLI
  `hydrology melton <watersheds> --dem <dem> --cell-size <m> [--output <ratio.tif>]`
  prints a per-basin table with a geomorphic class and can write the ratio back
  as a per-basin raster for use as a susceptibility conditioning factor.
- **Cloud module in the Python bindings** (`cloud` feature, on by default) — the
  PyPI wheel now exposes remote I/O so a full pipeline can live in one process:
  `cog_fetch(url, bbox)`, `cog_fetch_full(url)`, `cog_info(url)`,
  `stac_search(catalog, bbox=, datetime=, collections=, limit=)` and
  `stac_sign_href(href, collection)` (Planetary Computer SAS signing). Fetches
  return `(numpy_f64_array, meta)` where `meta["transform"]` is GDAL-order
  (`rasterio.transform.Affine.from_gdal(*meta["transform"])`). Datetimes are
  normalised to RFC 3339 so strict catalogs (Earth Search) accept bare dates.
  Disable with `--no-default-features` for pure-compute wheels.
- **SAR water / flood primitives** (`imagery::sar`) — `linear_to_db` /
  `db_to_linear` backscatter unit conversion, `dual_pol_water_index`
  ((VV − VH)/(VV + VH)), and `sar_water_mask` (threshold → binary u8 water mask,
  1=water / 0=land / 255=nodata). New CLI `imagery sar-db`, `imagery
  sar-water-index --co-pol --cross-pol`, and `imagery sar-water-mask --threshold
  [--water-above]`. Multi-temporal backscatter compositing and amplitude change
  detection reuse the existing `temporal_min` / `median_composite` and
  `raster_difference` / `ir_mad`. A Lee / refined-Lee speckle filter is a
  tracked follow-up.

  All three were requested by a multi-hazard susceptibility study in Ñuble,
  Chile (volcanic/lahar, landslide, wildfire, flood around Nevados de Chillán).

### Changed

- **Verbose COG tile-decode diagnostics are now opt-in.** The `[cog]` / `[pred]`
  per-read tile logging (added while fixing the `bps=15` striping bug) is
  silenced by default and re-enabled with `SURTGIS_COG_DEBUG=1`. FATAL decode
  errors are still always printed. Keeps the Python `cog_fetch` output clean.

## [0.15.5] - 2026-06-24

### Added

- **Multi-band GeoTIFF reading in `surtgis-core`** — `io::read_geotiff_bands`
  and `io::read_geotiff_bands_from_buffer` read every band of a multi-band
  GeoTIFF into a `Vec<Raster<T>>` (the read-side counterpart of the
  `write_geotiff_multiband` writer added in v0.14.8), preserving per-band
  georeferencing and nodata.
- **Line and MultiPoint GeoJSON geometries** are now read by the GeoJSON
  reader (previously points and polygons only), broadening vector ingestion.
- **`mimalloc` is now an optional CLI feature** (on by default). It compiles C,
  which blocked builds on minimal toolchains (no system C compiler) and
  complicated musl/static targets. Build with `--no-default-features` (plus the
  features you want) to fall back to the system allocator and a pure-Rust /
  no-cc build. Reported from a multi-hazard study in Ñuble, Chile.
- **`stac fetch-mosaic --max-scenes`** — visible alias for `--max-items`, and a
  cap-hit notice plus per-scene progress lines that remain visible in non-TTY
  pipelines (the spinner auto-hides off a terminal). Helps users scope wide
  datetime/bbox queries before they grow RAM.

### Fixed

- **STAC collection flag is now spelled consistently.** `stac search` took
  `--collections` (plural) while `stac fetch` / `fetch-mosaic` took
  `--collection` (singular); both spellings are now accepted on all three via
  visible aliases. No more failed runs from the mismatch.

## [0.15.4] - 2026-06-10

### Added

- **Complex rasters in `surtgis-core`** (new feature `complex`) —
  `Raster<Complex<f32>>` / `Raster<Complex<f64>>` for interferometric
  phase data, plus the minimal InSAR helpers in `raster::complex`:
  `complex_from_parts` / `complex_to_parts` (assemble/split via
  real+imaginary band rasters — also the persistence path: complex
  rasters travel as `(re, im)` GeoTIFF pairs through the existing
  writers) and `magnitude` / `phase` (wrapped phase via
  `atan2(im, re)`). Nodata convention: a complex cell is nodata when
  both parts are NaN. FFT helpers deliberately left to insar-rs.
  Implements SPEC_SURTGIS_ECOSYSTEM_FOUNDATION P2.1; trigger:
  insar-rs starting.

### Changed

- **`RasterElement` split into `RasterCell` + `RasterElement`** to
  admit unordered cell types. `RasterCell` (new) is the minimum a
  type needs to live in a `Raster` — copy, zero, nodata convention —
  and is now the bound on `Raster<T>` itself. `RasterElement` is now
  a subtrait (`RasterCell + PartialOrd + NumCast` + min/max/f64
  casts) and remains the bound for statistics, resampling, I/O and
  the algorithm crates. **Source-compatible for all consumers**:
  every numeric type still implements both traits, and generic code
  bounding on `RasterElement` compiles unchanged (the entire
  workspace needed zero downstream edits). Only a hand-written
  external `impl RasterElement for CustomType` would need to split
  its impl in two — no such implementor is known.

## [0.15.3] - 2026-06-10

### Added

- **Zarr write in `surtgis-cloud`** (feature `zarr`) —
  `zarr_writer::write_cube_zarr` persists a `Cube<f64>` as a Zarr v2
  directory store (one `[time, y, x]` f64 variable per band, chunked
  one time step per chunk, plus `time`/`y`/`x` coordinate arrays with
  CF time units), and `write_raster_zarr` does the 2-D single-raster
  case. Hand-rolled v2 writer following the xarray
  `_ARRAY_DIMENSIONS` convention — validated against two independent
  readers: `zarrs` (exact round-trip) and xarray (dims, CF-decoded
  datetimes, values, lazy dask chunks aligned). CRS/transform travel
  as `surtgis:epsg` / `surtgis:geotransform` group attributes.
  Completes the write half of SPEC_SURTGIS_ECOSYSTEM_FOUNDATION
  P1.3 (read existed since v0.6.0); trigger: datacube-rs as ARD
  backing store. v1 scope: uncompressed chunks, f64; compression and
  tuned chunking once datacube-rs profiles real loads.

## [0.15.2] - 2026-06-10

### Added

- **`cube::Cube` in `surtgis-core`** — aligned `(time, band)` raster
  stacks for data-cube workflows. Holds `n_times × n_bands`
  single-band rasters verified at construction to share one grid
  (shape, transform, CRS; strictly increasing Unix-epoch timestamps,
  unique band names). Aligned iteration: `pixel_series(row, col,
  band)` (per-pixel time series), `band_values(row, col, t)`, and
  `chunks(chunk_rows)` yielding one 2-D view per slice over the same
  rows — the bounded-memory access pattern for temporal algebra.
  Deliberately a **container only**: temporal analysis lives in
  consumer engines (datacube-rs, unmix-rs). Implements
  SPEC_SURTGIS_ECOSYSTEM_FOUNDATION P1.2, triggered by datacube-rs
  starting. End-to-end test: three dated GeoTIFFs from disk →
  Cube → aligned series and chunk iteration.

## [0.15.1] - 2026-06-10

### Added

- **GeoParquet point I/O in `surtgis-core`** (new feature `parquet`,
  off by default — pure-Rust via the `parquet` crate, no Arrow):
  `vector::geoparquet` with the columnar `PointTable` type (one f32
  column per embedding dimension scales without per-feature
  HashMaps), `write_geoparquet_points` / `read_geoparquet_points`,
  and `FeatureCollection` bridges `write_geoparquet` /
  `read_geoparquet`. Output is GeoParquet 1.0 (WKB geometry +
  `geo` file metadata, Snappy): DuckDB spatial auto-detects the
  geometry column as native GEOMETRY and GeoPandas reads it
  directly (both validated). Implements
  SPEC_SURTGIS_ECOSYSTEM_FOUNDATION P1.1, triggered by geoembed-rs
  starting. v1 scope: Point geometry, f64/f32/i64/string columns,
  no nulls; CRS travels as a `surtgis:epsg` metadata key (the
  GeoParquet `crs` field is `null` — PROJJSON generation is out of
  scope), so non-SurtGIS readers see correct geometry but must be
  told the CRS out-of-band.

## [0.15.0] - 2026-06-10 — "Ecosystem Foundation"

Minor release consolidating SurtGIS as the base of the Rust
geoscience engine family (datacube-rs, geoembed-rs, insar-rs,
rainflow, sedlink, …). Implements P0 of
`SPEC_SURTGIS_ECOSYSTEM_FOUNDATION`: the public API of
`surtgis-core` is now **declared stable under SemVer** (breaking
changes pre-announced one release ahead), sampling and 2-D tiling
are library APIs, and the D∞ pipeline is complete and
cross-validated. No breaking changes vs v0.14.10.

New documentation: book chapters *Using surtgis-core as a library*
and *Extension contract* (the four rules a sibling engine signs:
depend on core only, exchange `Raster<T>` + `GeoTransform` + `CRS`,
stay autonomous, pin the 0.15 baseline).

### Added

- **`sampling` module in `surtgis-algorithms`** — the library-level
  primitives behind `surtgis extract` / `extract-patches`, exposed
  for sibling crates (geoembed-rs, roam, pointproc):
  - `sample_at_points(rasters, points)` — multi-raster values at
    world-coordinate points; `None` for out-of-bounds or non-finite.
  - `extract_patches(bands, centers, params)` — fixed-size
    multi-band windows as a flat `[n, bands, size, size]` f32 tensor
    with NaN-fraction filtering and a `kept` index for label
    alignment. Same window semantics as the CLI.
  - `grid_points_in_polygon(polygon, reference, stride, margin)` —
    regular grid of cell-centre points inside a polygon; pass
    `margin = size / 2` to guarantee valid patch centres.
  The `extract` CLI handler now delegates its sampling loop to
  `sample_at_points`.
- **`tiling::TileGrid` in `surtgis-core`** — 2-D tile iteration with
  overlap. Each `Tile` carries a core extent (cores tile the grid
  exactly) and a read extent (core + clamped halo), the standard
  pattern for seam-free tiled kernels and embedding inference. The
  2-D counterpart of `StripProcessor`.
- **`hydrology::flow_accumulation_dinf`** — D-infinity flow
  accumulation as a standalone function taking only the D∞ angle
  raster (output of `flow_direction_dinf`; TauDEM `ang` rasters use
  the same convention). Flow from each cell splits between the two
  D8 neighbors bracketing its angle, proportional to the angular
  offset (Tarboton 1997, eq. 6). Processing is a dependency-driven
  topological sort (Kahn), matching the D8 `flow_accumulation`
  structure — no DEM or elevation sort needed. Headwater cells have
  accumulation 0 (same convention as D8). `flow_dinf` is now a thin
  wrapper over `flow_direction_dinf` + `flow_accumulation_dinf`.
- **CLI `hydrology flow-accumulation-dinf <angles> <output>`** —
  closes the D∞ pipeline gap (direction existed since v0.10.x,
  accumulation was library-only inside `flow_dinf`).
- **CLI `hydrology twi --method d8|dinf|mfd`** — the TWI pipeline
  accumulation step can now use D-infinity or MFD routing instead
  of D8. Default remains `d8` (no behaviour change).
- **`benchmarks/validate_dinf_accumulation.py`** — cross-validation
  of D∞ accumulation against WhiteboxTools 2.4.0 on the Andes 30 m
  fixture: log-accumulation Pearson r = 0.9796, median relative
  error 0.000%, 96.8% of cells within 10%, and identical
  contributing area at the main outlet (251,222 cells). Residual
  differences are confined to flats, where WBT applies its own
  flat-routing and SurtGIS marks flats as pits.

### Changed

- `flow_dinf` accumulation no longer sorts cells by elevation
  (O(n log n) + 24 B/cell) — the dependency-driven pass is O(n)
  with 4 B/cell. Results are unchanged for Tarboton-conformant
  angles (receivers are always strictly downslope). Nodata cells in
  the accumulation output are now NaN (previously 0.0), consistent
  with the raster's NaN nodata metadata.

## [0.14.10] - 2026-06-08

Patch release. Closes **ROADMAP item O** ("Mosaic seams + colour
balance") from the 2026-06-08 OTB-survey backlog. **The
2026-06-06 survey backlog is now entirely shipped or
trigger-driven** — items G–H, J–O all done; only I (3D Tiles
export) remains, gated on a Cesium-user request.

### Added

- **`imagery::histogram_match`** — empirical CDF (rank-based)
  matching. For every source pixel `v`, computes its rank in the
  sorted finite source values, looks up the value at the same
  rank in the sorted reference, and emits a linearly-interpolated
  value between adjacent reference samples. Non-linear; matches
  the full distribution shape, not just `(μ, σ)`. Use when
  source and reference were captured under noticeably different
  conditions (different sun angle, atmosphere, sensor gain).

- **`imagery::moment_match`** — linear `(μ, σ)` matching:
  `v' = (v − μ_src) · (σ_ref / σ_src) + μ_ref`. Cheap and
  monotonic. Fits Gaussian-shaped reflectance distributions
  well. Rejects sources with zero variance.

- **`imagery::feather_mosaic`** — distance-weighted blend of
  aligned rasters. For each input raster, computes the Chamfer
  3-4 distance transform from each valid pixel to the nearest
  invalid cell *or* raster boundary; pixels deep inside a tile
  get a high weight, pixels at edges fade to zero. Output is
  the distance-weighted average across inputs:
  `out(p) = Σᵢ d_i(p) · v_i(p) / Σᵢ d_i(p)`. Hides hard tile
  seams without the cost of solving the full min-cost-path seam
  problem. All-NaN positions propagate to NaN.

### Internal numerical kernel

- **`chamfer_distance_to_invalid`** — two-pass Chamfer 3-4
  distance transform on a `Raster<f64>`. Initialises invalid
  cells *and* the raster border to distance 0 so the feather
  fades naturally at tile edges. Returns `Vec<f32>` in
  raster-pixel units scaled by 3 (irrelevant for the feather
  weight ratio).

### CLI

- **`surtgis imagery color-balance histogram <SOURCE> <REFERENCE>
  <OUTPUT>`** — full CDF matching, single band.
- **`surtgis imagery color-balance moments <SOURCE> <REFERENCE>
  <OUTPUT>`** — linear (μ, σ) matching.
- **`surtgis imagery mosaic-feather <OUTPUT> --input PATH
  [--input ...]`** — distance-weighted feather blend, ≥ 2
  aligned inputs.

### Tests

10/10 new tests pass:

- **Histogram-match invariants**: `src == ref` produces output
  ≈ source bit-for-bit; remapping `[0..100)` to `[200..300)`
  recovers the reference's `(μ, σ)` within 0.5; NaN
  passthrough.
- **Moment-match invariants**: output `(μ, σ)` equal reference's
  to 1e-9; zero-variance source rejected; NaN passthrough.
- **Feather-mosaic invariants**: two identical-value rasters
  preserve the value (interior cells); two non-overlapping
  halves with mid-stripe overlap give monotonically-increasing
  blend across the overlap zone; all-NaN positions produce NaN;
  mismatched-shape error.

### Backlog status

Survey 2026-06-06 status after this release: G ✓ (Martini RTIN),
H ✓ (TopoToolbox depth), I 📋 trigger-driven (3D Tiles export
pending Cesium user), J ✓ (texture deepening, v0.14.4), K ✓
(SLIC + Felzenszwalb, v0.14.5), L ✓ (radiometric calibration,
v0.14.6), M ✓ (pansharpening, v0.14.7), N ✓ (MAD/IR-MAD,
v0.14.9), **O ✓** (this release). Plus the v0.14.8 stack
utility surfaced during the M validation.

**Net: 7 patch releases this cycle, every non-trigger-driven
backlog item shipped.**

## [0.14.9] - 2026-06-08

Patch release. Advances **ROADMAP item N** ("MAD / IR-MAD change
detection") from the 2026-06-08 OTB-survey backlog. After this
release: G ✓, H ✓, I trigger-driven, J ✓, K ✓, L ✓, M ✓,
**N ✓**. Only O (mosaic seams + color balance) remains as a
non-trigger-driven backlog item.

### Added

- **`imagery::mad`** — Multivariate Alteration Detection
  (Nielsen, Conradsen & Simpson 1998). Solves the canonical
  correlation analysis (CCA) between two `B`-band timestamps and
  returns the `B` MAD variates `MAD_i = U_i − V_i` plus the
  canonical correlations `ρ_i ∈ [0, 1]`. MAD_1 captures the
  strongest change signal, with decreasing magnitude as `i` grows.

- **`imagery::ir_mad`** — Iteratively Reweighted MAD (Nielsen
  2007). Re-estimates the joint covariance over pixels weighted
  by `1 − F_χ²(T², B)` where T² is the standardised squared norm
  of the MAD vector. Pixels that look "unchanged" get higher
  weights, so the covariance is increasingly estimated on the
  no-change population — the canonical signal-to-noise regime.
  Returns the MAD variates, the canonical correlations, a
  per-pixel **no-change probability** raster (the χ² weight),
  and the iteration count. Supports diagonal regularisation
  (Nielsen 2007 §IV.B) for near-singular covariance.

- **CLI: `surtgis imagery change-detection`** with two subcommands:
  - `mad --output-dir DIR --t1 PATH [--t1 ...] --t2 PATH [--t2 ...]`
    — repeated `--t1` / `--t2` for multi-band inputs; outputs
    `mad_variate{NN}.tif` and prints the canonical correlations.
  - `ir-mad --output-dir DIR --t1 ... --t2 ... [--max-iter 25]
    [--tol 0.001] [--regularisation 0.0]` — same plus
    `irmad_nochange_prob.tif`.

### Internal numerical kernels

The CCA solver needed Cholesky factorisation and triangular
solves on small `B × B` matrices (typically `B ≤ 10`). All three
are implemented in-module without pulling in BLAS / LAPACK:

- **`cholesky_lower`** — Cholesky-Banachiewicz. Rejects
  non-positive-definite input with a clear error pointing to the
  problematic diagonal.
- **`solve_lower_tri`** / **`solve_upper_tri_transposed`** —
  forward / back substitution.
- **`lower_incomplete_gamma`** — series + continued-fraction
  expansion (Numerical Recipes pattern) for χ²(k) CDF.
- **`gamma_ln`** — Lanczos approximation, ~1e-14 precision.

The symmetric eigenproblem on `MᵀM` reuses the existing Jacobi
solver from `classification::pca::jacobi_eigen` (made
`pub(crate)` in v0.14.7) — single source of truth for the
algorithms-crate symmetric eigensolver across PCA, PCA
pansharpening, and now MAD.

### Tests

11/11 tests in `imagery::mad::tests` pass:

- **Numerical-kernel tests**: Cholesky roundtrip on a 2×2 SPD
  matrix, Cholesky rejection on a non-positive-definite matrix,
  triangular-solve roundtrip, χ²(k) CDF at known values
  (`χ²(2)` at `x=2` matches `1 − e⁻¹`).
- **MAD invariants**: `T1 == T2` yields MAD variates `< 1e-9`
  and all canonical correlations `≈ 1.0`; the planted-change
  test plants a 4×4 high-DN block into T2's band 1 (on a
  20×20 two-band scene with linearly independent bands) and
  asserts the mean `|MAD_1|` inside the block is more than 3×
  the outside mean.
- **Validity**: canonical correlations stay in `[0, 1]` under
  deterministic noise; NaN in any input propagates to NaN in
  every MAD variate.
- **IR-MAD**: converges within 15 iterations on the planted-
  change scenario and the per-pixel no-change probability is
  significantly lower inside the change block than outside.
- **Error paths**: mismatched band counts and empty inputs
  reject cleanly.

### Backlog status

The 2026-06-06 OTB-deepening survey is now substantively
complete: J, K, L, M, N all shipped. **Remaining backlog**: O
(mosaic seams + color balance — polish on the existing
composite pipeline). I (3D Tiles export) remains trigger-driven
pending a Cesium-user request.

## [0.14.8] - 2026-06-08

Patch release. Closes a gap surfaced during the v0.14.7 pansharpening
validation: there was no native way to package multiple
single-band outputs as a single multi-band GeoTIFF for visualization
in QGIS / ArcGIS. Until now the workaround was an external
`rasterio` / `gdal_translate` round-trip.

### Added

- **`core::io::write_geotiff_multiband`** — new native writer for
  multi-band GeoTIFFs. Generic over `RasterElement`. Supports 1, 3
  or 4 bands (`Gray32Float`, `RGB32Float`, `RGBA32Float`); for
  `N > 4` the function returns a clear error pointing to the
  GDAL backend. All band rasters must share shape; GeoTIFF metadata
  (CRS, transform, nodata) is inherited from `bands[0]`. Per-band
  values are cast to `f32` and interleaved pixel-by-pixel
  (chunky planar config).

- **`encode_multiband_image::<CT>`** — internal generic helper
  parameterised over the tiff crate's `ColorType` trait, so the
  same GeoTIFF-tag plumbing (scale, tiepoint, geokey, nodata)
  serves Gray / RGB / RGBA. Single source of truth for the tag
  emission across band counts.

- **CLI: `surtgis imagery stack <OUTPUT> --band <PATH> [--band ...]`**
  — stacks 1, 3 or 4 single-band rasters into a multi-band
  GeoTIFF. Order of `--band` flags determines band order in the
  output (e.g. pass R, G, B for a standard RGB stack).

### Performance note

The validation pipeline (which previously used `rasterio` to
stack the per-band pansharpening outputs) now uses the native
`stack` command. Wall time for a 1.93 M-cell 3-band stack:
**~90 ms** end-to-end (read 3 single-band TIFFs + interleave +
write 3-band TIFF), comparable to the `rasterio` round-trip
within measurement noise.

### Tests

5 new tests in `core::io::native::tests`:
- `multiband_rgb_roundtrip` — round-trip 3 ramp bands, decode the
  written file with the `tiff` crate directly, assert
  `SamplesPerPixel == 3`.
- `multiband_rgba_writes_four_samples` — same for 4 bands.
- `multiband_rejects_unsupported_band_count` — passing 2 bands
  produces a clear error message mentioning the allowed counts.
- `multiband_rejects_mismatched_shapes` — bands with different
  dimensions error out.
- `multiband_preserves_crs_and_transform` — write a 3-band RGB
  stack with `EPSG:32719` and a UTM-like geotransform; inspect the
  raw GeoTIFF tags (33550 scale, 33922 tiepoint, 34735 geokey
  directory) and assert each round-trips.

### Backlog status

Survey 2026-06-06 remains: G ✓, H ✓, I trigger-driven, J ✓, K ✓,
L ✓, M ✓. The stack writer is a foundation-layer enabler — not a
ROADMAP item by itself, but a precondition for cleaner CLI demos
of every multi-band algorithm shipped in J–M.

## [0.14.7] - 2026-06-08

Patch release. Advances **ROADMAP item M** ("Pansharpening") from
the 2026-06-08 OTB-survey backlog. Ships three classical
pansharpening algorithms covering the practical spread: a
spectral-ratio method (Brovey), a covariance-based method (PCA),
and a Gram-Schmidt orthogonalisation method (Laben & Brower 2000).
All operate on arbitrary band counts and follow a common contract.

### Added

- **`pansharpening::brovey`** — Brovey transform (Gillespie et al.
  1987). `out_b = MS_b · pan / mean(MS)`. Fast, simple, can
  oversaturate radiometric values when synthetic and real pan
  differ strongly.

- **`pansharpening::pca_pansharpen`** — PCA pansharpening (Chavez
  et al. 1991). Builds the covariance matrix of the MS bands,
  diagonalises it with the shared Jacobi solver from
  `classification::pca::jacobi_eigen` (now `pub(crate)`),
  histogram-matches the pan to the first PC, and back-projects.
  Works for any number of bands.

- **`pansharpening::gram_schmidt`** — Gram-Schmidt pansharpening
  (Laben & Brower 2000, US patent 6,011,875 expired January 2018,
  now public domain). Builds the centred Gram-Schmidt sequence
  `(P_lr, MS_1, ..., MS_B)`, records the regression coefficients
  `φ_{k,i}`, substitutes the histogram-matched pan into the first
  GS slot, and inverts. The same `φ` coefficients are used in
  the forward and inverse passes — that's the elegant property
  that makes the substitution clean.

- **CLI: `surtgis imagery pansharpen`** with three subcommands:
  - `brovey --pan PATH --band PATH [--band ...] --output-dir DIR`
  - `pca --pan PATH --band PATH [--band ...] --output-dir DIR`
  - `gram-schmidt --pan PATH --band PATH [--band ...] --output-dir DIR`
  Each writes one `<prefix>_band<NN>.tif` per output band into
  `output-dir`.

### Contract

All three functions take `pan: &Raster<f64>` and `ms: &[&Raster<f64>]`
with **the MS bands already upsampled to pan resolution** (upstream
caller's responsibility — use `vector::resample` or any other
bilinear/cubic upsampler). They return `Vec<Raster<f64>>` of the
same length as `ms`. Pixels where any input is NaN propagate to
NaN in every output band; output `nodata` is set to `Some(NaN)`.

### Internal change

`classification::pca::jacobi_eigen` is now `pub(crate)`. PCA
pansharpening reuses it instead of duplicating the symmetric-matrix
eigensolver — single source of truth for the numerical kernel.
Public API of `classification::pca` is unchanged.

### Tests

- **Brovey**: pan-equal-synth preserves MS bit-for-bit, pan-doubled
  doubles every output, NaN propagation, zero-synth guard,
  empty-bands and mismatched-shape errors.

- **PCA**: pan-matching-PC1 yields near-lossless reconstruction
  (≤ 5 units mean-abs drift on a 4×4 gradient test), NaN
  propagation, empty/mismatched-shape errors.

- **Gram-Schmidt**: pan-equal-synth recovers MS at 1e-9 tolerance
  (the deepest correctness check — pins the orthogonalisation +
  inversion math), per-band mean preservation property
  (centred-pan injection has zero mean by construction), NaN
  propagation, empty/mismatched-shape errors.

15/15 pansharpening tests pass.

### Backlog status

Survey 2026-06-06 status after this release: G ✓, H ✓, I
trigger-driven, J ✓ (v0.14.4), K ✓ (v0.14.5), L ✓ (v0.14.6),
**M ✓** (this release). Remaining: N (MAD / IR-MAD), O (mosaic
seams + color balance).

## [0.14.6] - 2026-06-08

Patch release. Advances **ROADMAP item L** ("Radiometric
calibration chain") from the 2026-06-08 OTB-survey backlog. Ships
the pure-math primitives for DN → reflectance conversion of
Landsat 8/9 and Sentinel-2, plus a simplified DOS1 atmospheric
correction. Metadata parsing (`MTL.txt`, `MTD_MSIL1C.xml`) is
intentionally out of scope — the existing STAC pipeline already
exposes the calibration coefficients in item properties and can
pass them straight through.

### Added

- **`imagery::dn_to_toa_landsat`** — Landsat 8/9 Collection 2
  Level-1: `ρ_TOA = (M_p · DN + A_p) / sin(SUN_ELEVATION)`. Caller
  supplies `M_p`, `A_p`, `sun_elevation_deg` from the MTL file.
  USGS Landsat 8/9 C2 L1 Data Format Control Book reference.

- **`imagery::dn_to_surface_reflectance_landsat_c2`** — Landsat
  Collection 2 Level-2 with the fixed USGS coefficients
  (`ρ = 2.75e-5 · DN − 0.2`). `DN == 0` is treated as "fill" per
  USGS spec and emitted as NaN.

- **`imagery::dn_to_reflectance_s2`** — Sentinel-2 L1C TOA or
  L2A BOA: `ρ = (DN + offset) / quantification_value`. Default
  params (`quantification = 10000`, `offset = 0`) work for
  pre-baseline products. For PSD baseline 04.00+ (post
  2022-01-25) harmonised products pass `offset = −1000`.

- **`imagery::dos1`** — simplified per-band dark-object
  subtraction (Chavez 1988). Estimates path radiance as the
  configurable quantile of finite cells (default `0.001` — the
  0.1 % darkest pixel, robust to sensor artefacts), subtracts it,
  clamps to non-negative. Full Chavez 1988 DOS1 with `E_SUN`
  rescaling is out of scope — chain with `dn_to_toa_landsat` /
  `dn_to_reflectance_s2` for an equivalent effect.

- **CLI: `surtgis imagery calibrate`** with four subcommands:
  - `landsat-toa --mult M --add A --sun-elevation DEG`
  - `landsat-sr-c2` (no params — fixed C2 coefficients)
  - `s2 [--quantification 10000] [--offset 0]`
  - `dos1 [--quantile 0.001]`

All calibrators preserve NaN cells, set output `nodata` to
`Some(NaN)`, and use the rayon-parallel per-pixel pattern shared
across the imagery module.

### Tests

- **3 hand-computed reference values**:
  - Landsat TOA: `M=2e-5, A=-0.1, elev=60°, DN=10000 →
    ρ = 0.1 / sin(60°) ≈ 0.11547` (1e-12 tolerance)
  - Landsat C2 SR: `DN=20000 → ρ = 0.35` (1e-12 tolerance)
  - S2 TOA: `DN=2500, q=10000 → ρ = 0.25` (1e-12 tolerance)
  - S2 post-baseline: `DN=2500, q=10000, offset=-1000 → ρ = 0.15`
    (1e-12 tolerance)
- NaN passthrough on all four calibrators.
- Sun-elevation domain check, quantification > 0 check.
- DOS1 invariants: dark cell → 0, monotonicity preserved across
  the raster, NaN passthrough, all-NaN raster errors cleanly.

12/12 calibration tests pass.

### Backlog status

Survey 2026-06-06 status after this release: G ✓, H ✓, I
trigger-driven, J ✓ (v0.14.4), K ✓ (v0.14.5), **L ✓** (this
release). Remaining: M (pansharpening), N (MAD / IR-MAD), O
(mosaic seams + color balance).

## [0.14.5] - 2026-06-08

Patch release. Advances **ROADMAP item K** ("Image segmentation —
SLIC + Felzenszwalb-Huttenlocher") from the 2026-06-08 OTB-survey
backlog. Ships both algorithms with multi-band support and chains
cleanly into the existing zonal-statistics pipeline.

### Added

- **`segmentation::slic`** — new module. SLIC superpixels
  (Achanta et al. 2012). Multi-band input (`&[&Raster<f64>]`),
  per-band [0, 1] normalisation so the conventional `compactness`
  range (m ≈ 10) works regardless of input units. Each cluster
  centre searches only its `2S × 2S` window, keeping the algorithm
  linear in pixel count. Optional 4-connectivity post-pass merges
  small disconnected islands into the largest adjacent component.
  Output: dense `1..=N` labels in `Raster<i32>`; NaN pixels and
  output nodata sentinel = `0`.

- **`segmentation::felzenszwalb`** — new module. Graph-based
  segmentation (Felzenszwalb & Huttenlocher 2004) over an
  8-connected pixel graph with edge weight = Euclidean distance
  between multi-band feature vectors. Union-Find with size + Int(C)
  tracking; merge condition `w ≤ min(Int(C₁) + k/|C₁|, Int(C₂) +
  k/|C₂|)`. Optional `min_size` post-pass merges components below
  threshold into their cheapest neighbour. Output convention
  matches SLIC.

- **CLI: `surtgis segmentation slic <OUTPUT> --band <PATH>
  [--band ...] [-n N] [-m M] [--max-iter N] [--no-connectivity]`**
  — repeated `--band` flag for multi-band input.

- **CLI: `surtgis segmentation felzenszwalb <OUTPUT> --band <PATH>
  [--band ...] [-k SCALE] [--min-size N]`** — same multi-band
  convention as `slic`.

### Tests

- **`integration_tests::slic_then_zonal_statistics_recovers_per_segment_means`**
  — chains `slic` → `statistics::zonal_statistics` on a
  half-bright/half-dark raster and asserts every per-segment mean
  lands at one of the two underlying values (no boundary-straddling
  segments). Validates the label-convention contract (`1..=N`,
  zone 0 = nodata) between segmentation and downstream tools.

- **`integration_tests::felzenszwalb_then_zonal_statistics_recovers_two_zones`**
  — same chain for Felzenszwalb; asserts exactly two zones with
  per-segment means `{0, 100}`.

- **SLIC tests** (6 new): uniform-raster grid recovery, two-blob
  separation at low compactness, multi-band shape preservation,
  NaN sentinel propagation, mismatched-shape error, empty-bands
  error.

- **Felzenszwalb tests** (6 new): flat raster → single segment,
  two distinct regions → exactly two segments, `min_size`
  post-pass absorbs singletons, NaN sentinel propagation,
  multi-band sanity check, mismatched-shape error.

14/14 segmentation tests pass.

### Label-convention contract

Both segmentation algorithms emit `1..=N` labels with `0` as
nodata, matching `landscape::connected_components` and the
`statistics::zonal_statistics` contract (which treats zone `0` as
nodata). This makes the chain `segmentation → zonal_statistics →
per-segment band statistics` a one-liner.

### Backlog status

Survey 2026-06-06 status after this release: G ✓, H ✓, I
trigger-driven, J ✓ (v0.14.4), **K ✓** (this release).
Remaining: L (radiometric calibration), M (pansharpening), N
(MAD / IR-MAD), O (mosaic seams + color balance).

## [0.14.4] - 2026-06-08

Patch release. Opens **ROADMAP item J** ("Texture/imagery deepening
toward Orfeo ToolBox parity"). Adds Local Binary Patterns (Ojala
2002), a single-pass multi-feature GLCM, an analytical reference
test that pins the GLCM math to hand-computed values, and the
matching CLI surface.

### Added

- **`texture::lbp`** — new module. Local Binary Patterns (Ojala,
  Pietikäinen & Mäenpää 2002) over the 3×3 ring of each pixel
  (P=8, R=1). Two variants:
  - `LbpVariant::Standard` — raw 8-bit code (0..=255), neighbours
    ordered clockwise from the top-left. Bit i is set when
    neighbour i ≥ centre.
  - `LbpVariant::RotationInvariantUniform` — Ojala riu2. For
    uniform patterns (≤2 circular bit transitions): the popcount
    of the code (0..=8). For non-uniform patterns: P+1 = 9.
    Collapses all rotations of a uniform pattern to a single
    value, the canonical feature for rotation-invariant
    classification.
  Border pixels and pixels with any NaN in the 3×3 neighbourhood
  are written as NaN. Output `nodata` metadata is set to NaN.
  Public API: `lbp(&Raster<f64>, LbpParams) -> Result<Raster<f64>>`.

- **`texture::haralick_glcm_multi`** — new entry point. Computes
  N GLCM features in a single pass by building the co-occurrence
  matrix once per pixel window and evaluating every requested
  feature on it. The GLCM construction dominates per-pixel cost,
  so requesting all 6 features runs ~6× faster than calling
  `haralick_glcm` six times. The existing `haralick_glcm` is a
  thin wrapper around the multi-pass entry point — behaviour is
  unchanged.

- **CLI: `surtgis texture lbp <INPUT> <OUTPUT> [--variant
  standard|riu2]`** — exposes LBP from the command line.

- **CLI: `surtgis texture glcm-all <INPUT> <OUTPUT_DIR> [--radius
  N] [--levels N]`** — computes all 6 Haralick textures in one
  pass and writes them as `glcm_{energy,contrast,homogeneity,
  correlation,entropy,dissimilarity}.tif` under `OUTPUT_DIR`.

### Tests

- **GLCM analytical reference**: hand-computed normalised GLCM on
  a 3×3 binary raster (`reference_binary_3x3`) pinning all 6
  Haralick features against analytical values (closed-form
  features at 1e-12 tolerance, entropy and correlation at 1e-4).
  This is the first regression test that catches subtle math
  drift independently of cross-pixel consistency checks.

- **GLCM multi vs per-feature equivalence**: asserts the
  multi-pass output equals `haralick_glcm` cell-by-cell on a
  gradient raster at 1e-12 tolerance.

- **LBP tests** (8 new): all-brighter / all-darker neighbourhoods,
  known clockwise pattern, riu2 uniform/non-uniform / rotation
  invariance, NaN propagation, border-NaN output, raster-size
  error, and a unit test on the circular bit-transition helper.

20/20 texture tests pass. The new `multi` entry point is the
single source of truth for the GLCM math; `haralick_glcm` and
`haralick_glcm_multi` share it.

### Backlog status

Survey 2026-06-06 status after this release: G ✓, H ✓, I backlog
(3D Tiles, trigger-driven), **J in progress** (texture +
multi-feature + LBP shipped; still to land in later patches:
GLCM-vs-scikit-image cross-validation, more texture features —
LBP-TOP, GLRLM, GLSZM — as user demand surfaces).

## [0.14.3] - 2026-06-07

Patch release. Closes **ROADMAP item G** ("Martini RTIN terrain
mesher", backlog with pickup-trigger "paper figure has flat regions
wasting triangles"). After this release the original 2026-06-06
survey backlog is down to one item: I (3D Tiles export), still
backlog pending a Cesium-user trigger.

### Added

- **`relief_3d::martini`** — new module. Port of the Mapbox Martini
  algorithm (Agafonkin 2019, <https://github.com/mapbox/martini>).
  Adaptive triangulation of heightmaps by curvature: more triangles
  where the surface bends, fewer on flat regions. Three building
  blocks:

  - `Martini::new(grid_size)` — precompute the per-triangle
    coordinate table for a `(2^n + 1) × (2^n + 1)` grid. `O(N²)`.
    Reuse across many tiles of the same grid size.
  - `Martini::tile(heights)` — build the error pyramid in `O(N²)`,
    bottom-up.
  - `Tile::mesh(max_error)` — extract a triangle mesh whose
    worst-case vertical error is bounded by `max_error`. Smaller
    error → more triangles; larger → fewer.

- **`relief_3d::lod::MartiniMeshParams`** + **`QuadtreeMesh::from_dem_martini`**
  — drop-in alternative to `from_dem` that emits one Martini-meshed
  `ChunkLod` per chunk instead of stride-decimated LOD levels. Same
  skirts, vertex compression (`VertexC`), and `LodPool` lazy upload
  as the quadtree path; the render pipeline is unchanged. `lod_levels`
  is set to 1 on the returned mesh (Martini's error metric is
  already curvature-adaptive — distance-band LOD on top of it would
  be redundant).

  Default `MartiniMeshParams { chunk_cells: 64, max_error: 0.002 }`
  — same chunk size as the quadtree default, error tuned for DEMs
  normalised to `[0, 0.45]` scene units (≈ 0.5 % of the vertical
  range).

### Changed

- **`QuadtreeMesh::select()`** now clamps the chosen LOD index to
  `chunk.lod_data.len() - 1`. The quadtree path was unaffected
  because every chunk had the same `lod_levels`; the Martini path
  needs the clamp so a `select_lod` choice of band 2 or 3 doesn't
  panic when the chunk only owns LOD 0.

### Verified

- `cargo test --release -p surtgis-relief-3d`: 22/22 green (was 15
  pre-sprint; +5 Martini core + 2 integration).
- `cargo build --release -p surtgis-relief-3d` clean.
- Flat 129² DEM, 4 chunks: quadtree LOD 0 = 32 768 triangles vs
  Martini = ≤ 3 200 triangles. **>10× reduction on flat regions**
  — exactly the use case documented in the ROADMAP pickup trigger.
- Bumpy DEM at `max_error = 0`: Martini reproduces the quadtree
  LOD 0 grid within ±50 %.

### Constraints / known limits

- `chunk_cells` must be a power of two (Martini RTIN requirement).
  Default 64 is the natural fit.
- DEMs whose side isn't a multiple of `chunk_cells` have their rim
  chunks fall back to the stride-1 quadtree surface mesh. Internal
  chunks Martini-adaptive.
- Build time slower than `from_dem` — the error pyramid + tree walk
  is `O(N²)` per chunk. Acceptable for static-camera paper figures
  and headless renders; not ideal for very large DEMs where stream-
  load time dominates.

### ROADMAP status after v0.14.3

  - ~~G. Martini RTIN~~ — **done** with this release.
  - ~~H. TopoToolbox channel network depth~~ — done in v0.14.2.
  - I. 3D Tiles export — backlog, pickup trigger: a user with a
    Cesium pipeline asks for it.

## [0.14.2] - 2026-06-07

Patch release. Closes **ROADMAP item H** ("TopoToolbox-style channel
network depth") which was marked *partially done* in the 2026-06-06
survey. Two additive phases land in `surtgis-algorithms::fluvial`;
no breaking changes; existing API surface preserved.

### Added — Phase 1 (PR #22)

- **`KsnSegment.ksn_ci: (f64, f64)`** — 95 % CI on `ksn_mean` from
  a deterministic percentile bootstrap over the segment's per-cell
  ksn values.
- **`KsnParams.bootstrap_n: usize`** (default 200) and
  **`KsnParams.seed: u64`** (default 42). Mirrors the
  `concavity_index` recipe — `hash(seed, segment_id, boot_k, cell_k)`
  generates resampling indices. `bootstrap_n = 0` disables the CI;
  it then equals `(ksn_mean, ksn_mean)`, preserving legacy behaviour.
- **`fluvial::long_profile`** — new module. Walks each outlet
  upstream along the largest-area path at every confluence, emits
  per-node `(coord, distance_from_outlet_m, elevation, area_m2,
  chi, ksn)`. χ and ksn are passed in as **optional rasters** —
  caller computes them with whatever θ_ref / smoothing-window choices
  they want; the extractor stays agnostic.
  
  New types: `LongProfile`, `LongProfileNode`, `LongProfileParams`,
  `LongProfileError`. Re-exported at `fluvial::`.

### Added — Phase 2 (PR #23)

- **`fluvial::swath_profile`** — new module. Standard TopoToolbox
  swath: densify a polyline baseline to `step_along_m` spacing,
  cast perpendiculars at every vertex, sample the raster within
  `±half_width_m` at `step_cross_m` steps, emit per-bin
  `min / max / mean / median / p25 / p75 / n_samples`.

  Baseline is caller-supplied — works equally for `LongProfile`
  nodes (`.iter().map(|n| n.coord).collect()`), fault traces, hand-
  picked transects, etc.

  New types: `SwathProfile`, `SwathStats`, `SwathParams`,
  `SwathError`. Re-exported at `fluvial::`.

- **`fluvial::export`** — new module. CSV / JSON writers for the
  four data products:

  | Product | CSV | JSON |
  |---------|-----|------|
  | `LongProfile` | row per node, `profile_id` groups outlets | hierarchical, nested `nodes` array |
  | `SwathProfile` | row per bin | row array |
  | `KsnResult` segments | row per segment + `geometry_wkt` LINESTRING | row array with nested coords |
  | χ raster | row per finite cell | (use the CSV) |

  The core types stay free of `serde` derives — wrapper structs in
  `export.rs` own the serialisation. Adds `serde_json` and `csv`
  to `surtgis-algorithms` dependencies.

  Free functions: `write_long_profiles_csv`, `write_long_profiles_json`,
  `write_swath_csv`, `write_swath_json`, `write_ksn_segments_csv`,
  `write_ksn_segments_json`, `write_chi_csv`. All take any
  `impl Write` so callers can stream to files, buffers, gz wrappers,
  whatever.

### Verified

- `cargo test --release -p surtgis-algorithms fluvial`: 42/42 green
  (was 34 pre-sprint; +8 new tests across Phase 1 and Phase 2).
- `cargo build --release --workspace`: clean.

### Deferred and rule-outs

The 2026-06-06 ROADMAP survey listed several follow-up items under
H. Status now:

  - **Bootstrap CI on ksn** — done (Phase 1).
  - **Long-profile extractor** — done (Phase 1).
  - **Swath profiles** — done (Phase 2).
  - **CSV/JSON export** — done (Phase 2).
  - **Knickpoint detection** — already shipped pre-sprint
    (Neely et al. 2017 KZP variant).
  - **Divide migration** — already shipped pre-sprint
    (Whipple/Forte/Willett 2017).
  - **Per-tributary profile walker convenience helper** —
    deferred. Callers can iterate confluences and call
    `long_profile` themselves; the convenience wrapper isn't a
    blocker.
  - **In-Rust figure generators** — **out of scope**. Users will
    re-plot the data in matplotlib / ggplot2 / R for paper-grade
    figures anyway; the CSV / JSON export is what they actually
    need.

After this release ROADMAP item H moves from *partially done* to
**done**. See `ROADMAP.md` for the updated entry.

## [0.14.1] - 2026-06-06

Patch release. Closes **M4** of `SPEC_SURTGIS_RELIEF_P4.md` — the
CLI `relief-3d` headless command gains a `--lod` flag that routes
through the quadtree + `LodPool` pipeline established by M1–M3.
Default path unchanged; opt-in for big DEMs.

After this release P4 is complete: M0 (spec) + M1 (native spike) +
M2 (skirts) + M3 (WASM browser) + M4 (CLI headless).

### Added

- **`surtgis-relief-3d::headless::render_to_rgba_lod(dem, texture, cfg)`**.
  Public entry mirroring `render_to_rgba`. Builds a `QuadtreeMesh`
  from the DEM (using `LodParams::default`) and renders via a 768 MB
  GPU pool (384 MB vertex + 384 MB index). Browser still uses 192 MB
  because the WebGL2 single-buffer cap binds there; headless has no
  such cap so we size aggressively to handle 10 K DEMs without
  overflow.

- **`surtgis relief-3d --lod`**. New CLI flag, default `false`. Help
  text points users at the use case: DEMs ≥ 3 K side where the
  default single-buffer upload would blow the adapter's max buffer
  size.

  ```bash
  # 10 K × 10 K DEM (100 M cells), default path is infeasible
  surtgis relief-3d big_dem.tif out.png \
      --width 1280 --height 720 --colormap imhof1 --lod
  ```

### Verified

- 1 K DEM (1000×1000): non-LOD 0.35 s, LOD 0.67 s. The LOD path is
  slower on small DEMs because of the quadtree build + per-chunk
  batching overhead, which is the expected trade-off. The output
  is visually identical.
- 10 K DEM (10000×10000, 100 M cells): non-LOD path infeasible
  (~3.2 GB single vertex buffer would blow every consumer GPU's
  `max_buffer_size`). LOD path produces a 1280×720 PNG in 30 s —
  most of which is the 2D `sphere_shade` recipe on 100 M cells in
  a single thread, not the LOD render itself.

### Deferred to a later patch

- CLI knobs for `chunk_cells` / `distance_bands`. 20 K+ DEMs may
  hit the 768 MB pool ceiling and need tighter bands or smaller
  chunks; if anyone trips it in practice we add the flags then.
- Auto-sized pool based on `mesh.cpu_bytes()` worst case.
- Screen-space-error LOD metric (polish; the distance-band metric
  is well-tuned in practice).

## [0.14.0] - 2026-06-06

Minor release. **Quadtree LOD lands in `surtgis-relief-3d`**, closing
the last named rayshader-gap that had measurable user impact:
production-scale DEMs (4 K × 4 K = 16 M cells) now render fluidly on
native ≥60 FPS, and load + render in WebGL2 browsers without
exceeding the 256 MB single-buffer cap. After this release the only
honest "rayshader wins" line is path-tracing (out of scope, will
stay out — use `rayrender` for that figure).

Implements M0–M3 of `SPEC_SURTGIS_RELIEF_P4.md`. M4 (CLI headless
path with LOD) is deferred to v0.14.1.

### Added

- **`surtgis_relief_3d::lod`** — quadtree LOD pipeline:
  - `QuadtreeMesh::from_dem(dem, vertical_exaggeration, params)`
    subdivides a DEM into `chunk_cells × chunk_cells` chunks and
    builds per-chunk per-LOD self-contained vertex + index buffers.
    LOD k uses stride `2^k`.
  - `LodParams { chunk_cells: 64, lod_levels: 4,
    distance_bands: [0.6, 1.8, 5.0] }` (tuned on a synthetic 4 K
    spike; a screen-space-error metric would be a polish follow-up).
  - `Aabb` + Gribb–Hartmann frustum extraction + 8-corners-vs-6-planes
    culling.
  - `QuadtreeMesh::batch_visible(view_proj, camera_pos, params, frame)`
    culls, selects a LOD per chunk, and fills a `FrameUpload` (vertex
    + index scratch vecs + draw commands). Cache hit when the visible
    set is identical to the previous frame; the upload is skipped
    entirely.
  - Skirts at every LOD edge — vertical strips dropping below the
    chunk's lowest vertex by `1.5 × chunk_y_span` so cracks at
    transitions between different LOD levels stay hidden.

- **`crate::VertexC`** — 16-byte compressed vertex (snorm16x4 position,
  unorm16x2 UV, snorm8x4 normal). Halves GPU memory for the same mesh;
  `pipeline::build_pipeline` converts `Vertex` → `VertexC` once at
  upload time. The WGSL shader takes the snorm/unorm formats as
  decoded vec4<f32> and extracts `.xyz` from the packed inputs.

- **`LodPool`** — fixed-size ring buffers (96 MB vertex + 96 MB index)
  for lazy GPU upload. Per frame the pool's vertex + index buffers are
  refreshed via a single `write_buffer` call each, the visible-chunks'
  data pre-batched into scratch vecs by `batch_visible`. The GPU side
  never exceeds 192 MB regardless of DEM size; the full mesh
  (~990 MB for a 4 K DEM) lives in the WASM heap, which has a 4 GB
  budget.

- **Native viewer paths**:
  - `native::run_lod_viewer(mesh, params, texture, label)` and
    `native::run_lod_viewer_with_mode(..., mode)` accept a
    `QuadtreeMesh`. Mouse controls + damping + screenshot + help key
    work unchanged.
  - `crates/relief-3d/examples/spike_lod_4k.rs` generates a 4096²
    procedural DEM (16.78 M cells) and renders it via the LOD path.
    The §12.6 lesson at the spec level: the M1 acceptance must be
    measured at production-workload size, not `dem_filled.tif` which
    already passes at 60 FPS without any LOD work.

- **Browser viewer paths**:
  - `#[wasm_bindgen] fn run_relief3d_synthetic_lod_canvas(canvas_id,
    side, colormap, sun_az, sun_alt, shadows, ambient,
    vertical_exaggeration)` generates a procedural DEM in-WASM and
    runs the LOD pipeline. Avoids the GeoTIFF round-trip from JS for
    big synthetic test meshes.
  - `surtgis-demo/relief3d-wgpu.html` gains "Synthetic 2K + LOD" and
    "Synthetic 4K + LOD" buttons. The 4 K button forces
    `shadows = false, ambient = false` because those 2D recipes block
    the single-threaded WASM event loop for several minutes on 16 M
    cells; the 4 K path exists to validate the 3D rendering pipeline
    + memory budget, not the 2D recipe quality. The 2 K button keeps
    the full recipe.

### Acceptance

  - **Native** (M1+M2 bar): the 4 K spike sustains 58–60 FPS across a
    full orbit cycle, all 4096 chunks visible. The cache hit rate is
    ~95 %; rebuild frames stay sub-10 ms.
  - **Browser** (M3 bar): a 4 K DEM loads + renders in Chrome (ANGLE +
    WebGL2) under the 256 MB single-buffer cap. Frame rate scales
    with the visible-chunk count; on default camera the GPU pool
    consumption hovers around ~30 MB.

### Changed

- Workspace bump 0.13.1 → 0.14.0; inter-crate deps swept from `"0.13.1"`
  to `"0.14"` across `surtgis-core`, `surtgis-algorithms`,
  `surtgis-parallel`, `surtgis-cloud`, `surtgis-colormap`,
  `surtgis-relief`, `surtgis-relief-3d`.
- `pipeline::build_pipeline` now compresses incoming `Vec<Vertex>` to
  `Vec<VertexC>` before upload. Public API surface unchanged — the
  conversion is internal.
- `crates/relief-3d/shaders/relief.wgsl` vertex stage inputs change
  from `vec3<f32>` / `vec2<f32>` / `vec3<f32>` to `vec4<f32>` /
  `vec2<f32>` / `vec4<f32>` (snorm16x4 / unorm16x2 / snorm8x4 from
  Rust), with `.xyz` extraction on the packed inputs. Output and the
  fragment stage unchanged.

### Fixed

- **WebGL2 `drawElementsInstancedBaseVertex` panic**. The LOD path
  initially issued `pass.draw_indexed(range, base_vertex, ...)` with
  per-chunk non-zero `base_vertex`. WebGL2 doesn't expose
  `drawElementsInstancedBaseVertexBaseInstance`, so the GL backend
  panics. Fix: in `QuadtreeMesh::batch_visible`, rebase chunk-local
  indices to pool-absolute by adding the chunk's vertex offset at
  copy time, then call `draw_indexed` with `base_vertex = 0` always.
  Native (Vulkan/Metal/DX12) supports the base-vertex draw natively;
  the rebase costs ~few-ns per index on rebuild frames and is
  free on cache hits.

### Deferred to v0.14.1

- **M4 — CLI headless path** (`surtgis relief-3d --lod`). Same wgpu
  pipeline rendered offscreen via the existing headless path; needs
  the LOD pool sized aggressively for single-frame renders. ~2 days.
- **Screen-space-error LOD metric** instead of camera-distance bands.
  Polish.

## [0.13.1] - 2026-06-05

Patch release with a single, focused perf improvement: **5.66× faster
sky-view factor**, which propagates to ~5× faster `ambient_shade` in
the relief composite pipeline. Bit-equivalent to within f64 rounding
(max delta 2.9e-15 over 363090 cells on `dem_filled.tif`); no
behavioural change for downstream callers.

### Added

- `surtgis_algorithms::terrain::sky_view_factor_fast(dem, params)` — a
  hot-loop-optimised SVF that keeps the same `SvfParams`, output
  format, and NaN semantics as `sky_view_factor`. Three optimisations:
  - **Pre-computed integer step offsets** per direction. The Bresenham-
    style `.round()` that the reference implementation runs every
    iteration is hoisted out once per direction × step; the inner
    march becomes pure integer index arithmetic, which lets the
    autovectorisator keep up.
  - **`inv_dist[k]` table** for `1 / (k * cell_size)` — replaces a
    division in the hot loop with one multiplication, shared across
    every cell × direction.
  - **`sin²(atan(t)) = t² / (1 + t²)`** algebraic identity at the
    direction summary — skips one `atan` and one `sin` per
    direction per cell (5.8 M trig calls saved on `dem_filled.tif`
    with 16 directions).

  Four new unit tests cover endpoint behaviour, pit-vs-open
  ordering, bit-equivalence against the reference SVF on a synthetic
  bumpy DEM, and zero-radius / zero-directions rejection.

- `crates/algorithms/examples/bench_svf.rs` — head-to-head benchmark
  used to validate the ≥3× acceptance bar before committing the swap.

### Changed

- `surtgis_relief::ambient_shade` now calls `sky_view_factor_fast`.
  No public API change. Pipeline benchmark
  (`crates/relief/examples/render_relief.rs` on `dem_filled.tif`,
  radius=30):
  - `ambient_shade`: 6.36 s → 1.12 s (**5.7×**)
  - relief composite total: 6.6 s → 1.35 s (**4.9×**)

## [0.13.0] - 2026-06-05

Polish release bringing `surtgis-relief` / `surtgis-relief-3d` to a
"rayshader peer" on every axis except path-tracing and LOD adaptativo.
Closes M1, M2, M3 of `SPEC_SURTGIS_RELIEF_P3.md`; M4 (R wrapper) was
skipped mid-sprint with a documented post-mortem in spec §6.

This release jumps from 0.11.0 over 0.12.0. P2 (the `surtgis-relief-3d`
crate — native wgpu viewer + WebGL2 browser viewer + headless CLI
screenshot) shipped on `main` between the two but was never tagged.
Both P2 and P3 changes are included below.

### Added — `surtgis-relief-3d` (P2, retroactively v0.12.0)

- New workspace crate `crates/relief-3d` exposing a single wgpu
  pipeline (mesh + textured fragment + Lambertian per-fragment light)
  reachable from three frontends with one cfg-split layer:
  - **Native** — winit window via `run_viewer(vertices, indices,
    rgba_pixels, width, height, label)`. Mouse-drag rotate,
    right-drag pan, wheel zoom. Keyboard controls for sun azimuth /
    altitude, vertical exaggeration, and ambient term.
  - **Browser** — `run_relief3d_canvas(canvas_id, tiff_bytes,
    colormap, sun_az, sun_alt, shadows, ambient, vertical_exag)`
    `#[wasm_bindgen]` entry point. WebGL2 only (WebGPU is too
    unreliable on Firefox today). 2.6 MB WASM bundle. Demo page at
    `surtgis-demo/relief3d-wgpu.html`. Returns a `ReliefHandle` so
    JS sliders can retune the sun and exaggeration without
    re-running the 2D composite.
  - **Headless** — `surtgis relief-3d DEM.tif --output PNG` (CLI
    feature `relief-3d`) renders a 1920×1080 PNG without opening a
    window. Same wgpu pipeline; the surface target is a
    `wgpu::Texture` instead of a `Surface`. Output is copied through
    a row-aligned staging buffer and channel-swapped if the format
    requires it.

- Mesh primitives in `crates/relief-3d/src/mesh.rs`:
  - `from_dem(dem, vertical_exaggeration)` — grid mesh with normals
    computed from neighbour heights via central differences (one-
    sided at borders). Up-facing `(-dh/dx, 1, -dh/dz)` convention.
    NaN cells map to flat (zero height + up-pointing normal).
  - `grid_mesh(rows, cols, extent, height_fn)` + the M1-spike
    `cosine_test_heights` so the 1 M-vertex perf validation can run
    without a real DEM.

- Orbit camera with mouse-driven control in
  `crates/relief-3d/src/camera.rs`.

- Acceptance: 1024×1024 grid (1,048,576 vertices) renders at vsync-
  capped 60 FPS in the native spike. `dem_filled.tif` (570×637,
  363090 vertices) renders at 60 FPS sustained through the full
  pipeline (sphere shade + ray shade + ambient + texture upload +
  mesh upload), confirming the architecture scales to production
  workloads (spec §M1 explicit application of §12.6 lesson from
  the 2D spec).

### Added — P3 polish (this release)

- **Imhof-style palettes.** Eight new `ColorScheme` variants in
  `surtgis-colormap::scheme`: `Imhof1` (greens → straw → ochre →
  snow), `Imhof2` (alpine cool), `Imhof3` (desert-leaning), `Imhof4`
  (twilight purples), `Bw1` (smooth grayscale), `Bw2` (high-contrast
  grayscale with sharp mid-ridge), `DesertDry`, `Pastel`. All
  curated against `dem_filled.tif`; not bit-equivalent to rayshader's
  imhof1..4 (rayshader does not publish source values). `ALL` grows
  from 8 to 16; CLI parsers and WASM string parsers accept the new
  names (`imhof1`..`imhof4`, `bw1`, `bw2`, `desert`, `pastel`).

- **Atmospheric haze in the 3D shader.** Vertex shader passes a
  `linear_depth` varying (clip-space `w` under standard
  perspective). Fragment shader applies
  `mix(shaded, fog_color, density * smoothstep(near, far, depth))`.
  `Uniforms` grew to 144 B (added `fog_color: vec4` and
  `fog_range: vec4`). Density defaults to 0 so the output is bit-
  equivalent to pre-P3 unless turned on. New CLI flag
  `--haze <0..1>`. Native viewer keys: `H` toggles, `F`/`G`
  fine-tune. WASM gains `ReliefHandle.set_haze(density)`.

- **Water depth shading.**
  `surtgis-relief::water_depth(mask: &Raster<u8>) -> Raster<f32>` —
  8-connected Chebyshev distance transform via multi-source BFS,
  O(N), with the implicit off-grid border seeded as land so lakes
  touching the raster edge read depth 1 at the edge instead of
  measuring against the opposite shore. New
  `ReliefBuilder.add_water_depth(mask, scheme)` samples the scheme
  at `t = depth / max_depth`. Shore → light end (t ≈ 0); centre →
  dark end (t = 1). With `ColorScheme::Water` (white → cyan →
  navy), a synthetic 200 × 200 lake renders centre `(8, 48, 107)`,
  shore `(204, 235, 252)` — continuous gradient. The CLI
  `surtgis relief --water` now uses depth shading by default; the
  old binary `add_water` stays in the public API.

- **Native viewer UX polish.**
  - Camera damping with `τ = 80 ms` exponential smoothing. Mouse
    handlers still write to `camera`; `camera_smooth` lerps each
    frame with `factor = 1 - exp(-dt/τ)`. Azimuth wraps through the
    short arc so 5°→355° does not animate the long way around.
  - `?` (slash) prints a formatted keybindings table to stderr.
    Visual on-window overlay deferred behind a font dep.
  - `S` saves a timestamped PNG to cwd
    (`relief3d-<unix_secs>.png`). Surface usage gains COPY_SRC; the
    just-rendered frame is copied via `copy_texture_to_buffer` into
    a staging buffer before submit/present, then mapped, row-
    padding-stripped, and channel-swapped if the surface format is
    BGRA before going through `surtgis_colormap::rgba_to_png_bytes`.

### Skipped — P3 deliberate omissions

- **M3.4 GLB export.** `gltf` crate is ~50 KLOC of dep weight for a
  feature only a handful of users would touch. Deferred.
- **M4 R wrapper (`surtgisr`).** Skipped mid-sprint, see
  `SPEC_SURTGIS_RELIEF_P3.md` §6 for the ROI post-mortem. Short
  version: R users we'd capture already have rayshader working,
  and where we differentiate (browser, headless CI, Python) does
  not overlap with R.
- **Path-tracing (P3 non-goal).** Stays out. Use rayshader for
  photorealistic figures.
- **LOD adaptativo for DEMs ≥ 2K side.** P4 sprint candidate, not
  P3.

### Changed

- Workspace version bumped 0.11.0 → 0.13.0 (jumps 0.12.0).
- Inter-crate dependency declarations swept from
  `version = "0.11"` to `version = "0.13"` across `surtgis-core`,
  `surtgis-algorithms`, `surtgis-parallel`, `surtgis-cloud`,
  `surtgis-colormap`, `surtgis-relief`, `surtgis-relief-3d`.
- `surtgis-cli` gains a new `relief-3d` feature gate. `surtgis`
  CLI binary built without that feature lacks the `relief-3d`
  subcommand but is otherwise unchanged.
- `crates/relief-3d/src/shadow_ray.rs` and friends gain a cfg-split
  `map_rows()` helper so the WASM target builds without rayon
  (mirrors the `maybe_rayon` pattern from `surtgis-algorithms`).
- `crates/colormap/src/scheme.rs` depth-changed
  `wgpu::TextureFormat::Depth32Float` → `Depth24Plus` in the 3D
  pipeline to match the WebGL2 baseline.

## [0.11.0] - 2026-06-04

Minor release introducing **`surtgis-relief`** — a rayshader-style 2D
shaded-relief composite layer on top of the existing terrain primitives.
Single binary on the CLI, browser-runnable via WASM (categorically
unique vs rayshader, which is desktop R only), and importable from
Python as a numpy `(H, W, 4) uint8` array.

This release closes M1–M5 of the `SPEC_SURTGIS_RELIEF.md` handoff.
3D (P2) is intentionally out of scope and will land in a later minor.
A working Three.js + WASM preview lives in `surtgis-demo/relief3d.html`
to validate that the 2D textures hold up under 3D mesh display.

### Added

- **`surtgis-relief`** workspace crate. Public API:
  - `ray_shade(dem, &RayShadeParams)` — ray-traced cast shadows.
    `RayShadeParams::with_soft_shadow_altitude(azimuth_deg,
    low_alt_deg, high_alt_deg, n_samples)` matches rayshader's
    `anglebreaks = seq(low, high, by=1)` recipe. When every sun sample
    shares an azimuth, the implementation routes through an amortised
    fast path; with differing azimuths it falls back to per-sun
    ray-marches transparently.
  - `sphere_shade(dem, HillshadeParams)` — normal-based intensity
    layer, thin wrapper over `hillshade` forcing `normalized = true`.
  - `ambient_shade(dem, radius)` — sky-view-factor wrapper for ambient
    occlusion in `[0, 1]`.
  - `detect_water(dem, &WaterParams)` — heuristic water mask via
    flat-area connected components (8-neighbour flatness test, 4-CC
    union-find, min-area filter), returning `Raster<u8>`.
  - `ReliefBuilder` fluent compositor — `.base_colormap(scheme)` +
    `.add_shade()` / `.add_shadow()` / `.add_ambient()` (multiply
    blend) + `.add_water()` (alpha-over with scheme-sampled colour) +
    `.add_rgba_over()` (alpha-over arbitrary RGBA) → `.render()` →
    `RgbaImage`.

- **Two new ray-march primitives** in `surtgis-relief::shadow_ray`:
  - `cast_shadow_ray_mask` — per-cell early-exit ray-march, binary
    lit/shadow mask. Incremental position state (no per-step
    multiplications) + `unsafe get_unchecked`. ~3× faster per call
    than `horizon_angle_map`.
  - `horizon_tan_map` — full-radius march tracking
    `max_k (z(k) - z0) / dist(k)`. With pre-computed `inv_dist[k]` so
    the inner-loop tan is one multiply (no division). Shared-azimuth
    amortisation primitive: one call serves every altitude; each
    altitude is then an O(N) thresholding.

- **CLI `surtgis relief DEM.tif OUT.png`** — full composite to PNG.
  Flags: `--colormap`, `--sun-azimuth`, `--sun-altitude`, `--shadows`,
  `--soft N`, `--ambient`, `--water`, `--z-factor`, `--radius`.

- **WASM binding `relief_compute`** in `surtgis-wasm` — returns a raw
  RGBA `Vec<u8>` for direct canvas/WebGL upload.

- **Python binding `surtgis.relief_compute`** — returns an
  `(H, W, 4) uint8` numpy array.

- `crates/relief/examples/render_relief.rs` — end-to-end example.
- `crates/relief/examples/bench_vs_rayshader.rs` — M2 acceptance
  benchmark mirroring `benchmarks/rayshader_baseline.R`.
- `benchmarks/rayshader_baseline.R` — 5-rep + warmup R baseline on
  `dem_filled.tif`; results in `benchmarks/results/rayshader_baseline.csv`.
- `surtgis-demo/relief.html` and `surtgis-demo/relief3d.html` — 2D and
  3D in-browser demos.

### Performance

On `dem_filled.tif` (637×570, Andes, EPSG:32719) with the rayshader
anglebreaks recipe (azimuth 315°, 11 altitudes 40°–50°, radius 850):

| Component | Before amortisation | After amortisation |
|---|---|---|
| `ray_shade` median | 1.98 s | **0.19 s** (10.4×) |
| `sphere_shade` median | 0.06 s | 0.06 s |
| **TOTAL median** | 2.04 s | **0.26 s** |
| **vs rayshader 1.80 s** | 0.88× FAIL | **6.99× PASS** |

The amortisation insight (single-azimuth `horizon_tan_map` + N cheap
thresholdings) is what makes the WASM and Python `relief_compute`
calls interactive on real DEMs.

### Spec deltas

`SPEC_SURTGIS_RELIEF.md` claimed "no new terrain math". After
implementation that was wrong — three new primitives shipped
(`cast_shadow_ray_mask`, `horizon_tan_map`, `detect_water`). The
spec's §11 was preserved verbatim with a "this paragraph was wrong"
prefix; new §12 Reality check documents the perf trap with
`horizon_angle_map`, the outstanding `ambient_shade` follow-up
(~6.4 s, SVF-dominated), and the meta-lesson for the next handoff
spec: measure the production-workload configuration in the spike,
not the cheapest one.

### Changed

- Workspace version bump 0.10.4 → 0.11.0.
- Inter-crate dependency declarations swept from `version = "0.10"` to
  `version = "0.11"` across `surtgis-core`, `surtgis-algorithms`,
  `surtgis-parallel`, `surtgis-cloud`, `surtgis-colormap`,
  `surtgis-relief`.
- `surtgis-cli` and the `surtgis-wasm` / `surtgis-python` frontends
  pick up direct dependencies on `surtgis-relief`.
- `crates/relief/src/shadow_ray.rs` cfg-gates the `rayon` import on
  `cfg(not(target_arch = "wasm32"))` and routes through an internal
  `map_rows()` helper that is parallel on native and sequential on
  WASM. Mirrors the `maybe_rayon` pattern from `surtgis-algorithms`.

## [0.10.4] - 2026-06-02

Patch release that adds in-tree PNG output. Pre-M1 for the
`surtgis-relief` crate (see `SPEC_SURTGIS_RELIEF.md`); doing this in
`surtgis-colormap` rather than `surtgis-relief` so every crate that
produces an RGBA buffer (curvature previews, hypsometric maps, fluvial
GeoJSON-paired rasters, future relief composites) can save PNG without
pulling its own dependency.

### Added

- `RgbaImage` struct in `surtgis_colormap::encode` with the row-major
  RGBA pixel layout that `raster_to_rgba` already produces. Methods:
  `from_rgba`, `from_intensity` (single-channel `[0, 1]` raster →
  greyscale), `over` (alpha-over composite), `multiply` (multiply blend
  for shadows on a colored base).
- `RgbaImage::to_png_bytes()` and `RgbaImage::save_png(path)` for
  native PNG output. Standalone helpers `rgba_to_png_bytes(width,
  height, &[u8])` and `save_png(path, …)` are also re-exported at the
  crate root.
- `EncodeError` typed via `thiserror`; includes shape-mismatch,
  `image::ImageError`, and `std::io::Error` variants.

### Notes

- All PNG paths are gated on `cfg(not(target_arch = "wasm32"))`. The
  `image` crate is added as a workspace-target dependency under the
  same gate, so WASM builds stay lightweight. On WASM, consumers are
  expected to pass the raw RGBA buffer back to JS, where the canvas or
  Blob path encodes the image.
- No algorithm changes elsewhere; `surtgis-core`, `surtgis-algorithms`,
  `surtgis-parallel`, `surtgis-cloud`, `surtgis` (CLI) ship unchanged
  but bump in lockstep with the workspace version.

## [0.10.3] - 2026-05-26

Patch release that adds the benchmark suite for the `extract-patches`
subcommand. Companion-paper submission to *Computers & Geosciences*
("Compute-efficient preparation of training data for geospatial
foundation models in native Rust") cites this version as the
reproducibility anchor for §6 of the manuscript.

### Added

- `benchmarks/measure_memory_gfm_prep.sh` — peak resident-set
  measurement via `/usr/bin/time -v`. Writes
  `benchmarks/results/gfm_prep/memory.csv`.
- `benchmarks/verify_outputs_gfm_prep.sh` — SHA-256 hashes plus
  element-wise diff statistics on the single-timestamp comparable
  workload. Writes `output_verification.txt`.
- `benchmarks/run_gfm_prep_sweeps.sh` — scaling sweeps over point
  count, patch size, and timestamp count. Forces `LC_ALL=C` for
  locale-independent printf. Writes
  `benchmarks/results/gfm_prep/scaling.csv`.
- `benchmarks/run_gfm_prep_3way.sh` — 3-way comparison harness
  (SurtGIS vs naive Python vs InstaGeo-style xarray+rioxarray
  proxy). Writes `three_way.csv`.
- `benchmarks/bench_gfm_prep_instageo_style.py` — InstaGeo-style
  chip-extraction proxy that reproduces the
  `RasterDataPipeline.process_tile` inner loop on local data
  without installing the CC-BY-NC-SA-licensed `instageo` package.
- `benchmarks/measure_stac_latency.py` — one-off STAC query
  latency probe against Element 84 Earth Search v1.

### Notes

- All five new bench scripts and their committed result files
  (memory.csv, scaling.csv, three_way.csv, output_verification.txt)
  are the reproducibility anchors cited in the paper-gfm-prep
  submission. The paper's repository
  ([github.com/franciscoparrao/paper-gfm-prep](https://github.com/franciscoparrao/paper-gfm-prep))
  references these files by path; the v0.10.3 tag ensures the
  cited paths resolve.
- No algorithm changes vs v0.10.2. Same library functionality;
  only the benchmark suite expands.

## [0.10.2] - 2026-05-24

Patch release that fixes a silent-corruption bug in
`surtgis hydrology fill-sinks` when the input DEM contains NaN
cells that form a barrier between the physical raster border and a
valid interior region. This is the typical post-reprojection pattern:
WGS84 → UTM leaves the rotated corners as NaN, and earlier versions
of `fill-sinks` could not find drainage paths around the NaN
curtains, leaving interior valid cells stuck at `f64::MAX / 2.0 ≈
8.99e307`. Every downstream stage (flow-direction, flow-accumulation,
χ, ksn, knickpoints) then operated on garbage. The defensive
workaround — `surtgis clip --bbox` to a NaN-free interior — is no
longer required.

Surfaced during the Sprint 7 Smugglers Notch validation against
Perron & Royden (2013); the validation example now runs without the
clip step and produces R² = 0.82 on the main catchment as before.

### Fixed

- **`surtgis hydrology fill-sinks`** now treats NaN / nodata cells as
  drainage exits in the Planchon-Darboux init, matching the GIS
  convention (and the `priority_flood` implementation in the same
  crate). Concretely: interior cells 8-adjacent to a NaN/nodata
  neighbour are initialised to their DEM value, so they can act as
  drains for cells further in. NaN cells themselves are preserved in
  the output (no `inf` propagation, no `big_value` leak).
- **`fill-sinks` output nodata metadata** is now uniformly
  `Some(f64::NAN)` regardless of input sentinel. Explicit sentinel
  values (e.g. `-9999.0`) in the input are converted to NaN in the
  output for consistency with `priority_flood`.

### Added

- Three new unit tests for `fill-sinks` covering: interior NaN
  preservation with adjacent sink, NaN curtain at all four corners
  (synthetic analogue of reprojected UTM), and explicit
  sentinel-nodata-to-NaN conversion.
- Debug assertion that catches `big_value` leaks before they reach
  the output raster; no cost in release builds.

### Changed

- **`examples/smugglers_notch_validation/run_validation.sh`** drops
  the defensive `clip` step (steps renumbered 1–7 instead of 1–8).
  README updated to reflect the v0.10.2 fix.

## [0.10.1] - 2026-05-24

Patch release that fixes a real bug surfaced by the first external use
of the v0.10.0 fluvial module (Frente Puerto Aysén / Pangal AOI
analysis): the three vector outputs of `surtgis fluvial *` were
writing coordinates in the source projected CRS but omitting any CRS
declaration. Per RFC 7946 the absence of a `crs` member means WGS84,
so every standards-compliant client (geopandas, MapLibre, deck.gl,
QGIS via modern OGR) interpreted UTM metres as lat/lon and produced
garbage values.

### Fixed

- **`surtgis fluvial ksn --segments`**, **`fluvial knickpoints`**,
  **`fluvial divide-migration`** now reproject coordinates to WGS84
  (EPSG:4326) before serialisation by default. Output is RFC 7946
  strict — no `crs` member, coordinates interpreted as lat/lon. Works
  unmodified with `geopandas.read_file()`, MapLibre, deck.gl, and
  modern QGIS workflows.

### Added

- **`--keep-crs` flag** on the three vector-output subcommands
  preserves the source raster's CRS in the GeoJSON output and declares
  it via a legacy GeoJSON 2008 `crs` member naming the EPSG. Use this
  when you need submetre precision preserved (scientific analysis,
  cross-tool comparison against QGIS / R sf / OGR-modern). Non-strict
  RFC 7946 but understood by all real GIS tooling.

### Implementation notes

Shared helpers `to_wgs84`, `project_coord`, `crs_member`,
`feature_collection_json`, `raster_epsg` at the top of
`crates/cli/src/handlers/fluvial.rs` (mirrors the pattern from
`stac_writer.rs` shipped in v0.9.0). Same proj4rs path under the
`projections` feature flag; fallback when projection fails preserves
source coordinates rather than failing the command.

Validated end-to-end on `fbm_1000_raw.tif` (EPSG:32719): same 3037
knickpoints in both modes; default emits valid Patagonia lon/lat,
`--keep-crs` emits UTM metres with explicit `urn:ogc:def:crs:EPSG::32719`.
geopandas roundtrip confirms `crs=EPSG:4326` with sensible bbox.

## [0.10.0] - 2026-05-23

Headline: closes the **fluvial-tectonic morphometry spec** —
`docs/SPEC_morfometria_fluvial_tectonica.md` — that Dr. Paulo Quezada
(LAMIR/UFPR + UDD) contributed for the Frente Puerto Aysén project.
Adds the five canonical metrics of tectonic geomorphology as a new
`crates/algorithms/src/fluvial/` submodule, all reachable from a single
binary as `surtgis fluvial <sub>`.

### Added — `crates/algorithms/src/fluvial/`

- **`chi-transform`** (Perron & Royden 2013): base-level reference
  distance χ, the path integral of `(A₀/A(x))^θref`. BFS upstream from
  every outlet on a `StreamGraph` built from the binary stream raster +
  D8 flow_dir. Riemann-sum convention matching TopoToolbox 2 for
  bit-for-bit parity.
- **`channel-steepness` (`ksn`)** (Wobus 2006): channel-following slope
  `S = (z_up − z_down)/dx_along_channel`, raw `ksn = S · A^θref`,
  smoothed over a moving window (default 500 m) along the network with
  main-stem traversal at confluences. Optional `--segments` flag emits a
  LineString GeoJSON of per-segment averages.
- **`knickpoints`** (Neely 2017): per-segment χ–z profile + 1-D TVD
  denoising (inline Condat 2013, ~80 LOC; the `tvr` crate that the spec
  recommended doesn't exist on crates.io) + non-uniform 3-point
  curvature stencil + magnitude / polarity classification (concave =
  decreasing slope downstream → likely lithology; convex = increasing
  slope downstream → likely transient/tectonic). Confluence buffer to
  suppress edge artefacts. GeoJSON Points + optional categorical raster.
- **`concavity`** (Perron & Royden 2013): per-basin θ estimation via
  grid search minimising elevation~χ scatter, with deterministic seeded
  bootstrap (default n=200) for 95 % CI. CSV output with `theta_opt`,
  `theta_ci_low`, `theta_ci_high`, `n_cells`, `rmse`.
- **`divide-migration`** (Willett 2014, Whipple 2017): scan 4-connected
  basin boundaries, group by sorted basin-id pair, compute median Δχ +
  Gilbert Δelev + Δrelief (local 3×3 max−min). Greedy nearest-neighbour
  polyline geometry per divide. LineString GeoJSON output.

Foundation primitive used by all five: `StreamGraph` +
`build_stream_graph(stream, flow_dir)` in `fluvial/stream_traversal.rs`.

29 unit tests, including the five `§7.2` headline golden tests from
the spec (all pass).

### Added — CLI

Five new subcommands under `surtgis fluvial`:

  - `fluvial chi STREAM FLOW_DIR FLOW_ACC OUTPUT [--theta-ref 0.45] [--a-0-m2 1e6]`
  - `fluvial ksn STREAM FLOW_DIR FLOW_ACC DEM OUTPUT [--segment-length-m 500] [--segments GEOJSON]`
  - `fluvial knickpoints STREAM FLOW_DIR FLOW_ACC DEM OUTPUT [--tvd-lambda 0.5] [--raster RASTER]`
  - `fluvial concavity STREAM FLOW_DIR FLOW_ACC DEM BASINS OUTPUT [--bootstrap-n 200]`
  - `fluvial divide-migration BASINS DEM FLOW_ACC OUTPUT [--chi CHI] [--min-divide-length-m 500]`

All handlers share a CRS-validation heuristic: reject inputs with
pixel size `< 1.0` unit (likely degrees) unless `--cell-size-m` is
supplied explicitly. Warns when cell size `> 30 m` per spec §8 pitfall
#1 (ksn / knickpoint sensitivity).

### Added — `hydrology stream-network --from-facc`

Optional flag to interpret the input as a pre-computed flow_accumulation
raster (skipping the DEM-side recomputation). Required for composable
workflows where stream-network must be topologically consistent with an
externally-computed flow_dir. Backward-compatible (default behaviour
unchanged).

### Notes

- Versioning is minor (0.9 → 0.10) because all additions are
  non-breaking; the existing public API surface is untouched. No
  `Breaking` heading required this release.
- Inter-crate path-deps bumped 0.9 → 0.10 across all 8 crates.
- Sprints 7-9 of the spec (TopoToolbox / pyTopoToolbox parity
  validation against published Smugglers Notch case + the Quezada
  Pangal AOI cross-check + mdBook chapter) are deferred to a follow-up
  release. They validate the algorithms against the literature but do
  not block usability for early adopters.

## [0.9.0] - 2026-05-22

Headline: closes the **G2 axis** of the roadmap — Geospatial Foundation
Model preprocessing pipeline. SurtGIS is now the first end-to-end
preparation tool for Prithvi-EO-2.0 / Clay v1.5 training data,
addressing the gap identified by InstaGeo (arxiv 2510.05617):
*"no published GFM includes its preprocessing pipeline."*

This is also the first release with **green CI on every job** after
several days of red — see commit `47f65ea` for the rebaseline (cargo
fmt + system libs + cfg gates + clippy correctness-only enforcement).

### Added — Geospatial Foundation Model (GFM) preprocessing

- **`extract-patches --profile <name>`** preset that targets a specific
  pre-trained foundation model. When set, the handler validates band
  count, applies the model's published per-band z-score normalization
  in place, and records full provenance (model target, band order, mean
  and std arrays, source URL, unit convention) in `meta.json` under the
  `gfm_profile` key. Supported profiles:
  - `prithvi-v2` → `ibm-nasa-geospatial/Prithvi-EO-2.0-300M`. 6 bands
    (B02, B03, B04, B05, B06, B07), tile 224×224, DN units (SR ×10000).
  - `clay-v1.5` → `made-with-clay/Clay/v1.5`. 10 Sentinel-2 bands
    (B02–B12 less B09, B10), tile 256×256, reflectance [0, 1].
- **Multi-timestamp input** for temporal foundation models like Prithvi.
  When `--features-dir` contains subdirectories (each with the same set
  of band rasters), each subdir is treated as a timestamp and the output
  tensor becomes `[N, C, T, H, W]` instead of `[N, C, H, W]`. Timestamp
  order follows lexicographic sort of subdir names (use ISO dates for
  natural ordering). Per-band z-score normalization runs across the full
  T·H·W block of each band. `meta.json` reports `n_timestamps`,
  `timestamps`, `tensor_layout`, and `tensor_shape`. Mixed mode (both
  top-level .tifs and subdirs with .tifs) is rejected with a clear error.
  Single-timestamp mode (top-level .tifs only) is preserved bit-for-bit
  for backward compatibility.
- New module `crates/cli/src/handlers/gfm_profiles.rs` with six unit
  tests covering name aliases, spec consistency, z-score correctness,
  NaN preservation, and the temporal-block normalization helper.
- **`extract-patches --output-format {npy,zarr}`** chooses the tensor
  serialisation. `npy` (default) is the existing single-file path; `zarr`
  emits a chunked Zarr v2 directory at `patches.zarr/` with one chunk per
  chip (`[1, C, (T,) H, W]`). Hand-written Zarr v2 writer at
  `crates/cli/src/handlers/zarr_writer.rs` — no new crate dependencies.
  `.zattrs` mirrors the meta.json keys so a Zarr-only consumer has full
  context. Output is bit-for-bit readable by `zarr` Python (`zarr.open()`)
  without any post-processing. Labels and manifest remain `.npy` / `.csv`.
- **`extract-patches --emit-stac`** writes STAC ML-AOI Collection + Items
  describing the chips as labelled training data. Output structure:
  `<output>/stac/collection.json` + `<output>/stac/items/chip_NNNNNN.json`.
  Each item declares `ml-aoi:role: label`, `ml-aoi:label_class` (int) or
  `ml-aoi:label_value` (float), bbox + Polygon geometry, and asset href
  pointing at the chip data. When a `--profile` is set, the Collection
  embeds the [STAC MLM extension](https://github.com/stac-extensions/mlm)
  declaring the target foundation model (e.g. `mlm:model_target:
  ibm-nasa-geospatial/Prithvi-EO-2.0-300M`) plus a full `mlm:input`
  descriptor with band order, tensor shape `[-1, C, T, H, W]`, dim_order,
  data_type, and per-channel normalization statistics. Coords are
  reprojected to WGS84 via proj4rs when the `projections` feature is
  compiled (default precompiled binaries); when not, items fall back to
  source-CRS coords stamped with `proj:epsg`. Hand-written STAC writer at
  `crates/cli/src/handlers/stac_writer.rs` with one unit test; smoke
  tested end-to-end with EPSG:32719 input verifying correct lon/lat
  output.

- **HLS Fmask cloud-mask strategy** for the Harmonized Landsat-Sentinel
  collection — the canonical input for NASA/IBM's Prithvi-EO-2.0. HLS
  Fmask is a 16-bit bitmask with HLS-specific bit assignments (cloud is
  bit 1, vs Landsat C2 QA_PIXEL where cloud is bit 3), so it needs its
  own strategy rather than reusing `LandsatQaMask`.
  - Default `HlsFmask::new()`: excludes cloud (bit 1), adjacent-to-cloud
    (bit 2), and cloud shadow (bit 3). Keeps clear, cirrus, snow, water.
  - `HlsFmask::strict()`: also excludes cirrus (bit 0) and snow (bit 4)
    for downstream uses sensitive to cirrus contamination.
  - `HlsFmask::with_bits(u16)`: arbitrary bit-set override.
  - Auto-routing: `stac::create_cloud_mask_strategy()` now inspects the
    asset name on a `CloudMaskType::Bitmask` and routes to `HlsFmask`
    when it contains "fmask" (case-insensitive), else `LandsatQaMask`.
    Existing Sentinel-2 SCL and SAR no-mask paths unchanged.
  - 2 new unit tests in `crates/algorithms/src/imagery/cloud_mask.rs`.

- **New how-to guide**: `docs/book/src/how-to/gfm-prithvi-prep.md`
  walks through the full pipeline from STAC bbox to a tensor ready to
  fine-tune Prithvi-EO-2.0 or Clay v1.5. Frames the work against
  InstaGeo's identified gap (no GFM ships preprocessing). Added to the
  user-guide nav.

- **GFM-prep benchmark harness** under `benchmarks/`:
  - `gfm_prep_make_dataset.py` generates a synthetic Prithvi-shaped
    dataset (6 HLS bands × T timestamps × G×G grid + N point labels).
  - `bench_gfm_prep_py.py` is a reference Python implementation of
    extract-patches (rasterio + numpy) with the same per-band z-score
    convention. Used as the "stay in Python" baseline.
  - `run_gfm_prep_bench.sh` orchestrates dataset generation, runs both
    implementations BENCH_REPS times per --size, writes a tidy CSV at
    `benchmarks/results/gfm_prep/timings.csv`.
  - `plot_gfm_prep.R` renders a paper-grade figure to
    `paper/figures/gfm_prep_throughput.pdf`.
  - First reference run (grid=1024, T=3, N=200, tile=224, 3 reps):
    SurtGIS mean 4.14s vs Python mean 6.36s — 1.54x speedup while
    SurtGIS additionally processes 3 timestamps per chip (output
    [N,C,T,H,W] vs Python's single-timestamp [N,C,H,W]). A like-for-like
    single-timestamp comparison would widen the gap further.
  - InstaGeo and raster-vision hooks intentionally out-of-scope: those
    add STAC + cloud-fetch overhead that's orthogonal to the local hot
    loop measured here. Documented in the script headers.

- **Auto-reprojection of vector input** in `extract-patches`. New
  `--points-crs <EPSG>` flag declares the EPSG of the points/polygons
  file (default 4326 — the GeoJSON spec mandates WGS84 lon/lat). When
  the raster's CRS differs, SurtGIS reprojects each point on the fly
  via proj4rs; polygons are reprojected vertex-by-vertex (exterior +
  interior rings). Before this fix, users passing a standard WGS84
  GeoJSON against a UTM raster got "No patch candidates produced"
  because lon/lat coords were being treated as projected meters.
  Validated end-to-end on the Maule mini example: 20/20 patches
  extracted from WGS84 GeoJSON against EPSG:32718 rasters, bit-exact
  match against the prior pyproj-preprocessed workflow. 3 unit tests
  added (identity, WGS84→UTM 18S sanity, round-trip).

This completes axis G2 of the roadmap (GFM preprocessing pipeline).

## [0.8.1] - 2026-05-17

Web demo expansion (roadmap axis A2.b). Closes the loop on v0.8.0:
the 23 new WASM functions that shipped in v0.8.0 are now exposed in
the live demo at https://franciscoparrao.github.io/surtgis/. Users
can now invoke Florinsky's 14 curvatures, MFD/D-∞ hydrology, and the
expanded spectral-index suite directly from a browser without
writing JavaScript.

### Added — web demo (`web/`)

- **Algorithm groups expanded** in `AlgoPanel.svelte` — 56 buttons
  organised into Terrain (21), Hydrology (8), Imagery (14),
  Morphology (4), Statistics (9). Headline new button:
  *Curvature (Florinsky 14)* with a dropdown for the 14 types.
- **Multi-band uploaders** in `App.svelte` — algorithms that need
  more than two bands (EVI: 3, BSI: 4) now render extra `FileUpload`
  slots automatically based on a `MULTI_BAND_LABELS` map. The
  primary uploader label changes too (e.g. "NIR Band" for NDVI,
  "SWIR Band" for BSI).
- **New parameter controls**: Florinsky type selector (14 options),
  openness directions + radius, focal percentile Q.
- **`SCHEME_MAP` updated** with reasonable colour schemes for all
  56 algorithms.

### Changed

- `runAlgorithm(name, demBytes, params, extraBands)` signature: the
  4th argument is now an array of additional band buffers (was a
  single optional `secondBand`). Lets EVI/BSI receive 2/3 extra
  bands cleanly. App.svelte's caller updated to pass the array.

### Notes

- No binary change vs v0.8.0; the WASM `.wasm` produced by
  `wasm-pack build` is bit-identical. Only the JS/CSS bundle changes.
- The `pkg/` artifacts (wasm + JS wrappers) include all 56 exports
  as a single TypeScript-typed surface.

## [0.8.0] - 2026-05-17

WASM expansion release. The browser-deployable surface grows from 33 to
**56 algorithms** (70 % expansion), substantially strengthening SurtGIS's
core differentiator: no other comprehensive terrain library runs
client-side in a web browser. This is the WASM-side counterpart to
roadmap axis A (see `ROADMAP.md`).

Minor version bump because the public binding surface for
`surtgis-wasm` grew incompatibly (new functions added; existing
functions unchanged). Inter-crate version pins moved from `0.7` to
`0.8` across the workspace; downstream Rust users who pin SurtGIS
sub-crates will need to bump the requirement.

### Added — Terrain (+5 new WASM bindings)

- **`advanced_curvature`** — exposes all 14 Florinsky curvatures
  (mean H, Gaussian K, unsphericity M, difference E, kmin, kmax, kh,
  kv, khe, kve, ka, kr, rotor, Laplacian) in the browser. The paper's
  headline contribution is now invocable from JavaScript. Aliases
  accepted (`mean`/`H`, `gaussian`/`K`, etc.).
- **`openness_positive`**, **`openness_negative`** — visibility-based
  terrain indices for ridge/valley discrimination.
- **`mrvbf`** — Multi-Resolution Valley Bottom Flatness (Gallant &
  Dowling 2003).

### Added — Hydrology (+3)

- **`flow_accumulation_mfd_compute`** — MFD accumulation in one pass
  from a DEM, with `MfdParams::default()`.
- **`flow_direction_dinf_compute`** — D-infinity flow angle raster.
- **`flow_accumulation_dinf_compute`** — D-infinity accumulation
  (computed alongside the directions via `flow_dinf`).

### Added — Imagery (+10)

- **`mndwi`**, **`nbr`**, **`ndre`**, **`gndvi`**, **`ndbi`**, **`ndmi`**,
  **`msavi`**, **`evi2`** — two-band spectral indices.
- **`evi`** — three-band Enhanced Vegetation Index (NIR + Red + Blue).
- **`bsi`** — four-band Bare Soil Index (SWIR + Red + NIR + Blue).

### Added — Focal Statistics (+6)

- **`focal_min`**, **`focal_max`**, **`focal_sum`**, **`focal_median`**,
  **`focal_majority`**, **`focal_percentile`** complete the window-based
  statistics suite (joining the existing mean, std, range).

### Notes for users

- Calling convention unchanged: GeoTIFF bytes in, GeoTIFF bytes out.
- Solar radiation is intentionally not exposed via WASM in this
  release: the native API requires pre-computed slope and aspect
  rasters, which is awkward for the single-buffer in / single-buffer
  out browser contract. Will revisit if there is demand.
- The web demo at `https://franciscoparrao.github.io/surtgis/` does
  not yet exercise all 56 functions; expanding the demo is the next
  follow-up (axis A2 in `ROADMAP.md`).

## [0.7.5] - 2026-05-16

**Critical correctness fix** for `stac composite` against Planetary
Computer Sentinel-2 L2A. Previous releases produced silently striped
outputs on basins where PC's COGs used `BitsPerSample=15` packed
encoding (`BUG_TILE_DECODE_BPS15_STRIPING.md`). Outputs from v0.7.0–
v0.7.4 should be re-validated against this release if Planetary Computer
was the source.

### Fixed
- **COG decoder now correctly unpacks `bps=15` tiles.** Previously the
  `(15, UNSIGNED_INT)` branch in `crates/cloud/src/decompress.rs` cast
  the raw buffer as if samples were padded `u16`, which is the wrong
  assumption — PC writes Sentinel-2 L2A red-edge bands (B05–B08, B8A,
  B11, B12) at *packed* 15 bits/sample for ~6.25 % disk savings. The
  cast produced only 245 760 samples per 512×512 tile (raw_len /
  sizeof(u16)) instead of the required 262 144, silently dropping the
  last 16 384 samples and producing rectangular striping when the
  median composite couldn't wash out the pattern across scenes. New
  `unpack_packed_to_u16` does proper MSB-first bit unpacking for any
  `bps ∈ 9..=16` packed stream.
- **Tile size mismatch is now a hard error** in `cog_reader.rs`. The
  pre-fix code logged a warning ("expected N px, got M") and copied the
  partial buffer into the output, producing the striping artefacts.
  Now any decode that produces a different sample count than the tile
  dimensions imply aborts the run with a clear error message pointing
  to the bug report.

### Added
- Three unit tests in `decompress.rs`:
  - Pack-then-unpack roundtrip with 16 hand-chosen 15-bit values.
  - Tile-size invariant (491 520 bytes ⇒ 262 144 samples).
  - Truncated buffer (not a clean multiple of bps bits) ⇒ error.

### Notes for users
- This is a correctness fix; v0.7.0–v0.7.4 outputs from Planetary
  Computer composites have systematic 6.25 % decoding losses on
  affected bands. Earth Search is unaffected (different COG encoding
  pipeline that does not use `bps=15`).
- The new hard-abort on size mismatch is intentional: silent partial
  decode is far worse than a loud failure for scientific workflows.
- Recommendation for the postdoc 15-cuencas pipeline: re-run any
  basins previously processed with PC under v0.7.0–v0.7.4 once v0.7.5
  wheels are published.

## [0.7.4] - 2026-05-16

Closes the last operational gap that forced users to fall back to
`gdalwarp` for raster reprojection. With `surtgis reproject`, the full
SurtGIS pipeline (STAC composite → reproject → terrain analysis →
output) now runs end-to-end without any system GDAL dependency.
Roadmap axis B (see `ROADMAP.md`).

### Added
- **New `surtgis reproject` command** (under `projections` feature):
  reprojects a GeoTIFF from any EPSG to any EPSG using proj4rs for the
  coordinate transform and a Rayon-parallelised inverse-mapping for
  per-pixel sampling.
  - `--to EPSG:XXXX` (required): target CRS.
  - `--from EPSG:XXXX` (optional): override the source CRS when the
    input GeoTIFF doesn't carry one.
  - `--method nearest|bilinear` (default bilinear): resampling kernel.
  - `--pixel-size <X>` (optional): output pixel size in target CRS
    units. Defaults to an auto-inferred value that preserves
    approximate resolution.
  - Smoke test reprojecting a 1000×1000 UTM 19S DEM → WGS84 takes
    0.13 s on the i7-1270P benchmark machine; → Web Mercator
    (1546×1550 output) takes 0.59 s.

### Notes for users
- The same-CRS case is short-circuited (input is copied to output
  without resampling).
- Cubic interpolation is not yet supported; nearest and bilinear cover
  the common cases.
- The output extent is computed from the source raster's corners plus
  edge midpoints, which handles most projection deformations but may
  underestimate for highly curved projections. Use `--pixel-size` and
  ensure the extent fits if working with extreme cases.

## [0.7.3] - 2026-05-16

Fixes the strip-pair even/odd peak-RAM pattern observed on the postdoc's
15-cuencas pipeline (`BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC.md` item #6),
where even-numbered strips (2, 4, ...) reached +2.5 GB transient peak vs
odd strips (1, 3, ...) at floor. Root cause confirmed by direct
measurement (see `crates/cli/examples/test_mi_collect.rs`): mimalloc
retains free'd pages in its per-thread free list and does not return
them to the OS until a decay tick or until pressure triggers cleanup.
`drop(scene_masks)` therefore appears as zero RSS reduction in
`/proc/self/status`, and the next strip's allocations pile on top of the
retained pages — producing the alternating pattern.

### Fixed
- **`stac composite`** now calls `mi_collect(true)` at the end of every
  strip iteration, immediately after the explicit `drop(scene_masks)`.
  This forces mimalloc to return abandoned segments to the OS so each
  strip's baseline RSS is the true working set, not the working set plus
  retained free pages. Expected peak reduction on Maule PC: ~13 GB →
  ~10 GB (−23%). A new `[ram] mi_collect(true): RSS=X MB → Y MB` log
  line at strip end reports the delta so users can verify the cleanup.

### Added
- `crates/cli/examples/test_mi_collect.rs`: reproducible 30-line test
  that demonstrates the mimalloc retain-vs-collect behaviour outside
  of the STAC composite codepath. Useful for future debugging if the
  pattern reappears under different load.

### Notes for users
- This is purely a RAM-behaviour fix; outputs are bit-identical to
  v0.7.2.
- The next strip's Phase A re-allocates from the OS rather than reusing
  warm mimalloc segments, costing a small amount of `mmap`/`madvise`
  overhead on each strip boundary. On a multi-hour-per-strip workload
  this is negligible compared to the peak-RAM benefit.
- WASM target unaffected (the fix is gated with `cfg(not(wasm32))`).

## [0.7.2] - 2026-05-15

Network-layer optimisation pass for `stac composite`. First step toward
closing the throughput gap with Planetary-Computer-Hub-native tooling
(`stackstac + dask`) when running SurtGIS from a workstation that is *not*
co-located with the data. On a Chile→Azure run the dominant cost is HTTP
latency × number of tile fetches; bigger gains come from running the binary
on the Hub itself, but this patch picks up the available headroom on the
client side.

### Changed
- **HTTP client retuned** in `crates/cloud/src/http.rs` and
  `stac_client.rs` for high-concurrency parallel fetches: 64 idle
  connections per host with a 60 s pool idle timeout, TCP_NODELAY, TCP
  keepalive at 30 s, HTTP/2 keepalive PINGs every 30 s with a 10 s timeout,
  HTTP/2 adaptive flow-control window. Together they cut the per-tile
  setup cost (TLS handshake, slow-start) for the long-running tile-fetch
  pattern. Required adding `"http2"` to the workspace `reqwest` features.
- **`tile_concurrency` defaults raised** to take advantage of HTTP/2
  multiplexing over the kept-alive connections: Earth Search 8 → 32,
  Planetary Computer 16 → 48, Custom 8 → 32. The pool size (64 per host)
  leaves headroom over the new caps.

### Notes for users
- Wider tile_concurrency increases the transient `decode` budget term
  (Planetary Computer: ~64 MB → ~192 MB worst case). Comfortably within
  the recalibrated peak introduced by v0.7.1; the budget print accounts
  for the new value automatically.
- Outputs remain bit-identical to v0.7.1; this is purely a network-layer
  patch.
- The biggest single-machine speedup for the cross-continent deployment
  remains running SurtGIS inside Planetary Computer Hub (compute next to
  data); the changes here are the available improvement when that is not
  possible.

## [0.7.1] - 2026-05-15

Diagnostic and budget-calibration patch for `stac composite`, addressing the
items raised in `BUG_RAM_V070_BUDGET_VS_REAL_MAULE_PC.md` from the postdoc's
15-cuencas pipeline. The structural refactor and allocator change from
v0.6.24 / v0.6.26 are unchanged; this release is purely about making the
budget print honest and the in-flight observability good enough to plan
capacity on memory-pressured systems.

### Changed
- **STAC composite budget formula recalibrated.** The mask-cache term now
  carries an empirical 1.8× inflation (postdoc observed real footprint of
  ~5–6 GB on Maule PC vs the previously-modelled 3 GB), and a 10% allocator-
  overhead term has been added on top of the variable working set. Both are
  fed into the auto-cap calculation and into the printed peak so users get
  a realistic figure (previously the print under-predicted observed peak by
  ~56% on Maule PC).
- **Budget print** now shows `allocator overhead: X.X GB` as a separate
  component and closes with a note that actual peak may be ±10% of estimate
  depending on system memory pressure.

### Added
- **Intra-chunk RSS watchdog.** A background thread samples `/proc/self/status`
  every 2 s and tracks the maximum RSS since the last chunk boundary; chunk-
  end log lines now include `peak_intra=NNNN MB (Δ=±M MB)` alongside the
  point-in-time RSS. This catches transient peaks (postdoc observed +815 MB
  undersample on Maule v0.7.0 — log reported 12.3 GB while real-time RSS
  reached 13.1 GB during the same chunk).
- **Phase A teardown log line.** Per-strip mask cache is now dropped
  explicitly at the end of each strip iteration and bracketed by RSS reads,
  emitting `phase A teardown: RSS=X MB → Y MB (Δ=±N MB)`. Together with the
  renamed `phase A masks loaded` line this lets users separate the strip→
  strip transition cost from the masks-loaded cost.

### Notes for users
- No algorithmic change; outputs are bit-identical to v0.7.0.
- The recalibrated budget will produce smaller `auto_strip_rows` on tight
  budgets (e.g. `SURTGIS_RAM_BUDGET_GB=8`), erring on the side of safety.
  If a previous v0.7.0 run completed without OOM at strip_rows=N, that
  configuration is still safe under v0.7.1 with the same budget.

## [0.7.0] - 2026-04-23

First release with precompiled binaries and a consolidation pass over docs
and distribution. Functionally identical to v0.6.28 — no new algorithms, no
API changes — but polished for evaluation by users outside the original
developer/postdoc feedback loop.

### Added
- **Precompiled binaries on GitHub Releases** for Linux x86_64, macOS arm64
  (Apple Silicon), and Windows x86_64. Produced by
  `.github/workflows/release.yml` on every `v*` tag push. Feature set:
  `cloud,zarr,projections` (STAC, COG, climate-data zarr readers, UTM
  reprojection). Users needing `netcdf` or `grib` can still
  `cargo install surtgis --all-features`.
- **Validated `Quick start` section** in README with a working end-to-end
  command (STAC composite → NDVI, ~65 s against Microsoft Planetary
  Computer) that a reader can paste verbatim.
- **Full v0.6.x CHANGELOG** with Keep-a-Changelog-formatted Added / Changed /
  Fixed sections for every release since 0.6.0.

### Policy
- From here on we pre-announce breaking changes one release before they
  land, flagged under a `Breaking` heading in the CHANGELOG entry.

## [0.6.28] - 2026-04-23
### Added
- COG reader now supports `s3://bucket/key` URLs by rewriting them to
  `https://bucket.s3.amazonaws.com/key` at the HTTP boundary. Earth Search
  returns raw `s3://` hrefs for several collections (notably
  `cop-dem-glo-30`), which previously failed with a "builder error" because
  reqwest doesn't handle the `s3://` scheme directly. Anonymous public-bucket
  reads now work through the default `--catalog es` path, and reqwest
  auto-follows the 307 redirect to the bucket's regional endpoint so no
  region config is needed.
### Internal
- New `Quick start` section in README with a validated end-to-end example
  (STAC composite → NDVI, ~65s against Planetary Computer).
- CHANGELOG.md rewritten with proper Keep-a-Changelog-formatted version
  history for the full v0.6.x series.

## [0.6.27] - 2026-04-23
### Added
- `extract-patches` top-level CLI command for CNN training sets. Reads a
  directory of aligned feature rasters (same discovery rules as `extract`) and
  a vector of either points or polygons, writes `patches.npy`
  (`[N, bands, H, W]` f32) + `labels.npy` (i64 or f32, auto-detected) +
  `manifest.csv` + `meta.json`. Points mode: one patch centred on each point.
  Polygons mode: grid sampling at `--stride` pixels with point-in-polygon test.
  No new crate deps — NPY header hand-rolled, subsample via seeded
  `DefaultHasher`.

## [0.6.26] - 2026-04-22
### Fixed
- STAC composite long-run RSS leak on Earth Search. Root cause: glibc malloc
  heap fragmentation under the mixed-size alloc/free pattern of decoded tile
  rasters — live data was bounded but RSS grew ~1.3 GB/min indefinitely.
  Resolved by switching to `mimalloc` as the global allocator for the CLI.
### Added
- `[ram]` diagnostic log lines at strip / phase / band-chunk transitions with
  a cumulative tile counter, so future RAM issues self-localise to a phase
  without requiring a rebuild with extra instrumentation.

## [0.6.25] - 2026-04-22
### Added
- `--band-chunk-size K` flag for `stac composite`: RAM↔HTTP dial. K=1 is the
  minimum-RAM default; higher K reduces HTTP request count at proportional RAM
  cost. Budget model updated to account for K.
- Exponential backoff with jitter in COG tile downloads: 4 attempts, base
  500 ms × 2^attempt, differentiates 429 (extra 2 s) / 503 / 404.
- Nightly CI job `stac-ram-bench` running `benchmarks/bench_stac_ram.sh`
  against real Earth Search. Asserts peak RSS below threshold and checks for
  monotonic-growth leak signature.
- `docs/postmortems/2026-04-stac-composite-ram.md` documenting the full
  v0.6.19 → v0.6.26 debugging arc.

## [0.6.24] - 2026-04-22
### Changed (structural)
- `stac composite` refactored from scene-outer to band-outer loop. Cloud masks
  (SCL / QA_PIXEL) are precomputed once per scene and reused across all bands;
  each band then iterates scenes independently. Peak RAM no longer scales with
  `n_bands`. Measured on Maule 7274×5725 × 10 bands × 42 dates: ~5 GB steady
  state vs >30 GB on v0.6.19.
### Fixed
- The v0.6.19–23 empirical RAM budgets underestimated real peak by 50–700%.
  v0.6.24 replaces them with a 5-component model (output + mask cache + scene
  strips + band working + decode) that matches observed behaviour.

## [0.6.23] - 2026-04-22
### Changed
- STAC composite RAM accounting adds a `per_scene_cache` term linear in
  `strip_rows`. **Partial fix** — real peak still underestimated by ~53%
  because the cache is near-constant in `strip_rows` (see v0.6.24).

## [0.6.22] - 2026-04-22
### Fixed
- STAC composite RAM spike on Earth Search. ES uses 1024² COG internal tiles
  vs 512² on Planetary Computer, producing ~4× larger decoded tile buffers.
  Auto-capped `strip_rows` + chunked tile downloads. **Partial fix** — see
  v0.6.24 for the structural resolution.

## [0.6.21] - 2026-04-20
### Added
- TIFF floating-point predictor (`predictor=3`) support in COG reader. Enables
  correct reads of Copernicus DEM GLO-30 and other float-predicted COGs.
  Verified bit-exact against GDAL.
- `pipeline susceptibility` auto-downloads DEM from STAC when given a
  collection ID (`cop-dem-glo-30`, `cop-dem-glo-90`, `nasadem`, `3dep-seamless`).

## [0.6.20] - 2026-04-20
### Changed (performance)
- `solar-radiation-annual` 7× faster in wall clock and 5× lower RAM.
  Copiapó 45 M cells went from OOM on v0.6.19 to 67 s / 3.5 GB. Pre-computed
  sun geometry outside the cell loop plus an `annual_only` path that drops
  per-month buffers after accumulating into the annual total.
### Fixed
- CLI parser accepts negative latitudes: `--latitude -27.5` now parses
  correctly (`allow_hyphen_values` on the argument).

## [0.6.19] - 2026-04-17
### Added
- Top-level `extract` command (moved out from behind `--features ml`; no
  longer requires `smelt-ml`).
- `clip --bbox xmin,ymin,xmax,ymax` as an alternative to `--polygon`.
- `terrain neighbours --stat <name>` for single-variable output (keeps the
  previous directory-of-five-files behaviour when `--stat` is not given).
- Python bindings: `extract_at_points` and `predict_raster` for integration
  with XGBoost / scikit-learn / numpy pipelines.
### Changed
- `extract` auto-discovers `.tif` files not listed in `features.json`.
- `--compress` flag unified across `pipeline` subcommands (removed per-command
  override that silently conflicted).
- `advanced-curvature --help` shows long-name aliases (`horizontal`,
  `vertical`, `accumulation`, etc.) alongside the short codes.

## [0.6.18] - 2026-04-17
### Fixed
- GDAL_NODATA TIFF tag (42113) is now read and written in the native GeoTIFF
  I/O path. Prior versions silently ignored it and treated nodata values as
  real data, corrupting downstream algorithms that depended on NaN propagation.
### Changed (performance)
- Gaussian smoothing: separable 1D row+column passes replace direct 2D
  convolution. Complexity drops from O(n·k²) to O(n·k). ~150× speedup for
  `sigma=50`.

## [0.6.17] - 2026-04-15
### Fixed
- Per-catalog STAC page-size limits: Planetary Computer accepts 1000, Earth
  Search caps at 250. Fixes 502 errors against ES when page > 250.

## [0.6.16] - 2026-04-15
### Changed
- Shared tokio runtime across all strips (was one per tile). Eliminates
  ~3,200 runtime creations per composite run and preserves HTTP connection
  pools across tiles.
- `--strip-rows` exposed as a CLI flag (was hardcoded).

## [0.6.15] - 2026-04-14
### Changed
- Multi-orbit Sentinel-2 composite: dates are now selected by coverage-aware
  ranking instead of pure chronological order, which produces denser coverage
  for small `--max-scenes`.

## [0.6.14] - 2026-04-14
### Fixed
- Landsat `QA_PIXEL`: fill pixels (bit 0) are now correctly excluded from
  composites. Previously they were treated as valid, producing edge artefacts.

## [0.6.13] - 2026-04-14
### Added
- 39 algorithms newly exposed in the CLI (coverage 55% → 78%). New command
  groups: `classification`, `texture`, `statistics`, `interpolation` (with 5
  methods: IDW, nearest, natural-neighbour, TPS, TIN), plus expansions to
  existing terrain / hydrology / morphology groups.

## [0.6.12] - 2026-04-14
### Changed
- HTTP retry in composites now operates at the per-band level rather than
  per-tile. Reduces redundant retries and preserves scene-level consistency
  when only one band of a scene fails.

## [0.6.11] - 2026-04-13
### Added
- Error logging restored in the async download path (was silently swallowed
  earlier). Per-strip diagnostic lines for OK / outside / partial / mask
  counts.

## [0.6.10] - 2026-04-13
### Fixed
- Planetary Computer STAC API 400 error when page size > 1000. Capped at 1000.

## [0.6.9] - 2026-04-13
### Fixed
- `search_limit` cap in multi-band composite was ignored. User-supplied
  values are now honoured.

## [0.6.8] - 2026-04-13
### Added
- Generic STAC collection support. Any collection with appropriate asset keys
  works (tested on WorldCover, custom DEMs, etc.) — previously restricted to
  Sentinel-2 L2A and Landsat C2 L2.

## [0.6.7] - 2026-04-13
### Added
- `tokio::spawn` parallelism for COG tile reads.
- Local disk cache for COG tiles at `~/.cache/surtgis/cog/` (enable with
  `--cache`). Fast re-runs across strips.

## [0.6.6] - 2026-04-12
### Changed
- Multi-band composite: concurrent band downloads (was serial per tile).

## [0.6.5] - 2026-04-12
### Added
- `--naming=asset` option for multi-band output files — writes `{band}.tif`
  instead of `{stem}_{band}.tif`.
### Changed
- Reduced log verbosity for `BBoxOutside` tiles. Typical runs see hundreds of
  these and the volume drowned out meaningful output.

## [0.6.4] - 2026-04-12
### Fixed
- Empty strip in multi-band composite when every tile of a given strip
  returned `BBoxOutside` for a band. Now handled without crashing.

## [0.6.3] - 2026-04-11
### Changed
- Multi-band composite refactored to single-pass: shared STAC search + shared
  cloud mask across bands. `N` bands is ~N× faster than running `N`
  single-band composites back-to-back.

## [0.6.2] - 2026-04-11
### Fixed
- SAS token expiration on long-running composites. Tokens are now re-signed
  when within 30 minutes of expiry.
- RFC 3339 datetime normalisation in STAC queries (trailing `Z` handling).

## [0.6.1] - 2026-04-10
### Fixed
- Multi-UTM composite coverage: bbox is reprojected to each tile's native CRS
  before the COG read. Previously, tiles in non-reference UTM zones returned
  `BBoxOutside` and were silently dropped.

## [0.6.0] - 2026-04-09
### Added
- Zarr reader (ERA5, TerraClimate) with Azure Blob + Planetary Computer auth.
- NetCDF reader (CMIP6) with CF Conventions support.
- GRIB2 reader (HRRR, ECMWF, GFS).
- `CloudRasterReader` trait for format-agnostic access across COG / Zarr /
  NetCDF / GRIB2.
- `stac download-climate` CLI with temporal aggregation.
- Atomic file writes for the write path (prevents corrupt partials on crash).
- Index-parameter reproducibility via `indices_metadata.json` companion files.
### Validated
- 15 Chilean watersheds (3800 km span, Arica → Punta Arenas) on real ERA5,
  CMIP6, and GFS data.
- ~100% STAC format coverage: COG + Zarr + NetCDF + GRIB2.

## [0.5.0] and earlier

Release history before v0.6.0 covers the initial workspace bring-up
(core, algorithms, parallel, cli, wasm, python, colormap, gui crates),
the first ~100 algorithms across terrain / hydrology / imagery / morphology /
landscape / classification / texture / statistics / interpolation /
temporal / vector, the native GeoTIFF I/O path, WASM bindings, Python
bindings via PyO3, the desktop GUI (`surtgis-gui`), and the first wave of
cross-validation tests against GDAL 3.11 and GRASS 8.3.

## [0.0.1] - 2024-12-01
### Added
- Initial project structure with workspace layout.
- Basic `Raster<T>` type and GeoTIFF support.
- First terrain algorithms (slope, aspect, hillshade).
- Hydrology module (fill sinks, flow direction, flow accumulation).
