# Changelog

All notable changes to SurtGIS are documented in this file.

Format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer 2.0.0](https://semver.org/). The project is still in
the `0.x` series, so minor versions may contain breaking changes; we try to
call them out under a `Breaking` heading when they happen.

## [Unreleased]

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
