# SurtGIS — Future Roadmap

> High-performance geospatial analysis library in Rust
> Last updated: 2026-02-03

---

## Dashboard

### Progress by Priority

| # | Priority | Category | Total | Done | Remaining | % |
|---|----------|----------|-------|------|-----------|---|
| 1 | Web Demo + npm Package | WASM | 8 | 6 | 2 | 75% |
| 2 | Publish crates.io + PyPI | Publishing | 8 | 0 | 8 | 0% |
| 3 | Cloud Optimized GeoTIFF | Cloud-native | 5 | 3 | 2 | 60% |
| 4 | Tests con datos reales + Clippy clean | Quality | 8 | 8 | 0 | 100% |
| 5 | wgpu Compute Shaders | GPU | 5 | 0 | 5 | 0% |
| 6 | LAS/LAZ → DEM Pipeline | LiDAR | 4 | 0 | 4 | 0% |
| 7 | Desktop GUI (egui) | GUI | 8 | 8 | 0 | 100% |
| **Total** | | | **46** | **25** | **21** | **54%** |

### Progress by Category

| Category | Items | Done | Status |
|----------|-------|------|--------|
| WASM / Web | 8 | 6 | **In progress** |
| Publishing (crates.io + PyPI) | 8 | 0 | Not started |
| Cloud-native I/O | 5 | 3 | **In progress** |
| Quality / Testing | 8 | 8 | **Complete** |
| GPU Compute | 5 | 0 | Not started |
| LiDAR | 4 | 0 | Not started |
| Desktop GUI | 8 | 8 | **Complete** |

---

## Current State (Baseline)

Before these future priorities, SurtGIS already has:

- **9 crates**: core, algorithms, parallel, cli, wasm, python, cloud, colormap, gui
- **88 algorithm modules** across 7 categories (terrain, hydrology, imagery, interpolation, vector, morphology, statistics)
- **591+ tests** passing (508 unit + 19 integration + 38 cloud + 12 colormap + 14 other)
- **26+ CLI commands** with full terrain/hydrology/imagery coverage + `cog` and `stac` subcommands
- **Desktop GUI** with SAGA-style dock layout, 44 algorithms, basemap tiles, STAC browser, 3D wireframe
- **WASM target** compiles (`wasm32-unknown-unknown`) — reproject module WASM-compatible
- **Python bindings** via PyO3 + maturin
- **CI/CD** with 7 jobs (check, clippy, test, test-no-parallel, wasm, python, fmt, bench)
- **Cloud-native**: COG reader (HTTP Range) + STAC client (Earth Search, Planetary Computer) + auto WGS84→UTM reprojection
- **Performance**: 2-21x faster than GDAL, SAGA, WhiteboxTools, R/terra on terrain operations
- **License**: MIT OR Apache-2.0

---

## Priority 1: Web Demo + npm Package (WASM diferenciador)

**Goal**: Make SurtGIS the first Rust GIS library usable from JavaScript/TypeScript in the browser.

**Impact**: Unique differentiator — no existing library provides high-performance terrain analysis in WebAssembly.

### Tasks

- [x] **1.1** Configure `wasm-pack build` for bundler + web targets ✓
  - Output: `pkg/` directory with `.wasm`, `.js`, `.d.ts` files
  - Target: `--target bundler` (webpack/vite)
  - Optimized: wasm-opt via wasm-pack, 572 KB binary (195 KB gzip)
  - 33 functions exported

- [x] **1.2** npm package setup ✓
  - `package.json` with name `surtgis`, version, description, keywords
  - `README.md` with installation, API reference, examples
  - `scripts/post_wasm_build.sh` to re-apply metadata after wasm-pack
  - License: MIT OR Apache-2.0

- [x] **1.3** TypeScript type definitions (`.d.ts`) ✓
  - Auto-generated `surtgis_wasm.d.ts` with 33 exported functions
  - Hand-written `surtgis.d.ts` for ergonomic wrapper (SurtGIS class)
  - `surtgis-worker.d.ts` for Web Worker API
  - All algorithm signatures with option interfaces

- [x] **1.4** JavaScript API wrapper (ergonomic) ✓
  - `SurtGIS.init()` initialization
  - 33 methods organized in 5 groups: Terrain, Hydrology, Imagery, Morphology, Statistics
  - Options objects with sensible defaults
  - Error handling with descriptive messages

- [x] **1.5** Web Worker integration ✓
  - `worker.js` loads WASM and processes messages
  - `SurtGISWorker` class in main thread (message-based API)
  - `postMessage` / `onmessage` protocol with request IDs
  - Transferable ArrayBuffers for zero-copy data transfer

- [x] **1.6** Interactive web demo ✓
  - Drag & drop GeoTIFF upload
  - 33 algorithms in 5 collapsible groups with parameter controls
  - Canvas visualization with 8 color schemes (terrain, divergent, grayscale, ndvi, blue_white_red, geomorphons, water, accumulation)
  - Side-by-side comparison (DEM vs. output)
  - Performance timer showing processing time
  - GitHub Pages deploy workflow created

- [ ] **1.7** Publish to npmjs.com
  - `npm publish` with 2FA
  - Verify install: `npm install surtgis` works
  - Test in fresh project (Vite + vanilla JS)
  - Test in React/Vue/Svelte starter

- [ ] **1.8** JavaScript/TypeScript documentation with examples
  - Getting started guide
  - API reference (auto-generated from `.d.ts`)
  - Examples: Node.js (server-side), Browser (client-side), React component
  - Performance tips (Web Workers, chunk processing)
  - Migration guide for users of geotiff.js / georaster-layer-for-leaflet

---

## Priority 2: Publish on crates.io + PyPI

**Goal**: Make SurtGIS installable via `cargo add surtgis` and `pip install surtgis`.

**Impact**: Standard distribution channels maximize reach.

### Tasks

- [ ] **2.1** Prepare crates.io metadata
  - `description` field in all workspace Cargo.toml
  - `keywords`: gis, geospatial, raster, terrain, hydrology
  - `categories`: science::geo, command-line-utilities
  - `repository`: github.com/franciscoparrao/surtgis
  - `documentation` pointing to docs.rs
  - `readme` field pointing to crate-level README
  - `exclude` patterns for test data, docs, benchmarks

- [ ] **2.2** Verify license in all crates
  - All 6 crates must have `license = "MIT OR Apache-2.0"`
  - LICENSE-MIT and LICENSE-APACHE files at workspace root
  - `cargo deny check licenses` passes

- [ ] **2.3** Publish crates in dependency order
  1. `surtgis-core` → verify on crates.io
  2. `surtgis-parallel` → depends on core
  3. `surtgis-algorithms` → depends on core + parallel
  4. `surtgis` (CLI) → depends on all above
  5. `surtgis-wasm` → separate, depends on core + algorithms
  - Use `cargo publish --dry-run` first
  - Verify `cargo install surtgis` works

- [ ] **2.4** Python: configure maturin + PyPI
  - `pyproject.toml` with maturin backend
  - Python version support: 3.9-3.13
  - Platform wheels: linux-x86_64, macos-arm64, macos-x86_64, windows-x86_64
  - `maturin publish` to PyPI (or TestPyPI first)
  - GitHub Actions for automated wheel builds

- [ ] **2.5** Native Raster type in Python
  - `Raster.__repr__()` showing shape, dtype, CRS, bounds
  - `Raster.plot()` using matplotlib (optional dependency)
  - `Raster.to_numpy()` → numpy array (zero-copy if possible)
  - `Raster.from_numpy(array, geotransform, crs)` constructor
  - `Raster.save(path)` and `Raster.load(path)` convenience methods
  - `Raster.shape`, `.dtype`, `.crs`, `.bounds`, `.resolution` properties

- [ ] **2.6** xarray integration
  - `Raster.to_xarray()` → `xarray.DataArray` with spatial coordinates
  - `surtgis.from_xarray(da)` → Raster
  - CRS and geotransform preserved in xarray attrs
  - Compatibility with rioxarray for `.rio.to_raster()` roundtrip

- [ ] **2.7** `pip install surtgis` functional end-to-end
  - Verify: `pip install surtgis && python -c "import surtgis; print(surtgis.__version__)"`
  - Test basic workflow: load → process → save
  - Ensure GDAL is NOT a runtime dependency (pure Rust I/O)
  - Fallback error message if optional deps missing

- [ ] **2.8** Jupyter notebook examples
  - `examples/01_terrain_analysis.ipynb` — slope, aspect, hillshade
  - `examples/02_hydrology.ipynb` — fill sinks, flow direction, TWI
  - `examples/03_spectral_indices.ipynb` — NDVI, NDWI from satellite imagery
  - `examples/04_interpolation.ipynb` — IDW, kriging, TPS
  - Runnable on Google Colab with `!pip install surtgis`

---

## Priority 3: Cloud Optimized GeoTIFF (COG) Reader

**Goal**: Read raster data directly from cloud storage without downloading entire files.

**Impact**: Essential for working with large satellite datasets (Sentinel-2, Landsat, Copernicus DEM).

### Tasks

- [x] **3.1** COG reader nativo (HTTP Range requests) ✓
  - Custom TIFF IFD parser (~300 LOC) via HTTP Range requests
  - Support overview levels (pyramids) with automatic selection
  - HTTP client: reqwest with async + connection pooling
  - Authentication: AWS S3 (SigV4) + extensible CloudAuth trait
  - Retry logic with exponential backoff (3 retries default)
  - Cache: LRU cache for tiles (128 tiles default)
  - New crate: `surtgis-cloud` (12 modules, 19 tests)
  - Verified with Copernicus DEM 30m on AWS S3

- [x] **3.2** Tiled reading (only read necessary tiles) ✓
  - BBox → tile IDs calculation with GeoTransform
  - Merge tiles into contiguous Raster<T>
  - Compression support: DEFLATE (zlib), LZW
  - Concurrent tile fetching (8 parallel default, FuturesOrdered)
  - CLI integration: 7 `surtgis cog` subcommands (info, fetch, slope, aspect, hillshade, tpi, fill-sinks)
  - Sync API (CogReaderBlocking) + WASM bindings

- [ ] **3.3** Zarr/NetCDF reader
  - Zarr v2 store reader (local + HTTP)
  - NetCDF-4/HDF5 via zarr compatibility
  - Chunk-based reading (similar to COG tiles)
  - Support climate/weather data conventions (CF conventions)

- [x] **3.4** STAC client con reproyeccion automatica ✓
  - `StacClient` with POST /search: bbox, datetime, collections, pagination (next links)
  - 3 built-in catalogs: Planetary Computer (`pc`), Earth Search (`es`), custom URL
  - Planetary Computer SAS token signing (automatic, cached)
  - `StacItem` model with `epsg()` helper (proj:epsg extension)
  - `read_stac_asset()` / `search_and_read()` bridge STAC → COG reader
  - Automatic WGS84 → UTM bbox reprojection (Snyder 1987 formulas, pure Rust, WASM-compatible)
    - Supports EPSG 326xx (UTM North) and 327xx (UTM South), zones 1-60
    - Detects CRS from STAC `proj:epsg` (preferred) or COG metadata (fallback)
    - Precision < 1m vs PROJ/pyproj
  - Blocking sync wrappers (`StacClientBlocking`)
  - CLI: `surtgis stac search` and `surtgis stac fetch` subcommands
  - 12 unit tests (models, pagination, EPSG parsing) + 11 reprojection tests

- [ ] **3.5** Streaming I/O for large datasets
  - Process raster in chunks without loading full dataset in memory
  - `StreamingRaster` type with `chunks()` iterator
  - Pipeline: read chunk → process → write chunk
  - Memory budget control (e.g., max 512 MB)
  - Compatible with parallel processing (tiled strategy)

---

## Priority 4: Tests con datos reales + Clippy clean

**Goal**: Zero clippy warnings, real-data integration tests, GDAL backend fix.

**Impact**: Production readiness and confidence in numerical correctness.

### Tasks

- [x] **4.1** Auto-fix clippy warnings ✓
  - Run `cargo clippy --fix --workspace --exclude surtgis-python --allow-dirty --allow-staged`
  - ~88 warnings fixed automatically (needless_range_loop, unnecessary_map_or, manual_map, etc.)

- [x] **4.2** Manual clippy fixes ✓
  - 59 `needless_range_loop`: converted to `iter_mut().enumerate()`
  - 9 `too_many_arguments`: added `#[allow(clippy::too_many_arguments)]`
  - 5 `unnecessary_parentheses`: removed extra parens
  - 4 `struct_update_has_no_effect`: removed `..Default::default()`
  - 3 `clamp-like pattern`: converted to `.clamp()`
  - 2 `&mut Vec` → `&mut [_]`, 1 `&PathBuf` → `&Path`
  - 1 `type_complexity`, 1 `empty_line_after_doc_comment`
  - Result: **0 clippy warnings** with `-D warnings`

- [x] **4.3** Fix GDAL backend (3 compilation errors in `gdal_io.rs`) ✓
  - Fixed `no_data_value()` return type (Option, not Result)
  - Fixed `create_with_band_type_with_options` params (usize, CslStringList)
  - Fixed `band.write` to use `Buffer<T>` type
  - `cargo check --package surtgis-core --features gdal` compiles clean

- [x] **4.4** Download real DEM of Chile (Copernicus 30m, Andes) ✓
  - Area: Andes centrales `[-70.0, -33.35, -69.8, -33.15]`
  - 720×720 pixels, elevation 2858-5981m
  - Downloaded via Planetary Computer (Copernicus GLO-30)
  - Size: 1.66 MB

- [x] **4.5** Create `tests/fixtures/` directory with DEM ✓
  - File: `tests/fixtures/andes_chile_30m.tif`
  - CRS: EPSG:4326, 100% valid coverage

- [x] **4.6** Integration tests with real DEM ✓
  - File: `crates/algorithms/tests/integration_real_dem.rs`
  - 13 tests: I/O roundtrip, slope, aspect, hillshade, curvature (general/profile/plan),
    fill_sinks, flow pipeline, TWI, focal mean/std, no-Inf check
  - Sanity checks: slope ∈ [0°, 90°], aspect ∈ [0°, 360°], hillshade ∈ [0, 255]
  - All 13 tests pass in ~5s

- [x] **4.7** Cross-validation vs GDAL/GRASS outputs ✓
  - DEM reprojected to UTM (EPSG:32719) for unambiguous cellsize comparison
  - Reference: GDAL 3.11 `gdaldem slope/aspect`, GRASS 8.3 `r.slope.aspect -n`
  - 6 integration tests in `cross_validation_gdal_grass.rs` (516K+ pixels compared)
  - Results (all three use Horn's 1981 method):
    - Slope SurtGIS vs GDAL: RMSE=0.0002°, MAE=0.0001° (threshold: 0.5°)
    - Slope SurtGIS vs GRASS: RMSE=0.0000°, MAE=0.0000° (threshold: 0.5°)
    - Aspect SurtGIS vs GDAL: RMSE=0.0021°, MAE=0.0005° (threshold: 1.0°)
    - Aspect SurtGIS vs GRASS: RMSE=0.0000°, MAE=0.0000° (threshold: 1.0°)
  - Differences documented: GDAL uses float32 internally (tiny rounding diffs),
    GRASS uses float64 (identical to SurtGIS). GDAL 3.11+ auto-scales geographic CRS.
  - Edge handling: all three exclude 1-pixel border (set to nodata)

- [x] **4.8** CI: add conditional integration test ✓
  - Added `integration` job to `.github/workflows/ci.yml`
  - Checks for fixture existence before running
  - `cargo test --test integration_real_dem --package surtgis-algorithms`

---

## Priority 5: wgpu Compute Shaders

**Goal**: GPU-accelerated geospatial processing via WebGPU/wgpu.

**Impact**: 10-100x speedup for focal operations on large rasters.

### Tasks

- [ ] **5.1** New crate: `surtgis-gpu`
  - Dependency: `wgpu` crate
  - GPU device enumeration and selection
  - Buffer management for Raster → GPU buffer → Raster
  - Error handling for missing GPU support
  - Feature flag: `gpu` in workspace

- [ ] **5.2** Compute shader for slope/hillshade
  - WGSL shader: 3×3 neighborhood kernel for slope (Horn's method)
  - WGSL shader: hillshade with configurable azimuth/altitude
  - GPU buffer layout: input DEM + output raster + uniform params
  - Workgroup size tuning (8×8 or 16×16)

- [ ] **5.3** Compute shader for focal statistics
  - WGSL shader: generic NxN moving window
  - Operations: mean, median, std, min, max, sum, range
  - Shared memory optimization for overlapping windows
  - Support for NoData handling in shader

- [ ] **5.4** Benchmark CPU vs GPU
  - Test matrix: 256×256, 1024×1024, 4096×4096, 8192×8192
  - Algorithms: slope, hillshade, focal_mean(5×5), focal_mean(11×11)
  - Include data transfer overhead in GPU timing
  - Criterion benchmarks with comparison groups
  - Document crossover point (size where GPU beats CPU)

- [ ] **5.5** WebGPU compatibility (browser)
  - Verify shaders work with browser WebGPU
  - WASM + wgpu (WebGPU backend) integration
  - Fallback to CPU if WebGPU not available
  - Demo page with GPU-accelerated terrain analysis

---

## Priority 6: LAS/LAZ → DEM Pipeline

**Goal**: Complete point cloud to DEM pipeline, from raw LiDAR to terrain analysis.

**Impact**: LiDAR data is increasingly available (USGS 3DEP, Spain PNOA, etc.).

### Tasks

- [ ] **6.1** Integrate `las-rs` crate
  - Read LAS 1.2-1.4 and LAZ (compressed) files
  - Point attributes: x, y, z, classification, return number, intensity
  - Lazy reading for large files (streaming iterator)
  - Basic statistics: point count, bounds, density, classification histogram

- [ ] **6.2** Ground classification (simple morphological filter)
  - Progressive Morphological Filter (PMF) — Zhang et al. 2003
  - Parameters: initial window size, max window size, slope threshold, height threshold
  - Output: classification label (ground=2, non-ground=1)
  - Optional: Cloth Simulation Filter (CSF) as alternative

- [ ] **6.3** Point cloud → DEM rasterization
  - Grid creation from point extent + specified resolution
  - Binning: assign points to grid cells
  - Aggregation: min (DTM), max (DSM), mean, IDW within cell
  - NoData filling: IDW interpolation for empty cells
  - Output: Raster<f64> with GeoTransform and CRS

- [ ] **6.4** Complete pipeline: LAZ → classify → rasterize → analyze
  - `surtgis lidar pipeline input.laz --resolution 1.0 --output dem.tif`
  - Steps: read → classify ground → rasterize → fill voids → terrain analysis
  - CLI subcommands: `lidar info`, `lidar classify`, `lidar rasterize`, `lidar pipeline`
  - Example with real LiDAR data (USGS 3DEP sample)

---

## Priority 7: Desktop GUI (egui) — COMPLETE

**Goal**: SAGA GIS-style desktop application for interactive geospatial analysis.

**Impact**: Makes SurtGIS accessible to non-programmers and provides interactive visual feedback.

### Tasks

- [x] **7.1** Phase 0: egui + eframe scaffold ✓
  - `surtgis-gui` binary crate with eframe 0.33 + wgpu renderer
  - Dark theme, resizable window (1400×900 default)

- [x] **7.2** Phase 1: Map canvas + GeoTIFF I/O ✓
  - `MapCanvasState` with zoom (mouse wheel) and pan (drag)
  - Raster → RGBA texture via `surtgis-colormap` crate (8 color schemes)
  - File > Open (GeoTIFF via rfd dialog) / File > Save
  - Cursor position tracking (pixel + geographic coordinates)

- [x] **7.3** Phase 2: SAGA-style dock layout ✓
  - 6-panel layout via egui_dock: Map, Algorithms, Properties, Data, Layers, Console
  - 44 algorithms in collapsible tree with tool dialog (parameter UI auto-generated)
  - Tools menu auto-generated from algorithm registry (7 categories)
  - Background execution via crossbeam channels + std::thread
  - Scale bar with auto-ranging (m/km/degrees)
  - Dataset selector bar, layer visibility/opacity, colormap selector

- [x] **7.4** Phase 3: Upgrade egui 0.33 ✓
  - Migrated from egui 0.31 to 0.33, eframe 0.33, egui_dock 0.18
  - Updated deprecated API: `MenuBar::new().ui()`, `ui.close()`
  - Fixed wgpu 27 backend feature propagation

- [x] **7.5** Phase 3: OpenStreetMap basemap tiles ✓
  - walkers 0.52 integration with `HttpTiles` (OSM tile source)
  - `RasterOverlay` plugin draws raster texture on top of basemap
  - Toggle via View > Map Mode (Simple / Basemap)
  - Lazy initialization centered on active dataset

- [x] **7.6** Phase 3: STAC browser panel ✓
  - Catalog selector: Planetary Computer / Earth Search / Custom URL
  - Search filters: bbox, datetime range, collection, max cloud cover (slider)
  - "Use Map Extent" button to fill bbox from active dataset
  - Results table with date, cloud%, platform, GSD, collection columns
  - Background search/download via `StacClientBlocking` + crossbeam channels
  - Signed URL support for Planetary Computer assets
  - Feature-gated (`cloud`), enabled by default in GUI

- [x] **7.7** Phase 3: 3D wireframe view ✓
  - Isometric 3D→2D projection via egui Painter (no wgpu pipeline)
  - Elevation-colored wireframe grid using active colormap
  - Interactive controls: azimuth (0-360), elevation (5-85), z-exaggeration (0.1-20x), step (1-32)
  - Mouse drag for rotation
  - Subsample step configurable for performance vs detail

- [x] **7.8** Phase 3: Tiled renderer for large rasters ✓
  - LRU cache of 256 tiles (512×512 pixels each)
  - Activates when raster > 16M pixels (4096×4096)
  - Generation-based invalidation for data/colormap changes
  - `visible_tiles()` calculates on-screen tiles from viewport geometry

---

## Dependencies Between Priorities

```
Priority 1 (WASM)  ←── Priority 5 (GPU, WebGPU compat)
Priority 2 (Publish) ←── Priority 4 (Quality, must pass CI)
Priority 3 (COG)    ←── independent
Priority 6 (LiDAR)  ←── independent
Priority 7 (GUI)    ←── COMPLETE
```

**Recommended execution order**:
1. **Priority 4** (Quality) — foundation for everything else
2. **Priority 2** (Publish) — depends on clean clippy + tests
3. **Priority 1** (WASM) — differentiator, most visible impact
4. **Priority 3** (COG) — enables cloud workflows
5. **Priority 6** (LiDAR) — new data source
6. **Priority 5** (GPU) — performance optimization, most complex
7. ~~Priority 7~~ (GUI) — **DONE**

---

## Architecture Notes

### Crate Dependency Graph

```
surtgis-core         (types, I/O, raster)
    ↑                       ↑
surtgis-parallel     surtgis-cloud  (COG reader, HTTP Range, tile cache)
    ↑                       ↑
surtgis-algorithms          |
    ↑           ↑           |
surtgis (CLI) ──────────────┘   surtgis-wasm    surtgis-python    surtgis-gpu (new)
    ↑
surtgis-colormap
    ↑
surtgis-gui  (egui desktop, walkers basemap, STAC browser, 3D view)
```

### Key Types

| Type | Location | Purpose |
|------|----------|---------|
| `Raster<T>` | `core::raster` | N-dimensional raster with geotransform |
| `GeoTransform` | `core::raster` | Affine transform (origin, resolution, rotation) |
| `CRS` | `core::crs` | Coordinate Reference System wrapper |
| `Neighborhood` | `core::raster::neighborhood` | NxN window accessor for focal operations |
| `TiledStrategy` | `parallel::tiled` | Tiled parallel processing |
| `SurtGisApp` | `gui::app` | Main desktop application (eframe::App) |
| `BasemapState` | `gui::render::map_tiles` | OSM basemap state (walkers) |
| `TiledRenderer` | `gui::render::tiled_renderer` | LRU tile cache for large rasters |
| `StacBrowserState` | `gui::panels::stac_browser` | STAC search UI state |

### Feature Flags

| Feature | Default | Purpose |
|---------|---------|---------|
| `parallel` | yes | Enable rayon parallelism |
| `gdal` | no | GDAL I/O backend |
| `simd` | no | SIMD optimizations (planned) |
| `cloud` | yes (CLI, GUI) | COG reader + STAC client via surtgis-cloud |
| `gpu` | no | wgpu compute shaders (Priority 5) |

---

## References

- [Florinsky, I.V. (2012) "Digital Terrain Analysis in Soil Science and Geology"](https://doi.org/10.1016/C2010-0-65718-X)
- [Horn, B.K.P. (1981) "Hill Shading and the Reflectance Map"](https://doi.org/10.1109/PROC.1981.11918)
- [Zhang et al. (2003) "A Progressive Morphological Filter for Removing Nonground Measurements from Airborne LIDAR Data"](https://doi.org/10.1109/TGRS.2003.810682)
- [COG Specification](https://www.cogeo.org/)
- [STAC Specification](https://stacspec.org/)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
