# SurtGIS vs SAGA: Parity Roadmap

> Source: External review, April 2026
> SAGA: 800+ tools, 20+ years. SurtGIS: 136 algorithms, architectural advantages (streaming, Rayon, cloud-native).

## Current SurtGIS Advantages over SAGA

- Streaming I/O (~200MB RAM for arbitrarily large DEMs)
- Cloud-native (COG, Zarr, NetCDF, GRIB2 via STAC)
- Zero external deps (no GDAL/PROJ install required)
- 2-20x faster on terrain/hydrology benchmarks
- Single binary: CLI, Rust lib, Python, WASM, GUI

## Gap Analysis (what SAGA has that SurtGIS doesn't)

### Priority 1: QGIS Processing Provider (SHORT TERM)
- **Impact**: 90% of SAGA users access it through QGIS, not SAGA's own GUI
- **Approach**: Python plugin wrapping surtgis-python bindings or CLI
- **Effort**: Medium — QGIS Processing API is well-documented
- **ROI**: Highest. Instant visibility to millions of QGIS users.

### Priority 2: Smelt-ML Spatial Geostatistics (MEDIUM TERM)
- **Impact**: SAGA's geostatistics module is one of its strongest features
- **What to add**:
  - Semivariogram modeling (experimental, theoretical, fitting)
  - Universal Kriging, Kriging with External Drift (KED)
  - Regression-Kriging (Random Forest residuals + Kriging)
  - Spatial cross-validation (already in Smelt-ML)
- **Approach**: Smelt-ML already has RF, GBM, spatial CV. Add kriging + semivariogram to complete the geostatistics stack. Expose via `surtgis ml` CLI.
- **Effort**: High — kriging math is non-trivial
- **ROI**: High. Differentiator vs all GIS tools (none have native ML+kriging in one binary).

### Priority 3: LiDAR / Point Cloud (MEDIUM TERM)
- **Impact**: SAGA is popular for LiDAR processing (forestry, topography)
- **What to add**:
  - LAS/LAZ reading (crates: `las`, `laz`)
  - Ground filtering (CSF, morphological)
  - Point cloud to DEM/DSM rasterization
  - Canopy height model (CHM = DSM - DTM)
  - Forest metrics (height percentiles, density)
- **Approach**: New `surtgis-lidar` crate. Already in future_roadmap.md as Priority 6.
- **Effort**: High
- **ROI**: Medium-high. Opens forestry/ecological use cases.

### Priority 4: CRS / Reprojection Engine (MEDIUM TERM)
- **Impact**: SurtGIS has WGS84↔UTM only. SAGA uses full PROJ.
- **What to add**:
  - More projections (Lambert, Albers, Mercator, polar stereographic)
  - Datum transformations (grid shifts)
  - On-the-fly reprojection for rasters and vectors
- **Approach**: `proj4rs` crate (pure Rust PROJ reimplementation) or feature-gated `proj` C bindings
- **Effort**: Medium
- **ROI**: Medium. Required for global datasets but most users work in UTM/WGS84.

### Priority 5: Vector Geoprocessing (LONG TERM)
- **Impact**: SAGA has robust vector topology engine
- **What to add**:
  - Intersection, Union, Difference (overlay operations)
  - Buffer, Dissolve, Spatial Join
  - Vector to TIN (Delaunay triangulation)
- **Approach**: `geo` crate already has boolean ops. `geos` crate for topology.
- **Effort**: Very high (topology is hard)
- **ROI**: Medium. Most SurtGIS users are raster-focused.

### Priority 6: Network Analysis (LONG TERM)
- **Impact**: Niche but valuable for transportation/accessibility
- **What to add**:
  - Least-cost path (Dijkstra, A*)
  - Service areas (isochrones)
  - Cumulative viewshed
- **Effort**: Medium
- **ROI**: Low-medium.

## Recommended Sequence

```
Now:     QGIS plugin (distribution) → immediate visibility
Q3 2026: Smelt-ML geostatistics (kriging + regression-kriging)
Q4 2026: LiDAR support (LAS/LAZ → DEM pipeline)
2027:    Full CRS engine + vector geoprocessing
```

## SurtGIS Algorithm Count by Category

| Category | SurtGIS | SAGA (approx) |
|----------|:-------:|:-------------:|
| Terrain | 40+ | ~80 |
| Hydrology | 25+ | ~60 |
| Imagery/Spectral | 20+ | ~40 |
| Interpolation | 10+ | ~30 |
| Morphology | 10+ | ~20 |
| Statistics | 15+ | ~30 |
| Climate/Cloud | 10+ | 0 (no cloud-native) |
| Vector | 5 | ~80 |
| LiDAR | 0 | ~30 |
| Geostatistics | 3 (IDW, Spline, Kriging) | ~40 |
| Network | 1 (viewshed) | ~20 |
| **Total** | **~136** | **~800** |
