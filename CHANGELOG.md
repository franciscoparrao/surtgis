# Changelog

All notable changes to SurtGIS will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

**Desktop GUI** (`surtgis-gui`)
- egui-based desktop application with SAGA-style workspace layout
- 6-panel dock layout: Algo Tree, Properties, Map Canvas, Layers, Output Log, Data Manager
- 105 algorithms accessible from GUI across 9 categories
- Tools menu with all algorithms organized by category
- Scale bar, coordinate display, zoom controls
- Basemap tiles via walkers (OpenStreetMap)
- STAC catalog browser with search, bbox, date filters
- 3D wireframe DEM visualization (isometric projection)
- Tiled renderer for large rasters (LRU cache, 512x512 tiles)
- Automatic colormap selection based on algorithm type

**Algorithms** (`surtgis-algorithms`)
- **Terrain** (30): slope, aspect, hillshade, curvature (plan/profile/tangential), TPI, TRI,
  roughness, geomorphons, convergence index, TWI, SPI, STI, LS factor, relative heights,
  wind exposition, sky view factor, contour lines, cost distance, and more
- **Imagery** (22): NDVI, NDWI, SAVI, NDSI, NDBI, NDMI, MSAVI, EVI2, band math,
  reclassify, histogram equalize, raster difference, change vector analysis, and more
- **Hydrology** (18): fill sinks, flow direction (D8), flow accumulation, watershed,
  stream network, strahler order, flow path length, isobasins, flood fill simulation,
  stream power index, topographic wetness, and more
- **Statistics** (11): focal mean/min/max/range/stddev/median/majority/diversity/percentile,
  zonal statistics
- **Morphology** (7): erosion, dilation, opening, closing, gradient, top-hat, black-hat
- **Interpolation** (6): IDW, nearest neighbor, TIN, natural neighbor, spline, kriging
- **Classification** (5): PCA, K-means, ISODATA, minimum distance, maximum likelihood
- **Landscape** (3): Shannon diversity, Simpson index, patch density
- **Texture** (3): Haralick GLCM (6 measures), Sobel edge, Laplacian

**Colormap** (`surtgis-colormap`)
- 8 color schemes: Terrain, Viridis, Magma, Grayscale, NDVI, BlueWhiteRed,
  Geomorphons, Water, Accumulation
- Auto parameter detection from raster statistics
- PNG rendering with legend

**Cloud** (`surtgis-cloud`)
- Cloud Optimized GeoTIFF (COG) reader with HTTP range requests
- STAC API client (search, pagination, asset download)
- Support for Planetary Computer and Earth Search catalogs
- Automatic WGS84-to-UTM bbox reprojection

**WASM** (`surtgis-wasm`)
- 33 algorithms compiled to WebAssembly
- npm package (`surtgis`)
- Web Worker integration for non-blocking computation

**Python** (`surtgis-python`)
- PyO3 bindings for core raster operations
- GeoTIFF read/write via Python API

**CLI** (`surtgis-cli`)
- Command-line interface for all algorithms
- GeoTIFF I/O, batch processing

**Core** (`surtgis-core`)
- `Raster<T>` generic raster type with nodata handling
- GeoTransform with named fields
- GeoTIFF reader/writer (TIFF + geospatial metadata)
- Parallel computation via rayon (optional)

**Cross-validation**
- Automated comparison tests: SurtGIS vs GDAL 3.11 and GRASS 8.3
- Validates slope, aspect, hillshade, flow accumulation, watershed results
- Tolerance-based comparison with statistical summaries

### Fixed
- Borrow checker error in ISODATA clustering (indexed loop instead of iterator)
- GeoTransform field access (named fields instead of array indexing)
- Error type conversion between surtgis_core::Error and anyhow::Error
- All clippy warnings resolved across workspace

## [0.0.1] - 2024-12-01

### Added
- Initial project structure with workspace layout
- Basic raster type and GeoTIFF support
- First terrain algorithms (slope, aspect, hillshade)
- Hydrology module (fill sinks, flow direction, flow accumulation)
