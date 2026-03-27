# SurtGIS 0.4.0: Interactive Web Platform

**Release Date:** March 27, 2026

## Overview

SurtGIS 0.4.0 transforms the project from a CLI tool into a **full interactive web platform**. Run 27 geospatial algorithms in your browser, browse 113+ STAC catalogs, download satellite data, and analyze it — all without installing anything.

**Live Demo:** https://surtgis-demo.ngrok-free.dev

## What's New

### 1. Interactive Web Demo (WASM)

- **27 algorithms** compiled to WebAssembly (581 KB binary)
- **Zero installation** — runs in any modern browser (Chrome, Firefox, Safari)
- **Mobile-friendly** — tested on phones and low-resource computers
- Upload your own GeoTIFF or use the built-in Andes demo DEM

**Terrain (15):** Slope, Aspect, Hillshade, Multi-Hillshade, TPI, TRI, Northness, Eastness, Curvature (general/profile/plan), DEV, Geomorphons, Shape Index, Curvedness

**Hydrology (6):** Fill Depressions, Priority Flood, Flow Direction, Flow Accumulation, TWI, HAND

**Morphology (4):** Erosion, Dilation, Opening, Closing

**Statistics (3):** Focal Mean, Focal Std Dev, Focal Range

### 2. STAC Browser in the Web UI

- **113+ catalogs** searchable from stacindex.org
- **Quick access** to Planetary Computer, Earth Search, Copernicus DEM, Sentinel-2, Landsat
- Browse collections, search items by **bbox drawn on map** + date range
- Download COG assets directly into the map viewer
- **Auto-signing** for Planetary Computer SAS tokens
- **S3 URL conversion** for AWS-hosted assets
- **File size check** before download (max 100 MB in browser, CLI for larger)
- **CORS proxy** built into the demo server for any STAC API
- **RFC3339 datetime** compliance for all STAC APIs

### 3. GIS Viewer Features

- **Leaflet + OpenStreetMap** (no API key required)
- **Layer manager** (QGIS-style): each result is an independent layer
  - Toggle visibility per layer
  - Reorder layers (move up/down arrows)
  - Remove individual layers
  - Clear all
- **4 basemaps:** OpenStreetMap, Satellite (ESRI), Topographic, Dark (CartoDB)
- **Graticule** (coordinate grid): adaptive interval by zoom, labeled
- **Opacity slider** for raster layers
- **Pixel value query**: hover mouse → see raster value in status bar
- **Mouse coordinates**: lat/lon displayed in real-time

### 4. CRS & Projection Support

- **Automatic EPSG detection** from GeoTIFF GeoKeys
- **UTM reprojection** (all 120 zones, north + south) via proj4js
- **Web Mercator** (EPSG:3857) support
- **Fallback bbox** from STAC metadata for files without affine transform (e.g., Sentinel-1 GRD)

### 5. Colormaps

6 colormaps with proper interpolation: Viridis, Plasma, Terrain, Grayscale, Seismic, HSV. Auto-selection per algorithm type.

## Architecture

```
Browser
├── index.html          — UI (Leaflet + Tailwind CSS)
├── app.js              — WASM orchestration, layer manager, STAC UI
├── stac.js             — STAC client (catalog search, collections, items, COG download)
├── server.py           — HTTP server with CORS proxy for STAC APIs
├── wasm/
│   └── surtgis_wasm.js + .wasm  — 27 algorithms compiled from Rust
└── data/
    └── demo_dem.tif    — 512×512 Andes sample (445 KB)
```

## WASM Compilation Fixes

- Fixed `ParallelSliceMut` sequential fallback in `maybe_rayon.rs` for non-parallel builds
- Fixed rayon trait imports for `wasm32-unknown-unknown` target
- Synchronized all crate versions across workspace

## Performance

| Operation | 512×512 | Notes |
|-----------|---------|-------|
| WASM load | <1s | 581 KB binary |
| Slope | <0.5s | |
| Hillshade | <0.5s | |
| Flow Accumulation | ~2s | Fill + flow dir + accumulation |
| TWI | ~3s | Full hydrological pipeline |
| COG download | 2-10s | Depends on file size and server |

## Backward Compatibility

- All CLI commands unchanged
- All 136+ native algorithms unchanged
- Web demo is additive (new `surtgis-demo/` directory)
- `demo.sh` script for one-command startup with ngrok

## Known Limitations

- Browser memory: max ~100 MB files (use CLI for larger)
- Single-threaded WASM (no Rayon in browser)
- Sentinel-1 GRD from Earth Search are raw TIFFs (687 MB), use Planetary Computer `sentinel-1-rtc` instead
- Some STAC APIs may require specific authentication not yet supported

## Commits

```
98a0d5c Add interactive web demo for SurtGIS v0.3.0
0a55a05 Add development guide for web demo
492bbbd Add demo DEM (Andes 512x512) and one-click load button
[current] v0.4.0: Web platform with STAC browser, layer manager, 27 WASM algorithms
```

## What's Next (v0.5.0)

- [ ] COG overview reading (load only thumbnail for large files)
- [ ] Multi-band visualization (RGB composite from 3 bands)
- [ ] NDVI/NDWI from two downloaded bands
- [ ] Save/load workspace (layers + settings)
- [ ] Vector overlay (GeoJSON polygons on map)
- [ ] Batch processing (multiple algorithms in sequence)
