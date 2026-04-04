# SurtGIS 0.5.0: Production-Ready Satellite Composite Pipeline

**Release Date:** April 4, 2026

## Overview

SurtGIS 0.5.0 makes the satellite composite pipeline production-ready. The COG reader now correctly handles DEFLATE compression with horizontal differencing predictor — the format used by 90%+ of modern Cloud-Optimized GeoTIFFs including all Sentinel-2 data on Planetary Computer. This was the root cause of all composite artifacts since v0.3.0.

## Critical Fix: COG Horizontal Differencing Predictor

The CogReader decompressed DEFLATE tiles correctly but never applied the TIFF Predictor=2 inverse (horizontal differencing). This produced values ~10× too high and visible "interference" patterns in all composites.

**Root cause:** TIFF Predictor=2 stores pixel differences as uint16 samples. The undo must accumulate as u16 with carry propagation between low and high bytes. Byte-level accumulation loses carry, corrupting values.

**Fix:** Sample-level accumulation for uint16 (and uint32 for float data).

**Before:** mean=33717, gradient=29398, range 0-65535
**After:** mean=3041, gradient=<100, range 0-10000 (correct reflectance)

## Satellite Composite Pipeline (Fixed End-to-End)

| Component | v0.4.0 | v0.5.0 |
|-----------|--------|--------|
| COG reader | Values 10× inflated | Correct (Predictor=2 support) |
| STAC search | Truncated at 200 items | 1000+ items (all MGRS tiles) |
| SAS signing | Rate-limited (1/7 dates) | Throttle 200ms + retry 5× backoff |
| Cloud mask | Discarded 99.7% (SCL=0) | SCL=0 passes through (no cloud info = assume clear) |
| Align-to | Post-composite resample (often 0%) | Direct write to DEM grid |
| Gap-fill | None | 3-phase: median → temporal nearest → spatial 3×3 |
| Resample | NaN at tile boundaries → stripes | NaN-tolerant bilinear (weighted valid neighbors) |
| 15-bit COG | Error | Supported (treat as uint16) |
| S2 nodata | No filtering | DN=0 → NaN |
| Output | Corrupt file (0-100KB) | Valid GeoTIFF, pixel-perfect with DEM |

## Validated on Real Data

- **Río Salado** (Atacama, 26°S): 100% coverage, values 0-4680, mean=3041
- **Río Huasco** (28°S, 14,921 km²): 57% coverage, pixel-perfect alignment to 30m DEM
- 13/13 dates in composite, 4 MGRS tiles per date, DEFLATE+Predictor=2

## Other New Features

### Machine Learning Integration (smelt-ml)

```bash
surtgis ml extract-samples --features features/ --points landslides.geojson --target class samples.csv
surtgis ml train samples.csv --target class --learner random_forest -o model.json
surtgis ml predict --features features/ --model model.json susceptibility.tif
surtgis ml benchmark samples.csv --target class --learners random_forest,gradient_boosting,xgboost
```

10 learners, cross-validation, feature importance, model serialization. First GIS tool with native ML.

### Feature Engineering Pipeline

```bash
surtgis pipeline features dem.tif --outdir features/
```

25+ geomorphometric variables in one command. Ready for scikit-learn, XGBoost, or smelt-ml.

### Python Bindings: 95 Functions

42 → 95 PyO3 functions covering terrain, hydrology, imagery (17 indices), classification, interpolation, landscape ecology, texture, and morphology.

### Web Demo Enhancements

- STAC browser with RGB composite and NDVI computation
- Vector overlay with zonal statistics and clip
- Workspace save/load
- COG overview preview

## WoS Bibliometric Analysis

Analyzed 19,062 papers across 15 queries. Key finding: ML is #1 method (952 mentions) and needs geospatial features as input — exactly what SurtGIS generates.

## Bug Fixes (17 fixes in composite pipeline)

1. SAS token signing retry with exponential backoff
2. 15-bit COG support (bits_per_sample=15)
3. Cross-UTM tile handling (per-tile EPSG tracking)
4. STAC search limit increased (200 → 1000+ items)
5. Strip bbox padding for tile edge coverage
6. Direct write to DEM grid (bypass post-composite resample)
7. S2 nodata filtering (DN=0 → NaN)
8. SCL class 0 passthrough (no cloud info = assume clear)
9. 3-phase gap-fill (median + temporal + spatial)
10. NaN-tolerant bilinear resample
11. Per-date diagnostic logging (tiles, signing, coverage)
12. File size verification after composite write
13. Composite extent logging for alignment debugging
14. COG tile assembly stats (tiles written/skipped)
15. SCL histogram for cloud mask debugging
16. TIFF Predictor tag (317) reading from IFD
17. **Horizontal differencing predictor undo (sample-level u16 accumulation)**

## Commits Since v0.4.0

34 commits addressing composite pipeline reliability, ML integration, Python bindings, and web demo features.

## What's Next (v0.6.0)

- [ ] Multi-temporal analysis (time series + change detection)
- [ ] Speckle filter for SAR data
- [ ] QGIS plugin (via Python bindings)
- [ ] Export workspace as Python script
- [ ] Object-based classification
