# SurtGIS

**High-performance geospatial analysis library and CLI in Rust.**

136 algorithms, 90 CLI subcommands, streaming I/O for arbitrarily large DEMs, and an end-to-end satellite composite pipeline — all from a single binary with no external dependencies.

> **Quickstart**: [Análisis de terreno de los Andes en 5 minutos](https://territorio-digital.cl/blog/surtgis-quickstart-analisis-terreno)
>
> **🌐 Try it now**: [SurtGIS Web Demo](surtgis-demo/) — No installation, run in your browser!

## Quick example

```bash
cargo install surtgis

# Download DEM + compute all terrain factors
surtgis stac fetch-mosaic --collection cop-dem-glo-30 --bbox -70.5,-33.6,-70.2,-33.3 dem.tif
surtgis terrain all dem.tif --outdir factors/ --compress

# Cloud-free Sentinel-2 composite
surtgis stac composite --collection sentinel-2-l2a --asset red \
  --datetime 2024-01-01/2024-12-31 --bbox -70.5,-33.6,-70.2,-33.3 composite.tif

# Clip by watershed polygon
surtgis clip --polygon watershed.shp factors/slope.tif slope_clipped.tif
```

## Highlights

- **136 algorithms** — terrain, hydrology, imagery, landscape, classification, statistics, morphology, interpolation
- **90 CLI subcommands** — batch modes (`terrain all`, `hydrology all`), expression-based indices, landscape metrics
- **Streaming I/O** — 12 terrain algorithms process DEMs of any size with ~200MB RAM
- **STAC composite** — end-to-end satellite pipeline: search → mosaic → cloud-mask → median composite
- **Vector formats** — GeoJSON, Shapefile (.shp), GeoPackage (.gpkg) for clip and rasterize
- **Cloud native** — COG reader with HTTP range requests, STAC client for Planetary Computer / Earth Search
- **Cross-platform** — compiles to native (Rayon), WebAssembly (browser), Python (PyO3)
- **No external dependencies** — native GeoTIFF I/O with DEFLATE compression, no GDAL required

## Architecture

```
surtgis/
├── surtgis-core           # Raster<T>, GeoTransform, CRS, GeoTIFF I/O, streaming, mosaic, vector
├── surtgis-algorithms     # 136 algorithms in 10 modules
├── surtgis-parallel       # Rayon-based parallel strategies
├── surtgis-cli            # 90 CLI subcommands (14 modular files)
├── surtgis-cloud          # COG reader, STAC client, bbox reprojection
├── surtgis-wasm           # WebAssembly bindings (33 algos)
├── surtgis-python         # Python bindings (PyO3)
├── surtgis-gui            # egui desktop application
└── surtgis-colormap       # Color schemes and raster rendering
```

## Performance

Benchmarks on full GeoTIFF pipelines (read → compute → write), DEMs up to 20K² (400M cells):

| Algorithm | vs GDAL | vs GRASS GIS | vs WhiteboxTools |
|-----------|---------|-------------|-----------------|
| Slope | 1.7–1.8× faster | 4.5–4.9× | — |
| Aspect | 2.0× | 3.5–4.8× | 4.2–7.9× |
| Flow accumulation | — | 7.5–23.1× | 7.5–7.7× |
| Depression filling | — | — | 1.6–2.6× |

## Algorithms (105)

### Terrain Analysis (30)

Slope, aspect, hillshade, curvature (plan/profile/tangential), multiscale curvatures (Florinsky),
TPI, TRI, roughness, landform classification, geomorphons, TWI, SPI, STI, LS factor,
viewshed, sky view factor, openness, convergence index, wind exposure, solar radiation,
MRVBF/MRRTF, relative heights, feature-preserving smoothing, contour lines, cost distance.

### Imagery (22)

NDVI, NDWI, MNDWI, NBR, SAVI, EVI, BSI, NDSI, NDBI, NDMI, MSAVI, EVI2,
band math, reclassify, histogram equalize, normalize, threshold,
raster difference, change vector analysis.

### Hydrology (18)

Fill sinks, flow direction (D8), flow accumulation, watershed, stream network,
stream power index, topographic wetness, Strahler order, flow path length,
isobasins, flood fill simulation.

### Statistics (11)

Focal statistics (mean/min/max/range/stddev/median/majority/diversity/percentile),
zonal statistics, Global Moran's I, Local Getis-Ord Gi*.

### Morphology (7)

Erosion, dilation, opening, closing, gradient, top-hat, black-hat.
Structuring elements: square, cross, disk, custom.

### Interpolation (6)

IDW, nearest neighbor, TIN, natural neighbor, spline, kriging.

### Classification (5)

PCA, K-means, ISODATA, minimum distance, maximum likelihood.

### Landscape Ecology (3)

Shannon diversity, Simpson index, patch density.

### Texture (3)

Haralick GLCM (6 measures: contrast, correlation, energy, homogeneity, dissimilarity, entropy),
Sobel edge detection, Laplacian filter.

## Desktop GUI

The GUI provides a SAGA-style workspace:

- **Algorithm Tree** — browse 105 algorithms by category
- **Property Panel** — configure parameters with type-appropriate widgets
- **Map Canvas** — raster visualization with zoom/pan and automatic colormap
- **Layers Panel** — manage loaded datasets
- **Basemap** — OpenStreetMap tile overlay via walkers
- **STAC Browser** — search Planetary Computer / Earth Search catalogs
- **3D View** — wireframe DEM visualization with adjustable azimuth/elevation

```bash
cargo run -p surtgis-gui --release
```

## Usage

### As a library

```rust
use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};
use surtgis_algorithms::terrain::{slope, SlopeParams, SlopeUnits};

fn main() -> anyhow::Result<()> {
    let dem = read_geotiff("dem.tif", None)?;

    let result = slope(&dem, SlopeParams {
        units: SlopeUnits::Degrees,
        z_factor: 1.0,
    })?;

    println!("Max slope: {:.1}", result.statistics().max.unwrap());
    write_geotiff(&result, "slope.tif", Some(GeoTiffOptions::default()))?;
    Ok(())
}
```

### As a CLI

```bash
# Raster info
surtgis info dem.tif

# Terrain
surtgis terrain slope dem.tif slope.tif --units degrees
surtgis terrain hillshade dem.tif hillshade.tif --azimuth 315 --altitude 45
surtgis terrain curvature dem.tif curv.tif --curvature-type profile

# Hydrology
surtgis hydrology fill-sinks dem.tif filled.tif
surtgis hydrology flow-direction dem.tif fdir.tif
surtgis hydrology flow-accumulation fdir.tif facc.tif

# Imagery
surtgis imagery ndvi --nir nir.tif --red red.tif ndvi.tif
surtgis imagery band-math -a band1.tif -b band2.tif result.tif --op subtract

# Morphology
surtgis morphology erode input.tif eroded.tif --shape disk --radius 2
```

### WASM (browser)

```javascript
import init, { slope, ndvi } from 'surtgis';
await init();

const result = slope(demData, rows, cols, { units: 'degrees', zFactor: 1.0 });
```

## Building

```bash
# Default (native TIFF I/O)
cargo build --release

# With cloud support (COG + STAC)
cargo build --release --features cloud

# Desktop GUI
cargo run -p surtgis-gui --release

# WASM package
cd crates/wasm && wasm-pack build --target web
```

## Testing

```bash
# Full test suite (637 tests)
cargo test --workspace

# Cross-validation against GDAL/GRASS
cargo test --test cross_validation
```

## License

MIT OR Apache-2.0
