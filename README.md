# SurtGIS

**High-performance geospatial analysis platform written in Rust.**

SurtGIS provides 105 geospatial algorithms across 9 categories, a desktop GUI, WASM support for browser-based analysis, and cloud data access via STAC/COG. Comparable in scope to SAGA-GIS, with Rust's performance, memory safety, and native parallelism.

## Highlights

- **105 algorithms** in 9 categories: terrain, hydrology, imagery, classification, texture, statistics, morphology, interpolation, landscape
- **Desktop GUI** — egui-based SAGA-style workspace with algorithm tree, basemap tiles, STAC browser, 3D view
- **WASM** — 33 algorithms compiled to WebAssembly, usable from JavaScript/TypeScript
- **Cloud** — COG reader with HTTP range requests, STAC client for Planetary Computer / Earth Search
- **Python bindings** — PyO3 interface for GeoTIFF I/O and raster processing
- **Parallel by default** — Rayon-based multi-core execution
- **No external dependencies** — native GeoTIFF reader/writer, no GDAL required

## Architecture

```
surtgis/
├── surtgis-core           # Raster<T>, GeoTransform, CRS, GeoTIFF I/O
├── surtgis-algorithms     # 105 algorithms in 10 modules
├── surtgis-parallel       # Tiled processing, parallel strategies
├── surtgis-gui            # egui desktop application
├── surtgis-cli            # Command-line interface
├── surtgis-cloud          # COG reader, STAC client
├── surtgis-wasm           # WebAssembly bindings (33 algos)
├── surtgis-python         # Python bindings (PyO3)
└── surtgis-colormap       # Color schemes and raster rendering
```

## Performance

Benchmarks on a 4096x4096 DEM (16.8M cells), release mode:

| Algorithm | SurtGIS | GDAL 3.11 | SAGA 9.9.1 | WhiteboxTools 2.4 | R / terra 1.8 |
|-----------|--------:|----------:|-----------:|-------------------:|--------------:|
| **Slope** | 187 ms | 516 ms (2.8x) | 3675 ms (19.6x) | 3937 ms (21.0x) | 640 ms (3.4x) |
| **Aspect** | 471 ms | 1232 ms (2.6x) | 3901 ms (8.3x) | 4254 ms (9.0x) | 928 ms (2.0x) |
| **Hillshade** | 444 ms | 616 ms (1.4x) | 2768 ms (6.2x) | 3217 ms (7.3x) | 3550 ms (8.0x) |

> Multipliers show `tool_time / surtgis_time`. SurtGIS times are pure computation; external tools include process startup and I/O.

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
