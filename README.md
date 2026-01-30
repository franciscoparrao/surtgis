# SurtGis

**High-performance geospatial analysis library written in Rust.**

SurtGis aims to provide a comprehensive set of geospatial algorithms — similar in scope to SAGA-GIS — leveraging Rust's performance, memory safety, and native parallelism.

## Features

- **Generic raster type** — `Raster<T>` supports `i8` through `f64` with zero-cost abstractions
- **Parallel by default** — algorithms use [Rayon](https://github.com/rayon-rs/rayon) for automatic multi-core execution
- **Tiled processing** — handles rasters larger than available memory
- **Native I/O** — reads/writes GeoTIFF without external dependencies; optional GDAL backend via feature flag
- **Modular workspace** — use only what you need (`core`, `algorithms`, `parallel`, or the full CLI)

## Performance

Benchmarks on a 4096x4096 DEM (16.8M cells), release mode:

| Algorithm | SurtGis | GDAL 3.11 | SAGA 9.9.1 | WhiteboxTools 2.4 | R / terra 1.8 |
|-----------|--------:|----------:|-----------:|-------------------:|--------------:|
| **Slope** | 187 ms | 516 ms (2.8x) | 3675 ms (19.6x) | 3937 ms (21.0x) | 640 ms (3.4x) |
| **Aspect** | 471 ms | 1232 ms (2.6x) | 3901 ms (8.3x) | 4254 ms (9.0x) | 928 ms (2.0x) |
| **Hillshade** | 444 ms | 616 ms (1.4x) | 2768 ms (6.2x) | 3217 ms (7.3x) | 3550 ms (8.0x) |

> Multipliers show `tool_time / surtgis_time` — higher means SurtGis is faster.
> SurtGis times are pure computation (in-memory); external tools include process startup and I/O.

Reproduce with:
```bash
cargo run -p surtgis-algorithms --example bench_comparison --release -- --size 4096
```

Criterion micro-benchmarks:
```bash
cargo bench -p surtgis-algorithms
```

## Architecture

```
surtgis/
├── surtgis-core         # Raster<T>, GeoTransform, CRS, I/O
├── surtgis-algorithms   # Terrain, hydrology, imagery, interpolation, vector, morphology
├── surtgis-parallel     # Tiled processing, parallel strategies
└── surtgis (cli)        # Command-line interface
```

## Algorithms

### Terrain Analysis

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Slope | `terrain::slope` | Rate of change of elevation (Horn 1981). Degrees, percent, or radians. |
| Aspect | `terrain::aspect` | Direction of steepest descent. Degrees, radians, or 8-direction compass. |
| Hillshade | `terrain::hillshade` | Shaded relief with configurable sun position. |
| Curvature | `terrain::curvature` | General, profile, and plan curvature (Zevenbergen & Thorne 1987). |
| TPI | `terrain::tpi` | Topographic Position Index with configurable neighborhood radius. |
| TRI | `terrain::tri` | Terrain Ruggedness Index (Riley 1999). |
| Landform | `terrain::landform_classification` | 11-class landform classification using multi-scale TPI + slope (Weiss 2001). |

### Hydrology

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Fill Sinks | `hydrology::fill_sinks` | Depression filling (Planchon-Darboux 2001). |
| Flow Direction | `hydrology::flow_direction` | D8 single flow direction. |
| Flow Accumulation | `hydrology::flow_accumulation` | Upstream contributing area. |
| Watershed | `hydrology::watershed` | Watershed delineation from pour points. |

### Imagery

| Algorithm | Function | Description |
|-----------|----------|-------------|
| NDVI | `imagery::ndvi` | Normalized Difference Vegetation Index |
| NDWI | `imagery::ndwi` | Normalized Difference Water Index |
| MNDWI | `imagery::mndwi` | Modified NDWI |
| NBR | `imagery::nbr` | Normalized Burn Ratio |
| SAVI | `imagery::savi` | Soil-Adjusted Vegetation Index |
| EVI | `imagery::evi` | Enhanced Vegetation Index |
| BSI | `imagery::bsi` | Bare Soil Index |
| Band Math | `imagery::band_math_binary` | Raster algebra (add, subtract, multiply, divide, power, min, max) |
| Reclassify | `imagery::reclassify` | Value reclassification by ranges |

### Interpolation

| Algorithm | Function | Description |
|-----------|----------|-------------|
| IDW | `interpolation::idw` | Inverse Distance Weighting (Shepard) |
| Nearest Neighbor | `interpolation::nearest_neighbor` | Nearest neighbor interpolation |
| TIN | `interpolation::tin_interpolation` | Triangulated Irregular Network (Bowyer-Watson + barycentric) |

### Vector Operations

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Buffer | `vector::buffer_points`, `buffer_geometry` | Circular and distance-based buffering |
| Simplify | `vector::simplify_dp`, `simplify_vw` | Douglas-Peucker and Visvalingam-Whyatt |
| Clip | `vector::clip_by_rect` | Rectangular clipping (Cohen-Sutherland / Sutherland-Hodgman) |
| Spatial Ops | `vector::centroid`, `convex_hull`, `bounding_box`, `dissolve` | Geometric operations |
| Measurements | `vector::area`, `length`, `perimeter` | Geometry measurements |

### Mathematical Morphology

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Erode | `morphology::erode` | Minimum filter (parallel) |
| Dilate | `morphology::dilate` | Maximum filter (parallel) |
| Opening | `morphology::opening` | Erosion then dilation — removes small bright features |
| Closing | `morphology::closing` | Dilation then erosion — removes small dark features |
| Gradient | `morphology::gradient` | Dilation minus erosion — edge detection |
| Top-hat | `morphology::top_hat` | Original minus opening — bright feature extraction |
| Black-hat | `morphology::black_hat` | Closing minus original — dark feature extraction |

Structuring elements: `Square`, `Cross`, `Disk`, `Custom` with arbitrary radius.

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
surtgis terrain aspect dem.tif aspect.tif
surtgis terrain hillshade dem.tif hillshade.tif --azimuth 315 --altitude 45
surtgis terrain curvature dem.tif curv.tif --curvature-type profile
surtgis terrain tpi dem.tif tpi.tif --radius 5
surtgis terrain tri dem.tif tri.tif
surtgis terrain landform dem.tif landform.tif --small-radius 3 --large-radius 10

# Hydrology
surtgis hydrology fill-sinks dem.tif filled.tif
surtgis hydrology flow-direction dem.tif fdir.tif
surtgis hydrology flow-accumulation fdir.tif facc.tif
surtgis hydrology watershed fdir.tif basins.tif --pour-points "100,200;300,400"

# Imagery
surtgis imagery ndvi --nir nir.tif --red red.tif ndvi.tif
surtgis imagery band-math -a band1.tif -b band2.tif result.tif --op subtract

# Morphology
surtgis morphology erode input.tif eroded.tif --shape disk --radius 2
surtgis morphology opening input.tif opened.tif
surtgis morphology gradient input.tif edges.tif
```

## Building

```bash
# Default (native TIFF I/O)
cargo build --release

# With GDAL support (requires libgdal-dev)
cargo build --release --features gdal
```

## Running tests

```bash
cargo test
```

## Roadmap

- [x] Core raster types and I/O
- [x] Terrain: slope, aspect, hillshade, curvature, TPI, TRI, landform classification
- [x] Hydrology: fill sinks, flow direction, flow accumulation, watersheds
- [x] Imagery: spectral indices (NDVI, NDWI, MNDWI, NBR, SAVI, EVI, BSI), band math, reclassify
- [x] Interpolation: IDW, nearest neighbor, TIN
- [x] Vector: buffer, simplify, clip, spatial operations, measurements
- [x] Mathematical morphology: erode, dilate, opening, closing, gradient, top-hat, black-hat
- [x] Parallel processing with Rayon
- [x] CLI with 26+ commands
- [x] Benchmarks vs GDAL, SAGA, WhiteboxTools, R/terra
- [ ] Python bindings via PyO3
- [ ] WASM support for browser-based analysis

## License

MIT OR Apache-2.0
