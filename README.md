# SurtGis

**High-performance geospatial analysis library written in Rust.**

SurtGis aims to provide a comprehensive set of geospatial algorithms — similar in scope to SAGA-GIS — leveraging Rust's performance, memory safety, and native parallelism.

## Features

- **Generic raster type** — `Raster<T>` supports `i8` through `f64` with zero-cost abstractions
- **Parallel by default** — algorithms use [Rayon](https://github.com/rayon-rs/rayon) for automatic multi-core execution
- **Tiled processing** — handles rasters larger than available memory
- **Native I/O** — reads/writes GeoTIFF without external dependencies; optional GDAL backend via feature flag
- **Modular workspace** — use only what you need (`core`, `algorithms`, `parallel`, or the full CLI)

## Architecture

```
surtgis/
├── surtgis-core         # Raster<T>, GeoTransform, CRS, I/O
├── surtgis-algorithms   # Terrain, hydrology, imagery, interpolation
├── surtgis-parallel     # Tiled processing, parallel strategies
└── surtgis (cli)        # Command-line interface
```

## Algorithms

### Terrain Analysis

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Slope | `terrain::slope` | Rate of change of elevation (Horn's method). Output in degrees, percent, or radians. |
| Aspect | `terrain::aspect` | Direction of steepest descent. Output in degrees, radians, or 8-direction compass. |
| Hillshade | `terrain::hillshade` | Shaded relief visualization with configurable sun position. |

### Hydrology

| Algorithm | Function | Description |
|-----------|----------|-------------|
| Fill Sinks | `hydrology::fill_sinks` | Depression filling using Planchon-Darboux (2001) method. |
| Flow Direction | `hydrology::flow_direction` | D8 single flow direction from filled DEM. |
| Flow Accumulation | `hydrology::flow_accumulation` | Upstream contributing area for each cell. |
| Watershed | `hydrology::watershed` | Watershed delineation from pour points. |

## Usage

### As a library

```rust
use surtgis_core::io::read_geotiff;
use surtgis_algorithms::terrain::{slope, SlopeParams, SlopeUnits};

fn main() -> anyhow::Result<()> {
    let dem = read_geotiff("dem.tif", None)?;

    let result = slope(&dem, SlopeParams {
        units: SlopeUnits::Degrees,
        z_factor: 1.0,
    })?;

    println!("Max slope: {:.1}°", result.statistics().max.unwrap());
    Ok(())
}
```

### As a CLI

```bash
# Raster info
surtgis info dem.tif

# Terrain analysis
surtgis terrain slope dem.tif slope.tif --units degrees
surtgis terrain aspect dem.tif aspect.tif
surtgis terrain hillshade dem.tif hillshade.tif --azimuth 315 --altitude 45
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
- [x] Terrain: slope, aspect, hillshade
- [x] Parallel processing with Rayon
- [ ] Hydrology: fill sinks, flow direction, flow accumulation, watersheds
- [ ] Terrain: curvatures, TPI, TRI, landform classification
- [ ] Imagery: spectral indices (NDVI, NDWI), band math
- [ ] Vector: buffer, clip, intersect, simplify
- [ ] Interpolation: IDW, kriging, splines
- [ ] Python bindings via PyO3
- [ ] WASM support for browser-based analysis

## License

MIT OR Apache-2.0
