# CLI reference

Every top-level `surtgis` subcommand, with flags and a minimal example.
Pages are generated from `--help` output and regenerated on every
CLI-surface change.

## Top-level commands

- [`surtgis info`](./info.md) — Show information about a raster file
- [`surtgis terrain`](./terrain.md) — Terrain analysis algorithms
- [`surtgis hydrology`](./hydrology.md) — Hydrology algorithms
- [`surtgis imagery`](./imagery.md) — Imagery / spectral index algorithms
- [`surtgis morphology`](./morphology.md) — Mathematical morphology algorithms
- [`surtgis landscape`](./landscape.md) — Landscape ecology metrics (global patch/class/landscape level)
- [`surtgis extract`](./extract.md) — Extract raster values at point locations to CSV
- [`surtgis extract-patches`](./extract-patches.md) — Extract raster patches centered on points or sampled from polygons for CNN training
- [`surtgis clip`](./clip.md) — Clip a raster by polygon or bounding box
- [`surtgis rasterize`](./rasterize.md) — Rasterize a vector file to a raster grid (.geojson, .shp, .gpkg)
- [`surtgis resample`](./resample.md) — Resample a raster to match the grid of a reference raster
- [`surtgis mosaic`](./mosaic.md) — Mosaic multiple rasters into one covering the union extent
- [`surtgis cog`](./cog.md) — Read and process Cloud Optimized GeoTIFFs (COGs) via HTTP
- [`surtgis stac`](./stac.md) — Search and fetch data from STAC catalogs (Planetary Computer, Earth Search)
- [`surtgis pipeline`](./pipeline.md) — Pipeline: integrated workflows for specific use cases
- [`surtgis vector`](./vector.md) — Vector geoprocessing: intersection, union, difference, dissolve, buffer
- [`surtgis interpolation`](./interpolation.md) — Geostatistical interpolation: variogram, kriging, universal kriging, regression kriging
- [`surtgis temporal`](./temporal.md) — Temporal analysis: trend, anomaly, phenology, statistics
- [`surtgis classification`](./classification.md) — Classification: unsupervised/supervised raster classification (k-means, PCA, etc.)
- [`surtgis texture`](./texture.md) — Texture analysis: edge detection and GLCM texture measures
- [`surtgis statistics`](./statistics.md) — Statistics: focal, zonal, and spatial autocorrelation

## Global flags

Available on every subcommand:

```text
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
  -V, --version                  Print version
```
