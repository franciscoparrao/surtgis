# CLI reference

Every top-level `surtgis` subcommand, with flags and a minimal example.
Pages are generated from `--help` output and regenerated on every
CLI-surface change.

## Top-level commands

- [`surtgis info`](./info.md) — Show information about a raster file
- [`surtgis provenance`](./provenance.md) — Show the provenance record embedded in a SurtGIS output (engine, command, hashed inputs, output digest)
- [`surtgis verify`](./verify.md) — Verify a SurtGIS output against its embedded provenance: recompute the digest of the pixel array and re-hash the recorded inputs. Exits non-zero on any mismatch
- [`surtgis terrain`](./terrain.md) — Terrain analysis algorithms
- [`surtgis hydrology`](./hydrology.md) — Hydrology algorithms
- [`surtgis fluvial`](./fluvial.md) — Fluvial-tectonic morphometry (chi, ksn, knickpoints, divide migration)
- [`surtgis imagery`](./imagery.md) — Imagery / spectral index algorithms
- [`surtgis morphology`](./morphology.md) — Mathematical morphology algorithms
- [`surtgis landscape`](./landscape.md) — Landscape ecology metrics (global patch/class/landscape level)
- [`surtgis extract`](./extract.md) — Extract raster values at point locations to CSV
- [`surtgis extract-patches`](./extract-patches.md) — Extract raster patches centered on points or sampled from polygons for CNN training
- [`surtgis relief`](./relief.md) — Rayshader-style shaded relief composite to PNG
- [`surtgis clip`](./clip.md) — Clip a raster by polygon or bounding box
- [`surtgis reproject`](./reproject.md) — Reproject a raster from one CRS to another (native, no GDAL dependency)
- [`surtgis rasterize`](./rasterize.md) — Rasterize a vector file to a raster grid (.geojson, .shp, .gpkg)
- [`surtgis resample`](./resample.md) — Resample a raster to match the grid of a reference raster
- [`surtgis mosaic`](./mosaic.md) — Mosaic multiple rasters into one covering the union extent
- [`surtgis cog`](./cog.md) — Read and process Cloud Optimized GeoTIFFs (COGs) via HTTP
- [`surtgis stac`](./stac.md) — Search and fetch data from STAC catalogs (Planetary Computer, Earth Search)
- [`surtgis pipeline`](./pipeline.md) — Pipeline: integrated workflows for specific use cases (cloud feature only)
- [`surtgis vector`](./vector.md) — Vector geoprocessing: intersection, union, difference, dissolve, buffer
- [`surtgis interpolation`](./interpolation.md) — Geostatistical interpolation: variogram, kriging, universal kriging, regression kriging
- [`surtgis temporal`](./temporal.md) — Temporal analysis: trend, anomaly, phenology, statistics
- [`surtgis classification`](./classification.md) — Classification: unsupervised/supervised raster classification (k-means, PCA, etc.)
- [`surtgis texture`](./texture.md) — Texture analysis: edge detection and GLCM texture measures
- [`surtgis embeddings`](./embeddings.md) — Foundation-model embeddings (AlphaEarth, TESSERA, your own encoder) as rasters: similarity maps, PCA false colour, norms
- [`surtgis segmentation`](./segmentation.md) — Image segmentation: SLIC superpixels and Felzenszwalb-Huttenlocher
- [`surtgis statistics`](./statistics.md) — Statistics: focal, zonal, and spatial autocorrelation

## Global flags

Available on every subcommand:

```text
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
  -V, --version                      Print version
```
