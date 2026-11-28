# `surtgis extract`

_Extract raster values at point locations to CSV_

## Synopsis

```text
Extract raster values at point locations to CSV

Usage: surtgis extract [OPTIONS] --features-dir <FEATURES_DIR> --points <POINTS> --target <TARGET> <OUTPUT>

Arguments:
  <OUTPUT>  Output CSV file

Options:
      --features-dir <FEATURES_DIR>  Directory with features.json and feature rasters (from `pipeline features`)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --points <POINTS>              Vector file with labeled points (.geojson, .shp, .gpkg)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --target <TARGET>              Property name containing the target label/value
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```
