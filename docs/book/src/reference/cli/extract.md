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
  -h, --help                         Print help
```
