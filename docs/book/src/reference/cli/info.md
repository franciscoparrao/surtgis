# `surtgis info`

_Show information about a raster file_

## Synopsis

```text
Show information about a raster file

Usage: surtgis info [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input raster file

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```
