# `surtgis mosaic`

_Mosaic multiple rasters into one covering the union extent_

## Synopsis

```text
Mosaic multiple rasters into one covering the union extent

Usage: surtgis mosaic [OPTIONS] --input <INPUT> <OUTPUT>

Arguments:
  <OUTPUT>  Output GeoTIFF file

Options:
  -i, --input <INPUT>            Input raster files (at least 2)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```
