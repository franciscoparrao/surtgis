# `surtgis reproject`

_Reproject a raster from one CRS to another (native, no GDAL dependency)_

## Synopsis

```text
Reproject a raster from one CRS to another (native, no GDAL dependency)

Usage: surtgis reproject [OPTIONS] --to <TO> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster file (GeoTIFF)
  <OUTPUT>  Output raster file (GeoTIFF)

Options:
      --to <TO>                  Target CRS (e.g. EPSG:32719 or just 32719)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --from <FROM>              Source CRS override (e.g. EPSG:4326); defaults to value embedded in input
      --method <METHOD>          Resampling method: nearest | bilinear (default) [default: bilinear]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --pixel-size <PIXEL_SIZE>  Output pixel size in target CRS units; defaults to a sensible auto-inferred value preserving approximate resolution
  -h, --help                     Print help
```
