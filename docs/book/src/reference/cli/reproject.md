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
      --to <TO>                      Target CRS (e.g. EPSG:32719 or just 32719)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --from <FROM>                  Source CRS override (e.g. EPSG:4326); defaults to value embedded in input
      --method <METHOD>              Resampling method: nearest | bilinear (default) [default: bilinear]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --pixel-size <PIXEL_SIZE>      Output pixel size in target CRS units; defaults to a sensible auto-inferred value preserving approximate resolution
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```
