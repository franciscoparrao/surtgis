# `surtgis resample`

_Resample a raster to match the grid of a reference raster_

## Synopsis

```text
Resample a raster to match the grid of a reference raster

Usage: surtgis resample [OPTIONS] --reference <REFERENCE> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster to resample
  <OUTPUT>  Output resampled raster

Options:
      --reference <REFERENCE>    Reference raster defining the target grid
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -m, --method <METHOD>          Interpolation method: nearest or bilinear [default: bilinear]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```
