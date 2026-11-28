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
      --reference <REFERENCE>        Reference raster defining the target grid
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
  -m, --method <METHOD>              Interpolation method: nearest or bilinear [default: bilinear]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```
