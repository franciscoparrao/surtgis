# `surtgis mosaic`

_Mosaic multiple rasters into one covering the union extent_

## Synopsis

```text
Mosaic multiple rasters into one covering the union extent

Usage: surtgis mosaic [OPTIONS] --input <INPUT> <OUTPUT>

Arguments:
  <OUTPUT>  Output GeoTIFF file

Options:
  -i, --input <INPUT>                Input raster files (at least 2)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```
