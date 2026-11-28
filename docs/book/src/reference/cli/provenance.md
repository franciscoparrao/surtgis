# `surtgis provenance`

_Show the provenance record embedded in a SurtGIS output (engine, command, hashed inputs, output digest)_

## Synopsis

```text
Show the provenance record embedded in a SurtGIS output (engine, command, hashed inputs, output digest)

Usage: surtgis provenance [OPTIONS] <INPUT>

Arguments:
  <INPUT>  GeoTIFF/COG written by SurtGIS

Options:
      --json                         Print the raw JSON record (URL query strings removed)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --stac                         Print the record as STAC Processing extension v1.3 fields (`processing:*` properties and `derived_from` links to the STAC items read), to merge into a STAC Item describing the file
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```
