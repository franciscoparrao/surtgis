# `surtgis verify`

_Verify a SurtGIS output against its embedded provenance: recompute the digest of the pixel array and re-hash the recorded inputs. Exits non-zero on any mismatch_

## Synopsis

```text
Verify a SurtGIS output against its embedded provenance: recompute the digest of the pixel array and re-hash the recorded inputs. Exits non-zero on any mismatch

Usage: surtgis verify [OPTIONS] <INPUT>

Arguments:
  <INPUT>  GeoTIFF/COG written by SurtGIS

Options:
      --skip-inputs                  Only check the output digest, not the inputs
  -v, --verbose                      Verbose output
      --against <FILE>               File of affected STAC item ids, one per line (`#` comments and item URLs accepted), e.g. a provider's list of corrupted items. Fails when the output read any of them
      --compress                     Compress output GeoTIFFs (deflate)
      --affected-from <TIME>         With --against: only items accessed at or after this time (RFC 3339 or YYYY-MM-DD, UTC)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --affected-to <TIME>           With --against: only items accessed at or before this time (a bare date counts until the end of that day)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```
