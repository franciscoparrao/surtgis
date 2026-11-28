# `surtgis segmentation`

_Image segmentation: SLIC superpixels and Felzenszwalb-Huttenlocher_

## Overview

```text
Image segmentation: SLIC superpixels and Felzenszwalb-Huttenlocher

Usage: surtgis segmentation [OPTIONS] <COMMAND>

Commands:
  slic          SLIC superpixels (Achanta 2012)
  felzenszwalb  Felzenszwalb-Huttenlocher graph segmentation (2004)
  help          Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `segmentation slic` {#slic}

```text
SLIC superpixels (Achanta 2012)

Usage: surtgis segmentation slic [OPTIONS] --band <BANDS> <OUTPUT>

Arguments:
  <OUTPUT>  Output label raster (i32 GeoTIFF, 1..=N labels, 0 = nodata)

Options:
      --band <BANDS>                 One or more input band rasters (repeatable)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
  -n, --n-segments <N_SEGMENTS>      Target number of superpixels [default: 100]
  -m, --compactness <COMPACTNESS>    Compactness (higher = more geometric superpixels) [default: 10.0]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-iter <MAX_ITER>          Maximum k-means iterations [default: 10]
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --no-connectivity              Disable connectivity enforcement post-pass
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `segmentation felzenszwalb` {#felzenszwalb}

```text
Felzenszwalb-Huttenlocher graph segmentation (2004)

Usage: surtgis segmentation felzenszwalb [OPTIONS] --band <BANDS> <OUTPUT>

Arguments:
  <OUTPUT>  Output label raster (i32 GeoTIFF, 1..=N labels, 0 = nodata)

Options:
      --band <BANDS>                 One or more input band rasters (repeatable)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
  -k, --scale <SCALE>                Scale parameter k (higher = larger segments) [default: 1.0]
      --min-size <MIN_SIZE>          Minimum component size (smaller components are merged) [default: 20]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

