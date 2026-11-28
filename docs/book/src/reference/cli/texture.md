# `surtgis texture`

_Texture analysis: edge detection and GLCM texture measures_

## Overview

```text
Texture analysis: edge detection and GLCM texture measures

Usage: surtgis texture [OPTIONS] <COMMAND>

Commands:
  glcm       GLCM texture (Haralick): energy, contrast, homogeneity, correlation, entropy
  glcm-all   GLCM all 6 textures in a single pass (one output file per feature)
  lbp        Local Binary Patterns (Ojala 2002)
  sobel      Sobel edge detection (gradient magnitude)
  laplacian  Laplacian edge detection (second derivative)
  help       Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `texture glcm` {#glcm}

```text
GLCM texture (Haralick): energy, contrast, homogeneity, correlation, entropy

Usage: surtgis texture glcm [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output texture raster

Options:
  -t, --texture <TEXTURE>            Texture measure: energy, contrast, homogeneity, correlation, entropy, dissimilarity [default: contrast]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>              Window radius [default: 3]
      --levels <LEVELS>              Quantization levels [default: 32]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `texture glcm-all` {#glcm-all}

```text
GLCM all 6 textures in a single pass (one output file per feature)

Usage: surtgis texture glcm-all [OPTIONS] <INPUT> <OUTPUT_DIR>

Arguments:
  <INPUT>       Input raster
  <OUTPUT_DIR>  Output directory (one .tif per feature is written here)

Options:
  -r, --radius <RADIUS>              Window radius [default: 3]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --levels <LEVELS>              Quantization levels [default: 32]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `texture lbp` {#lbp}

```text
Local Binary Patterns (Ojala 2002)

Usage: surtgis texture lbp [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output LBP raster

Options:
  -v, --verbose                      Verbose output
      --variant <VARIANT>            Variant: standard (0..255 raw code) or riu2 (rotation-invariant uniform) [default: standard]
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `texture sobel` {#sobel}

```text
Sobel edge detection (gradient magnitude)

Usage: surtgis texture sobel [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output edge raster

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `texture laplacian` {#laplacian}

```text
Laplacian edge detection (second derivative)

Usage: surtgis texture laplacian [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output edge raster

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

