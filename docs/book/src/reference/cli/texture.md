# `surtgis texture`

_Texture analysis: edge detection and GLCM texture measures_

## Overview

```text
Texture analysis: edge detection and GLCM texture measures

Usage: surtgis texture [OPTIONS] <COMMAND>

Commands:
  glcm       GLCM texture (Haralick): energy, contrast, homogeneity, correlation, entropy
  sobel      Sobel edge detection (gradient magnitude)
  laplacian  Laplacian edge detection (second derivative)
  help       Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `texture glcm` {#glcm}

```text
GLCM texture (Haralick): energy, contrast, homogeneity, correlation, entropy

Usage: surtgis texture glcm [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output texture raster

Options:
  -t, --texture <TEXTURE>        Texture measure: energy, contrast, homogeneity, correlation, entropy, dissimilarity [default: contrast]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          Window radius [default: 3]
      --levels <LEVELS>          Quantization levels [default: 32]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `texture sobel` {#sobel}

```text
Sobel edge detection (gradient magnitude)

Usage: surtgis texture sobel [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output edge raster

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `texture laplacian` {#laplacian}

```text
Laplacian edge detection (second derivative)

Usage: surtgis texture laplacian [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output edge raster

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

