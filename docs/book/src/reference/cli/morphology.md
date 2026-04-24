# `surtgis morphology`

_Mathematical morphology algorithms_

## Overview

```text
Mathematical morphology algorithms

Usage: surtgis morphology [OPTIONS] <COMMAND>

Commands:
  erode      Erosion (minimum filter)
  dilate     Dilation (maximum filter)
  opening    Opening (erosion then dilation) — removes small bright features
  closing    Closing (dilation then erosion) — removes small dark features
  gradient   Morphological gradient (dilation - erosion) — edge detection
  top-hat    Top-hat transform (original - opening) — bright feature extraction
  black-hat  Black-hat transform (closing - original) — dark feature extraction
  help       Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `morphology erode` {#erode}

```text
Erosion (minimum filter)

Usage: surtgis morphology erode [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output file

Options:
      --shape <SHAPE>            Structuring element shape: square, cross, disk [default: square]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          Structuring element radius in cells [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `morphology dilate` {#dilate}

```text
Dilation (maximum filter)

Usage: surtgis morphology dilate [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output file

Options:
      --shape <SHAPE>            [default: square]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `morphology opening` {#opening}

```text
Opening (erosion then dilation) — removes small bright features

Usage: surtgis morphology opening [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output file

Options:
      --shape <SHAPE>            [default: square]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `morphology closing` {#closing}

```text
Closing (dilation then erosion) — removes small dark features

Usage: surtgis morphology closing [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output file

Options:
      --shape <SHAPE>            [default: square]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `morphology gradient` {#gradient}

```text
Morphological gradient (dilation - erosion) — edge detection

Usage: surtgis morphology gradient [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output file

Options:
      --shape <SHAPE>            [default: square]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `morphology top-hat` {#top-hat}

```text
Top-hat transform (original - opening) — bright feature extraction

Usage: surtgis morphology top-hat [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output file

Options:
      --shape <SHAPE>            [default: square]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `morphology black-hat` {#black-hat}

```text
Black-hat transform (closing - original) — dark feature extraction

Usage: surtgis morphology black-hat [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output file

Options:
      --shape <SHAPE>            [default: square]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

