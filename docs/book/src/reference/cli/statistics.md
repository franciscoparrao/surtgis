# `surtgis statistics`

_Statistics: focal, zonal, and spatial autocorrelation_

## Overview

```text
Statistics: focal, zonal, and spatial autocorrelation

Usage: surtgis statistics [OPTIONS] <COMMAND>

Commands:
  focal         Focal (moving window) statistics
  zonal         Zonal statistics (JSON output by zone)
  zonal-raster  Zonal statistics as raster (each cell = zone statistic)
  morans-i      Global Moran's I spatial autocorrelation (prints result)
  getis-ord     Local Getis-Ord Gi* hotspot analysis
  help          Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `statistics focal` {#focal}

```text
Focal (moving window) statistics

Usage: surtgis statistics focal [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output raster

Options:
  -s, --stat <STAT>              Statistic: mean, std, min, max, range, sum, count, median [default: mean]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          Window radius (size = 2r+1) [default: 3]
      --circular                 Use circular window instead of square
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `statistics zonal` {#zonal}

```text
Zonal statistics (JSON output by zone)

Usage: surtgis statistics zonal [OPTIONS] --zones <ZONES> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input values raster
  <OUTPUT>  Output JSON file

Options:
  -v, --verbose                  Verbose output
      --zones <ZONES>            Zone raster (integer classes)
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `statistics zonal-raster` {#zonal-raster}

```text
Zonal statistics as raster (each cell = zone statistic)

Usage: surtgis statistics zonal-raster [OPTIONS] --zones <ZONES> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input values raster
  <OUTPUT>  Output raster

Options:
  -v, --verbose                  Verbose output
      --zones <ZONES>            Zone raster (integer classes)
      --compress                 Compress output GeoTIFFs (deflate)
  -s, --stat <STAT>              Statistic: mean, std, min, max, range, sum, count, median [default: mean]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `statistics morans-i` {#morans-i}

```text
Global Moran's I spatial autocorrelation (prints result)

Usage: surtgis statistics morans-i [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input raster

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `statistics getis-ord` {#getis-ord}

```text
Local Getis-Ord Gi* hotspot analysis

Usage: surtgis statistics getis-ord [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output z-scores raster

Options:
  -r, --radius <RADIUS>          Neighborhood radius [default: 3]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

