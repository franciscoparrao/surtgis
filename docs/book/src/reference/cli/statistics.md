# `surtgis statistics`

_Statistics: focal, zonal, and spatial autocorrelation_

## Overview

```text
Statistics: focal, zonal, and spatial autocorrelation

Usage: surtgis statistics [OPTIONS] <COMMAND>

Commands:
  focal         Focal (moving window) statistics
  zonal         Zonal statistics per zone, written as JSON (or CSV)
  zonal-raster  Zonal statistics as raster (each cell = zone statistic)
  morans-i      Global Moran's I spatial autocorrelation (prints result)
  getis-ord     Local Getis-Ord Gi* hotspot analysis
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

## `statistics focal` {#focal}

```text
Focal (moving window) statistics

Usage: surtgis statistics focal [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output raster

Options:
  -s, --stat <STAT>                  Statistic: mean, std, min, max, range, sum, count, median [default: mean]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>              Window radius (size = 2r+1) [default: 3]
      --circular                     Use circular window instead of square
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `statistics zonal` {#zonal}

```text
Zonal statistics per zone, written as JSON (or CSV)

Zone 0 is background and is skipped unless --include-zero is given; cells equal to the zone raster's nodata and NaN values are always skipped.

Usage: surtgis statistics zonal [OPTIONS] --zones <ZONES> <INPUT> <OUTPUT>

Arguments:
  <INPUT>
          Input values raster

  <OUTPUT>
          Output file: JSON, or CSV when the name ends in .csv

Options:
  -v, --verbose
          Verbose output

      --zones <ZONES>
          Zone raster (integer classes)

      --compress
          Compress output GeoTIFFs (deflate)

      --format <FORMAT>
          Output format; by default taken from the output extension (.csv → csv, anything else → json)
          
          [possible values: json, csv]

      --include-zero
          Report zone 0 as a zone instead of treating it as background

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN)
          
          [default: f64]
          [possible values: f32, f64]

      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1

  -h, --help
          Print help (see a summary with '-h')
```

## `statistics zonal-raster` {#zonal-raster}

```text
Zonal statistics as raster (each cell = zone statistic)

Zone 0 is background (NaN in the output) unless --include-zero.

Usage: surtgis statistics zonal-raster [OPTIONS] --zones <ZONES> <INPUT> <OUTPUT>

Arguments:
  <INPUT>
          Input values raster

  <OUTPUT>
          Output raster

Options:
  -v, --verbose
          Verbose output

      --zones <ZONES>
          Zone raster (integer classes)

      --compress
          Compress output GeoTIFFs (deflate)

  -s, --stat <STAT>
          Statistic: mean, std, min, max, range, sum, count, median
          
          [default: mean]

      --include-zero
          Map zone 0 as a zone instead of treating it as background

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN)
          
          [default: f64]
          [possible values: f32, f64]

      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1

  -h, --help
          Print help (see a summary with '-h')
```

## `statistics morans-i` {#morans-i}

```text
Global Moran's I spatial autocorrelation (prints result)

Usage: surtgis statistics morans-i [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input raster

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `statistics getis-ord` {#getis-ord}

```text
Local Getis-Ord Gi* hotspot analysis

Usage: surtgis statistics getis-ord [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output z-scores raster

Options:
  -r, --radius <RADIUS>              Neighborhood radius [default: 3]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

