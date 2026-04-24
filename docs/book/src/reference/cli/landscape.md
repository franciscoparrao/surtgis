# `surtgis landscape`

_Landscape ecology metrics (global patch/class/landscape level)_

## Overview

```text
Landscape ecology metrics (global patch/class/landscape level)

Usage: surtgis landscape [OPTIONS] <COMMAND>

Commands:
  label-patches      Label connected patches in a classification raster
  patch-metrics      Compute per-patch metrics (PARA, FRAC, area, perimeter) as CSV
  class-metrics      Compute per-class metrics (AI, COHESION, proportion)
  landscape-metrics  Compute global landscape metrics (SHDI, SIDI)
  analyze            Full landscape analysis: label + patch metrics + class metrics + landscape metrics
  help               Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `landscape label-patches` {#label-patches}

```text
Label connected patches in a classification raster

Usage: surtgis landscape label-patches [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input classification raster (integer class values)
  <OUTPUT>  Output labeled patch raster (i32 IDs)

Options:
  -c, --connectivity <CONNECTIVITY>  Connectivity: 4 (cardinal) or 8 (cardinal+diagonal) [default: 4]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                         Print help
```

## `landscape patch-metrics` {#patch-metrics}

```text
Compute per-patch metrics (PARA, FRAC, area, perimeter) as CSV

Usage: surtgis landscape patch-metrics [OPTIONS] --output <OUTPUT> <INPUT>

Arguments:
  <INPUT>  Input classification raster

Options:
  -o, --output <OUTPUT>              Output CSV file with per-patch metrics
  -v, --verbose                      Verbose output
  -c, --connectivity <CONNECTIVITY>  Connectivity: 4 or 8 [default: 4]
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                         Print help
```

## `landscape class-metrics` {#class-metrics}

```text
Compute per-class metrics (AI, COHESION, proportion)

Usage: surtgis landscape class-metrics [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input classification raster

Options:
  -c, --connectivity <CONNECTIVITY>  Connectivity: 4 or 8 [default: 4]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                         Print help
```

## `landscape landscape-metrics` {#landscape-metrics}

```text
Compute global landscape metrics (SHDI, SIDI)

Usage: surtgis landscape landscape-metrics [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input classification raster

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `landscape analyze` {#analyze}

```text
Full landscape analysis: label + patch metrics + class metrics + landscape metrics

Usage: surtgis landscape analyze [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Input classification raster

Options:
      --output-labels <OUTPUT_LABELS>  Output labeled raster (optional)
  -v, --verbose                        Verbose output
      --compress                       Compress output GeoTIFFs (deflate)
      --output-csv <OUTPUT_CSV>        Output CSV with per-patch metrics (optional)
  -c, --connectivity <CONNECTIVITY>    Connectivity: 4 or 8 [default: 4]
      --streaming                      Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>        Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                           Print help
```

