# `surtgis cog`

_Read and process Cloud Optimized GeoTIFFs (COGs) via HTTP_

## Overview

```text
Read and process Cloud Optimized GeoTIFFs (COGs) via HTTP

Usage: surtgis cog [OPTIONS] <COMMAND>

Commands:
  info        Show metadata of a remote COG
  fetch       Read a bounding box from a remote COG and save to local file
  slope       Calculate slope from a remote COG DEM
  aspect      Calculate aspect from a remote COG DEM
  hillshade   Calculate hillshade from a remote COG DEM
  tpi         Calculate TPI from a remote COG DEM
  fill-sinks  Fill sinks from a remote COG DEM
  help        Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `cog info` {#info}

```text
Show metadata of a remote COG

Usage: surtgis cog info [OPTIONS] <URL>

Arguments:
  <URL>  URL of the COG file

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `cog fetch` {#fetch}

```text
Read a bounding box from a remote COG and save to local file

Usage: surtgis cog fetch [OPTIONS] --bbox <BBOX> <URL> <OUTPUT>

Arguments:
  <URL>     URL of the COG file
  <OUTPUT>  Output GeoTIFF file

Options:
      --bbox <BBOX>              Bounding box: min_x,min_y,max_x,max_y
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --overview <OVERVIEW>      Overview level (0 = full resolution)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `cog slope` {#slope}

```text
Calculate slope from a remote COG DEM

Usage: surtgis cog slope [OPTIONS] --bbox <BBOX> <URL> <OUTPUT>

Arguments:
  <URL>     URL of the COG DEM
  <OUTPUT>  Output file

Options:
      --bbox <BBOX>              Bounding box: min_x,min_y,max_x,max_y
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -u, --units <UNITS>            Output units: degrees, percent, radians [default: degrees]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
  -z, --z-factor <Z_FACTOR>      Z-factor for unit conversion [default: 1.0]
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `cog aspect` {#aspect}

```text
Calculate aspect from a remote COG DEM

Usage: surtgis cog aspect [OPTIONS] --bbox <BBOX> <URL> <OUTPUT>

Arguments:
  <URL>     URL of the COG DEM
  <OUTPUT>  Output file

Options:
      --bbox <BBOX>              Bounding box: min_x,min_y,max_x,max_y
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -f, --format <FORMAT>          Output format: degrees, radians, compass [default: degrees]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `cog hillshade` {#hillshade}

```text
Calculate hillshade from a remote COG DEM

Usage: surtgis cog hillshade [OPTIONS] --bbox <BBOX> <URL> <OUTPUT>

Arguments:
  <URL>     URL of the COG DEM
  <OUTPUT>  Output file

Options:
      --bbox <BBOX>              Bounding box: min_x,min_y,max_x,max_y
  -v, --verbose                  Verbose output
  -a, --azimuth <AZIMUTH>        Sun azimuth in degrees (0=North, clockwise) [default: 315]
      --compress                 Compress output GeoTIFFs (deflate)
  -l, --altitude <ALTITUDE>      Sun altitude in degrees above horizon [default: 45]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -z, --z-factor <Z_FACTOR>      Z-factor for vertical exaggeration [default: 1.0]
  -h, --help                     Print help
```

## `cog tpi` {#tpi}

```text
Calculate TPI from a remote COG DEM

Usage: surtgis cog tpi [OPTIONS] --bbox <BBOX> <URL> <OUTPUT>

Arguments:
  <URL>     URL of the COG DEM
  <OUTPUT>  Output file

Options:
      --bbox <BBOX>              Bounding box: min_x,min_y,max_x,max_y
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 1]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `cog fill-sinks` {#fill-sinks}

```text
Fill sinks from a remote COG DEM

Usage: surtgis cog fill-sinks [OPTIONS] --bbox <BBOX> <URL> <OUTPUT>

Arguments:
  <URL>     URL of the COG DEM
  <OUTPUT>  Output file

Options:
      --bbox <BBOX>              Bounding box: min_x,min_y,max_x,max_y
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --min-slope <MIN_SLOPE>    Minimum slope to enforce [default: 0.01]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

