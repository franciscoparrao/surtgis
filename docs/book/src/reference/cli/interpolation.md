# `surtgis interpolation`

_Geostatistical interpolation: variogram, kriging, universal kriging, regression kriging_

## Overview

```text
Geostatistical interpolation: variogram, kriging, universal kriging, regression kriging

Usage: surtgis interpolation [OPTIONS] <COMMAND>

Commands:
  variogram           Compute empirical variogram and fit a theoretical model
  kriging             Ordinary Kriging interpolation from points to raster
  universal-kriging   Universal Kriging with polynomial drift
  regression-kriging  Regression-Kriging: ML prediction + kriging on residuals
  idw                 Inverse Distance Weighting interpolation
  nearest-neighbor    Nearest Neighbor interpolation (Voronoi/Thiessen)
  natural-neighbor    Natural Neighbor interpolation (Sibson)
  tps                 Thin Plate Spline interpolation
  tin                 TIN (Triangulated Irregular Network) interpolation
  help                Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `interpolation variogram` {#variogram}

```text
Compute empirical variogram and fit a theoretical model

Usage: surtgis interpolation variogram [OPTIONS] --attribute <ATTRIBUTE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile with a numeric attribute)
  <OUTPUT>  Output JSON file with variogram data and fitted model

Options:
      --attribute <ATTRIBUTE>    Attribute name containing the values to analyze
  -v, --verbose                  Verbose output
      --bins <BINS>              Number of lag bins (default: 15) [default: 15]
      --compress                 Compress output GeoTIFFs (deflate)
      --max-lag <MAX_LAG>        Maximum lag distance (auto-detect if omitted)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `interpolation kriging` {#kriging}

```text
Ordinary Kriging interpolation from points to raster

Usage: surtgis interpolation kriging [OPTIONS] --attribute <ATTRIBUTE> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>          Attribute name with values
  -v, --verbose                        Verbose output
      --compress                       Compress output GeoTIFFs (deflate)
      --reference <REFERENCE>          Reference raster for output grid (resolution, extent, CRS)
      --model <MODEL>                  Variogram model: spherical, exponential, gaussian, matern [default: spherical]
      --streaming                      Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>        Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --range <RANGE>                  Variogram range (auto-fit if omitted)
      --sill <SILL>                    Variogram sill (auto-fit if omitted)
      --nugget <NUGGET>                Variogram nugget (default: 0) [default: 0]
      --max-neighbors <MAX_NEIGHBORS>  Maximum neighbors for kriging (default: 16) [default: 16]
  -h, --help                           Print help
```

## `interpolation universal-kriging` {#universal-kriging}

```text
Universal Kriging with polynomial drift

Usage: surtgis interpolation universal-kriging [OPTIONS] --attribute <ATTRIBUTE> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>          Attribute name with values
  -v, --verbose                        Verbose output
      --compress                       Compress output GeoTIFFs (deflate)
      --reference <REFERENCE>          Reference raster for output grid
      --drift <DRIFT>                  Drift order: linear or quadratic [default: linear]
      --streaming                      Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>        Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --model <MODEL>                  Variogram model: spherical, exponential, gaussian [default: spherical]
      --max-neighbors <MAX_NEIGHBORS>  Maximum neighbors (default: 16) [default: 16]
  -h, --help                           Print help
```

## `interpolation regression-kriging` {#regression-kriging}

```text
Regression-Kriging: ML prediction + kriging on residuals

Usage: surtgis interpolation regression-kriging [OPTIONS] --attribute <ATTRIBUTE> --features <FEATURES> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>    Attribute name with target values
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --features <FEATURES>      Directory with covariable rasters (features)
      --reference <REFERENCE>    Reference raster for output grid
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --model <MODEL>            Variogram model for residuals [default: spherical]
  -h, --help                     Print help
```

## `interpolation idw` {#idw}

```text
Inverse Distance Weighting interpolation

Usage: surtgis interpolation idw [OPTIONS] --attribute <ATTRIBUTE> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>    Attribute name with values
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --reference <REFERENCE>    Reference raster for output grid
      --power <POWER>            Power parameter (default: 2.0, higher = more local) [default: 2.0]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --max-radius <MAX_RADIUS>  Maximum search radius (default: unlimited)
      --max-points <MAX_POINTS>  Maximum neighbors (default: all)
  -h, --help                     Print help
```

## `interpolation nearest-neighbor` {#nearest-neighbor}

```text
Nearest Neighbor interpolation (Voronoi/Thiessen)

Usage: surtgis interpolation nearest-neighbor [OPTIONS] --attribute <ATTRIBUTE> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>    Attribute name with values
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --reference <REFERENCE>    Reference raster for output grid
      --max-radius <MAX_RADIUS>  Maximum search radius (default: unlimited)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `interpolation natural-neighbor` {#natural-neighbor}

```text
Natural Neighbor interpolation (Sibson)

Usage: surtgis interpolation natural-neighbor [OPTIONS] --attribute <ATTRIBUTE> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>    Attribute name with values
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --reference <REFERENCE>    Reference raster for output grid
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `interpolation tps` {#tps}

```text
Thin Plate Spline interpolation

Usage: surtgis interpolation tps [OPTIONS] --attribute <ATTRIBUTE> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>    Attribute name with values
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --reference <REFERENCE>    Reference raster for output grid
      --smoothing <SMOOTHING>    Smoothing parameter (0 = exact, >0 = smoothing spline) [default: 0.0]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `interpolation tin` {#tin}

```text
TIN (Triangulated Irregular Network) interpolation

Usage: surtgis interpolation tin [OPTIONS] --attribute <ATTRIBUTE> --reference <REFERENCE> <POINTS> <OUTPUT>

Arguments:
  <POINTS>  Input points (GeoJSON/Shapefile)
  <OUTPUT>  Output raster

Options:
      --attribute <ATTRIBUTE>    Attribute name with values
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --reference <REFERENCE>    Reference raster for output grid
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

