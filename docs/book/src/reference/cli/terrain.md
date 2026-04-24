# `surtgis terrain`

_Terrain analysis algorithms_

## Overview

```text
Terrain analysis algorithms

Usage: surtgis terrain [OPTIONS] <COMMAND>

Commands:
  slope                         Calculate slope from DEM
  aspect                        Calculate aspect from DEM
  hillshade                     Calculate hillshade from DEM
  curvature                     Calculate surface curvature from DEM
  tpi                           Calculate Topographic Position Index
  tri                           Calculate Terrain Ruggedness Index
  landform                      Landform classification (multi-scale TPI + slope)
  geomorphons                   Geomorphon landform classification (Jasiewicz & Stepinski 2013)
  northness                     Northness: cos(aspect), north-facing = 1, south-facing = -1
  eastness                      Eastness: sin(aspect), east-facing = 1, west-facing = -1
  openness-positive             Positive topographic openness (sky visibility above)
  openness-negative             Negative topographic openness (enclosure below)
  svf                           Sky View Factor (0=enclosed, 1=flat horizon)
  mrvbf                         MRVBF/MRRTF: Multi-Resolution Valley/Ridge Bottom Flatness
  dev                           Deviation from Mean Elevation
  vrm                           Vector Ruggedness Measure
  advanced-curvature            Florinsky advanced curvature (14 types)
  viewshed                      Viewshed: binary line-of-sight visibility from an observer point
  convergence                   Convergence Index (-100=convergent, +100=divergent)
  multi-hillshade               Multi-directional hillshade (6 azimuths combined)
  ls-factor                     LS-Factor for RUSLE soil erosion model
  valley-depth                  Valley depth: vertical distance to ridge surface
  relative-slope-position       Relative Slope Position (0=valley, 1=ridge)
  surface-area-ratio            Surface Area Ratio (3D/2D area roughness)
  all                           Compute all standard terrain factors in one pass
  solar-radiation               Solar radiation (clear-sky insolation for a given day/hour)
  solar-radiation-annual        Annual solar radiation (integrated over full year)
  contour-lines                 Contour lines as raster
  cost-distance                 Cost distance from source points
  shape-index                   Shape index (concavity/convexity, -1 to +1)
  curvedness                    Curvedness (magnitude of curvature)
  gaussian-smoothing            Gaussian smoothing
  feature-preserving-smoothing  Feature-preserving smoothing (edge-aware)
  wind-exposure                 Wind exposure index
  horizon-angle                 Horizon angles for a given azimuth
  accumulation-zones            Accumulation zones (contributing area classification)
  spi                           Stream Power Index (SPI = A × tan(slope))
  sti                           Sediment Transport Index (STI)
  twi                           Topographic Wetness Index (TWI = ln(A / tan(slope)))
  log-transform                 Log transform (ln(x+1))
  uncertainty                   DEM uncertainty analysis (Monte Carlo)
  viewshed-pderl                PDERL viewshed (reference plane algorithm)
  viewshed-xdraw                XDraw viewshed (approximate, fast)
  viewshed-multiple             Multiple-observer cumulative viewshed
  hypsometric-hillshade         Hypsometrically tinted hillshade (hillshade × normalized elevation)
  elev-relative                 Elevation relative to global min/max (normalized 0–1)
  diff-from-mean                Difference from mean elevation (non-normalized, in DEM units)
  percent-elev-range            Percent elevation range (local position 0–100%)
  elev-above-pit                Elevation above pit / depth in sink
  circular-variance-aspect      Circular variance of aspect (0=uniform, 1=dispersed)
  neighbours                    Neighbour elevation statistics (3×3 window, 5 outputs)
  pennock                       Pennock landform classification (7 classes)
  edge-density                  Edge density (proportion of edge pixels in focal window)
  relative-aspect               Relative aspect (local vs regional aspect difference, 0–180°)
  normal-deviation              Average normal vector angular deviation (degrees)
  spherical-std-dev             Spherical standard deviation of surface normals
  directional-relief            Directional relief (elevation range along azimuth)
  downslope-index               Downslope index (distance to reach elevation drop, Hjerdt 2004)
  max-branch-length             Maximum upstream branch length (longest D8 flow path)
  help                          Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain slope` {#slope}

```text
Calculate slope from DEM

Usage: surtgis terrain slope [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
  -u, --units <UNITS>            Output units: degrees, percent, radians [default: degrees]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -z, --z-factor <Z_FACTOR>      Z-factor for unit conversion [default: 1.0]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain aspect` {#aspect}

```text
Calculate aspect from DEM

Usage: surtgis terrain aspect [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
  -f, --format <FORMAT>          Output format: degrees, radians, compass [default: degrees]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain hillshade` {#hillshade}

```text
Calculate hillshade from DEM

Usage: surtgis terrain hillshade [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
  -a, --azimuth <AZIMUTH>        Sun azimuth in degrees (0=North, clockwise) [default: 315]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -l, --altitude <ALTITUDE>      Sun altitude in degrees above horizon [default: 45]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
  -z, --z-factor <Z_FACTOR>      Z-factor for vertical exaggeration [default: 1.0]
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain curvature` {#curvature}

```text
Calculate surface curvature from DEM

Usage: surtgis terrain curvature [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
  -t, --curvature-type <CURVATURE_TYPE>
          Curvature type: general, profile, plan [default: general]
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
  -z, --z-factor <Z_FACTOR>
          Z-factor for unit conversion [default: 1.0]
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help
          Print help
```

## `terrain tpi` {#tpi}

```text
Calculate Topographic Position Index

Usage: surtgis terrain tpi [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 1]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain tri` {#tri}

```text
Calculate Terrain Ruggedness Index

Usage: surtgis terrain tri [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 1]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain landform` {#landform}

```text
Landform classification (multi-scale TPI + slope)

Usage: surtgis terrain landform [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file (class codes 1-11)

Options:
      --small-radius <SMALL_RADIUS>
          Small-scale TPI radius [default: 3]
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
      --large-radius <LARGE_RADIUS>
          Large-scale TPI radius [default: 10]
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --threshold <THRESHOLD>
          Standardized TPI threshold (z-score) [default: 1.0]
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --slope-threshold <SLOPE_THRESHOLD>
          Slope threshold (degrees) for gentle/steep [default: 6.0]
  -h, --help
          Print help
```

## `terrain geomorphons` {#geomorphons}

```text
Geomorphon landform classification (Jasiewicz & Stepinski 2013)

Usage: surtgis terrain geomorphons [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Lookup radius in cells [default: 10]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -f, --flatness <FLATNESS>      Flatness threshold in degrees [default: 1.0]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain northness` {#northness}

```text
Northness: cos(aspect), north-facing = 1, south-facing = -1

Usage: surtgis terrain northness [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain eastness` {#eastness}

```text
Eastness: sin(aspect), east-facing = 1, west-facing = -1

Usage: surtgis terrain eastness [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain openness-positive` {#openness-positive}

```text
Positive topographic openness (sky visibility above)

Usage: surtgis terrain openness-positive [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Search radius in cells [default: 10]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -d, --directions <DIRECTIONS>  Number of azimuth directions [default: 8]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain openness-negative` {#openness-negative}

```text
Negative topographic openness (enclosure below)

Usage: surtgis terrain openness-negative [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          [default: 10]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -d, --directions <DIRECTIONS>  [default: 8]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain svf` {#svf}

```text
Sky View Factor (0=enclosed, 1=flat horizon)

Usage: surtgis terrain svf [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          [default: 10]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -d, --directions <DIRECTIONS>  [default: 16]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain mrvbf` {#mrvbf}

```text
MRVBF/MRRTF: Multi-Resolution Valley/Ridge Bottom Flatness

Usage: surtgis terrain mrvbf [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  Output MRVBF file

Options:
      --mrrtf-output <MRRTF_OUTPUT>  Optional MRRTF output file
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                         Print help
```

## `terrain dev` {#dev}

```text
Deviation from Mean Elevation

Usage: surtgis terrain dev [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          [default: 10]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain vrm` {#vrm}

```text
Vector Ruggedness Measure

Usage: surtgis terrain vrm [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          [default: 1]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain advanced-curvature` {#advanced-curvature}

```text
Florinsky advanced curvature (14 types)

Usage: surtgis terrain advanced-curvature [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -t, --curvature-type <CURVATURE_TYPE>
          Curvature type: mean_h|mean, gaussian_k|gaussian, kmin|minimal, kmax|maximal, kh|horizontal, kv|vertical, khe|horizontal_excess, kve|vertical_excess, ka|accumulation, kr|ring, rotor, laplacian, unsphericity, difference [default: mean_h]
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help
          Print help
```

## `terrain viewshed` {#viewshed}

```text
Viewshed: binary line-of-sight visibility from an observer point

Usage: surtgis terrain viewshed [OPTIONS] --observer-row <OBSERVER_ROW> --observer-col <OBSERVER_COL> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --observer-row <OBSERVER_ROW>
          Observer row (pixel coordinate)
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
      --observer-col <OBSERVER_COL>
          Observer column (pixel coordinate)
      --observer-height <OBSERVER_HEIGHT>
          Observer height above ground (meters) [default: 1.8]
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --target-height <TARGET_HEIGHT>
          Target height above ground (meters) [default: 0.0]
      --max-radius <MAX_RADIUS>
          Maximum visibility radius in cells (0 = unlimited) [default: 0]
  -h, --help
          Print help
```

## `terrain convergence` {#convergence}

```text
Convergence Index (-100=convergent, +100=divergent)

Usage: surtgis terrain convergence [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          [default: 3]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain multi-hillshade` {#multi-hillshade}

```text
Multi-directional hillshade (6 azimuths combined)

Usage: surtgis terrain multi-hillshade [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain ls-factor` {#ls-factor}

```text
LS-Factor for RUSLE soil erosion model

Usage: surtgis terrain ls-factor [OPTIONS] --flow-acc <FLOW_ACC> --slope <SLOPE> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --flow-acc <FLOW_ACC>      Flow accumulation raster
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --slope <SLOPE>            Slope raster (radians)
      --cell-size <CELL_SIZE>    Cell size in meters [default: 1.0]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain valley-depth` {#valley-depth}

```text
Valley depth: vertical distance to ridge surface

Usage: surtgis terrain valley-depth [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain relative-slope-position` {#relative-slope-position}

```text
Relative Slope Position (0=valley, 1=ridge)

Usage: surtgis terrain relative-slope-position [OPTIONS] --hand <HAND> --valley-depth <VALLEY_DEPTH> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --hand <HAND>                  HAND raster
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --valley-depth <VALLEY_DEPTH>  Valley depth raster
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                         Print help
```

## `terrain surface-area-ratio` {#surface-area-ratio}

```text
Surface Area Ratio (3D/2D area roughness)

Usage: surtgis terrain surface-area-ratio [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          [default: 1]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain all` {#all}

```text
Compute all standard terrain factors in one pass

Usage: surtgis terrain all [OPTIONS] --outdir <OUTDIR> <INPUT>

Arguments:
  <INPUT>  Input DEM file

Options:
  -o, --outdir <OUTDIR>          Output directory for all terrain products
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain solar-radiation` {#solar-radiation}

```text
Solar radiation (clear-sky insolation for a given day/hour)

Usage: surtgis terrain solar-radiation [OPTIONS] --day <DAY> --hour <HOUR> --latitude <LATITUDE> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --day <DAY>                Day of year (1-365)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --hour <HOUR>              Hour (solar time, 0-24)
      --latitude <LATITUDE>      Latitude in degrees (negative for southern hemisphere)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain solar-radiation-annual` {#solar-radiation-annual}

```text
Annual solar radiation (integrated over full year)

Usage: surtgis terrain solar-radiation-annual [OPTIONS] --latitude <LATITUDE> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --latitude <LATITUDE>      Latitude in degrees (negative for southern hemisphere)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain contour-lines` {#contour-lines}

```text
Contour lines as raster

Usage: surtgis terrain contour-lines [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --interval <INTERVAL>      Contour interval [default: 100]
  -v, --verbose                  Verbose output
      --base <BASE>              Base contour value [default: 0]
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain cost-distance` {#cost-distance}

```text
Cost distance from source points

Usage: surtgis terrain cost-distance [OPTIONS] --sources <SOURCES> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Cost surface raster
  <OUTPUT>  Output accumulated cost raster

Options:
      --sources <SOURCES>        Source points raster (non-zero = source)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain shape-index` {#shape-index}

```text
Shape index (concavity/convexity, -1 to +1)

Usage: surtgis terrain shape-index [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain curvedness` {#curvedness}

```text
Curvedness (magnitude of curvature)

Usage: surtgis terrain curvedness [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain gaussian-smoothing` {#gaussian-smoothing}

```text
Gaussian smoothing

Usage: surtgis terrain gaussian-smoothing [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --sigma <SIGMA>            Sigma (standard deviation in cells) [default: 1.0]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --radius <RADIUS>          Kernel radius in cells (default: ceil(3*sigma))
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain feature-preserving-smoothing` {#feature-preserving-smoothing}

```text
Feature-preserving smoothing (edge-aware)

Usage: surtgis terrain feature-preserving-smoothing [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --strength <STRENGTH>      Smoothing strength [default: 1.0]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --iterations <ITERATIONS>  Number of iterations [default: 3]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain wind-exposure` {#wind-exposure}

```text
Wind exposure index

Usage: surtgis terrain wind-exposure [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --direction <DIRECTION>    Wind direction in degrees from north (0=N, 90=E) [default: 270]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --radius <RADIUS>          Search radius in cells [default: 100]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain horizon-angle` {#horizon-angle}

```text
Horizon angles for a given azimuth

Usage: surtgis terrain horizon-angle [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --azimuth <AZIMUTH>        Azimuth in degrees from north [default: 180]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --radius <RADIUS>          Search radius in cells [default: 100]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain accumulation-zones` {#accumulation-zones}

```text
Accumulation zones (contributing area classification)

Usage: surtgis terrain accumulation-zones [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain spi` {#spi}

```text
Stream Power Index (SPI = A × tan(slope))

Usage: surtgis terrain spi [OPTIONS] --flow-acc <FLOW_ACC> --slope <SLOPE> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --flow-acc <FLOW_ACC>      Flow accumulation raster
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --slope <SLOPE>            Slope raster (radians)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain sti` {#sti}

```text
Sediment Transport Index (STI)

Usage: surtgis terrain sti [OPTIONS] --flow-acc <FLOW_ACC> --slope <SLOPE> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --flow-acc <FLOW_ACC>      Flow accumulation raster
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --slope <SLOPE>            Slope raster (radians)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain twi` {#twi}

```text
Topographic Wetness Index (TWI = ln(A / tan(slope)))

Usage: surtgis terrain twi [OPTIONS] --flow-acc <FLOW_ACC> --slope <SLOPE> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --flow-acc <FLOW_ACC>      Flow accumulation raster
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --slope <SLOPE>            Slope raster (radians)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain log-transform` {#log-transform}

```text
Log transform (ln(x+1))

Usage: surtgis terrain log-transform [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain uncertainty` {#uncertainty}

```text
DEM uncertainty analysis (Monte Carlo)

Usage: surtgis terrain uncertainty [OPTIONS] --outdir <OUTDIR> <INPUT>

Arguments:
  <INPUT>  

Options:
  -o, --outdir <OUTDIR>                Output directory (mean, std, etc.)
  -v, --verbose                        Verbose output
      --compress                       Compress output GeoTIFFs (deflate)
      --error-std <ERROR_STD>          Error standard deviation (meters) [default: 1.0]
      --n-simulations <N_SIMULATIONS>  Number of simulations [default: 100]
      --streaming                      Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>        Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                           Print help
```

## `terrain viewshed-pderl` {#viewshed-pderl}

```text
PDERL viewshed (reference plane algorithm)

Usage: surtgis terrain viewshed-pderl [OPTIONS] --row <ROW> --col <COL> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --row <ROW>                Observer row
  -v, --verbose                  Verbose output
      --col <COL>                Observer column
      --compress                 Compress output GeoTIFFs (deflate)
      --height <HEIGHT>          Observer height above ground [default: 1.7]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain viewshed-xdraw` {#viewshed-xdraw}

```text
XDraw viewshed (approximate, fast)

Usage: surtgis terrain viewshed-xdraw [OPTIONS] --row <ROW> --col <COL> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --row <ROW>                Observer row
  -v, --verbose                  Verbose output
      --col <COL>                Observer column
      --compress                 Compress output GeoTIFFs (deflate)
      --height <HEIGHT>          Observer height above ground [default: 1.7]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain viewshed-multiple` {#viewshed-multiple}

```text
Multiple-observer cumulative viewshed

Usage: surtgis terrain viewshed-multiple [OPTIONS] --observers <OBSERVERS> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --observers <OBSERVERS>    Observer locations as row,col pairs (e.g. "10,20;30,40;50,60")
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --height <HEIGHT>          Observer height above ground [default: 1.7]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain hypsometric-hillshade` {#hypsometric-hillshade}

```text
Hypsometrically tinted hillshade (hillshade × normalized elevation)

Usage: surtgis terrain hypsometric-hillshade [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -a, --azimuth <AZIMUTH>        Sun azimuth in degrees [default: 315]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -l, --altitude <ALTITUDE>      Sun altitude in degrees [default: 45]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
  -z, --z-factor <Z_FACTOR>      Z-factor [default: 1.0]
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain elev-relative` {#elev-relative}

```text
Elevation relative to global min/max (normalized 0–1)

Usage: surtgis terrain elev-relative [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain diff-from-mean` {#diff-from-mean}

```text
Difference from mean elevation (non-normalized, in DEM units)

Usage: surtgis terrain diff-from-mean [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 10]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain percent-elev-range` {#percent-elev-range}

```text
Percent elevation range (local position 0–100%)

Usage: surtgis terrain percent-elev-range [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 10]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain elev-above-pit` {#elev-above-pit}

```text
Elevation above pit / depth in sink

Usage: surtgis terrain elev-above-pit [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain circular-variance-aspect` {#circular-variance-aspect}

```text
Circular variance of aspect (0=uniform, 1=dispersed)

Usage: surtgis terrain circular-variance-aspect [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 3]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain neighbours` {#neighbours}

```text
Neighbour elevation statistics (3×3 window, 5 outputs)

Usage: surtgis terrain neighbours [OPTIONS] --output <OUTPUT> <INPUT>

Arguments:
  <INPUT>  

Options:
  -o, --output <OUTPUT>          Output path: directory (all 5 stats) or file (with --stat)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --stat <STAT>              Single statistic: max_downslope_change, min_downslope_change, max_upslope_change, num_downslope, num_upslope
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain pennock` {#pennock}

```text
Pennock landform classification (7 classes)

Usage: surtgis terrain pennock [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --slope-threshold <SLOPE_THRESHOLD>
          Slope threshold (degrees) for "level" class [default: 3.0]
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
      --curv-threshold <CURV_THRESHOLD>
          Profile curvature threshold for linear vs convex/concave [default: 0.1]
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help
          Print help
```

## `terrain edge-density` {#edge-density}

```text
Edge density (proportion of edge pixels in focal window)

Usage: surtgis terrain edge-density [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 3]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -t, --threshold <THRESHOLD>    Sobel magnitude threshold for edge detection [default: 0.5]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain relative-aspect` {#relative-aspect}

```text
Relative aspect (local vs regional aspect difference, 0–180°)

Usage: surtgis terrain relative-aspect [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --sigma <SIGMA>            Gaussian sigma for regional smoothing [default: 50]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain normal-deviation` {#normal-deviation}

```text
Average normal vector angular deviation (degrees)

Usage: surtgis terrain normal-deviation [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 3]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain spherical-std-dev` {#spherical-std-dev}

```text
Spherical standard deviation of surface normals

Usage: surtgis terrain spherical-std-dev [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Neighborhood radius in cells [default: 3]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain directional-relief` {#directional-relief}

```text
Directional relief (elevation range along azimuth)

Usage: surtgis terrain directional-relief [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          Search radius in cells [default: 10]
  -v, --verbose                  Verbose output
  -a, --azimuth <AZIMUTH>        Azimuth in degrees (0 = multidirectional average) [default: 0]
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain downslope-index` {#downslope-index}

```text
Downslope index (distance to reach elevation drop, Hjerdt 2004)

Usage: surtgis terrain downslope-index [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -d, --drop <DROP>              Elevation drop threshold in DEM units [default: 2.0]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `terrain max-branch-length` {#max-branch-length}

```text
Maximum upstream branch length (longest D8 flow path)

Usage: surtgis terrain max-branch-length [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

