# `surtgis hydrology`

_Hydrology algorithms_

## Overview

```text
Hydrology algorithms

Usage: surtgis hydrology [OPTIONS] <COMMAND>

Commands:
  fill-sinks             Fill sinks / depressions in DEM (Planchon-Darboux 2001)
  flow-direction         D8 flow direction from DEM
  flow-accumulation      Flow accumulation from flow direction raster
  watershed              Watershed delineation from flow direction
  priority-flood         Priority-Flood depression filling (Barnes 2014, optimal O(n log n))
  breach                 Breach depressions (carve channels through barriers)
  flow-direction-dinf    D-infinity flow direction (Tarboton 1997, continuous angles)
  flow-accumulation-mfd  Multiple Flow Direction accumulation (Quinn et al. 1991)
  twi                    Topographic Wetness Index (from DEM, full pipeline)
  hand                   Height Above Nearest Drainage (from DEM, full pipeline)
  stream-network         Stream network extraction (from DEM, full pipeline)
  drainage-density       Drainage density: stream length per unit area
  hypsometric-integral   Hypsometric integral per watershed
  sediment-connectivity  Sediment Connectivity Index (Borselli 2008)
  basin-morphometry      Basin morphometric parameters per watershed
  all                    Compute full hydrology pipeline from DEM
  help                   Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology fill-sinks` {#fill-sinks}

```text
Fill sinks / depressions in DEM (Planchon-Darboux 2001)

Usage: surtgis hydrology fill-sinks [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
      --min-slope <MIN_SLOPE>    Minimum slope to enforce [default: 0.01]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology flow-direction` {#flow-direction}

```text
D8 flow direction from DEM

Usage: surtgis hydrology flow-direction [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file (D8 codes: 1,2,4,8,16,32,64,128)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology flow-accumulation` {#flow-accumulation}

```text
Flow accumulation from flow direction raster

Usage: surtgis hydrology flow-accumulation [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input flow direction raster (D8 codes)
  <OUTPUT>  Output file (upstream cell count)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology watershed` {#watershed}

```text
Watershed delineation from flow direction

Usage: surtgis hydrology watershed [OPTIONS] --pour-points <POUR_POINTS> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input flow direction raster (D8 codes)
  <OUTPUT>  Output file (basin IDs)

Options:
      --pour-points <POUR_POINTS>  Pour points as "row,col;row,col;..."
  -v, --verbose                    Verbose output
      --compress                   Compress output GeoTIFFs (deflate)
      --streaming                  Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>    Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                       Print help
```

## `hydrology priority-flood` {#priority-flood}

```text
Priority-Flood depression filling (Barnes 2014, optimal O(n log n))

Usage: surtgis hydrology priority-flood [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --epsilon <EPSILON>        Minimum slope epsilon [default: 0.0001]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology breach` {#breach}

```text
Breach depressions (carve channels through barriers)

Usage: surtgis hydrology breach [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --max-depth <MAX_DEPTH>    Maximum breach depth (meters) [default: 100.0]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --max-length <MAX_LENGTH>  Maximum breach length (cells) [default: 1000]
      --fill-remaining           Fill remaining unfilled depressions
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology flow-direction-dinf` {#flow-direction-dinf}

```text
D-infinity flow direction (Tarboton 1997, continuous angles)

Usage: surtgis hydrology flow-direction-dinf [OPTIONS] <INPUT> <OUTPUT>

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

## `hydrology flow-accumulation-mfd` {#flow-accumulation-mfd}

```text
Multiple Flow Direction accumulation (Quinn et al. 1991)

Usage: surtgis hydrology flow-accumulation-mfd [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --exponent <EXPONENT>      Flow partition exponent [default: 1.1]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology twi` {#twi}

```text
Topographic Wetness Index (from DEM, full pipeline)

Usage: surtgis hydrology twi [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology hand` {#hand}

```text
Height Above Nearest Drainage (from DEM, full pipeline)

Usage: surtgis hydrology hand [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  

Options:
      --threshold <THRESHOLD>    Stream extraction threshold (contributing cells) [default: 1000]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology stream-network` {#stream-network}

```text
Stream network extraction (from DEM, full pipeline)

Usage: surtgis hydrology stream-network [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  

Options:
      --threshold <THRESHOLD>    Contributing area threshold [default: 1000]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology drainage-density` {#drainage-density}

```text
Drainage density: stream length per unit area

Usage: surtgis hydrology drainage-density [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Stream network raster (binary: 1=stream)
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>          [default: 10]
  -v, --verbose                  Verbose output
      --cell-size <CELL_SIZE>    [default: 1.0]
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology hypsometric-integral` {#hypsometric-integral}

```text
Hypsometric integral per watershed

Usage: surtgis hydrology hypsometric-integral [OPTIONS] --dem <DEM> --watersheds <WATERSHEDS>

Options:
      --dem <DEM>                DEM file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --watersheds <WATERSHEDS>  Watershed raster (i32 IDs)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology sediment-connectivity` {#sediment-connectivity}

```text
Sediment Connectivity Index (Borselli 2008)

Usage: surtgis hydrology sediment-connectivity [OPTIONS] --slope <SLOPE> --flow-acc <FLOW_ACC> --flow-dir <FLOW_DIR> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --slope <SLOPE>            Slope raster (radians)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --flow-acc <FLOW_ACC>      Flow accumulation raster
      --flow-dir <FLOW_DIR>      D8 flow direction raster
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --threshold <THRESHOLD>    Stream threshold (flow accumulation cells) [default: 1000]
  -h, --help                     Print help
```

## `hydrology basin-morphometry` {#basin-morphometry}

```text
Basin morphometric parameters per watershed

Usage: surtgis hydrology basin-morphometry [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Watershed raster (i32 IDs)

Options:
      --cell-size <CELL_SIZE>    [default: 1.0]
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `hydrology all` {#all}

```text
Compute full hydrology pipeline from DEM

Usage: surtgis hydrology all [OPTIONS] --outdir <OUTDIR> <INPUT>

Arguments:
  <INPUT>  Input DEM file

Options:
  -o, --outdir <OUTDIR>          Output directory
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --threshold <THRESHOLD>    Stream threshold [default: 1000]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

