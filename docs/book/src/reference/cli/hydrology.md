# `surtgis hydrology`

_Hydrology algorithms_

## Overview

```text
Hydrology algorithms

Usage: surtgis hydrology [OPTIONS] <COMMAND>

Commands:
  lake-catchment          Catchments of many lakes, with each lake as the outlet
  fill-sinks              Fill sinks / depressions in DEM (Planchon-Darboux 2001)
  flow-direction          D8 flow direction from DEM (flats resolved with Garbrecht-Martz 1997)
  flow-accumulation       Flow accumulation from flow direction raster
  watershed               Watershed delineation from flow direction
  priority-flood          Priority-Flood depression filling (Barnes 2014, optimal O(n log n))
  breach                  Breach depressions (carve channels through barriers)
  flow-direction-dinf     D-infinity flow direction (Tarboton 1997, continuous angles)
  flow-accumulation-dinf  D-infinity flow accumulation from a D-inf angle raster
  flow-accumulation-mfd   Multiple Flow Direction accumulation (Quinn et al. 1991)
  twi                     Topographic Wetness Index (from DEM, full pipeline)
  hand                    Height Above Nearest Drainage (from DEM, full pipeline)
  stream-network          Stream network extraction
  drainage-density        Drainage density: stream length per unit area
  hypsometric-integral    Hypsometric integral per watershed
  sediment-connectivity   Sediment Connectivity Index (Borselli 2008)
  basin-morphometry       Basin morphometric parameters per watershed
  melton                  Melton ruggedness ratio per watershed (debris-flow / lahar screening)
  energy-cone             Energy-cone lahar / mass-flow inundation (Malin & Sheridan 1982)
  laharz                  LAHARZ lahar / debris-flow inundation (Iverson, Schilling & Vallance 1998)
  all                     Compute full hydrology pipeline from DEM
  help                    Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology lake-catchment` {#lake-catchment}

```text
Catchments of many lakes, with each lake as the outlet

For every lake: a UTM window around it, each --dem warped onto it, the lake rasterized (polygons) or extracted from --water-mask (points), depressions filled everywhere except the lake, D8, and the union of everything that drains into any lake cell. The window doubles while the catchment or the lake reaches its edge. Writes one CSV row per lake and DEM; with two or more DEMs `dem_spread` = max/min catchment area across them (large values flag flat, uncertain terrain).

Usage: surtgis hydrology lake-catchment [OPTIONS] --dem <DEMS> <LAKES> <OUTPUT>

Arguments:
  <LAKES>
          Lakes: polygons (preferred) or points; GeoJSON, GeoPackage, Shapefile or GeoParquet, any CRS with an EPSG code

  <OUTPUT>
          Output CSV

Options:
      --dem <DEMS>
          DEM source, repeatable: `stac:<catalog>:<collection>[:asset]` (e.g. stac:pc:cop-dem-glo-30, stac:pc:alos-dem) or a local GeoTIFF/COG file or directory of tiles (e.g. FABDEM)

  -v, --verbose
          Verbose output

      --compress
          Compress output GeoTIFFs (deflate)

      --id-field <ID_FIELD>
          Attribute holding the lake id (default: feature id or index)

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --water-mask <WATER_MASKS>
          Water raster for point lakes, repeatable, tried in order (the first with data and a water body near the point wins). A rule may follow the spec: `stac:pc:jrc-gsw:occurrence>=90` (JRC GSW, to 78°N), `stac:pc:esa-worldcover:map=80` (to 82.75°N)

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

      --water-value <WATER_VALUE>
          Water value for --water-mask specs without a rule (WorldCover: 80)
          
          [default: 80]

      --min-lake-cells <MIN_LAKE_CELLS>
          Water bodies smaller than this (cells) are not used as the lake; the cell under the point is (status point_only)
          
          [default: 4]

      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN)
          
          [default: f64]
          [possible values: f32, f64]

      --max-neck-cells <MAX_NECK_CELLS>
          Parts of a water-mask lake joined across a narrows of at most this many cells removed by the river-cutting opening count as one lake; 0 keeps only the part at the point
          
          [default: 5]

      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1

      --min-compactness <MIN_COMPACTNESS>
          Water-mask lakes below this compactness 4πA/P² (or cut by the opening to under half their component) get status review: possibly a river or merged lakes
          
          [default: 0.06]

      --window-km <WINDOW_KM>
          Initial window side, km (grown to 3x the lake's extent if larger)
          
          [default: 16]

      --max-window-km <MAX_WINDOW_KM>
          Largest window side, km
          
          [default: 256]

      --pixel-size <PIXEL_SIZE>
          Cell size of the working UTM grid, metres
          
          [default: 30]

      --jobs <JOBS>
          Lakes processed in parallel
          
          [default: 4]

      --retries <RETRIES>
          Attempts per remote tile
          
          [default: 3]

      --masks-dir <MASKS_DIR>
          Also write each catchment mask (0 out, 1 catchment, 2 lake) here

      --sink-depth <SINK_DEPTH>
          Also report the local catchment: depressions at least this deep (m) keep their own drainage instead of spilling into the lake (`catchment_area_km2` is the connected, fill-and-spill one)

  -h, --help
          Print help (see a summary with '-h')
```

## `hydrology fill-sinks` {#fill-sinks}

```text
Fill sinks / depressions in DEM (Planchon-Darboux 2001)

Usage: surtgis hydrology fill-sinks [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file

Options:
      --min-slope <MIN_SLOPE>        Minimum slope to enforce [default: 1e-5]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology flow-direction` {#flow-direction}

```text
D8 flow direction from DEM (flats resolved with Garbrecht-Martz 1997)

Usage: surtgis hydrology flow-direction [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output file (D8 codes: 0 = no outflow, 1-8 counter-clockwise from East)

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology flow-accumulation` {#flow-accumulation}

```text
Flow accumulation from flow direction raster

Usage: surtgis hydrology flow-accumulation [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input flow direction raster (D8 codes)
  <OUTPUT>  Output file (upstream cell count, or upstream weight sum with --weights)

Options:
  -v, --verbose                      Verbose output
      --weights <RASTER>             Weight raster on the same grid: each cell receives the sum of the weights of every cell upstream of it (susceptible source area, runoff, sediment or pollutant load). NaN/nodata weights contribute nothing but still pass flow through. Read as f32
      --compress                     Compress output GeoTIFFs (deflate)
      --include-self                 Add each cell's own weight (1 without --weights) to its result. This is the convention of terra::flowAccumulation and WhiteboxTools; without it a headwater cell is 0
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology watershed` {#watershed}

```text
Watershed delineation from flow direction

Usage: surtgis hydrology watershed [OPTIONS] --pour-points <POUR_POINTS> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input flow direction raster (D8 codes)
  <OUTPUT>  Output file (basin IDs)

Options:
      --pour-points <POUR_POINTS>    Pour points as "row,col;row,col;..."
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology priority-flood` {#priority-flood}

```text
Priority-Flood depression filling (Barnes 2014, optimal O(n log n))

Usage: surtgis hydrology priority-flood [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --epsilon <EPSILON>            Minimum slope epsilon [default: 0.0001]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology breach` {#breach}

```text
Breach depressions (carve channels through barriers)

Usage: surtgis hydrology breach [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --max-depth <MAX_DEPTH>        Maximum breach depth (meters) [default: 100.0]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --max-length <MAX_LENGTH>      Maximum breach length (cells) [default: 1000]
      --fill-remaining               Fill remaining unfilled depressions
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology flow-direction-dinf` {#flow-direction-dinf}

```text
D-infinity flow direction (Tarboton 1997, continuous angles)

Usage: surtgis hydrology flow-direction-dinf [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology flow-accumulation-dinf` {#flow-accumulation-dinf}

```text
D-infinity flow accumulation from a D-inf angle raster

Usage: surtgis hydrology flow-accumulation-dinf [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input D-inf angle raster (output of flow-direction-dinf)
  <OUTPUT>  

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology flow-accumulation-mfd` {#flow-accumulation-mfd}

```text
Multiple Flow Direction accumulation (Quinn et al. 1991)

Usage: surtgis hydrology flow-accumulation-mfd [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --exponent <EXPONENT>          Flow partition exponent [default: 1.1]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology twi` {#twi}

```text
Topographic Wetness Index (from DEM, full pipeline)

Usage: surtgis hydrology twi [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  

Options:
      --method <METHOD>              Flow routing method for the accumulation step [default: d8] [possible values: d8, dinf, mfd]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology hand` {#hand}

```text
Height Above Nearest Drainage (from DEM, full pipeline)

Usage: surtgis hydrology hand [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  

Options:
      --threshold <THRESHOLD>        Stream extraction threshold (contributing cells) [default: 1000]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology stream-network` {#stream-network}

```text
Stream network extraction.

By default the input is treated as a DEM and the handler runs the full pipeline (priority_flood → flow_direction → flow_accumulation → threshold). When `--from-facc` is passed, the input is treated as a pre-computed flow_accumulation raster and only the threshold step runs — this is the path you want when composing with an externally computed flow direction / accumulation, e.g. for the fluvial module's chi/ksn workflow.

Usage: surtgis hydrology stream-network [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>
          Input raster. DEM by default; flow_accumulation when `--from-facc`

  <OUTPUT>
          

Options:
      --threshold <THRESHOLD>
          Contributing area threshold (cell counts)
          
          [default: 1000]

  -v, --verbose
          Verbose output

      --compress
          Compress output GeoTIFFs (deflate)

      --from-facc
          Treat `input` as a pre-computed flow_accumulation raster and skip the DEM → fdir → facc recomputation. Use when you already have flow_dir/acc and want the resulting `stream-network` to be topologically consistent with them

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

## `hydrology drainage-density` {#drainage-density}

```text
Drainage density: stream length per unit area

Usage: surtgis hydrology drainage-density [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Stream network raster (binary: 1=stream)
  <OUTPUT>  

Options:
  -r, --radius <RADIUS>              [default: 10]
  -v, --verbose                      Verbose output
      --cell-size <CELL_SIZE>        [default: 1.0]
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology hypsometric-integral` {#hypsometric-integral}

```text
Hypsometric integral per watershed

Usage: surtgis hydrology hypsometric-integral [OPTIONS] --dem <DEM> --watersheds <WATERSHEDS>

Options:
      --dem <DEM>                    DEM file
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --watersheds <WATERSHEDS>      Watershed raster (i32 IDs)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology sediment-connectivity` {#sediment-connectivity}

```text
Sediment Connectivity Index (Borselli 2008)

Usage: surtgis hydrology sediment-connectivity [OPTIONS] --slope <SLOPE> --flow-acc <FLOW_ACC> --flow-dir <FLOW_DIR> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --slope <SLOPE>                Slope raster (radians)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --flow-acc <FLOW_ACC>          Flow accumulation raster
      --flow-dir <FLOW_DIR>          D8 flow direction raster
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --threshold <THRESHOLD>        Stream threshold (flow accumulation cells) [default: 1000]
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology basin-morphometry` {#basin-morphometry}

```text
Basin morphometric parameters per watershed

Usage: surtgis hydrology basin-morphometry [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Watershed raster (i32 IDs)

Options:
      --cell-size <CELL_SIZE>        [default: 1.0]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology melton` {#melton}

```text
Melton ruggedness ratio per watershed (debris-flow / lahar screening)

Usage: surtgis hydrology melton [OPTIONS] --dem <DEM> <INPUT>

Arguments:
  <INPUT>  Watershed raster (i32 IDs)

Options:
      --dem <DEM>                    DEM raster aligned with the watershed raster
  -v, --verbose                      Verbose output
      --cell-size <CELL_SIZE>        [default: 1.0]
      --compress                     Compress output GeoTIFFs (deflate)
      --output <OUTPUT>              Optional output raster mapping each basin to its Melton ratio
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `hydrology energy-cone` {#energy-cone}

```text
Energy-cone lahar / mass-flow inundation (Malin & Sheridan 1982)

Usage: surtgis hydrology energy-cone [OPTIONS] --source <SOURCE> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output energy-height-above-ground raster (>0 = reached)

Options:
      --source <SOURCE>
          Source cell(s) as "row,col" (multiple separated by ';')
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
      --cone-angle <CONE_ANGLE>
          Energy-cone angle φ in degrees (H/L = tan φ); smaller = more mobile [default: 10.0]
      --collapse-height <COLLAPSE_HEIGHT>
          Collapse height added to the source elevation to set the apex [default: 0.0]
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help
          Print help
```

## `hydrology laharz` {#laharz}

```text
LAHARZ lahar / debris-flow inundation (Iverson, Schilling & Vallance 1998)

Usage: surtgis hydrology laharz [OPTIONS] --flow-dir <FLOW_DIR> --source <SOURCE> --volume <VOLUME> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output inundation-depth raster (>0 = inundated)

Options:
      --flow-dir <FLOW_DIR>            D8 flow-direction raster (u8, from `hydrology flow-direction`)
  -v, --verbose                        Verbose output
      --compress                       Compress output GeoTIFFs (deflate)
      --source <SOURCE>                Source cell(s) as "row,col" (multiple separated by ';'). Seed proximal CHANNEL cells, not the summit — a summit cell's D8 descent often runs down the wrong drainage. Sources route independently and the footprint is their union
      --streaming                      Force streaming mode for large rasters (auto-detected if >500MB)
      --volume <VOLUME>                Flow volume in m³ (applied to each source)
      --flow-type <FLOW_TYPE>          Flow type preset: lahar | debris-flow | rock-avalanche [default: lahar]
      --max-memory <MAX_MEMORY>        Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>    Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --spread-aspect <SPREAD_ASPECT>  Override the lateral-spread aspect ratio (width:depth). Omit to use the preset default; 0 = canonical fill-to-area-A (long thin ribbons)
      --no-provenance                  Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                           Print help
```

## `hydrology all` {#all}

```text
Compute full hydrology pipeline from DEM

Usage: surtgis hydrology all [OPTIONS] --outdir <OUTDIR> <INPUT>

Arguments:
  <INPUT>  Input DEM file

Options:
  -o, --outdir <OUTDIR>              Output directory
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --threshold <THRESHOLD>        Stream threshold [default: 1000]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

