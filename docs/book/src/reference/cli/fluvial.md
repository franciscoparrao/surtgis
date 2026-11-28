# `surtgis fluvial`

_Fluvial-tectonic morphometry (chi, ksn, knickpoints, divide migration)_

## Overview

```text
Fluvial-tectonic morphometry (chi, ksn, knickpoints, divide migration)

Usage: surtgis fluvial [OPTIONS] <COMMAND>

Commands:
  chi               χ (chi) integral transform (Perron & Royden 2013)
  ksn               Normalised channel steepness index `ksn` (Wobus et al. 2006)
  knickpoints       Knickpoint detection (Neely et al. 2017, TVD denoising + curvature)
  concavity         Concavity index θ per basin (Perron & Royden 2013)
  divide-migration  Divide-migration metrics (Willett 2014, Whipple 2017)
  help              Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `fluvial chi` {#chi}

```text
χ (chi) integral transform (Perron & Royden 2013).

Computes the path integral of (A₀/A(x))^θref along each stream cell starting from the network's outlets. χ is a base-level reference distance that linearises elevation profiles in steady state; together with ksn it is one of the two most common metrics in geomorphology-from-topography.

Usage: surtgis fluvial chi [OPTIONS] <STREAM> <FLOW_DIR> <FLOW_ACC> <OUTPUT>

Arguments:
  <STREAM>
          Binary stream network raster (1 = stream, 0 = non-stream), e.g. the output of `surtgis hydrology stream-network`

  <FLOW_DIR>
          D8 flow direction raster from `surtgis hydrology flow-direction`

  <FLOW_ACC>
          Flow accumulation raster (cell counts) from `surtgis hydrology flow-accumulation`

  <OUTPUT>
          Output χ raster (Float32 GeoTIFF, NaN = non-stream)

Options:
      --theta-ref <THETA_REF>
          Reference concavity exponent. Default 0.45 (bedrock channels in steady state)
          
          [default: 0.45]

  -v, --verbose
          Verbose output

      --a-0-m2 <A_0_M2>
          Reference drainage area in m². Default 1·10⁶ m² (1 km²)
          
          [default: 1e6]

      --compress
          Compress output GeoTIFFs (deflate)

      --cell-size-m <CELL_SIZE_M>
          Cell size override in metres. Default: read from the raster transform. Required when the input does not declare a projected CRS; not used to override an existing transform

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

## `fluvial ksn` {#ksn}

```text
Normalised channel steepness index `ksn` (Wobus et al. 2006).

Channel-following slope × A^θref, smoothed over a moving window along the network. `ksn` is the workhorse proxy for U/K (uplift rate divided by erodibility) used in tectonic geomorphology.

Usage: surtgis fluvial ksn [OPTIONS] <STREAM> <FLOW_DIR> <FLOW_ACC> <DEM> <OUTPUT>

Arguments:
  <STREAM>
          Binary stream network raster (1 = stream, 0 = non-stream)

  <FLOW_DIR>
          D8 flow direction raster

  <FLOW_ACC>
          Flow accumulation raster (cell counts)

  <DEM>
          DEM (elevation in metres)

  <OUTPUT>
          Output `ksn` raster (Float32 GeoTIFF, NaN = non-stream / outlet)

Options:
      --theta-ref <THETA_REF>
          Reference concavity exponent. Default 0.45
          
          [default: 0.45]

  -v, --verbose
          Verbose output

      --compress
          Compress output GeoTIFFs (deflate)

      --segment-length-m <SEGMENT_LENGTH_M>
          Smoothing window length in metres. Default 500 (Wobus 2006 standard)
          
          [default: 500]

      --min-drainage-area-m2 <MIN_DRAINAGE_AREA_M2>
          Minimum drainage area in m² for a cell to contribute. Default 1e6 (1 km²)
          
          [default: 1e6]

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --cell-size-m <CELL_SIZE_M>
          Cell size override in metres. Defaults to the raster transform's pixel size

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN)
          
          [default: f64]
          [possible values: f32, f64]

      --segments <SEGMENTS>
          Optional path to write per-segment vector output as a GeoJSON LineString FeatureCollection (one feature per stream segment between confluences/outlet, attributes: `ksn_mean`, `n_cells`)

      --keep-crs
          Preserve the source raster's CRS in the GeoJSON output and declare it via a legacy `crs` member (GeoJSON 2008 compat). Default: reproject coordinates to WGS84 (RFC 7946 compliant) — recommended for web-mapping pipelines (MapLibre, deck.gl) and modern geopandas / QGIS workflows

      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1

  -h, --help
          Print help (see a summary with '-h')
```

## `fluvial knickpoints` {#knickpoints}

```text
Knickpoint detection (Neely et al. 2017, TVD denoising + curvature).

Detects sharp slope breaks along each river long profile. Knickpoints are classified as `concave` (slope decreases downstream → likely lithologic contrast) or `convex` (slope increases downstream → likely transient tectonic pulse).

Usage: surtgis fluvial knickpoints [OPTIONS] <STREAM> <FLOW_DIR> <FLOW_ACC> <DEM> <OUTPUT>

Arguments:
  <STREAM>
          Binary stream network raster (1 = stream, 0 = non-stream)

  <FLOW_DIR>
          D8 flow direction raster

  <FLOW_ACC>
          Flow accumulation raster (cell counts)

  <DEM>
          DEM (elevation in metres)

  <OUTPUT>
          Output GeoJSON Point FeatureCollection of knickpoints

Options:
      --theta-ref <THETA_REF>
          Reference concavity exponent. Default 0.45
          
          [default: 0.45]

  -v, --verbose
          Verbose output

      --compress
          Compress output GeoTIFFs (deflate)

      --tvd-lambda <TVD_LAMBDA>
          TVD regularization (larger = stronger smoothing). Default 0.5
          
          [default: 0.5]

      --curvature-threshold <CURVATURE_THRESHOLD>
          Threshold on |d²z/dχ²| (units: 1/m). Default 1.0
          
          [default: 1.0]

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

      --min-magnitude-m <MIN_MAGNITUDE_M>
          Minimum elevation magnitude across the knickpoint window (metres). Default 10 m
          
          [default: 10]

      --confluence-buffer-cells <CONFLUENCE_BUFFER_CELLS>
          Number of cells to exclude at each segment end (per spec pitfall §8.9: confluences and outlets induce spurious curvature). Default 5
          
          [default: 5]

      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN)
          
          [default: f64]
          [possible values: f32, f64]

      --cell-size-m <CELL_SIZE_M>
          Cell size override in metres. Default: read from raster

      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1

      --raster <RASTER>
          Optional path to write a categorical raster (0=no knickpoint, 1=concave, 2=convex) alongside the GeoJSON points

      --keep-crs
          Preserve the source raster's CRS in the GeoJSON output and declare it via a legacy `crs` member. Default: reproject to WGS84 (RFC 7946 compliant)

  -h, --help
          Print help (see a summary with '-h')
```

## `fluvial concavity` {#concavity}

```text
Concavity index θ per basin (Perron & Royden 2013).

Grid-search θ ∈ [low, high] minimising the residual RMSE of elevation~χ regression. Bootstrap (default n=200) for a 95% CI. Output is a CSV table with one row per qualifying basin.

Usage: surtgis fluvial concavity [OPTIONS] <STREAM> <FLOW_DIR> <FLOW_ACC> <DEM> <BASINS> <OUTPUT>

Arguments:
  <STREAM>
          Binary stream network raster

  <FLOW_DIR>
          D8 flow direction raster

  <FLOW_ACC>
          Flow accumulation raster (cell counts)

  <DEM>
          DEM (elevation in metres)

  <BASINS>
          Basin id raster (e.g. from `surtgis hydrology watershed`), i32 with 0 = no basin

  <OUTPUT>
          Output CSV path. Columns: basin_id, theta_opt, theta_ci_low, theta_ci_high, n_cells, rmse

Options:
      --theta-range <THETA_RANGE>
          θ search range as comma-separated (low,high). Default "0.1,0.9"
          
          [default: 0.1,0.9]

  -v, --verbose
          Verbose output

      --compress
          Compress output GeoTIFFs (deflate)

      --theta-step <THETA_STEP>
          θ grid step. Default 0.05
          
          [default: 0.05]

      --bootstrap-n <BOOTSTRAP_N>
          Bootstrap iterations for the 95% CI. Default 200; set to 0 to skip bootstrap (CI returned as [theta_opt, theta_opt])
          
          [default: 200]

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

      --min-basin-cells <MIN_BASIN_CELLS>
          Minimum basin stream-cell count to attempt an estimate. Default 30
          
          [default: 30]

      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN)
          
          [default: f64]
          [possible values: f32, f64]

      --seed <SEED>
          Seed for the deterministic bootstrap resampler. Default 42
          
          [default: 42]

      --cell-size-m <CELL_SIZE_M>
          Cell size override in metres

      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1

  -h, --help
          Print help (see a summary with '-h')
```

## `fluvial divide-migration` {#divide-migration}

```text
Divide-migration metrics (Willett 2014, Whipple 2017).

For each pair of adjacent watershed basins, computes the median asymmetry across the divide: Gilbert elevation/relief difference and (optionally) χ across-divide difference. Output is a GeoJSON LineString FeatureCollection — one feature per adjacent-basin pair.

Usage: surtgis fluvial divide-migration [OPTIONS] <BASINS> <DEM> <FLOW_ACC> <OUTPUT>

Arguments:
  <BASINS>
          Basin id raster (i32, 0 = no basin)

  <DEM>
          DEM (elevation in metres)

  <FLOW_ACC>
          Flow accumulation raster (currently informational; reserved for future area-weighted statistics)

  <OUTPUT>
          Output GeoJSON path

Options:
      --chi <CHI>
          Optional pre-computed χ raster from `surtgis fluvial chi`. When supplied, median_chi_diff is reported; otherwise NaN

  -v, --verbose
          Verbose output

      --compress
          Compress output GeoTIFFs (deflate)

      --min-divide-length-m <MIN_DIVIDE_LENGTH_M>
          Minimum cumulative divide polyline length (m) to report. Default 500 m — short divides are rarely robust
          
          [default: 500]

      --cell-size-m <CELL_SIZE_M>
          Cell size override in metres

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --keep-crs
          Preserve the source raster's CRS in the GeoJSON output and declare it via a legacy `crs` member. Default: reproject to WGS84 (RFC 7946 compliant)

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

