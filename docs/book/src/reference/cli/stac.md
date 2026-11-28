# `surtgis stac`

_Search and fetch data from STAC catalogs (Planetary Computer, Earth Search)_

## Overview

```text
Search and fetch data from STAC catalogs (Planetary Computer, Earth Search)

Usage: surtgis stac [OPTIONS] <COMMAND>

Commands:
  search            Search a STAC catalog and list matching items
  fetch             Search a STAC catalog, fetch the first COG asset, and save to a GeoTIFF
  fetch-mosaic      Search STAC catalog, fetch ALL matching COG assets, mosaic, and save
  composite         End-to-end satellite composite: search -> mosaic per date -> cloud-mask -> median composite
  time-series       Download a time series: one cloud-free composite per interval (monthly, biweekly, etc.)
  download-climate  Download climate data (Zarr) for a region and time range
  list-catalogs     List all available STAC catalogs (curated + indexed)
  list-collections  List collections available in a STAC catalog
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

## `stac search` {#search}

```text
Search a STAC catalog and list matching items

Usage: surtgis stac search [OPTIONS]

Options:
      --catalog <CATALOG>            Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL [default: es]
  -v, --verbose                      Verbose output
      --bbox <BBOX>                  Bounding box: west,south,east,north
      --compress                     Compress output GeoTIFFs (deflate)
      --datetime <DATETIME>          Datetime or range (e.g. "2024-06-01/2024-06-30")
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --collections <COLLECTIONS>    Collections (comma-separated, e.g. "sentinel-2-l2a") [aliases: --collection]
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --limit <LIMIT>                Maximum items to return [default: 10]
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `stac fetch` {#fetch}

```text
Search a STAC catalog, fetch the first COG asset, and save to a GeoTIFF

Usage: surtgis stac fetch [OPTIONS] --bbox <BBOX> --collection <COLLECTION> <OUTPUT>

Arguments:
  <OUTPUT>  Output GeoTIFF file

Options:
      --catalog <CATALOG>            Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL [default: es]
  -v, --verbose                      Verbose output
      --bbox <BBOX>                  Bounding box: west,south,east,north
      --compress                     Compress output GeoTIFFs (deflate)
      --collection <COLLECTION>      Collection (e.g. "sentinel-2-l2a") [aliases: --collections]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --asset <ASSET>                Asset key to fetch (e.g. "red", "nir", "B04"). Auto-detects COG if omitted
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --datetime <DATETIME>          Datetime or range
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
      --variable <VARIABLE>          Variable name for Zarr stores (e.g. "precipitation_amount_1hour_Accumulation")
      --time-step <TIME_STEP>        Time step for Zarr: "first", "last", or ISO datetime (e.g. "2020-06-15") [default: first]
  -h, --help                         Print help
```

## `stac fetch-mosaic` {#fetch-mosaic}

```text
Search STAC catalog, fetch ALL matching COG assets, mosaic, and save

Usage: surtgis stac fetch-mosaic [OPTIONS] --bbox <BBOX> --collection <COLLECTION> <OUTPUT>

Arguments:
  <OUTPUT>  Output GeoTIFF file

Options:
      --catalog <CATALOG>            Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL [default: es]
  -v, --verbose                      Verbose output
      --bbox <BBOX>                  Bounding box: west,south,east,north
      --compress                     Compress output GeoTIFFs (deflate)
      --collection <COLLECTION>      Collection (e.g. "cop-dem-glo-30", "sentinel-2-l2a") [aliases: --collections]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --asset <ASSET>                Asset key to fetch (e.g. "data", "red", "B04"). Auto-detects COG if omitted
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --datetime <DATETIME>          Datetime or range
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --max-items <MAX_ITEMS>        Maximum items to fetch and mosaic (caps RAM/time on wide queries) [default: 20] [aliases: --max-scenes]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
      --retries <RETRIES>            Attempts per tile before giving it up (transient HTTP errors are retried with exponential backoff) [default: 3]
      --allow-partial                Save the mosaic even if some tiles could not be fetched. Off by default: an incomplete mosaic is an error, because tile-sized NoData holes are indistinguishable from legitimate NoData in the output. With this flag, missing tiles are listed loudly and in a `<output>.partial.json` sidecar
  -h, --help                         Print help
```

## `stac composite` {#composite}

```text
End-to-end satellite composite: search -> mosaic per date -> cloud-mask -> median composite

Usage: surtgis stac composite [OPTIONS] --bbox <BBOX> --collection <COLLECTION> --asset <ASSET> --datetime <DATETIME> <OUTPUT>

Arguments:
  <OUTPUT>  Output GeoTIFF file

Options:
      --catalog <CATALOG>
          Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL [default: es]
  -v, --verbose
          Verbose output
      --bbox <BBOX>
          Bounding box: west,south,east,north
      --compress
          Compress output GeoTIFFs (deflate)
      --collection <COLLECTION>
          Collection (e.g. "sentinel-2-l2a")
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --asset <ASSET>
          Data asset to composite (e.g. "red", "nir", "B04")
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --datetime <DATETIME>
          Datetime range (e.g. "2024-01-01/2024-12-31")
      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --max-scenes <MAX_SCENES>
          Maximum number of temporal scenes to composite [default: 12]
      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
      --scl-asset <SCL_ASSET>
          SCL asset key for cloud masking [default: scl]
      --scl-keep <SCL_KEEP>
          SCL classes to keep (comma-separated, default: vegetation,soil,water,snow) [default: 4,5,6,11]
      --align-to <ALIGN_TO>
          Align output to this raster's grid (resamples to match origin, cell size, dims)
      --naming <NAMING>
          Multi-band output naming: "prefix" → {stem}_{band}.tif (default), "asset" → {band}.tif [default: prefix]
      --cache
          Cache downloaded COG tiles locally (~/.cache/surtgis/cog/) for fast re-runs
      --strip-rows <STRIP_ROWS>
          Rows per processing strip (larger = fewer HTTP requests but more RAM). Default: 512 [default: 512]
      --band-chunk-size <BAND_CHUNK_SIZE>
          Bands to download + process together per scene (RAM↔HTTP dial). Higher = fewer HTTP requests (less rate-limit pressure) but more RAM per strip. 1 = minimum RAM (default). For 38 GB hosts, 3-5 is comfortable with ES; for PC, up to n_bands is fine [default: 1]
      --max-tile-failures <MAX_TILE_FAILURES>
          Abort the composite if more than this many tiles fail after retries (0 = never abort; just print a summary of how many tiles/scenes were affected). Multi-band composites (--asset with commas) only [default: 0]
      --no-harmonize
          Keep Sentinel-2 L2A digital numbers as served. By default tiles with processing baseline >= 04.00 get ESA's BOA_ADD_OFFSET (-1000) applied when the catalog has not (Planetary Computer), so every scene is on the pre-2022 scale: reflectance = DN / 10000
  -h, --help
          Print help
```

## `stac time-series` {#time-series}

```text
Download a time series: one cloud-free composite per interval (monthly, biweekly, etc.)

Usage: surtgis stac time-series [OPTIONS] --bbox <BBOX> --collection <COLLECTION> --asset <ASSET> --datetime <DATETIME> <OUTPUT>

Arguments:
  <OUTPUT>  Output directory (one GeoTIFF per interval)

Options:
      --catalog <CATALOG>            Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL [default: es]
  -v, --verbose                      Verbose output
      --bbox <BBOX>                  Bounding box: west,south,east,north
      --compress                     Compress output GeoTIFFs (deflate)
      --collection <COLLECTION>      Collection (e.g. "sentinel-2-l2a")
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --asset <ASSET>                Data asset to download (e.g. "B04", "nir", "red")
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --datetime <DATETIME>          Full datetime range (e.g. "2020-01-01/2024-12-31")
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --interval <INTERVAL>          Temporal interval: "monthly", "biweekly", "weekly", or custom days (e.g. "30") [default: monthly]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
      --scl-asset <SCL_ASSET>        SCL asset key for cloud masking (use "none" to skip) [default: scl]
      --max-scenes <MAX_SCENES>      Maximum scenes per interval [default: 8]
      --align-to <ALIGN_TO>          Align output to this raster's grid (e.g., a DEM)
      --no-harmonize                 Keep Sentinel-2 L2A digital numbers as served (see `composite`)
  -h, --help                         Print help
```

## `stac download-climate` {#download-climate}

```text
Download climate data (Zarr) for a region and time range

Searches a STAC catalog for climate datasets (ERA5, TerraClimate), aggregates over time intervals, and writes one GeoTIFF per interval.

Usage: surtgis stac download-climate [OPTIONS] --bbox <BBOX> --collection <COLLECTION> --variable <VARIABLE> --datetime <DATETIME> <OUTPUT>

Arguments:
  <OUTPUT>
          Output directory (one GeoTIFF per interval)

Options:
      --catalog <CATALOG>
          Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
          
          [default: pc]

  -v, --verbose
          Verbose output

      --bbox <BBOX>
          Bounding box: west,south,east,north

      --compress
          Compress output GeoTIFFs (deflate)

      --collection <COLLECTION>
          Collection (e.g. "era5-pds")

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

      --variable <VARIABLE>
          Variable (STAC asset / Zarr array) to read; several separated by commas (each variable then goes to its own `<output>/<variable>/` folder)

      --datetime <DATETIME>
          Datetime range (e.g. "2020-01-01/2020-12-31")

      --output-dtype <OUTPUT_DTYPE>
          Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN)
          
          [default: f64]
          [possible values: f32, f64]

      --aggregate <AGGREGATE>
          Temporal aggregation: none, daily-sum, daily-mean, monthly-mean, monthly-sum, yearly-mean, yearly-sum
          
          [default: monthly-mean]

      --no-provenance
          Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1

      --concurrency <CONCURRENCY>
          Stores (STAC items) read at the same time. Climate archives publish one store per period, and each store costs a few latency-bound requests, so reading several at once is the main speed-up
          
          [default: 4]

  -h, --help
          Print help (see a summary with '-h')
```

## `stac list-catalogs` {#list-catalogs}

```text
List all available STAC catalogs (curated + indexed)

Usage: surtgis stac list-catalogs [OPTIONS]

Options:
      --search <SEARCH>              Search for catalogs by keyword (e.g., "sentinel-2", "dem", "thermal")
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `stac list-collections` {#list-collections}

```text
List collections available in a STAC catalog

Usage: surtgis stac list-collections [OPTIONS]

Options:
      --catalog <CATALOG>            Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL [default: pc]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

