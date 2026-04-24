# `surtgis pipeline`

_Pipeline: integrated workflows for specific use cases_

## Overview

```text
Pipeline: integrated workflows for specific use cases

Usage: surtgis pipeline [OPTIONS] <COMMAND>

Commands:
  susceptibility  Compute susceptibility factors from DEM + S2 imagery
  features        Generate geomorphometric feature stack from DEM
  temporal        End-to-end temporal analysis: STAC download → spectral index → trend/stats/phenology
  help            Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `pipeline susceptibility` {#susceptibility}

```text
Compute susceptibility factors from DEM + S2 imagery

Usage: surtgis pipeline susceptibility [OPTIONS] --dem <DEM> --s2 <S2> --bbox <BBOX> --datetime <DATETIME> --outdir <OUTDIR>

Options:
      --dem <DEM>                DEM source: STAC collection ID (cop-dem-glo-30, cop-dem-glo-90, nasadem, 3dep-seamless) or local file path
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --s2 <S2>                  S2 source: "sentinel-2-l2a" or "earth-search" or "skip"
      --bbox <BBOX>              Bounding box: "west,south,east,north"
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --datetime <DATETIME>      Date range: "YYYY-MM-DD/YYYY-MM-DD"
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --outdir <OUTDIR>          Output directory (will be created)
      --max-scenes <MAX_SCENES>  Max scenes for S2 (default: 12) [default: 12]
      --scl-keep <SCL_KEEP>      Cloud mask classes to keep for S2 (default: "4,5,6,11") [default: 4,5,6,11]
  -h, --help                     Print help
```

## `pipeline features` {#features}

```text
Generate geomorphometric feature stack from DEM

Usage: surtgis pipeline features [OPTIONS] --outdir <OUTDIR> <INPUT>

Arguments:
  <INPUT>  Input DEM file

Options:
  -o, --outdir <OUTDIR>          Output directory
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --skip-hydrology           Skip hydrology features (faster)
      --extras                   Include extra features (valley depth, surface area ratio, landform, wind exposure, accumulation zones)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `pipeline temporal` {#temporal}

```text
End-to-end temporal analysis: STAC download → spectral index → trend/stats/phenology

Usage: surtgis pipeline temporal [OPTIONS] --bbox <BBOX> --collection <COLLECTION> --datetime <DATETIME> --index <INDEX> --analysis <ANALYSIS> --outdir <OUTDIR>

Options:
      --catalog <CATALOG>        STAC catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL [default: es]
  -v, --verbose                  Verbose output
      --bbox <BBOX>              Bounding box: west,south,east,north
      --compress                 Compress output GeoTIFFs (deflate)
      --collection <COLLECTION>  Collection (e.g. "sentinel-2-l2a", "landsat-c2-l2")
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --datetime <DATETIME>      Date range: "YYYY-MM-DD/YYYY-MM-DD"
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --interval <INTERVAL>      Temporal interval: "monthly", "biweekly", "weekly", "quarterly", "yearly", or custom days [default: monthly]
      --index <INDEX>            Spectral index to compute: ndvi, ndwi, mndwi, nbr, savi, evi, evi2, bsi, ndbi, ndmi, ndsi, gndvi, ngrdi, ndre, msavi
      --analysis <ANALYSIS>      Analysis type (comma-separated): stats, trend, phenology
      --method <METHOD>          Trend method (when analysis includes "trend"): linear, mann-kendall [default: linear]
      --threshold <THRESHOLD>    Phenology threshold for SOS/EOS (0-1) [default: 0.5]
      --smooth <SMOOTH>          Phenology smoothing window (odd number) [default: 5]
      --stats <STATS>            Statistics to compute (when analysis includes "stats") [default: mean,std,min,max,count]
      --max-scenes <MAX_SCENES>  Maximum scenes per interval window [default: 8]
      --outdir <OUTDIR>          Output directory
      --keep-intermediates       Keep per-interval index rasters as intermediate outputs
      --align-to <ALIGN_TO>      Align output to this reference raster's grid (e.g., a DEM)
  -h, --help                     Print help
```

