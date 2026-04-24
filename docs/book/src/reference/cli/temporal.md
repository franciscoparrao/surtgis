# `surtgis temporal`

_Temporal analysis: trend, anomaly, phenology, statistics_

## Overview

```text
Temporal analysis: trend, anomaly, phenology, statistics

Usage: surtgis temporal [OPTIONS] <COMMAND>

Commands:
  stats      Per-pixel temporal statistics (mean, std, min, max, count, percentile)
  trend      Pixel-wise linear trend analysis (OLS regression)
  change     Change detection between two dates
  anomaly    Anomaly detection vs reference period
  phenology  Vegetation phenology metrics from NDVI/EVI time series
  help       Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `temporal stats` {#stats}

```text
Per-pixel temporal statistics (mean, std, min, max, count, percentile)

Usage: surtgis temporal stats [OPTIONS] --input <INPUT> --outdir <OUTDIR>

Options:
  -i, --input <INPUT>            Input rasters (time-ordered GeoTIFFs)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -o, --outdir <OUTDIR>          Output directory for statistic rasters
      --stats <STATS>            Which statistics to compute (comma-separated): mean,std,min,max,count,p10,p25,p50,p75,p90 [default: mean,std,min,max,count]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `temporal trend` {#trend}

```text
Pixel-wise linear trend analysis (OLS regression)

Usage: surtgis temporal trend [OPTIONS] --input <INPUT> --outdir <OUTDIR>

Options:
  -i, --input <INPUT>            Input rasters (time-ordered GeoTIFFs)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -o, --outdir <OUTDIR>          Output directory for trend rasters (slope, intercept, r2, pvalue)
      --method <METHOD>          Method: "linear" (OLS) or "mann-kendall" (non-parametric) [default: linear]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --times <TIMES>            Time values (comma-separated, e.g. fractional years). If omitted, uses 0,1,2,...
  -h, --help                     Print help
```

## `temporal change` {#change}

```text
Change detection between two dates

Usage: surtgis temporal change [OPTIONS] --before <BEFORE> --after <AFTER> <OUTPUT>

Arguments:
  <OUTPUT>  Output difference raster

Options:
      --before <BEFORE>
          Before raster (time T1)
  -v, --verbose
          Verbose output
      --after <AFTER>
          After raster (time T2)
      --compress
          Compress output GeoTIFFs (deflate)
      --decrease-threshold <DECREASE_THRESHOLD>
          Threshold for significant decrease [default: -1.0]
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --increase-threshold <INCREASE_THRESHOLD>
          Threshold for significant increase [default: 1.0]
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help
          Print help
```

## `temporal anomaly` {#anomaly}

```text
Anomaly detection vs reference period

Usage: surtgis temporal anomaly [OPTIONS] --reference <REFERENCE> --target <TARGET> --outdir <OUTDIR>

Options:
      --reference <REFERENCE>    Reference period rasters (baseline, at least 2)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --target <TARGET>          Target rasters to evaluate
  -o, --outdir <OUTDIR>          Output directory
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --method <METHOD>          Method: "zscore", "difference", or "percent" [default: zscore]
  -h, --help                     Print help
```

## `temporal phenology` {#phenology}

```text
Vegetation phenology metrics from NDVI/EVI time series

Usage: surtgis temporal phenology [OPTIONS] --input <INPUT> --outdir <OUTDIR>

Options:
  -i, --input <INPUT>            Input rasters (time-ordered NDVI/EVI GeoTIFFs, at least 6)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
  -o, --outdir <OUTDIR>          Output directory for phenology rasters (sos, eos, peak, amplitude, etc.)
      --doys <DOYS>              Day-of-year for each input (comma-separated). If omitted, uses indices
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --threshold <THRESHOLD>    Threshold for SOS/EOS as fraction of amplitude (0-1) [default: 0.5]
      --smooth <SMOOTH>          Smoothing window size (odd number, 0=none) [default: 5]
  -h, --help                     Print help
```

