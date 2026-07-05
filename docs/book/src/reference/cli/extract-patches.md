# `surtgis extract-patches`

_Extract raster patches centered on points or sampled from polygons for CNN training_

## Synopsis

```text
Extract raster patches centered on points or sampled from polygons for CNN training

Usage: surtgis extract-patches [OPTIONS] --features-dir <FEATURES_DIR> --label-col <LABEL_COL> <OUTPUT>

Arguments:
  <OUTPUT>  Output directory (creates patches.npy or patches.zarr/, plus labels.npy, manifest.csv, meta.json, and optional stac/)

Options:
      --features-dir <FEATURES_DIR>
          Directory with features.json and feature rasters (auto-discovers .tif if no json)
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
      --points <POINTS>
          Vector file with POINTS (.geojson, .shp, .gpkg). Mutually exclusive with --polygons
      --polygons <POLYGONS>
          Vector file with POLYGONS for grid-sampling. Mutually exclusive with --points
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --label-col <LABEL_COL>
          Property name on the vector feature that carries the label
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --size <SIZE>
          Patch side length in pixels (square). Default 256 [default: 256]
      --stride <STRIDE>
          Grid stride in pixels when sampling polygons. Default = size (no overlap)
      --skip-nan-threshold <SKIP_NAN_THRESHOLD>
          Skip patches where fraction of NaN pixels exceeds this threshold. Default 0.1 [default: 0.1]
      --max-patches <MAX_PATCHES>
          Optional random subsample cap (uses deterministic seed)
      --seed <SEED>
          Random seed used for subsampling. Default 42 [default: 42]
      --profile <PROFILE>
          Geospatial Foundation Model profile to target. When set, applies per-band z-score normalization using the model's published stats and validates the band count. Use --size to override tile size. Supported: prithvi-v2, clay-v1.5
      --output-format <OUTPUT_FORMAT>
          Output format for the patch tensor: `npy` (single .npy file, default) or `zarr` (chunked Zarr v2 directory, one chunk per chip). Labels and manifest are always emitted as .npy / .csv regardless [default: npy]
      --emit-stac
          Emit STAC ML-AOI Collection + Items describing the chips as labelled training data. Writes `<output>/stac/collection.json` and `<output>/stac/items/chip_NNNNNN.json`. When a --profile is also set, the Collection embeds the MLM extension declaring the target foundation model
      --points-crs <POINTS_CRS>
          EPSG code of the input vector file (--points or --polygons). Defaults to 4326 — the GeoJSON spec mandates WGS84 lon/lat, and most shapefiles in the wild also use it. When the raster's CRS differs, vector coordinates are reprojected on the fly via proj4rs. Set explicitly when your vector is in projected coords already (e.g. 32718 for UTM 18S) to skip reprojection [default: 4326]
  -h, --help
          Print help
```
