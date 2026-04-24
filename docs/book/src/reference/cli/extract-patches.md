# `surtgis extract-patches`

_Extract raster patches centered on points or sampled from polygons for CNN training_

## Synopsis

```text
Extract raster patches centered on points or sampled from polygons for CNN training

Usage: surtgis extract-patches [OPTIONS] --features-dir <FEATURES_DIR> --label-col <LABEL_COL> <OUTPUT>

Arguments:
  <OUTPUT>  Output directory (creates patches.npy, labels.npy, manifest.csv, meta.json)

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
  -h, --help
          Print help
```
