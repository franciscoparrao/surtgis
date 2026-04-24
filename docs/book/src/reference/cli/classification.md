# `surtgis classification`

_Classification: unsupervised/supervised raster classification (k-means, PCA, etc.)_

## Overview

```text
Classification: unsupervised/supervised raster classification (k-means, PCA, etc.)

Usage: surtgis classification [OPTIONS] <COMMAND>

Commands:
  kmeans          K-means unsupervised clustering
  isodata         ISODATA adaptive clustering (auto-splits/merges clusters)
  pca             Principal Component Analysis (multi-band input)
  min-distance    Minimum distance supervised classification
  max-likelihood  Maximum likelihood supervised classification
  help            Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `classification kmeans` {#kmeans}

```text
K-means unsupervised clustering

Usage: surtgis classification kmeans [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output classified raster

Options:
  -c, --classes <CLASSES>          Number of clusters [default: 5]
  -v, --verbose                    Verbose output
      --compress                   Compress output GeoTIFFs (deflate)
      --max-iter <MAX_ITER>        Maximum iterations [default: 100]
      --convergence <CONVERGENCE>  Convergence threshold [default: 0.001]
      --streaming                  Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>    Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --seed <SEED>                Random seed [default: 42]
  -h, --help                       Print help
```

## `classification isodata` {#isodata}

```text
ISODATA adaptive clustering (auto-splits/merges clusters)

Usage: surtgis classification isodata [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output classified raster

Options:
  -c, --classes <CLASSES>          Initial number of clusters [default: 5]
  -v, --verbose                    Verbose output
      --compress                   Compress output GeoTIFFs (deflate)
      --min-classes <MIN_CLASSES>  Minimum clusters [default: 2]
      --max-classes <MAX_CLASSES>  Maximum clusters [default: 10]
      --streaming                  Force streaming mode for large rasters (auto-detected if >500MB)
      --max-iter <MAX_ITER>        Maximum iterations [default: 50]
      --max-memory <MAX_MEMORY>    Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                       Print help
```

## `classification pca` {#pca}

```text
Principal Component Analysis (multi-band input)

Usage: surtgis classification pca [OPTIONS] --bands <BANDS> <OUTPUT>

Arguments:
  <OUTPUT>  Output directory for PC rasters

Options:
      --bands <BANDS>            Input raster bands (comma-separated paths)
  -v, --verbose                  Verbose output
  -c, --components <COMPONENTS>  Number of components (default: all)
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `classification min-distance` {#min-distance}

```text
Minimum distance supervised classification

Usage: surtgis classification min-distance [OPTIONS] --training <TRAINING> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output classified raster

Options:
      --training <TRAINING>      Training raster (class labels)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `classification max-likelihood` {#max-likelihood}

```text
Maximum likelihood supervised classification

Usage: surtgis classification max-likelihood [OPTIONS] --training <TRAINING> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster
  <OUTPUT>  Output classified raster

Options:
      --training <TRAINING>      Training raster (class labels)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

