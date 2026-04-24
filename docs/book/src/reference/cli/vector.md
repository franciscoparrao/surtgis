# `surtgis vector`

_Vector geoprocessing: intersection, union, difference, dissolve, buffer_

## Overview

```text
Vector geoprocessing: intersection, union, difference, dissolve, buffer

Usage: surtgis vector [OPTIONS] <COMMAND>

Commands:
  intersection    Geometric intersection of two vector layers (A ∩ B)
  union           Geometric union of two vector layers (A ∪ B)
  difference      Geometric difference: features of A not covered by B (A - B)
  sym-difference  Symmetric difference: areas in A or B but not both (A ⊕ B)
  dissolve        Dissolve all features into a single geometry
  buffer          Buffer features by a distance
  help            Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `vector intersection` {#intersection}

```text
Geometric intersection of two vector layers (A ∩ B)

Usage: surtgis vector intersection [OPTIONS] <INPUT_A> <INPUT_B> <OUTPUT>

Arguments:
  <INPUT_A>  Input layer A (GeoJSON/Shapefile/GeoPackage)
  <INPUT_B>  Input layer B (overlay)
  <OUTPUT>   Output file

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `vector union` {#union}

```text
Geometric union of two vector layers (A ∪ B)

Usage: surtgis vector union [OPTIONS] <INPUT_A> <INPUT_B> <OUTPUT>

Arguments:
  <INPUT_A>  
  <INPUT_B>  
  <OUTPUT>   

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `vector difference` {#difference}

```text
Geometric difference: features of A not covered by B (A - B)

Usage: surtgis vector difference [OPTIONS] <INPUT_A> <INPUT_B> <OUTPUT>

Arguments:
  <INPUT_A>  
  <INPUT_B>  
  <OUTPUT>   

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `vector sym-difference` {#sym-difference}

```text
Symmetric difference: areas in A or B but not both (A ⊕ B)

Usage: surtgis vector sym-difference [OPTIONS] <INPUT_A> <INPUT_B> <OUTPUT>

Arguments:
  <INPUT_A>  
  <INPUT_B>  
  <OUTPUT>   

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `vector dissolve` {#dissolve}

```text
Dissolve all features into a single geometry

Usage: surtgis vector dissolve [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input layer
  <OUTPUT>  Output file

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `vector buffer` {#buffer}

```text
Buffer features by a distance

Usage: surtgis vector buffer [OPTIONS] --distance <DISTANCE> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input layer
  <OUTPUT>  Output file

Options:
      --distance <DISTANCE>      Buffer distance (in CRS units)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --segments <SEGMENTS>      Number of segments per quarter circle (default: 8) [default: 8]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

