# `surtgis embeddings`

_Foundation-model embeddings (AlphaEarth, TESSERA, your own encoder) as rasters: similarity maps, PCA false colour, norms_

## Overview

```text
Foundation-model embeddings (AlphaEarth, TESSERA, your own encoder) as rasters: similarity maps, PCA false colour, norms

Usage: surtgis embeddings [OPTIONS] <COMMAND>

Commands:
  info        Describe a stack: dimensions, valid vectors, L2 norm statistics (is it unit-normalised?), variance captured by the first PCs
  similarity  Similarity of every cell to a reference vector ("more like this")
  pca         Principal components of the field: fit on a sample, project every cell, write one band per component (PC1..PCn; view the first three as RGB)
  locate      Find the AlphaEarth tile(s) covering a location in the product's GeoParquet index (`aef_index.parquet`, next to the annual folders)
  norm        L2 norm of the vector at every cell
  help        Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `embeddings info` {#info}

```text
Describe a stack: dimensions, valid vectors, L2 norm statistics (is it unit-normalised?), variance captured by the first PCs

Usage: surtgis embeddings info [OPTIONS] <INPUT>

Arguments:
  <INPUT>  Embedding stack (multi-band GeoTIFF/COG)

Options:
      --bbox <MINX,MINY,MAXX,MAXY>   Read only this window, `min_x,min_y,max_x,max_y` in the stack's CRS (an AlphaEarth tile is 8192² × 64 bands: 34 GB as f64)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --dequantize <DEQUANTIZE>      Stored-value code: none | alphaearth | linear:SCALE[,OFFSET] [default: none]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `embeddings similarity` {#similarity}

```text
Similarity of every cell to a reference vector ("more like this")

Usage: surtgis embeddings similarity [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Embedding stack (multi-band GeoTIFF/COG)
  <OUTPUT>  Output similarity raster (Float64)

Options:
      --bbox <MINX,MINY,MAXX,MAXY>   Read only this window, `min_x,min_y,max_x,max_y` in the stack's CRS (an AlphaEarth tile is 8192² × 64 bands: 34 GB as f64)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --dequantize <DEQUANTIZE>      Stored-value code: none | alphaearth | linear:SCALE[,OFFSET]. AlphaEarth int8 tiles need `alphaearth` (non-linear code) [default: none]
      --ref-lonlat <LON,LAT>         Reference as a longitude,latitude (EPSG:4326) inside the stack
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --ref-cell <ROW,COL>           Reference as a row,col cell of the stack
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --ref-vec <V1,V2,...>          Reference as an explicit vector, one value per band
      --metric <METRIC>              cosine (scale-free, [-1,1]) | dot | euclidean [default: cosine]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `embeddings pca` {#pca}

```text
Principal components of the field: fit on a sample, project every cell, write one band per component (PC1..PCn; view the first three as RGB)

Usage: surtgis embeddings pca [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Embedding stack (multi-band GeoTIFF/COG)
  <OUTPUT>  Output stack of component scores

Options:
      --bbox <MINX,MINY,MAXX,MAXY>   Read only this window, `min_x,min_y,max_x,max_y` in the stack's CRS (an AlphaEarth tile is 8192² × 64 bands: 34 GB as f64)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --dequantize <DEQUANTIZE>      Stored-value code: none | alphaearth | linear:SCALE[,OFFSET] [default: none]
      --components <COMPONENTS>      Components to keep [default: 3]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --samples <SAMPLES>            Vectors used for the fit (regular subsample) [default: 200000]
      --model <MODEL>                Reuse a model saved with --save-model instead of fitting (same colours across tiles and years)
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
      --save-model <SAVE_MODEL>      Save the fitted model as JSON
  -h, --help                         Print help
```

## `embeddings locate` {#locate}

```text
Find the AlphaEarth tile(s) covering a location in the product's GeoParquet index (`aef_index.parquet`, next to the annual folders)

Usage: surtgis embeddings locate [OPTIONS] --index <INDEX> --lon <LON> --lat <LAT>

Options:
      --index <INDEX>                `aef_index.parquet` (download once from https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --lon <LON>                    Longitude (EPSG:4326)
      --lat <LAT>                    Latitude (EPSG:4326)
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --year <YEAR>                  Year (default: every year in the index)
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

## `embeddings norm` {#norm}

```text
L2 norm of the vector at every cell

Usage: surtgis embeddings norm [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Embedding stack (multi-band GeoTIFF/COG)
  <OUTPUT>  Output raster (Float64)

Options:
      --bbox <MINX,MINY,MAXX,MAXY>   Read only this window, `min_x,min_y,max_x,max_y` in the stack's CRS (an AlphaEarth tile is 8192² × 64 bands: 34 GB as f64)
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --dequantize <DEQUANTIZE>      Stored-value code: none | alphaearth | linear:SCALE[,OFFSET] [default: none]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
  -h, --help                         Print help
```

