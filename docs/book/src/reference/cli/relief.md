# `surtgis relief`

_Rayshader-style shaded relief composite to PNG_

## Synopsis

```text
Rayshader-style shaded relief composite to PNG

Usage: surtgis relief [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input DEM file
  <OUTPUT>  Output PNG file

Options:
      --colormap <COLORMAP>          Color scheme for the base layer [default: terrain]
  -v, --verbose                      Verbose output
      --compress                     Compress output GeoTIFFs (deflate)
      --sun-azimuth <SUN_AZIMUTH>    Sun azimuth in degrees (0 = North, clockwise) [default: 315]
      --streaming                    Force streaming mode for large rasters (auto-detected if >500MB)
      --sun-altitude <SUN_ALTITUDE>  Sun altitude in degrees above the horizon [default: 45]
      --max-memory <MAX_MEMORY>      Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --shadows                      Enable ray-traced cast shadows
      --output-dtype <OUTPUT_DTYPE>  Sample type of Float64 outputs: `f64` (default) or `f32` (half the size; nodata stays NaN) [default: f64] [possible values: f32, f64]
      --soft <SOFT>                  Number of sun samples for the soft-shadow penumbra (>=2 enables soft shadows). Default 1 = hard shadows [default: 1]
      --ambient                      Enable ambient occlusion via sky-view factor
      --no-provenance                Do not embed a provenance record (command, hashed inputs, output digest) in the written GeoTIFFs, and skip hashing the inputs. Also honoured as SURTGIS_NO_PROVENANCE=1
      --water                        Detect water bodies (large flat regions) and tint them
      --z-factor <Z_FACTOR>          Vertical exaggeration for the normal-based sphere shade [default: 1.0]
      --radius <RADIUS>              Search radius (cells) for the ambient SVF computation [default: 30]
  -h, --help                         Print help
```
