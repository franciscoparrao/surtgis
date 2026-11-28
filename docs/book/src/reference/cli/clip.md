# `surtgis clip`

_Clip a raster by polygon or bounding box_

## Synopsis

```text
Clip a raster by polygon or bounding box

With --polygon, cells outside the polygons become NaN and the output is cropped to the polygons' extent (like `gdalwarp -crop_to_cutline`); --keep-extent keeps the input grid and only masks. Polygons and bbox must use the raster's CRS. A clip that leaves no valid cell fails.

Usage: surtgis clip [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>
          Input raster file

  <OUTPUT>
          Output file

Options:
      --polygon <POLYGON>
          Vector file with polygon(s) (.geojson, .shp, .gpkg)

  -v, --verbose
          Verbose output

      --bbox <BBOX>
          Bounding box: xmin,ymin,xmax,ymax (same CRS as input)

      --compress
          Compress output GeoTIFFs (deflate)

      --keep-extent
          With --polygon: mask only, keeping the input's full extent

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
