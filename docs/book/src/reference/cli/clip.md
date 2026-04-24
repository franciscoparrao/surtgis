# `surtgis clip`

_Clip a raster by polygon or bounding box_

## Synopsis

```text
Clip a raster by polygon or bounding box

Usage: surtgis clip [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster file
  <OUTPUT>  Output file

Options:
      --polygon <POLYGON>        Vector file with polygon(s) (.geojson, .shp, .gpkg)
  -v, --verbose                  Verbose output
      --bbox <BBOX>              Bounding box: xmin,ymin,xmax,ymax (same CRS as input)
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```
