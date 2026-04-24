# Mosaic rasters in different CRSes

`surtgis mosaic` assumes all inputs share a CRS and grid. If they don't,
you'll hit an error. Two recommended approaches.

## Pre-reproject everything to a target CRS

Pick a reference raster (or create an empty one) in your target CRS, then
use `resample` on each input to align them:

```bash
for f in tile_*.tif; do
    surtgis resample "$f" "aligned_$f" --reference reference_utm.tif
done
surtgis mosaic aligned_*.tif out.tif
```

`resample` handles both the CRS change and the grid alignment in one step,
so `aligned_*.tif` are all on the same grid by construction.

## Use `stac composite` if the sources are on a STAC catalog

The multi-UTM case that drove SurtGIS 0.6.1's critical fix is exactly
this: a bbox spanning two UTM zones, with different scenes in each zone.
`stac composite` handles it automatically by reprojecting each tile's
bbox into the tile's native CRS before reading, then unifying at mosaic
time. You don't think about zone boundaries:

```bash
surtgis stac composite \
    --bbox=-71.5,-35.5,-70.5,-34.5 \   # spans UTM 19S and 18S
    --asset red,nir \
    ...
```

## If you really want to mosaic mixed-CRS files yourself

Reproject them all to a common CRS first (either via `resample --reference`
or `gdalwarp`), then `surtgis mosaic`. There is no "mosaic with automatic
reprojection" command; deliberately so, because implicit reprojection
during mosaic usually produces surprising seams when the target grid isn't
chosen carefully.

## Overlap handling

`surtgis mosaic` uses NaN-aware last-write-wins: a source pixel only
overwrites the output if it is finite. This means irregular tile shapes
(Sentinel-2 scenes with masked borders, Landsat fill pixels) do not
overwrite valid data from adjacent tiles. Order of the input arguments
matters only for genuinely overlapping valid regions — there, the last
file wins.
