# Reproject between UTM zones

SurtGIS 0.7.0 does not ship a standalone reproject command yet. The
recommended paths today, in order of preference:

## 1. Use `resample --reference` if you have a template raster

If you already have a raster in the target CRS (for example, you're
aligning a DEM to an existing Sentinel-2 tile):

```bash
surtgis resample source.tif target.tif --reference template_in_target_crs.tif
```

`resample` handles CRS mismatch: it reprojects `source.tif` into the
reference's CRS, resamples to the reference's grid (origin, pixel size,
dimensions), and writes the result.

Supported methods via `--method`: `bilinear` (default, good for continuous
data like elevation or NDVI), `nearest` (for categorical data like
classification results), `cubic`.

## 2. Use STAC's on-the-fly reprojection for cloud data

If the source is a STAC asset, many collections are served natively in
multiple projections. `stac composite` and `stac fetch-mosaic` accept
`--align-to path/to/grid.tif` and reproject-on-read directly:

```bash
surtgis stac fetch-mosaic \
    --catalog pc \
    --collection cop-dem-glo-30 \
    --bbox=... \
    --align-to reference_utm.tif \
    dem_aligned.tif
```

This is usually faster than download-then-reproject because only the tiles
intersecting the target grid are fetched.

## 3. Fall back to `gdalwarp` for a one-shot reprojection

For a standalone reprojection with no template, the fastest thing today is
GDAL:

```bash
gdalwarp -t_srs EPSG:32719 -tr 30 30 -r bilinear dem_wgs84.tif dem_utm.tif
```

SurtGIS and GDAL interoperate fine through GeoTIFFs — no data loss either
direction.

## Why SurtGIS doesn't have its own reproject yet

The `proj4rs` crate gives us the math, and `surtgis-cloud` already uses it
for STAC bbox reprojection. A standalone `reproject` command is a
straightforward wrapper around what already exists — it just hasn't hit
the top of the priority queue. Track the issue at
[#reproject-command](https://github.com/franciscoparrao/surtgis/issues).

## Choosing a target CRS

For terrain analysis at mid-latitudes, use the local UTM zone. Find it
with the formula:

```
zone = floor((longitude + 180) / 6) + 1
hemisphere = "S" if latitude < 0 else "N"
```

Chile around Curicó (−71°, −35°) → zone 19 South → EPSG:32719.
Spain around Madrid (−3°, 40°) → zone 30 North → EPSG:32630.

For very large areas that span multiple UTM zones, use an equal-area
projection appropriate to your region (Lambert Azimuthal for continental
Europe, Albers Equal Area Conic for continental US, etc.). Terrain slope
computations are less sensitive to distortion than area computations, so
UTM within one zone is almost always fine.
