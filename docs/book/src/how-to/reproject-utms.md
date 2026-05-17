# Reproject a raster between CRSes

SurtGIS 0.7.4 added the `surtgis reproject` command, which uses
`proj4rs` for the coordinate transform and a Rayon-parallelised inverse
mapping for the per-pixel sampling. There is no system GDAL dependency
— this works from a single static `surtgis` binary.

## The 30-second version

```bash
# UTM 19S → WGS84 lat/lon
surtgis reproject dem_utm.tif dem_wgs84.tif --to EPSG:4326

# WGS84 → UTM 19S, preserving roughly the same resolution
surtgis reproject dem_wgs84.tif dem_utm.tif --to EPSG:32719

# UTM 19S → Web Mercator at 30 m, nearest-neighbour (for categorical
# rasters like a classification result)
surtgis reproject classes.tif classes_mercator.tif \
    --to EPSG:3857 --pixel-size 30 --method nearest
```

The output pixel size is auto-inferred when both CRSes are the same kind
(metric ↔ metric, or geographic ↔ geographic). For unit-changing
reprojections you can pass `--pixel-size <X>` in target-CRS units.

## Required flags

- `--to EPSG:XXXX` (or just `--to XXXX`): the target CRS. Any EPSG code
  supported by proj4rs works (UTM zones, Web Mercator 3857, Lambert,
  national grids, etc.).

## Optional flags

- `--from EPSG:XXXX`: override the source CRS when the input GeoTIFF
  doesn't embed one. Most modern GeoTIFFs do, in which case you can
  omit this.
- `--method nearest|bilinear`: resampling kernel. Default is `bilinear`
  which is right for continuous data (elevation, NDVI, reflectance).
  Use `nearest` for categorical data like geomorphons or land-cover
  classifications, where interpolating values across class boundaries
  would produce meaningless "tween" classes.
- `--pixel-size <X>`: output pixel size in target-CRS units (metres
  for projected CRSes, degrees for geographic). When omitted, SurtGIS
  picks a sensible default that preserves the source resolution.

## What it produces

```
$ surtgis reproject benchmarks/results/dems/fbm_1000_raw.tif out.tif --to EPSG:4326
Reprojecting EPSG:32719 → EPSG:4326 (1000, bilinear method)
✓ wrote out.tif (649×1001) in 0.13 s
```

On the i7-1270P benchmark machine: a 1 000 × 1 000 UTM 19S DEM
reprojects to WGS84 in 0.13 s, and to Web Mercator in 0.59 s
(the output grid is bigger because Web Mercator stretches near
the poles). The reproject is parallelised across output rows, so
multi-core machines scale roughly linearly until memory bandwidth
saturates.

## Same-CRS shortcut

If the source and target CRSes are the same, SurtGIS writes the
input verbatim and tells you what it did:

```
$ surtgis reproject foo.tif foo_copy.tif --to EPSG:32719
source and target CRS are the same (EPSG:32719); copying input to output
✓ wrote foo_copy.tif in 0.04 s
```

This makes `reproject` safe to put in scripts where the source CRS
varies; the no-op case is cheap.

## Alternative paths

The `reproject` command is the right tool for a standalone
reprojection. For two adjacent use cases, prefer the dedicated
shortcut:

### 1. Aligning to a template raster

If you already have a raster in the target CRS and want the output
on the same grid (same origin, pixel size, dimensions), use
`resample`:

```bash
surtgis resample source.tif aligned.tif --reference template.tif
```

`resample` handles CRS mismatch internally; you don't need to call
`reproject` first.

### 2. Pulling cloud data already in the target CRS

If the source is a STAC asset, the cloud commands accept `--align-to`
and reproject-on-read so only the tiles intersecting the target grid
are fetched:

```bash
surtgis stac composite \
    --catalog pc \
    --collection cop-dem-glo-30 \
    --bbox=... \
    --align-to reference_utm.tif \
    dem_aligned.tif
```

This is usually faster than `download → reproject` because only the
tiles within the target extent are pulled.

## Choosing a target CRS

For terrain analysis at mid-latitudes, the local UTM zone is almost
always the right choice. The formula:

```
zone = floor((longitude + 180) / 6) + 1
hemisphere = "S" if latitude < 0 else "N"
EPSG = 32600 + zone   (Northern hemisphere)
EPSG = 32700 + zone   (Southern hemisphere)
```

Examples:
- Chile around Curicó (−71°, −35°) → zone 19 South → **EPSG:32719**.
- Spain around Madrid (−3°, 40°) → zone 30 North → **EPSG:32630**.
- Western US around Los Angeles (−118°, 34°) → zone 11 North →
  **EPSG:32611**.

For continental-scale areas that span multiple UTM zones, use an
equal-area projection appropriate to your region (Lambert Azimuthal
for continental Europe, Albers Equal Area Conic for the continental
US, etc.). Terrain slope is less sensitive to projection distortion
than area, so UTM within a single zone is almost always good enough
for analyses smaller than ~500 km across.

## Caveats

- Only `nearest` and `bilinear` are implemented today. Cubic
  resampling is on the roadmap. For elevation and most reflectance
  data, bilinear is the right default; cubic's main advantage is
  smoother contour rendering, which `reproject` is not the right tool
  for anyway.
- The output extent is computed by transforming the source raster's
  four corners plus three edge midpoints, which covers most
  projections. Highly curved projections (large-area polar or
  cylindrical projections crossing the antimeridian) may need
  `--pixel-size` set explicitly.
- For geographic-to-metric reprojections without `--pixel-size`,
  SurtGIS infers a default that roughly preserves the output
  column count. If you need an exact resolution (say 30 m to match
  Copernicus DEM), pass it explicitly.
