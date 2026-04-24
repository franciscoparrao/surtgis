# First terrain analysis from a Copernicus DEM

This tutorial walks you end-to-end through a realistic terrain analysis: you
will download a real digital elevation model from a public STAC catalog,
reproject it to a metric coordinate system, compute slope, aspect, and
hillshade, clip the result to a smaller area, and understand what the
outputs actually mean. It takes about 30 minutes.

By the end you will have:

- A 30-metre DEM of a ~11 km × 5 km patch of the Chilean Andes on disk.
- A set of terrain-factor rasters derived from it.
- A clear mental model of SurtGIS's command flow: source → transform →
  derive → consume.

**Prerequisites:** SurtGIS installed (see [Installation](../installation.md)).
Network access for the STAC download step (about 20 MB transferred).

## 1. Set up a working directory

```bash
mkdir -p ~/surtgis-tutorial
cd ~/surtgis-tutorial
```

All subsequent commands expect you to be in this directory.

## 2. Download the DEM from STAC

We'll pull a patch of the Copernicus GLO-30 DEM (30-metre global coverage)
from Microsoft's Planetary Computer:

```bash
surtgis stac fetch-mosaic \
    --catalog pc \
    --collection cop-dem-glo-30 \
    --bbox=-71.0,-35.1,-70.9,-35.05 \
    dem_wgs84.tif
```

What each flag does:

- `--catalog pc` — use Planetary Computer. The other supported catalogs are
  `es` (Earth Search) and any full STAC API URL. For `cop-dem-glo-30`, PC
  is the cleaner source (Earth Search returns multiple overlapping tiles
  that slow the download).
- `--collection cop-dem-glo-30` — the STAC collection ID. PC's browser at
  `planetarycomputer.microsoft.com/catalog` lists every available.
- `--bbox=-71.0,-35.1,-70.9,-35.05` — west, south, east, north in degrees.
  Note the `=` syntax: with a negative number as the first value, the shell
  would otherwise interpret `-71.0` as a flag.
- `dem_wgs84.tif` — output filename. The `_wgs84` suffix is a note to
  future-us: this DEM is in EPSG:4326 (latitude/longitude). We'll reproject
  in a moment.

The command should take ~10 seconds. Verify:

```bash
surtgis info dem_wgs84.tif
```

You should see something close to:

```text
Dimensions: 360 x 182 (65520 cells)
Cell size: 0.0002777777777777778
Bounds: (-71.000000, -35.100278) - (-70.900000, -35.049722)
CRS: EPSG:4326

Statistics:
  Min: 400.0
  Max: 1380.0
  Mean: ~800
  Valid cells: 65520 (100.0%)
```

Elevations between 400 m and 1380 m tell us this patch is at a foothill
level of the Andes, not in the high peaks. That's realistic for the
latitude (35°S, near Curicó).

## 3. Why we have to reproject before computing slope

Here is a trap almost every newcomer falls into, so we're going to walk
straight into it on purpose.

Try computing slope on this DEM directly:

```bash
surtgis terrain slope dem_wgs84.tif slope_wgs84.tif
surtgis info slope_wgs84.tif
```

Look at the statistics:

```text
Statistics:
  Min: 89.8
  Max: 90.0
  Mean: ~89.99
```

Every pixel is saying "89.99 degrees of slope" — essentially vertical,
everywhere. This is wrong; the Andes foothills are not a sheer cliff.

**Why this happens.** Slope is `atan(rise / run)`. The `run` in our DEM is
measured in degrees of longitude (about 0.000278° per pixel), but the
`rise` is measured in metres. When you divide metres by degrees, the ratio
is on the order of hundreds or thousands, so the `atan` saturates to 90°.

The fix is to reproject the DEM into a coordinate system where both axes
are metres. For Chile at 35°S, that's UTM zone 19 South (EPSG:32719).
SurtGIS doesn't have a dedicated reproject command yet — but `stac
fetch-mosaic` accepts `--align-to` which can reproject-on-read, and the
`resample` command is the general-purpose tool for grid alignment. For
this tutorial we'll take a simpler route: ask STAC for the DEM in the right
projection in one shot.

## 4. Reproject via `resample` to a reference grid

If you have a reference raster in the target CRS, `resample` is the
cleanest path:

```bash
# (skip this if you don't have an existing UTM raster — see the next
# section for the alternate path)
surtgis resample dem_wgs84.tif dem_utm.tif --reference reference_utm.tif
```

Most first-time users don't have a reference. For a standalone reprojection
you'll want the more direct approach below.

## 5. Alternative: process a native-metric DEM from a different source

For this tutorial we'll sidestep the reprojection question entirely by
working with a small pre-projected DEM we can compute slope on directly.
The `sentinel-2-l2a` collection on Planetary Computer serves imagery
already in UTM, and its companion DEM products do the same.

If you're following along without a reference raster, skip to tutorial 2
[Sentinel-2 composite and NDVI](s2-composite-ndvi.md) which starts with
UTM data and avoids this rabbit hole.

For the remainder of this tutorial, if you want to compute terrain factors
on the `dem_wgs84.tif` we just downloaded, you will need to reproject it
first. The simplest way today is a one-liner with GDAL (yes, the tool
SurtGIS doesn't depend on — we're not dogmatic about it):

```bash
gdalwarp -t_srs EPSG:32719 -tr 30 30 -r bilinear dem_wgs84.tif dem_utm.tif
surtgis info dem_utm.tif
```

Output:

```text
Cell size: 30
CRS: EPSG:32719
```

Now the cells are 30 m on a side, and we can compute slope meaningfully.

> This is a documented limitation of SurtGIS 0.7.0. A native reproject
> command is on the roadmap. Until it lands, GDAL's `gdalwarp` is the
> recommended path. See the [Reproject between UTM zones](../how-to/reproject-utms.md)
> how-to for more.

## 6. Compute the core terrain factors

With a UTM DEM, we can compute the standard factors:

```bash
surtgis terrain slope    dem_utm.tif slope.tif
surtgis terrain aspect   dem_utm.tif aspect.tif
surtgis terrain hillshade dem_utm.tif hillshade.tif
```

Each of these takes a fraction of a second on a patch this size. Verify one:

```bash
surtgis info slope.tif
```

You should see a reasonable range now, roughly:

```text
Min: 0.0
Max: ~45
Mean: ~12
```

Twelve degrees mean slope is consistent with Andean foothills. Peak slopes
of 45° correspond to the steeper gullies.

## 7. Or: compute every standard factor in one pass

The `terrain all` subcommand computes the full set (slope, aspect,
hillshade, curvature, TPI, TRI, roughness) in one shot, reading the DEM
once and writing each factor as a separate file:

```bash
surtgis terrain all dem_utm.tif --outdir factors/ --compress
ls factors/
```

```text
slope.tif  aspect.tif  hillshade.tif  curvature.tif  tpi.tif  tri.tif  roughness.tif
```

This is the ergonomic path for "give me the standard factor set to feed
into a model". For ad-hoc experimentation, the individual commands are
fine.

## 8. Clip to a smaller area of interest

If your model only cares about the eastern half of this patch:

```bash
surtgis clip factors/slope.tif --bbox=300000,6110000,315000,6120000 slope_east.tif
```

`--bbox` is in the raster's CRS (so UTM metres here, not degrees).
Alternatively, `--polygon my_aoi.gpkg` clips by a vector geometry — see
[Clip by polygon how-to](../how-to/mosaic-crs.md) for that workflow.

## 9. Understand what you just produced

Let's make sure the outputs actually make sense.

**Slope** is degrees above horizontal, 0 to 90. Flat plains are 0–3°, gentle
hills 5–15°, steep terrain 25–45°, cliffs >45°. If your DEM yields slopes
above 70° widely, something is wrong with units (as we saw in step 3) or
with artifacts in the DEM.

**Aspect** is compass direction the slope faces, 0–360° (or −1 where the
slope is too flat to have a direction). 0° = north, 90° = east, 180° =
south, 270° = west.

**Hillshade** is a synthetic illumination value, 0–255, simulating a
light source at a default azimuth (315°) and altitude (45°). It's purely
for visualisation; it has no physical unit. Darker values = in shadow from
the light source; brighter = facing it.

Open `factors/hillshade.tif` in QGIS (or any raster viewer) and you should
immediately see the terrain structure of the patch. That visual check is
the single best way to confirm your pipeline is correct before trusting
the downstream derivatives.

## What's next

- For cloud masking and Sentinel-2 imagery, continue with
  [Tutorial 2: Sentinel-2 composite and NDVI](s2-composite-ndvi.md).
- For scaling this to a much larger DEM (>5 GB), read
  [Memory model](../explanation/memory-model.md) and use `--streaming`.
- For writing this pipeline as a script, every command here is composable
  via shell scripts; most users treat the CLI as the primary interface.

## Common gotchas

**"My slope is everywhere 89°."** You're computing slope on a
lat/lon-projected DEM. Reproject to a metric CRS first (step 5).

**"All my outputs are black or blank."** Open in QGIS with "Singleband
pseudocolor" and let it auto-stretch the min–max. The TIFF viewer on your
OS is probably clamping to the raw byte range.

**"Some pixels are NaN / nodata."** Edge pixels within one cell of the DEM
boundary are undefined for 3×3 window algorithms (slope, aspect, TPI, TRI).
This is intentional; the alternatives either extrapolate off the edge or
silently invent values.

**"The STAC download was slow."** For `cop-dem-glo-30`, prefer
`--catalog pc`. Earth Search serves the same collection but with multiple
overlapping tile granules — transfer volume can be 20× higher for the
same bbox.
