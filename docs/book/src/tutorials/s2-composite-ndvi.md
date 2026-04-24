# Sentinel-2 composite and NDVI from STAC

This tutorial builds a cloud-free Sentinel-2 composite from Microsoft's
Planetary Computer and derives the NDVI vegetation index from it — the
classic "hello world" of satellite remote sensing. It takes about 15
minutes of wall clock (most of it network transfer).

By the end you will have:

- Red and NIR band GeoTIFFs, mosaicked across multiple cloud-free scenes
  from one month, already in UTM at 10-metre resolution.
- An NDVI raster ready to visualise or feed into a model.
- An understanding of the STAC composite pipeline: search → mask → median
  → reproject.

**Prerequisites:** SurtGIS installed. Network access for STAC reads
(~100 MB transferred for this tutorial).

## 1. Working directory

```bash
mkdir -p ~/surtgis-s2-tutorial
cd ~/surtgis-s2-tutorial
```

## 2. Pick a study area

We'll use a 5 km × 5 km patch in central Chile (near Talca), Southern
Hemisphere summer, so vegetation is green and cloud cover is manageable:

- Bbox: `-71.0, -35.1, -70.95, -35.05` (west, south, east, north in degrees)
- Date range: `2024-01-01` to `2024-01-31`

Small bboxes are kind to your network and to Planetary Computer's rate
limits. Once the pipeline works at 5 km, scaling to a watershed is just
changing one flag.

## 3. Fetch a median composite of red + NIR

```bash
surtgis stac composite \
    --catalog pc \
    --collection sentinel-2-l2a \
    --asset "red,nir" \
    --bbox=-71.0,-35.1,-70.95,-35.05 \
    --datetime 2024-01-01/2024-01-31 \
    --max-scenes 3 \
    composite.tif
```

What's happening, step by step:

1. **STAC search.** The client queries Planetary Computer's STAC API for
   `sentinel-2-l2a` items intersecting the bbox within the datetime range.
   Typically you get 5–10 candidate scenes for one month on a small bbox.
2. **Scene ranking.** Up to `--max-scenes` (3 here) are picked by coverage
   of the bbox and reasonable cloud cover metadata.
3. **Per-scene fetch.** For each selected scene, the COG reader opens the
   red and NIR assets via HTTP range requests, downloading only the tiles
   that intersect the bbox.
4. **Cloud masking.** Sentinel-2 L2A ships with an SCL (Scene
   Classification Layer) raster that labels every pixel as cloud, cloud
   shadow, water, vegetation, etc. SurtGIS applies the mask to remove
   cloudy pixels.
5. **Median composite.** Across the 3 cloud-masked scenes, each pixel takes
   the median value per band. The result is one pixel-stack with one value
   per band.
6. **Output.** Written as UTM-projected GeoTIFFs, one per band:
   `composite_red.tif` and `composite_nir.tif`. The `_red` and `_nir`
   suffixes come from the asset names.

On a warm connection this should take about 65 seconds. Verify:

```bash
ls *.tif
surtgis info composite_red.tif
```

```text
composite_nir.tif  composite_red.tif
...
Dimensions: 467 x 564 (263388 cells)
Cell size: 10
Bounds: (317595.06, 6114033.69) - (322265.06, 6119673.69)
CRS: EPSG:32719
```

10-metre cells, EPSG:32719 (UTM 19S), about 4.7 km × 5.6 km. Note that the
output extent is slightly larger than the bbox you requested — the
composite rounds to the Sentinel-2 native 10-metre grid, so you get a
couple of extra pixels of padding.

## 4. Compute NDVI

NDVI = (NIR − Red) / (NIR + Red). It's a vegetation index: roughly, higher
values mean denser, greener vegetation. Bare soil is around 0.1, sparse
grass 0.2–0.3, active crops 0.4–0.7, dense forest 0.7–0.9.

```bash
surtgis imagery ndvi \
    --red composite_red.tif \
    --nir composite_nir.tif \
    ndvi.tif

surtgis info ndvi.tif
```

You should see:

```text
Statistics:
  Min: ~0.03
  Max: ~0.69
  Mean: ~0.40
```

A mean of 0.40 is consistent with a mix of crop fields and some bare land
in mid-January. Open the TIFF in QGIS with the "pseudocolor" style and
apply a red → yellow → green ramp; you'll immediately see agricultural
patterns.

## 5. Interpret the output

A few sanity checks to confirm your pipeline is working:

**Histogram shape.** NDVI over a real landscape has a bimodal or trimodal
distribution: bare soil / pavement (~0.1), mixed vegetation (~0.3–0.5),
dense vegetation (~0.7+). A single peak at 0 suggests the red/NIR bands
got swapped. A single peak at 0.9 suggests the composite is hallucinating
vegetation, possibly because the SCL mask is removing all non-vegetation
pixels (unlikely but worth checking).

**No NaN border.** Unlike terrain derivatives (which have a 1-pixel NaN
border from the 3×3 window), NDVI is a per-pixel formula and should have
valid values everywhere the red and NIR bands both have values. If you see
a large NaN region, that's likely cloud cover that the SCL mask filtered
out across _every_ selected scene — increase `--max-scenes` to get more
dates.

**Distribution of valid pixels.** `surtgis info ndvi.tif` reports "Valid
cells: N/M". For a cloud-free January in this region you should see 100%.
In cloudier regions or seasons, expect 85–95%.

## 6. Scale to more bands and more dates

The real workflow for a machine-learning pipeline wants more than two
bands. Here's a 10-band composite over three months:

```bash
surtgis stac composite \
    --catalog pc \
    --collection sentinel-2-l2a \
    --asset "blue,green,red,rededge1,rededge2,rededge3,nir,nir08,swir16,swir22" \
    --bbox=-71.0,-35.1,-70.95,-35.05 \
    --datetime 2024-01-01/2024-03-31 \
    --max-scenes 8 \
    --cache \
    composite_all.tif
```

What's new:

- 10 bands at once. Writes `composite_all_blue.tif`, `composite_all_green.tif`,
  etc.
- 3 months of data, 8 scenes median-composited.
- `--cache` persists downloaded COG tiles at `~/.cache/surtgis/cog/` so
  re-runs reuse them instead of re-fetching.

Expected wall time: ~5 minutes. RAM peak: see
[the RAM budget how-to](../how-to/ram-budget.md) — SurtGIS will print a
budget line at startup estimating peak usage.

## 7. Use NDVI as input to a model

Now you have a feature raster. Two common next steps:

**Combine with terrain.** Run the [first tutorial](first-terrain-analysis.md)
on a DEM covering the same bbox, then the derived terrain factors (slope,
aspect, TPI) plus this NDVI form a multi-feature stack that most
landslide-susceptibility or vegetation-classification models want.

**Extract training samples at labelled points.** If you have a vector file
of labelled points (e.g. `wells.gpkg` with a `vegetation_class` column):

```bash
surtgis extract \
    --features-dir ./composite_all_bands/ \
    --points wells.gpkg \
    --target vegetation_class \
    samples.csv
```

The result is a CSV with one row per point, columns = bands + target.
Feed into XGBoost, scikit-learn, or any tabular model.

For CNN-based spatial classification, use `extract-patches` instead to
generate image chips (see the [extract-patches how-to](../how-to/extract-patches.md)).

## Common gotchas

**"The composite has huge black regions."** Your bbox straddles a
Sentinel-2 MGRS tile boundary, and the scenes selected don't all cover
both sides. Increase `--max-scenes` to get scenes from both orbits.

**"The download is slow or times out."** Planetary Computer occasionally
rate-limits (HTTP 429). SurtGIS 0.6.25+ has exponential backoff with
jitter built into the retry logic; transient failures should recover
automatically. Persistent rate limits might warrant switching to Earth
Search (`--catalog es`) or spreading the work across time.

**"RAM climbs during long runs."** See
[Debug a stac composite using too much RAM](../how-to/debug-stac-ram.md).
Short version: the `[ram]` log lines SurtGIS prints at strip/band
transitions pinpoint where growth starts. Raise `--band-chunk-size` if
you have headroom, or lower `SURTGIS_RAM_BUDGET_GB` if you don't.

**"I want the full scene, not just my bbox."** For a full MGRS tile (~110
km on a side at 10 m resolution), expect 10–100× the runtime and transfer
volume. Use `--strip-rows 256` and enable `--cache`. For tile-size extents
the streaming pipeline really earns its keep.
