# STAC integration philosophy

The `surtgis stac` command group is designed around one principle:
**fetch the minimum bytes you need, compose them into the output the user
actually asked for, in one command.**

This rules out the two obvious alternatives:

1. **Download the whole file, then process.** A full Sentinel-2 MGRS
   tile is ~700 MB. Over a year that's hundreds of GB of transfer for a
   workflow that might only care about a 5 km × 5 km bbox. The COG format
   exists specifically so we don't have to do this.

2. **Use GDAL's VSI layer.** GDAL can read COGs over HTTP
   (`/vsicurl/https://...`). This works, but puts GDAL between SurtGIS
   and the byte layout of the file, which means any bug or performance
   quirk in GDAL's VSI cache shows up as mysterious SurtGIS behaviour.
   Keeping the COG reader in `surtgis-cloud` means when something is wrong
   we can debug it, not just report it to GDAL upstream.

## What the pipeline does

`surtgis stac composite` walks this chain:

1. **Search.** Hit the STAC API for items matching bbox + datetime +
   collection. Page through results, honouring per-catalog page-size limits
   (PC=1000, ES=250).
2. **Rank and select.** From the candidates, pick `--max-scenes` by
   coverage-aware ranking.
3. **Per scene: resolve asset URLs.** Each band and the cloud mask become
   a signed or unsigned HTTPS URL (or `s3://` for Earth Search — which
   SurtGIS rewrites to `https://bucket.s3.amazonaws.com/key` internally).
4. **Per strip of output: fetch the tiles that intersect.** COG tiles are
   typically 512² or 1024² pixels. For a strip of output, the reader
   computes which internal tiles to pull, issues parallel HTTP range
   requests, decompresses.
5. **Mosaic within scene.** If the bbox crosses UTM zones, the scene's
   tiles may be in different CRSes; `surtgis-cloud::reproject` aligns them
   before `surtgis_core::mosaic` stitches.
6. **Mask clouds.** Apply the scene's SCL (Sentinel-2) or QA_PIXEL
   (Landsat) to nullify cloudy pixels.
7. **Median across scenes.** For each output pixel, the median of the
   non-null values across all selected scenes.
8. **Temporal + spatial fill.** If a pixel is still null after the
   median (every scene was cloudy there), try temporal fallback from the
   least-cloudy scene, then 3×3 spatial mean.

## Why median and not mean

Median is robust to cloud-mask leakage. A single bright cloud pixel that
slipped past SCL distorts a mean significantly but affects a median of
3+ values almost not at all. Cost: slightly noisier output on very sparse
scene counts (N=3 gives a noisier median than N=20).

## Why per-strip processing

The composite for a 100 km × 100 km extent at 10 m resolution is
`10,000 × 10,000 × n_bands × 8 bytes` = 8 GB for 10 bands. Holding that
plus all the per-scene intermediate arrays in RAM blows any host. Strip
processing bounds the peak regardless of extent size.

The trade-off was learned expensively over v0.6.19 through v0.6.26 — see
the [post-mortem](https://github.com/franciscoparrao/surtgis/blob/main/docs/postmortems/2026-04-stac-composite-ram.md)
for the full story.

## What's intentionally not supported

- **Arbitrary temporal aggregations.** Median only. Mean, max, percentile
  are easy to add if someone asks; nobody has yet.
- **Sub-pixel reprojection.** Bilinear resampling is the only option. If
  you want cubic or Lanczos for a composite, reach for GDAL's `gdalwarp`
  after the fact.
- **Streaming output.** The output band rasters are written whole at the
  end, not incrementally. For extents where the output itself exceeds RAM
  (very rare in practice, requires `>> 100` GB bbox coverage), you need
  a different tool.
