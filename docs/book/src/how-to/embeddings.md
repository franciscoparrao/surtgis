# Work with foundation-model embeddings

Products such as AlphaEarth Foundations (Google DeepMind, 64 dimensions,
10 m, one image per year) or TESSERA (Cambridge, 128 dimensions) publish
one vector per pixel. SurtGIS treats such a stack as a field of vectors:
every band is one dimension, a cell is valid when every band is, and the
operations that make the field useful run in the CLI, in the library and
as dynamic tiles in `surtgis serve`, with no GPU and no notebook.

## AlphaEarth tiles as published

The annual embeddings live in a public Google Cloud Storage bucket, one
COG per 8192 × 8192 tile, and an index lists every tile with its bounds:

```
https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet
https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/{year}/{utm_zone}/{id}-{row}-{col}.tiff
```

Three things about these files matter, and SurtGIS handles all three:

- **Layout.** 64 bands of `Int8`, nodata `-128`, `ZSTD` compression,
  band-interleaved (`INTERLEAVE=BAND`), and stored bottom-up (positive
  pixel height). The COG reader decodes ZSTD in pure Rust, reads planar
  tiles per band, and presents bottom-up files north-up; locally, the CLI
  and the server fall back to that reader when the native GeoTIFF reader
  declines the layout.
- **Code.** Values are a non-linear int8 code: `f = sign(v) · (v/127.5)²`
  restores unit-norm vectors, so that the dot product equals the cosine.
  Pass `--dequantize alphaearth` (CLI) or `params=dequantize:alphaearth`
  (server). Cosine similarity is scale-free, but this code is not linear,
  so de-quantise before comparing.
- **Size.** A tile is 4 GB of int8 and would be 34 GB as f64: read a
  window with `--bbox` (UTM metres of the tile's zone).

Find the tile for a place, then read a 2 km window around it:

```bash
curl -O https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet
surtgis embeddings locate --index aef_index.parquet --lon -70.65 --lat -33.45 --year 2024
#   2024  zone 19S  utm [336160 6231680 418080 6313600]
#   https://storage.googleapis.com/…/2024/19S/xzm4a9wt2bne1ov7f-0000000000-0000000000.tiff
curl -O https://storage.googleapis.com/…/2024/19S/xzm4a9wt2bne1ov7f-0000000000-0000000000.tiff
surtgis embeddings info --dequantize alphaearth --bbox 345000,6296000,347000,6298000 xzm4….tiff
```

`info` reports the L2 norm statistics: a unit-normalised product reads
"norm ≈ 1.000 everywhere".

## Similarity ("more like this")

```bash
surtgis embeddings similarity --dequantize alphaearth --bbox 345000,6296000,347000,6298000 \
  --ref-lonlat=-70.65,-33.45 xzm4….tiff sim.tif
```

The reference can be a location (`--ref-lonlat`, EPSG:4326), a cell
(`--ref-cell ROW,COL`) or an explicit vector (`--ref-vec v1,v2,…`); the
metric is `cosine` (default, in [-1, 1]), `dot` or `euclidean`. A nodata
reference cell falls back to the mean of its 3 × 3 neighbours.

## PCA false colour

```bash
surtgis embeddings pca --dequantize alphaearth --bbox … --components 3 --save-model pca.json xzm4….tiff pca.tif
surtgis embeddings pca --model pca.json other_tile.tiff pca2.tif   # same colours
```

The model (means, axes, variance, the code it was fitted on) is JSON:
fit once, project anywhere, so colours mean the same thing on every tile
and every year. Band `PC1..PC3` of the output are the scores; view them as
RGB.

## Embeddings as features

`surtgis extract` and `surtgis extract-patches` expand a multi-band
feature raster into one feature per band (`name:b1`, `name:b2`, …), so an
embedding tile dropped into the features directory becomes 64 columns of
the training table next to slope, TWI or any other feature.

## Dynamic tiles

With the server, the same operations render on the fly from a local tile
or a COG URL in the allowlist:

```
/tiles/{z}/{x}/{y}.png?url=xzm4….tiff&alg=similarity&ref=-70.65,-33.45&params=dequantize:alphaearth
/tiles/{z}/{x}/{y}.png?url=xzm4….tiff&alg=similarity&vec=0.1,-0.2,…&params=metric:euclidean
/tiles/{z}/{x}/{y}.png?url=xzm4….tiff&alg=pca&params=dequantize:alphaearth,samples:4096
```

`ref` is resolved once per source and location (the cell's vector,
cached); `pca` is fitted once per source on a coarse sample of the whole
extent, so every tile shares the axes and the 2–98 % stretch. `/statistics`
works for `similarity`; `pca` renders RGB only.

## Other products

- **TESSERA** ships `.npy` int8 arrays with a per-pixel `float32` scale and
  a land-mask GeoTIFF that carries the georeference; its vectors are not
  unit-normalised (use cosine). Converting a grid to a GeoTIFF stack makes
  it usable with every command above.
- **Major TOM `Core-AlphaEarth-Embeddings`** re-grids AlphaEarth to
  1068 × 1068 `Float64` GeoTIFFs per cell, already de-quantised (no
  `--dequantize` needed), pixel-interleaved DEFLATE: the native reader
  handles them directly.
