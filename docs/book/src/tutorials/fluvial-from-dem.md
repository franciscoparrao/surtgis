# Fluvial-tectonic morphometry from a DEM

This tutorial walks you from a raw digital elevation model to the five
canonical metrics of tectonic geomorphology: χ (chi), normalised
channel steepness (ksn), knickpoints, per-basin concavity (θ), and
divide-migration asymmetry. It takes about 10 minutes once you have a
DEM in hand and produces inputs that match the convention TopoToolbox
(Schwanghart & Scherler 2014) established.

By the end you will have:

- A reproducible CLI sequence that turns any projected DEM into the
  full v0.10 fluvial product family.
- Three GeoJSON files (ksn segments, knickpoints, divides) ready to
  drape on a hillshade in QGIS or load into geopandas.
- A CSV of per-basin θ with bootstrap-derived 95 % CIs.
- Working intuition for which knob to turn when an output looks off.

**Prerequisites:** SurtGIS ≥ 0.10.1 installed
(`cargo install surtgis` or `pip install surtgis`). A DEM in a
**projected** coordinate system in metres — UTM is the standard
choice. If yours is in EPSG:4326 (latitude/longitude), reproject first:
`surtgis reproject dem_wgs84.tif --to EPSG:32719 dem.tif`.

If you don't have one, the
[First terrain analysis tutorial](first-terrain-analysis.md) walks
through downloading a real 30 m Andes DEM in two steps.

## 1. Set up a working directory

```bash
mkdir -p ~/surtgis-fluvial-tutorial
cd ~/surtgis-fluvial-tutorial
cp /path/to/your/dem.tif dem.tif
```

All subsequent commands expect `dem.tif` to be present in the working
directory and to be in a projected CRS with metre units.

## 2. Build the hydrology stack

The fluvial algorithms operate on three derived rasters, not on the DEM
directly. Producing them is the standard SurtGIS hydrology pipeline:

```bash
surtgis hydrology fill-sinks       dem.tif       filled.tif
surtgis hydrology flow-direction   filled.tif    fdir.tif
surtgis hydrology flow-accumulation fdir.tif      facc.tif
```

Each command takes seconds on a 1000 × 1000 raster. The three outputs
together represent the full hydrologic topology: every cell knows where
its water comes from (`facc`) and where it goes next (`fdir`).

Now extract the stream network. **This is where the new
`--from-facc` flag matters**: without it, `stream-network` would
recompute the whole pipeline internally and the resulting stream cells
would NOT be topologically consistent with the `fdir.tif` you just
produced. With it, the command thresholds your existing `facc.tif`
and the topology is coherent across all four files.

```bash
surtgis hydrology stream-network --from-facc --threshold 1000 \
                                  facc.tif    streams.tif
```

The `--threshold 1000` is in cell counts. With a 30 m DEM this is
~0.9 km² of contributing area — a reasonable threshold for mountainous
terrain that captures bedrock channels but suppresses noise in
zero-order hollows. For a 10 m DEM, scale to `--threshold 9000` to
keep the area threshold constant.

## 3. Compute χ (chi)

χ is the path integral of `(A₀/A(x))^θref` along each river profile,
measured from the network's outlets upstream. In a steady-state
landscape it linearises the elevation profile against a single
slope — that slope is `ksn`.

```bash
surtgis fluvial chi streams.tif fdir.tif facc.tif chi.tif
```

Defaults: `--theta-ref 0.45` (the canonical bedrock-channel reference,
Perron & Royden 2013) and `--a-0-m2 1e6` (1 km² normalisation). The
output is a Float32 GeoTIFF where every stream cell has a χ value in
metres; non-stream cells are NaN.

Quick visual check in QGIS: load `chi.tif`, style with a continuous
ramp — χ should climb monotonically from blue (low, near outlets) to
red (high, near divides). Any spatial discontinuity along a single
channel flags a topology problem upstream (usually flow-direction
ambiguity at a confluence).

## 4. Compute ksn (channel steepness)

```bash
surtgis fluvial ksn streams.tif fdir.tif facc.tif filled.tif ksn.tif \
                    --segments ksn_segments.geojson
```

Two outputs:

- `ksn.tif` — per-cell raster, smoothed over a 500 m window along the
  channel (Wobus et al. 2006 standard).
- `ksn_segments.geojson` — one LineString per stream segment between
  graph terminals (confluence ↔ confluence, confluence ↔ outlet),
  attributes `ksn_mean` and `n_cells`. Coordinates are reprojected to
  WGS84 by default (RFC 7946); pass `--keep-crs` to preserve source
  CRS metres with a legacy `crs` declaration.

High ksn cells correspond to either fast uplift or resistant
lithology. Sub-basins with anomalously high ksn relative to their
neighbours are the first thing to investigate for tectonic signal.

## 5. Detect knickpoints

```bash
surtgis fluvial knickpoints streams.tif fdir.tif facc.tif filled.tif \
                            knickpoints.geojson \
                            --raster knickpoint_raster.tif
```

The output GeoJSON Points carry four properties:

- `elevation_m` — denoised z at the knickpoint cell
- `magnitude_m` — elevation drop across a ±2-cell window
- `chi` — χ value at the cell
- `polarity` — `concave` (slope decreases downstream, often lithology
  contrast) or `convex` (slope increases downstream, often transient
  response to a tectonic pulse)

The categorical raster (`--raster`) writes 0 for non-knickpoint,
1 for concave, 2 for convex — useful when you want to overlay knickpoint
density on a hillshade.

Tuning: if you see false positives in headwaters, raise
`--min-magnitude-m` from 10 (default) to 20-30. If you suspect real
knickpoints are being smoothed away, lower `--tvd-lambda` from 0.5 to
0.3.

## 6. Per-basin concavity θ

You'll need a basins raster first. Identify pour-points (rows/cols of
the outlets of the sub-basins you care about) — usually the cells
with the highest `facc` values along the AOI boundary. If you have
them as coordinates from a GIS, convert to (row, col) using the
`facc.tif` transform.

```bash
surtgis hydrology watershed --pour-points "1429,778;1613,466" \
                            fdir.tif basins.tif

surtgis fluvial concavity streams.tif fdir.tif facc.tif filled.tif basins.tif \
                          concavity.csv \
                          --bootstrap-n 200
```

The CSV has one row per basin with `theta_opt`, `theta_ci_low`,
`theta_ci_high`, `n_cells`, `rmse`. Typical bedrock channels in steady
state cluster around θ = 0.45; values < 0.3 or > 0.6 are interesting
and warrant inspection — they suggest transient response, lithologic
heterogeneity, or a non-uniform uplift signal.

## 7. Divide migration

```bash
surtgis fluvial divide-migration basins.tif filled.tif facc.tif \
                                 divides.geojson \
                                 --chi chi.tif
```

One LineString feature per adjacent-basin pair, with median Δχ
(Willett 2014), median Gilbert Δelev, median Δrelief, and `n_pairs`
(cell-pair count along the divide).

Interpretation: a divide where `median_chi_diff > 0` and basin_a
(the smaller numeric ID) is on the higher-elevation side suggests
basin_a is "losing area" to basin_b. The threshold for significance
is empirical — in well-studied cases like Willett's southern Sierra
Madre, |Δχ| of a few hundred metres is diagnostic; for your AOI
calibrate against the obvious cases first.

## 8. Verify in QGIS

Load the rasters and vectors in QGIS:

| Layer | Style |
|---|---|
| `filled.tif` | hillshade (Tools → Raster → Analysis → Hillshade) |
| `ksn.tif` | graduated, viridis 10–500 range, transparent for NaN |
| `knickpoints.geojson` | categorized on `polarity` (concave = blue, convex = red), graduated size by `magnitude_m` |
| `ksn_segments.geojson` | graduated line on `ksn_mean`, viridis |
| `divides.geojson` | graduated line on `median_chi_diff`, diverging colormap centred at 0 |

If you used the default WGS84 output, everything aligns directly on
a basemap layer (OpenStreetMap, satellite). If you used `--keep-crs`,
make sure the project CRS matches your source raster CRS.

## Where to next

- **"What do these numbers actually mean?"** —
  [Tectonic geomorphology with the fluvial module](../explanation/fluvial-tectonics.md)
  walks through the interpretation framework.
- **"My output looks weird."** — the pitfalls section of the explanation
  chapter covers the eight failure modes the spec calls out, including
  the cell-size-vs-resolution sensitivity that bites on coarse DEMs.
- **"I want to publish this."** — every output above carries provenance
  (parameters, source CRS, etc.) in its filename or properties; for a
  paper-grade pipeline log, capture the full command sequence and the
  SurtGIS version (`surtgis --version`) alongside the figures.
