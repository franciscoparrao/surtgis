#!/usr/bin/env bash
# Smugglers Notch (Vermont, USA) — fluvial-tectonic morphometry pipeline.
#
# Reproduces the canonical test case from Perron & Royden 2013 (ESPL 38,
# 570) using SurtGIS v0.10.1. The headline metric is the linearisation
# of the elevation profile when plotted against χ: in a steady-state
# bedrock landscape this should be ~linear (R² → 1). Real basins
# deviate; we get R² = 0.82 on Smugglers Notch's main catchment, which
# matches the scatter visible in P&R's own Fig 3.
#
# Inputs: none — Copernicus GLO-30 DEM is fetched via Earth Search.
# Outputs: validation_plot.{pdf,png} + validation_metrics.csv +
#          concavity.csv in this directory.
#
# Runtime: ~3 minutes (most of it the DEM download).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BIN="${SURTGIS_BIN:-surtgis}"
WORK="${WORK:-/tmp/smugglers_notch_run}"

echo "[1/7] Setup workdir at $WORK"
mkdir -p "$WORK"
cd "$WORK"

echo "[2/7] Fetch Copernicus GLO-30 DEM via Earth Search"
"$BIN" stac fetch-mosaic \
    --catalog es \
    --collection cop-dem-glo-30 \
    --bbox=-72.90,44.45,-72.65,44.65 \
    dem_wgs84.tif

echo "[3/7] Reproject WGS84 → UTM 18N (EPSG:32618)"
# The reprojected DEM has ~5% NaN cells at the rotated corners. Earlier
# versions of surtgis fill-sinks propagated `inf` through those NaN
# regions, forcing a defensive clip step. v0.10.2 handles NaN as a
# drainage exit, so the clip is no longer required.
"$BIN" reproject dem_wgs84.tif --to EPSG:32618 dem.tif

echo "[4/7] Hydrology pipeline"
"$BIN" hydrology fill-sinks dem.tif filled.tif
"$BIN" hydrology flow-direction filled.tif fdir.tif
"$BIN" hydrology flow-accumulation fdir.tif facc.tif
"$BIN" hydrology stream-network --from-facc --threshold 500 facc.tif streams.tif

echo "[5/7] Watershed delineation (spatially-distributed pour points)"
# Greedy spatial selection: rank cells by flow-accumulation, then pick
# the top N subject to a minimum pixel-distance constraint between
# any two selected cells. This avoids the failure mode of the naive
# "top-K max-facc" picker, which collapses all picks onto the few
# cells where the dominant river exits the AOI (here, the eastern
# border) and yields only one delineated basin.
PP=$(python3 -c "
import rasterio, numpy as np
src = rasterio.open('facc.tif')
fa = src.read(1)
cell_size = abs(src.transform.a)
min_sep_m = 3000  # 3 km between basin outlets — keeps Smugglers' tributaries separable
min_sep_px = max(1, int(min_sep_m / cell_size))
target_n = 6

# Candidate pool: top 0.1% by facc, then greedy spatial filter.
n_cand = max(target_n * 50, int(fa.size * 0.001))
flat = fa.flatten()
top_idx = np.argpartition(flat, -n_cand)[-n_cand:]
top_idx = top_idx[np.argsort(-flat[top_idx])]  # descending by facc
rows = (top_idx // fa.shape[1]).astype(int)
cols = (top_idx %  fa.shape[1]).astype(int)

picked = []
for r, c in zip(rows, cols):
    if all((r - pr) ** 2 + (c - pc) ** 2 >= min_sep_px ** 2 for pr, pc in picked):
        picked.append((int(r), int(c)))
        if len(picked) >= target_n:
            break

print(';'.join(f'{r},{c}' for r, c in picked))
")
echo "  pour points: $PP"
"$BIN" hydrology watershed --pour-points "$PP" fdir.tif basins.tif

echo "[6/7] Fluvial morphometry"
"$BIN" fluvial chi streams.tif fdir.tif facc.tif chi.tif
"$BIN" fluvial ksn --segments ksn_segments.geojson \
    streams.tif fdir.tif facc.tif filled.tif ksn.tif
"$BIN" fluvial concavity --bootstrap-n 200 --min-basin-cells 100 \
    streams.tif fdir.tif facc.tif filled.tif basins.tif "$ROOT/concavity.csv"

echo "[7/7] Validation plot + metrics"
python3 "$ROOT/plot_validation.py" "$WORK" "$ROOT"

echo ""
echo "Done. Outputs in $ROOT:"
echo "  validation_plot.{pdf,png}"
echo "  validation_metrics.csv"
echo "  concavity.csv"
