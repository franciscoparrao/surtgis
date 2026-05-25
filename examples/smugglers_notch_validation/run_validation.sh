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

echo "[1/8] Setup workdir at $WORK"
mkdir -p "$WORK"
cd "$WORK"

echo "[2/8] Fetch Copernicus GLO-30 DEM via Earth Search"
"$BIN" stac fetch-mosaic \
    --catalog es \
    --collection cop-dem-glo-30 \
    --bbox=-72.90,44.45,-72.65,44.65 \
    dem_wgs84.tif

echo "[3/8] Reproject WGS84 → UTM 18N (EPSG:32618)"
"$BIN" reproject dem_wgs84.tif --to EPSG:32618 dem_utm.tif

echo "[4/8] Clip to NaN-free interior (reprojection leaves corners as NaN)"
# The reprojected DEM has ~5% NaN cells at the rotated corners. fill-sinks
# propagates `inf` through any NaN region (a known surtgis issue tracked
# separately). Clip to the inner box to side-step it.
"$BIN" clip dem_utm.tif --bbox 668769,4926295,684684,4944484 dem.tif

echo "[5/8] Hydrology pipeline"
"$BIN" hydrology fill-sinks dem.tif filled.tif
"$BIN" hydrology flow-direction filled.tif fdir.tif
"$BIN" hydrology flow-accumulation fdir.tif facc.tif
"$BIN" hydrology stream-network --from-facc --threshold 500 facc.tif streams.tif

echo "[6/8] Watershed delineation (top-8 max-facc cells as pour points)"
PP=$(python3 -c "
import rasterio, numpy as np
fa = rasterio.open('facc.tif').read(1)
flat = fa.flatten()
top_idx = np.argpartition(flat, -8)[-8:]
print(';'.join(f'{int(i//fa.shape[1])},{int(i%fa.shape[1])}' for i in top_idx))
")
"$BIN" hydrology watershed --pour-points "$PP" fdir.tif basins.tif

echo "[7/8] Fluvial morphometry"
"$BIN" fluvial chi streams.tif fdir.tif facc.tif chi.tif
"$BIN" fluvial ksn --segments ksn_segments.geojson \
    streams.tif fdir.tif facc.tif filled.tif ksn.tif
"$BIN" fluvial concavity --bootstrap-n 200 --min-basin-cells 100 \
    streams.tif fdir.tif facc.tif filled.tif basins.tif "$ROOT/concavity.csv"

echo "[8/8] Validation plot + metrics"
python3 "$ROOT/plot_validation.py" "$WORK" "$ROOT"

echo ""
echo "Done. Outputs in $ROOT:"
echo "  validation_plot.{pdf,png}"
echo "  validation_metrics.csv"
echo "  concavity.csv"
