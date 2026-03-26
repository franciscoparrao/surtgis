#!/bin/bash
# Measure peak RSS (maxresident) for SurtGIS, GDAL, GRASS, WBT
# on 10K DEM for slope, fill, flow_acc
#
# Usage: bash benchmarks/bench_memory.sh
# Output: benchmarks/results/experiment_memory.csv

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEM_10K="$ROOT/benchmarks/results/dems/fbm_10000_raw.tif"
DEM_5K="$ROOT/benchmarks/results/dems/fbm_5000_raw.tif"
OUTDIR="$ROOT/benchmarks/results"
TMPDIR=$(mktemp -d)
CSV="$OUTDIR/experiment_memory.csv"

echo "algorithm,size,tool,peak_rss_mb" > "$CSV"

measure() {
    local algo="$1" size="$2" tool="$3" cmd="$4"
    local outfile="$TMPDIR/time_${tool}_${algo}_${size}.txt"

    /usr/bin/time -v bash -c "$cmd" 2>"$outfile" >/dev/null || true
    local rss_kb=$(grep "Maximum resident" "$outfile" | awk '{print $NF}')
    local rss_mb=$(echo "scale=1; $rss_kb / 1024" | bc)
    echo "$algo,$size,$tool,$rss_mb" >> "$CSV"
    echo "  $tool $algo ${size}: ${rss_mb} MB"
}

echo "═══════════════════════════════════════"
echo "  SurtGIS Memory Benchmark (peak RSS)"
echo "═══════════════════════════════════════"

# ── SurtGIS (10K) ──
echo ""
echo "── SurtGIS (native Rust) ──"
SURTGIS="$ROOT/target/release/examples/bench_comparison"

# Build if not exists
if [ ! -f "$SURTGIS" ]; then
    echo "Building bench_comparison..."
    cargo build --release --example bench_comparison -p surtgis-algorithms 2>/dev/null
fi

measure "slope" "10000" "surtgis" \
    "$SURTGIS --size 10000 --algo slope --reps 1 --warmup 0"

measure "fill" "10000" "surtgis" \
    "$SURTGIS --size 10000 --algo fill --reps 1 --warmup 0"

measure "flow_acc" "10000" "surtgis" \
    "$SURTGIS --size 10000 --algo flow_acc --reps 1 --warmup 0"

# ── GDAL (10K) ──
echo ""
echo "── GDAL ──"
measure "slope" "10000" "gdal" \
    "gdaldem slope $DEM_10K $TMPDIR/gdal_slope.tif -of GTiff -b 1"

# ── GRASS (10K) ──
echo ""
echo "── GRASS ──"
measure "slope" "10000" "grass" \
    "grass --tmp-location EPSG:32719 --exec bash -c 'r.in.gdal input=$DEM_10K output=dem --quiet 2>/dev/null && r.slope.aspect elevation=dem slope=slope --quiet 2>/dev/null && r.out.gdal input=slope output=$TMPDIR/grass_slope.tif --quiet 2>/dev/null'"

# GRASS fill (5K, since 10K times out)
measure "fill" "5000" "grass" \
    "grass --tmp-location EPSG:32719 --exec bash -c 'r.in.gdal input=$DEM_5K output=dem --quiet 2>/dev/null && r.fill.dir input=dem output=filled direction=dir --quiet 2>/dev/null && r.out.gdal input=filled output=$TMPDIR/grass_fill.tif --quiet 2>/dev/null'"

measure "flow_acc" "10000" "grass" \
    "grass --tmp-location EPSG:32719 --exec bash -c 'r.in.gdal input=$DEM_10K output=dem --quiet 2>/dev/null && r.watershed elevation=dem accumulation=acc --quiet 2>/dev/null && r.out.gdal input=acc output=$TMPDIR/grass_flowacc.tif --quiet 2>/dev/null'"

# ── WBT (10K) ──
echo ""
echo "── WhiteboxTools ──"
WBT=$(python3 -c "import whitebox; print(whitebox.download.get_wbt_path())" 2>/dev/null || echo "")
if [ -n "$WBT" ]; then
    measure "slope" "10000" "wbt" \
        "$WBT -r=Slope --dem=$DEM_10K -o=$TMPDIR/wbt_slope.tif 2>/dev/null"

    measure "fill" "10000" "wbt" \
        "$WBT -r=FillDepressions --dem=$DEM_10K -o=$TMPDIR/wbt_fill.tif 2>/dev/null"

    measure "flow_acc" "10000" "wbt" \
        "$WBT -r=D8FlowAccumulation --dem=$DEM_10K -o=$TMPDIR/wbt_flowacc.tif --pntr 2>/dev/null"
else
    echo "  WBT not found, skipping"
fi

echo ""
echo "Results: $CSV"
cat "$CSV"

# Cleanup
rm -rf "$TMPDIR"
