#!/usr/bin/env bash
# Peak resident set size (RSS) measurement for both extract-patches
# implementations. Uses /usr/bin/time -v "Maximum resident set size"
# (which Linux reports in KB).
#
# Writes results to benchmarks/results/gfm_prep/memory.csv with
# columns: implementation,rep,peak_rss_mb,wall_clock_s.
#
# This is a peak-RSS measurement, not a streaming-trace; it reflects
# the maximum memory the process held at any instant during its
# lifetime. For both implementations the dominant contribution is the
# materialised feature rasters (3 timestamps × 6 bands × G×G float32).
#
# Re-run: `bash benchmarks/measure_memory_gfm_prep.sh`

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SURTGIS_BIN:-$ROOT/target/release/surtgis}"
OUT_DIR="$ROOT/benchmarks/results/gfm_prep"
DATA_DIR="$OUT_DIR/synthetic_dataset"
CSV="$OUT_DIR/memory.csv"

SIZE="${SIZE:-224}"
N_REPS="${N_REPS:-3}"
PROFILE="${PROFILE:-prithvi-v2}"

if [ ! -x "$BIN" ]; then
    echo "Building release binary..."
    cargo build --release --features cloud,projections -p surtgis
fi

if [ ! -d "$DATA_DIR/features" ]; then
    echo "Synthetic dataset not found at $DATA_DIR — run benchmarks/run_gfm_prep_bench.sh first."
    exit 1
fi

if ! /usr/bin/time -v true >/dev/null 2>&1; then
    echo "GNU /usr/bin/time required (bash builtin 'time' doesn't expose RSS)."
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "implementation,rep,peak_rss_mb,wall_clock_s" > "$CSV"

extract_rss_kb () {
    grep "Maximum resident set size" "$1" | awk '{print $NF}'
}
extract_wall_s () {
    # /usr/bin/time -v formats elapsed as h:mm:ss or m:ss.cc
    grep "Elapsed (wall clock) time" "$1" | awk -F': ' '{print $2}' | awk -F: '{
        if (NF == 3) print $1 * 3600 + $2 * 60 + $3
        else if (NF == 2) print $1 * 60 + $2
        else print $1
    }'
}

for REP in $(seq 1 "$N_REPS"); do
    # SurtGIS
    OUT_SUR="$OUT_DIR/mem_run_surtgis_r${REP}"
    LOG_SUR="$OUT_DIR/mem_run_surtgis_r${REP}.time"
    rm -rf "$OUT_SUR"
    /usr/bin/time -v -o "$LOG_SUR" "$BIN" extract-patches \
        --features-dir "$DATA_DIR/features" \
        --points "$DATA_DIR/points.geojson" \
        --points-crs 32719 \
        --label-col cls \
        --profile "$PROFILE" \
        --size "$SIZE" \
        "$OUT_SUR" > /dev/null
    RSS_KB=$(extract_rss_kb "$LOG_SUR")
    WC=$(extract_wall_s "$LOG_SUR")
    RSS_MB=$(python3 -c "print(${RSS_KB} / 1024)")
    echo "surtgis,$REP,$RSS_MB,$WC" >> "$CSV"
    echo "  [rep=$REP] surtgis : peak ${RSS_MB} MB, ${WC} s"

    # Python reference (single-timestamp — same as run_gfm_prep_bench.sh)
    OUT_PY="$OUT_DIR/mem_run_python_r${REP}"
    LOG_PY="$OUT_DIR/mem_run_python_r${REP}.time"
    rm -rf "$OUT_PY"
    /usr/bin/time -v -o "$LOG_PY" python3 "$ROOT/benchmarks/bench_gfm_prep_py.py" \
        --features-dir "$DATA_DIR/features/t0" \
        --points "$DATA_DIR/points.geojson" \
        --label-col cls \
        --profile "$PROFILE" \
        --size "$SIZE" \
        --output "$OUT_PY" > /dev/null 2>&1
    RSS_KB=$(extract_rss_kb "$LOG_PY")
    WC=$(extract_wall_s "$LOG_PY")
    RSS_MB=$(python3 -c "print(${RSS_KB} / 1024)")
    echo "python,$REP,$RSS_MB,$WC" >> "$CSV"
    echo "  [rep=$REP] python  : peak ${RSS_MB} MB, ${WC} s"
done

echo ""
echo "Wrote $CSV"
echo ""
echo "Note: surtgis processes all 3 timestamps; the Python reference processes"
echo "1 timestamp. The comparison is therefore biased AGAINST surtgis on the"
echo "memory axis (3x the raster pixels loaded)."
