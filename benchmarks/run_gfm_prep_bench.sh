#!/usr/bin/env bash
# Benchmark: SurtGIS extract-patches vs reference Python on the local
# extraction hot loop. Generates a synthetic Prithvi-shaped dataset
# (6 bands × T timestamps × G×G pixels, with N point labels) and runs
# both implementations N_REPS times, writing wall-clock results to
# `benchmarks/results/gfm_prep/timings.csv`.
#
# What this bench is NOT: a head-to-head against InstaGeo or
# raster-vision. Those add STAC + cloud-fetch + tile-decode overhead
# that we don't reimplement here. The point is the local hot loop
# that both InstaGeo and surtgis run after data is on disk.
#
# Configure via env vars (defaults in parens):
#   BENCH_SIZES (224)     comma-separated patch sizes
#   BENCH_TIMESTAMPS (3)  number of HLS-equivalent timestamps
#   BENCH_GRID (2048)     synthetic raster side length
#   BENCH_N_POINTS (200)  number of training points
#   BENCH_REPS (5)        repetitions per configuration
#   BENCH_PROFILE (prithvi-v2)
#
# Re-run: `bash benchmarks/run_gfm_prep_bench.sh`

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SURTGIS_BIN:-$ROOT/target/release/surtgis}"
OUT_DIR="$ROOT/benchmarks/results/gfm_prep"
DATA_DIR="$OUT_DIR/synthetic_dataset"
CSV="$OUT_DIR/timings.csv"

BENCH_SIZES="${BENCH_SIZES:-224}"
BENCH_TIMESTAMPS="${BENCH_TIMESTAMPS:-3}"
BENCH_GRID="${BENCH_GRID:-2048}"
BENCH_N_POINTS="${BENCH_N_POINTS:-200}"
BENCH_REPS="${BENCH_REPS:-5}"
BENCH_PROFILE="${BENCH_PROFILE:-prithvi-v2}"

if [ ! -x "$BIN" ]; then
    echo "Building release binary..."
    cargo build --release --features cloud,projections -p surtgis
fi

mkdir -p "$OUT_DIR"

# ── 1. Generate synthetic dataset matching Prithvi conventions ────────
echo "[bench] Generating synthetic dataset at $DATA_DIR"
python3 "$ROOT/benchmarks/gfm_prep_make_dataset.py" \
    --out "$DATA_DIR" \
    --grid "$BENCH_GRID" \
    --timestamps "$BENCH_TIMESTAMPS" \
    --n-points "$BENCH_N_POINTS"

# ── 2. Run both implementations BENCH_REPS times per --size ──────────
echo "implementation,size,n_timestamps,grid,n_points,rep,wall_clock_s" > "$CSV"

for SIZE in ${BENCH_SIZES//,/ }; do
    for REP in $(seq 1 "$BENCH_REPS"); do
        # SurtGIS
        OUT_SUR="$OUT_DIR/run_surtgis_${SIZE}_r${REP}"
        rm -rf "$OUT_SUR"
        T0=$(python3 -c 'import time; print(time.perf_counter())')
        "$BIN" extract-patches \
            --features-dir "$DATA_DIR/features" \
            --points "$DATA_DIR/points.geojson" \
            --label-col cls \
            --profile "$BENCH_PROFILE" \
            --size "$SIZE" \
            "$OUT_SUR" > /dev/null
        T1=$(python3 -c 'import time; print(time.perf_counter())')
        WC_SUR=$(python3 -c "print(${T1} - ${T0})")
        echo "surtgis,$SIZE,$BENCH_TIMESTAMPS,$BENCH_GRID,$BENCH_N_POINTS,$REP,$WC_SUR" >> "$CSV"
        echo "  [size=$SIZE rep=$REP] surtgis   : ${WC_SUR}s"

        # Python reference (single-timestamp only — collapse subdirs)
        OUT_PY="$OUT_DIR/run_python_${SIZE}_r${REP}"
        rm -rf "$OUT_PY"
        T0=$(python3 -c 'import time; print(time.perf_counter())')
        python3 "$ROOT/benchmarks/bench_gfm_prep_py.py" \
            --features-dir "$DATA_DIR/features/t0" \
            --points "$DATA_DIR/points.geojson" \
            --label-col cls \
            --profile "$BENCH_PROFILE" \
            --size "$SIZE" \
            --output "$OUT_PY" > /dev/null 2>&1
        T1=$(python3 -c 'import time; print(time.perf_counter())')
        WC_PY=$(python3 -c "print(${T1} - ${T0})")
        echo "python,$SIZE,$BENCH_TIMESTAMPS,$BENCH_GRID,$BENCH_N_POINTS,$REP,$WC_PY" >> "$CSV"
        echo "  [size=$SIZE rep=$REP] python    : ${WC_PY}s"
    done
done

echo ""
echo "Wrote $CSV"
echo "Render figure: Rscript $ROOT/benchmarks/plot_gfm_prep.R"
