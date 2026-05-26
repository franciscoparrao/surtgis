#!/usr/bin/env bash
# Scaling sweeps for SurtGIS extract-patches vs Python reference.
#
# Three axes:
#   N      (point count):  controls patch-loop iteration count
#   PATCH  (H = W):         controls per-patch slice cost (O(H²))
#   T      (timestamps):    SurtGIS-only (Python ref does not support T>1)
#
# For each axis value the dataset is regenerated with the swept
# parameter varying and the other two held at canonical values
# (N=200, PATCH=224, T=3). Each (axis, value) point is benchmarked
# at 3 reps per implementation. Results appended to scaling.csv
# with columns:
#   axis,axis_value,implementation,rep,wall_clock_s
#
# Wall-clock total budget: ~20-30 minutes on a typical workstation.
# To shorten, override the sweep value lists via env vars (see below).
#
# Re-run: `bash benchmarks/run_gfm_prep_sweeps.sh`

set -euo pipefail
# Force C locale so printf %f uses '.' decimal separator regardless of
# the user's LC_NUMERIC setting (es_CL, fr_FR, etc. use ',').
export LC_ALL=C
export LC_NUMERIC=C

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SURTGIS_BIN:-$ROOT/target/release/surtgis}"
OUT_DIR="$ROOT/benchmarks/results/gfm_prep"
DATA_DIR="$OUT_DIR/sweep_dataset"
CSV="$OUT_DIR/scaling.csv"

# Sweep values (override via env vars to subset)
SWEEP_N="${SWEEP_N:-50,200,500,1000}"
SWEEP_PATCH="${SWEEP_PATCH:-64,128,224}"
SWEEP_T="${SWEEP_T:-1,3,6}"
SWEEP_REPS="${SWEEP_REPS:-3}"

# Canonical hold values
CANONICAL_N="200"
CANONICAL_PATCH="224"
CANONICAL_T="3"

if [ ! -x "$BIN" ]; then
    echo "Building release binary..."
    cargo build --release --features cloud,projections -p surtgis
fi

mkdir -p "$OUT_DIR"
echo "axis,axis_value,implementation,rep,wall_clock_s" > "$CSV"

# Helper: regenerate dataset with given N, T (PATCH is set at extract time)
regen_dataset () {
    local n="$1"; local t="$2"
    rm -rf "$DATA_DIR"
    python3 "$ROOT/benchmarks/gfm_prep_make_dataset.py" \
        --out "$DATA_DIR" \
        --grid 2048 \
        --timestamps "$t" \
        --n-points "$n" >/dev/null
}

# Helper: run one (impl, rep) pair, return wall-clock seconds
bench_one () {
    local impl="$1"; local patch="$2"; local rep_idx="$3"
    local out="$OUT_DIR/sweep_run_${impl}_p${patch}_r${rep_idx}"
    rm -rf "$out"
    local t0=$(python3 -c 'import time; print(time.perf_counter())')
    if [ "$impl" = "surtgis" ]; then
        "$BIN" extract-patches \
            --features-dir "$DATA_DIR/features" \
            --points "$DATA_DIR/points.geojson" \
            --points-crs 32719 \
            --label-col cls \
            --profile prithvi-v2 \
            --size "$patch" \
            "$out" >/dev/null
    else
        # Python ref: single timestamp only (t0/)
        python3 "$ROOT/benchmarks/bench_gfm_prep_py.py" \
            --features-dir "$DATA_DIR/features/t0" \
            --points "$DATA_DIR/points.geojson" \
            --label-col cls \
            --profile prithvi-v2 \
            --size "$patch" \
            --output "$out" >/dev/null 2>&1
    fi
    local t1=$(python3 -c 'import time; print(time.perf_counter())')
    python3 -c "print(${t1} - ${t0})"
}

# ── Axis 1: N (point count) ──────────────────────────────────────────
echo ""
echo "=== Axis 1: N (point count) at PATCH=$CANONICAL_PATCH, T=$CANONICAL_T ==="
for N in ${SWEEP_N//,/ }; do
    regen_dataset "$N" "$CANONICAL_T"
    for REP in $(seq 1 "$SWEEP_REPS"); do
        for IMPL in surtgis python; do
            WC=$(bench_one "$IMPL" "$CANONICAL_PATCH" "$REP")
            echo "N,$N,$IMPL,$REP,$WC" >> "$CSV"
            printf "  [N=%5s rep=%d] %-8s : %.3f s\n" "$N" "$REP" "$IMPL" "$WC"
        done
    done
done

# ── Axis 2: PATCH (patch size) ───────────────────────────────────────
echo ""
echo "=== Axis 2: PATCH (patch H=W) at N=$CANONICAL_N, T=$CANONICAL_T ==="
regen_dataset "$CANONICAL_N" "$CANONICAL_T"
for PATCH in ${SWEEP_PATCH//,/ }; do
    for REP in $(seq 1 "$SWEEP_REPS"); do
        for IMPL in surtgis python; do
            WC=$(bench_one "$IMPL" "$PATCH" "$REP")
            echo "PATCH,$PATCH,$IMPL,$REP,$WC" >> "$CSV"
            printf "  [PATCH=%4s rep=%d] %-8s : %.3f s\n" "$PATCH" "$REP" "$IMPL" "$WC"
        done
    done
done

# ── Axis 3: T (timestamps) — SurtGIS only ────────────────────────────
echo ""
echo "=== Axis 3: T (timestamps) at N=$CANONICAL_N, PATCH=$CANONICAL_PATCH (surtgis only) ==="
for T in ${SWEEP_T//,/ }; do
    regen_dataset "$CANONICAL_N" "$T"
    for REP in $(seq 1 "$SWEEP_REPS"); do
        WC=$(bench_one "surtgis" "$CANONICAL_PATCH" "$REP")
        echo "T,$T,surtgis,$REP,$WC" >> "$CSV"
        printf "  [T=%2s rep=%d] surtgis  : %.3f s\n" "$T" "$REP" "$WC"
    done
done

echo ""
echo "Wrote $CSV"
echo "Total rows: $(wc -l < "$CSV")"
