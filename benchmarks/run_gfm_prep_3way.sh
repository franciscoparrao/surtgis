#!/usr/bin/env bash
# 3-way comparison of GFM training-data extraction:
#
#   surtgis             — Rust extract-patches (this work)
#   python              — naive Python (rasterio + numpy, materialise + slice)
#   instageo-style      — Python (xarray + rioxarray + per-chip GeoTIFF write)
#                         mirrors the InstaGeo chip_creator internal loop on
#                         local data, without the STAC orchestration overhead
#                         that InstaGeo additionally incurs upstream
#
# All three operate on the same canonical synthetic dataset
# (N=200 points, 6 bands, 224×224 patches; SurtGIS at T=3, the two
# Python implementations at T=1 since neither supports multi-timestamp
# inputs without significant refactoring).
#
# Re-run: `bash benchmarks/run_gfm_prep_3way.sh`

set -euo pipefail
export LC_ALL=C
export LC_NUMERIC=C

ROOT="$(cd "$(dirname "$0")" && cd .. && pwd)"
BIN="${SURTGIS_BIN:-$ROOT/target/release/surtgis}"
VENV_PY="${VENV_PY:-/home/franciscoparrao/proyectos/paper-gfm-prep/venv/bin/python}"
OUT_DIR="$ROOT/benchmarks/results/gfm_prep"
DATA_DIR="$OUT_DIR/synthetic_dataset"
CSV="$OUT_DIR/three_way.csv"

N_REPS="${N_REPS:-3}"
SIZE="${SIZE:-224}"

if [ ! -x "$BIN" ]; then
    echo "Building release binary..."
    cargo build --release --features cloud,projections -p surtgis
fi

if [ ! -d "$DATA_DIR/features" ]; then
    echo "Synthetic dataset not found at $DATA_DIR — generating..."
    python3 "$ROOT/benchmarks/gfm_prep_make_dataset.py" \
        --out "$DATA_DIR" --grid 2048 --timestamps 3 --n-points 200
fi

mkdir -p "$OUT_DIR"
echo "implementation,rep,wall_clock_s" > "$CSV"

bench_run () {
    local impl="$1"; local rep="$2"
    local out="$OUT_DIR/threeway_${impl}_r${rep}"
    rm -rf "$out"
    local t0=$(python3 -c 'import time; print(time.perf_counter())')
    case "$impl" in
        surtgis)
            "$BIN" extract-patches \
                --features-dir "$DATA_DIR/features" \
                --points "$DATA_DIR/points.geojson" \
                --points-crs 32719 \
                --label-col cls --profile prithvi-v2 --size "$SIZE" \
                "$out" >/dev/null
            ;;
        python)
            "$VENV_PY" "$ROOT/benchmarks/bench_gfm_prep_py.py" \
                --features-dir "$DATA_DIR/features/t0" \
                --points "$DATA_DIR/points.geojson" \
                --label-col cls --profile prithvi-v2 --size "$SIZE" \
                --output "$out" >/dev/null 2>&1
            ;;
        instageo-style)
            "$VENV_PY" "$ROOT/benchmarks/bench_gfm_prep_instageo_style.py" \
                --features-dir "$DATA_DIR/features/t0" \
                --points "$DATA_DIR/points.geojson" \
                --label-col cls --profile prithvi-v2 --size "$SIZE" \
                --output "$out" >/dev/null 2>&1
            ;;
        *) echo "unknown impl: $impl"; exit 1 ;;
    esac
    local t1=$(python3 -c 'import time; print(time.perf_counter())')
    python3 -c "print(${t1} - ${t0})"
}

for REP in $(seq 1 "$N_REPS"); do
    for IMPL in surtgis python instageo-style; do
        WC=$(bench_run "$IMPL" "$REP")
        echo "${IMPL},${REP},${WC}" >> "$CSV"
        printf "  [rep=%d] %-15s : %.3f s\n" "$REP" "$IMPL" "$WC"
    done
done

echo ""
echo "Wrote $CSV"
echo ""
echo "Means:"
python3 -c "
import csv
from collections import defaultdict
rows = defaultdict(list)
with open('$CSV') as f:
    for r in csv.DictReader(f):
        rows[r['implementation']].append(float(r['wall_clock_s']))
for k, v in rows.items():
    mean = sum(v) / len(v)
    print(f'  {k:18s} : {mean:.3f} s (n={len(v)})')
"
