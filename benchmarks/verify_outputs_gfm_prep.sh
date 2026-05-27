#!/usr/bin/env bash
# Output-correctness verification between SurtGIS and Python reference.
#
# Both extract-patches implementations should produce comparable NumPy
# tensors for the patch payloads. This script:
#   1. Re-runs both implementations with a fixed seed on the
#      single-timestamp slice (so the shapes match exactly).
#   2. Computes SHA-256 hashes of patches.npy and labels.npy.
#   3. If hashes diverge (expected because of float-precision quirks
#      and nodata-encoding differences), computes element-wise diff
#      statistics so the divergence is quantified, not hand-waved.
#
# Writes results to benchmarks/results/gfm_prep/output_verification.txt.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SURTGIS_BIN:-$ROOT/target/release/surtgis}"
OUT_DIR="$ROOT/benchmarks/results/gfm_prep"
DATA_DIR="$OUT_DIR/synthetic_dataset"
REPORT="$OUT_DIR/output_verification.txt"

if [ ! -d "$DATA_DIR/features/t0" ]; then
    echo "Synthetic dataset not found — run benchmarks/run_gfm_prep_bench.sh first."
    exit 1
fi

OUT_SUR="$OUT_DIR/verify_surtgis"
OUT_PY="$OUT_DIR/verify_python"
rm -rf "$OUT_SUR" "$OUT_PY"

echo "[verify] Running surtgis on single-timestamp slice"
"$BIN" extract-patches \
    --features-dir "$DATA_DIR/features/t0" \
    --points "$DATA_DIR/points.geojson" \
    --points-crs 32719 \
    --label-col cls \
    --profile prithvi-v2 \
    --size 224 \
    "$OUT_SUR" > /dev/null

echo "[verify] Running Python reference"
python3 "$ROOT/benchmarks/bench_gfm_prep_py.py" \
    --features-dir "$DATA_DIR/features/t0" \
    --points "$DATA_DIR/points.geojson" \
    --label-col cls \
    --profile prithvi-v2 \
    --size 224 \
    --output "$OUT_PY" > /dev/null

{
    echo "Output verification: SurtGIS vs Python reference"
    echo "Generated: $(date -Iseconds)"
    echo ""
    echo "----- SHA-256 hashes -----"
    for f in patches.npy labels.npy; do
        S=$(sha256sum "$OUT_SUR/$f" 2>/dev/null | awk '{print $1}' || echo "MISSING")
        P=$(sha256sum "$OUT_PY/$f" 2>/dev/null | awk '{print $1}' || echo "MISSING")
        if [ "$S" = "$P" ]; then
            echo "  $f: MATCH ($S)"
        else
            echo "  $f: DIFFER"
            echo "    surtgis: $S"
            echo "    python : $P"
        fi
    done
    echo ""
    echo "----- Element-wise diff statistics -----"
    python3 - "$OUT_SUR" "$OUT_PY" <<'PY'
import sys
import numpy as np
sur = sys.argv[1]
pyd = sys.argv[2]
for name in ["patches.npy", "labels.npy"]:
    try:
        a = np.load(f"{sur}/{name}")
        b = np.load(f"{pyd}/{name}")
    except FileNotFoundError as e:
        print(f"  {name}: SKIP ({e})")
        continue
    print(f"  {name}")
    print(f"    shapes : surtgis={a.shape} python={b.shape}")
    print(f"    dtypes : surtgis={a.dtype} python={b.dtype}")
    if a.shape != b.shape:
        print("    (shape mismatch; cannot compare elementwise)")
        continue
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    finite = np.isfinite(af) & np.isfinite(bf)
    diff = (af - bf)[finite]
    nan_only_sur = np.isnan(af) & ~np.isnan(bf)
    nan_only_py  = ~np.isnan(af) & np.isnan(bf)
    print(f"    n_total         : {a.size}")
    print(f"    n_finite_both   : {int(finite.sum())}")
    print(f"    n_nan_surtgis_only : {int(nan_only_sur.sum())}")
    print(f"    n_nan_python_only  : {int(nan_only_py.sum())}")
    if diff.size:
        print(f"    diff abs (finite): max={np.abs(diff).max():.6g}  mean={np.abs(diff).mean():.6g}")
        print(f"    diff rel (finite): max={(np.abs(diff)/(np.abs(bf[finite])+1e-12)).max():.6g}")
PY
} | tee "$REPORT"

echo ""
echo "Wrote $REPORT"
