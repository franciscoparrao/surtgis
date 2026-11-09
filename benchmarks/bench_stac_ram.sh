#!/bin/bash
# RAM regression benchmark for `stac composite` on Earth Search.
#
# Rationale: v0.6.19–v0.6.23 repeatedly shipped budget estimates that were off
# by 50–700% because per-band tile caches scaled differently than modelled.
# v0.6.24's outer-band refactor brought the real peak down to ~5 GB on Maule,
# but there's no CI guard to catch future regressions of that class.
#
# This script:
#   1. Runs a small ES bbox composite (one MGRS tile, 3 bands, 3 dates)
#   2. Samples RSS of the surtgis process every 0.25s
#   3. Asserts peak RSS stays below THRESHOLD_MB (default 2000 MB)
#   4. Asserts the RSS peak does not grow late in the run (no leak pattern)
#
# Usage: bash benchmarks/bench_stac_ram.sh
# Env:   THRESHOLD_MB=2000  (peak RSS fail threshold)
#        BBOX="-71.5,-35.5,-71.4,-35.4"  (tiny, single MGRS tile)
#        DATETIME="2024-01-01/2024-01-31"
#        LEAK_MIN_MB=64  (late-run peak growth below this never fails)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${SURTGIS_BIN:-$ROOT/target/release/surtgis}"
THRESHOLD_MB="${THRESHOLD_MB:-2000}"
BBOX="${BBOX:--71.5,-35.5,-71.4,-35.4}"
DATETIME="${DATETIME:-2024-01-01/2024-01-31}"
TMPDIR=$(mktemp -d)
LOG="$TMPDIR/composite.log"
RSS_LOG="$TMPDIR/rss.log"
OUTPUT="$TMPDIR/bench_out.tif"

trap "rm -rf $TMPDIR" EXIT

if [ ! -x "$BIN" ]; then
    echo "ERROR: surtgis binary not found at $BIN"
    echo "       Build first: cargo build --release -p surtgis"
    exit 2
fi

echo "=== STAC composite RAM benchmark ==="
echo "  binary:     $BIN"
echo "  threshold:  ${THRESHOLD_MB} MB"
echo "  bbox:       $BBOX"
echo "  datetime:   $DATETIME"
echo "  assets:     red,green,blue"
echo

# Start the composite in background; capture its PID
"$BIN" stac composite \
    --catalog es \
    --bbox "$BBOX" \
    --collection sentinel-2-l2a \
    --asset "red,green,blue" \
    --datetime "$DATETIME" \
    --max-scenes 3 \
    --strip-rows 256 \
    "$OUTPUT" \
    >"$LOG" 2>&1 &
PID=$!

echo "timestamp_s,rss_mb" > "$RSS_LOG"
START=$(date +%s.%N)
while kill -0 "$PID" 2>/dev/null; do
    if [ -r "/proc/$PID/status" ]; then
        RSS_KB=$(awk '/VmRSS:/ {print $2}' "/proc/$PID/status" 2>/dev/null || echo 0)
        RSS_MB=$((RSS_KB / 1024))
        NOW=$(date +%s.%N)
        T=$(awk "BEGIN{printf \"%.2f\", $NOW - $START}")
        echo "$T,$RSS_MB" >> "$RSS_LOG"
    fi
    sleep 0.25
done

# Under `set -e` a bare failing `wait` would end the script here, before
# the log tail below is printed.
STATUS=0
wait "$PID" || STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo "FAIL: surtgis exited with status $STATUS"
    echo "--- log tail ---"
    tail -30 "$LOG"
    exit "$STATUS"
fi

# Summary
PEAK_MB=$(awk -F, 'NR>1 && $2+0 > max {max=$2} END {print max+0}' "$RSS_LOG")
N_SAMPLES=$(($(wc -l < "$RSS_LOG") - 1))

echo
echo "=== Results ==="
echo "  samples:    $N_SAMPLES"
echo "  peak RSS:   ${PEAK_MB} MB"
echo "  threshold:  ${THRESHOLD_MB} MB"

# Check 1: peak below threshold
if [ "$PEAK_MB" -gt "$THRESHOLD_MB" ]; then
    echo "FAIL: peak ${PEAK_MB} MB exceeds threshold ${THRESHOLD_MB} MB"
    echo "--- RSS trajectory (last 20 samples) ---"
    tail -20 "$RSS_LOG"
    exit 1
fi

# Check 2: no growth over the final third of the run. RSS is a sawtooth
# (each strip allocates and frees ~25 MB), so single samples depend on the
# phase they land on: compare the peak of the final third with the peak of
# the middle third. Growth must also exceed LEAK_MIN_MB, so allocator noise
# on a ~100 MB run is not read as the GB-per-minute leaks this guards
# against.
LEAK_MIN_MB="${LEAK_MIN_MB:-64}"
if [ "$N_SAMPLES" -ge 12 ]; then
    read -r MID_PEAK END_PEAK < <(awk -F, -v n="$N_SAMPLES" '
        NR > 1 { i = NR - 1
                 if (i > n / 3 && i <= 2 * n / 3 && $2 + 0 > mid) mid = $2 + 0
                 if (i > 2 * n / 3 && $2 + 0 > end) end = $2 + 0 }
        END { print mid + 0, end + 0 }' "$RSS_LOG")
    GROWTH_MB=$((END_PEAK - MID_PEAK))
    GROWTH_PCT=$(awk "BEGIN{if($MID_PEAK>0) printf \"%.1f\", $GROWTH_MB*100/$MID_PEAK; else print 0}")
    echo "  late-run peak: ${MID_PEAK} MB → ${END_PEAK} MB (${GROWTH_PCT}%, ${GROWTH_MB} MB)"
    # Allow up to 20% (or LEAK_MIN_MB) late-run growth; more suggests a leak
    if awk "BEGIN{exit !($GROWTH_PCT > 20)}" && [ "$GROWTH_MB" -gt "$LEAK_MIN_MB" ]; then
        echo "FAIL: RSS peak grew ${GROWTH_PCT}% (${GROWTH_MB} MB) in the final third (leak signature)"
        echo "--- RSS trajectory ---"
        tr '\n' ' ' < "$RSS_LOG"
        echo
        exit 1
    fi
fi

echo "PASS: peak RSS within budget, no leak signature."
