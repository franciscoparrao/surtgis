#!/usr/bin/env python3
"""One-off STAC query latency measurement against Element 84 Earth Search.

Purpose: provide a bounded estimate of the STAC orchestration overhead
that InstaGeo incurs upstream of its chip-extraction loop, used in
paper-gfm-prep §6.4 to quantify the gap between the "InstaGeo-style"
proxy benchmark (which skips STAC) and InstaGeo-as-deployed.

Methodology: a single-bbox, single-date-window catalog search returning
on the order of 1-10 items. This is the cheapest STAC query that
InstaGeo would issue per AOI; bulk-catalog workflows pay
proportionally more. We report median over 5 reps with 1-second
spacing to discount transient network jitter.

Run from any directory:
  python3 benchmarks/measure_stac_latency.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
import urllib.error


CATALOG = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BBOX = [-72.90, 44.45, -72.65, 44.65]  # Smugglers Notch, VT (~22 km AOI)
DATE_RANGE = "2024-06-01T00:00:00Z/2024-08-31T23:59:59Z"
N_REPS = 5
SPACING_S = 1.0
TIMEOUT_S = 30


def stac_search() -> tuple[float, int]:
    """Issue one POST /search and return (latency_s, n_items)."""
    body = {
        "collections": [COLLECTION],
        "bbox": BBOX,
        "datetime": DATE_RANGE,
        "limit": 50,
    }
    req = urllib.request.Request(
        f"{CATALOG}/search",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/geo+json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        payload = json.loads(r.read())
    t = time.perf_counter() - t0
    return t, len(payload.get("features", []))


def main() -> int:
    print(f"STAC API: {CATALOG}")
    print(f"Collection: {COLLECTION}")
    print(f"BBox: {BBOX}  (~22 km AOI)")
    print(f"Date range: {DATE_RANGE}")
    print()
    latencies = []
    n_items = 0
    for i in range(N_REPS):
        try:
            lat, n = stac_search()
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"  rep {i + 1}: FAILED ({type(exc).__name__}: {exc})")
            continue
        latencies.append(lat)
        n_items = n
        print(f"  rep {i + 1}: {lat:.3f} s  ({n} items returned)")
        if i < N_REPS - 1:
            time.sleep(SPACING_S)
    if not latencies:
        print("All reps failed.")
        return 1
    print()
    print(f"Latency stats (n={len(latencies)}, items per query={n_items}):")
    print(f"  median  : {statistics.median(latencies):.3f} s")
    print(f"  mean    : {statistics.mean(latencies):.3f} s")
    if len(latencies) > 1:
        print(f"  std     : {statistics.stdev(latencies):.3f} s")
    print(f"  min/max : {min(latencies):.3f} / {max(latencies):.3f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
