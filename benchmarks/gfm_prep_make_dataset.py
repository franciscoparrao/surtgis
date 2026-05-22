"""Generate a synthetic Prithvi-shaped dataset for the GFM-prep benchmark.

Produces: `<out>/features/t{0..T-1}/B0{2..7}.tif` (6 HLS bands × T timestamps,
all aligned to the same EPSG:32719 grid) plus `<out>/points.geojson` with
N labelled points in UTM coords. Bands carry random values in
the typical HLS surface reflectance range (0-10000 DN).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


PRITHVI_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--grid", type=int, default=2048)
    p.add_argument("--timestamps", type=int, default=3)
    p.add_argument("--n-points", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    # Maule-region anchor in EPSG:32719 (10 m pixels)
    origin_x, origin_y = 700000.0, 6100000.0
    px = 10.0
    transform = from_origin(origin_x, origin_y, px, px)
    crs = "EPSG:32719"

    feats_dir = args.out / "features"
    feats_dir.mkdir(parents=True, exist_ok=True)

    for ti in range(args.timestamps):
        ts_dir = feats_dir / f"t{ti}"
        ts_dir.mkdir(exist_ok=True)
        for band in PRITHVI_BANDS:
            arr = rng.uniform(200, 4000, (args.grid, args.grid)).astype(np.float32)
            with rasterio.open(
                ts_dir / f"{band}.tif",
                "w",
                driver="GTiff",
                height=args.grid,
                width=args.grid,
                count=1,
                dtype="float32",
                crs=crs,
                transform=transform,
                compress=None,
            ) as dst:
                dst.write(arr, 1)

    # Random points safely inside the grid (avoiding the margin so any
    # reasonable patch size fits without overflow)
    margin = max(256, 0)
    rows = rng.integers(margin, args.grid - margin, args.n_points)
    cols = rng.integers(margin, args.grid - margin, args.n_points)
    labels = rng.integers(0, 5, args.n_points)
    features = []
    for r, c, lbl in zip(rows, cols, labels):
        x = origin_x + (c + 0.5) * px
        y = origin_y - (r + 0.5) * px
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
            "properties": {"cls": int(lbl)},
        })
    gj = {"type": "FeatureCollection", "features": features}
    (args.out / "points.geojson").write_text(json.dumps(gj))

    print(f"Wrote {args.timestamps} timestamps × {len(PRITHVI_BANDS)} bands "
          f"(grid {args.grid}×{args.grid}) + {args.n_points} points to {args.out}")


if __name__ == "__main__":
    main()
