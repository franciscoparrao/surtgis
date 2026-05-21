"""Reference Python implementation of `surtgis extract-patches`.

Why this exists: companion benchmark for SurtGIS extract-patches.
Reproduces the same logic (read N feature rasters, extract patches at
point locations, apply per-band z-score normalization, write NPY)
using the canonical Python stack (rasterio + numpy). The wall-clock
difference between this script and `surtgis extract-patches` on the
same inputs is the "switching cost" a user pays for staying in
Python — the load-bearing claim of the InstaGeo gap analysis (arxiv
2510.05617) that motivates SurtGIS's G2 work.

Not an apples-to-apples reimplementation of InstaGeo or raster-vision:
those add full STAC + cloud-masking + tile-fetching, which we don't
bench here. Our scope is the deterministic local-disk hot loop.

Usage:
    python bench_gfm_prep_py.py --features-dir DIR --points GEOJSON \\
        --label-col LABEL --size 224 --output OUT_DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio


# Prithvi-EO-2.0-300M per-band z-score statistics
# (huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
PRITHVI_V2_MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32
)
PRITHVI_V2_STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32
)


def list_tifs_shallow(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})


def load_band_stack(features_dir: Path) -> tuple[list[str], list[np.ndarray], dict]:
    """Read all top-level .tifs in features_dir as a band stack.

    Returns (band_names, band_arrays, geo_info). Subdirs are ignored —
    callers in multi-timestamp mode iterate them explicitly.
    """
    tifs = list_tifs_shallow(features_dir)
    if not tifs:
        raise SystemExit(f"No .tif files in {features_dir}")
    band_names: list[str] = []
    arrays: list[np.ndarray] = []
    geo_info: dict | None = None
    for tif in tifs:
        with rasterio.open(tif) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr == src.nodata] = np.nan if src.nodata is not None else arr[0, 0]
            arrays.append(arr)
            if geo_info is None:
                geo_info = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "shape": src.shape,
                }
        band_names.append(tif.stem)
    return band_names, arrays, geo_info


def geo_to_pixel(x: float, y: float, transform) -> tuple[float, float]:
    """Inverse affine transform: world coords -> (col, row)."""
    inv = ~transform
    col, row = inv * (x, y)
    return col, row


def read_points(geojson_path: Path, label_col: str) -> list[tuple[float, float, float]]:
    """Read [(x, y, label), ...] from a GeoJSON FeatureCollection."""
    with open(geojson_path) as f:
        gj = json.load(f)
    out: list[tuple[float, float, float]] = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] != "Point":
            continue
        props = feat.get("properties", {})
        if label_col not in props:
            continue
        x, y = geom["coordinates"][:2]
        out.append((x, y, props[label_col]))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path, required=True)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("--label-col", required=True)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--skip-nan-threshold", type=float, default=0.1)
    p.add_argument(
        "--profile",
        choices=["prithvi-v2", "none"],
        default="prithvi-v2",
        help="Apply per-band z-score normalization for this GFM target",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    t0 = time.perf_counter()

    band_names, arrays, geo = load_band_stack(args.features_dir)
    n_bands = len(arrays)
    rows, cols = geo["shape"]
    half = args.size // 2

    if args.profile == "prithvi-v2":
        if n_bands != 6:
            raise SystemExit(
                f"Profile prithvi-v2 expects 6 bands, got {n_bands}"
            )

    t_load = time.perf_counter() - t0

    # Build candidate specs from points
    t1 = time.perf_counter()
    pts = read_points(args.points, args.label_col)
    specs: list[tuple[int, int, float]] = []  # (row, col, label)
    for x, y, lbl in pts:
        col_f, row_f = geo_to_pixel(x, y, geo["transform"])
        r, c = int(np.floor(row_f)), int(np.floor(col_f))
        if r < half or c < half:
            continue
        if r + (args.size - half) > rows or c + (args.size - half) > cols:
            continue
        specs.append((r, c, lbl))
    t_specs = time.perf_counter() - t1

    # Extract patches
    t2 = time.perf_counter()
    kept_bufs: list[np.ndarray] = []
    kept_labels: list[float] = []
    nan_skipped = 0
    for r, c, lbl in specs:
        r0, c0 = r - half, c - half
        patch = np.empty((n_bands, args.size, args.size), dtype=np.float32)
        for bi, band in enumerate(arrays):
            patch[bi] = band[r0:r0 + args.size, c0:c0 + args.size]
        nan_count = np.isnan(patch).sum()
        if nan_count / patch.size > args.skip_nan_threshold:
            nan_skipped += 1
            continue
        if args.profile == "prithvi-v2":
            for bi in range(n_bands):
                m = PRITHVI_V2_MEAN[bi]
                s = PRITHVI_V2_STD[bi] if PRITHVI_V2_STD[bi] > 1e-10 else 1e-10
                mask = np.isfinite(patch[bi])
                patch[bi, mask] = (patch[bi, mask] - m) / s
        kept_bufs.append(patch)
        kept_labels.append(lbl)
    t_extract = time.perf_counter() - t2

    # Write outputs
    t3 = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    patches = np.stack(kept_bufs, axis=0) if kept_bufs else np.empty((0, n_bands, args.size, args.size), dtype=np.float32)
    np.save(args.output / "patches.npy", patches)
    labels = np.array(kept_labels)
    if np.issubdtype(labels.dtype, np.integer) or labels.dtype == np.int64:
        np.save(args.output / "labels.npy", labels.astype(np.int64))
    else:
        np.save(args.output / "labels.npy", labels.astype(np.float32))
    meta = {
        "bands": band_names,
        "patch_size": args.size,
        "n_patches": int(patches.shape[0]),
        "nan_skipped": int(nan_skipped),
        "profile": args.profile,
        "implementation": "python_reference",
        "timings_seconds": {
            "load_bands": round(t_load, 4),
            "build_specs": round(t_specs, 4),
            "extract_patches": round(t_extract, 4),
            "write_outputs": 0.0,  # filled in after this line
        },
    }
    t_write = time.perf_counter() - t3
    meta["timings_seconds"]["write_outputs"] = round(t_write, 4)
    meta["timings_seconds"]["total"] = round(time.perf_counter() - t0, 4)
    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))

    # Summary to stderr so timing CSV consumers can scrape stdout cleanly
    print(f"Python reference: {patches.shape[0]} patches, "
          f"total {meta['timings_seconds']['total']:.3f}s "
          f"(load {t_load:.3f}s, specs {t_specs:.3f}s, "
          f"extract {t_extract:.3f}s, write {t_write:.3f}s)",
          file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
