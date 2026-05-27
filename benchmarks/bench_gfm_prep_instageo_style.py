"""InstaGeo-style chip extraction benchmark (Python xarray + rioxarray
+ per-chip GeoTIFF write).

This is NOT a verbatim re-execution of InstaGeo's chip_creator module.
That module is tightly coupled to STAC queries, a multi-source config
system (HLS/Sentinel-2/Sentinel-1), and the CC-BY-NC-SA-licensed
`instageo` package. Running it on local synthetic data would require
mocking STAC items and accepting a non-commercial license in the
benchmark dependency tree.

Instead, we reproduce the pattern that InstaGeo's internal loop
follows on local data:

  1. Load rasters as an xr.Dataset via rioxarray (the InstaGeo
     load_data() return type).
  2. Slice patches via xr.Dataset.isel() (the InstaGeo
     RasterDataPipeline.process_tile() inner loop, lines 933-936 of
     /tmp/instageo/instageo/data/data_pipeline.py).
  3. Apply z-score normalisation.
  4. Write each chip as a per-chip GeoTIFF (the InstaGeo output
     contract — one file per chip + one per seg_map).

The wall-clock measurement isolates the local extraction-and-write
cost from the STAC-orchestration cost that InstaGeo additionally
incurs upstream. The comparison against SurtGIS extract-patches and
the rasterio+numpy Python reference is thus an upper bound on the
SurtGIS advantage — InstaGeo as deployed is slower than this script
because it also does STAC, dask scheduling, and segmentation-map
generation.

Output: same wall-clock format as bench_gfm_prep_py.py so a single
CSV row per rep is appended.
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
import rioxarray
import xarray as xr

PRITHVI_V2_MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32
)
PRITHVI_V2_STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32
)


def list_tifs(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})


def load_xrds(features_dir: Path) -> xr.Dataset:
    """Stack feature rasters into a single xr.Dataset. This mirrors
    InstaGeo's `BaseDataPipeline.load_data()` return type."""
    tifs = list_tifs(features_dir)
    if not tifs:
        raise SystemExit(f"No .tif files in {features_dir}")
    data_vars = {}
    for tif in tifs:
        # rioxarray opens a single-band TIFF as a DataArray with band dim
        da = rioxarray.open_rasterio(tif, masked=True)
        if da.shape[0] == 1:
            da = da.squeeze("band", drop=True)
        data_vars[tif.stem] = da
    return xr.Dataset(data_vars)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True, type=Path)
    ap.add_argument("--points", required=True, type=Path)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--profile", default="prithvi-v2")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    if args.profile != "prithvi-v2":
        raise SystemExit("only prithvi-v2 profile is implemented in this bench")

    args.output.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # Load all rasters into a single xr.Dataset (mirrors InstaGeo)
    t0 = time.perf_counter()
    ds = load_xrds(args.features_dir)
    band_names = sorted(ds.data_vars)
    t_load = time.perf_counter() - t0

    transform = ds[band_names[0]].rio.transform()
    crs = ds[band_names[0]].rio.crs

    # Read points
    t0 = time.perf_counter()
    with open(args.points) as f:
        feats = json.load(f)["features"]
    point_specs = []
    for feat in feats:
        if feat["geometry"]["type"] != "Point":
            continue
        x, y = feat["geometry"]["coordinates"]
        label = feat["properties"].get(args.label_col)
        if label is None:
            continue
        col_f, row_f = ~transform * (x, y)
        col_i = int(col_f)
        row_i = int(row_f)
        half = args.size // 2
        if row_i < half or col_i < half:
            continue
        if (row_i + (args.size - half)) > ds.sizes["y"]:
            continue
        if (col_i + (args.size - half)) > ds.sizes["x"]:
            continue
        point_specs.append((row_i, col_i, label))
    t_specs = time.perf_counter() - t0

    # Extract chips via xr.Dataset.isel() (the InstaGeo pattern) and
    # write per-chip GeoTIFFs (the InstaGeo output contract).
    t0 = time.perf_counter()
    half = args.size // 2
    chips_dir = args.output / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    labels_log = []

    n_bands = len(band_names)
    for idx, (row_i, col_i, label) in enumerate(point_specs):
        y0 = row_i - half
        y1 = y0 + args.size
        x0 = col_i - half
        x1 = x0 + args.size

        chip_ds = ds.isel(
            x=slice(x0, x1),
            y=slice(y0, y1),
        )

        # Stack bands -> numpy [C, H, W], apply z-score
        arr = np.stack([chip_ds[b].values for b in band_names], axis=0).astype(np.float32)
        for c in range(n_bands):
            arr[c] = (arr[c] - PRITHVI_V2_MEAN[c]) / PRITHVI_V2_STD[c]

        # Write per-chip GeoTIFF (InstaGeo's output format)
        chip_path = chips_dir / f"chip_{idx:05d}.tif"
        chip_transform = rasterio.transform.from_origin(
            transform.c + col_i * transform.a - half * transform.a,
            transform.f + row_i * transform.e - half * transform.e,
            transform.a,
            -transform.e,
        )
        with rasterio.open(
            chip_path,
            "w",
            driver="GTiff",
            count=n_bands,
            height=args.size,
            width=args.size,
            dtype=rasterio.float32,
            crs=crs,
            transform=chip_transform,
        ) as dst:
            for c in range(n_bands):
                dst.write(arr[c], c + 1)
        labels_log.append((idx, label))
    t_extract_write = time.perf_counter() - t0

    # Sidecar labels JSON (InstaGeo writes them as a separate file)
    with open(args.output / "labels.json", "w") as f:
        json.dump({"items": [{"idx": i, "label": l} for i, l in labels_log]}, f)

    t_total = time.perf_counter() - t_start
    print(
        f"InstaGeo-style ref: {len(point_specs)} chips, total {t_total:.3f}s "
        f"(xrload {t_load:.3f}s, specs {t_specs:.3f}s, "
        f"extract+per-chip-write {t_extract_write:.3f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
