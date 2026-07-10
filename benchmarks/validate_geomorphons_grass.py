#!/usr/bin/env python3
"""D2 validation: geomorphons vs GRASS r.geomorphon, cell by cell.

SurtGIS reimplements r.geomorphon's default mode (comparison=anglev1, basic
correction) exactly, so — unlike flat-routing, where any monotone drainage is
valid — the right metric here IS cell-by-cell agreement. Small disagreement
is expected only from float precision (GRASS computes elevation deltas in
f32, SurtGIS in f64), which can flip cells whose angles sit exactly on the
flatness threshold or on a zenith/nadir tie.

Runs several parameter combinations (search/skip/dist) on the fbm_1000
fractal DEM and reports the agreement and the disagreement histogram.

Usage: python3 benchmarks/validate_geomorphons_grass.py
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
SURTGIS = REPO / "target/release/surtgis"
DEM_IN = REPO / "benchmarks/results/dems/fbm_1000_raw.tif"

FORM_NAMES = {
    0: "nodata", 1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur",
    6: "slope", 7: "hollow", 8: "footslope", 9: "valley", 10: "pit",
}

COMBOS = [
    # (search, skip, flat, dist) — GRASS defaults first
    (3, 0, 1.0, 0.0),
    (10, 0, 1.0, 0.0),
    (10, 2, 1.0, 0.0),
    (15, 0, 1.0, 5.0),
]


def read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def run_surtgis(work, search, skip, flat, dist):
    out = work / f"sg_{search}_{skip}_{dist}.tif"
    subprocess.run(
        [str(SURTGIS), "terrain", "geomorphons", str(DEM_IN), str(out),
         "--radius", str(search), "--flatness", str(flat),
         "--skip", str(skip), "--flat-dist", str(dist)],
        check=True, capture_output=True,
    )
    return read(out)


def run_grass(work, search, skip, flat, dist):
    out = work / f"grass_{search}_{skip}_{dist}.tif"
    script = (
        f"r.in.gdal input={DEM_IN} output=dem --overwrite -o && "
        "g.region raster=dem && "
        f"r.geomorphon elevation=dem forms=forms search={search} skip={skip} "
        f"flat={flat} dist={dist} --overwrite && "
        f"r.out.gdal input=forms output={out} format=GTiff type=Int32 "
        "createopt=COMPRESS=DEFLATE --overwrite"
    )
    subprocess.run(
        ["grass", "--tmp-location", str(DEM_IN), "--exec", "bash", "-c", script],
        check=True, capture_output=True,
    )
    return read(out)


def main():
    if not SURTGIS.exists():
        sys.exit("build first: cargo build --release -p surtgis")
    if not DEM_IN.exists():
        sys.exit(f"missing DEM: {DEM_IN}")

    work = Path(tempfile.mkdtemp(prefix="validate_geomorphons_"))
    print(f"work dir: {work}")

    for search, skip, flat, dist in COMBOS:
        sg = run_surtgis(work, search, skip, flat, dist)
        gr = run_grass(work, search, skip, flat, dist)

        gr = np.where((gr >= 1) & (gr <= 10), gr, 0).astype(np.uint8)
        both = (sg != 0) & (gr != 0)
        same = (sg == gr) & both
        n_both, n_same = int(both.sum()), int(same.sum())

        print(f"\n== search={search} skip={skip} flat={flat} dist={dist} ==")
        print(f"valid in SurtGIS: {(sg != 0).sum()}, in GRASS: {(gr != 0).sum()}, "
              f"in both: {n_both}")
        print(f"cell-by-cell agreement: {n_same}/{n_both} "
              f"({100 * n_same / n_both:.3f}%)")

        if n_same < n_both:
            diff = both & ~same
            pairs, counts = np.unique(
                np.stack([sg[diff], gr[diff]]).reshape(2, -1).T,
                axis=0, return_counts=True,
            )
            top = sorted(zip(counts, pairs.tolist()), reverse=True)[:5]
            print("top disagreements (surtgis -> grass):")
            for c, (a, b) in top:
                print(f"  {FORM_NAMES[a]:>9} -> {FORM_NAMES[b]:<9} {c}")


if __name__ == "__main__":
    main()
