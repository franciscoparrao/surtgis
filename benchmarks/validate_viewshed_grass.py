#!/usr/bin/env python3
"""D4 validation: viewshed vs GRASS r.viewshed, cell-by-cell.

Compares the SurtGIS Bresenham viewshed against GRASS r.viewshed (boolean
mode) on the fbm_1000 fractal DEM for several observers, with and without
the Earth curvature + refraction correction. The algorithms differ in the
line-of-sight discretization (Bresenham rays vs r.viewshed's sweep with
interpolated events), so ~100% agreement is not expected — disagreement
concentrates on grazing cells whose LOS passes within float precision of an
occluder. Agreement is reported per configuration.

Usage: python3 benchmarks/validate_viewshed_grass.py
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

OBSERVERS = [(500, 500), (200, 700), (800, 300)]  # (row, col)
OBS_HEIGHT = 1.75
MAX_RADIUS = 400  # cells


def read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.transform


def run_surtgis(work, row, col, curvature):
    out = work / f"sg_{row}_{col}_{int(curvature)}.tif"
    cmd = [str(SURTGIS), "terrain", "viewshed", str(DEM_IN), str(out),
           "--observer-row", str(row), "--observer-col", str(col),
           "--observer-height", str(OBS_HEIGHT),
           "--max-radius", str(MAX_RADIUS)]
    if curvature:
        cmd.append("--curvature")
    subprocess.run(cmd, check=True, capture_output=True)
    return read(out)[0]


def run_grass(work, row, col, curvature):
    out = work / f"grass_{row}_{col}_{int(curvature)}.tif"
    with rasterio.open(DEM_IN) as src:
        east, north = src.transform * (col + 0.5, row + 0.5)
        cell = src.transform.a
    flags = "-b" + ("cr" if curvature else "")
    script = (
        f"r.in.gdal input={DEM_IN} output=dem --overwrite -o && "
        "g.region raster=dem && "
        f"r.viewshed {flags} input=dem output=vs coordinates={east},{north} "
        f"observer_elevation={OBS_HEIGHT} target_elevation=0.0 "
        f"max_distance={MAX_RADIUS * cell} --overwrite && "
        f"r.out.gdal input=vs output={out} format=GTiff type=Byte --overwrite"
    )
    subprocess.run(
        ["grass", "--tmp-location", str(DEM_IN), "--exec", "bash", "-c", script],
        check=True, capture_output=True,
    )
    return read(out)[0]


def main():
    if not SURTGIS.exists():
        sys.exit("build first: cargo build --release -p surtgis")
    if not DEM_IN.exists():
        sys.exit(f"missing DEM: {DEM_IN}")

    work = Path(tempfile.mkdtemp(prefix="validate_viewshed_"))
    print(f"work dir: {work}")

    # Restrict comparison to the max-radius disk around each observer.
    yy, xx = np.mgrid[0:1000, 0:1000]

    for curvature in (False, True):
        label = "curvature+refraction" if curvature else "no curvature"
        print(f"\n== {label} ==")
        for row, col in OBSERVERS:
            sg = run_surtgis(work, row, col, curvature)
            gr = run_grass(work, row, col, curvature)
            disk = (yy - row) ** 2 + (xx - col) ** 2 <= MAX_RADIUS**2
            sg_v = (sg == 1) & disk
            gr_v = (gr == 1) & disk
            agree = (sg_v == gr_v) & disk
            iou = (sg_v & gr_v).sum() / max((sg_v | gr_v).sum(), 1)
            print(f"observer ({row},{col}): agreement "
                  f"{100 * agree.sum() / disk.sum():.2f}% | visible IoU {iou:.3f} | "
                  f"visible cells surtgis {sg_v.sum()}, grass {gr_v.sum()}")


if __name__ == "__main__":
    main()
