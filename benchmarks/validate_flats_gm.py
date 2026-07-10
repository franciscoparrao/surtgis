#!/usr/bin/env python3
"""D6 validation: Garbrecht-Martz flat resolution vs WhiteboxTools and GRASS.

Builds a DEM with large flat regions (quantizing the fbm_1000 fractal DEM to
integer steps), then runs the D8 pipeline of three tools and compares:

  1. Flat coverage: fraction of flat cells that receive a flow direction
     (SurtGIS resolve_flats vs WBT fix_flats fill).
  2. Cell-by-cell direction agreement (overall and inside flats). Exact
     agreement is not expected on flats — G-M double gradient (SurtGIS) vs
     flat-increment fill (WBT) are different constructions — the metric that
     matters is where the water ends up:
  3. Flow accumulation agreement: Pearson r on log1p(accumulation), and
     IoU of the extracted stream network (accumulation >= threshold).

Usage: python3 benchmarks/validate_flats_gm.py [--step 10] [--threshold 100]
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
SURTGIS = REPO / "target/release/surtgis"
DEM_IN = REPO / "benchmarks/results/dems/fbm_1000_raw.tif"

# WBT d8_pointer codes (clockwise powers of two) -> (drow, dcol)
WBT_OFFSETS = {
    1: (-1, 1),   # NE
    2: (0, 1),    # E
    4: (1, 1),    # SE
    8: (1, 0),    # S
    16: (1, -1),  # SW
    32: (0, -1),  # W
    64: (-1, -1), # NW
    128: (-1, 0), # N
}
# SurtGIS codes (counter-clockwise from East) -> (drow, dcol)
SG_OFFSETS = {
    1: (0, 1),    # E
    2: (-1, 1),   # NE
    3: (-1, 0),   # N
    4: (-1, -1),  # NW
    5: (0, -1),   # W
    6: (1, -1),   # SW
    7: (1, 0),    # S
    8: (1, 1),    # SE
}
# GRASS r.watershed drainage codes (counter-clockwise from NE) -> (drow, dcol)
GRASS_OFFSETS = {
    1: (-1, 1),   # NE
    2: (-1, 0),   # N
    3: (-1, -1),  # NW
    4: (0, -1),   # W
    5: (1, -1),   # SW
    6: (1, 0),    # S
    7: (1, 1),    # SE
    8: (0, 1),    # E
}


def read(path):
    with rasterio.open(path) as src:
        return src.read(1), src.profile


def write(path, data, profile, dtype="float32", nodata=None):
    p = dict(profile)
    p.update(dtype=dtype, count=1, nodata=nodata)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(dtype), 1)


def to_offsets(dirs, table):
    """Direction codes -> (drow, dcol) arrays; (0,0) where no direction."""
    drow = np.zeros(dirs.shape, dtype=np.int8)
    dcol = np.zeros(dirs.shape, dtype=np.int8)
    for code, (dr, dc) in table.items():
        m = dirs == code
        drow[m] = dr
        dcol[m] = dc
    return drow, dcol


def flat_mask(dem):
    """Cells with no strictly lower 8-neighbor but >=1 equal neighbor."""
    z = np.pad(dem, 1, mode="edge")
    has_lower = np.zeros(dem.shape, dtype=bool)
    has_equal = np.zeros(dem.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = z[1 + dr : 1 + dr + dem.shape[0], 1 + dc : 1 + dc + dem.shape[1]]
            has_lower |= nb < dem
            has_equal |= nb == dem
    return ~has_lower & has_equal


def stream_iou(acc_a, acc_b, threshold):
    sa, sb = acc_a >= threshold, acc_b >= threshold
    inter, union = (sa & sb).sum(), (sa | sb).sum()
    return inter / union if union else float("nan")


def pearson_log(acc_a, acc_b):
    a, b = np.log1p(np.abs(acc_a)).ravel(), np.log1p(np.abs(acc_b)).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def run_surtgis(quant_tif, work):
    filled = work / "sg_filled.tif"
    fdir = work / "sg_fdir.tif"
    acc = work / "sg_acc.tif"
    for cmd in (
        [SURTGIS, "hydrology", "priority-flood", quant_tif, filled, "--epsilon", "0"],
        [SURTGIS, "hydrology", "flow-direction", filled, fdir],
        [SURTGIS, "hydrology", "flow-accumulation", fdir, acc],
    ):
        subprocess.run([str(c) for c in cmd], check=True, capture_output=True)
    return read(filled)[0], read(fdir)[0], read(acc)[0]


def run_wbt(quant_tif, work):
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_working_dir(str(work))
    wbt.set_verbose_mode(False)
    filled = work / "wbt_filled.tif"
    fdir = work / "wbt_fdir.tif"
    acc = work / "wbt_acc.tif"
    wbt.fill_depressions(str(quant_tif), str(filled), fix_flats=True)
    wbt.d8_pointer(str(filled), str(fdir))
    wbt.d8_flow_accumulation(str(fdir), str(acc), out_type="cells", pntr=True)
    return read(fdir)[0], read(acc)[0]


def run_grass(quant_tif, work):
    """r.watershed SFD on the quantized DEM (its own conditioning)."""
    grassdir = work / "grassdata"
    script = (
        f"r.in.gdal input={quant_tif} output=dem --overwrite -o && "
        "g.region raster=dem && "
        "r.watershed -s elevation=dem drainage=drain accumulation=acc --overwrite && "
        f"r.out.gdal input=drain output={work}/grass_drain.tif format=GTiff --overwrite && "
        f"r.out.gdal input=acc output={work}/grass_acc.tif format=GTiff --overwrite"
    )
    subprocess.run(
        ["grass", "--tmp-location", str(quant_tif), "--exec", "bash", "-c", script],
        check=True,
        capture_output=True,
        cwd=grassdir.parent,
    )
    return read(work / "grass_drain.tif")[0], read(work / "grass_acc.tif")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=10.0, help="quantization step (m)")
    ap.add_argument("--threshold", type=float, default=100.0, help="stream accum threshold (cells)")
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    args = ap.parse_args()

    if not SURTGIS.exists():
        sys.exit("build first: cargo build --release -p surtgis-cli")
    if not DEM_IN.exists():
        sys.exit(f"missing DEM: {DEM_IN}")

    work = Path(tempfile.mkdtemp(prefix="validate_flats_"))
    print(f"work dir: {work}")

    dem, profile = read(DEM_IN)
    quant = np.floor(dem / args.step) * args.step
    quant_tif = work / "quantized.tif"
    write(quant_tif, quant, profile)

    # --- SurtGIS ------------------------------------------------------------
    sg_filled, sg_fdir, sg_acc = run_surtgis(quant_tif, work)
    flats = flat_mask(sg_filled)
    n_flats = int(flats.sum())
    sg_resolved = int((sg_fdir[flats] != 0).sum())
    print(f"\nDEM {dem.shape}, step {args.step} m -> {n_flats} flat cells "
          f"({100 * n_flats / dem.size:.1f}% of raster) after epsilon-0 fill")
    print("\n== SurtGIS (Garbrecht-Martz) ==")
    print(f"flat cells with direction: {sg_resolved}/{n_flats} "
          f"({100 * sg_resolved / n_flats:.2f}%)")
    print(f"cells with no direction overall: {int((sg_fdir == 0).sum())}")

    # --- WhiteboxTools -------------------------------------------------------
    wbt_fdir, wbt_acc = run_wbt(quant_tif, work)
    wbt_resolved = int((wbt_fdir[flats] != 0).sum())
    print("\n== WhiteboxTools (fix_flats fill) ==")
    print(f"flat cells with direction: {wbt_resolved}/{n_flats} "
          f"({100 * wbt_resolved / n_flats:.2f}%)")

    def report_pair(name, dirs_a, offs_a, dirs_b, offs_b, acc_a, acc_b):
        a_dr, a_dc = to_offsets(dirs_a, offs_a)
        b_dr, b_dc = to_offsets(dirs_b, offs_b)
        both = (a_dr != 0) | (a_dc != 0)
        both &= (b_dr != 0) | (b_dc != 0)
        same = (a_dr == b_dr) & (a_dc == b_dc) & both
        nf = both & ~flats
        fl = both & flats
        print(f"\n== {name} ==")
        print(f"direction agreement (both directed): {100 * same.sum() / both.sum():.1f}% overall | "
              f"{100 * same[nf].sum() / nf.sum():.1f}% outside flats | "
              f"{100 * same[fl].sum() / fl.sum():.1f}% inside flats")
        print(f"log-accumulation Pearson r: {pearson_log(acc_a, acc_b):.4f}")
        print(f"stream network IoU (acc>={args.threshold:.0f}): "
              f"{stream_iou(np.abs(acc_a), np.abs(acc_b), args.threshold):.3f}")

    report_pair("SurtGIS vs WhiteboxTools", sg_fdir, SG_OFFSETS,
                wbt_fdir.astype(np.int32), WBT_OFFSETS, sg_acc, wbt_acc)

    # --- GRASS (optional) -----------------------------------------------------
    try:
        (work / "grassdata").mkdir(exist_ok=True)
        gr_drain, gr_acc = run_grass(quant_tif, work)
        gr_dirs = np.where(gr_drain > 0, gr_drain, 0).astype(np.int32)
        print("\n(GRASS r.watershed -s does its own conditioning — no fill)")
        report_pair("SurtGIS vs GRASS", sg_fdir, SG_OFFSETS, gr_dirs, GRASS_OFFSETS,
                    sg_acc, gr_acc)
        report_pair("WhiteboxTools vs GRASS (reference baseline)",
                    wbt_fdir.astype(np.int32), WBT_OFFSETS, gr_dirs, GRASS_OFFSETS,
                    wbt_acc, gr_acc)
    except Exception as e:  # GRASS is a secondary reference; report and move on
        print(f"\nGRASS comparison skipped: {e}")

    if args.keep:
        print(f"\noutputs kept in {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
