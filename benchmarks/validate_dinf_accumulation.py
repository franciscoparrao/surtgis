#!/usr/bin/env python3
"""Validate SurtGIS D-infinity flow accumulation against WhiteboxTools.

Both implementations follow Tarboton (1997): flow split between the two
D8 neighbors bracketing the continuous flow angle, proportional to the
angular offset.

Steps:
1. Fill the Andes UTM DEM once with WhiteboxTools FillDepressions
2. WBT: DInfPointer + DInfFlowAccumulation (out_type=cells) on the filled DEM
3. SurtGIS: flow-direction-dinf + flow-accumulation-dinf CLI on the same DEM
4. Compare log-accumulation pixel-by-pixel (Pearson r, relative error)

Conventions reconciled here:
- WBT cells accumulation INCLUDES the cell itself (headwater = 1)
- SurtGIS accumulation EXCLUDES the cell itself (headwater = 0)
  → compare surtgis + 1 vs WBT.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
ANDES_UTM = BASE / "tests" / "fixtures" / "andes_chile_30m_utm.tif"
SURTGIS = BASE / "target" / "release" / "surtgis"


def read_geotiff(path):
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def main():
    if not ANDES_UTM.exists():
        sys.exit(f"fixture not found: {ANDES_UTM}")
    if not SURTGIS.exists():
        sys.exit(f"surtgis release binary not found: {SURTGIS} (cargo build --release -p surtgis)")

    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)

    def wbt_run(name, fn, *args, **kwargs):
        ret = fn(*args, **kwargs)
        if ret != 0:
            sys.exit(f"WBT {name} failed (exit {ret})")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        filled = tmp / "filled.tif"
        wbt_pntr = tmp / "wbt_dinf_pntr.tif"
        wbt_acc = tmp / "wbt_dinf_acc.tif"
        sg_ang = tmp / "sg_dinf_ang.tif"
        sg_acc = tmp / "sg_dinf_acc.tif"

        print("[1/4] WBT FillDepressions...")
        wbt_run("fill_depressions", wbt.fill_depressions, str(ANDES_UTM), str(filled), fix_flats=True)

        print("[2/4] WBT DInfPointer + DInfFlowAccumulation (cells)...")
        wbt_run("d_inf_pointer", wbt.d_inf_pointer, str(filled), str(wbt_pntr))
        wbt_run(
            "d_inf_flow_accumulation",
            wbt.d_inf_flow_accumulation,
            str(wbt_pntr),
            str(wbt_acc),
            out_type="cells",
            pntr=True,
        )

        print("[3/4] SurtGIS flow-direction-dinf + flow-accumulation-dinf...")
        subprocess.run(
            [str(SURTGIS), "hydrology", "flow-direction-dinf", str(filled), str(sg_ang)],
            check=True,
        )
        subprocess.run(
            [str(SURTGIS), "hydrology", "flow-accumulation-dinf", str(sg_ang), str(sg_acc)],
            check=True,
        )

        print("[4/4] Comparing...")
        wbt_a = read_geotiff(wbt_acc)
        sg_a = read_geotiff(sg_acc) + 1.0  # align self-inclusion convention

        valid = np.isfinite(wbt_a) & np.isfinite(sg_a)
        # Exclude a 1-cell border: edge-handling policies differ between tools
        valid[0, :] = valid[-1, :] = valid[:, 0] = valid[:, -1] = False
        n = valid.sum()
        w = wbt_a[valid]
        s = sg_a[valid]

        log_w = np.log10(w)
        log_s = np.log10(s)
        r = np.corrcoef(log_w, log_s)[0, 1]
        rel = np.abs(s - w) / np.maximum(w, 1.0)
        within_10 = (rel < 0.10).mean() * 100
        within_50 = (rel < 0.50).mean() * 100

        print(f"\n  valid cells           : {n}")
        print(f"  Pearson r (log10 acc) : {r:.6f}")
        print(f"  median rel. error     : {np.median(rel) * 100:.3f}%")
        print(f"  cells within 10%      : {within_10:.2f}%")
        print(f"  cells within 50%      : {within_50:.2f}%")
        print(f"  max acc WBT / SurtGIS : {w.max():.0f} / {s.max():.0f}")

        ok = r > 0.95
        print(f"\n  {'PASS' if ok else 'FAIL'} (criterion: log-acc Pearson r > 0.95)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
