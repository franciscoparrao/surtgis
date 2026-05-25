#!/usr/bin/env python3
"""Plot Smugglers Notch elevation vs χ + R² per basin + ksn map.

Inputs: chi.tif, filled.tif, basins.tif, streams.tif, ksn.tif in $WORK.
Outputs: validation_plot.{pdf,png} and validation_metrics.csv in $OUT.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: plot_validation.py <work_dir> <out_dir>", file=sys.stderr)
        return 1
    work = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    chi = rasterio.open(work / "chi.tif").read(1)
    dem = rasterio.open(work / "filled.tif").read(1)
    basins = rasterio.open(work / "basins.tif").read(1)
    streams = rasterio.open(work / "streams.tif").read(1)
    ksn = rasterio.open(work / "ksn.tif").read(1)

    mask = (streams == 1) & np.isfinite(chi) & np.isfinite(dem)
    all_basins = np.unique(basins[basins > 0])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    results = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_basins), 1)))
    basin_color = {}
    for bid, col in zip(all_basins, colors):
        basin_color[int(bid)] = col
        bmask = mask & (basins == bid)
        if bmask.sum() < 50:
            continue
        x = chi[bmask]
        z = dem[bmask]
        slope, intercept = np.polyfit(x, z, 1)
        pred = slope * x + intercept
        ss_res = ((z - pred) ** 2).sum()
        ss_tot = ((z - z.mean()) ** 2).sum()
        r2 = float(1.0 - ss_res / ss_tot)
        results.append({
            "basin": int(bid),
            "n": int(bmask.sum()),
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": r2,
        })
        ax.scatter(x, z, s=2, alpha=0.4, color=col,
                   label=f"Basin {int(bid)} (n={bmask.sum()}, R²={r2:.3f})")
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=col, lw=1.5)

    ax.set_xlabel("χ (m)")
    ax.set_ylabel("Elevation (m)")
    ax.set_title("Smugglers Notch — elevation vs χ per basin\n"
                 "(Perron & Royden 2013 linearisation test)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    basin_rgba = np.zeros((*basins.shape, 4), dtype=float)
    basin_rgba[..., 3] = 0.0  # transparent default
    for bid, col in basin_color.items():
        m = basins == bid
        basin_rgba[m] = col
        basin_rgba[m, 3] = 0.85
    ax.imshow(basin_rgba)
    # Stream skeleton overlay for context.
    stream_overlay = np.where(streams == 1, 1.0, np.nan)
    ax.imshow(stream_overlay, cmap="gray_r", vmin=0, vmax=1, alpha=0.7)
    ax.set_title(f"Basin delineation ({len(results)} basins)\nstreams overlaid")
    ax.axis("off")

    ax = axes[2]
    ksn_disp = np.where(np.isfinite(ksn), ksn, np.nan)
    im = ax.imshow(ksn_disp, cmap="viridis", vmin=0,
                   vmax=float(np.nanpercentile(ksn_disp, 95)))
    ax.set_title("ksn map (Wobus 2006)")
    plt.colorbar(im, ax=ax, label="ksn", shrink=0.7)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(out / "validation_plot.pdf", dpi=200, bbox_inches="tight")
    plt.savefig(out / "validation_plot.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out}/validation_plot.{{pdf,png}}")

    with open(out / "validation_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["basin_id", "n_cells", "slope_z_vs_chi", "intercept", "r2"])
        for r in results:
            w.writerow([r["basin"], r["n"], f"{r['slope']:.6f}",
                        f"{r['intercept']:.4f}", f"{r['r2']:.4f}"])

    print()
    for r in results:
        print(f"  Basin {r['basin']}: n={r['n']}, R²={r['r2']:.4f}, "
              f"slope={r['slope']:.4f} m/m_χ")
    if results:
        all_r2 = [r["r2"] for r in results]
        print(f"\nN basins: {len(results)}")
        print(f"Median R²: {np.median(all_r2):.4f}  "
              f"min: {min(all_r2):.4f}  max: {max(all_r2):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
