#!/usr/bin/env python3
"""Generate graphical abstract for the SurtGIS EMS paper.

Layout (landscape):
  ┌─────────────────────────────────────────────────────────┐
  │  [DEM input]  →  [SurtGIS core]  →  [Output panels]    │
  │                                                          │
  │     Andes          100+ algorithms     Slope  Geomorph.  │
  │     hillshade      Rust / Rayon        HAND   Curvature  │
  │                                                          │
  │  ┌──────────────────────────────────────────────────┐    │
  │  │  Desktop    Browser (WASM)    Python  │ Speedups │    │
  │  └──────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.gridspec as gridspec

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

import surtgis

ROOT = Path(__file__).resolve().parent.parent
DEM_PATH = ROOT / "tests" / "fixtures" / "andes_chile_30m_utm.tif"
FIGURES = ROOT / "benchmarks" / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def read_dem(path):
    if HAS_GDAL:
        ds = gdal.Open(str(path))
        data = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        gt = ds.GetGeoTransform()
        return data, gt[1]
    else:
        raise ImportError("Need GDAL to read GeoTIFF")


def main():
    dem, cell_size = read_dem(DEM_PATH)
    nodata_mask = dem < 100
    dem_masked = np.where(nodata_mask, np.nan, dem)

    # Compute outputs
    slope = np.where(nodata_mask, np.nan, surtgis.slope(dem, cell_size))
    hs = np.where(nodata_mask, np.nan,
                  surtgis.hillshade_compute(dem, cell_size, azimuth=315.0, altitude=45.0))
    geomorphons = np.where(nodata_mask, np.nan,
                           surtgis.geomorphons_compute(dem, cell_size, flatness=1.0, radius=18).astype(float))
    mean_curv = np.where(nodata_mask, np.nan,
                         surtgis.advanced_curvature(dem, cell_size, ctype="mean_h"))
    hand = np.where(nodata_mask, np.nan, surtgis.hand_compute(dem, cell_size))
    streams = surtgis.stream_network_compute(dem, cell_size, threshold=100.0)

    # Normalize hillshade
    hs_valid = hs[np.isfinite(hs)]
    hs_norm = (hs - np.nanmin(hs_valid)) / (np.nanmax(hs_valid) - np.nanmin(hs_valid) + 1e-10)
    hs_norm = np.where(nodata_mask, np.nan, hs_norm)

    # ── Figure layout ──
    fig = plt.figure(figsize=(16, 9), facecolor="white")

    # Main grid: top row (maps), bottom row (targets + speedup)
    gs_main = fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.12,
                               left=0.02, right=0.98, top=0.88, bottom=0.04)

    # Top: input → core → outputs
    gs_top = gs_main[0].subgridspec(1, 7, width_ratios=[2.2, 0.3, 2.2, 0.3, 2.2, 2.2, 2.2],
                                     wspace=0.08)

    # ── Title ──
    fig.text(0.5, 0.95, "SurtGIS: High-Performance Geospatial Analysis in Rust",
             fontsize=20, fontweight="bold", ha="center", va="center",
             color="#1a1a2e")
    fig.text(0.5, 0.905, "100+ terrain, hydrology & remote sensing algorithms  ·  Native + WebAssembly + Python from one codebase",
             fontsize=11, ha="center", va="center", color="#555555")

    # ── Panel 1: Input DEM ──
    ax_dem = fig.add_subplot(gs_top[0])
    ax_dem.imshow(dem_masked, cmap="terrain", origin="upper",
                  vmin=np.nanpercentile(dem_masked, 2), vmax=np.nanpercentile(dem_masked, 98))
    ax_dem.imshow(hs_norm, cmap="gray", alpha=0.4, origin="upper")
    ax_dem.set_xticks([])
    ax_dem.set_yticks([])
    ax_dem.set_title("Input DEM", fontsize=11, fontweight="bold", pad=6)
    # Label
    ax_dem.text(0.5, -0.06, "Copernicus 30 m\nAndes, Chile",
                transform=ax_dem.transAxes, ha="center", fontsize=8, color="#555")

    # ── Arrow 1 ──
    ax_arr1 = fig.add_subplot(gs_top[1])
    ax_arr1.axis("off")
    ax_arr1.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                     arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.3",
                                     color="#2196F3", lw=3),
                     xycoords="axes fraction")
    ax_arr1.text(0.5, 0.7, "SurtGIS", fontsize=10, fontweight="bold",
                 ha="center", va="center", color="#2196F3",
                 transform=ax_arr1.transAxes)

    # ── Panel 2: Slope ──
    ax_slope = fig.add_subplot(gs_top[2])
    ax_slope.imshow(np.clip(slope, 0, 70), cmap="YlOrRd", origin="upper", vmin=0, vmax=70)
    ax_slope.set_xticks([])
    ax_slope.set_yticks([])
    ax_slope.set_title("Slope", fontsize=11, fontweight="bold", pad=6)
    ax_slope.text(0.5, -0.06, "1.8× faster than GDAL\nat 400M cells",
                  transform=ax_slope.transAxes, ha="center", fontsize=8, color="#555")

    # ── Arrow 2 ──
    ax_arr2 = fig.add_subplot(gs_top[3])
    ax_arr2.axis("off")

    # ── Panel 3: Mean curvature ──
    ax_curv = fig.add_subplot(gs_top[4])
    curv_abs = np.abs(mean_curv[np.isfinite(mean_curv)])
    vmax_c = np.percentile(curv_abs, 99) if len(curv_abs) > 0 else 0.01
    ax_curv.imshow(mean_curv, cmap="RdBu_r", origin="upper", vmin=-vmax_c, vmax=vmax_c)
    ax_curv.set_xticks([])
    ax_curv.set_yticks([])
    ax_curv.set_title("Florinsky curvature", fontsize=11, fontweight="bold", pad=6)
    ax_curv.text(0.5, -0.06, "14 curvature types\nopen-source first",
                 transform=ax_curv.transAxes, ha="center", fontsize=8, color="#555")

    # ── Panel 4: Geomorphons ──
    import matplotlib.colors as mcolors
    ax_geo = fig.add_subplot(gs_top[5])
    geo_colors = ["#CCCCCC", "#FF0000", "#FF8800", "#FFCC00", "#FFFF00",
                  "#88CC00", "#00CC88", "#0088FF", "#0000FF", "#880088"]
    cmap_geo = mcolors.ListedColormap(geo_colors)
    bounds_geo = np.arange(0.5, 11.5, 1)
    norm_geo = mcolors.BoundaryNorm(bounds_geo, cmap_geo.N)
    ax_geo.imshow(geomorphons, cmap=cmap_geo, norm=norm_geo, origin="upper")
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])
    ax_geo.set_title("Geomorphons", fontsize=11, fontweight="bold", pad=6)
    ax_geo.text(0.5, -0.06, "Automated landform\nclassification",
                transform=ax_geo.transAxes, ha="center", fontsize=8, color="#555")

    # ── Panel 5: HAND + streams ──
    ax_hand = fig.add_subplot(gs_top[6])
    hand_display = np.clip(hand, 0, np.nanpercentile(hand[np.isfinite(hand)], 95))
    ax_hand.imshow(hand_display, cmap="YlGnBu_r", origin="upper")
    stream_mask = np.ma.masked_where(streams < 0.5, streams)
    ax_hand.imshow(stream_mask, cmap=mcolors.ListedColormap(["#0000FF"]),
                   origin="upper", alpha=0.8, interpolation="nearest")
    ax_hand.set_xticks([])
    ax_hand.set_yticks([])
    ax_hand.set_title("HAND + drainage", fontsize=11, fontweight="bold", pad=6)
    ax_hand.text(0.5, -0.06, "Integrated hydrology\npipeline",
                 transform=ax_hand.transAxes, ha="center", fontsize=8, color="#555")

    # ── Bottom row: targets + speedup ──
    gs_bot = gs_main[1].subgridspec(1, 2, width_ratios=[3, 2], wspace=0.15)

    # Left: deployment targets
    ax_targets = fig.add_subplot(gs_bot[0])
    ax_targets.axis("off")

    targets = [
        ("Desktop / Server", "Native Rust + Rayon\n12-core parallel", "#2196F3", "crates.io"),
        ("Web Browser", "WebAssembly via wasm-pack\nClient-side processing", "#4CAF50", "npm"),
        ("Python", "PyO3 bindings + NumPy\npip install surtgis", "#FF9800", "PyPI"),
    ]

    box_w, box_h = 0.28, 0.75
    gap = 0.05
    start_x = 0.05

    for i, (title, desc, color, registry) in enumerate(targets):
        x = start_x + i * (box_w + gap)
        # Box
        rect = FancyBboxPatch((x, 0.1), box_w, box_h,
                               boxstyle="round,pad=0.02",
                               facecolor=color, alpha=0.12,
                               edgecolor=color, linewidth=2,
                               transform=ax_targets.transAxes)
        ax_targets.add_patch(rect)
        # Title
        ax_targets.text(x + box_w / 2, 0.72, title,
                       transform=ax_targets.transAxes, ha="center", va="center",
                       fontsize=11, fontweight="bold", color=color)
        # Description
        ax_targets.text(x + box_w / 2, 0.45, desc,
                       transform=ax_targets.transAxes, ha="center", va="center",
                       fontsize=8, color="#333")
        # Registry badge
        ax_targets.text(x + box_w / 2, 0.2, registry,
                       transform=ax_targets.transAxes, ha="center", va="center",
                       fontsize=9, fontweight="bold", color=color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.15,
                                 edgecolor=color, linewidth=1))

    ax_targets.text(0.5, -0.05, "One codebase → three deployment targets",
                   transform=ax_targets.transAxes, ha="center", fontsize=10,
                   fontstyle="italic", color="#555")

    # Right: speedup bars
    ax_speed = fig.add_subplot(gs_bot[1])

    algorithms = ["Slope", "Aspect", "Fill", "Flow acc."]
    # Speedups at 10K² (100M cells), median of 10 runs (paper Tables 1-2)
    # WBT slope excluded: uses different derivative method (RMSE=6.53° vs Horn)
    gdal_speedup = [1.8, 2.0, None, None]
    grass_speedup = [4.8, 4.8, None, 23.1]
    wbt_speedup = [None, 7.9, 1.6, 7.7]

    y = np.arange(len(algorithms))
    bar_h = 0.22

    for i, alg in enumerate(algorithms):
        bars_data = []
        if gdal_speedup[i]:
            bars_data.append(("GDAL", gdal_speedup[i], "#4CAF50"))
        if grass_speedup[i]:
            bars_data.append(("GRASS", grass_speedup[i], "#FF9800"))
        if wbt_speedup[i]:
            bars_data.append(("WBT", wbt_speedup[i], "#F44336"))

        for j, (tool, val, color) in enumerate(bars_data):
            offset = (j - len(bars_data) / 2 + 0.5) * bar_h
            bar = ax_speed.barh(i + offset, val, bar_h * 0.9, color=color, alpha=0.8,
                                edgecolor="white", linewidth=0.5)
            ax_speed.text(val + 0.3, i + offset, f"{val}×",
                         va="center", ha="left", fontsize=8, fontweight="bold", color=color)

    ax_speed.set_yticks(y)
    ax_speed.set_yticklabels(algorithms, fontsize=10)
    ax_speed.set_xlabel("Speedup over competitor", fontsize=9)
    ax_speed.set_xlim(0, 26)
    ax_speed.invert_yaxis()
    ax_speed.grid(True, axis="x", alpha=0.2)
    ax_speed.set_title("Performance (full GeoTIFF pipeline)", fontsize=11, fontweight="bold")

    # Legend for speedup
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", alpha=0.8, label="vs GDAL"),
        mpatches.Patch(facecolor="#FF9800", alpha=0.8, label="vs GRASS"),
        mpatches.Patch(facecolor="#F44336", alpha=0.8, label="vs WBT"),
    ]
    ax_speed.legend(handles=legend_elements, loc="lower right", fontsize=8,
                    framealpha=0.9)

    # Save
    out_pdf = FIGURES / "graphical_abstract.pdf"
    out_png = FIGURES / "graphical_abstract.png"
    fig.savefig(out_pdf, dpi=300, facecolor="white")
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Graphical abstract saved to:\n  {out_pdf}\n  {out_png}")


if __name__ == "__main__":
    main()
