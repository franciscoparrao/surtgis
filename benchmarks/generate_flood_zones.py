#!/usr/bin/env python3
"""Generate HAND-based flood zone delineation figure for the SurtGIS EMS paper.

Uses HAND (Height Above Nearest Drainage) to classify flood susceptibility
zones: <5m (high), 5-10m (moderate), 10-15m (low), >15m (minimal).

This extends the environmental case study (Section 4) to demonstrate
SurtGIS's applicability to operational flood hazard mapping.
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    raise ImportError("Need GDAL Python bindings")

import surtgis

ROOT = Path(__file__).resolve().parent.parent
DEM_PATH = ROOT / "tests" / "fixtures" / "andes_chile_30m_utm.tif"
FIGURES = ROOT / "benchmarks" / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def read_dem(path):
    ds = gdal.Open(str(path))
    data = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    cell_size = gt[1]
    return data, cell_size


def main():
    print("Reading DEM...")
    dem, cell_size = read_dem(DEM_PATH)
    nodata_mask = dem < 100
    dem_masked = np.where(nodata_mask, np.nan, dem)
    valid_mask = ~nodata_mask
    cell_area_m2 = cell_size ** 2
    cell_area_km2 = cell_area_m2 / 1e6

    print(f"  Shape: {dem.shape}, Cell size: {cell_size:.2f} m")
    print(f"  Valid cells: {np.sum(valid_mask)}")

    # Compute HAND and streams
    print("Computing HAND...")
    hand = surtgis.hand_compute(dem, cell_size)
    hand_masked = np.where(nodata_mask, np.nan, hand)

    print("Computing stream network...")
    streams = surtgis.stream_network_compute(dem, cell_size, threshold=100.0)

    print("Computing hillshade...")
    hs = surtgis.hillshade_compute(dem, cell_size, azimuth=315.0, altitude=45.0)
    hs_valid = hs[np.isfinite(hs) & ~nodata_mask]
    hs_norm = (hs - np.nanmin(hs_valid)) / (np.nanmax(hs_valid) - np.nanmin(hs_valid) + 1e-10)
    hs_norm = np.where(nodata_mask, np.nan, hs_norm)

    # Define flood zones based on HAND thresholds
    thresholds = [5, 10, 15, 25, 50]
    zone_labels = [
        f"< {thresholds[0]} m (high risk)",
        f"{thresholds[0]}–{thresholds[1]} m (moderate)",
        f"{thresholds[1]}–{thresholds[2]} m (low)",
        f"{thresholds[2]}–{thresholds[3]} m (minimal)",
        f"> {thresholds[3]} m (negligible)",
    ]
    zone_colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#1a9850"]

    # Classify
    flood_zones = np.full(dem.shape, np.nan)
    for i in range(len(thresholds)):
        if i == 0:
            mask = valid_mask & (hand_masked < thresholds[0])
        else:
            mask = valid_mask & (hand_masked >= thresholds[i-1]) & (hand_masked < thresholds[i])
        flood_zones[mask] = i + 1
    # Last zone: >= last threshold
    flood_zones[valid_mask & (hand_masked >= thresholds[-1])] = len(thresholds) + 1

    # Actually, simplify to 5 zones
    flood_zones = np.full(dem.shape, np.nan)
    flood_zones[valid_mask & (hand_masked < 5)] = 1
    flood_zones[valid_mask & (hand_masked >= 5) & (hand_masked < 10)] = 2
    flood_zones[valid_mask & (hand_masked >= 10) & (hand_masked < 15)] = 3
    flood_zones[valid_mask & (hand_masked >= 15) & (hand_masked < 25)] = 4
    flood_zones[valid_mask & (hand_masked >= 25)] = 5

    # Statistics
    total_valid = np.sum(valid_mask)
    total_area_km2 = total_valid * cell_area_km2

    print("\n=== FLOOD ZONE STATISTICS ===")
    print(f"Total study area: {total_area_km2:.2f} km²")
    print(f"Cell size: {cell_size:.2f} m, Cell area: {cell_area_m2:.1f} m²")
    print()

    for i, (label, color) in enumerate(zip(zone_labels, zone_colors)):
        count = np.sum(flood_zones == (i + 1))
        area = count * cell_area_km2
        pct = 100.0 * count / total_valid
        print(f"  Zone {i+1}: {label}")
        print(f"    Cells: {count:,}, Area: {area:.2f} km², {pct:.1f}%")

    # Extent in km
    extent_km = [0, dem.shape[1] * cell_size / 1000,
                 0, dem.shape[0] * cell_size / 1000]

    # ── Figure: 2 panels ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # Left panel: HAND continuous with streams
    ax = axes[0]
    hand_display = np.clip(hand_masked, 0, np.nanpercentile(hand_masked[valid_mask], 95))
    ax.imshow(hs_norm, cmap="gray", extent=extent_km, origin="upper", alpha=0.5)
    im = ax.imshow(hand_display, cmap="YlGnBu_r", extent=extent_km, origin="upper",
                   alpha=0.7)
    # Overlay streams
    stream_overlay = np.ma.masked_where(streams < 0.5, streams)
    ax.imshow(stream_overlay, cmap=mcolors.ListedColormap(["#0066CC"]),
              extent=extent_km, origin="upper", alpha=0.9, interpolation="nearest")
    ax.set_title("(a) HAND + drainage network", fontsize=11, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("HAND (m)", fontsize=9)

    # Right panel: Flood zones
    ax = axes[1]
    cmap_zones = mcolors.ListedColormap(zone_colors)
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = mcolors.BoundaryNorm(bounds, cmap_zones.N)

    ax.imshow(hs_norm, cmap="gray", extent=extent_km, origin="upper", alpha=0.4)
    im = ax.imshow(flood_zones, cmap=cmap_zones, norm=norm,
                   extent=extent_km, origin="upper", alpha=0.75)
    # Overlay streams
    ax.imshow(stream_overlay, cmap=mcolors.ListedColormap(["#000066"]),
              extent=extent_km, origin="upper", alpha=0.8, interpolation="nearest")

    ax.set_title("(b) HAND-based flood susceptibility zones", fontsize=11, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")

    # Legend
    legend_patches = []
    for i, (label, color) in enumerate(zip(zone_labels, zone_colors)):
        count = np.sum(flood_zones == (i + 1))
        pct = 100.0 * count / total_valid
        legend_patches.append(
            mpatches.Patch(facecolor=color, edgecolor="gray", linewidth=0.5,
                          label=f"{label} ({pct:.1f}%)")
        )
    ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
              framealpha=0.9, title="HAND zone", title_fontsize=8)

    fig.suptitle(
        f"Flood susceptibility mapping using HAND "
        f"(Andes 30 m DEM, {dem.shape[0]}$\\times${dem.shape[1]} cells)",
        fontsize=12, fontweight="bold", y=0.98
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    out_pdf = FIGURES / "fig_flood_zones.pdf"
    out_png = FIGURES / "fig_flood_zones.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"\nFigure saved to:\n  {out_pdf}\n  {out_png}")

    # Print data for paper text
    print("\n=== DATA FOR PAPER ===")
    for i, label in enumerate(zone_labels):
        count = np.sum(flood_zones == (i + 1))
        area = count * cell_area_km2
        pct = 100.0 * count / total_valid
        print(f"Zone {i+1} ({label}): {area:.2f} km² ({pct:.1f}%)")

    high_risk = np.sum(flood_zones == 1) * cell_area_km2
    moderate = np.sum(flood_zones == 2) * cell_area_km2
    combined = high_risk + moderate
    combined_pct = 100.0 * (np.sum(flood_zones == 1) + np.sum(flood_zones == 2)) / total_valid
    print(f"\nHigh+moderate risk (HAND<10m): {combined:.2f} km² ({combined_pct:.1f}%)")
    print(f"Mean HAND: {np.nanmean(hand_masked):.1f} m")
    print(f"Median HAND: {np.nanmedian(hand_masked[valid_mask]):.1f} m")
    print(f"Stream cells: {np.sum(streams > 0.5):,}")


if __name__ == "__main__":
    main()
