#!/usr/bin/env python3
"""Generate the environmental case study figure for the SurtGIS EMS paper.

Uses the Copernicus 30m DEM (Andes, Chile) to compute terrain derivatives,
geomorphon classification, and hydrological indices, then creates a
publication-quality multi-panel figure.
"""

import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource

# Try to read with rasterio, fallback to GDAL
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

import surtgis

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DEM_PATH = ROOT / "tests" / "fixtures" / "andes_chile_30m_utm.tif"
FIGURES = ROOT / "benchmarks" / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def read_dem(path):
    """Read DEM and return (data, cell_size, transform, crs_info)."""
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64)
            cell_size = src.res[0]
            transform = src.transform
            bounds = src.bounds
            return data, cell_size, transform, bounds
    elif HAS_GDAL:
        ds = gdal.Open(str(path))
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray().astype(np.float64)
        gt = ds.GetGeoTransform()
        cell_size = gt[1]
        # Create simple bounds
        nrows, ncols = data.shape
        bounds = type('Bounds', (), {
            'left': gt[0], 'right': gt[0] + ncols * gt[1],
            'bottom': gt[3] + nrows * gt[5], 'top': gt[3]
        })()
        return data, cell_size, gt, bounds
    else:
        raise ImportError("Need rasterio or GDAL to read GeoTIFF")


def main():
    print(f"Reading DEM: {DEM_PATH}")
    dem, cell_size, transform, bounds = read_dem(DEM_PATH)
    # Mask NoData (elevation=0 in high Andes is NoData)
    nodata_mask = dem < 100  # Andes region is 2800-6000m
    dem_masked = np.where(nodata_mask, np.nan, dem)

    print(f"  Shape: {dem.shape}, Cell size: {cell_size:.2f} m")
    print(f"  Elevation range: {np.nanmin(dem_masked):.0f} - {np.nanmax(dem_masked):.0f} m")
    print(f"  NoData cells: {np.sum(nodata_mask)} ({100*np.sum(nodata_mask)/dem.size:.1f}%)")

    # Coordinate extent in km (relative to bottom-left)
    extent_m = [0, dem.shape[1] * cell_size, 0, dem.shape[0] * cell_size]
    extent_km = [e / 1000 for e in extent_m]

    # ── Compute all derivatives with SurtGIS ──
    t0 = time.time()

    print("Computing slope...")
    slope_deg = surtgis.slope(dem, cell_size)

    print("Computing hillshade...")
    hs = surtgis.hillshade_compute(dem, cell_size, azimuth=315.0, altitude=45.0)

    print("Computing geomorphons...")
    geomorphons = surtgis.geomorphons_compute(dem, cell_size, flatness=1.0, radius=18)

    print("Computing mean curvature (Florinsky)...")
    mean_curv = surtgis.advanced_curvature(dem, cell_size, ctype="mean_h")

    print("Computing priority-flood fill...")
    filled = surtgis.priority_flood_fill(dem, cell_size)

    print("Computing TWI...")
    twi = surtgis.twi_compute(dem, cell_size)

    print("Computing HAND...")
    hand = surtgis.hand_compute(dem, cell_size)

    print("Computing stream network...")
    streams = surtgis.stream_network_compute(dem, cell_size, threshold=100.0)

    total_time = time.time() - t0
    print(f"\nTotal computation time: {total_time:.2f} s")

    # ── Create figure ──
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("white")

    # Apply nodata mask to all outputs
    def mask_nodata(arr):
        return np.where(nodata_mask, np.nan, arr)

    slope_deg = mask_nodata(slope_deg)
    hs = mask_nodata(hs)
    mean_curv = mask_nodata(mean_curv)
    geomorphons = mask_nodata(geomorphons).astype(float)
    twi = mask_nodata(twi)
    hand = mask_nodata(hand)

    # Panel (a): Hillshaded DEM with elevation
    ax = axes[0, 0]
    hs_valid = hs[np.isfinite(hs)]
    hs_norm = (hs - np.nanmin(hs_valid)) / (np.nanmax(hs_valid) - np.nanmin(hs_valid) + 1e-10)
    im = ax.imshow(dem_masked, cmap="terrain", extent=extent_km, origin="upper",
                   vmin=np.nanpercentile(dem_masked, 2), vmax=np.nanpercentile(dem_masked, 98))
    ax.imshow(np.where(nodata_mask, np.nan, hs_norm), cmap="gray", alpha=0.35,
              extent=extent_km, origin="upper")
    ax.set_title("(a) Elevation + hillshade", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Elevation (m)", fontsize=8)

    # Panel (b): Slope
    ax = axes[0, 1]
    slope_display = np.clip(slope_deg, 0, 70)
    im = ax.imshow(slope_display, cmap="YlOrRd", extent=extent_km, origin="upper",
                   vmin=0, vmax=70)
    ax.set_title("(b) Slope", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Slope (degrees)", fontsize=8)

    # Panel (c): Mean curvature (Florinsky)
    ax = axes[0, 2]
    # Clip curvature to ±p99 for visualization
    curv_abs = np.abs(mean_curv[np.isfinite(mean_curv)])
    if len(curv_abs) > 0:
        vmax_curv = np.percentile(curv_abs, 99)
    else:
        vmax_curv = 0.01
    im = ax.imshow(mean_curv, cmap="RdBu_r", extent=extent_km, origin="upper",
                   vmin=-vmax_curv, vmax=vmax_curv)
    ax.set_title("(c) Mean curvature $H$ (Florinsky)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("$H$ (m$^{-1}$)", fontsize=8)

    # Panel (d): Geomorphons
    ax = axes[1, 0]
    # Geomorphon classes: 1=flat, 2=peak, 3=ridge, 4=shoulder, 5=spur,
    # 6=slope, 7=hollow, 8=footslope, 9=valley, 10=pit
    geomorph_labels = ["", "Flat", "Peak", "Ridge", "Shoulder", "Spur",
                       "Slope", "Hollow", "Footslope", "Valley", "Pit"]
    geomorph_colors = ["#ffffff", "#CCCCCC", "#FF0000", "#FF8800", "#FFCC00",
                       "#FFFF00", "#88CC00", "#00CC88", "#0088FF", "#0000FF", "#880088"]
    cmap_geo = mcolors.ListedColormap(geomorph_colors[1:])
    bounds_geo = np.arange(0.5, 11.5, 1)
    norm_geo = mcolors.BoundaryNorm(bounds_geo, cmap_geo.N)
    im = ax.imshow(geomorphons, cmap=cmap_geo, norm=norm_geo,
                   extent=extent_km, origin="upper")
    ax.set_title("(d) Geomorphons", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, ticks=range(1, 11))
    cb.ax.set_yticklabels(geomorph_labels[1:], fontsize=6)

    # Panel (e): TWI
    ax = axes[1, 1]
    twi_valid = twi[np.isfinite(twi)]
    twi_p2, twi_p98 = np.percentile(twi_valid, [2, 98])
    twi_display = np.clip(twi, twi_p2, twi_p98)
    im = ax.imshow(twi_display, cmap="Blues", extent=extent_km, origin="upper",
                   vmin=twi_p2, vmax=twi_p98)
    ax.set_title("(e) Topographic Wetness Index", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("TWI", fontsize=8)

    # Panel (f): HAND with stream network overlay
    ax = axes[1, 2]
    hand_display = np.clip(hand, 0, np.nanpercentile(hand[np.isfinite(hand)], 95))
    im = ax.imshow(hand_display, cmap="YlGnBu_r", extent=extent_km, origin="upper")
    # Overlay streams
    stream_mask = np.ma.masked_where(streams < 0.5, streams)
    ax.imshow(stream_mask, cmap=mcolors.ListedColormap(["#0000FF"]),
              extent=extent_km, origin="upper", alpha=0.8, interpolation="nearest")
    ax.set_title("(f) HAND + drainage network", fontsize=10, fontweight="bold")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("HAND (m)", fontsize=8)

    fig.suptitle(
        f"Terrain characterization of Andes region (Copernicus 30 m DEM, "
        f"{dem.shape[0]}$\\times${dem.shape[1]} cells, "
        f"computed in {total_time:.1f} s with SurtGIS)",
        fontsize=11, fontweight="bold", y=0.98
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    out_pdf = FIGURES / "fig5_casestudy.pdf"
    out_png = FIGURES / "fig5_casestudy.png"
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"\nFigure saved to:\n  {out_pdf}\n  {out_png}")

    # Print summary stats for the paper
    print("\n=== SUMMARY STATS FOR PAPER ===")
    print(f"DEM: {dem.shape[0]} x {dem.shape[1]} cells, {cell_size:.2f} m resolution")
    print(f"Elevation: {np.nanmin(dem_masked):.0f} - {np.nanmax(dem_masked):.0f} m")
    print(f"Mean slope: {np.nanmean(slope_deg):.1f} degrees")
    print(f"Total computation time: {total_time:.2f} s")

    # Geomorphon class distribution
    print("\nGeomorphon class distribution:")
    total_cells = np.sum(np.isfinite(geomorphons))
    for i, label in enumerate(geomorph_labels[1:], 1):
        count = np.sum(geomorphons == i)
        pct = 100.0 * count / total_cells if total_cells > 0 else 0
        if pct > 0.1:
            print(f"  {label:12s}: {pct:5.1f}%")

    print(f"\nTWI range: {np.nanmin(twi):.1f} - {np.nanmax(twi):.1f}")
    print(f"HAND range: {np.nanmin(hand):.1f} - {np.nanmax(hand):.1f} m")
    print(f"Stream cells: {np.sum(streams > 0.5)}")


if __name__ == "__main__":
    main()
