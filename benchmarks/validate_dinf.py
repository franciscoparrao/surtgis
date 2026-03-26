#!/usr/bin/env python3
"""C1: Validate SurtGIS D-infinity flow direction against TauDEM reference.

Both implementations follow Tarboton (1997): 8 triangular facets, continuous
flow angles in radians [0, 2π), counter-clockwise from East.

Steps:
1. Run TauDEM PitRemove + DinfFlowDir on Andes UTM DEM
2. Run SurtGIS flow_direction_dinf via Python bindings
3. Pixel-by-pixel angular comparison (circular RMSE)
"""

import time
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
ANDES_UTM = BASE / "tests" / "fixtures" / "andes_chile_30m_utm.tif"
TAUDEM_DIR = Path("/tmp/TauDEM/build/src")


def read_geotiff(path):
    """Read GeoTIFF as numpy array using GDAL."""
    from osgeo import gdal
    ds = gdal.Open(str(path))
    if ds is None:
        raise FileNotFoundError(f"Cannot open {path}")
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    ds = None
    return arr


def angular_diff(a, b):
    """Compute angular difference in radians, handling circular wrapping."""
    diff = a - b
    # Wrap to [-π, π]
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    return diff


def run_taudem(dem_path, tmpdir):
    """Run TauDEM PitRemove + DinfFlowDir."""
    filled = Path(tmpdir) / "dem_filled.tif"
    angles = Path(tmpdir) / "taudem_ang.tif"
    slopes = Path(tmpdir) / "taudem_slp.tif"

    pitremove = TAUDEM_DIR / "pitremove"
    dinfflowdir = TAUDEM_DIR / "dinfflowdir"

    # PitRemove
    print("  Running TauDEM PitRemove...")
    t0 = time.perf_counter()
    result = subprocess.run(
        ["mpiexec", "--allow-run-as-root", "-n", "1",
         str(pitremove), "-z", str(dem_path), "-fel", str(filled)],
        capture_output=True, text=True, timeout=120
    )
    t1 = time.perf_counter()
    if result.returncode != 0:
        # Try without --allow-run-as-root
        result = subprocess.run(
            ["mpiexec", "-n", "1",
             str(pitremove), "-z", str(dem_path), "-fel", str(filled)],
            capture_output=True, text=True, timeout=120
        )
    if result.returncode != 0:
        print(f"  PitRemove FAILED: {result.stderr[:300]}")
        return None, None, None
    print(f"  PitRemove done in {t1-t0:.1f}s")

    # DinfFlowDir
    print("  Running TauDEM DinfFlowDir...")
    t0 = time.perf_counter()
    result = subprocess.run(
        ["mpiexec", "--allow-run-as-root", "-n", "1",
         str(dinfflowdir), "-fel", str(filled),
         "-ang", str(angles), "-slp", str(slopes)],
        capture_output=True, text=True, timeout=120
    )
    t1 = time.perf_counter()
    if result.returncode != 0:
        result = subprocess.run(
            ["mpiexec", "-n", "1",
             str(dinfflowdir), "-fel", str(filled),
             "-ang", str(angles), "-slp", str(slopes)],
            capture_output=True, text=True, timeout=120
        )
    if result.returncode != 0:
        print(f"  DinfFlowDir FAILED: {result.stderr[:300]}")
        return None, None, None
    print(f"  DinfFlowDir done in {t1-t0:.1f}s")

    ang_arr = read_geotiff(angles)
    slp_arr = read_geotiff(slopes)
    filled_arr = read_geotiff(filled)

    return ang_arr, slp_arr, filled_arr


def run_surtgis_dinf(filled_dem_arr, cell_size=28.4):
    """Run SurtGIS D-infinity flow direction on pre-filled DEM."""
    import surtgis

    print("  Running SurtGIS flow_direction_dinf on filled DEM...")
    t0 = time.perf_counter()
    # SurtGIS returns angles in radians [0, 2π), -1 for pits
    angles = surtgis.flow_direction_dinf(filled_dem_arr, cell_size)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.3f}s")

    return angles


def diagnose_convention(taudem_ang, surtgis_ang):
    """Diagnose angle convention difference between SurtGIS and TauDEM."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Testing angle convention transformations")
    print("=" * 60)

    valid = (np.isfinite(taudem_ang) & np.isfinite(surtgis_ang) &
             (taudem_ang >= 0) & (surtgis_ang >= 0) &
             (taudem_ang < 2 * np.pi + 0.1) & (surtgis_ang < 2 * np.pi + 0.1))

    t = taudem_ang[valid]
    s = surtgis_ang[valid]

    transformations = {
        "Direct (SurtGIS = TauDEM)": s,
        "Mirror Y: 2π - SurtGIS": (2 * np.pi - s) % (2 * np.pi),
        "Mirror X: π - SurtGIS": (np.pi - s) % (2 * np.pi),
        "Rotate 180°: SurtGIS + π": (s + np.pi) % (2 * np.pi),
        "CW↔CCW swap: 2π - SurtGIS": (2 * np.pi - s) % (2 * np.pi),
        "N↔S flip (mirror rows)": (2 * np.pi - s) % (2 * np.pi),
    }

    print(f"\n{'Transformation':<40} {'RMSE (°)':<12} {'Within 1°':<12} {'Within 5°':<12}")
    print("-" * 80)

    best_rmse = 999.0
    best_name = ""

    for name, s_transformed in transformations.items():
        diff = angular_diff(s_transformed, t)
        abs_diff_deg = np.degrees(np.abs(diff))
        rmse_deg = np.degrees(np.sqrt(np.mean(diff**2)))
        pct_1 = 100 * np.sum(abs_diff_deg < 1.0) / len(t)
        pct_5 = 100 * np.sum(abs_diff_deg < 5.0) / len(t)
        print(f"{name:<40} {rmse_deg:<12.4f} {pct_1:<12.1f}% {pct_5:<12.1f}%")
        if rmse_deg < best_rmse:
            best_rmse = rmse_deg
            best_name = name

    print(f"\nBest: {best_name} (RMSE = {best_rmse:.4f}°)")

    # Also check specific quadrant patterns
    print("\nQuadrant analysis (TauDEM angle ranges → SurtGIS mean offset):")
    for lo, hi, label in [(0, 0.5, "E (~0)"), (1.3, 1.8, "N (~π/2)"),
                          (2.9, 3.4, "W (~π)"), (4.5, 5.0, "S (~3π/2)")]:
        mask = (t >= lo) & (t < hi)
        if np.sum(mask) > 100:
            diff_deg = np.degrees(angular_diff(s[mask], t[mask]))
            print(f"  {label}: N={np.sum(mask):,}, mean Δ={np.mean(diff_deg):+.1f}°, "
                  f"median Δ={np.median(diff_deg):+.1f}°")


def compare_angles(taudem_ang, surtgis_ang, filled_dem=None):
    """Pixel-by-pixel comparison of flow direction angles."""
    print("\n" + "=" * 60)
    print("Pixel-by-pixel comparison: SurtGIS vs TauDEM D-infinity")
    print("=" * 60)

    # Valid mask: exclude NaN, nodata, and pit cells
    # TauDEM uses nodata for edges, SurtGIS uses -1 for pits
    valid = (np.isfinite(taudem_ang) & np.isfinite(surtgis_ang) &
             (taudem_ang >= 0) & (surtgis_ang >= 0) &
             (taudem_ang < 2 * np.pi + 0.1) & (surtgis_ang < 2 * np.pi + 0.1))

    n_total = taudem_ang.size
    n_valid = np.sum(valid)
    print(f"Total cells: {n_total:,}")
    print(f"Valid cells: {n_valid:,} ({100*n_valid/n_total:.1f}%)")

    t_ang = taudem_ang[valid]
    s_ang = surtgis_ang[valid]

    # Angular difference (circular)
    diff = angular_diff(s_ang, t_ang)
    abs_diff = np.abs(diff)
    abs_diff_deg = np.degrees(abs_diff)

    # Circular RMSE
    rmse_rad = np.sqrt(np.mean(diff**2))
    rmse_deg = np.degrees(rmse_rad)

    # MAE
    mae_rad = np.mean(abs_diff)
    mae_deg = np.degrees(mae_rad)

    # Circular correlation (using cos similarity)
    cos_diff = np.cos(diff)
    mean_cos = np.mean(cos_diff)

    # Percentile statistics
    pct_1deg = 100 * np.sum(abs_diff_deg < 1.0) / n_valid
    pct_5deg = 100 * np.sum(abs_diff_deg < 5.0) / n_valid
    pct_10deg = 100 * np.sum(abs_diff_deg < 10.0) / n_valid
    pct_45deg = 100 * np.sum(abs_diff_deg < 45.0) / n_valid

    print(f"\nAngular difference statistics:")
    print(f"  RMSE:          {rmse_rad:.6f} rad ({rmse_deg:.4f}°)")
    print(f"  MAE:           {mae_rad:.6f} rad ({mae_deg:.4f}°)")
    print(f"  Mean cos(Δ):   {mean_cos:.6f} (1.0 = perfect)")
    print(f"  Median diff:   {np.degrees(np.median(abs_diff)):.4f}°")
    print(f"  95th pctl:     {np.degrees(np.percentile(abs_diff, 95)):.4f}°")
    print(f"  Max diff:      {np.degrees(np.max(abs_diff)):.2f}°")

    print(f"\nAgreement thresholds:")
    print(f"  Within 1°:     {pct_1deg:.1f}%")
    print(f"  Within 5°:     {pct_5deg:.1f}%")
    print(f"  Within 10°:    {pct_10deg:.1f}%")
    print(f"  Within 45°:    {pct_45deg:.1f}%")

    # Range comparison
    print(f"\nAngle ranges:")
    print(f"  TauDEM:  [{np.min(t_ang):.4f}, {np.max(t_ang):.4f}] rad")
    print(f"  SurtGIS: [{np.min(s_ang):.4f}, {np.max(s_ang):.4f}] rad")

    # Slope-stratified analysis if DEM provided
    if filled_dem is not None:
        print(f"\n--- Slope-stratified analysis ---")
        # Compute slope from the filled DEM using numpy (simple central diff)
        rows, cols = filled_dem.shape
        # Gradient magnitude (Horn method approximation)
        slope_mag = np.zeros_like(filled_dem)
        cs = 28.4
        for r in range(1, rows-1):
            for c in range(1, cols-1):
                dzdx = (filled_dem[r, c+1] - filled_dem[r, c-1]) / (2 * cs)
                dzdy = (filled_dem[r-1, c] - filled_dem[r+1, c]) / (2 * cs)
                slope_mag[r, c] = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

        slope_valid = slope_mag[valid]
        thresholds = [(0, 2, "0-2°"), (2, 5, "2-5°"), (5, 15, "5-15°"),
                      (15, 30, "15-30°"), (30, 90, "30-90°")]
        print(f"{'Slope range':<12} {'N cells':<12} {'RMSE (°)':<12} {'Within 1°':<12} {'Within 5°':<12}")
        print("-" * 60)
        for lo, hi, label in thresholds:
            mask = (slope_valid >= lo) & (slope_valid < hi)
            n = np.sum(mask)
            if n < 10:
                continue
            d = angular_diff(s_ang[mask], t_ang[mask])
            ad = np.degrees(np.abs(d))
            rmse = np.degrees(np.sqrt(np.mean(d**2)))
            p1 = 100 * np.sum(ad < 1.0) / n
            p5 = 100 * np.sum(ad < 5.0) / n
            print(f"{label:<12} {n:<12,} {rmse:<12.4f} {p1:<12.1f}% {p5:<12.1f}%")

        # Steep terrain only (>5°)
        steep_mask = slope_valid >= 5.0
        n_steep = np.sum(steep_mask)
        if n_steep > 0:
            d_steep = angular_diff(s_ang[steep_mask], t_ang[steep_mask])
            ad_steep = np.degrees(np.abs(d_steep))
            rmse_steep = np.degrees(np.sqrt(np.mean(d_steep**2)))
            p1_steep = 100 * np.sum(ad_steep < 1.0) / n_steep
            p5_steep = 100 * np.sum(ad_steep < 5.0) / n_steep
            print(f"\nSteep terrain (slope>5°): N={n_steep:,}")
            print(f"  RMSE = {rmse_steep:.4f}°, {p1_steep:.1f}% within 1°, {p5_steep:.1f}% within 5°")

    return {
        'n_valid': n_valid,
        'rmse_rad': rmse_rad,
        'rmse_deg': rmse_deg,
        'mae_deg': mae_deg,
        'mean_cos': mean_cos,
        'pct_1deg': pct_1deg,
        'pct_5deg': pct_5deg,
        'pct_10deg': pct_10deg,
    }


if __name__ == "__main__":
    print("C1: D-infinity validation — SurtGIS vs TauDEM\n")

    if not ANDES_UTM.exists():
        print(f"ERROR: DEM not found at {ANDES_UTM}")
        sys.exit(1)

    if not (TAUDEM_DIR / "dinfflowdir").exists():
        print(f"ERROR: TauDEM not built at {TAUDEM_DIR}")
        sys.exit(1)

    # Run TauDEM
    print("[1/2] TauDEM D-infinity:")
    with tempfile.TemporaryDirectory() as tmpdir:
        taudem_ang, taudem_slp, filled_dem = run_taudem(ANDES_UTM, tmpdir)

        if taudem_ang is None:
            print("TauDEM failed. Exiting.")
            sys.exit(1)

        # Run SurtGIS on the SAME filled DEM (fair comparison)
        print("\n[2/2] SurtGIS D-infinity (on TauDEM-filled DEM):")
        surtgis_ang = run_surtgis_dinf(filled_dem, cell_size=28.4)

        # Diagnose convention
        diagnose_convention(taudem_ang, surtgis_ang)

        # Compare
        results = compare_angles(taudem_ang, surtgis_ang, filled_dem)

    print("\n" + "=" * 60)
    print("SUMMARY for paper:")
    print(f"  RMSE = {results['rmse_deg']:.4f}° ({results['rmse_rad']:.6f} rad)")
    print(f"  {results['pct_1deg']:.1f}% within 1°, {results['pct_5deg']:.1f}% within 5°")
    print(f"  N = {results['n_valid']:,} cells")
    print("=" * 60)
