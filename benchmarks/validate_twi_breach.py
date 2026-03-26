#!/usr/bin/env python3
"""R3: Breach timing + E5: TWI cross-tool validation.

Runs:
  - SurtGIS breach_fill on Andes DEM (timing, 5 reps)
  - SurtGIS TWI vs GRASS r.topidx vs WBT WetnessIndex on Andes UTM DEM
"""

import time
import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np

# Paths
BASE = Path(__file__).resolve().parent.parent
ANDES_UTM = BASE / "tests" / "fixtures" / "andes_chile_30m_utm.tif"
RESULTS_DIR = BASE / "benchmarks" / "results"

def read_geotiff(path):
    """Read GeoTIFF as numpy array using GDAL."""
    from osgeo import gdal
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray().astype(np.float64)
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    ds = None
    return arr

def write_geotiff_from_reference(data, ref_path, out_path):
    """Write array as GeoTIFF using ref file for metadata."""
    from osgeo import gdal
    ref = gdal.Open(str(ref_path))
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = data.shape
    ds = driver.Create(str(out_path), cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(ref.GetGeoTransform())
    ds.SetProjection(ref.GetProjection())
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.SetNoDataValue(-9999.0)
    ds = None
    ref = None

# ── R3: Breach timing ──────────────────────────────────────────
def benchmark_breach():
    """Time SurtGIS breach_fill on the Andes DEM."""
    import surtgis

    print("=" * 60)
    print("R3: Breach timing (SurtGIS breach_fill)")
    print("=" * 60)

    dem = read_geotiff(ANDES_UTM)
    cell_size = 28.4
    rows, cols = dem.shape
    total_cells = rows * cols
    print(f"DEM: {rows}x{cols} = {total_cells:,} cells")
    print(f"Cell size: {cell_size} m")

    # Warmup
    print("\nWarmup (1 run)...")
    _ = surtgis.breach_fill(dem, cell_size)

    # Timed runs
    n_reps = 5
    times = []
    print(f"Timing ({n_reps} runs)...")
    for i in range(n_reps):
        t0 = time.perf_counter()
        result = surtgis.breach_fill(dem, cell_size)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.3f} s")

    median_t = np.median(times)
    iqr = np.percentile(times, 75) - np.percentile(times, 25)
    print(f"\nMedian: {median_t:.3f} s (IQR: {iqr:.3f} s)")
    print(f"Cells: {total_cells:,}")

    # Count cells modified
    dem_clean = np.nan_to_num(dem, nan=0.0)
    result_clean = np.nan_to_num(result, nan=0.0)
    diff = np.abs(result_clean - dem_clean)
    modified = np.sum(diff > 1e-10)
    print(f"Cells modified by breach: {modified:,} ({100*modified/total_cells:.2f}%)")
    print(f"Max depth change: {np.max(diff):.2f} m")

    return median_t, total_cells


# ── E5: TWI cross-tool validation ──────────────────────────────
def validate_twi():
    """Compare SurtGIS TWI vs GRASS r.topidx vs WBT WetnessIndex."""
    import surtgis

    print("\n" + "=" * 60)
    print("E5: TWI cross-tool validation")
    print("=" * 60)

    dem = read_geotiff(ANDES_UTM)
    cell_size = 28.4
    rows, cols = dem.shape
    print(f"DEM: {rows}x{cols}, cell_size={cell_size} m")

    # 1. SurtGIS TWI
    print("\n[1/3] Computing SurtGIS TWI...")
    t0 = time.perf_counter()
    twi_sg = surtgis.twi_compute(dem, cell_size)
    t1 = time.perf_counter()
    print(f"  Done in {t1-t0:.3f} s")
    print(f"  Range: [{np.nanmin(twi_sg):.2f}, {np.nanmax(twi_sg):.2f}]")
    print(f"  Mean: {np.nanmean(twi_sg):.2f}, Std: {np.nanstd(twi_sg):.2f}")

    # 2. GRASS r.topidx
    print("\n[2/3] Computing GRASS r.topidx...")
    with tempfile.TemporaryDirectory() as tmpdir:
        grass_out = Path(tmpdir) / "twi_grass.tif"
        grass_cmd = (
            f"r.in.gdal input={ANDES_UTM} output=dem --overwrite 2>/dev/null && "
            f"g.region raster=dem && "
            f"r.topidx input=dem output=twi --overwrite 2>/dev/null && "
            f"r.out.gdal input=twi output={grass_out} format=GTiff type=Float64 "
            f"createopt=COMPRESS=NONE --overwrite 2>/dev/null"
        )
        t0 = time.perf_counter()
        result = subprocess.run(
            ["grass", "--tmp-location", "EPSG:32719", "--exec", "bash", "-c", grass_cmd],
            capture_output=True, text=True, timeout=120,
            env={**__import__('os').environ, "LD_LIBRARY_PATH": "/lib/x86_64-linux-gnu"}
        )
        t1 = time.perf_counter()

        if result.returncode != 0:
            print(f"  GRASS FAILED: {result.stderr[:500]}")
            twi_grass = None
        else:
            print(f"  Done in {t1-t0:.3f} s")
            twi_grass = read_geotiff(grass_out)
            print(f"  Range: [{np.nanmin(twi_grass):.2f}, {np.nanmax(twi_grass):.2f}]")
            print(f"  Mean: {np.nanmean(twi_grass):.2f}, Std: {np.nanstd(twi_grass):.2f}")

    # 3. WBT WetnessIndex
    print("\n[3/3] Computing WBT WetnessIndex...")
    try:
        import whitebox
        wbt = whitebox.WhiteboxTools()
        wbt.set_verbose_mode(False)

        with tempfile.TemporaryDirectory() as tmpdir:
            wbt_out = Path(tmpdir) / "twi_wbt.tif"
            t0 = time.perf_counter()
            wbt.wetness_index(str(ANDES_UTM.resolve()), str(wbt_out.resolve()))
            t1 = time.perf_counter()
            print(f"  Done in {t1-t0:.3f} s")
            twi_wbt = read_geotiff(wbt_out)
            print(f"  Range: [{np.nanmin(twi_wbt):.2f}, {np.nanmax(twi_wbt):.2f}]")
            print(f"  Mean: {np.nanmean(twi_wbt):.2f}, Std: {np.nanstd(twi_wbt):.2f}")
    except Exception as e:
        print(f"  WBT FAILED: {e}")
        twi_wbt = None

    # ── Cross-tool comparison ──
    print("\n" + "-" * 40)
    print("Cross-tool comparison (valid pixels only)")
    print("-" * 40)

    def compare(name, a, b):
        """Compare two TWI arrays, handling NaN/Inf."""
        mask = np.isfinite(a) & np.isfinite(b)
        n_valid = np.sum(mask)
        a_v, b_v = a[mask], b[mask]
        rmse = np.sqrt(np.mean((a_v - b_v) ** 2))
        mae = np.mean(np.abs(a_v - b_v))
        ss_res = np.sum((a_v - b_v) ** 2)
        ss_tot = np.sum((b_v - np.mean(b_v)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        corr = np.corrcoef(a_v, b_v)[0, 1]
        pct_01 = 100 * np.sum(np.abs(a_v - b_v) < 0.1) / n_valid
        pct_1 = 100 * np.sum(np.abs(a_v - b_v) < 1.0) / n_valid
        print(f"\n{name}:")
        print(f"  Valid pixels: {n_valid:,} / {a.size:,}")
        print(f"  RMSE:         {rmse:.4f}")
        print(f"  MAE:          {mae:.4f}")
        print(f"  R²:           {r2:.6f}")
        print(f"  Pearson r:    {corr:.6f}")
        print(f"  Within 0.1:   {pct_01:.1f}%")
        print(f"  Within 1.0:   {pct_1:.1f}%")
        return rmse, r2, corr

    results = {}
    if twi_grass is not None:
        results['sg_vs_grass'] = compare("SurtGIS vs GRASS", twi_sg, twi_grass)
    if twi_wbt is not None:
        results['sg_vs_wbt'] = compare("SurtGIS vs WBT", twi_sg, twi_wbt)
    if twi_grass is not None and twi_wbt is not None:
        results['grass_vs_wbt'] = compare("GRASS vs WBT", twi_grass, twi_wbt)

    return results


if __name__ == "__main__":
    print("SurtGIS R3 (Breach) + E5 (TWI) validation\n")

    # Check DEM exists
    if not ANDES_UTM.exists():
        print(f"ERROR: DEM not found at {ANDES_UTM}")
        sys.exit(1)

    # R3: Breach
    breach_time, breach_cells = benchmark_breach()

    # E5: TWI
    twi_results = validate_twi()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"R3 Breach: {breach_time:.3f} s on {breach_cells:,} cells")
    if twi_results:
        print("E5 TWI cross-tool:")
        for key, (rmse, r2, corr) in twi_results.items():
            print(f"  {key}: RMSE={rmse:.4f}, R²={r2:.4f}, r={corr:.4f}")
