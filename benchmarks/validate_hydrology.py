#!/usr/bin/env python3
"""Cross-tool validation of hydrological algorithms for SurtGIS EMS paper.

Compares SurtGIS outputs against GRASS GIS and WhiteboxTools for:
  1. Depression filling (Priority-Flood / r.fill.dir / fill_depressions)
  2. D8 flow accumulation
  3. Stream network extraction

Metrics: RMSE, R², MAE, % cells agreeing within threshold.

Usage:
    python benchmarks/validate_hydrology.py
"""

import os
import sys
import subprocess
import tempfile
import time
import numpy as np
from pathlib import Path

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    import whitebox
    HAS_WBT = True
except ImportError:
    HAS_WBT = False

import surtgis

# ── Paths ──
ROOT = Path(__file__).resolve().parent.parent
DEM_PATH = ROOT / "tests" / "fixtures" / "andes_chile_30m_utm.tif"
RESULTS = ROOT / "benchmarks" / "results"
FIGURES = RESULTS / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def read_dem(path):
    """Read DEM and return (data, cell_size, geotransform, projection)."""
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray().astype(np.float64)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    cell_size = gt[1]
    return data, cell_size, gt, proj


def write_geotiff(path, data, gt, proj, nodata=None):
    """Write a 2D array as GeoTIFF."""
    driver = gdal.GetDriverByName("GTiff")
    rows, cols = data.shape
    ds = driver.Create(str(path), cols, rows, 1, gdal.GDT_Float64)
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(data)
    band.FlushCache()
    ds = None


def read_geotiff(path):
    """Read single-band GeoTIFF as float64 array."""
    ds = gdal.Open(str(path))
    data = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    ds = None
    return data


# ── Tool runners ──

def surtgis_fill(dem, cell_size):
    return surtgis.priority_flood_fill(dem, cell_size)


def surtgis_flow_acc(dem, cell_size):
    return surtgis.flow_accumulation_d8(dem, cell_size)


def surtgis_streams(dem, cell_size, threshold=100.0):
    return surtgis.stream_network_compute(dem, cell_size, threshold=threshold)


def grass_fill(dem_path, output_path):
    """Run GRASS r.fill.dir."""
    cmd = [
        "grass", "--tmp-location", "EPSG:32719", "--exec",
        "bash", "-c",
        f"r.in.gdal input={dem_path} output=dem --overwrite && "
        f"g.region raster=dem && "
        f"r.fill.dir input=dem output=filled direction=dir --overwrite && "
        f"r.out.gdal input=filled output={output_path} format=GTiff type=Float64 createopt=COMPRESS=NONE --overwrite"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"GRASS fill stderr: {result.stderr[:500]}")
    return result.returncode == 0


def grass_flow_acc(dem_path, output_path):
    """Run GRASS r.watershed for D8 flow accumulation."""
    cmd = [
        "grass", "--tmp-location", "EPSG:32719", "--exec",
        "bash", "-c",
        f"r.in.gdal input={dem_path} output=dem --overwrite && "
        f"g.region raster=dem && "
        f"r.watershed elevation=dem accumulation=acc -s --overwrite && "
        f"r.out.gdal input=acc output={output_path} format=GTiff type=Float64 createopt=COMPRESS=NONE --overwrite"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"GRASS flow_acc stderr: {result.stderr[:500]}")
    return result.returncode == 0


def grass_streams(dem_path, output_path, threshold=100):
    """Run GRASS r.watershed for stream extraction."""
    cmd = [
        "grass", "--tmp-location", "EPSG:32719", "--exec",
        "bash", "-c",
        f"r.in.gdal input={dem_path} output=dem --overwrite && "
        f"g.region raster=dem && "
        f"r.watershed elevation=dem stream=streams threshold={threshold} -s --overwrite && "
        f"r.out.gdal input=streams output={output_path} format=GTiff type=Float64 createopt=COMPRESS=NONE --overwrite"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"GRASS streams stderr: {result.stderr[:500]}")
    return result.returncode == 0


def wbt_fill(dem_path, output_path):
    """Run WBT fill_depressions."""
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.fill_depressions(str(dem_path), str(output_path))
    return Path(output_path).exists()


def wbt_flow_acc(dem_path, output_path):
    """Run WBT d8_flow_accumulation."""
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    wbt.d8_flow_accumulation(str(dem_path), str(output_path))
    return Path(output_path).exists()


def wbt_streams(dem_path, output_path, threshold=100):
    """Run WBT to extract streams via flow accumulation threshold."""
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    # WBT: compute flow acc, then threshold
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_acc = tmp.name
    wbt.d8_flow_accumulation(str(dem_path), tmp_acc)
    # Read and threshold
    if Path(tmp_acc).exists():
        acc = read_geotiff(tmp_acc)
        streams = (acc >= threshold).astype(np.float64)
        # We need gt/proj to write
        ds = gdal.Open(str(dem_path))
        gt = ds.GetGeoTransform()
        proj = ds.GetProjection()
        ds = None
        write_geotiff(output_path, streams, gt, proj)
        os.unlink(tmp_acc)
        return True
    return False


# ── Metrics ──

def compute_metrics(a, b, mask=None, name_a="A", name_b="B"):
    """Compute comparison metrics between two arrays."""
    if mask is None:
        mask = np.isfinite(a) & np.isfinite(b)

    va = a[mask]
    vb = b[mask]

    if len(va) == 0:
        return {"n_cells": 0}

    diff = va - vb
    rmse = np.sqrt(np.mean(diff**2))
    mae = np.mean(np.abs(diff))

    # R²
    ss_res = np.sum(diff**2)
    ss_tot = np.sum((vb - np.mean(vb))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # % within thresholds
    pct_exact = 100.0 * np.sum(diff == 0) / len(diff)
    pct_01 = 100.0 * np.sum(np.abs(diff) < 0.1) / len(diff)
    pct_1 = 100.0 * np.sum(np.abs(diff) < 1.0) / len(diff)

    # Correlation
    corr = np.corrcoef(va, vb)[0, 1] if len(va) > 1 else float("nan")

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n_cells": len(va),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "correlation": corr,
        "pct_exact": pct_exact,
        "pct_within_0.1": pct_01,
        "pct_within_1.0": pct_1,
        "max_abs_diff": np.max(np.abs(diff)),
        "mean_a": np.mean(va),
        "mean_b": np.mean(vb),
    }


def compute_stream_metrics(a, b, mask=None, name_a="A", name_b="B"):
    """Compute binary classification metrics for stream networks."""
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)

    sa = (a[mask] > 0.5).astype(int)
    sb = (b[mask] > 0.5).astype(int)

    tp = np.sum((sa == 1) & (sb == 1))
    fp = np.sum((sa == 1) & (sb == 0))
    fn = np.sum((sa == 0) & (sb == 1))
    tn = np.sum((sa == 0) & (sb == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    agreement = 100.0 * (tp + tn) / len(sa)

    return {
        "comparison": f"{name_a} vs {name_b}",
        "n_cells": len(sa),
        "stream_cells_a": int(np.sum(sa)),
        "stream_cells_b": int(np.sum(sb)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "iou": iou,
        "agreement_pct": agreement,
    }


def main():
    print("=" * 70)
    print("HYDROLOGICAL CROSS-TOOL VALIDATION")
    print("=" * 70)

    # Read DEM
    print(f"\nReading DEM: {DEM_PATH}")
    dem, cell_size, gt, proj = read_dem(DEM_PATH)
    nodata_mask = dem < 100  # Andes NoData
    dem_clean = np.where(nodata_mask, np.nan, dem)
    valid_mask = ~nodata_mask
    print(f"  Shape: {dem.shape}, Cell size: {cell_size:.2f} m")
    print(f"  Valid cells: {np.sum(valid_mask)} / {dem.size}")

    # Write DEM with proper NoData for GRASS/WBT
    # Set cells with dem < 100 to NoData value
    dem_for_tools = dem.copy()
    dem_for_tools[nodata_mask] = -9999.0

    with tempfile.TemporaryDirectory() as tmpdir:
        dem_file = os.path.join(tmpdir, "dem.tif")
        write_geotiff(dem_file, dem_for_tools, gt, proj, nodata=-9999.0)

        all_results = []

        # ═══════════════════════════════════════════
        # 1. DEPRESSION FILLING
        # ═══════════════════════════════════════════
        print("\n" + "─" * 50)
        print("1. DEPRESSION FILLING")
        print("─" * 50)

        # SurtGIS
        print("  Computing SurtGIS fill...")
        t0 = time.time()
        fill_surtgis = surtgis.priority_flood_fill(dem, cell_size)
        t_surtgis = time.time() - t0
        fill_surtgis = np.where(nodata_mask, np.nan, fill_surtgis)
        print(f"    Done in {t_surtgis:.3f}s")

        # GRASS
        fill_grass = None
        grass_fill_file = os.path.join(tmpdir, "fill_grass.tif")
        print("  Computing GRASS fill...")
        t0 = time.time()
        if grass_fill(dem_file, grass_fill_file):
            fill_grass = read_geotiff(grass_fill_file)
            fill_grass = np.where(nodata_mask, np.nan, fill_grass)
            t_grass = time.time() - t0
            print(f"    Done in {t_grass:.3f}s")
        else:
            print("    GRASS fill FAILED")

        # WBT
        fill_wbt = None
        wbt_fill_file = os.path.join(tmpdir, "fill_wbt.tif")
        print("  Computing WBT fill...")
        t0 = time.time()
        if wbt_fill(dem_file, wbt_fill_file):
            fill_wbt = read_geotiff(wbt_fill_file)
            fill_wbt = np.where(nodata_mask, np.nan, fill_wbt)
            t_wbt = time.time() - t0
            print(f"    Done in {t_wbt:.3f}s")
        else:
            print("    WBT fill FAILED")

        # Compare fills
        print("\n  Fill comparison metrics:")

        # Basic check: filled >= original
        fill_diff = fill_surtgis - dem_clean
        filled_cells = np.sum(fill_diff[valid_mask] > 0)
        print(f"    SurtGIS filled cells: {filled_cells} ({100*filled_cells/np.sum(valid_mask):.2f}%)")
        print(f"    Max fill depth: {np.nanmax(fill_diff):.3f} m")
        print(f"    Mean fill depth (filled cells only): {np.nanmean(fill_diff[fill_diff > 0]):.4f} m")

        if fill_grass is not None:
            m = compute_metrics(fill_surtgis, fill_grass, valid_mask,
                               "SurtGIS", "GRASS")
            print(f"\n    SurtGIS vs GRASS fill:")
            print(f"      RMSE: {m['rmse']:.6f} m")
            print(f"      MAE:  {m['mae']:.6f} m")
            print(f"      R²:   {m['r2']:.10f}")
            print(f"      Max diff: {m['max_abs_diff']:.6f} m")
            print(f"      % within 0.01m: {100*np.sum(np.abs(fill_surtgis[valid_mask]-fill_grass[valid_mask])<0.01)/np.sum(valid_mask):.2f}%")
            all_results.append(("fill", m))

        if fill_wbt is not None:
            m = compute_metrics(fill_surtgis, fill_wbt, valid_mask,
                               "SurtGIS", "WBT")
            print(f"\n    SurtGIS vs WBT fill:")
            print(f"      RMSE: {m['rmse']:.6f} m")
            print(f"      MAE:  {m['mae']:.6f} m")
            print(f"      R²:   {m['r2']:.10f}")
            print(f"      Max diff: {m['max_abs_diff']:.6f} m")
            all_results.append(("fill", m))

        if fill_grass is not None and fill_wbt is not None:
            m = compute_metrics(fill_grass, fill_wbt, valid_mask,
                               "GRASS", "WBT")
            print(f"\n    GRASS vs WBT fill:")
            print(f"      RMSE: {m['rmse']:.6f} m")
            print(f"      MAE:  {m['mae']:.6f} m")
            print(f"      R²:   {m['r2']:.10f}")
            all_results.append(("fill", m))

        # ═══════════════════════════════════════════
        # 2. D8 FLOW ACCUMULATION
        # ═══════════════════════════════════════════
        print("\n" + "─" * 50)
        print("2. D8 FLOW ACCUMULATION")
        print("─" * 50)

        # SurtGIS: flow_direction first, then flow_accumulation
        print("  Computing SurtGIS D8 flow direction...")
        t0 = time.time()
        fdir_surtgis = surtgis.flow_direction_d8(dem, cell_size)
        print(f"    Flow dir done in {time.time()-t0:.3f}s")

        print("  Computing SurtGIS D8 flow acc...")
        t0 = time.time()
        acc_surtgis = surtgis.flow_accumulation_d8(fdir_surtgis, cell_size)
        t_surtgis = time.time() - t0
        acc_surtgis_masked = np.where(nodata_mask, np.nan, acc_surtgis.astype(np.float64))
        print(f"    Done in {t_surtgis:.3f}s")
        print(f"    Max accumulation: {np.nanmax(acc_surtgis_masked):.0f}")

        # GRASS
        acc_grass = None
        grass_acc_file = os.path.join(tmpdir, "acc_grass.tif")
        print("  Computing GRASS D8 flow acc...")
        t0 = time.time()
        if grass_flow_acc(dem_file, grass_acc_file):
            acc_grass_raw = read_geotiff(grass_acc_file)
            # GRASS r.watershed with -s returns negative values for streams
            # The absolute value is the accumulation
            acc_grass = np.abs(acc_grass_raw)
            acc_grass = np.where(nodata_mask, np.nan, acc_grass)
            t_grass = time.time() - t0
            print(f"    Done in {t_grass:.3f}s")
            print(f"    Max accumulation: {np.nanmax(acc_grass):.0f}")
        else:
            print("    GRASS flow_acc FAILED")

        # WBT
        acc_wbt = None
        wbt_acc_file = os.path.join(tmpdir, "acc_wbt.tif")
        print("  Computing WBT D8 flow acc...")
        t0 = time.time()
        if wbt_flow_acc(dem_file, wbt_acc_file):
            acc_wbt = read_geotiff(wbt_acc_file)
            acc_wbt = np.where(nodata_mask, np.nan, acc_wbt)
            t_wbt = time.time() - t0
            print(f"    Done in {t_wbt:.3f}s")
            print(f"    Max accumulation: {np.nanmax(acc_wbt):.0f}")
        else:
            print("    WBT flow_acc FAILED")

        # Compare flow accumulation
        # Use log10 for comparison since flow acc spans many orders of magnitude
        print("\n  Flow accumulation comparison metrics:")

        def log_acc(arr, mask):
            """Log10 of flow accumulation for comparison."""
            a = arr.copy()
            a[a <= 0] = 1  # Avoid log(0)
            return np.log10(a)

        if acc_grass is not None:
            # Raw comparison
            m = compute_metrics(acc_surtgis_masked, acc_grass, valid_mask,
                               "SurtGIS", "GRASS")
            print(f"\n    SurtGIS vs GRASS D8 flow acc:")
            print(f"      Correlation: {m['correlation']:.6f}")
            print(f"      % cells exact match: {m['pct_exact']:.2f}%")

            # Log comparison
            log_s = log_acc(acc_surtgis_masked, valid_mask)
            log_g = log_acc(acc_grass, valid_mask)
            m_log = compute_metrics(log_s, log_g, valid_mask,
                                   "SurtGIS(log)", "GRASS(log)")
            print(f"      Log10 RMSE: {m_log['rmse']:.4f}")
            print(f"      Log10 R²:   {m_log['r2']:.6f}")
            all_results.append(("flow_acc", m))
            all_results.append(("flow_acc_log", m_log))

        if acc_wbt is not None:
            m = compute_metrics(acc_surtgis_masked, acc_wbt, valid_mask,
                               "SurtGIS", "WBT")
            print(f"\n    SurtGIS vs WBT D8 flow acc:")
            print(f"      Correlation: {m['correlation']:.6f}")
            print(f"      % cells exact match: {m['pct_exact']:.2f}%")

            log_s = log_acc(acc_surtgis_masked, valid_mask)
            log_w = log_acc(acc_wbt, valid_mask)
            m_log = compute_metrics(log_s, log_w, valid_mask,
                                   "SurtGIS(log)", "WBT(log)")
            print(f"      Log10 RMSE: {m_log['rmse']:.4f}")
            print(f"      Log10 R²:   {m_log['r2']:.6f}")
            all_results.append(("flow_acc", m))
            all_results.append(("flow_acc_log", m_log))

        # ═══════════════════════════════════════════
        # 3. STREAM NETWORK
        # ═══════════════════════════════════════════
        print("\n" + "─" * 50)
        print("3. STREAM NETWORK (threshold=100)")
        print("─" * 50)

        # SurtGIS
        print("  Computing SurtGIS streams...")
        streams_surtgis = surtgis.stream_network_compute(dem, cell_size, threshold=100.0)
        streams_surtgis = np.where(nodata_mask, 0, streams_surtgis)
        print(f"    Stream cells: {np.sum(streams_surtgis > 0.5)}")

        # GRASS
        streams_grass = None
        grass_stream_file = os.path.join(tmpdir, "streams_grass.tif")
        print("  Computing GRASS streams...")
        if grass_streams(dem_file, grass_stream_file, threshold=100):
            streams_grass_raw = read_geotiff(grass_stream_file)
            # GRASS streams: positive values = stream ID, 0 = no stream
            streams_grass = (streams_grass_raw > 0).astype(np.float64)
            streams_grass = np.where(nodata_mask, 0, streams_grass)
            print(f"    Stream cells: {np.sum(streams_grass > 0.5)}")
        else:
            print("    GRASS streams FAILED")

        # WBT
        streams_wbt = None
        wbt_stream_file = os.path.join(tmpdir, "streams_wbt.tif")
        print("  Computing WBT streams...")
        if wbt_streams(dem_file, wbt_stream_file, threshold=100):
            streams_wbt = read_geotiff(wbt_stream_file)
            streams_wbt = np.where(nodata_mask, 0, streams_wbt)
            print(f"    Stream cells: {np.sum(streams_wbt > 0.5)}")
        else:
            print("    WBT streams FAILED")

        # Compare streams (binary classification metrics)
        print("\n  Stream network comparison:")

        if streams_grass is not None:
            m = compute_stream_metrics(streams_surtgis, streams_grass, valid_mask,
                                       "SurtGIS", "GRASS")
            print(f"\n    SurtGIS vs GRASS:")
            print(f"      Agreement: {m['agreement_pct']:.2f}%")
            print(f"      F1 score:  {m['f1_score']:.4f}")
            print(f"      IoU:       {m['iou']:.4f}")
            print(f"      Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
            all_results.append(("streams", m))

        if streams_wbt is not None:
            m = compute_stream_metrics(streams_surtgis, streams_wbt, valid_mask,
                                       "SurtGIS", "WBT")
            print(f"\n    SurtGIS vs WBT:")
            print(f"      Agreement: {m['agreement_pct']:.2f}%")
            print(f"      F1 score:  {m['f1_score']:.4f}")
            print(f"      IoU:       {m['iou']:.4f}")
            print(f"      Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}")
            all_results.append(("streams", m))

        # ═══════════════════════════════════════════
        # 4. GENERATE FIGURE
        # ═══════════════════════════════════════════
        print("\n" + "─" * 50)
        print("4. GENERATING VALIDATION FIGURE")
        print("─" * 50)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        extent_km = [0, dem.shape[1] * cell_size / 1000,
                     0, dem.shape[0] * cell_size / 1000]

        fig, axes = plt.subplots(3, 3, figsize=(15, 13))

        # Row 1: Fill comparison
        # (a) SurtGIS fill depth
        ax = axes[0, 0]
        fill_depth = fill_surtgis - dem_clean
        fill_depth_display = np.where(fill_depth <= 0, np.nan, fill_depth)
        im = ax.imshow(fill_depth_display, cmap="Reds", extent=extent_km,
                       origin="upper", vmin=0, vmax=np.nanpercentile(fill_depth_display[np.isfinite(fill_depth_display)], 99))
        ax.set_title("(a) SurtGIS fill depth", fontsize=10, fontweight="bold")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("Fill depth (m)", fontsize=8)

        # (b) Fill difference SurtGIS - GRASS
        ax = axes[0, 1]
        if fill_grass is not None:
            fill_diff_sg = fill_surtgis - fill_grass
            vmax_d = max(np.nanpercentile(np.abs(fill_diff_sg[valid_mask]), 99), 1e-6)
            im = ax.imshow(fill_diff_sg, cmap="RdBu_r", extent=extent_km,
                           origin="upper", vmin=-vmax_d, vmax=vmax_d)
            ax.set_title("(b) Fill diff: SurtGIS - GRASS", fontsize=10, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label("Difference (m)", fontsize=8)
        else:
            ax.text(0.5, 0.5, "GRASS unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(b) Fill diff: SurtGIS - GRASS", fontsize=10)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

        # (c) Fill difference SurtGIS - WBT
        ax = axes[0, 2]
        if fill_wbt is not None:
            fill_diff_sw = fill_surtgis - fill_wbt
            vmax_d = max(np.nanpercentile(np.abs(fill_diff_sw[valid_mask]), 99), 1e-6)
            im = ax.imshow(fill_diff_sw, cmap="RdBu_r", extent=extent_km,
                           origin="upper", vmin=-vmax_d, vmax=vmax_d)
            ax.set_title("(c) Fill diff: SurtGIS - WBT", fontsize=10, fontweight="bold")
            cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label("Difference (m)", fontsize=8)
        else:
            ax.text(0.5, 0.5, "WBT unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(c) Fill diff: SurtGIS - WBT", fontsize=10)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

        # Row 2: Flow accumulation comparison (log scale)
        # (d) SurtGIS log flow acc
        ax = axes[1, 0]
        log_acc_s = np.where(valid_mask & (acc_surtgis > 0),
                             np.log10(acc_surtgis.astype(np.float64)), np.nan)
        im = ax.imshow(log_acc_s, cmap="Blues", extent=extent_km, origin="upper",
                       vmin=0, vmax=np.nanpercentile(log_acc_s[np.isfinite(log_acc_s)], 99))
        ax.set_title("(d) SurtGIS log10(flow acc)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("log10(cells)", fontsize=8)

        # (e) Scatter SurtGIS vs GRASS flow acc
        ax = axes[1, 1]
        if acc_grass is not None:
            s_vals = acc_surtgis_masked[valid_mask]
            g_vals = acc_grass[valid_mask]
            # Subsample for scatter
            idx = np.random.RandomState(42).choice(len(s_vals),
                                                    min(50000, len(s_vals)),
                                                    replace=False)
            ax.scatter(np.log10(np.maximum(s_vals[idx], 1)),
                      np.log10(np.maximum(g_vals[idx], 1)),
                      s=0.5, alpha=0.3, c="steelblue")
            ax.plot([0, 6], [0, 6], "r--", lw=1, label="1:1")
            ax.set_xlabel("SurtGIS log10(acc)")
            ax.set_ylabel("GRASS log10(acc)")
            ax.set_title("(e) Flow acc: SurtGIS vs GRASS", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
            ax.set_xlim(0, 5.5)
            ax.set_ylim(0, 5.5)
        else:
            ax.text(0.5, 0.5, "GRASS unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(e) Flow acc: SurtGIS vs GRASS", fontsize=10)

        # (f) Scatter SurtGIS vs WBT flow acc
        ax = axes[1, 2]
        if acc_wbt is not None:
            s_vals = acc_surtgis_masked[valid_mask]
            w_vals = acc_wbt[valid_mask]
            idx = np.random.RandomState(42).choice(len(s_vals),
                                                    min(50000, len(s_vals)),
                                                    replace=False)
            ax.scatter(np.log10(np.maximum(s_vals[idx], 1)),
                      np.log10(np.maximum(w_vals[idx], 1)),
                      s=0.5, alpha=0.3, c="darkorange")
            ax.plot([0, 6], [0, 6], "r--", lw=1, label="1:1")
            ax.set_xlabel("SurtGIS log10(acc)")
            ax.set_ylabel("WBT log10(acc)")
            ax.set_title("(f) Flow acc: SurtGIS vs WBT", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.set_aspect("equal")
            ax.set_xlim(0, 5.5)
            ax.set_ylim(0, 5.5)
        else:
            ax.text(0.5, 0.5, "WBT unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(f) Flow acc: SurtGIS vs WBT", fontsize=10)

        # Row 3: Stream network comparison
        # (g) SurtGIS streams
        ax = axes[2, 0]
        streams_display = np.where(nodata_mask, np.nan, dem_clean)
        ax.imshow(streams_display, cmap="terrain", extent=extent_km, origin="upper",
                  alpha=0.5,
                  vmin=np.nanpercentile(dem_clean, 2), vmax=np.nanpercentile(dem_clean, 98))
        stream_overlay = np.ma.masked_where(streams_surtgis < 0.5, streams_surtgis)
        ax.imshow(stream_overlay, cmap=mcolors.ListedColormap(["blue"]),
                  extent=extent_km, origin="upper", alpha=0.9)
        ax.set_title("(g) SurtGIS streams", fontsize=10, fontweight="bold")
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

        # (h) Stream overlap SurtGIS vs GRASS
        ax = axes[2, 1]
        if streams_grass is not None:
            overlap = np.zeros(dem.shape + (3,), dtype=np.float32)
            # Green = both agree, Red = SurtGIS only, Blue = GRASS only
            both = (streams_surtgis > 0.5) & (streams_grass > 0.5)
            surt_only = (streams_surtgis > 0.5) & (streams_grass < 0.5)
            grass_only = (streams_surtgis < 0.5) & (streams_grass > 0.5)
            bg = np.where(nodata_mask, 1.0, 0.9)
            overlap[:, :, 0] = bg.copy()
            overlap[:, :, 1] = bg.copy()
            overlap[:, :, 2] = bg.copy()
            overlap[both, :] = [0, 0.7, 0]       # Green
            overlap[surt_only, :] = [1, 0, 0]     # Red
            overlap[grass_only, :] = [0, 0, 1]     # Blue
            overlap[nodata_mask, :] = [1, 1, 1]
            ax.imshow(overlap, extent=extent_km, origin="upper")
            ax.set_title("(h) Streams: SurtGIS vs GRASS", fontsize=10, fontweight="bold")
            # Legend
            from matplotlib.patches import Patch
            legend_items = [Patch(facecolor="green", label="Both"),
                           Patch(facecolor="red", label="SurtGIS only"),
                           Patch(facecolor="blue", label="GRASS only")]
            ax.legend(handles=legend_items, fontsize=7, loc="lower right")
        else:
            ax.text(0.5, 0.5, "GRASS unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(h) Streams: SurtGIS vs GRASS", fontsize=10)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

        # (i) Stream overlap SurtGIS vs WBT
        ax = axes[2, 2]
        if streams_wbt is not None:
            overlap = np.zeros(dem.shape + (3,), dtype=np.float32)
            both = (streams_surtgis > 0.5) & (streams_wbt > 0.5)
            surt_only = (streams_surtgis > 0.5) & (streams_wbt < 0.5)
            wbt_only = (streams_surtgis < 0.5) & (streams_wbt > 0.5)
            bg = np.where(nodata_mask, 1.0, 0.9)
            overlap[:, :, 0] = bg.copy()
            overlap[:, :, 1] = bg.copy()
            overlap[:, :, 2] = bg.copy()
            overlap[both, :] = [0, 0.7, 0]
            overlap[surt_only, :] = [1, 0, 0]
            overlap[wbt_only, :] = [0.8, 0.5, 0]
            overlap[nodata_mask, :] = [1, 1, 1]
            ax.imshow(overlap, extent=extent_km, origin="upper")
            ax.set_title("(i) Streams: SurtGIS vs WBT", fontsize=10, fontweight="bold")
            from matplotlib.patches import Patch
            legend_items = [Patch(facecolor="green", label="Both"),
                           Patch(facecolor="red", label="SurtGIS only"),
                           Patch(facecolor=[0.8, 0.5, 0], label="WBT only")]
            ax.legend(handles=legend_items, fontsize=7, loc="lower right")
        else:
            ax.text(0.5, 0.5, "WBT unavailable", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("(i) Streams: SurtGIS vs WBT", fontsize=10)
        ax.set_xlabel("Easting (km)")
        ax.set_ylabel("Northing (km)")

        fig.suptitle(
            f"Hydrological algorithm cross-tool validation "
            f"(Andes 30 m DEM, {dem.shape[0]}$\\times${dem.shape[1]} cells)",
            fontsize=12, fontweight="bold", y=0.99
        )
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_pdf = FIGURES / "fig_hydro_validation.pdf"
        out_png = FIGURES / "fig_hydro_validation.png"
        fig.savefig(out_pdf, dpi=300)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"\n  Figure saved to: {out_pdf}")

    # ═══════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    print(f"{'Algorithm':<16} {'Comparison':<22} {'RMSE':<14} {'R²':<12} {'Correlation':<12}")
    print("─" * 76)
    for algo, m in all_results:
        if "rmse" in m:
            print(f"{algo:<16} {m['comparison']:<22} {m['rmse']:<14.6f} {m.get('r2', float('nan')):<12.6f} {m.get('correlation', float('nan')):<12.6f}")

    # Save CSV
    csv_path = RESULTS / "hydro_validation.csv"
    with open(csv_path, "w") as f:
        f.write("algorithm,comparison,metric,value\n")
        for algo, m in all_results:
            comp = m.get("comparison", "")
            for k, v in m.items():
                if k in ("comparison",):
                    continue
                f.write(f"{algo},{comp},{k},{v}\n")
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
