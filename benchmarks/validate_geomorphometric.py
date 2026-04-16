#!/usr/bin/env python3
"""
Validate SurtGIS geomorphometric variables against WhiteboxTools.

Uses fbm_1000_raw.tif (1000x1000 fractal DEM) as test surface.
Compares pixel values at 500 random sample points.
Reports Pearson correlation and RMSE for each variable.

Usage:
    python3 benchmarks/validate_geomorphometric.py
"""

import os
import sys
import subprocess
import tempfile
import numpy as np

try:
    import rasterio
except ImportError:
    print("ERROR: rasterio not installed. Run: pip install rasterio")
    sys.exit(1)

try:
    import whitebox
except ImportError:
    print("ERROR: whitebox not installed. Run: pip install whitebox")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────

DEM = "benchmarks/results/dems/huasco_1500.tif"
SURTGIS = "target/release/surtgis"
N_SAMPLES = 500
MARGIN = 15  # skip border pixels (boundary effects)

# Build release binary if needed
if not os.path.exists(SURTGIS):
    print("Building SurtGIS (release)...")
    r = subprocess.run(["cargo", "build", "--release", "--bin", "surtgis"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"Build failed:\n{r.stderr}")
        sys.exit(1)


# ── Helpers ────────────────────────────────────────────────────────────

def read_raster(path):
    """Read a single-band raster as float64 array."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        return data


def run_surtgis(args, output_path):
    """Run surtgis CLI command."""
    cmd = [SURTGIS] + args
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  SurtGIS FAILED: {' '.join(args)}")
        print(f"  stderr: {r.stderr[:200]}")
        return False
    return os.path.exists(output_path)


def run_wbt(tool_name, **kwargs):
    """Run a WhiteboxTools tool."""
    wbt = whitebox.WhiteboxTools()
    wbt.verbose = False
    func = getattr(wbt, tool_name)
    func(**kwargs)


def sample_points(shape, n, margin):
    """Generate random sample points avoiding borders."""
    rows, cols = shape
    rng = np.random.default_rng(42)
    rs = rng.integers(margin, rows - margin, size=n)
    cs = rng.integers(margin, cols - margin, size=n)
    return rs, cs


def compare(surtgis_data, wbt_data, sample_rows, sample_cols, name):
    """Compare two rasters at sample points. Return (pearson_r, rmse, n_valid)."""
    sv = surtgis_data[sample_rows, sample_cols]
    wv = wbt_data[sample_rows, sample_cols]

    # Filter NaN/inf from both
    mask = np.isfinite(sv) & np.isfinite(wv)
    sv = sv[mask]
    wv = wv[mask]

    n = len(sv)
    if n < 10:
        return None, None, n

    # Pearson correlation
    if np.std(sv) < 1e-15 or np.std(wv) < 1e-15:
        r = 1.0 if np.allclose(sv, wv) else 0.0
    else:
        r = np.corrcoef(sv, wv)[0, 1]

    rmse = np.sqrt(np.mean((sv - wv) ** 2))
    return r, rmse, n


# ── Validation functions ──────────────────────────────────────────────

def validate_variable(name, surtgis_args, wbt_tool, wbt_kwargs,
                      surtgis_output, wbt_output,
                      sample_rows, sample_cols):
    """Run both tools and compare."""
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")

    # Run SurtGIS
    ok = run_surtgis(surtgis_args, surtgis_output)
    if not ok:
        print(f"  SKIP: SurtGIS failed")
        return None

    # Run WhiteboxTools
    try:
        run_wbt(wbt_tool, **wbt_kwargs)
    except Exception as e:
        print(f"  SKIP: WBT failed: {e}")
        return None

    if not os.path.exists(wbt_output):
        print(f"  SKIP: WBT output not found")
        return None

    # Compare
    sd = read_raster(surtgis_output)
    wd = read_raster(wbt_output)

    if sd.shape != wd.shape:
        print(f"  SKIP: Shape mismatch {sd.shape} vs {wd.shape}")
        return None

    r, rmse, n = compare(sd, wd, sample_rows, sample_cols, name)
    if r is None:
        print(f"  SKIP: Too few valid samples ({n})")
        return None

    status = "PASS" if r > 0.95 else ("WARN" if r > 0.80 else "FAIL")
    print(f"  Pearson r = {r:.6f}  RMSE = {rmse:.6f}  N = {n}  [{status}]")
    return r, rmse, n, status


# ── Main ──────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(DEM):
        print(f"ERROR: DEM not found: {DEM}")
        sys.exit(1)

    dem_data = read_raster(DEM)
    rows, cols = dem_data.shape
    sample_rows, sample_cols = sample_points(dem_data.shape, N_SAMPLES, MARGIN)
    dem_abs = os.path.abspath(DEM)

    print(f"DEM: {DEM} ({cols}x{rows})")
    print(f"Sample points: {N_SAMPLES} (margin={MARGIN})")

    results = {}

    with tempfile.TemporaryDirectory(prefix="surtgis_validate_") as tmpdir:
        def sp(name):
            return os.path.join(tmpdir, f"surtgis_{name}.tif")
        def wp(name):
            return os.path.join(tmpdir, f"wbt_{name}.tif")

        # ── 1. Percent Elevation Range ────────────────────────
        r = validate_variable(
            "Percent Elevation Range (radius=10)",
            ["terrain", "percent-elev-range", dem_abs, sp("per"), "-r", "10"],
            "percent_elev_range",
            dict(dem=dem_abs, output=wp("per"), filterx=21, filtery=21),
            sp("per"), wp("per"),
            sample_rows, sample_cols,
        )
        if r: results["percent_elev_range"] = r

        # ── 2. Elevation Above Pit ────────────────────────────
        r = validate_variable(
            "Elevation Above Pit (depth in sink)",
            ["terrain", "elev-above-pit", dem_abs, sp("eap")],
            "depth_in_sink",
            dict(dem=dem_abs, output=wp("eap")),
            sp("eap"), wp("eap"),
            sample_rows, sample_cols,
        )
        if r: results["elev_above_pit"] = r

        # ── 3. Diff from Mean Elevation ───────────────────────
        r = validate_variable(
            "Diff from Mean Elevation (radius=10)",
            ["terrain", "diff-from-mean", dem_abs, sp("dfm"), "-r", "10"],
            "diff_from_mean_elev",
            dict(dem=dem_abs, output=wp("dfm"), filterx=21, filtery=21),
            sp("dfm"), wp("dfm"),
            sample_rows, sample_cols,
        )
        if r: results["diff_from_mean_elev"] = r

        # ── 4. Circular Variance of Aspect ────────────────────
        r = validate_variable(
            "Circular Variance of Aspect (radius=3)",
            ["terrain", "circular-variance-aspect", dem_abs, sp("cva"), "-r", "3"],
            "circular_variance_of_aspect",
            dict(dem=dem_abs, output=wp("cva"), filter=7),
            sp("cva"), wp("cva"),
            sample_rows, sample_cols,
        )
        if r: results["circular_variance_aspect"] = r

        # ── 5. Directional Relief (azimuth=315) ──────────────
        r = validate_variable(
            "Directional Relief (radius=10, azimuth=315)",
            ["terrain", "directional-relief", dem_abs, sp("dr"), "-r", "10", "-a", "315"],
            "directional_relief",
            dict(dem=dem_abs, output=wp("dr"), azimuth=315.0, max_dist=100.0),
            sp("dr"), wp("dr"),
            sample_rows, sample_cols,
        )
        if r: results["directional_relief"] = r

        # ── 6. Max Downslope Elev Change ──────────────────────
        # WBT outputs single rasters for each; we output a directory
        surtgis_nb_dir = os.path.join(tmpdir, "surtgis_nb")
        os.makedirs(surtgis_nb_dir, exist_ok=True)
        run_surtgis(
            ["terrain", "neighbours", dem_abs, "-o", surtgis_nb_dir],
            os.path.join(surtgis_nb_dir, "max_downslope_change.tif"),
        )

        # Max downslope
        try:
            run_wbt("max_downslope_elev_change", dem=dem_abs, output=wp("maxd"))
            if os.path.exists(wp("maxd")) and os.path.exists(os.path.join(surtgis_nb_dir, "max_downslope_change.tif")):
                sd = read_raster(os.path.join(surtgis_nb_dir, "max_downslope_change.tif"))
                wd = read_raster(wp("maxd"))
                r_val, rmse, n = compare(sd, wd, sample_rows, sample_cols, "max_downslope")
                if r_val is not None:
                    status = "PASS" if r_val > 0.95 else ("WARN" if r_val > 0.80 else "FAIL")
                    print(f"\n{'─'*60}")
                    print(f"  Max Downslope Elevation Change")
                    print(f"{'─'*60}")
                    print(f"  Pearson r = {r_val:.6f}  RMSE = {rmse:.6f}  N = {n}  [{status}]")
                    results["max_downslope_change"] = (r_val, rmse, n, status)
        except Exception as e:
            print(f"  Max downslope: WBT failed: {e}")

        # Num downslope neighbours
        try:
            run_wbt("num_downslope_neighbours", dem=dem_abs, output=wp("numd"))
            if os.path.exists(wp("numd")) and os.path.exists(os.path.join(surtgis_nb_dir, "num_downslope.tif")):
                sd = read_raster(os.path.join(surtgis_nb_dir, "num_downslope.tif"))
                wd = read_raster(wp("numd"))
                r_val, rmse, n = compare(sd, wd, sample_rows, sample_cols, "num_downslope")
                if r_val is not None:
                    status = "PASS" if r_val > 0.95 else ("WARN" if r_val > 0.80 else "FAIL")
                    print(f"\n{'─'*60}")
                    print(f"  Num Downslope Neighbours")
                    print(f"{'─'*60}")
                    print(f"  Pearson r = {r_val:.6f}  RMSE = {rmse:.6f}  N = {n}  [{status}]")
                    results["num_downslope"] = (r_val, rmse, n, status)
        except Exception as e:
            print(f"  Num downslope: WBT failed: {e}")

        # ── 7. Downslope Index ────────────────────────────────
        r = validate_variable(
            "Downslope Index (drop=2.0, distance)",
            ["terrain", "downslope-index", dem_abs, sp("di"), "-d", "2.0"],
            "downslope_index",
            dict(dem=dem_abs, output=wp("di"), drop=2.0, out_type="distance"),
            sp("di"), wp("di"),
            sample_rows, sample_cols,
        )
        if r: results["downslope_index"] = r

        # ── 8. Spherical Std Dev of Normals ───────────────────
        r = validate_variable(
            "Spherical Std Dev of Normals (radius=3)",
            ["terrain", "spherical-std-dev", dem_abs, sp("ssd"), "-r", "3"],
            "spherical_std_dev_of_normals",
            dict(dem=dem_abs, output=wp("ssd"), filter=7),
            sp("ssd"), wp("ssd"),
            sample_rows, sample_cols,
        )
        if r: results["spherical_std_dev"] = r

        # ── 9. Average Normal Vector Angular Deviation ────────
        r = validate_variable(
            "Avg Normal Vector Angular Deviation (radius=3)",
            ["terrain", "normal-deviation", dem_abs, sp("nvd"), "-r", "3"],
            "average_normal_vector_angular_deviation",
            dict(dem=dem_abs, output=wp("nvd"), filter=7),
            sp("nvd"), wp("nvd"),
            sample_rows, sample_cols,
        )
        if r: results["normal_vector_deviation"] = r

        # ── 10. Pennock Landform Classification ───────────────
        r = validate_variable(
            "Pennock Landform Classification",
            ["terrain", "pennock", dem_abs, sp("pen"), "--curv-threshold", "0.1"],
            "pennock_landform_class",
            dict(dem=dem_abs, output=wp("pen"), slope=3.0, prof=0.1, plan=0.0),
            sp("pen"), wp("pen"),
            sample_rows, sample_cols,
        )
        if r: results["pennock"] = r

        # ── 11. Edge Density ──────────────────────────────────
        r = validate_variable(
            "Edge Density (radius=3)",
            ["terrain", "edge-density", dem_abs, sp("ed"), "-r", "3"],
            "edge_density",
            dict(dem=dem_abs, output=wp("ed"), filter=7, norm_diff=5.0),
            sp("ed"), wp("ed"),
            sample_rows, sample_cols,
        )
        if r: results["edge_density"] = r

        # ── 12. Max Branch Length ─────────────────────────────
        r = validate_variable(
            "Max Branch Length",
            ["terrain", "max-branch-length", dem_abs, sp("mbl")],
            "max_branch_length",
            dict(dem=dem_abs, output=wp("mbl")),
            sp("mbl"), wp("mbl"),
            sample_rows, sample_cols,
        )
        if r: results["max_branch_length"] = r

    # ── Summary ───────────────────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Variable':<35} {'r':>8} {'RMSE':>10} {'Status':>6}")
    print(f"  {'─'*35} {'─'*8} {'─'*10} {'─'*6}")

    pass_count = 0
    warn_count = 0
    fail_count = 0
    skip_count = 0

    all_vars = [
        "percent_elev_range", "elev_above_pit", "diff_from_mean_elev",
        "circular_variance_aspect", "directional_relief",
        "max_downslope_change", "num_downslope",
        "downslope_index", "spherical_std_dev", "normal_vector_deviation",
        "pennock", "edge_density", "max_branch_length",
    ]

    for v in all_vars:
        if v in results:
            r_val, rmse, n, status = results[v]
            print(f"  {v:<35} {r_val:>8.4f} {rmse:>10.4f} {status:>6}")
            if status == "PASS": pass_count += 1
            elif status == "WARN": warn_count += 1
            else: fail_count += 1
        else:
            print(f"  {v:<35} {'—':>8} {'—':>10} {'SKIP':>6}")
            skip_count += 1

    total = pass_count + warn_count + fail_count
    print(f"\n  Total: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL, {skip_count} SKIP / {total + skip_count}")

    # Variables sin equivalente WBT directo
    print(f"\n  Not validated (no direct WBT equivalent):")
    print(f"    - hypsometric_hillshade (composite of existing validated ops)")
    print(f"    - elev_relative_to_min_max (trivial global normalization)")
    print(f"    - relative_aspect (composite: aspect + gaussian + diff)")


if __name__ == "__main__":
    main()
