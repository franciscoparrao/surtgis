#!/usr/bin/env python3
"""Validate SurtGIS 14-curvature system against analytical solutions and SAGA GIS.

Generates a Gaussian hill DEM with known analytical curvatures, computes all 14
curvature types with SurtGIS, and validates against:

  1. Analytical (closed-form) curvatures derived from the Gaussian surface
  2. SAGA GIS curvatures (ta_morphometry module 0)

The Gaussian hill z = A * exp(-(x^2 + y^2) / (2*sigma^2)) has known partial
derivatives and therefore exact curvature values at every point, making it an
ideal analytical test surface.

Reference: Florinsky, I.V. (2025) "Digital Terrain Analysis" 3rd ed., Ch. 2.

Usage:
    .venv/bin/python3 benchmarks/validate_curvatures.py
"""

import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import surtgis
except ImportError:
    sys.exit("ERROR: surtgis not installed. Run: maturin develop --release")

try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

# ============================================================================
# Configuration
# ============================================================================

# Gaussian hill parameters
A = 500.0         # Amplitude (meters)
SIGMA = 2000.0    # Standard deviation (meters)
CELL_SIZE = 10.0  # Cell size (meters)
GRID_SIZE = 1000  # 1000 x 1000 pixels

# Derived
SIGMA2 = SIGMA ** 2
SIGMA4 = SIGMA ** 4

# Paths
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SAGA_CMD = "/usr/local/bin/saga_cmd"
SAGA_ENV = {**os.environ, "LD_LIBRARY_PATH": "/lib/x86_64-linux-gnu"}

# Border exclusion: SurtGIS uses a 3x3 window, so the 1-pixel border is NaN.
BORDER = 2

# Threshold for excluding pixels near the peak where gradient is near-zero.
GRAD_EPSILON = 1e-10

# All 14 curvature types in SurtGIS
ALL_CURVATURE_TYPES = [
    "mean_h", "gaussian_k", "unsphericity_m", "difference_e",
    "minimal_kmin", "maximal_kmax", "horizontal_kh", "vertical_kv",
    "horizontal_excess_khe", "vertical_excess_kve", "accumulation_ka",
    "ring_kr", "rotor", "laplacian",
]

# Curvatures that require p^2+q^2 > 0 (involve division by gradient magnitude)
GRADIENT_DEPENDENT = {
    "horizontal_kh", "vertical_kv",
    "horizontal_excess_khe", "vertical_excess_kve",
    "accumulation_ka", "ring_kr",
    "difference_e",
    "rotor",
}

# SAGA output parameter names mapped to SurtGIS curvature types
# SAGA ta_morphometry 0 outputs (using correct SAGA parameter names):
SAGA_TO_SURTGIS = {
    "C_PLAN": "horizontal_kh",      # plan curvature
    "C_PROF": "vertical_kv",        # profile curvature
    "C_MINI": "minimal_kmin",       # minimal curvature
    "C_MAXI": "maximal_kmax",       # maximal curvature
    "C_TOTA": "gaussian_k",         # total curvature (Gaussian)
    "C_GENE": "laplacian",          # general curvature (Laplacian)
    "C_ROTO": "rotor",              # flow line curvature (rotor)
}


# ============================================================================
# Gaussian Hill: DEM Generation
# ============================================================================

def generate_gaussian_hill():
    """Generate a 1000x1000 Gaussian hill DEM.

    z(x,y) = A * exp(-(x^2 + y^2) / (2 * sigma^2))

    The grid is centered so the peak is at (row=500, col=500).
    x increases to the east (column direction).
    y increases to the north (opposite to row direction).

    Returns:
        dem: (GRID_SIZE, GRID_SIZE) float64 array
        x_grid: (GRID_SIZE, GRID_SIZE) x-coordinates
        y_grid: (GRID_SIZE, GRID_SIZE) y-coordinates
    """
    cx = GRID_SIZE // 2
    cy = GRID_SIZE // 2

    cols = np.arange(GRID_SIZE, dtype=np.float64)
    rows = np.arange(GRID_SIZE, dtype=np.float64)
    col_grid, row_grid = np.meshgrid(cols, rows)

    x_grid = (col_grid - cx) * CELL_SIZE
    # Row 0 is top (north), row increases southward -> y decreases
    y_grid = (cy - row_grid) * CELL_SIZE

    r2 = x_grid ** 2 + y_grid ** 2
    dem = A * np.exp(-r2 / (2.0 * SIGMA2))

    return dem, x_grid, y_grid


# ============================================================================
# Analytical Curvatures for Gaussian Hill
# ============================================================================

def analytical_partial_derivatives(x, y):
    """Compute the five partial derivatives (p,q,r,s,t) for the Gaussian hill.

    For z = A * exp(-(x^2 + y^2) / (2*sigma^2)):

        p = dz/dx = -A*x/sigma^2 * exp(...)
        q = dz/dy = -A*y/sigma^2 * exp(...)
        r = d^2z/dx^2 = A*(x^2 - sigma^2)/sigma^4 * exp(...)
        t = d^2z/dy^2 = A*(y^2 - sigma^2)/sigma^4 * exp(...)
        s = d^2z/dxdy = A*x*y/sigma^4 * exp(...)

    Returns:
        p, q, r, s, t: arrays of partial derivatives
    """
    r2 = x ** 2 + y ** 2
    exp_term = A * np.exp(-r2 / (2.0 * SIGMA2))

    p = -x / SIGMA2 * exp_term
    q = -y / SIGMA2 * exp_term
    r = (x ** 2 - SIGMA2) / SIGMA4 * exp_term
    t = (y ** 2 - SIGMA2) / SIGMA4 * exp_term
    s = (x * y) / SIGMA4 * exp_term

    return p, q, r, s, t


def analytical_curvatures(x, y):
    """Compute all 14 analytical curvatures for the Gaussian hill.

    Uses the exact Florinsky formulas.

    Returns:
        dict mapping curvature_type -> 2D array
    """
    p, q, r, s, t = analytical_partial_derivatives(x, y)

    p2 = p * p
    q2 = q * q
    p2q2 = p2 + q2
    w = 1.0 + p2q2        # 1 + p^2 + q^2
    w_sqrt = np.sqrt(w)
    w_32 = w * w_sqrt      # (1 + p^2 + q^2)^(3/2)

    # Mean curvature H
    mean_h = -((1.0 + q2) * r - 2.0 * p * q * s + (1.0 + p2) * t) / (2.0 * w_32)

    # Gaussian curvature K
    gaussian_k = (r * t - s * s) / (w * w)

    # Laplacian (does not depend on gradient direction)
    laplacian = r + t

    # Horizontal (plan) curvature kh  and  Vertical (profile) curvature kv
    with np.errstate(divide="ignore", invalid="ignore"):
        kh = np.where(
            p2q2 > GRAD_EPSILON,
            -(q2 * r - 2.0 * p * q * s + p2 * t) / (p2q2 * w_sqrt),
            np.nan,
        )
        kv = np.where(
            p2q2 > GRAD_EPSILON,
            -(p2 * r + 2.0 * p * q * s + q2 * t) / (p2q2 * w_32),
            np.nan,
        )

    # Unsphericity M = sqrt(H^2 - K), clamped to 0 if H^2 < K
    disc = mean_h * mean_h - gaussian_k
    unsphericity_m = np.sqrt(np.maximum(disc, 0.0))

    # Difference curvature E = (kv - kh) / 2
    difference_e = np.where(
        p2q2 > GRAD_EPSILON,
        (kv - kh) / 2.0,
        np.nan,
    )

    # Principal curvatures
    minimal_kmin = mean_h - unsphericity_m
    maximal_kmax = mean_h + unsphericity_m

    # Excess curvatures: khe = M - E, kve = M + E
    horizontal_excess_khe = np.where(
        p2q2 > GRAD_EPSILON,
        unsphericity_m - difference_e,
        np.nan,
    )
    vertical_excess_kve = np.where(
        p2q2 > GRAD_EPSILON,
        unsphericity_m + difference_e,
        np.nan,
    )

    # Accumulation curvature Ka = kh * kv
    accumulation_ka = np.where(
        p2q2 > GRAD_EPSILON,
        kh * kv,
        np.nan,
    )

    # Ring curvature Kr = M^2 - E^2
    ring_kr = np.where(
        p2q2 > GRAD_EPSILON,
        unsphericity_m ** 2 - difference_e ** 2,
        np.nan,
    )

    # Rotor
    with np.errstate(divide="ignore", invalid="ignore"):
        rotor = np.where(
            p2q2 > GRAD_EPSILON,
            ((p2 - q2) * s - p * q * (r - t)) / (p2q2 ** 1.5),
            np.nan,
        )

    return {
        "mean_h": mean_h,
        "gaussian_k": gaussian_k,
        "unsphericity_m": unsphericity_m,
        "difference_e": difference_e,
        "minimal_kmin": minimal_kmin,
        "maximal_kmax": maximal_kmax,
        "horizontal_kh": kh,
        "vertical_kv": kv,
        "horizontal_excess_khe": horizontal_excess_khe,
        "vertical_excess_kve": vertical_excess_kve,
        "accumulation_ka": accumulation_ka,
        "ring_kr": ring_kr,
        "rotor": rotor,
        "laplacian": laplacian,
    }


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(computed, reference, mask):
    """Compute RMSE, MAE, R-squared, and max absolute error."""
    valid = mask & np.isfinite(computed) & np.isfinite(reference)
    n = np.sum(valid)

    if n == 0:
        return {
            "rmse": np.nan, "mae": np.nan, "r2": np.nan,
            "max_error": np.nan, "n_valid_pixels": 0,
        }

    c = computed[valid]
    r = reference[valid]
    diff = c - r

    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    max_error = np.max(np.abs(diff))

    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((r - np.mean(r)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "rmse": rmse, "mae": mae, "r2": r2,
        "max_error": max_error, "n_valid_pixels": int(n),
    }


# ============================================================================
# GeoTIFF I/O (for SAGA cross-validation)
# ============================================================================

def write_geotiff(path, data, cell_size, origin_x=0.0, origin_y=None):
    """Write a 2D float64 array as a GeoTIFF for use with SAGA."""
    if not HAS_GDAL:
        raise RuntimeError("GDAL/osgeo not available, cannot write GeoTIFF")

    rows, cols = data.shape
    if origin_y is None:
        origin_y = rows * cell_size

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), cols, rows, 1, gdal.GDT_Float64,
                       options=["COMPRESS=NONE"])

    gt = (origin_x, cell_size, 0.0, origin_y, 0.0, -cell_size)
    ds.SetGeoTransform(gt)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32719)
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    band.FlushCache()
    ds = None


def read_geotiff(path):
    """Read a single-band GeoTIFF as float64 array."""
    if not HAS_GDAL:
        raise RuntimeError("GDAL/osgeo not available")
    ds = gdal.Open(str(path))
    if ds is None:
        return None
    data = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    ds = None
    if nodata is not None:
        data[data == nodata] = np.nan
    return data


# ============================================================================
# SAGA GIS Cross-Validation
# ============================================================================

def run_saga_curvatures(dem_path, tmpdir):
    """Run SAGA ta_morphometry 0 on a DEM.

    Returns:
        dict mapping SAGA output name -> numpy array, or None on failure
    """
    if not os.path.isfile(SAGA_CMD):
        print(f"  SAGA not found at {SAGA_CMD}, skipping")
        return None

    # Import GeoTIFF to SAGA grid
    saga_dem = os.path.join(tmpdir, "dem_saga")
    cmd_import = [
        SAGA_CMD, "io_gdal", "0",
        "-GRIDS", saga_dem,
        "-FILES", dem_path,
    ]
    result = subprocess.run(cmd_import, capture_output=True, text=True, timeout=120, env=SAGA_ENV)
    if result.returncode != 0:
        print(f"  SAGA import failed: {result.stderr[:300]}")
        return None

    # Find the .sgrd file
    saga_dem_sgrd = saga_dem + ".sgrd"
    if not os.path.isfile(saga_dem_sgrd):
        import glob
        candidates = glob.glob(saga_dem + "*.sgrd")
        if candidates:
            saga_dem_sgrd = candidates[0]
        else:
            print("  SAGA import: .sgrd file not found")
            return None

    # Define output paths using correct SAGA parameter names
    outputs = {}
    saga_outputs = {}
    for name in SAGA_TO_SURTGIS.keys():
        outpath = os.path.join(tmpdir, f"saga_{name.lower()}")
        outputs[name] = outpath
        saga_outputs[name] = outpath + ".sgrd"

    # Run SAGA ta_morphometry 0 with Evans (1979) method (METHOD=3)
    # which matches SurtGIS's Evans-Young derivative scheme
    cmd_curv = [
        SAGA_CMD, "ta_morphometry", "0",
        "-ELEVATION", saga_dem_sgrd,
        "-UNIT_SLOPE", "0",      # radians
        "-UNIT_ASPECT", "0",     # radians
        "-METHOD", "3",          # Evans 1979 (= Evans-Young)
    ]
    for name, path in outputs.items():
        cmd_curv.extend([f"-{name}", path])

    result = subprocess.run(cmd_curv, capture_output=True, text=True, timeout=300, env=SAGA_ENV)
    if result.returncode != 0:
        print(f"  SAGA curvature (METHOD=6) failed, trying default...")
        cmd_curv_alt = [
            SAGA_CMD, "ta_morphometry", "0",
            "-ELEVATION", saga_dem_sgrd,
            "-UNIT_SLOPE", "0",
            "-UNIT_ASPECT", "0",
        ]
        for name, path in outputs.items():
            cmd_curv_alt.extend([f"-{name}", path])
        result = subprocess.run(cmd_curv_alt, capture_output=True, text=True, timeout=300, env=SAGA_ENV)
        if result.returncode != 0:
            print(f"  SAGA curvature (default) also failed: {result.stderr[:500]}")
            return None

    # Export .sgrd -> GeoTIFF and read back
    saga_results = {}
    for name, sgrd_path in saga_outputs.items():
        if not os.path.isfile(sgrd_path):
            continue

        tif_path = os.path.join(tmpdir, f"saga_{name.lower()}.tif")
        cmd_export = [
            SAGA_CMD, "io_gdal", "2",
            "-GRIDS", sgrd_path,
            "-FILE", tif_path,
        ]
        result = subprocess.run(cmd_export, capture_output=True, text=True, timeout=60, env=SAGA_ENV)
        if result.returncode == 0 and os.path.isfile(tif_path):
            data = read_geotiff(tif_path)
            if data is not None:
                saga_results[name] = data

    if not saga_results:
        print("  SAGA: No output curvatures could be read")
        return None

    return saga_results


# ============================================================================
# Validation Engine
# ============================================================================

def build_valid_mask(dem, x_grid, y_grid, curvature_type):
    """Build a boolean mask of valid pixels for comparison."""
    rows, cols = dem.shape
    mask = np.ones((rows, cols), dtype=bool)

    mask[:BORDER, :] = False
    mask[-BORDER:, :] = False
    mask[:, :BORDER] = False
    mask[:, -BORDER:] = False

    if curvature_type in GRADIENT_DEPENDENT:
        p, q, _, _, _ = analytical_partial_derivatives(x_grid, y_grid)
        p2q2 = p * p + q * q
        mask &= (p2q2 > GRAD_EPSILON)

    return mask


def validate_surtgis_vs_analytical(dem, x_grid, y_grid, analytical):
    """Compare SurtGIS curvatures against analytical solutions."""
    results = []

    print("\n" + "=" * 72)
    print("PART 1: SurtGIS vs ANALYTICAL (Gaussian hill)")
    print("=" * 72)
    print(f"  Surface: z = {A} * exp(-(x^2+y^2) / (2*{SIGMA}^2))")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}, cell_size={CELL_SIZE} m")
    print(f"  Border exclusion: {BORDER} px")
    print(f"  Gradient threshold: p^2+q^2 > {GRAD_EPSILON}")

    header = f"{'Curvature':<24} {'RMSE':<14} {'MAE':<14} {'R^2':<12} {'MaxErr':<14} {'N_valid':<10}"
    print(f"\n  {header}")
    print(f"  {'=' * len(header)}")

    for ctype in ALL_CURVATURE_TYPES:
        t0 = time.time()
        surtgis_result = surtgis.advanced_curvature(dem, CELL_SIZE, ctype=ctype)
        t_elapsed = time.time() - t0

        ref = analytical[ctype]
        mask = build_valid_mask(dem, x_grid, y_grid, ctype)
        m = compute_metrics(surtgis_result, ref, mask)
        m["curvature_type"] = ctype
        m["validation_method"] = "analytical"
        m["time_s"] = t_elapsed
        results.append(m)

        rmse_str = f"{m['rmse']:.6e}" if np.isfinite(m['rmse']) else "N/A"
        mae_str = f"{m['mae']:.6e}" if np.isfinite(m['mae']) else "N/A"
        r2_str = f"{m['r2']:.8f}" if np.isfinite(m['r2']) else "N/A"
        maxe_str = f"{m['max_error']:.6e}" if np.isfinite(m['max_error']) else "N/A"
        print(f"  {ctype:<24} {rmse_str:<14} {mae_str:<14} {r2_str:<12} {maxe_str:<14} {m['n_valid_pixels']:<10}")

    return results


def validate_surtgis_vs_saga(dem, x_grid, y_grid):
    """Compare SurtGIS curvatures against SAGA GIS."""
    results = []

    print("\n" + "=" * 72)
    print("PART 2: SurtGIS vs SAGA GIS")
    print("=" * 72)

    if not HAS_GDAL:
        print("  GDAL not available - cannot write GeoTIFF for SAGA. Skipping.")
        return results

    if not os.path.isfile(SAGA_CMD):
        print(f"  SAGA not found at {SAGA_CMD}. Skipping.")
        return results

    with tempfile.TemporaryDirectory(prefix="surtgis_curv_") as tmpdir:
        dem_tif = os.path.join(tmpdir, "gaussian_hill.tif")
        print(f"  Writing DEM to {dem_tif}")
        origin_x = -(GRID_SIZE // 2) * CELL_SIZE
        origin_y = (GRID_SIZE // 2) * CELL_SIZE
        write_geotiff(dem_tif, dem, CELL_SIZE, origin_x=origin_x, origin_y=origin_y)

        print("  Running SAGA ta_morphometry 0...")
        t0 = time.time()
        saga_results = run_saga_curvatures(dem_tif, tmpdir)
        t_saga = time.time() - t0
        print(f"  SAGA completed in {t_saga:.2f}s")

        if saga_results is None:
            print("  SAGA failed, no cross-validation results.")
            return results

        print(f"  SAGA returned curvatures: {list(saga_results.keys())}")

        header = f"{'Curvature':<24} {'SAGA_name':<14} {'RMSE':<14} {'MAE':<14} {'R^2':<12} {'MaxErr':<14} {'N_valid':<10}"
        print(f"\n  {header}")
        print(f"  {'=' * len(header)}")

        for saga_name, surtgis_name in SAGA_TO_SURTGIS.items():
            if saga_name not in saga_results:
                print(f"  {surtgis_name:<24} {saga_name:<14} --- SAGA output not available ---")
                continue

            saga_data = saga_results[saga_name]

            if saga_data.shape != dem.shape:
                print(f"  {surtgis_name:<24} {saga_name:<14} --- Shape mismatch ---")
                continue

            surtgis_result = surtgis.advanced_curvature(dem, CELL_SIZE, ctype=surtgis_name)

            mask = build_valid_mask(dem, x_grid, y_grid, surtgis_name)
            mask &= np.isfinite(saga_data)

            # Try both sign conventions (SAGA may negate some curvatures)
            m_direct = compute_metrics(surtgis_result, saga_data, mask)
            m_negated = compute_metrics(surtgis_result, -saga_data, mask)

            if (np.isfinite(m_negated["rmse"]) and np.isfinite(m_direct["rmse"])
                    and m_negated["rmse"] < m_direct["rmse"] * 0.5):
                m = m_negated
                sign_note = " (SAGA negated)"
            else:
                m = m_direct
                sign_note = ""

            m["curvature_type"] = surtgis_name
            m["saga_name"] = saga_name
            m["validation_method"] = "saga"
            m["sign_note"] = sign_note
            results.append(m)

            rmse_str = f"{m['rmse']:.6e}" if np.isfinite(m['rmse']) else "N/A"
            mae_str = f"{m['mae']:.6e}" if np.isfinite(m['mae']) else "N/A"
            r2_str = f"{m['r2']:.8f}" if np.isfinite(m['r2']) else "N/A"
            maxe_str = f"{m['max_error']:.6e}" if np.isfinite(m['max_error']) else "N/A"
            print(f"  {surtgis_name:<24} {saga_name:<14} {rmse_str:<14} {mae_str:<14} "
                  f"{r2_str:<12} {maxe_str:<14} {m['n_valid_pixels']:<10}{sign_note}")

    return results


# ============================================================================
# Internal Consistency Checks (Florinsky Theorems)
# ============================================================================

def check_florinsky_identities(dem):
    """Verify Florinsky's algebraic identities between curvature types."""
    print("\n" + "=" * 72)
    print("PART 3: FLORINSKY INTERNAL CONSISTENCY CHECKS")
    print("=" * 72)

    curvs = {}
    for ctype in ALL_CURVATURE_TYPES:
        curvs[ctype] = surtgis.advanced_curvature(dem, CELL_SIZE, ctype=ctype)

    rows, cols = dem.shape
    mask = np.ones((rows, cols), dtype=bool)
    mask[:BORDER, :] = False
    mask[-BORDER:, :] = False
    mask[:, :BORDER] = False
    mask[:, -BORDER:] = False
    for ctype in ALL_CURVATURE_TYPES:
        mask &= np.isfinite(curvs[ctype])

    n_valid = np.sum(mask)
    print(f"  Valid interior pixels: {n_valid}")

    identities = [
        ("kmin = H - M",
         curvs["minimal_kmin"],
         curvs["mean_h"] - curvs["unsphericity_m"]),

        ("kmax = H + M",
         curvs["maximal_kmax"],
         curvs["mean_h"] + curvs["unsphericity_m"]),

        ("khe = M - E",
         curvs["horizontal_excess_khe"],
         curvs["unsphericity_m"] - curvs["difference_e"]),

        ("kve = M + E",
         curvs["vertical_excess_kve"],
         curvs["unsphericity_m"] + curvs["difference_e"]),

        ("K = H^2 - M^2",
         curvs["gaussian_k"],
         curvs["mean_h"] ** 2 - curvs["unsphericity_m"] ** 2),

        ("Ka = kh * kv",
         curvs["accumulation_ka"],
         curvs["horizontal_kh"] * curvs["vertical_kv"]),

        ("Kr = M^2 - E^2",
         curvs["ring_kr"],
         curvs["unsphericity_m"] ** 2 - curvs["difference_e"] ** 2),

        ("Ka = K + Kr",
         curvs["accumulation_ka"],
         curvs["gaussian_k"] + curvs["ring_kr"]),
    ]

    header = f"  {'Identity':<24} {'MaxAbsDiff':<14} {'RMSE':<14} {'Status':<10}"
    print(f"\n{header}")
    print(f"  {'=' * (len(header) - 2)}")

    all_pass = True
    for name, left, right in identities:
        diff = left - right
        valid = mask & np.isfinite(diff)
        if np.sum(valid) == 0:
            print(f"  {name:<24} {'N/A':<14} {'N/A':<14} {'SKIP':<10}")
            continue

        d = diff[valid]
        max_abs = np.max(np.abs(d))
        rmse = np.sqrt(np.mean(d ** 2))
        status = "PASS" if max_abs < 1e-10 else ("WARN" if max_abs < 1e-6 else "FAIL")
        if status == "FAIL":
            all_pass = False
        print(f"  {name:<24} {max_abs:<14.6e} {rmse:<14.6e} {status:<10}")

    if all_pass:
        print("\n  All Florinsky identities verified successfully.")
    else:
        print("\n  WARNING: Some identities showed deviations beyond tolerance.")


# ============================================================================
# Output
# ============================================================================

def save_csv(analytical_results, saga_results, csv_path):
    """Save all results to CSV."""
    fieldnames = [
        "curvature_type", "rmse", "mae", "r2", "max_error",
        "n_valid_pixels", "validation_method",
    ]

    all_rows = []
    for m in analytical_results + saga_results:
        row = {k: m.get(k, "") for k in fieldnames}
        all_rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\n  CSV saved to: {csv_path}")


def print_summary_table(analytical_results, saga_results):
    """Print a nicely formatted summary table to stdout."""
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)

    all_results = analytical_results + saga_results

    header = (f"{'Curvature':<24} {'Method':<12} {'RMSE':<14} "
              f"{'MAE':<14} {'R^2':<12} {'MaxErr':<14} {'N':<10}")
    print(header)
    print("-" * len(header))

    for m in all_results:
        ctype = m.get("curvature_type", "?")
        method = m.get("validation_method", "?")
        rmse_str = f"{m['rmse']:.6e}" if np.isfinite(m.get('rmse', np.nan)) else "N/A"
        mae_str = f"{m['mae']:.6e}" if np.isfinite(m.get('mae', np.nan)) else "N/A"
        r2_str = f"{m['r2']:.8f}" if np.isfinite(m.get('r2', np.nan)) else "N/A"
        maxe_str = f"{m['max_error']:.6e}" if np.isfinite(m.get('max_error', np.nan)) else "N/A"
        n_str = f"{m.get('n_valid_pixels', 0)}"
        print(f"{ctype:<24} {method:<12} {rmse_str:<14} {mae_str:<14} "
              f"{r2_str:<12} {maxe_str:<14} {n_str:<10}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("SURTGIS 14-CURVATURE VALIDATION")
    print("Gaussian Hill: z = A*exp(-(x^2+y^2)/(2*sigma^2))")
    print(f"A={A}, sigma={SIGMA}, cell_size={CELL_SIZE}, grid={GRID_SIZE}x{GRID_SIZE}")
    print("=" * 72)

    # Step 1: Generate Gaussian hill DEM
    print("\nGenerating Gaussian hill DEM...")
    t0 = time.time()
    dem, x_grid, y_grid = generate_gaussian_hill()
    print(f"  Done in {time.time()-t0:.3f}s")
    print(f"  Shape: {dem.shape}")
    print(f"  Elevation range: {dem.min():.4f} to {dem.max():.4f} m")
    print(f"  Peak at center: z({GRID_SIZE//2},{GRID_SIZE//2}) = {dem[GRID_SIZE//2, GRID_SIZE//2]:.4f} m")

    # Step 2: Compute analytical curvatures
    print("\nComputing analytical curvatures...")
    t0 = time.time()
    analytical = analytical_curvatures(x_grid, y_grid)
    print(f"  Done in {time.time()-t0:.3f}s")
    for ctype in ALL_CURVATURE_TYPES:
        arr = analytical[ctype]
        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            print(f"  {ctype:<28} range: [{finite.min():.6e}, {finite.max():.6e}]")
        else:
            print(f"  {ctype:<28} all NaN")

    # Step 3: Validate SurtGIS vs analytical
    analytical_results = validate_surtgis_vs_analytical(dem, x_grid, y_grid, analytical)

    # Step 4: Internal consistency checks
    check_florinsky_identities(dem)

    # Step 5: Cross-validate against SAGA GIS
    saga_results = validate_surtgis_vs_saga(dem, x_grid, y_grid)

    # Step 6: Summary and CSV output
    print_summary_table(analytical_results, saga_results)

    csv_path = RESULTS / "curvature_validation.csv"
    save_csv(analytical_results, saga_results, csv_path)

    # Final verdict
    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)

    n_good = sum(1 for m in analytical_results
                 if np.isfinite(m.get("r2", np.nan)) and m["r2"] > 0.99)
    n_total = len(analytical_results)
    print(f"  Analytical validation: {n_good}/{n_total} curvatures with R^2 > 0.99")

    n_saga = len(saga_results)
    if n_saga > 0:
        n_saga_good = sum(1 for m in saga_results
                          if np.isfinite(m.get("r2", np.nan)) and m["r2"] > 0.99)
        print(f"  SAGA cross-validation: {n_saga_good}/{n_saga} curvatures with R^2 > 0.99")
    else:
        print("  SAGA cross-validation: not performed (SAGA or GDAL unavailable)")

    print(f"\n  NOTE: Small RMSE values are expected due to the difference between")
    print(f"  exact analytical derivatives and finite-difference (Evans-Young 3x3)")
    print(f"  discrete approximations used by SurtGIS. These errors decrease with")
    print(f"  finer cell size relative to the surface's characteristic length scale.")
    print(f"  For sigma={SIGMA}m and cell_size={CELL_SIZE}m, the ratio is {SIGMA/CELL_SIZE:.0f}:1,")
    print(f"  which should give very accurate finite-difference approximations.")

    print("\nDone.")


if __name__ == "__main__":
    main()
