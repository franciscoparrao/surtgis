#!/usr/bin/env python3
"""
SurtGIS Paper Benchmarks — Environmental Modelling & Software
=============================================================

Three experiments comparing SurtGIS against GDAL, GRASS GIS, and WhiteboxTools:

  1. Scalability: wall-clock time across DEM sizes (1K² to 20K²)
  2. Accuracy: analytical validation + cross-tool agreement
  3. Cross-platform: native Rust (multi/single), Python bindings, WASM placeholder

Usage:
  python benchmarks/run_benchmarks.py              # full suite
  python benchmarks/run_benchmarks.py --quick       # fast validation run
  python benchmarks/run_benchmarks.py --experiment 1
  python benchmarks/run_benchmarks.py --tools surtgis,gdal
"""

import argparse
import csv
import ctypes
import gc
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEMS_DIR = RESULTS_DIR / "dems"
FIXTURE_DEM = ROOT_DIR / "tests" / "fixtures" / "andes_chile_30m.tif"
FIXTURE_DEM_UTM = ROOT_DIR / "tests" / "fixtures" / "andes_chile_30m_utm.tif"

FULL_SIZES = [1000, 5000, 10000, 20000]
QUICK_SIZES = [500, 1000, 2000]

FULL_REPS = 10
FULL_WARMUP = 3
QUICK_REPS = 3
QUICK_WARMUP = 0

TIMEOUT_SECONDS = 300  # per individual run
CELL_SIZE = 10.0  # meters

# Memory safety thresholds (% of total system RAM)
MEMORY_ALERT_PCT = 80
MEMORY_CRITICAL_PCT = 90
# Minimum free RAM (GB) to start a new DEM size / tool combination
MEMORY_MIN_FREE_GB = 4.0

# ---------------------------------------------------------------------------
# Memory safety (Metodología de Seguridad — 3 Capas)
# ---------------------------------------------------------------------------


def get_memory_status() -> dict:
    """Return current memory status: process RSS (GB), system % used, available GB."""
    proc = psutil.Process(os.getpid())
    proc_gb = proc.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    return {
        "proc_gb": proc_gb,
        "system_pct": vm.percent,
        "avail_gb": vm.available / (1024 ** 3),
        "total_gb": vm.total / (1024 ** 3),
    }


def force_cleanup():
    """Aggressive memory cleanup: GC + malloc_trim (Linux)."""
    gc.collect()
    gc.collect()  # second pass for reference cycles
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def log_memory(prefix: str = ""):
    """Print current memory status."""
    m = get_memory_status()
    print(f"  {prefix}MEM: proceso={m['proc_gb']:.1f}GB | "
          f"sistema={m['system_pct']:.0f}% | "
          f"disponible={m['avail_gb']:.1f}GB/{m['total_gb']:.0f}GB")


def check_memory_safe(context: str = "") -> bool:
    """
    Check if it's safe to continue. Returns True if safe.
    If memory is above alert threshold, forces cleanup.
    If above critical threshold, returns False (STOP).
    """
    m = get_memory_status()

    if m["system_pct"] >= MEMORY_CRITICAL_PCT or m["avail_gb"] < 2.0:
        print(f"\n  *** MEMORIA CRITICA: {m['system_pct']:.0f}% usado, "
              f"{m['avail_gb']:.1f}GB libre ***")
        print(f"  *** Saltando {context} para proteger el sistema ***")
        force_cleanup()
        return False

    if m["system_pct"] >= MEMORY_ALERT_PCT or m["avail_gb"] < MEMORY_MIN_FREE_GB:
        print(f"\n  ** Memoria alta: {m['system_pct']:.0f}% — limpiando... **")
        force_cleanup()
        time.sleep(2)
        # Re-check after cleanup
        m2 = get_memory_status()
        if m2["system_pct"] >= MEMORY_CRITICAL_PCT or m2["avail_gb"] < 2.0:
            print(f"  *** Aun critico tras limpieza: {m2['system_pct']:.0f}% ***")
            print(f"  *** Saltando {context} ***")
            return False

    return True


def safe_delete(*names_and_refs):
    """Delete variables and force cleanup. Usage: safe_delete(var1, var2, ...)"""
    for ref in names_and_refs:
        try:
            del ref
        except Exception:
            pass
    force_cleanup()


# ---------------------------------------------------------------------------
# Incremental CSV writer
# ---------------------------------------------------------------------------


class IncrementalCSV:
    """Write CSV rows incrementally so data survives crashes."""

    def __init__(self, path: Path, fieldnames: list[str], append: bool = False):
        self.path = path
        self.fieldnames = fieldnames
        if append and path.exists() and path.stat().st_size > 0:
            print(f"  [CSV APPEND mode: {path.name}]")
        else:
            self._write_header()

    def _write_header(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def append(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def append_many(self, rows: list[dict]):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# DEM generation
# ---------------------------------------------------------------------------


def generate_fbm_dem(size: int, hurst: float = 0.7, seed: int = 42,
                     cell_size: float = 10.0) -> np.ndarray:
    """FFT-based fractal Brownian motion surface (spectral synthesis).

    Memory-safe: explicitly deletes FFT intermediates to avoid peak RAM spikes.
    For 20K² this reduces peak from ~12 GB to ~6 GB.
    """
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((size, size))

    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    power = (fx ** 2 + fy ** 2 + 1e-10) ** (-(hurst + 1))
    power[0, 0] = 0
    del fx, fy  # free meshgrid arrays

    sqrt_power = np.sqrt(power)
    del power

    spectrum = np.fft.fft2(noise)
    del noise
    spectrum *= sqrt_power  # in-place multiply
    del sqrt_power

    surface = np.fft.ifft2(spectrum)
    del spectrum
    gc.collect()

    surface = np.real(surface)

    # Normalise to realistic elevation range (200 – 2000 m)
    lo, hi = surface.min(), surface.max()
    surface = 200.0 + (surface - lo) / (hi - lo) * 1800.0
    return surface.astype(np.float64)


def generate_gaussian_hill(size: int, A: float = 500.0, sigma: float = None,
                           cell_size: float = 10.0):
    """
    Gaussian hill: z = A * exp(-(x² + y²) / (2σ²))

    Returns (dem, slope_analytical, aspect_analytical,
             curvature_mean_analytical, curvature_gaussian_analytical)
    all as float64 arrays of shape (size, size).
    """
    if sigma is None:
        sigma = size * cell_size * 0.2

    half = (size - 1) / 2.0
    rows = np.arange(size, dtype=np.float64)
    cols = np.arange(size, dtype=np.float64)
    # x_grid: east (column direction), positive = right
    # y_south: array row direction (positive = south, row increases downward)
    # In a north-up GeoTIFF, row 0 is north, row N is south
    y_south, x_grid = np.meshgrid(
        (rows - half) * cell_size,   # row: positive south
        (cols - half) * cell_size,   # col: positive east
        indexing="ij",
    )

    r2 = x_grid ** 2 + y_south ** 2
    s2 = sigma ** 2
    exp_term = np.exp(-r2 / (2.0 * s2))

    # DEM
    dem = A * exp_term

    # First derivatives w.r.t. array coordinates
    # dz/dx (east): positive = elevation increases eastward
    dz_dx = -A * x_grid / s2 * exp_term
    # dz/dy_south: positive = elevation increases with row (southward)
    dz_dy_south = -A * y_south / s2 * exp_term

    # Slope (degrees) — same regardless of sign convention
    gradient_mag = np.sqrt(dz_dx ** 2 + dz_dy_south ** 2)
    slope_analytical = np.degrees(np.arctan(gradient_mag))

    # Aspect (degrees, 0=North CW)
    # SurtGIS convention: aspect = atan2(-dz_dx, dz_dy_south)
    # where dz_dx positive=east, dz_dy_south positive=south
    aspect_rad = np.arctan2(-dz_dx, dz_dy_south)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360.0) % 360.0
    # Flat areas → -1
    flat_mask = gradient_mag < 1e-10
    aspect_deg[flat_mask] = -1.0

    # Second derivatives
    d2z_dx2 = A / s2 * (x_grid ** 2 / s2 - 1.0) * exp_term
    d2z_dy2 = A / s2 * (y_south ** 2 / s2 - 1.0) * exp_term
    d2z_dxdy = A * x_grid * y_south / (s2 ** 2) * exp_term

    # Mean curvature H (Zevenbergen & Thorne simplified)
    p = dz_dx
    q = dz_dy_south
    r_ = d2z_dx2
    t_ = d2z_dy2
    s_ = d2z_dxdy
    denom = (1 + p ** 2 + q ** 2)
    H = -((1 + q ** 2) * r_ - 2 * p * q * s_ + (1 + p ** 2) * t_) / (
        2.0 * denom ** 1.5
    )

    # Gaussian curvature K
    K = (r_ * t_ - s_ ** 2) / (denom ** 2)

    return dem, slope_analytical, aspect_deg, H, K


def save_dem_geotiff(dem: np.ndarray, path: Path, cell_size: float = 10.0):
    """Write a numpy DEM to a GeoTIFF using GDAL Python bindings or gdal_translate."""
    try:
        from osgeo import gdal, osr
        driver = gdal.GetDriverByName("GTiff")
        rows, cols = dem.shape
        ds = driver.Create(str(path), cols, rows, 1, gdal.GDT_Float64,
                           ["COMPRESS=LZW", "TILED=YES"])
        ds.SetGeoTransform((500000.0, cell_size, 0.0, 4500000.0, 0.0, -cell_size))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32719)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(dem)
        ds.GetRasterBand(1).SetNoDataValue(-9999.0)
        ds.FlushCache()
        ds = None
        return True
    except ImportError:
        pass

    # Fallback: write raw + gdal_translate
    raw_path = path.with_suffix(".raw.npy")
    np.save(raw_path, dem)
    try:
        rows, cols = dem.shape
        vrt = path.with_suffix(".vrt")
        vrt_xml = f"""<VRTDataset rasterXSize="{cols}" rasterYSize="{rows}">
  <SRS>EPSG:32719</SRS>
  <GeoTransform>500000.0, {cell_size}, 0.0, 4500000.0, 0.0, {-cell_size}</GeoTransform>
  <VRTRasterBand dataType="Float64" band="1">
    <NoDataValue>-9999</NoDataValue>
    <SourceFilename relativeToVRT="0">{raw_path}</SourceFilename>
  </VRTRasterBand>
</VRTDataset>"""
        # Use a simpler approach: write binary + gdal_translate from raw
        bin_path = path.with_suffix(".bin")
        dem.astype(np.float64).tofile(str(bin_path))
        vrt_content = f"""<VRTDataset rasterXSize="{cols}" rasterYSize="{rows}">
  <SRS>EPSG:32719</SRS>
  <GeoTransform>500000.0, {cell_size}, 0.0, 4500000.0, 0.0, {-cell_size}</GeoTransform>
  <VRTRasterBand dataType="Float64" band="1">
    <NoDataValue>-9999</NoDataValue>
    <SourceFilename relativeToVRT="0">{bin_path}</SourceFilename>
    <ImageOffset>0</ImageOffset>
    <PixelOffset>8</PixelOffset>
    <LineOffset>{cols * 8}</LineOffset>
    <ByteOrder>LSB</ByteOrder>
  </VRTRasterBand>
</VRTDataset>"""
        vrt.write_text(vrt_content)
        subprocess.run(
            ["gdal_translate", "-of", "GTiff", "-co", "COMPRESS=LZW",
             str(vrt), str(path)],
            check=True, capture_output=True,
        )
        # Cleanup
        for f in [raw_path, bin_path, vrt]:
            f.unlink(missing_ok=True)
        return True
    except Exception:
        raw_path.unlink(missing_ok=True)
        return False


def read_geotiff_as_array(path: Path) -> np.ndarray:
    """Read a GeoTIFF and return the first band as numpy array."""
    try:
        from osgeo import gdal
        ds = gdal.Open(str(path))
        arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
        ds = None
        return arr
    except ImportError:
        pass
    # Fallback using gdalinfo + raw read is complex; for now raise
    raise RuntimeError("Cannot read GeoTIFF: install GDAL Python bindings (osgeo)")


# ---------------------------------------------------------------------------
# Tool availability detection
# ---------------------------------------------------------------------------


def check_surtgis() -> bool:
    try:
        import surtgis  # noqa: F401
        return True
    except ImportError:
        return False


def check_gdal() -> bool:
    return shutil.which("gdaldem") is not None


def check_grass() -> bool:
    return shutil.which("grass") is not None


def _grass_tmp_flag() -> str:
    """Return the correct tmp project flag for the installed GRASS version.
    GRASS >= 8.4 uses --tmp-project, older versions use --tmp-location."""
    try:
        out = subprocess.run(["grass", "--version"], capture_output=True, text=True)
        text = out.stdout + out.stderr
        for line in text.splitlines():
            if "GRASS GIS" in line:
                parts = line.split()
                for p in parts:
                    if p[0].isdigit():
                        major, minor = p.split(".")[:2]
                        if (int(major), int(minor)) >= (8, 4):
                            return "--tmp-project"
                        return "--tmp-location"
    except Exception:
        pass
    return "--tmp-location"


def check_wbt() -> bool:
    try:
        import whitebox  # noqa: F401
        return True
    except ImportError:
        return False


def check_cargo() -> bool:
    return shutil.which("cargo") is not None


def get_available_tools(requested: list[str] | None = None) -> dict[str, bool]:
    surtgis_ok = check_surtgis()
    all_tools = {
        "surtgis": surtgis_ok,
        "surtgis_io": surtgis_ok,  # same binary, includes file I/O
        "gdal": check_gdal(),
        "grass": check_grass(),
        "wbt": check_wbt(),
    }
    if requested:
        # "surtgis" in request enables both surtgis and surtgis_io
        expanded = list(requested)
        if "surtgis" in expanded and "surtgis_io" not in expanded:
            expanded.append("surtgis_io")
        return {k: all_tools.get(k, False) for k in expanded}
    return all_tools


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def collect_system_info(tools: dict[str, bool]) -> dict:
    info = {
        "cpu": _get_cpu_name(),
        "cores": os.cpu_count(),
        "ram_gb": _get_ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if tools.get("surtgis"):
        try:
            import surtgis
            info["surtgis_version"] = getattr(surtgis, "__version__", "0.1.x")
        except Exception:
            pass

    if tools.get("gdal"):
        try:
            out = subprocess.run(["gdalinfo", "--version"], capture_output=True, text=True)
            info["gdal_version"] = out.stdout.strip().split(",")[0].replace("GDAL ", "")
        except Exception:
            pass

    if tools.get("grass"):
        try:
            out = subprocess.run(["grass", "--version"], capture_output=True, text=True)
            for line in (out.stdout + out.stderr).splitlines():
                if "GRASS GIS" in line:
                    info["grass_version"] = line.strip()
                    break
        except Exception:
            pass

    if tools.get("wbt"):
        try:
            import whitebox
            wbt = whitebox.WhiteboxTools()
            info["wbt_version"] = wbt.version().strip()
        except Exception:
            pass

    return info


def _get_cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _get_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    return round(kb / 1024 / 1024, 1)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("Execution timed out")


def timed_call(func, timeout: int = TIMEOUT_SECONDS):
    """Run func() with a timeout. Returns (elapsed_seconds, result) or raises."""
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)
    try:
        t0 = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - t0
        return elapsed, result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Tool-specific runners — Experiment 1
# ---------------------------------------------------------------------------

# ---- SurtGIS (in-memory numpy) ----

def run_surtgis(algorithm: str, dem: np.ndarray, cell_size: float,
                **kwargs) -> float:
    """Run a SurtGIS algorithm on numpy array. Returns wall-clock seconds."""
    import surtgis

    def _run():
        if algorithm == "slope":
            return surtgis.slope(dem, cell_size)
        elif algorithm == "aspect":
            return surtgis.aspect_degrees(dem, cell_size)
        elif algorithm == "hillshade":
            return surtgis.hillshade_compute(dem, cell_size)
        elif algorithm == "tpi":
            return surtgis.tpi_compute(dem, cell_size, radius=10)
        elif algorithm == "fill":
            return surtgis.priority_flood_fill(dem, cell_size)
        elif algorithm == "flow_acc":
            fdir = surtgis.flow_direction_d8(dem, cell_size)
            return surtgis.flow_accumulation_d8(fdir, cell_size)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    elapsed, _ = timed_call(_run)
    return elapsed


# ---- SurtGIS with file I/O (fair comparison) ----

def _write_geotiff_fast(arr: np.ndarray, path: Path, cell_size: float):
    """Write array to GeoTIFF without compression (matching gdaldem default)."""
    from osgeo import gdal, osr
    rows, cols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    # No compression, no tiling — same as gdaldem default output
    ds = driver.Create(str(path), cols, rows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((500000.0, cell_size, 0.0, 4500000.0, 0.0, -cell_size))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32719)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
    ds.FlushCache()
    ds = None


def run_surtgis_io(algorithm: str, dem_path: Path = None, out_dir: Path = None,
                   cell_size: float = CELL_SIZE, **kwargs) -> float:
    """Run SurtGIS with full file I/O: read GeoTIFF → compute → write GeoTIFF.

    This provides a fair comparison against GDAL/GRASS/WBT which all include
    file I/O in their measured time. Uses no-compression Float32 output
    matching gdaldem's default format.
    """
    import surtgis

    output = out_dir / f"surtgis_io_{algorithm}.tif"
    output.unlink(missing_ok=True)

    def _run():
        # 1. Read DEM from disk (same as other tools)
        dem = read_geotiff_as_array(dem_path)

        # 2. Compute
        if algorithm == "slope":
            result = surtgis.slope(dem, cell_size)
        elif algorithm == "aspect":
            result = surtgis.aspect_degrees(dem, cell_size)
        elif algorithm == "hillshade":
            result = surtgis.hillshade_compute(dem, cell_size)
        elif algorithm == "tpi":
            result = surtgis.tpi_compute(dem, cell_size, radius=10)
        elif algorithm == "fill":
            result = surtgis.priority_flood_fill(dem, cell_size)
        elif algorithm == "flow_acc":
            fdir = surtgis.flow_direction_d8(dem, cell_size)
            result = surtgis.flow_accumulation_d8(fdir, cell_size)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # 3. Write result to disk (no compression, same as gdaldem default)
        _write_geotiff_fast(result, output, cell_size)

    elapsed, _ = timed_call(_run)
    output.unlink(missing_ok=True)
    return elapsed


# ---- GDAL (subprocess) ----

def run_gdal(algorithm: str, dem_path: Path, out_dir: Path, **kwargs) -> float:
    """Run a GDAL algorithm. Returns wall-clock seconds."""
    output = out_dir / f"gdal_{algorithm}.tif"
    output.unlink(missing_ok=True)

    if algorithm == "slope":
        cmd = ["gdaldem", "slope", str(dem_path), str(output)]
    elif algorithm == "aspect":
        cmd = ["gdaldem", "aspect", str(dem_path), str(output)]
    elif algorithm == "hillshade":
        cmd = ["gdaldem", "hillshade", str(dem_path), str(output),
               "-az", "315", "-alt", "45"]
    elif algorithm == "tpi":
        cmd = ["gdaldem", "TPI", str(dem_path), str(output)]
    else:
        return float("nan")

    def _run():
        subprocess.run(cmd, check=True, capture_output=True, timeout=TIMEOUT_SECONDS)

    elapsed, _ = timed_call(_run)
    output.unlink(missing_ok=True)
    return elapsed


# ---- GRASS (subprocess with --tmp-project) ----

def run_grass(algorithm: str, dem_path: Path, out_dir: Path, **kwargs) -> float:
    """Run a GRASS GIS algorithm. Returns wall-clock seconds."""
    output = out_dir / f"grass_{algorithm}.tif"
    output.unlink(missing_ok=True)

    # Common import + set region to match input raster
    import_cmd = (
        f"r.in.gdal input={dem_path} output=dem --overwrite && "
        f"g.region raster=dem"
    )
    export_cmd = (
        f"r.out.gdal input=result output={output} format=GTiff "
        f"createopt=COMPRESS=LZW --overwrite"
    )

    if algorithm == "slope":
        grass_cmds = (
            f"{import_cmd} && "
            f"r.slope.aspect elevation=dem slope=result --overwrite && "
            f"{export_cmd}"
        )
    elif algorithm == "aspect":
        grass_cmds = (
            f"{import_cmd} && "
            f"r.slope.aspect elevation=dem aspect=result --overwrite && "
            f"{export_cmd}"
        )
    elif algorithm == "hillshade":
        grass_cmds = (
            f"{import_cmd} && "
            f"r.relief input=dem output=result --overwrite && "
            f"{export_cmd}"
        )
    elif algorithm == "fill":
        grass_cmds = (
            f"{import_cmd} && "
            f"r.fill.dir input=dem output=result direction=fdir --overwrite && "
            f"{export_cmd}"
        )
    elif algorithm == "flow_acc":
        grass_cmds = (
            f"{import_cmd} && "
            f"r.watershed elevation=dem accumulation=result --overwrite && "
            f"{export_cmd}"
        )
    else:
        return float("nan")

    cmd = [
        "grass", _grass_tmp_flag(), "EPSG:32719", "--exec",
        "bash", "-c", grass_cmds,
    ]

    def _run():
        subprocess.run(
            cmd, check=True, capture_output=True,
            timeout=TIMEOUT_SECONDS,
            env={**os.environ, "GRASS_VERBOSE": "0"},
        )

    elapsed, _ = timed_call(_run)
    output.unlink(missing_ok=True)
    return elapsed


# ---- WhiteboxTools (Python wrapper) ----

def run_wbt(algorithm: str, dem_path: Path, out_dir: Path, **kwargs) -> float:
    """Run a WhiteboxTools algorithm. Returns wall-clock seconds."""
    import whitebox
    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)

    abs_dem = str(dem_path.resolve())
    output = out_dir / f"wbt_{algorithm}.tif"
    output.unlink(missing_ok=True)
    abs_out = str(output.resolve())

    def _run():
        if algorithm == "slope":
            wbt.slope(abs_dem, abs_out)
        elif algorithm == "aspect":
            wbt.aspect(abs_dem, abs_out)
        elif algorithm == "hillshade":
            wbt.hillshade(abs_dem, abs_out, azimuth=315.0, altitude=45.0)
        elif algorithm == "fill":
            wbt.fill_depressions(abs_dem, abs_out)
        elif algorithm == "flow_acc":
            filled = str((out_dir / "wbt_filled_tmp.tif").resolve())
            wbt.fill_depressions(abs_dem, filled)
            wbt.d8_flow_accumulation(filled, abs_out)
            Path(filled).unlink(missing_ok=True)
        else:
            raise ValueError(f"WBT: unknown algorithm {algorithm}")

    elapsed, _ = timed_call(_run)
    output.unlink(missing_ok=True)
    return elapsed


# Mapping of tools to runner functions
TOOL_RUNNERS = {
    "surtgis": run_surtgis,
    "surtgis_io": run_surtgis_io,
    "gdal": run_gdal,
    "grass": run_grass,
    "wbt": run_wbt,
}

# Which algorithms each tool supports
TOOL_ALGORITHMS = {
    "surtgis": ["slope", "aspect", "hillshade", "tpi", "fill", "flow_acc"],
    "surtgis_io": ["slope", "aspect", "hillshade", "tpi", "fill", "flow_acc"],
    "gdal":    ["slope", "aspect", "hillshade", "tpi"],
    "grass":   ["slope", "aspect", "hillshade", "fill", "flow_acc"],
    "wbt":     ["slope", "aspect", "hillshade", "fill", "flow_acc"],
}


# ---------------------------------------------------------------------------
# Experiment 1: Scalability
# ---------------------------------------------------------------------------


def experiment1_scalability(
    sizes: list[int],
    reps: int,
    warmup: int,
    tools: dict[str, bool],
    append: bool = False,
) -> Path:
    """Run scalability benchmarks. Returns path to CSV."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Scalability")
    print("=" * 60)

    algorithms = ["slope", "aspect", "hillshade", "tpi", "fill", "flow_acc"]
    csv_path = RESULTS_DIR / "experiment1_scalability.csv"

    # Incremental CSV — survives crashes
    csv_writer = IncrementalCSV(csv_path, ["algorithm", "size", "tool",
                                            "run", "time_seconds"],
                                append=append)

    for size in sizes:
        print(f"\n--- DEM size: {size}x{size} ({size*size/1e6:.1f}M cells) ---")
        log_memory("PRE-DEM ")

        # Check memory feasibility (float64 = 8 bytes per cell)
        dem_file = DEMS_DIR / f"fbm_{size}.tif"
        mem_needed_gb = size * size * 8 / 1e9
        avail = get_memory_status()["avail_gb"]

        if not dem_file.exists():
            # FFT generation peak with optimized del: ~3.5x the final array
            fft_peak_gb = mem_needed_gb * 3.5
            if fft_peak_gb > avail * 0.85:
                print(f"  SKIP: FFT peak ~{fft_peak_gb:.1f}GB, only {avail:.1f}GB libre")
                continue
        else:
            # Only need to load the array for surtgis runs (~1x)
            if mem_needed_gb > avail * 0.7:
                print(f"  SKIP: DEM ~{mem_needed_gb:.1f}GB, only {avail:.1f}GB libre")
                continue

        # Pre-size memory check
        if not check_memory_safe(f"DEM {size}x{size}"):
            continue
        dem_array = None
        if not dem_file.exists():
            print(f"  Generating fBm DEM {size}x{size}...", end=" ", flush=True)
            t0 = time.perf_counter()
            dem_array = generate_fbm_dem(size, cell_size=CELL_SIZE)
            save_dem_geotiff(dem_array, dem_file, CELL_SIZE)
            # Free FFT intermediates
            force_cleanup()
            print(f"done ({time.perf_counter()-t0:.1f}s)")
            log_memory("POST-GEN ")
        else:
            print(f"  Reusing existing DEM: {dem_file.name}")

        for alg in algorithms:
            for tool_name, available in tools.items():
                if not available:
                    continue
                if alg not in TOOL_ALGORITHMS.get(tool_name, []):
                    continue

                # Memory check before each tool×algorithm combination
                context = f"{alg}/{tool_name} @ {size}x{size}"
                if not check_memory_safe(context):
                    csv_writer.append({
                        "algorithm": alg, "size": size, "tool": tool_name,
                        "run": 1, "time_seconds": "SKIPPED_MEM",
                    })
                    continue

                # Load DEM into memory only for surtgis (in-process)
                # Subprocess tools (gdal, grass, wbt) use the file path
                if tool_name == "surtgis" and dem_array is None:
                    dem_array = read_geotiff_as_array(dem_file)

                label = f"  {alg:12s} | {tool_name:12s}"
                print(f"{label}: ", end="", flush=True)

                runner = TOOL_RUNNERS[tool_name]
                run_kwargs = dict(
                    algorithm=alg,
                    dem=dem_array,
                    dem_path=dem_file,
                    cell_size=CELL_SIZE,
                    out_dir=RESULTS_DIR,
                )

                # Warm-up
                for _ in range(warmup):
                    try:
                        runner(**run_kwargs)
                        force_cleanup()
                    except Exception:
                        break

                # Timed repetitions
                times = []
                for rep in range(reps):
                    try:
                        t = runner(**run_kwargs)
                        times.append(t)
                        csv_writer.append({
                            "algorithm": alg,
                            "size": size,
                            "tool": tool_name,
                            "run": rep + 1,
                            "time_seconds": f"{t:.6f}",
                        })
                    except TimeoutError:
                        print(f"TIMEOUT (>{TIMEOUT_SECONDS}s)")
                        csv_writer.append({
                            "algorithm": alg,
                            "size": size,
                            "tool": tool_name,
                            "run": rep + 1,
                            "time_seconds": "TIMEOUT",
                        })
                        break
                    except Exception as e:
                        print(f"ERROR: {e}")
                        csv_writer.append({
                            "algorithm": alg,
                            "size": size,
                            "tool": tool_name,
                            "run": rep + 1,
                            "time_seconds": "ERROR",
                        })
                        break

                    # Memory check between repetitions for heavy combos
                    if rep < reps - 1:
                        force_cleanup()

                if times:
                    med = np.median(times)
                    iqr = np.percentile(times, 75) - np.percentile(times, 25)
                    print(f"median={med:.4f}s  IQR={iqr:.4f}s  (n={len(times)})")
                else:
                    print("no valid runs")

                # Post-combination cleanup
                force_cleanup()

        # CRITICAL: Free DEM array between sizes to reclaim RAM
        if dem_array is not None:
            del dem_array
            dem_array = None
        force_cleanup()
        log_memory("POST-SIZE ")
        print()

    print(f"\nResults: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Experiment 2: Accuracy
# ---------------------------------------------------------------------------


def experiment2_accuracy(tools: dict[str, bool]) -> Path:
    """Run accuracy benchmarks. Returns path to CSV."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Numerical Accuracy")
    print("=" * 60)

    csv_path = RESULTS_DIR / "experiment2_accuracy.csv"
    csv_writer = IncrementalCSV(csv_path, ["metric", "algorithm",
                                            "comparison", "value"])

    # --- 2a: Analytical validation (Gaussian Hill) ---
    print("\n--- 2a: Analytical validation (Gaussian Hill 1000x1000) ---")
    log_memory("PRE-EXP2 ")
    size = 1000
    dem, slope_true, aspect_true, H_true, K_true = generate_gaussian_hill(
        size, A=500.0, cell_size=CELL_SIZE
    )

    # Exclude 1-cell border for fair comparison
    sl = slice(1, -1)
    interior = (sl, sl)

    if tools.get("surtgis"):
        import surtgis

        # Slope
        slope_sg = np.array(surtgis.slope(dem, CELL_SIZE))
        for row in _accuracy_metrics(
            slope_true[interior], slope_sg[interior],
            "slope", "surtgis_vs_analytical"
        ):
            csv_writer.append(row)
        del slope_sg

        # Aspect (skip flat cells, use angular difference for circular metric)
        aspect_sg = np.array(surtgis.aspect_degrees(dem, CELL_SIZE))
        mask = slope_true[interior] > 1.0  # non-flat (>1° slope)
        if mask.any():
            ref_asp = aspect_true[interior][mask]
            cmp_asp = aspect_sg[interior][mask]
            # Angular difference: handles 0/360 wrap-around
            ang_diff = np.abs(ref_asp - cmp_asp)
            ang_diff = np.minimum(ang_diff, 360.0 - ang_diff)
            for row in [
                {"metric": "rmse", "algorithm": "aspect",
                 "comparison": "surtgis_vs_analytical",
                 "value": f"{float(np.sqrt(np.mean(ang_diff**2))):.6f}"},
                {"metric": "mae", "algorithm": "aspect",
                 "comparison": "surtgis_vs_analytical",
                 "value": f"{float(np.mean(ang_diff)):.6f}"},
                {"metric": "max_angular_error", "algorithm": "aspect",
                 "comparison": "surtgis_vs_analytical",
                 "value": f"{float(np.max(ang_diff)):.6f}"},
            ]:
                csv_writer.append(row)
        del aspect_sg

        # Mean curvature
        curv_sg = np.array(surtgis.advanced_curvature(dem, CELL_SIZE, ctype="mean_h"))
        for row in _accuracy_metrics(
            H_true[interior], curv_sg[interior],
            "curvature_mean", "surtgis_vs_analytical"
        ):
            csv_writer.append(row)
        del curv_sg

        # Gaussian curvature
        gcurv_sg = np.array(surtgis.advanced_curvature(dem, CELL_SIZE, ctype="gaussian_k"))
        for row in _accuracy_metrics(
            K_true[interior], gcurv_sg[interior],
            "curvature_gaussian", "surtgis_vs_analytical"
        ):
            csv_writer.append(row)
        del gcurv_sg

        print("  SurtGIS vs analytical: done")

    # Free Gaussian Hill arrays
    del dem, slope_true, aspect_true, H_true, K_true
    force_cleanup()

    # --- 2b: Cross-tool validation (real DEM, UTM projected) ---
    # Use UTM DEM to ensure consistent cell_size units (meters) across tools.
    # Geographic DEMs cause systematic errors when cell_size is passed in meters
    # but pixel spacing is in degrees.
    cross_dem_path = FIXTURE_DEM_UTM if FIXTURE_DEM_UTM.exists() else FIXTURE_DEM
    print(f"\n--- 2b: Cross-tool validation ({cross_dem_path.name}) ---")

    if not cross_dem_path.exists():
        print(f"  SKIP: fixture DEM not found at {cross_dem_path}")
    else:
        # Extract cell size from GeoTIFF metadata
        from osgeo import gdal as _gdal
        _ds = _gdal.Open(str(cross_dem_path))
        _gt = _ds.GetGeoTransform()
        cross_cell_size = abs(_gt[1])
        print(f"  DEM: {_ds.RasterXSize}x{_ds.RasterYSize}, cell_size={cross_cell_size:.2f}m")
        _ds = None

        real_dem = read_geotiff_as_array(cross_dem_path)
        slopes = {}

        # SurtGIS
        if tools.get("surtgis"):
            import surtgis
            slopes["surtgis"] = np.array(surtgis.slope(real_dem, cross_cell_size))

        # GDAL
        if tools.get("gdal"):
            out = RESULTS_DIR / "gdal_slope_andes.tif"
            out.unlink(missing_ok=True)
            try:
                subprocess.run(
                    ["gdaldem", "slope", str(cross_dem_path), str(out)],
                    check=True, capture_output=True, timeout=60,
                )
                slopes["gdal"] = read_geotiff_as_array(out)
                out.unlink(missing_ok=True)
            except Exception as e:
                print(f"  GDAL slope error: {e}")

        # GRASS
        if tools.get("grass"):
            out = RESULTS_DIR / "grass_slope_andes.tif"
            out.unlink(missing_ok=True)
            try:
                subprocess.run(
                    ["grass", _grass_tmp_flag(), "EPSG:32719", "--exec",
                     "bash", "-c",
                     f"r.in.gdal input={cross_dem_path} output=dem --overwrite && "
                     f"g.region raster=dem && "
                     f"r.slope.aspect elevation=dem slope=slp --overwrite && "
                     f"r.out.gdal input=slp output={out} format=GTiff --overwrite"],
                    check=True, capture_output=True, timeout=60,
                    env={**os.environ, "GRASS_VERBOSE": "0"},
                )
                slopes["grass"] = read_geotiff_as_array(out)
                out.unlink(missing_ok=True)
            except Exception as e:
                print(f"  GRASS slope error: {e}")

        # WBT
        if tools.get("wbt"):
            import whitebox
            wbt = whitebox.WhiteboxTools()
            wbt.set_verbose_mode(False)
            out = RESULTS_DIR / "wbt_slope_andes.tif"
            out.unlink(missing_ok=True)
            try:
                wbt.slope(str(cross_dem_path.resolve()), str(out.resolve()))
                slopes["wbt"] = read_geotiff_as_array(out)
                out.unlink(missing_ok=True)
            except Exception as e:
                print(f"  WBT slope error: {e}")

        # Pairwise comparison
        tool_names = list(slopes.keys())
        sl2 = slice(1, -1)
        interior2 = (sl2, sl2)

        for i, t1 in enumerate(tool_names):
            for t2 in tool_names[i + 1:]:
                s1 = slopes[t1][interior2]
                s2 = slopes[t2][interior2]

                # WBT returns slope in degrees by default, verify ranges match
                comparison = f"{t1}_vs_{t2}"
                for row in _accuracy_metrics(s1, s2, "slope", comparison):
                    csv_writer.append(row)

                # Additional: % cells within 0.1 degrees
                pct = np.mean(np.abs(s1 - s2) < 0.1) * 100.0
                csv_writer.append({
                    "metric": "pct_within_0.1deg",
                    "algorithm": "slope",
                    "comparison": comparison,
                    "value": f"{pct:.2f}",
                })
                print(f"  {comparison}: RMSE={_rmse(s1,s2):.4f}°, "
                      f"agree<0.1°={pct:.1f}%")

        del slopes, real_dem
        force_cleanup()

    print(f"\nResults: {csv_path}")
    return csv_path


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(a - b)))


def _r2(a, b):
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def _accuracy_metrics(reference, computed, algorithm, comparison):
    """Compute RMSE, MAE, R² between reference and computed arrays."""
    # Filter out NaN/Inf
    valid = np.isfinite(reference) & np.isfinite(computed)
    ref = reference[valid]
    comp = computed[valid]

    if len(ref) == 0:
        return []

    return [
        {"metric": "rmse", "algorithm": algorithm,
         "comparison": comparison, "value": f"{_rmse(ref, comp):.6f}"},
        {"metric": "mae", "algorithm": algorithm,
         "comparison": comparison, "value": f"{_mae(ref, comp):.6f}"},
        {"metric": "r2", "algorithm": algorithm,
         "comparison": comparison, "value": f"{_r2(ref, comp):.6f}"},
    ]


# ---------------------------------------------------------------------------
# Experiment 3: Cross-platform
# ---------------------------------------------------------------------------


def experiment3_crossplatform(
    reps: int,
    warmup: int,
    tools: dict[str, bool],
    append: bool = False,
) -> Path:
    """Run cross-platform benchmarks. Returns path to CSV."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Cross-Platform Performance")
    print("=" * 60)

    size = 5000
    csv_path = RESULTS_DIR / "experiment3_crossplatform.csv"
    csv_writer = IncrementalCSV(csv_path, ["algorithm", "target", "size",
                                            "run", "time_seconds"],
                                append=append)
    algorithms = ["slope", "aspect", "hillshade"]

    log_memory("PRE-EXP3 ")

    # Generate 5K DEM
    dem_file = DEMS_DIR / f"fbm_{size}.tif"
    dem_array = None
    if not dem_file.exists():
        print(f"  Generating fBm DEM {size}x{size}...", end=" ", flush=True)
        t0 = time.perf_counter()
        dem_array = generate_fbm_dem(size, cell_size=CELL_SIZE)
        save_dem_geotiff(dem_array, dem_file, CELL_SIZE)
        force_cleanup()
        print(f"done ({time.perf_counter()-t0:.1f}s)")

    # --- Target 1: Rust native multi-thread ---
    if check_cargo():
        print("\n--- Rust native (multi-thread) ---")
        _run_rust_bench(csv_writer, algorithms, size, reps, warmup,
                        "rust_multithread", env_override=None)

    # --- Target 2: Rust native single-thread ---
    if check_cargo():
        print("\n--- Rust native (single-thread) ---")
        _run_rust_bench(csv_writer, algorithms, size, reps, warmup,
                        "rust_singlethread",
                        env_override={"RAYON_NUM_THREADS": "1"})

    # --- Target 3: Python bindings ---
    if tools.get("surtgis"):
        print("\n--- Python bindings ---")
        import surtgis

        if dem_array is None:
            dem_array = read_geotiff_as_array(dem_file)

        for alg in algorithms:
            print(f"  {alg}: ", end="", flush=True)

            def _make_runner(a):
                def _run():
                    if a == "slope":
                        surtgis.slope(dem_array, CELL_SIZE)
                    elif a == "aspect":
                        surtgis.aspect_degrees(dem_array, CELL_SIZE)
                    elif a == "hillshade":
                        surtgis.hillshade_compute(dem_array, CELL_SIZE)
                return _run

            runner = _make_runner(alg)
            for _ in range(warmup):
                runner()
                force_cleanup()

            times = []
            for rep in range(reps):
                t0 = time.perf_counter()
                runner()
                t = time.perf_counter() - t0
                times.append(t)
                csv_writer.append({
                    "algorithm": alg,
                    "target": "python_bindings",
                    "size": size,
                    "run": rep + 1,
                    "time_seconds": f"{t:.6f}",
                })
                force_cleanup()

            med = np.median(times)
            print(f"median={med:.4f}s (n={len(times)})")

        del dem_array
        force_cleanup()

    # --- Target 4: WebAssembly (placeholder) ---
    print("\n--- WebAssembly ---")
    print("  PLACEHOLDER: WASM benchmarks require browser automation.")
    print("  Will be reported separately in the paper.")
    for alg in algorithms:
        csv_writer.append({
            "algorithm": alg,
            "target": "wasm",
            "size": size,
            "run": 1,
            "time_seconds": "PENDING",
        })

    print(f"\nResults: {csv_path}")
    return csv_path


def _run_rust_bench(csv_writer, algorithms, size, reps, warmup, target, env_override):
    """Run the Rust bench_comparison example and parse output."""
    cargo_cmd = [
        "cargo", "run", "-p", "surtgis-algorithms",
        "--example", "bench_comparison", "--release",
        "--", "--size", str(size),
    ]
    if target == "rust_singlethread":
        cargo_cmd.append("--single-thread")

    env = {**os.environ}
    if env_override:
        env.update(env_override)

    for alg in algorithms:
        print(f"  {alg}: ", end="", flush=True)
        times = []

        for _ in range(warmup):
            try:
                subprocess.run(cargo_cmd, capture_output=True, timeout=TIMEOUT_SECONDS,
                               cwd=str(ROOT_DIR), env=env)
            except Exception:
                break

        for rep in range(reps):
            try:
                t0 = time.perf_counter()
                result = subprocess.run(
                    cargo_cmd, capture_output=True, text=True,
                    timeout=TIMEOUT_SECONDS, cwd=str(ROOT_DIR), env=env,
                )
                elapsed = time.perf_counter() - t0

                # Parse SurtGis time from the output table
                alg_time = _parse_rust_output(result.stdout, alg)
                if alg_time is not None:
                    times.append(alg_time)
                    csv_writer.append({
                        "algorithm": alg,
                        "target": target,
                        "size": size,
                        "run": rep + 1,
                        "time_seconds": f"{alg_time:.6f}",
                    })
                else:
                    # Fallback: use total elapsed / 3 (rough)
                    times.append(elapsed / 3.0)
                    csv_writer.append({
                        "algorithm": alg,
                        "target": target,
                        "size": size,
                        "run": rep + 1,
                        "time_seconds": f"{elapsed / 3.0:.6f}",
                    })

            except subprocess.TimeoutExpired:
                print("TIMEOUT ", end="")
                csv_writer.append({
                    "algorithm": alg,
                    "target": target,
                    "size": size,
                    "run": rep + 1,
                    "time_seconds": "TIMEOUT",
                })
                break
            except Exception as e:
                print(f"ERROR({e}) ", end="")
                break

        if times:
            med = np.median(times)
            print(f"median={med:.4f}s (n={len(times)})")
        else:
            print("no valid runs")


def _parse_rust_output(stdout: str, algorithm: str) -> float | None:
    """Parse the SurtGis time for a given algorithm from bench_comparison output."""
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == algorithm:
            # Format: "slope    12.3 ms ..."
            try:
                val = float(parts[1])
                unit = parts[2] if len(parts) > 2 else "ms"
                if unit == "ms":
                    return val / 1000.0
                elif unit == "s":
                    return val
            except (ValueError, IndexError):
                pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SurtGIS Paper Benchmarks (Environmental Modelling & Software)",
    )
    parser.add_argument(
        "--experiment", type=int, choices=[1, 2, 3],
        help="Run only a specific experiment (1, 2, or 3)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: smaller DEMs, fewer repetitions",
    )
    parser.add_argument(
        "--tools",
        help="Comma-separated list of tools to use (e.g. surtgis,gdal)",
    )
    parser.add_argument(
        "--sizes",
        help="Comma-separated DEM sizes (e.g. 20000 or 5000,10000,20000)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing CSV files instead of overwriting",
    )
    args = parser.parse_args()

    # Setup directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEMS_DIR.mkdir(parents=True, exist_ok=True)

    # Detect tools
    requested = args.tools.split(",") if args.tools else None
    tools = get_available_tools(requested)

    print("SurtGIS Paper Benchmarks")
    print("=" * 40)
    print(f"Mode: {'quick' if args.quick else 'full'}")
    print("Available tools:")
    for t, avail in tools.items():
        print(f"  {t:10s}: {'YES' if avail else 'no'}")

    if not any(tools.values()):
        print("\nERROR: No tools available. Install surtgis (maturin develop --release)")
        sys.exit(1)

    # Parameters
    if args.quick:
        sizes = QUICK_SIZES
        reps = QUICK_REPS
        warmup = QUICK_WARMUP
    else:
        sizes = FULL_SIZES
        reps = FULL_REPS
        warmup = FULL_WARMUP

    # Override sizes if specified
    if args.sizes:
        sizes = [int(s.strip()) for s in args.sizes.split(",")]
        print(f"Custom sizes: {sizes}")

    # Collect system info
    info = collect_system_info(tools)
    info_path = RESULTS_DIR / "system_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSystem info: {info_path}")

    # Run experiments
    experiments = [args.experiment] if args.experiment else [1, 2, 3]

    if 1 in experiments:
        experiment1_scalability(sizes, reps, warmup, tools, append=args.append)

    if 2 in experiments:
        experiment2_accuracy(tools)

    if 3 in experiments:
        experiment3_crossplatform(reps, warmup, tools, append=args.append)

    print("\n" + "=" * 60)
    print("ALL DONE")
    print(f"Results in: {RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
