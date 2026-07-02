"""Smoke tests for the surtgis Python bindings.

Run after building/installing the wheel:

    maturin build --manifest-path crates/python/Cargo.toml --release
    pip install target/wheels/surtgis-*.whl numpy pytest
    pytest crates/python/tests/ -v

Covers: module import, representative functions from each category
(terrain, hydrology, imagery, statistics), output shapes, plausible value
ranges, NaN/nodata propagation, and — critically — that heavy compute
releases the GIL (regression test for the Sprint 4 audit item S5).
"""

import threading
import time

import numpy as np
import pytest

import surtgis

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

SIZE = 20
CELL = 10.0


def make_dem(size=SIZE, seed=42):
    """Tilted plane + Gaussian hill: smooth, no flats, deterministic."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    plane = 2.0 * x + 1.0 * y
    cx = cy = size / 2.0
    hill = 50.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * (size / 5.0) ** 2))
    noise = rng.random((size, size)) * 0.1
    return (plane + hill + noise).astype(np.float64)


@pytest.fixture
def dem():
    return make_dem()


# ---------------------------------------------------------------------------
# Import / module surface
# ---------------------------------------------------------------------------


def test_import_and_surface():
    assert hasattr(surtgis, "slope")
    assert hasattr(surtgis, "flow_accumulation_d8")
    assert hasattr(surtgis, "ndvi_compute")


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------


def test_slope_range(dem):
    out = surtgis.slope(dem, cell_size=CELL)
    assert out.shape == dem.shape
    valid = out[np.isfinite(out)]
    assert valid.size > 0
    assert valid.min() >= 0.0
    assert valid.max() <= 90.0


def test_aspect_range(dem):
    out = surtgis.aspect_degrees(dem, cell_size=CELL)
    assert out.shape == dem.shape
    valid = out[np.isfinite(out)]
    assert valid.size > 0
    # 0-360 degrees; flat cells may be encoded as -1
    assert valid.min() >= -1.0
    assert valid.max() <= 360.0


def test_hillshade_range(dem):
    out = surtgis.hillshade_compute(dem, cell_size=CELL, azimuth=315.0, altitude=45.0)
    assert out.shape == dem.shape
    valid = out[np.isfinite(out)]
    assert valid.size > 0
    assert valid.min() >= 0.0
    assert valid.max() <= 255.0


def test_tpi_shape(dem):
    out = surtgis.tpi_compute(dem, cell_size=CELL, radius=3)
    assert out.shape == dem.shape
    # Hill summit should be locally elevated -> positive TPI at center
    c = SIZE // 2
    assert np.isfinite(out[c, c])
    assert out[c, c] > 0.0


def test_geomorphons_codes(dem):
    out = surtgis.geomorphons_compute(dem, cell_size=CELL)
    assert out.shape == dem.shape
    assert out.dtype == np.uint8
    # geomorphon classes are small integer codes
    assert out.max() <= 12


# ---------------------------------------------------------------------------
# Hydrology
# ---------------------------------------------------------------------------


def test_fill_sinks_monotone(dem):
    # Punch a pit into the DEM; fill must raise it, never lower anything.
    pitted = dem.copy()
    pitted[10, 10] = -100.0
    out = surtgis.fill_depressions(pitted, cell_size=CELL)
    assert out.shape == dem.shape
    mask = np.isfinite(out) & np.isfinite(pitted)
    assert np.all(out[mask] >= pitted[mask] - 1e-9)
    assert out[10, 10] > -100.0


def test_priority_flood_monotone(dem):
    pitted = dem.copy()
    pitted[5, 5] = -50.0
    out = surtgis.priority_flood_fill(pitted, cell_size=CELL)
    mask = np.isfinite(out) & np.isfinite(pitted)
    assert np.all(out[mask] >= pitted[mask] - 1e-9)


def test_flow_direction_d8(dem):
    filled = surtgis.fill_depressions(dem, cell_size=CELL)
    fdir = surtgis.flow_direction_d8(filled, cell_size=CELL)
    assert fdir.shape == dem.shape
    assert fdir.dtype == np.uint8
    # SurtGIS D8 encoding: 1..8 counter-clockwise from East (see
    # hydrology::d8), 0 for pits/nodata — NOT the ESRI powers-of-two codes.
    valid_codes = set(range(9))
    assert set(np.unique(fdir)).issubset(valid_codes)
    # After filling with default epsilon there is exactly one undirected
    # cell class allowed: the outlet(s)/edges may drain out, interior must flow
    interior = fdir[1:-1, 1:-1]
    assert (interior > 0).mean() > 0.95, "most interior cells must have a direction"


def test_flow_accumulation_d8(dem):
    filled = surtgis.fill_depressions(dem, cell_size=CELL)
    fdir = surtgis.flow_direction_d8(filled, cell_size=CELL)
    acc = surtgis.flow_accumulation_d8(fdir, cell_size=CELL)
    assert acc.shape == dem.shape
    valid = acc[np.isfinite(acc)]
    assert valid.min() >= 0.0
    # accumulation can never exceed the number of cells
    assert valid.max() <= dem.size
    # somewhere downstream must accumulate more than a single cell
    assert valid.max() > 1.0


def test_twi_finite(dem):
    out = surtgis.twi_compute(dem, cell_size=CELL)
    assert out.shape == dem.shape
    assert np.isfinite(out).any()


def test_watershed(dem):
    filled = surtgis.fill_depressions(dem, cell_size=CELL)
    fdir = surtgis.flow_direction_d8(filled, cell_size=CELL)
    ws = surtgis.watershed_compute(fdir, cell_size=CELL)
    assert ws.shape == dem.shape
    valid = ws[np.isfinite(ws)]
    assert valid.size > 0
    assert valid.min() >= 0.0


# ---------------------------------------------------------------------------
# Imagery
# ---------------------------------------------------------------------------


def test_ndvi_range():
    rng = np.random.default_rng(7)
    nir = rng.random((SIZE, SIZE)) * 0.8 + 0.1
    red = rng.random((SIZE, SIZE)) * 0.5 + 0.05
    out = surtgis.ndvi_compute(nir, red)
    assert out.shape == (SIZE, SIZE)
    valid = out[np.isfinite(out)]
    assert valid.size > 0
    assert valid.min() >= -1.0
    assert valid.max() <= 1.0


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_focal_mean(dem):
    out = surtgis.focal_mean(dem, radius=1)
    assert out.shape == dem.shape
    # interior cell: focal mean of a 3x3 window must sit within [min, max]
    win = dem[9:12, 9:12]
    assert win.min() - 1e-9 <= out[10, 10] <= win.max() + 1e-9


# ---------------------------------------------------------------------------
# NaN / nodata propagation
# ---------------------------------------------------------------------------


def test_nan_propagation_slope(dem):
    holed = dem.copy()
    holed[8, 8] = np.nan
    out = surtgis.slope(holed, cell_size=CELL)
    assert out.shape == dem.shape
    # the nodata hole must not silently become a valid value
    assert np.isnan(out[8, 8])


def test_nan_propagation_focal(dem):
    holed = dem.copy()
    holed[3:6, 3:6] = np.nan
    out = surtgis.focal_mean(holed, radius=1)
    # centre of the 3x3 NaN block has no valid neighbours at all
    assert np.isnan(out[4, 4])


# ---------------------------------------------------------------------------
# GIL release (audit item S5)
# ---------------------------------------------------------------------------


def test_gil_released_during_compute():
    """A heavy compute in a worker thread must not block the main thread.

    If the binding holds the GIL for the whole call, the main thread cannot
    execute any bytecode until the worker finishes, so the counter stays at
    ~0. With py.allow_threads the counter advances freely.
    """
    dem = make_dem(size=800, seed=3)

    done = threading.Event()
    result = {}

    def worker():
        try:
            # fill_sinks + MFD accumulation on 800x800: comfortably >100 ms
            result["out"] = surtgis.flow_accumulation_mfd_compute(dem, cell_size=1.0)
        finally:
            done.set()

    t = threading.Thread(target=worker)
    t0 = time.monotonic()
    t.start()

    counter = 0
    while not done.is_set():
        counter += 1
        if time.monotonic() - t0 > 120.0:  # safety net
            break

    t.join(timeout=120.0)
    assert "out" in result, "worker did not finish"
    assert result["out"].shape == dem.shape
    # With the GIL held throughout, counter would be ~0 (the main thread is
    # frozen). Require clear evidence of concurrent progress.
    assert counter > 1000, f"main thread only advanced {counter} steps: GIL not released"
