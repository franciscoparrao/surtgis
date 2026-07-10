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


def test_ordinary_kriging_interpolation():
    rng = np.random.default_rng(11)
    n = 25
    pts = rng.uniform(0, 100, size=(n, 2))
    vals = (pts[:, 0] + pts[:, 1]).reshape(-1, 1)
    grid = surtgis.ordinary_kriging_interpolation(pts, vals, 15, 15, 8.0, 8)
    assert grid.shape == (15, 15)
    assert np.isfinite(grid).any()


def test_zonal_statistics_mean():
    rng = np.random.default_rng(9)
    values = rng.random((SIZE, SIZE)) * 100
    zones = np.ones((SIZE, SIZE), dtype=np.float64)
    zones[SIZE // 2 :, :] = 2.0
    out = surtgis.zonal_statistics_compute(values, zones, CELL, "mean")
    assert out.shape == values.shape
    assert np.isclose(out[0, 0], values[: SIZE // 2, :].mean())
    assert np.isclose(out[-1, 0], values[SIZE // 2 :, :].mean())


def test_zonal_statistics_zone_zero_is_excluded():
    # Zone id 0 means "no zone" (same convention as watershed ids) and must
    # come back NaN, not silently included in some other zone's stats.
    values = np.ones((SIZE, SIZE), dtype=np.float64) * 5.0
    zones = np.zeros((SIZE, SIZE), dtype=np.float64)
    out = surtgis.zonal_statistics_compute(values, zones, CELL, "mean")
    assert np.all(np.isnan(out))


def test_linear_trend_compute_recovers_slope():
    rng = np.random.default_rng(3)
    t, r, c = 10, 12, 12
    stack = np.zeros((t, r, c), dtype=np.float64)
    for i in range(t):
        stack[i] = i * 3.0 + rng.normal(0, 0.01, size=(r, c))
    slope, intercept, r_squared, p_value = surtgis.linear_trend_compute(stack)
    assert slope.shape == (r, c)
    assert np.allclose(slope, 3.0, atol=0.1)
    assert np.all(r_squared[np.isfinite(r_squared)] > 0.99)


def test_mann_kendall_compute_detects_increasing_trend():
    t, r, c = 10, 10, 10
    stack = np.stack([np.full((r, c), float(i)) for i in range(t)])
    tau, p_value, trend, sens_slope = surtgis.mann_kendall_compute(stack)
    assert tau.shape == (r, c)
    assert np.allclose(trend, 1.0)
    assert np.allclose(sens_slope, 1.0)


def test_slic_compute_produces_multiple_segments():
    rng = np.random.default_rng(5)
    bands = rng.random((2, 30, 30)).astype(np.float64)
    labels = surtgis.slic_compute(bands, 25, 0.1, CELL)
    assert labels.shape == (30, 30)
    assert len(np.unique(labels)) > 1


def test_felzenszwalb_compute_produces_labels():
    rng = np.random.default_rng(6)
    bands = rng.random((2, 30, 30)).astype(np.float64)
    labels = surtgis.felzenszwalb_compute(bands, 1.0, 20, CELL)
    assert labels.shape == (30, 30)
    assert np.isfinite(labels).any()


@pytest.mark.parametrize("method", ["brovey", "gram_schmidt", "pca"])
def test_pansharpen_compute_all_methods(method):
    rng = np.random.default_rng(8)
    pan = rng.random((40, 40)).astype(np.float64)
    ms = rng.random((3, 40, 40)).astype(np.float64)
    sharp = surtgis.pansharpen_compute(pan, ms, CELL, method)
    assert sharp.shape == (3, 40, 40)
    assert np.isfinite(sharp).any()


def test_pansharpen_compute_rejects_unknown_method():
    rng = np.random.default_rng(8)
    pan = rng.random((10, 10)).astype(np.float64)
    ms = rng.random((2, 10, 10)).astype(np.float64)
    with pytest.raises(ValueError):
        surtgis.pansharpen_compute(pan, ms, CELL, "not-a-method")


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name", ["morph_erode", "morph_dilate", "morph_opening", "morph_closing"]
)
def test_morphology_ops_preserve_shape(dem, fn_name):
    fn = getattr(surtgis, fn_name)
    out = fn(dem, radius=1)
    assert out.shape == dem.shape
    assert np.isfinite(out).any()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_kmeans_compute():
    out = surtgis.kmeans_compute(make_dem(seed=1), k=3, max_iterations=50)
    assert out.shape == (SIZE, SIZE)
    valid = out[np.isfinite(out)]
    # Cluster labels are 1..k, not 0-indexed.
    assert set(np.unique(valid)).issubset(set(range(1, 4))) or valid.size == 0


def test_isodata_compute():
    out = surtgis.isodata_compute(make_dem(seed=2), initial_k=3, max_iterations=20)
    assert out.shape == (SIZE, SIZE)


# ---------------------------------------------------------------------------
# Interpolation (IDW / nearest / natural neighbor)
# ---------------------------------------------------------------------------


def _synthetic_points(n=20, seed=1):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, size=(n, 2))
    values = (xy[:, 0] * 0.5 + xy[:, 1] * 0.2).reshape(-1, 1)
    return xy, values


def test_idw_interpolation():
    xy, values = _synthetic_points()
    grid = surtgis.idw_interpolation(xy, values, grid_rows=15, grid_cols=15, cellsize=8.0, power=2.0)
    assert grid.shape == (15, 15)
    assert np.isfinite(grid).any()


def test_nearest_neighbor_interpolation():
    xy, values = _synthetic_points()
    grid = surtgis.nearest_neighbor_interpolation(xy, values, grid_rows=15, grid_cols=15, cellsize=8.0)
    assert grid.shape == (15, 15)
    assert np.isfinite(grid).any()


def test_natural_neighbor_interpolation():
    xy, values = _synthetic_points()
    grid = surtgis.natural_neighbor_interpolation(xy, values, grid_rows=15, grid_cols=15, cellsize=8.0)
    assert grid.shape == (15, 15)


# ---------------------------------------------------------------------------
# Landscape ecology
# ---------------------------------------------------------------------------


def _classification_raster():
    rng = np.random.default_rng(4)
    return rng.integers(1, 4, size=(SIZE, SIZE)).astype(np.float64)


def test_label_patches_compute():
    labels, n_patches = surtgis.label_patches_compute(_classification_raster(), connectivity8=True)
    assert labels.shape == (SIZE, SIZE)
    assert n_patches > 0


def test_shannon_diversity_compute():
    out = surtgis.shannon_diversity_compute(_classification_raster(), radius=3)
    assert out.shape == (SIZE, SIZE)
    valid = out[np.isfinite(out)]
    assert valid.size > 0
    assert valid.min() >= 0.0


def test_simpson_diversity_compute():
    out = surtgis.simpson_diversity_compute(_classification_raster(), radius=3)
    assert out.shape == (SIZE, SIZE)


def test_patch_density_compute():
    out = surtgis.patch_density_compute(_classification_raster(), radius=3)
    assert out.shape == (SIZE, SIZE)


def test_landscape_metrics_compute():
    shdi, sidi, num_patches, num_classes, total_area_m2, total_cells = (
        surtgis.landscape_metrics_compute(_classification_raster())
    )
    # num_patches is always 0 here by design: this function only computes
    # class-proportion diversity indices, not patch topology (see
    # class_metrics.rs — "filled by caller if patches are available").
    # Patch count comes from label_patches_compute separately.
    assert num_patches == 0
    assert num_classes > 0
    assert total_cells == SIZE * SIZE
    assert shdi >= 0.0
    assert sidi >= 0.0


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------


def test_sobel_edge_compute(dem):
    out = surtgis.sobel_edge_compute(dem)
    assert out.shape == dem.shape
    assert np.isfinite(out).any()


def test_laplacian_compute(dem):
    out = surtgis.laplacian_compute(dem)
    assert out.shape == dem.shape


@pytest.mark.parametrize(
    "texture", ["contrast", "energy", "homogeneity", "correlation", "entropy", "dissimilarity"]
)
def test_haralick_glcm_compute_all_textures(dem, texture):
    out = surtgis.haralick_glcm_compute(dem, radius=3, distance=1, texture=texture)
    assert out.shape == dem.shape


def test_haralick_glcm_compute_rejects_unknown_texture(dem):
    with pytest.raises(ValueError):
        surtgis.haralick_glcm_compute(dem, texture="not-a-texture")


# ---------------------------------------------------------------------------
# SAR
# ---------------------------------------------------------------------------


def test_sar_linear_to_db_and_back():
    rng = np.random.default_rng(13)
    backscatter = rng.random((SIZE, SIZE)) * 0.5 + 0.01
    db = surtgis.sar_linear_to_db(backscatter, CELL)
    assert db.shape == backscatter.shape
    linear = surtgis.sar_db_to_linear(db, CELL)
    assert np.allclose(linear, backscatter, atol=1e-6)


def test_sar_dual_pol_water_index():
    rng = np.random.default_rng(14)
    co = rng.random((SIZE, SIZE)) * 0.5 + 0.01
    cross = rng.random((SIZE, SIZE)) * 0.5 + 0.01
    out = surtgis.sar_dual_pol_water_index(co, cross, CELL)
    assert out.shape == co.shape
    valid = out[np.isfinite(out)]
    assert valid.min() >= -1.0 - 1e-9
    assert valid.max() <= 1.0 + 1e-9


def test_sar_water_mask():
    rng = np.random.default_rng(15)
    backscatter_db = rng.random((SIZE, SIZE)) * 20 - 20
    mask = surtgis.sar_water_mask(backscatter_db, -15.0, False, CELL)
    assert mask.shape == backscatter_db.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1, 255})


def test_sar_lee_filter():
    rng = np.random.default_rng(16)
    image = rng.random((SIZE, SIZE)) * 0.5 + 0.01
    out = surtgis.sar_lee_filter(image, 5, 1.0, CELL, False)
    assert out.shape == image.shape
    out_refined = surtgis.sar_lee_filter(image, 5, 1.0, CELL, True)
    assert out_refined.shape == image.shape


# ---------------------------------------------------------------------------
# Terrain: extras (viewshed, openness, contour, cost distance, MRVBF/VRM)
# ---------------------------------------------------------------------------


def test_viewshed_compute(dem):
    out = surtgis.viewshed_compute(dem, cell_size=CELL, observer_row=SIZE // 2, observer_col=SIZE // 2)
    assert out.shape == dem.shape
    assert out.dtype == np.uint8
    # observer cell must always see itself
    assert out[SIZE // 2, SIZE // 2] in (1, 255)


def test_openness_positive_and_negative(dem):
    pos = surtgis.openness_positive(dem, cell_size=CELL, directions=8, radius=5)
    neg = surtgis.openness_negative(dem, cell_size=CELL, directions=8, radius=5)
    assert pos.shape == dem.shape
    assert neg.shape == dem.shape


def test_contour_lines_compute(dem):
    out = surtgis.contour_lines_compute(dem, cell_size=CELL, interval=5.0)
    assert out.shape == dem.shape


def test_cost_distance_compute(dem):
    cost = np.ones_like(dem)
    out = surtgis.cost_distance_compute(cost, cell_size=CELL, source_row=0, source_col=0)
    assert out.shape == dem.shape
    assert out[0, 0] == 0.0
    assert out[-1, -1] > 0.0


def test_mrvbf_compute(dem):
    mrvbf, mrrtf = surtgis.mrvbf_compute(dem, cell_size=CELL)
    assert mrvbf.shape == dem.shape
    assert mrrtf.shape == dem.shape


def test_vrm_compute(dem):
    out = surtgis.vrm_compute(dem, cell_size=CELL, radius=1)
    assert out.shape == dem.shape
    valid = out[np.isfinite(out)]
    assert valid.min() >= 0.0


def test_solar_radiation_compute(dem):
    # Takes the DEM directly (slope/aspect computed internally), not
    # pre-computed slope/aspect rasters.
    out = surtgis.solar_radiation_compute(dem, cell_size=CELL, latitude=-33.0, day=172)
    assert out.shape == dem.shape


def test_melton_ruggedness_compute(dem):
    filled = surtgis.fill_depressions(dem, cell_size=CELL)
    fdir = surtgis.flow_direction_d8(filled, cell_size=CELL)
    ws = surtgis.watershed_compute(fdir, cell_size=CELL)
    metrics = surtgis.melton_ruggedness_compute(ws, dem, cell_size=CELL)
    assert isinstance(metrics, list)
    if metrics:
        assert "melton_ratio" in metrics[0]


# ---------------------------------------------------------------------------
# More vegetation indices (two-band NIR/Red-family functions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn_name",
    [
        "ndwi_compute",
        "savi_compute",
        "evi_compute",
        "ndre_compute",
        "gndvi_compute",
        "ngrdi_compute",
        "reci_compute",
    ],
)
def test_two_band_indices_shape(fn_name):
    rng = np.random.default_rng(20)
    a = rng.random((SIZE, SIZE)) * 0.8 + 0.1
    b = rng.random((SIZE, SIZE)) * 0.5 + 0.05
    fn = getattr(surtgis, fn_name)
    if fn_name == "evi_compute":
        c = rng.random((SIZE, SIZE)) * 0.5 + 0.05
        out = fn(a, b, c)
    else:
        out = fn(a, b)
    assert out.shape == (SIZE, SIZE)
    assert np.isfinite(out).any()


def test_bsi_compute():
    # BSI takes four bands: SWIR, NIR, Red, Blue (not the two-band shape
    # of the other spectral indices above).
    rng = np.random.default_rng(21)
    swir = rng.random((SIZE, SIZE)) * 0.5 + 0.1
    nir = rng.random((SIZE, SIZE)) * 0.8 + 0.1
    red = rng.random((SIZE, SIZE)) * 0.5 + 0.05
    blue = rng.random((SIZE, SIZE)) * 0.3 + 0.05
    out = surtgis.bsi_compute(swir, nir, red, blue)
    assert out.shape == (SIZE, SIZE)
    assert np.isfinite(out).any()


# ---------------------------------------------------------------------------
# ML: extract_at_points / predict_raster
# ---------------------------------------------------------------------------


def test_extract_at_points(dem):
    slope_r = surtgis.slope(dem, cell_size=CELL)
    cols = np.array([[2.0], [5.0], [10.0]])
    rows = np.array([[2.0], [5.0], [10.0]])
    out = surtgis.extract_at_points([dem, slope_r], cols, rows)
    assert out.shape[1] == 2
    assert out.shape[0] <= 3


def test_predict_raster(dem):
    def predict_fn(batch):
        return np.asarray(batch).sum(axis=1)

    out = surtgis.predict_raster([dem], predict_fn, batch_size=64)
    assert out.shape == dem.shape


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_idw_interpolation_rejects_mismatched_points_shape():
    xy = np.zeros((5, 3))  # wrong second dim
    values = np.zeros((5, 1))
    with pytest.raises(ValueError):
        surtgis.idw_interpolation(xy, values)


def test_zonal_statistics_rejects_unknown_statistic():
    values = np.ones((5, 5))
    zones = np.ones((5, 5))
    with pytest.raises(ValueError):
        surtgis.zonal_statistics_compute(values, zones, 1.0, "not-a-statistic")


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
