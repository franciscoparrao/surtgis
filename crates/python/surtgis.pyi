"""Type stubs for the surtgis PyO3 extension module.

Generated from the #[pyfunction] signatures in src/lib.rs and
src/cloud.rs. Regenerate with gen_stubs.py after adding/changing a
binding rather than hand-editing this file out of sync with the
Rust source.
"""

import typing

import numpy as np
import numpy.typing as npt

def accumulation_zones_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Classify terrain into accumulation/dispersion zones.
    """
    ...

def advanced_curvature(dem: npt.NDArray[np.float64], cell_size: float = 1.0, ctype: str = "mean_h") -> npt.NDArray[np.float64]:
    """Compute advanced curvature (Florinsky system).
ctype options: "mean_h", "gaussian_k", "unsphericity_m", "difference_e",
"minimal_kmin", "maximal_kmax", "horizontal_kh", "vertical_kv",
"horizontal_excess_khe", "vertical_excess_kve", "accumulation_ka",
"ring_kr", "rotor", "laplacian" 
    """
    ...

def aspect_degrees(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute aspect in degrees (0-360, 0=North, clockwise).
    """
    ...

def breach_fill(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Breach depressions (Lindsay 2016) - preferred over filling.
    """
    ...

def bsi_compute(swir: npt.NDArray[np.float64], nir: npt.NDArray[np.float64], red: npt.NDArray[np.float64], blue: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute BSI (Bare Soil Index) from SWIR, NIR, Red, and Blue bands.
    """
    ...

def chi_compute(stream: npt.NDArray[np.uint8], flow_dir: npt.NDArray[np.uint8], flow_acc: npt.NDArray[np.float64], theta_ref: float = 0.45, cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """χ (chi) transform of a stream network.

Args:
    stream: 2D numpy array (uint8), 1 = stream cell.
    flow_dir: 2D numpy array (uint8), D8 flow direction.
    flow_acc: 2D numpy array (f64), flow accumulation (cell counts).
    theta_ref: reference concavity (default 0.45).
    cell_size: cell size in metres.
Returns:
    2D numpy array (f64) of χ along the streams (NaN off-network).
    """
    ...

def cog_fetch(url: str, bbox: tuple[float, float, float, float], overview: int | None = None) -> tuple[npt.NDArray[np.float64], dict[str, typing.Any]]:
    """Fetch a bounding-box window from a remote Cloud-Optimized GeoTIFF.

Args:
    url: HTTP(S) or s3:// URL of the COG.
    bbox: (min_x, min_y, max_x, max_y) in the COG's CRS.
    overview: Optional overview level (0 = full resolution).

Returns:
    (array, meta): a 2D float64 numpy array and a metadata dict
    (transform, crs, nodata, width, height).
    """
    ...

def cog_fetch_full(url: str, overview: int | None = None) -> tuple[npt.NDArray[np.float64], dict[str, typing.Any]]:
    """Fetch the full extent of a remote COG (optionally at an overview level).

Returns (array, meta) like [`cog_fetch`]. Use an `overview` > 0 for large
scenes to avoid downloading every full-resolution tile.
    """
    ...

def cog_info(url: str) -> dict[str, typing.Any]:
    """Read a remote COG's header metadata without downloading pixel data.

Returns a dict: transform, crs, nodata, width, height, num_overviews,
bits_per_sample.
    """
    ...

def composite(catalog: str, collection: str, bbox: tuple[float, float, float, float], bands: list[str], datetime: str, max_scenes: int = 6, band_chunk_size: int = 1, budget_gb: float = 16.0, max_tile_failures: int = 0, use_cache: bool = False) -> tuple[dict[str, typing.Any], dict[str, typing.Any]]:
    """Build a cloud-free multiband composite from a STAC catalog.

Searches the catalog, downloads and mosaics each band across the selected
scene dates, applies cloud masking, and reduces per pixel to a median
composite (with coverage- and neighbour-based gap filling). Experimental
(mirrors `surtgis stac composite`); the API may change in a minor release.

Args:
    catalog: "pc", "es", or a full STAC API URL.
    collection: collection id (e.g. "sentinel-2-l2a").
    bbox: (min_x, min_y, max_x, max_y) in WGS84 degrees.
    bands: band/asset keys to composite (e.g. ["red", "nir"]).
    datetime: STAC datetime query (instant or "start/end").
    max_scenes: max scene dates to composite (default 6).
    band_chunk_size: bands downloaded together per chunk (default 1 = min RAM).
    budget_gb: RAM budget for strip sizing (default 16.0).
    max_tile_failures: abort after this many failed tiles (0 = never).
    use_cache: cache decoded tiles on disk between strips (default False).

Returns:
    (bands, meta): `bands` is a dict of band key → 2D float64 numpy array
    (NaN = no data); `meta` is a dict with `transform` (GDAL 6-tuple),
    `crs`, `width`, `height` shared by every band.
    """
    ...

def concavity_compute(stream: npt.NDArray[np.uint8], flow_dir: npt.NDArray[np.uint8], flow_acc: npt.NDArray[np.float64], dem: npt.NDArray[np.float64], basins: npt.NDArray[np.float64], cell_size: float = 1.0) -> list[dict[str, typing.Any]]:
    """Best-fit channel concavity θ per basin (χ–elevation collinearity).

Args:
    stream, flow_dir, flow_acc, dem: input rasters.
    basins: 2D numpy array of basin ids (f64, rounded to int).
    cell_size: cell size in metres.
Returns:
    A list of dicts: {basin_id, theta_opt, theta_ci_low, theta_ci_high,
    n_cells, rmse}.
    """
    ...

def contour_lines_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, interval: float = 10.0) -> npt.NDArray[np.float64]:
    """Generate contour lines as a raster.
    """
    ...

def convergence_index_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, radius: int = 1) -> npt.NDArray[np.float64]:
    """Compute convergence index.
    """
    ...

def cost_distance_compute(cost_surface: npt.NDArray[np.float64], cell_size: float = 1.0, source_row: int = 0, source_col: int = 0) -> npt.NDArray[np.float64]:
    """Compute cost distance from source cells.
    """
    ...

def curvature_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, ctype: str = "general") -> npt.NDArray[np.float64]:
    """Compute curvature. `ctype`: "general", "profile", or "plan".
    """
    ...

def curvedness_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute curvedness.
    """
    ...

def dev_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, radius: int = 10) -> npt.NDArray[np.float64]:
    """Compute DEV (Deviation from Mean Elevation).
    """
    ...

def divide_migration_compute(basins: npt.NDArray[np.float64], dem: npt.NDArray[np.float64], flow_acc: npt.NDArray[np.float64], chi: npt.NDArray[np.float64] | None = None, cell_size: float = 1.0) -> list[dict[str, typing.Any]]:
    """Drainage-divide migration metrics (across-divide χ / elevation asymmetry).

Args:
    basins: 2D numpy array of basin ids (f64, rounded to int).
    dem: 2D numpy array (f64).
    flow_acc: 2D numpy array (f64).
    chi: optional 2D numpy array (f64) of χ; enables the χ-difference metric.
    cell_size: cell size in metres.
Returns:
    A list of dicts: {basin_a, basin_b, median_chi_diff, median_elev_diff,
    median_relief_diff, n_pairs}.
    """
    ...

def eastness_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute eastness (sin of aspect).
    """
    ...

def energy_cone_compute(dem: npt.NDArray[np.float64], sources: list[tuple[int, int]], cone_angle_degrees: float = 10.0, collapse_height: float = 0.0, cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Energy-cone lahar / mass-flow inundation (Malin & Sheridan 1982).

Args:
    dem: 2D numpy array (f64) of elevations.
    sources: list of (row, col) source cells.
    cone_angle_degrees: energy-cone angle φ (H/L = tan φ); smaller = more mobile.
    collapse_height: height added to the source elevation to set the apex.
    cell_size: cell size in map units.
Returns:
    2D numpy array (f64): energy height above ground (>0 = reached, 0 = not).
    """
    ...

def evi2_compute(nir: npt.NDArray[np.float64], red: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute EVI2 (Two-band Enhanced Vegetation Index) from NIR and Red bands.
    """
    ...

def evi_compute(nir: npt.NDArray[np.float64], red: npt.NDArray[np.float64], blue: npt.NDArray[np.float64], g: float = 2.5, c1: float = 6.0, c2: float = 7.5, l: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute EVI (Enhanced Vegetation Index) from NIR, Red, and Blue bands.
    """
    ...

def excess_topography_compute(dem: npt.NDArray[np.float64], threshold_degrees: float = 30.0, cell_size: float = 1.0, max_iterations: int = 200) -> npt.NDArray[np.float64]:
    """Excess topography above a threshold hillslope angle (Blöthe et al. 2015).

Args:
    dem: 2D numpy array (f64) of elevations.
    threshold_degrees: threshold hillslope angle (0 < θ < 90), default 30.
    cell_size: cell size in map units.
    max_iterations: max fast-sweeping rounds (default 200).
Returns:
    2D numpy array (f64): excess height (z − slope-limited surface), >= 0.
    """
    ...

def extract_at_points(rasters: list[npt.NDArray[np.float64]], points_col: npt.NDArray[np.float64], points_row: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Extract pixel values from multiple rasters at point locations.

Args:
    rasters: list of 2D numpy arrays (all same shape)
    points_x: 1D array of X coordinates (in pixel space: column indices)
    points_y: 1D array of Y coordinates (in pixel space: row indices)

Returns:
    2D numpy array of shape (N_valid_points, N_rasters) with extracted values.
    Points outside bounds or with NaN in any raster are excluded.

Example:
    import surtgis, rasterio, numpy as np
    slope = rasterio.open("slope.tif").read(1).astype(np.float64)
    aspect = rasterio.open("aspect.tif").read(1).astype(np.float64)
    # Convert geo coords to pixel coords using rasterio transform
    cols, rows = rasterio.transform.rowcol(transform, xs, ys)
    X = surtgis.extract_at_points([slope, aspect], cols, rows)

GIL note: this function intentionally does NOT release the GIL — the
gather loop reads the numpy buffers in place (no copy), so the borrows
must stay under the GIL. Cost is O(n_points * n_rasters) reads.
    """
    ...

def feature_preserving_smoothing_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, iterations: int = 3, threshold: float = 15.0) -> npt.NDArray[np.float64]:
    """Feature-preserving DEM smoothing.
    """
    ...

def felzenszwalb_compute(bands: npt.NDArray[np.float64], scale: float = 1.0, min_size: int = 20, cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Felzenszwalb-Huttenlocher graph-based segmentation over a `(bands,
rows, cols)` multi-band stack. Returns an integer label raster (as
f64, same convention as `slic_compute`).
    """
    ...

def fill_depressions(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Fill sinks (depressions) in a DEM.
    """
    ...

def flow_accumulation_adaptive_mfd(dem: npt.NDArray[np.float64], cell_size: float = 1.0, convergence: float = 8.9) -> npt.NDArray[np.float64]:
    """Adaptive MFD flow accumulation (Qin 2011).
    """
    ...

def flow_accumulation_d8(fdir: npt.NDArray[np.uint8], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute flow accumulation from D8 flow direction raster.
    """
    ...

def flow_accumulation_mfd_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, exponent: float = 1.1) -> npt.NDArray[np.float64]:
    """Compute MFD (Multiple Flow Direction) accumulation.
    """
    ...

def flow_accumulation_tfga_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """TFGA (Facet-to-Facet) flow accumulation.
    """
    ...

def flow_direction_d8(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.uint8]:
    """Compute D8 flow direction. Returns u8 direction codes.
    """
    ...

def flow_direction_dinf(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute D-infinity flow direction (Tarboton 1997).
Returns continuous flow angles in radians [0, 2π), -1 for pits.
    """
    ...

def flow_path_length_compute(fdir: npt.NDArray[np.uint8], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute flow path length from D8 flow direction.
    """
    ...

def focal_max(data: npt.NDArray[np.float64], radius: int = 1, circular: bool = False) -> npt.NDArray[np.float64]:
    """Focal maximum statistic.
    """
    ...

def focal_mean(data: npt.NDArray[np.float64], radius: int = 1, circular: bool = False) -> npt.NDArray[np.float64]:
    """Focal mean statistic.
    """
    ...

def focal_median(data: npt.NDArray[np.float64], radius: int = 1, circular: bool = False) -> npt.NDArray[np.float64]:
    """Focal median statistic.
    """
    ...

def focal_min(data: npt.NDArray[np.float64], radius: int = 1, circular: bool = False) -> npt.NDArray[np.float64]:
    """Focal minimum statistic.
    """
    ...

def focal_range(data: npt.NDArray[np.float64], radius: int = 1, circular: bool = False) -> npt.NDArray[np.float64]:
    """Focal range (max - min).
    """
    ...

def focal_std(data: npt.NDArray[np.float64], radius: int = 1, circular: bool = False) -> npt.NDArray[np.float64]:
    """Focal standard deviation.
    """
    ...

def focal_sum(data: npt.NDArray[np.float64], radius: int = 1, circular: bool = False) -> npt.NDArray[np.float64]:
    """Focal sum statistic.
    """
    ...

def gaussian_smoothing_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, sigma: float = 1.0) -> npt.NDArray[np.float64]:
    """Gaussian smoothing of a DEM.
    """
    ...

def geomorphons_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, flatness: float = 1.0, radius: int = 10, skip: int = 0, flatness_distance: float = 0.0) -> npt.NDArray[np.uint8]:
    """Compute geomorphons landform classification (GRASS r.geomorphon parity).
Returns u8 landform codes (1-10, GRASS categories; 0 = nodata/border).
    """
    ...

def gndvi_compute(nir: npt.NDArray[np.float64], green: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute GNDVI (Green NDVI) from NIR and Green bands.
    """
    ...

def hand_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, stream_threshold: float = 1000.0) -> npt.NDArray[np.float64]:
    """Compute HAND from a DEM (fill → flow_dir → flow_acc → HAND).
    """
    ...

def haralick_glcm_compute(data: npt.NDArray[np.float64], radius: int = 3, distance: int = 1, texture: str = "contrast") -> npt.NDArray[np.float64]:
    """Compute GLCM (Haralick) texture feature.
texture: "contrast", "energy", "homogeneity", "correlation", "entropy", "dissimilarity" 
    """
    ...

def hillshade_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, azimuth: float = 315.0, altitude: float = 45.0) -> npt.NDArray[np.float64]:
    """Compute hillshade from a DEM.
    """
    ...

def horizon_angle_map_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, direction_degrees: float = 0.0, max_distance: int = 100) -> npt.NDArray[np.float64]:
    """Compute horizon angle map for a single azimuth direction.
    """
    ...

def idw_interpolation(points_xy: npt.NDArray[np.float64], values: npt.NDArray[np.float64], grid_rows: int = 100, grid_cols: int = 100, cellsize: float = 1.0, power: float = 2.0) -> npt.NDArray[np.float64]:
    """IDW interpolation from scattered points to a raster grid.
Points given as (N,2) array of (x,y) coords, values as (N,) array.
    """
    ...

def isodata_compute(data: npt.NDArray[np.float64], initial_k: int = 5, max_iterations: int = 50) -> npt.NDArray[np.float64]:
    """ISODATA unsupervised classification on a single raster.
    """
    ...

def kmeans_compute(data: npt.NDArray[np.float64], k: int = 5, max_iterations: int = 100) -> npt.NDArray[np.float64]:
    """K-means unsupervised classification on a single raster.
    """
    ...

def knickpoints_compute(stream: npt.NDArray[np.uint8], flow_dir: npt.NDArray[np.uint8], flow_acc: npt.NDArray[np.float64], dem: npt.NDArray[np.float64], theta_ref: float = 0.45, cell_size: float = 1.0) -> list[dict[str, typing.Any]]:
    """Detect knickpoints on a stream network.

Returns a list of dicts: {row, col, elevation_m, magnitude_m, chi, polarity}
where polarity is "concave" or "convex".
    """
    ...

def ksn_compute(stream: npt.NDArray[np.uint8], flow_dir: npt.NDArray[np.uint8], flow_acc: npt.NDArray[np.float64], dem: npt.NDArray[np.float64], theta_ref: float = 0.45, segment_length_m: float = 500.0, cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Normalized channel steepness index k_sn (per-cell raster).

Args:
    stream, flow_dir, flow_acc, dem: input rasters (uint8/uint8/f64/f64).
    theta_ref: reference concavity (default 0.45).
    segment_length_m: smoothing window along the channel (default 500).
    cell_size: cell size in metres.
Returns:
    2D numpy array (f64) of k_sn (NaN off-network).
    """
    ...

def label_patches_compute(classification: npt.NDArray[np.float64], connectivity8: bool = True) -> tuple[npt.NDArray[np.float64], int]:
    """Label connected patches in a classification raster.
Returns (patch_labels as f64 array, number_of_patches).
    """
    ...

def laharz_compute(dem: npt.NDArray[np.float64], flow_dir: npt.NDArray[np.uint8], sources: list[tuple[int, int]], volume_m3: float, flow_type: str = "lahar", cell_size: float = 1.0, spread_aspect: float | None = None) -> npt.NDArray[np.float64]:
    """LAHARZ lahar / debris-flow inundation (Iverson, Schilling & Vallance 1998).

Args:
    dem: 2D numpy array (f64) of elevations.
    flow_dir: 2D numpy array (uint8), D8 flow direction.
    sources: list of (row, col) source cells. Seed proximal CHANNEL cells,
        not the summit (a summit cell's D8 descent often runs down the wrong
        drainage). Sources route independently; the footprint is their union.
    volume_m3: flow volume in cubic metres (applied to each source).
    flow_type: "lahar" | "debris-flow" | "rock-avalanche".
    cell_size: cell size in metres.
Returns:
    2D numpy array (f64): inundation depth (>0 = inundated, 0 = dry).
    """
    ...

def landform_classification_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute landform classification (Weiss 2001, 11 classes).
    """
    ...

def landscape_metrics_compute(classification: npt.NDArray[np.float64]) -> tuple[float, float, int, int, float, int]:
    """Compute landscape-level metrics (SHDI, SIDI, etc.).
Returns a tuple (shdi, sidi, num_patches, num_classes, total_area_m2, total_cells).
    """
    ...

def laplacian_compute(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Laplacian edge detection (second derivative).
    """
    ...

def linear_trend_compute(stack: npt.NDArray[np.float64], times: npt.NDArray[np.float64] | None = None, cell_size: float = 1.0) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-pixel linear trend across a temporal stack. `stack` is
`(time, rows, cols)`; `times` (optional) are the time coordinate for
each band, default `0..T`. Returns `(slope, intercept, r_squared, p_value)`.
    """
    ...

def log_transform_compute(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Sign-preserving log transform: f(x) = sign(x) * ln(1 + |x|).
    """
    ...

def mann_kendall_compute(stack: npt.NDArray[np.float64], cell_size: float = 1.0) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Per-pixel Mann-Kendall trend test across a temporal stack `(time, rows,
cols)`. Returns `(tau, p_value, trend, sens_slope)` — `trend` is
1=increasing / -1=decreasing / 0=no significant trend at alpha=0.05.
    """
    ...

def melton_ruggedness_compute(watersheds: npt.NDArray[np.float64], dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> list[dict[str, typing.Any]]:
    """Melton ruggedness ratio per basin: (H_max - H_min) / sqrt(area).

Args:
    watersheds: 2D numpy array of watershed ids (f64, rounded to int;
        values <= 0 are ignored). The output of watershed_compute works.
    dem: 2D numpy array (f64) of elevations, aligned with watersheds.
    cell_size: cell size in metres.
Returns:
    A list of dicts, one per basin, with keys: watershed_id, area_cells,
    area_m2, min_elevation, max_elevation, relief, melton_ratio.
    """
    ...

def mndwi_compute(green: npt.NDArray[np.float64], swir: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute MNDWI (Modified NDWI) from Green and SWIR bands.
    """
    ...

def morph_black_hat(data: npt.NDArray[np.float64], radius: int = 1) -> npt.NDArray[np.float64]:
    """Black-hat transform (closing minus original). Extracts dark features.
    """
    ...

def morph_closing(data: npt.NDArray[np.float64], radius: int = 1) -> npt.NDArray[np.float64]:
    """Morphological closing (dilation followed by erosion).
    """
    ...

def morph_dilate(data: npt.NDArray[np.float64], radius: int = 1) -> npt.NDArray[np.float64]:
    """Morphological dilation with square kernel.
    """
    ...

def morph_erode(data: npt.NDArray[np.float64], radius: int = 1) -> npt.NDArray[np.float64]:
    """Morphological erosion with square kernel.
    """
    ...

def morph_gradient(data: npt.NDArray[np.float64], radius: int = 1) -> npt.NDArray[np.float64]:
    """Morphological gradient (dilation minus erosion).
    """
    ...

def morph_opening(data: npt.NDArray[np.float64], radius: int = 1) -> npt.NDArray[np.float64]:
    """Morphological opening (erosion followed by dilation).
    """
    ...

def morph_top_hat(data: npt.NDArray[np.float64], radius: int = 1) -> npt.NDArray[np.float64]:
    """Top-hat transform (original minus opening). Extracts bright features.
    """
    ...

def mrvbf_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute MRVBF (Multi-resolution Valley Bottom Flatness).
Returns a tuple (mrvbf, mrrtf).
    """
    ...

def msavi_compute(nir: npt.NDArray[np.float64], red: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute MSAVI (Modified Soil-Adjusted Vegetation Index) from NIR and Red bands.
    """
    ...

def multidirectional_hillshade(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute multidirectional hillshade.
    """
    ...

def natural_neighbor_interpolation(points_xy: npt.NDArray[np.float64], values: npt.NDArray[np.float64], grid_rows: int = 100, grid_cols: int = 100, cellsize: float = 1.0) -> npt.NDArray[np.float64]:
    """Natural neighbor (Sibson) interpolation from scattered points to a raster grid.
    """
    ...

def nbr_compute(nir: npt.NDArray[np.float64], swir: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NBR (Normalized Burn Ratio) from NIR and SWIR bands.
    """
    ...

def ndbi_compute(swir: npt.NDArray[np.float64], nir: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NDBI (Normalized Difference Built-up Index) from SWIR and NIR bands.
    """
    ...

def ndmi_compute(nir: npt.NDArray[np.float64], swir: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NDMI (Normalized Difference Moisture Index) from NIR and SWIR bands.
    """
    ...

def ndre_compute(nir: npt.NDArray[np.float64], red_edge: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NDRE (Normalized Difference Red Edge) from NIR and RedEdge bands.
    """
    ...

def ndsi_compute(green: npt.NDArray[np.float64], swir: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NDSI (Normalized Difference Snow Index) from Green and SWIR bands.
    """
    ...

def ndvi_compute(nir: npt.NDArray[np.float64], red: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NDVI from NIR and Red bands.
    """
    ...

def ndwi_compute(green: npt.NDArray[np.float64], nir: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NDWI from Green and NIR bands.
    """
    ...

def nearest_neighbor_interpolation(points_xy: npt.NDArray[np.float64], values: npt.NDArray[np.float64], grid_rows: int = 100, grid_cols: int = 100, cellsize: float = 1.0) -> npt.NDArray[np.float64]:
    """Nearest-neighbor interpolation from scattered points to a raster grid.
    """
    ...

def ngrdi_compute(green: npt.NDArray[np.float64], red: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute NGRDI (Normalized Green-Red Difference) from Green and Red bands.
    """
    ...

def normalized_diff(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute normalized difference: (A - B) / (A + B).
    """
    ...

def northness_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute northness (cos of aspect).
    """
    ...

def openness_negative(dem: npt.NDArray[np.float64], cell_size: float = 1.0, directions: int = 8, radius: int = 10) -> npt.NDArray[np.float64]:
    """Compute negative openness (below-horizon enclosure).
    """
    ...

def openness_positive(dem: npt.NDArray[np.float64], cell_size: float = 1.0, directions: int = 8, radius: int = 10) -> npt.NDArray[np.float64]:
    """Compute positive openness (above-horizon visibility).
    """
    ...

def ordinary_kriging_interpolation(points_xy: npt.NDArray[np.float64], values: npt.NDArray[np.float64], grid_rows: int = 100, grid_cols: int = 100, cellsize: float = 1.0, max_points: int = 16) -> npt.NDArray[np.float64]:
    """Ordinary Kriging interpolation from scattered points to a raster grid.
Points given as (N,2) array of (x,y) coords, values as (N,1) array —
same input shape as `idw_interpolation`/`nearest_neighbor_interpolation`.

Fits the best-RSS variogram model (spherical/exponential/Gaussian)
automatically from the empirical semivariance, then krige with it —
the common case; use the lower-level Rust API directly if you need to
choose or inspect the variogram model.
    """
    ...

def pansharpen_compute(pan: npt.NDArray[np.float64], ms: npt.NDArray[np.float64], cell_size: float = 1.0, method: str = "brovey") -> npt.NDArray[np.float64]:
    """Sharpen a multispectral stack `(bands, rows, cols)` using a
co-registered panchromatic band, on the same grid. `method`: "brovey"
(default), "gram_schmidt"/"gs", or "pca". Returns a `(bands, rows,
cols)` array, same band count as the input `ms` stack.
    """
    ...

def patch_density_compute(classification: npt.NDArray[np.float64], radius: int = 3) -> npt.NDArray[np.float64]:
    """Compute patch density in a moving window.
    """
    ...

def predict_raster(rasters: list[npt.NDArray[np.float64]], predict_fn: typing.Callable[..., typing.Any], batch_size: int = 100000) -> npt.NDArray[np.float64]:
    """Apply a Python prediction function to a stack of rasters, producing a prediction raster.

Args:
    rasters: list of 2D numpy arrays (all same shape), one per feature
    predict_fn: Python callable that takes a 2D array (N, n_features) and returns 1D array (N,)
    batch_size: number of pixels per batch (default 100000)

Returns:
    2D numpy array with predictions. NaN where any input raster has NaN.

Example:
    import surtgis, xgboost as xgb
    model = xgb.XGBRegressor().fit(X_train, y_train)
    prediction = surtgis.predict_raster([slope, aspect, tpi], model.predict)

GIL note: this function intentionally does NOT release the GIL — it calls
back into the Python `predict_fn` for every batch, which requires holding
the GIL by construction.
    """
    ...

def priority_flood_fill(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Priority-flood depression filling (Barnes 2014).
    """
    ...

def priority_flood_flat_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Priority-flood with flat resolution.
    """
    ...

def reci_compute(nir: npt.NDArray[np.float64], red_edge: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute RECI (Red Edge Chlorophyll Index) from NIR and RedEdge bands.
    """
    ...

def relief_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, colormap: str = "terrain", sun_azimuth: float = 315.0, sun_altitude: float = 45.0, shadows: bool = True, ambient: bool = False) -> npt.NDArray[np.uint8]:
    """SurtGis: High-performance geospatial analysis from Python.

Usage:
    import numpy as np
    import surtgis

    dem = np.random.rand(100, 100) * 1000  # elevation in meters
    slp = surtgis.slope(dem, cell_size=30.0)
    asp = surtgis.aspect_degrees(dem, cell_size=30.0)
    hs  = surtgis.hillshade_compute(dem, cell_size=30.0)
Full shaded-relief composite (terrain colormap + sphere shade + optional
ray-traced shadows + optional ambient occlusion). Returns an
`(H, W, 4) uint8` numpy array in row-major RGBA order.

`colormap` is one of `terrain`, `divergent`, `grayscale`, `ndvi`, `bwr`,
`geomorphons`, `water`, `accumulation`. Unknown names fall back to
`terrain`.
    """
    ...

def sar_db_to_linear(backscatter_db: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Convert decibel SAR backscatter back to linear power (10^(x/10)).
    """
    ...

def sar_dual_pol_water_index(co_pol: npt.NDArray[np.float64], cross_pol: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Dual-pol SAR water index: (co_pol - cross_pol) / (co_pol + cross_pol).

For Sentinel-1, pass co_pol = VV and cross_pol = VH (linear power).
    """
    ...

def sar_lee_filter(image: npt.NDArray[np.float64], window_size: int = 7, looks: float = 1.0, cell_size: float = 1.0, refined: bool = False) -> npt.NDArray[np.float64]:
    """Lee adaptive speckle filter for SAR backscatter.

Args:
    image: 2D numpy array (f64), single-band backscatter (linear power).
    window_size: odd window side length (default 7).
    looks: equivalent number of looks / ENL (default 1.0).
    cell_size: cell size in map units.
    refined: use the edge-aligned refined Lee (1981) instead of the classic
        Lee (1980); preserves edges and linear features better (default False).
Returns:
    2D numpy array (f64), speckle-filtered.
    """
    ...

def sar_linear_to_db(backscatter: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Convert linear-power SAR backscatter to decibels (10·log10).

Args:
    backscatter: 2D numpy array (f64), linear power.
    cell_size: cell size in map units.
Returns:
    2D numpy array of dB values (non-positive inputs -> NaN).
    """
    ...

def sar_water_mask(raster: npt.NDArray[np.float64], threshold: float, water_above: bool = False, cell_size: float = 1.0) -> npt.NDArray[np.uint8]:
    """Threshold a backscatter or index raster into a binary water mask.

Args:
    raster: 2D numpy array (f64), backscatter (dB) or a water index.
    threshold: decision boundary.
    water_above: if True, water is ABOVE the threshold (water index);
        if False (default), water is below it (backscatter).
    cell_size: cell size in map units.
Returns:
    2D numpy array (uint8): 1=water, 0=land, 255=nodata.
    """
    ...

def savi_compute(nir: npt.NDArray[np.float64], red: npt.NDArray[np.float64], l_factor: float = 0.5) -> npt.NDArray[np.float64]:
    """Compute SAVI from NIR and Red bands.
    """
    ...

def shannon_diversity_compute(classification: npt.NDArray[np.float64], radius: int = 3) -> npt.NDArray[np.float64]:
    """Compute Shannon Diversity Index in a moving window.
    """
    ...

def shape_index_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute shape index.
    """
    ...

def simpson_diversity_compute(classification: npt.NDArray[np.float64], radius: int = 3) -> npt.NDArray[np.float64]:
    """Compute Simpson Diversity Index in a moving window.
    """
    ...

def sky_view_factor_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, directions: int = 16, radius: int = 10) -> npt.NDArray[np.float64]:
    """Compute sky view factor.
    """
    ...

def slic_compute(bands: npt.NDArray[np.float64], n_segments: int = 100, compactness: float = 0.1, cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """SLIC superpixel segmentation over a `(bands, rows, cols)` multi-band
stack. Returns an integer label raster (as f64, matching this binding's
other integer-raster outputs like `watershed_compute`).
    """
    ...

def slope(dem: npt.NDArray[np.float64], cell_size: float = 1.0, units: str = "degrees") -> npt.NDArray[np.float64]:
    """Compute slope from a DEM.

Args:
    dem: 2D numpy array (f64) of elevation values
    cell_size: Cell size in map units (meters)
    units: "degrees" (default) or "percent"

Returns:
    2D numpy array of slope values
    """
    ...

def sobel_edge_compute(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Sobel edge detection (gradient magnitude).
    """
    ...

def solar_radiation_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, latitude: float = 45.0, day: int = 172) -> npt.NDArray[np.float64]:
    """Compute solar radiation (daily total Wh/m²) from a DEM.
    """
    ...

def stac_search(catalog: str, bbox: tuple[float, float, float, float] | None = None, datetime: str | None = None, collections: list[str] | None = None, limit: int | None = None) -> list[dict[str, typing.Any]]:
    """Search a STAC catalog and return the matching items.

Args:
    catalog: "pc" (Planetary Computer), "es"/"earth-search" (Earth Search),
        or a STAC API root URL.
    bbox: optional (min_x, min_y, max_x, max_y) in WGS84 lon/lat.
    datetime: optional ISO range, e.g. "2023-01-01/2023-12-31".
    collections: optional list of collection ids.
    limit: optional per-page item cap.

Returns:
    A list of dicts, one per item: {id, collection, datetime, bbox,
    assets: {key: href}}. Hrefs from Planetary Computer must be signed with
    [`stac_sign_href`] before passing them to [`cog_fetch`].
    """
    ...

def stac_sign_href(href: str, collection: str, catalog: str = "pc") -> str:
    """Sign a Planetary Computer asset href so it can be fetched.

PC asset hrefs are private blob-store URLs that require a short-lived SAS
token. Pass the unsigned href from [`stac_search`] and the item's collection
id; the returned URL can be handed directly to [`cog_fetch`].

Args:
    href: the unsigned asset href.
    collection: the STAC collection id (e.g. "sentinel-2-l2a").
    catalog: catalog to sign against (default "pc").
    """
    ...

def strahler_order_compute(fdir: npt.NDArray[np.uint8], stream_mask: npt.NDArray[np.uint8], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute Strahler stream order from flow direction and stream mask.
    """
    ...

def stream_network_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, threshold: float = 1000.0) -> npt.NDArray[np.uint8]:
    """Extract stream network from flow accumulation.
    """
    ...

def surface_area_ratio_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute surface area ratio (Jenness 2004).
    """
    ...

def tpi_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, radius: int = 3) -> npt.NDArray[np.float64]:
    """Compute TPI (Topographic Position Index).
    """
    ...

def tri_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute TRI (Terrain Ruggedness Index).
    """
    ...

def twi_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute TWI from a DEM (fill → flow_dir → flow_acc → slope → TWI).
    """
    ...

def uncertainty_slope(dem: npt.NDArray[np.float64], cell_size: float = 1.0, dem_rmse: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute uncertainty maps (slope RMSE) from DEM error.
    """
    ...

def valley_depth_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0) -> npt.NDArray[np.float64]:
    """Compute valley depth.
    """
    ...

def viewshed_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, observer_row: int = 0, observer_col: int = 0, observer_height: float = 1.8, target_height: float = 0.0, max_radius: int = 0, earth_curvature: bool = False, refraction_coeff: float = 0.14286) -> npt.NDArray[np.uint8]:
    """Compute viewshed from observer location.
    """
    ...

def viewshed_xdraw_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, observer_row: int = 0, observer_col: int = 0, observer_height: float = 1.8, target_height: float = 0.0, max_radius: int = 0, earth_curvature: bool = False, refraction_coeff: float = 0.14286) -> npt.NDArray[np.uint8]:
    """Compute XDraw viewshed (faster than Bresenham for large DEMs).
    """
    ...

def vrm_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, radius: int = 1) -> npt.NDArray[np.float64]:
    """Compute VRM (Vector Ruggedness Measure).
    """
    ...

def watershed_compute(fdir: npt.NDArray[np.uint8], cell_size: float = 1.0, pour_row: int = 0, pour_col: int = 0) -> npt.NDArray[np.float64]:
    """Delineate watershed from D8 flow direction with pour point.
    """
    ...

def wind_exposure_compute(dem: npt.NDArray[np.float64], cell_size: float = 1.0, directions: int = 8, max_distance: int = 30) -> npt.NDArray[np.float64]:
    """Compute wind exposure (Topex index).
    """
    ...

def zonal_statistics_compute(values: npt.NDArray[np.float64], zones: npt.NDArray[np.float64], cell_size: float = 1.0, statistic: str = "mean") -> npt.NDArray[np.float64]:
    """Zonal statistics: for each zone in `zones` (integer IDs), compute
`statistic` over the corresponding cells of `values`, mapped back onto
a raster of the same shape (every cell in a zone gets that zone's
value — the same "broadcast back to grid" shape as the focal_* family).

Zone id `0` means "no zone" (excluded from every statistic, its output
cells come back NaN) — the same convention as watershed/basin IDs
elsewhere in this crate. Use positive zone ids only.
    """
    ...

