//! Python bindings for SurtGis geospatial algorithms via PyO3.
//!
//! All terrain functions accept a 2D numpy array (f64) and a `cell_size` parameter.
//! Returns a 2D numpy array of the same shape.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_algorithms::terrain::{
    aspect, hillshade, curvature,
    slope as compute_slope,
    tpi as compute_tpi, tri as compute_tri,
    twi as compute_twi,
    geomorphons as compute_geomorphons,
    northness, eastness,
    dev as compute_dev,
    multidirectional_hillshade as compute_multi_hillshade,
    shape_index as compute_shape_index,
    curvedness as compute_curvedness,
    sky_view_factor as compute_svf,
    uncertainty as compute_uncertainty,
    viewshed as compute_viewshed,
    positive_openness, negative_openness,
    mrvbf as compute_mrvbf,
    vrm as compute_vrm,
    advanced_curvatures,
    AspectOutput, HillshadeParams, SlopeParams, SlopeUnits,
    CurvatureParams, CurvatureType, CurvatureFormula, DerivativeMethod,
    TpiParams, TriParams, GeomorphonParams, DevParams,
    MultiHillshadeParams, SvfParams, OpennessParams, MrvbfParams, VrmParams,
    UncertaintyParams, ViewshedParams, AdvancedCurvatureType,
    // New terrain imports
    solar_radiation as compute_solar_radiation, SolarParams,
    surface_area_ratio as compute_surface_area_ratio, SarParams,
    valley_depth as compute_valley_depth,
    convergence_index as compute_convergence_index, ConvergenceParams,
    landform_classification as compute_landform_classification, LandformParams,
    wind_exposure as compute_wind_exposure, WindExposureParams,
    contour_lines as compute_contour_lines, ContourParams,
    cost_distance as compute_cost_distance, CostDistanceParams,
    feature_preserving_smoothing as compute_fps, SmoothingParams,
    gaussian_smoothing as compute_gaussian_smoothing, GaussianSmoothingParams,
    log_transform as compute_log_transform,
    accumulation_zones as compute_accumulation_zones,
    viewshed_xdraw as compute_viewshed_xdraw,
    horizon_angle_map as compute_horizon_angle_map,
};
use surtgis_algorithms::hydrology::{
    fill_sinks, flow_direction, flow_accumulation,
    flow_direction_dinf as compute_flow_direction_dinf,
    priority_flood, hand as compute_hand,
    breach_depressions, flow_accumulation_mfd,
    stream_network as compute_stream_network,
    FillSinksParams, PriorityFloodParams, HandParams,
    BreachParams, MfdParams, StreamNetworkParams,
    // New hydrology imports
    watershed as compute_watershed, WatershedParams,
    flow_accumulation_tfga as compute_tfga, TfgaParams,
    flow_accumulation_mfd_adaptive as compute_adaptive_mfd, AdaptiveMfdParams,
    priority_flood_flat as compute_priority_flood_flat,
    strahler_order as compute_strahler_order,
    flow_path_length as compute_flow_path_length,
};
use surtgis_algorithms::imagery::{
    ndvi as compute_ndvi, ndwi as compute_ndwi, savi as compute_savi,
    normalized_difference,
    SaviParams,
    // New imagery imports
    mndwi as compute_mndwi, nbr as compute_nbr,
    evi as compute_evi, EviParams,
    bsi as compute_bsi,
    ndre as compute_ndre, gndvi as compute_gndvi, ngrdi as compute_ngrdi,
    reci as compute_reci,
    ndsi as compute_ndsi, ndbi as compute_ndbi, ndmi as compute_ndmi,
    msavi as compute_msavi, evi2 as compute_evi2,
};
use surtgis_algorithms::morphology::{
    erode as compute_erode, dilate as compute_dilate,
    opening as compute_opening, closing as compute_closing,
    gradient as compute_gradient, top_hat as compute_top_hat, black_hat as compute_black_hat,
    StructuringElement,
};
use surtgis_algorithms::statistics::{
    focal_statistics, FocalStatistic, FocalParams,
};
use surtgis_algorithms::classification::{
    kmeans_raster as compute_kmeans, KmeansParams,
    isodata as compute_isodata, IsodataParams,
};
use surtgis_algorithms::interpolation::{
    idw as compute_idw, IdwParams, SamplePoint,
    nearest_neighbor as compute_nearest_neighbor, NearestNeighborParams,
    natural_neighbor as compute_natural_neighbor, NaturalNeighborParams,
};
use surtgis_algorithms::landscape::{
    label_patches as compute_label_patches, Connectivity,
    shannon_diversity as compute_shannon_diversity,
    simpson_diversity as compute_simpson_diversity,
    patch_density as compute_patch_density,
    DiversityParams,
    landscape_metrics as compute_landscape_metrics,
};
use surtgis_algorithms::texture::{
    sobel_edge as compute_sobel_edge,
    laplacian as compute_laplacian,
    haralick_glcm as compute_haralick_glcm, GlcmParams, GlcmTexture,
};

/// Build a Raster<f64> from a numpy 2D array and cell_size.
fn numpy_to_raster(arr: &PyReadonlyArray2<'_, f64>, cell_size: f64) -> PyResult<Raster<f64>> {
    let shape = arr.shape();
    let rows = shape[0];
    let cols = shape[1];
    let data: Vec<f64> = arr.as_slice()
        .map_err(|_| PyValueError::new_err("array must be contiguous C-order"))?
        .to_vec();
    let mut raster = Raster::from_vec(data, rows, cols)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    raster.set_transform(GeoTransform::new(0.0, 0.0, cell_size, -cell_size));
    Ok(raster)
}

/// Build a Raster<u8> from a numpy 2D u8 array and cell_size.
fn numpy_u8_to_raster(arr: &PyReadonlyArray2<'_, u8>, cell_size: f64) -> PyResult<Raster<u8>> {
    let shape = arr.shape();
    let rows = shape[0];
    let cols = shape[1];
    let data: Vec<u8> = arr.as_slice()
        .map_err(|_| PyValueError::new_err("array must be contiguous C-order"))?
        .to_vec();
    let mut raster = Raster::from_vec(data, rows, cols)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    raster.set_transform(GeoTransform::new(0.0, 0.0, cell_size, -cell_size));
    Ok(raster)
}

/// Convert Raster<f64> back to a numpy 2D array.
fn raster_to_numpy<'py>(py: Python<'py>, raster: &Raster<f64>) -> Bound<'py, PyArray2<f64>> {
    let arr: Array2<f64> = raster.data().clone();
    arr.into_pyarray(py)
}

/// Convert Raster<u8> back to a numpy 2D array.
fn raster_u8_to_numpy<'py>(py: Python<'py>, raster: &Raster<u8>) -> Bound<'py, PyArray2<u8>> {
    let arr: Array2<u8> = raster.data().clone();
    arr.into_pyarray(py)
}

/// Convert Raster<i32> back to a numpy 2D f64 array (cast i32 to f64).
fn raster_i32_to_numpy_f64<'py>(py: Python<'py>, raster: &Raster<i32>) -> Bound<'py, PyArray2<f64>> {
    let arr: Array2<i32> = raster.data().clone();
    let arr_f64 = arr.mapv(|v| v as f64);
    arr_f64.into_pyarray(py)
}

// ===========================================================================
// Terrain
// ===========================================================================

/// Compute slope from a DEM.
///
/// Args:
///     dem: 2D numpy array (f64) of elevation values
///     cell_size: Cell size in map units (meters)
///     units: "degrees" (default) or "percent"
///
/// Returns:
///     2D numpy array of slope values
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, units="degrees"))]
fn slope<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    units: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let slope_units = match units {
        "percent" | "pct" => SlopeUnits::Percent,
        _ => SlopeUnits::Degrees,
    };
    let result = compute_slope(&raster, SlopeParams { units: slope_units, z_factor: 1.0 })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute aspect in degrees (0-360, 0=North, clockwise).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn aspect_degrees<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = aspect(&raster, AspectOutput::Degrees)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute hillshade from a DEM.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, azimuth=315.0, altitude=45.0))]
fn hillshade_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    azimuth: f64,
    altitude: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = hillshade(&raster, HillshadeParams { azimuth, altitude, z_factor: 1.0, normalized: false })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute multidirectional hillshade.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn multidirectional_hillshade<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_multi_hillshade(&raster, MultiHillshadeParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute curvature. `ctype`: "general", "profile", or "plan".
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, ctype="general"))]
fn curvature_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    ctype: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let ct = match ctype {
        "profile" => CurvatureType::Profile,
        "plan" | "tangential" => CurvatureType::Plan,
        _ => CurvatureType::General,
    };
    let result = curvature(&raster, CurvatureParams { curvature_type: ct, method: DerivativeMethod::EvansYoung, formula: CurvatureFormula::Full, z_factor: 1.0 })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute TPI (Topographic Position Index).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, radius=3))]
fn tpi_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_tpi(&raster, TpiParams { radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute TRI (Terrain Ruggedness Index).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn tri_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_tri(&raster, TriParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute geomorphons landform classification. Returns u8 landform codes.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, flatness=1.0, radius=10))]
fn geomorphons_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    flatness: f64,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_geomorphons(&raster, GeomorphonParams { flatness_threshold: flatness, radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_u8_to_numpy(py, &result))
}

/// Compute northness (cos of aspect).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn northness_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = northness(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute eastness (sin of aspect).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn eastness_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = eastness(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute DEV (Deviation from Mean Elevation).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, radius=10))]
fn dev_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_dev(&raster, DevParams { radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute shape index.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn shape_index_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_shape_index(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute curvedness.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn curvedness_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_curvedness(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute sky view factor.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, directions=16, radius=10))]
fn sky_view_factor_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    directions: usize,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_svf(&raster, SvfParams { directions, radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute uncertainty maps (slope RMSE) from DEM error.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, dem_rmse=1.0))]
fn uncertainty_slope<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    dem_rmse: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_uncertainty(&raster, UncertaintyParams { dem_rmse })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result.slope_rmse))
}

/// Compute viewshed from observer location.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, observer_row=0, observer_col=0, observer_height=1.8, target_height=0.0, max_radius=0))]
fn viewshed_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    observer_row: usize,
    observer_col: usize,
    observer_height: f64,
    target_height: f64,
    max_radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_viewshed(&raster, ViewshedParams {
        observer_row, observer_col,
        observer_height, target_height,
        max_radius,
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_u8_to_numpy(py, &result))
}

/// Compute positive openness (above-horizon visibility).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, directions=8, radius=10))]
fn openness_positive<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    directions: usize,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = positive_openness(&raster, OpennessParams { directions, radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute negative openness (below-horizon enclosure).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, directions=8, radius=10))]
fn openness_negative<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    directions: usize,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = negative_openness(&raster, OpennessParams { directions, radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute MRVBF (Multi-resolution Valley Bottom Flatness).
/// Returns a tuple (mrvbf, mrrtf).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn mrvbf_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let (mrvbf, mrrtf) = compute_mrvbf(&raster, MrvbfParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((raster_to_numpy(py, &mrvbf), raster_to_numpy(py, &mrrtf)))
}

/// Compute VRM (Vector Ruggedness Measure).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, radius=1))]
fn vrm_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_vrm(&raster, VrmParams { radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute advanced curvature (Florinsky system).
/// ctype options: "mean_h", "gaussian_k", "unsphericity_m", "difference_e",
/// "minimal_kmin", "maximal_kmax", "horizontal_kh", "vertical_kv",
/// "horizontal_excess_khe", "vertical_excess_kve", "accumulation_ka",
/// "ring_kr", "rotor", "laplacian"
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, ctype="mean_h"))]
fn advanced_curvature<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    ctype: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let curv_type = match ctype {
        "mean_h" | "mean" => AdvancedCurvatureType::MeanH,
        "gaussian_k" | "gaussian" => AdvancedCurvatureType::GaussianK,
        "unsphericity_m" | "unsphericity" => AdvancedCurvatureType::UnsphericitytM,
        "difference_e" | "difference" => AdvancedCurvatureType::DifferenceE,
        "minimal_kmin" | "kmin" => AdvancedCurvatureType::MinimalKmin,
        "maximal_kmax" | "kmax" => AdvancedCurvatureType::MaximalKmax,
        "horizontal_kh" | "kh" => AdvancedCurvatureType::HorizontalKh,
        "vertical_kv" | "kv" => AdvancedCurvatureType::VerticalKv,
        "horizontal_excess_khe" | "khe" => AdvancedCurvatureType::HorizontalExcessKhe,
        "vertical_excess_kve" | "kve" => AdvancedCurvatureType::VerticalExcessKve,
        "accumulation_ka" | "ka" => AdvancedCurvatureType::AccumulationKa,
        "ring_kr" | "kr" => AdvancedCurvatureType::RingKr,
        "rotor" => AdvancedCurvatureType::Rotor,
        "laplacian" => AdvancedCurvatureType::Laplacian,
        _ => return Err(PyValueError::new_err(format!("Unknown curvature type: {}", ctype))),
    };
    let result = advanced_curvatures(&raster, curv_type)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// Hydrology
// ===========================================================================

/// Fill sinks (depressions) in a DEM.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn fill_depressions<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = fill_sinks(&raster, FillSinksParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Priority-flood depression filling (Barnes 2014).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn priority_flood_fill<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = priority_flood(&raster, PriorityFloodParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute D8 flow direction. Returns u8 direction codes.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn flow_direction_d8<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = flow_direction(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_u8_to_numpy(py, &result))
}

/// Compute D-infinity flow direction (Tarboton 1997).
/// Returns continuous flow angles in radians [0, 2π), -1 for pits.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn flow_direction_dinf<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_flow_direction_dinf(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute flow accumulation from D8 flow direction raster.
#[pyfunction]
#[pyo3(signature = (fdir, cell_size=1.0))]
fn flow_accumulation_d8<'py>(
    py: Python<'py>,
    fdir: PyReadonlyArray2<'py, u8>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_u8_to_raster(&fdir, cell_size)?;
    let result = flow_accumulation(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute TWI from a DEM (fill → flow_dir → flow_acc → slope → TWI).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn twi_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let filled = fill_sinks(&raster, FillSinksParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let fdir = flow_direction(&filled)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let facc = flow_accumulation(&fdir)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let slp = compute_slope(&raster, SlopeParams { units: SlopeUnits::Degrees, z_factor: 1.0 })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = compute_twi(&facc, &slp)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute HAND from a DEM (fill → flow_dir → flow_acc → HAND).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, stream_threshold=1000.0))]
fn hand_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    stream_threshold: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let filled = fill_sinks(&raster, FillSinksParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let fdir = flow_direction(&filled)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let facc = flow_accumulation(&fdir)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = compute_hand(&raster, &fdir, &facc, HandParams { stream_threshold })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Breach depressions (Lindsay 2016) - preferred over filling.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn breach_fill<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = breach_depressions(&raster, BreachParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute MFD (Multiple Flow Direction) accumulation.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, exponent=1.1))]
fn flow_accumulation_mfd_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    exponent: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let filled = fill_sinks(&raster, FillSinksParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = flow_accumulation_mfd(&filled, MfdParams { exponent })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Extract stream network from flow accumulation.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, threshold=1000.0))]
fn stream_network_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let filled = fill_sinks(&raster, FillSinksParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let fdir = flow_direction(&filled)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let facc = flow_accumulation(&fdir)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = compute_stream_network(&facc, StreamNetworkParams { threshold })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_u8_to_numpy(py, &result))
}

// ===========================================================================
// Imagery
// ===========================================================================

/// Compute NDVI from NIR and Red bands.
#[pyfunction]
fn ndvi_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let red_r = numpy_to_raster(&red, 1.0)?;
    let result = compute_ndvi(&nir_r, &red_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute NDWI from Green and NIR bands.
#[pyfunction]
fn ndwi_compute<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f64>,
    nir: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let green_r = numpy_to_raster(&green, 1.0)?;
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let result = compute_ndwi(&green_r, &nir_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute SAVI from NIR and Red bands.
#[pyfunction]
#[pyo3(signature = (nir, red, l_factor=0.5))]
fn savi_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
    l_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let red_r = numpy_to_raster(&red, 1.0)?;
    let result = compute_savi(&nir_r, &red_r, SaviParams { l_factor })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute normalized difference: (A - B) / (A + B).
#[pyfunction]
fn normalized_diff<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_r = numpy_to_raster(&a, 1.0)?;
    let b_r = numpy_to_raster(&b, 1.0)?;
    let result = normalized_difference(&a_r, &b_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute MNDWI (Modified NDWI) from Green and SWIR bands.
#[pyfunction]
fn mndwi_compute<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f64>,
    swir: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let green_r = numpy_to_raster(&green, 1.0)?;
    let swir_r = numpy_to_raster(&swir, 1.0)?;
    let result = compute_mndwi(&green_r, &swir_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute NBR (Normalized Burn Ratio) from NIR and SWIR bands.
#[pyfunction]
fn nbr_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    swir: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let swir_r = numpy_to_raster(&swir, 1.0)?;
    let result = compute_nbr(&nir_r, &swir_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute EVI (Enhanced Vegetation Index) from NIR, Red, and Blue bands.
#[pyfunction]
#[pyo3(signature = (nir, red, blue, g=2.5, c1=6.0, c2=7.5, l=1.0))]
fn evi_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
    blue: PyReadonlyArray2<'py, f64>,
    g: f64,
    c1: f64,
    c2: f64,
    l: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let red_r = numpy_to_raster(&red, 1.0)?;
    let blue_r = numpy_to_raster(&blue, 1.0)?;
    let result = compute_evi(&nir_r, &red_r, &blue_r, EviParams { g, c1, c2, l })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute BSI (Bare Soil Index) from SWIR, NIR, Red, and Blue bands.
#[pyfunction]
fn bsi_compute<'py>(
    py: Python<'py>,
    swir: PyReadonlyArray2<'py, f64>,
    nir: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
    blue: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let swir_r = numpy_to_raster(&swir, 1.0)?;
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let red_r = numpy_to_raster(&red, 1.0)?;
    let blue_r = numpy_to_raster(&blue, 1.0)?;
    let result = compute_bsi(&swir_r, &red_r, &nir_r, &blue_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute NDRE (Normalized Difference Red Edge) from NIR and RedEdge bands.
#[pyfunction]
fn ndre_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    red_edge: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let re_r = numpy_to_raster(&red_edge, 1.0)?;
    let result = compute_ndre(&nir_r, &re_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute GNDVI (Green NDVI) from NIR and Green bands.
#[pyfunction]
fn gndvi_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    green: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let green_r = numpy_to_raster(&green, 1.0)?;
    let result = compute_gndvi(&nir_r, &green_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute NGRDI (Normalized Green-Red Difference) from Green and Red bands.
#[pyfunction]
fn ngrdi_compute<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let green_r = numpy_to_raster(&green, 1.0)?;
    let red_r = numpy_to_raster(&red, 1.0)?;
    let result = compute_ngrdi(&green_r, &red_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute RECI (Red Edge Chlorophyll Index) from NIR and RedEdge bands.
#[pyfunction]
fn reci_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    red_edge: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let re_r = numpy_to_raster(&red_edge, 1.0)?;
    let result = compute_reci(&nir_r, &re_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute NDSI (Normalized Difference Snow Index) from Green and SWIR bands.
#[pyfunction]
fn ndsi_compute<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f64>,
    swir: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let green_r = numpy_to_raster(&green, 1.0)?;
    let swir_r = numpy_to_raster(&swir, 1.0)?;
    let result = compute_ndsi(&green_r, &swir_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute NDBI (Normalized Difference Built-up Index) from SWIR and NIR bands.
#[pyfunction]
fn ndbi_compute<'py>(
    py: Python<'py>,
    swir: PyReadonlyArray2<'py, f64>,
    nir: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let swir_r = numpy_to_raster(&swir, 1.0)?;
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let result = compute_ndbi(&swir_r, &nir_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute NDMI (Normalized Difference Moisture Index) from NIR and SWIR bands.
#[pyfunction]
fn ndmi_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    swir: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let swir_r = numpy_to_raster(&swir, 1.0)?;
    let result = compute_ndmi(&nir_r, &swir_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute MSAVI (Modified Soil-Adjusted Vegetation Index) from NIR and Red bands.
#[pyfunction]
fn msavi_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let red_r = numpy_to_raster(&red, 1.0)?;
    let result = compute_msavi(&nir_r, &red_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute EVI2 (Two-band Enhanced Vegetation Index) from NIR and Red bands.
#[pyfunction]
fn evi2_compute<'py>(
    py: Python<'py>,
    nir: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nir_r = numpy_to_raster(&nir, 1.0)?;
    let red_r = numpy_to_raster(&red, 1.0)?;
    let result = compute_evi2(&nir_r, &red_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// Morphology
// ===========================================================================

/// Morphological erosion with square kernel.
#[pyfunction]
#[pyo3(signature = (data, radius=1))]
fn morph_erode<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let elem = StructuringElement::Square(radius);
    let result = compute_erode(&raster, &elem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Morphological dilation with square kernel.
#[pyfunction]
#[pyo3(signature = (data, radius=1))]
fn morph_dilate<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let elem = StructuringElement::Square(radius);
    let result = compute_dilate(&raster, &elem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Morphological opening (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (data, radius=1))]
fn morph_opening<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let elem = StructuringElement::Square(radius);
    let result = compute_opening(&raster, &elem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Morphological closing (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (data, radius=1))]
fn morph_closing<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let elem = StructuringElement::Square(radius);
    let result = compute_closing(&raster, &elem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Morphological gradient (dilation minus erosion).
#[pyfunction]
#[pyo3(signature = (data, radius=1))]
fn morph_gradient<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let elem = StructuringElement::Square(radius);
    let result = compute_gradient(&raster, &elem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Top-hat transform (original minus opening). Extracts bright features.
#[pyfunction]
#[pyo3(signature = (data, radius=1))]
fn morph_top_hat<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let elem = StructuringElement::Square(radius);
    let result = compute_top_hat(&raster, &elem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Black-hat transform (closing minus original). Extracts dark features.
#[pyfunction]
#[pyo3(signature = (data, radius=1))]
fn morph_black_hat<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let elem = StructuringElement::Square(radius);
    let result = compute_black_hat(&raster, &elem)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// Statistics
// ===========================================================================

/// Focal mean statistic.
#[pyfunction]
#[pyo3(signature = (data, radius=1, circular=false))]
fn focal_mean<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    circular: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = focal_statistics(&raster, FocalParams { statistic: FocalStatistic::Mean, radius, circular })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Focal standard deviation.
#[pyfunction]
#[pyo3(signature = (data, radius=1, circular=false))]
fn focal_std<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    circular: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = focal_statistics(&raster, FocalParams { statistic: FocalStatistic::StdDev, radius, circular })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Focal range (max - min).
#[pyfunction]
#[pyo3(signature = (data, radius=1, circular=false))]
fn focal_range<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    circular: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = focal_statistics(&raster, FocalParams { statistic: FocalStatistic::Range, radius, circular })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Focal minimum statistic.
#[pyfunction]
#[pyo3(signature = (data, radius=1, circular=false))]
fn focal_min<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    circular: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = focal_statistics(&raster, FocalParams { statistic: FocalStatistic::Min, radius, circular })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Focal maximum statistic.
#[pyfunction]
#[pyo3(signature = (data, radius=1, circular=false))]
fn focal_max<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    circular: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = focal_statistics(&raster, FocalParams { statistic: FocalStatistic::Max, radius, circular })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Focal sum statistic.
#[pyfunction]
#[pyo3(signature = (data, radius=1, circular=false))]
fn focal_sum<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    circular: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = focal_statistics(&raster, FocalParams { statistic: FocalStatistic::Sum, radius, circular })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Focal median statistic.
#[pyfunction]
#[pyo3(signature = (data, radius=1, circular=false))]
fn focal_median<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    circular: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = focal_statistics(&raster, FocalParams { statistic: FocalStatistic::Median, radius, circular })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// New Terrain Functions
// ===========================================================================

/// Compute solar radiation (daily total Wh/m²) from a DEM.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, latitude=45.0, day=172))]
fn solar_radiation_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    latitude: f64,
    day: u32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    // Compute slope and aspect in degrees, then convert to radians
    let slope_deg = compute_slope(&raster, SlopeParams { units: SlopeUnits::Degrees, z_factor: 1.0 })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let aspect_deg = aspect(&raster, AspectOutput::Degrees)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut slope_rad = slope_deg;
    for v in slope_rad.data_mut().iter_mut() {
        if v.is_finite() { *v = v.to_radians(); }
    }
    let mut aspect_rad = aspect_deg;
    for v in aspect_rad.data_mut().iter_mut() {
        if v.is_finite() { *v = v.to_radians(); }
    }
    let params = SolarParams { day, latitude, ..SolarParams::default() };
    let result = compute_solar_radiation(&slope_rad, &aspect_rad, params)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result.total))
}

/// Compute surface area ratio (Jenness 2004).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn surface_area_ratio_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_surface_area_ratio(&raster, SarParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute valley depth.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn valley_depth_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_valley_depth(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute convergence index.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, radius=1))]
fn convergence_index_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_convergence_index(&raster, ConvergenceParams { radius })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute landform classification (Weiss 2001, 11 classes).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn landform_classification_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_landform_classification(&raster, LandformParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute wind exposure (Topex index).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, directions=8, max_distance=30))]
fn wind_exposure_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    directions: usize,
    max_distance: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_wind_exposure(&raster, WindExposureParams {
        directions,
        radius: max_distance,
        ..WindExposureParams::default()
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Generate contour lines as a raster.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, interval=10.0))]
fn contour_lines_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    interval: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_contour_lines(&raster, ContourParams { interval, base: 0.0 })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute cost distance from source cells.
#[pyfunction]
#[pyo3(signature = (cost_surface, cell_size=1.0, source_row=0, source_col=0))]
fn cost_distance_compute<'py>(
    py: Python<'py>,
    cost_surface: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    source_row: usize,
    source_col: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&cost_surface, cell_size)?;
    let result = compute_cost_distance(&raster, CostDistanceParams {
        sources: vec![(source_row, source_col)],
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Feature-preserving DEM smoothing.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, iterations=3, threshold=15.0))]
fn feature_preserving_smoothing_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    iterations: usize,
    threshold: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_fps(&raster, SmoothingParams { radius: 2, iterations, threshold })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Gaussian smoothing of a DEM.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, sigma=1.0))]
fn gaussian_smoothing_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    sigma: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let radius = (sigma * 3.0).ceil() as usize;
    let radius = if radius == 0 { 1 } else { radius };
    let result = compute_gaussian_smoothing(&raster, GaussianSmoothingParams { radius, sigma })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Sign-preserving log transform: f(x) = sign(x) * ln(1 + |x|).
#[pyfunction]
fn log_transform_compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = compute_log_transform(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Classify terrain into accumulation/dispersion zones.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn accumulation_zones_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_accumulation_zones(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute XDraw viewshed (faster than Bresenham for large DEMs).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, observer_row=0, observer_col=0, observer_height=1.8, target_height=0.0, max_radius=0))]
fn viewshed_xdraw_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    observer_row: usize,
    observer_col: usize,
    observer_height: f64,
    target_height: f64,
    max_radius: usize,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_viewshed_xdraw(&raster, ViewshedParams {
        observer_row, observer_col,
        observer_height, target_height,
        max_radius,
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_u8_to_numpy(py, &result))
}

/// Compute horizon angle map for a single azimuth direction.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, direction_degrees=0.0, max_distance=100))]
fn horizon_angle_map_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    direction_degrees: f64,
    max_distance: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let azimuth_rad = direction_degrees.to_radians();
    let result = compute_horizon_angle_map(&raster, azimuth_rad, max_distance)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// New Hydrology Functions
// ===========================================================================

/// Delineate watershed from D8 flow direction with pour point.
#[pyfunction]
#[pyo3(signature = (fdir, cell_size=1.0, pour_row=0, pour_col=0))]
fn watershed_compute<'py>(
    py: Python<'py>,
    fdir: PyReadonlyArray2<'py, u8>,
    cell_size: f64,
    pour_row: usize,
    pour_col: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_u8_to_raster(&fdir, cell_size)?;
    let params = if pour_row == 0 && pour_col == 0 {
        WatershedParams { pour_points: vec![] }
    } else {
        WatershedParams { pour_points: vec![(pour_row, pour_col)] }
    };
    let result = compute_watershed(&raster, params)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_i32_to_numpy_f64(py, &result))
}

/// TFGA (Facet-to-Facet) flow accumulation.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn flow_accumulation_tfga_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let filled = fill_sinks(&raster, FillSinksParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = compute_tfga(&filled, TfgaParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Adaptive MFD flow accumulation (Qin 2011).
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0, convergence=8.9))]
fn flow_accumulation_adaptive_mfd<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
    convergence: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let filled = fill_sinks(&raster, FillSinksParams::default())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let result = compute_adaptive_mfd(&filled, AdaptiveMfdParams {
        scale_factor: convergence,
        ..AdaptiveMfdParams::default()
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Priority-flood with flat resolution.
#[pyfunction]
#[pyo3(signature = (dem, cell_size=1.0))]
fn priority_flood_flat_compute<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&dem, cell_size)?;
    let result = compute_priority_flood_flat(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute Strahler stream order from flow direction and stream mask.
#[pyfunction]
#[pyo3(signature = (fdir, stream_mask, cell_size=1.0))]
fn strahler_order_compute<'py>(
    py: Python<'py>,
    fdir: PyReadonlyArray2<'py, u8>,
    stream_mask: PyReadonlyArray2<'py, u8>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let fdir_r = numpy_u8_to_raster(&fdir, cell_size)?;
    let mask_r = numpy_u8_to_raster(&stream_mask, cell_size)?;
    let result = compute_strahler_order(&fdir_r, &mask_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute flow path length from D8 flow direction.
#[pyfunction]
#[pyo3(signature = (fdir, cell_size=1.0))]
fn flow_path_length_compute<'py>(
    py: Python<'py>,
    fdir: PyReadonlyArray2<'py, u8>,
    cell_size: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let fdir_r = numpy_u8_to_raster(&fdir, cell_size)?;
    let result = compute_flow_path_length(&fdir_r)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// Classification
// ===========================================================================

/// K-means unsupervised classification on a single raster.
#[pyfunction]
#[pyo3(signature = (data, k=5, max_iterations=100))]
fn kmeans_compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    k: usize,
    max_iterations: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = compute_kmeans(&raster, KmeansParams {
        k,
        max_iterations,
        ..KmeansParams::default()
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// ISODATA unsupervised classification on a single raster.
#[pyfunction]
#[pyo3(signature = (data, initial_k=5, max_iterations=50))]
fn isodata_compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    initial_k: usize,
    max_iterations: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = compute_isodata(&raster, IsodataParams {
        initial_k,
        max_iterations,
        ..IsodataParams::default()
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// Interpolation
// ===========================================================================

/// IDW interpolation from scattered points to a raster grid.
/// Points given as (N,2) array of (x,y) coords, values as (N,) array.
#[pyfunction]
#[pyo3(signature = (points_xy, values, grid_rows=100, grid_cols=100, cellsize=1.0, power=2.0))]
fn idw_interpolation<'py>(
    py: Python<'py>,
    points_xy: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    grid_rows: usize,
    grid_cols: usize,
    cellsize: f64,
    power: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xy_shape = points_xy.shape();
    let n = xy_shape[0];
    if xy_shape[1] != 2 {
        return Err(PyValueError::new_err("points_xy must have shape (N, 2)"));
    }
    let val_shape = values.shape();
    if val_shape[0] != n || val_shape[1] != 1 {
        return Err(PyValueError::new_err("values must have shape (N, 1)"));
    }
    let xy_slice = points_xy.as_slice()
        .map_err(|_| PyValueError::new_err("points_xy must be contiguous"))?;
    let val_slice = values.as_slice()
        .map_err(|_| PyValueError::new_err("values must be contiguous"))?;
    let mut sample_points = Vec::with_capacity(n);
    for i in 0..n {
        sample_points.push(SamplePoint::new(xy_slice[i * 2], xy_slice[i * 2 + 1], val_slice[i]));
    }
    let transform = GeoTransform::new(0.0, grid_rows as f64 * cellsize, cellsize, -cellsize);
    let params = IdwParams {
        power,
        rows: grid_rows,
        cols: grid_cols,
        transform,
        ..IdwParams::default()
    };
    let result = compute_idw(&sample_points, params)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Nearest-neighbor interpolation from scattered points to a raster grid.
#[pyfunction]
#[pyo3(signature = (points_xy, values, grid_rows=100, grid_cols=100, cellsize=1.0))]
fn nearest_neighbor_interpolation<'py>(
    py: Python<'py>,
    points_xy: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    grid_rows: usize,
    grid_cols: usize,
    cellsize: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xy_shape = points_xy.shape();
    let n = xy_shape[0];
    if xy_shape[1] != 2 {
        return Err(PyValueError::new_err("points_xy must have shape (N, 2)"));
    }
    let val_shape = values.shape();
    if val_shape[0] != n || val_shape[1] != 1 {
        return Err(PyValueError::new_err("values must have shape (N, 1)"));
    }
    let xy_slice = points_xy.as_slice()
        .map_err(|_| PyValueError::new_err("points_xy must be contiguous"))?;
    let val_slice = values.as_slice()
        .map_err(|_| PyValueError::new_err("values must be contiguous"))?;
    let mut sample_points = Vec::with_capacity(n);
    for i in 0..n {
        sample_points.push(SamplePoint::new(xy_slice[i * 2], xy_slice[i * 2 + 1], val_slice[i]));
    }
    let transform = GeoTransform::new(0.0, grid_rows as f64 * cellsize, cellsize, -cellsize);
    let params = NearestNeighborParams {
        rows: grid_rows,
        cols: grid_cols,
        transform,
        ..NearestNeighborParams::default()
    };
    let result = compute_nearest_neighbor(&sample_points, params)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Natural neighbor (Sibson) interpolation from scattered points to a raster grid.
#[pyfunction]
#[pyo3(signature = (points_xy, values, grid_rows=100, grid_cols=100, cellsize=1.0))]
fn natural_neighbor_interpolation<'py>(
    py: Python<'py>,
    points_xy: PyReadonlyArray2<'py, f64>,
    values: PyReadonlyArray2<'py, f64>,
    grid_rows: usize,
    grid_cols: usize,
    cellsize: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let xy_shape = points_xy.shape();
    let n = xy_shape[0];
    if xy_shape[1] != 2 {
        return Err(PyValueError::new_err("points_xy must have shape (N, 2)"));
    }
    let val_shape = values.shape();
    if val_shape[0] != n || val_shape[1] != 1 {
        return Err(PyValueError::new_err("values must have shape (N, 1)"));
    }
    let xy_slice = points_xy.as_slice()
        .map_err(|_| PyValueError::new_err("points_xy must be contiguous"))?;
    let val_slice = values.as_slice()
        .map_err(|_| PyValueError::new_err("values must be contiguous"))?;
    let mut sample_points = Vec::with_capacity(n);
    for i in 0..n {
        sample_points.push(SamplePoint::new(xy_slice[i * 2], xy_slice[i * 2 + 1], val_slice[i]));
    }
    let transform = GeoTransform::new(0.0, grid_rows as f64 * cellsize, cellsize, -cellsize);
    let params = NaturalNeighborParams {
        rows: grid_rows,
        cols: grid_cols,
        transform,
        ..NaturalNeighborParams::default()
    };
    let result = compute_natural_neighbor(&sample_points, params)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// Landscape Ecology
// ===========================================================================

/// Label connected patches in a classification raster.
/// Returns (patch_labels as f64 array, number_of_patches).
#[pyfunction]
#[pyo3(signature = (classification, connectivity8=true))]
fn label_patches_compute<'py>(
    py: Python<'py>,
    classification: PyReadonlyArray2<'py, f64>,
    connectivity8: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, usize)> {
    let raster = numpy_to_raster(&classification, 1.0)?;
    let conn = if connectivity8 { Connectivity::Eight } else { Connectivity::Four };
    let (labels, n_patches) = compute_label_patches(&raster, conn)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((raster_i32_to_numpy_f64(py, &labels), n_patches))
}

/// Compute Shannon Diversity Index in a moving window.
#[pyfunction]
#[pyo3(signature = (classification, radius=3))]
fn shannon_diversity_compute<'py>(
    py: Python<'py>,
    classification: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&classification, 1.0)?;
    let result = compute_shannon_diversity(&raster, DiversityParams { radius, circular: false })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute Simpson Diversity Index in a moving window.
#[pyfunction]
#[pyo3(signature = (classification, radius=3))]
fn simpson_diversity_compute<'py>(
    py: Python<'py>,
    classification: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&classification, 1.0)?;
    let result = compute_simpson_diversity(&raster, DiversityParams { radius, circular: false })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute patch density in a moving window.
#[pyfunction]
#[pyo3(signature = (classification, radius=3))]
fn patch_density_compute<'py>(
    py: Python<'py>,
    classification: PyReadonlyArray2<'py, f64>,
    radius: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&classification, 1.0)?;
    let result = compute_patch_density(&raster, DiversityParams { radius, circular: false })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute landscape-level metrics (SHDI, SIDI, etc.).
/// Returns a tuple (shdi, sidi, num_patches, num_classes, total_area_m2, total_cells).
#[pyfunction]
fn landscape_metrics_compute<'py>(
    _py: Python<'py>,
    classification: PyReadonlyArray2<'py, f64>,
) -> PyResult<(f64, f64, usize, usize, f64, usize)> {
    let raster = numpy_to_raster(&classification, 1.0)?;
    let metrics = compute_landscape_metrics(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((metrics.shdi, metrics.sidi, metrics.num_patches, metrics.num_classes, metrics.total_area_m2, metrics.total_cells))
}

// ===========================================================================
// Texture
// ===========================================================================

/// Sobel edge detection (gradient magnitude).
#[pyfunction]
fn sobel_edge_compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = compute_sobel_edge(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Laplacian edge detection (second derivative).
#[pyfunction]
fn laplacian_compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let result = compute_laplacian(&raster)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

/// Compute GLCM (Haralick) texture feature.
/// texture: "contrast", "energy", "homogeneity", "correlation", "entropy", "dissimilarity"
#[pyfunction]
#[pyo3(signature = (data, radius=3, distance=1, texture="contrast"))]
fn haralick_glcm_compute<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    radius: usize,
    distance: usize,
    texture: &str,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let raster = numpy_to_raster(&data, 1.0)?;
    let tex = match texture {
        "energy" | "asm" => GlcmTexture::Energy,
        "contrast" => GlcmTexture::Contrast,
        "homogeneity" | "idm" => GlcmTexture::Homogeneity,
        "correlation" => GlcmTexture::Correlation,
        "entropy" => GlcmTexture::Entropy,
        "dissimilarity" => GlcmTexture::Dissimilarity,
        _ => return Err(PyValueError::new_err(format!("Unknown texture type: {}", texture))),
    };
    let result = compute_haralick_glcm(&raster, GlcmParams {
        radius,
        distance,
        texture: tex,
        ..GlcmParams::default()
    }).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(raster_to_numpy(py, &result))
}

// ===========================================================================
// Module definition
// ===========================================================================

/// SurtGis: High-performance geospatial analysis from Python.
///
/// Usage:
///     import numpy as np
///     import surtgis
///
///     dem = np.random.rand(100, 100) * 1000  # elevation in meters
///     slp = surtgis.slope(dem, cell_size=30.0)
///     asp = surtgis.aspect_degrees(dem, cell_size=30.0)
///     hs  = surtgis.hillshade_compute(dem, cell_size=30.0)
#[pymodule]
fn surtgis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Terrain
    m.add_function(wrap_pyfunction!(slope, m)?)?;
    m.add_function(wrap_pyfunction!(aspect_degrees, m)?)?;
    m.add_function(wrap_pyfunction!(hillshade_compute, m)?)?;
    m.add_function(wrap_pyfunction!(multidirectional_hillshade, m)?)?;
    m.add_function(wrap_pyfunction!(curvature_compute, m)?)?;
    m.add_function(wrap_pyfunction!(tpi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(tri_compute, m)?)?;
    m.add_function(wrap_pyfunction!(geomorphons_compute, m)?)?;
    m.add_function(wrap_pyfunction!(northness_compute, m)?)?;
    m.add_function(wrap_pyfunction!(eastness_compute, m)?)?;
    m.add_function(wrap_pyfunction!(dev_compute, m)?)?;
    m.add_function(wrap_pyfunction!(shape_index_compute, m)?)?;
    m.add_function(wrap_pyfunction!(curvedness_compute, m)?)?;
    m.add_function(wrap_pyfunction!(sky_view_factor_compute, m)?)?;
    m.add_function(wrap_pyfunction!(uncertainty_slope, m)?)?;
    m.add_function(wrap_pyfunction!(viewshed_compute, m)?)?;
    m.add_function(wrap_pyfunction!(openness_positive, m)?)?;
    m.add_function(wrap_pyfunction!(openness_negative, m)?)?;
    m.add_function(wrap_pyfunction!(mrvbf_compute, m)?)?;
    m.add_function(wrap_pyfunction!(vrm_compute, m)?)?;
    m.add_function(wrap_pyfunction!(advanced_curvature, m)?)?;

    // Hydrology
    m.add_function(wrap_pyfunction!(fill_depressions, m)?)?;
    m.add_function(wrap_pyfunction!(priority_flood_fill, m)?)?;
    m.add_function(wrap_pyfunction!(flow_direction_d8, m)?)?;
    m.add_function(wrap_pyfunction!(flow_direction_dinf, m)?)?;
    m.add_function(wrap_pyfunction!(flow_accumulation_d8, m)?)?;
    m.add_function(wrap_pyfunction!(twi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(hand_compute, m)?)?;
    m.add_function(wrap_pyfunction!(breach_fill, m)?)?;
    m.add_function(wrap_pyfunction!(flow_accumulation_mfd_compute, m)?)?;
    m.add_function(wrap_pyfunction!(stream_network_compute, m)?)?;

    // Imagery
    m.add_function(wrap_pyfunction!(ndvi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ndwi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(savi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_diff, m)?)?;
    m.add_function(wrap_pyfunction!(mndwi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(nbr_compute, m)?)?;
    m.add_function(wrap_pyfunction!(evi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(bsi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ndre_compute, m)?)?;
    m.add_function(wrap_pyfunction!(gndvi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ngrdi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(reci_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ndsi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ndbi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ndmi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(msavi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(evi2_compute, m)?)?;

    // Morphology
    m.add_function(wrap_pyfunction!(morph_erode, m)?)?;
    m.add_function(wrap_pyfunction!(morph_dilate, m)?)?;
    m.add_function(wrap_pyfunction!(morph_opening, m)?)?;
    m.add_function(wrap_pyfunction!(morph_closing, m)?)?;
    m.add_function(wrap_pyfunction!(morph_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(morph_top_hat, m)?)?;
    m.add_function(wrap_pyfunction!(morph_black_hat, m)?)?;

    // Statistics
    m.add_function(wrap_pyfunction!(focal_mean, m)?)?;
    m.add_function(wrap_pyfunction!(focal_std, m)?)?;
    m.add_function(wrap_pyfunction!(focal_range, m)?)?;
    m.add_function(wrap_pyfunction!(focal_min, m)?)?;
    m.add_function(wrap_pyfunction!(focal_max, m)?)?;
    m.add_function(wrap_pyfunction!(focal_sum, m)?)?;
    m.add_function(wrap_pyfunction!(focal_median, m)?)?;

    // New Terrain
    m.add_function(wrap_pyfunction!(solar_radiation_compute, m)?)?;
    m.add_function(wrap_pyfunction!(surface_area_ratio_compute, m)?)?;
    m.add_function(wrap_pyfunction!(valley_depth_compute, m)?)?;
    m.add_function(wrap_pyfunction!(convergence_index_compute, m)?)?;
    m.add_function(wrap_pyfunction!(landform_classification_compute, m)?)?;
    m.add_function(wrap_pyfunction!(wind_exposure_compute, m)?)?;
    m.add_function(wrap_pyfunction!(contour_lines_compute, m)?)?;
    m.add_function(wrap_pyfunction!(cost_distance_compute, m)?)?;
    m.add_function(wrap_pyfunction!(feature_preserving_smoothing_compute, m)?)?;
    m.add_function(wrap_pyfunction!(gaussian_smoothing_compute, m)?)?;
    m.add_function(wrap_pyfunction!(log_transform_compute, m)?)?;
    m.add_function(wrap_pyfunction!(accumulation_zones_compute, m)?)?;
    m.add_function(wrap_pyfunction!(viewshed_xdraw_compute, m)?)?;
    m.add_function(wrap_pyfunction!(horizon_angle_map_compute, m)?)?;

    // New Hydrology
    m.add_function(wrap_pyfunction!(watershed_compute, m)?)?;
    m.add_function(wrap_pyfunction!(flow_accumulation_tfga_compute, m)?)?;
    m.add_function(wrap_pyfunction!(flow_accumulation_adaptive_mfd, m)?)?;
    m.add_function(wrap_pyfunction!(priority_flood_flat_compute, m)?)?;
    m.add_function(wrap_pyfunction!(strahler_order_compute, m)?)?;
    m.add_function(wrap_pyfunction!(flow_path_length_compute, m)?)?;

    // Classification
    m.add_function(wrap_pyfunction!(kmeans_compute, m)?)?;
    m.add_function(wrap_pyfunction!(isodata_compute, m)?)?;

    // Interpolation
    m.add_function(wrap_pyfunction!(idw_interpolation, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_neighbor_interpolation, m)?)?;
    m.add_function(wrap_pyfunction!(natural_neighbor_interpolation, m)?)?;

    // Landscape Ecology
    m.add_function(wrap_pyfunction!(label_patches_compute, m)?)?;
    m.add_function(wrap_pyfunction!(shannon_diversity_compute, m)?)?;
    m.add_function(wrap_pyfunction!(simpson_diversity_compute, m)?)?;
    m.add_function(wrap_pyfunction!(patch_density_compute, m)?)?;
    m.add_function(wrap_pyfunction!(landscape_metrics_compute, m)?)?;

    // Texture
    m.add_function(wrap_pyfunction!(sobel_edge_compute, m)?)?;
    m.add_function(wrap_pyfunction!(laplacian_compute, m)?)?;
    m.add_function(wrap_pyfunction!(haralick_glcm_compute, m)?)?;

    Ok(())
}
