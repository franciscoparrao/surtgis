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
    AspectOutput, HillshadeParams, SlopeParams, SlopeUnits,
    CurvatureParams, CurvatureType, DerivativeMethod,
    TpiParams, TriParams, GeomorphonParams, DevParams,
    MultiHillshadeParams, SvfParams,
    UncertaintyParams,
};
use surtgis_algorithms::hydrology::{
    fill_sinks, flow_direction, flow_accumulation,
    priority_flood, hand as compute_hand,
    FillSinksParams, PriorityFloodParams, HandParams,
};
use surtgis_algorithms::imagery::{
    ndvi as compute_ndvi, ndwi as compute_ndwi, savi as compute_savi,
    normalized_difference,
    SaviParams,
};
use surtgis_algorithms::morphology::{
    erode as compute_erode, dilate as compute_dilate,
    opening as compute_opening, closing as compute_closing,
    StructuringElement,
};
use surtgis_algorithms::statistics::{
    focal_statistics, FocalStatistic, FocalParams,
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
    let result = curvature(&raster, CurvatureParams { curvature_type: ct, method: DerivativeMethod::EvansYoung, ..Default::default() })
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
    let result = compute_tpi(&raster, TpiParams { radius, ..Default::default() })
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
    let result = compute_geomorphons(&raster, GeomorphonParams { flatness_threshold: flatness, radius, ..Default::default() })
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
    let result = compute_dev(&raster, DevParams { radius, ..Default::default() })
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
    let result = compute_svf(&raster, SvfParams { directions, radius, ..Default::default() })
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

    // Hydrology
    m.add_function(wrap_pyfunction!(fill_depressions, m)?)?;
    m.add_function(wrap_pyfunction!(priority_flood_fill, m)?)?;
    m.add_function(wrap_pyfunction!(flow_direction_d8, m)?)?;
    m.add_function(wrap_pyfunction!(flow_accumulation_d8, m)?)?;
    m.add_function(wrap_pyfunction!(twi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(hand_compute, m)?)?;

    // Imagery
    m.add_function(wrap_pyfunction!(ndvi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(ndwi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(savi_compute, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_diff, m)?)?;

    // Morphology
    m.add_function(wrap_pyfunction!(morph_erode, m)?)?;
    m.add_function(wrap_pyfunction!(morph_dilate, m)?)?;
    m.add_function(wrap_pyfunction!(morph_opening, m)?)?;
    m.add_function(wrap_pyfunction!(morph_closing, m)?)?;

    // Statistics
    m.add_function(wrap_pyfunction!(focal_mean, m)?)?;
    m.add_function(wrap_pyfunction!(focal_std, m)?)?;
    m.add_function(wrap_pyfunction!(focal_range, m)?)?;

    Ok(())
}
