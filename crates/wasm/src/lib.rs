//! WebAssembly bindings for SurtGis geospatial algorithms.
//!
//! Exposes terrain, hydrology, imagery, interpolation, morphology, and
//! statistics functions to JavaScript. Input/output is via in-memory
//! GeoTIFF byte buffers (`&[u8]` / `Vec<u8>`).
//!
//! No competitor offers terrain analysis + hydrology in WASM.

use wasm_bindgen::prelude::*;

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
    ssa_2d as compute_ssa_2d,
    AspectOutput, HillshadeParams, SlopeParams, SlopeUnits,
    CurvatureParams, CurvatureType, DerivativeMethod,
    TpiParams, TriParams, GeomorphonParams, DevParams,
    MultiHillshadeParams, SvfParams,
    UncertaintyParams, Ssa2dParams,
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
use surtgis_core::io::{read_geotiff_from_buffer, write_geotiff_to_buffer};

/// Helper: read DEM, apply function, write result
macro_rules! dem_op {
    ($tiff:expr, $body:expr) => {{
        let dem = read_geotiff_from_buffer::<f64>($tiff, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = ($body)(&dem)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        write_geotiff_to_buffer(&result, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }};
}

// ===========================================================================
// Terrain — Basic
// ===========================================================================

/// Compute slope from a DEM. `units`: "degrees" (default) or "percent".
#[wasm_bindgen]
pub fn slope(tiff_bytes: &[u8], units: &str) -> Result<Vec<u8>, JsValue> {
    let slope_units = match units.to_lowercase().as_str() {
        "percent" | "pct" => SlopeUnits::Percent,
        _ => SlopeUnits::Degrees,
    };
    dem_op!(tiff_bytes, |dem: &_| compute_slope(
        dem,
        SlopeParams { units: slope_units, z_factor: 1.0 }
    ))
}

/// Compute aspect in degrees (0-360, 0=North, clockwise).
#[wasm_bindgen]
pub fn aspect_degrees(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| aspect(dem, AspectOutput::Degrees))
}

/// Compute hillshade. `azimuth`: sun azimuth (default 315). `altitude`: sun altitude (default 45).
#[wasm_bindgen]
pub fn hillshade_compute(
    tiff_bytes: &[u8],
    azimuth: f64,
    altitude: f64,
) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| hillshade(
        dem,
        HillshadeParams { azimuth, altitude, z_factor: 1.0, normalized: false }
    ))
}

/// Compute multidirectional hillshade (6 azimuths combined).
#[wasm_bindgen]
pub fn multidirectional_hillshade(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_multi_hillshade(
        dem, MultiHillshadeParams::default()
    ))
}

/// Compute curvature. `ctype`: "general", "profile", "plan".
#[wasm_bindgen]
pub fn curvature_compute(tiff_bytes: &[u8], ctype: &str) -> Result<Vec<u8>, JsValue> {
    let ct = match ctype.to_lowercase().as_str() {
        "profile" => CurvatureType::Profile,
        "plan" | "tangential" => CurvatureType::Plan,
        _ => CurvatureType::General,
    };
    dem_op!(tiff_bytes, |dem: &_| curvature(
        dem,
        CurvatureParams { curvature_type: ct, method: DerivativeMethod::EvansYoung, ..Default::default() }
    ))
}

/// Compute TPI (Topographic Position Index). `radius`: window radius in cells.
#[wasm_bindgen]
pub fn tpi_compute(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_tpi(
        dem, TpiParams { radius }
    ))
}

/// Compute TRI (Terrain Ruggedness Index).
#[wasm_bindgen]
pub fn tri_compute(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_tri(dem, TriParams::default()))
}

/// Compute TWI (Topographic Wetness Index) from a DEM.
/// Internally computes fill → flow_direction → flow_accumulation → slope → TWI.
#[wasm_bindgen]
pub fn twi_compute(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let filled = fill_sinks(&dem, FillSinksParams::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let fdir = flow_direction(&filled)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let facc = flow_accumulation(&fdir)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let slp = compute_slope(&dem, SlopeParams { units: SlopeUnits::Degrees, z_factor: 1.0 })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_twi(&facc, &slp)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute geomorphons landform classification.
/// `flatness`: flatness threshold degrees. `radius`: search radius in cells.
#[wasm_bindgen]
pub fn geomorphons_compute(tiff_bytes: &[u8], flatness: f64, radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_geomorphons(
        dem, GeomorphonParams { flatness_threshold: flatness, radius }
    ))
}

/// Compute northness (cos of aspect).
#[wasm_bindgen]
pub fn northness_compute(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| northness(dem))
}

/// Compute eastness (sin of aspect).
#[wasm_bindgen]
pub fn eastness_compute(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| eastness(dem))
}

/// Compute DEV (Deviation from Mean Elevation). `radius`: window radius.
#[wasm_bindgen]
pub fn dev_compute(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_dev(
        dem, DevParams { radius }
    ))
}

/// Compute shape index.
#[wasm_bindgen]
pub fn shape_index(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_shape_index(dem))
}

/// Compute curvedness.
#[wasm_bindgen]
pub fn curvedness(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_curvedness(dem))
}

/// Compute sky view factor. `n_dirs`: number of directions. `max_radius`: search radius in cells.
#[wasm_bindgen]
pub fn sky_view_factor(tiff_bytes: &[u8], n_dirs: usize, max_radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_svf(
        dem, SvfParams { directions: n_dirs, radius: max_radius }
    ))
}

/// Compute slope uncertainty (RMSE). `dem_rmse`: DEM vertical RMSE in meters.
#[wasm_bindgen]
pub fn uncertainty_slope(tiff_bytes: &[u8], dem_rmse: f64) -> Result<Vec<u8>, JsValue> {
    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_uncertainty(&dem, UncertaintyParams { dem_rmse })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result.slope_rmse, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Denoise DEM using 2D-SSA. `window`: window size, `components`: number of signal components.
#[wasm_bindgen]
pub fn ssa_2d_denoise(tiff_bytes: &[u8], window: usize, components: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_ssa_2d(
        dem, Ssa2dParams { window_rows: window, window_cols: window, n_components: components }
    ))
}

// ===========================================================================
// Hydrology
// ===========================================================================

/// Fill sinks (depressions) in a DEM.
#[wasm_bindgen]
pub fn fill_depressions(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| fill_sinks(dem, FillSinksParams::default()))
}

/// Priority-flood depression filling (Barnes 2014).
#[wasm_bindgen]
pub fn priority_flood_fill(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| priority_flood(dem, PriorityFloodParams::default()))
}

/// Compute D8 flow direction from a filled DEM. Returns encoded direction raster.
#[wasm_bindgen]
pub fn flow_direction_d8(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = flow_direction(&dem)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute flow accumulation from a D8 flow direction raster.
#[wasm_bindgen]
pub fn flow_accumulation_d8(fdir_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let fdir = read_geotiff_from_buffer::<u8>(fdir_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = flow_accumulation(&fdir)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute HAND (Height Above Nearest Drainage) from a DEM.
/// Internally computes fill → flow_direction → flow_accumulation → HAND.
/// `stream_threshold`: accumulation threshold to define streams (default 1000).
#[wasm_bindgen]
pub fn hand_compute(tiff_bytes: &[u8], stream_threshold: f64) -> Result<Vec<u8>, JsValue> {
    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let filled = fill_sinks(&dem, FillSinksParams::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let fdir = flow_direction(&filled)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let facc = flow_accumulation(&fdir)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_hand(&dem, &fdir, &facc, HandParams { stream_threshold })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ===========================================================================
// Imagery / Spectral Indices
// ===========================================================================

/// Compute NDVI from NIR and Red bands.
#[wasm_bindgen]
pub fn ndvi(nir_bytes: &[u8], red_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let nir = read_geotiff_from_buffer::<f64>(nir_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let red = read_geotiff_from_buffer::<f64>(red_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_ndvi(&nir, &red)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute NDWI (Normalized Difference Water Index) from Green and NIR bands.
#[wasm_bindgen]
pub fn ndwi(green_bytes: &[u8], nir_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let green = read_geotiff_from_buffer::<f64>(green_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nir = read_geotiff_from_buffer::<f64>(nir_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_ndwi(&green, &nir)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute SAVI (Soil-Adjusted Vegetation Index). `l_factor`: soil adjustment (default 0.5).
#[wasm_bindgen]
pub fn savi(nir_bytes: &[u8], red_bytes: &[u8], l_factor: f64) -> Result<Vec<u8>, JsValue> {
    let nir = read_geotiff_from_buffer::<f64>(nir_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let red = read_geotiff_from_buffer::<f64>(red_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_savi(&nir, &red, SaviParams { l_factor })
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Compute generic Normalized Difference Index: (A - B) / (A + B).
#[wasm_bindgen]
pub fn normalized_diff(a_bytes: &[u8], b_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let a = read_geotiff_from_buffer::<f64>(a_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let b = read_geotiff_from_buffer::<f64>(b_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = normalized_difference(&a, &b)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

// ===========================================================================
// Morphology
// ===========================================================================

/// Morphological erosion with square kernel of given `radius`.
#[wasm_bindgen]
pub fn morph_erode(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    let elem = StructuringElement::Square(radius);
    dem_op!(tiff_bytes, |dem: &_| compute_erode(dem, &elem))
}

/// Morphological dilation with square kernel of given `radius`.
#[wasm_bindgen]
pub fn morph_dilate(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    let elem = StructuringElement::Square(radius);
    dem_op!(tiff_bytes, |dem: &_| compute_dilate(dem, &elem))
}

/// Morphological opening (erosion followed by dilation).
#[wasm_bindgen]
pub fn morph_opening(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    let elem = StructuringElement::Square(radius);
    dem_op!(tiff_bytes, |dem: &_| compute_opening(dem, &elem))
}

/// Morphological closing (dilation followed by erosion).
#[wasm_bindgen]
pub fn morph_closing(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    let elem = StructuringElement::Square(radius);
    dem_op!(tiff_bytes, |dem: &_| compute_closing(dem, &elem))
}

// ===========================================================================
// Statistics
// ===========================================================================

/// Focal mean statistic. `radius`: window radius in cells.
#[wasm_bindgen]
pub fn focal_mean(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Mean, radius, circular: false }
    ))
}

/// Focal standard deviation. `radius`: window radius in cells.
#[wasm_bindgen]
pub fn focal_std(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::StdDev, radius, circular: false }
    ))
}

/// Focal range (max - min). `radius`: window radius in cells.
#[wasm_bindgen]
pub fn focal_range(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Range, radius, circular: false }
    ))
}
