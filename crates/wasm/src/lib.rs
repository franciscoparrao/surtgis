//! WebAssembly bindings for SurtGis geospatial algorithms.
//!
//! Exposes terrain, hydrology, imagery, interpolation, morphology, and
//! statistics functions to JavaScript. Input/output is via in-memory
//! GeoTIFF byte buffers (`&[u8]` / `Vec<u8>`).
//!
//! No competitor offers terrain analysis + hydrology in WASM.

use wasm_bindgen::prelude::*;

use surtgis_algorithms::terrain::{
    aspect, hillshade, curvature, advanced_curvatures,
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
    positive_openness as compute_positive_openness,
    negative_openness as compute_negative_openness,
    mrvbf as compute_mrvbf,
    uncertainty as compute_uncertainty,
    ssa_2d as compute_ssa_2d,
    AspectOutput, HillshadeParams, SlopeParams, SlopeUnits,
    CurvatureParams, CurvatureType, DerivativeMethod, AdvancedCurvatureType,
    TpiParams, TriParams, GeomorphonParams, DevParams,
    MultiHillshadeParams, SvfParams,
    OpennessParams, MrvbfParams,
    UncertaintyParams, Ssa2dParams,
};
use surtgis_algorithms::hydrology::{
    fill_sinks, flow_direction, flow_accumulation,
    flow_accumulation_mfd, flow_direction_dinf, flow_dinf,
    priority_flood, hand as compute_hand,
    FillSinksParams, PriorityFloodParams, HandParams, MfdParams,
};
use surtgis_algorithms::imagery::{
    ndvi as compute_ndvi, ndwi as compute_ndwi, savi as compute_savi,
    mndwi as compute_mndwi, nbr as compute_nbr,
    evi as compute_evi, bsi as compute_bsi,
    ndre as compute_ndre, gndvi as compute_gndvi,
    ndbi as compute_ndbi, ndmi as compute_ndmi,
    msavi as compute_msavi, evi2 as compute_evi2,
    normalized_difference,
    SaviParams, EviParams,
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

/// Compute one of Florinsky's 14 curvatures from a DEM. `ctype` accepts:
/// `mean_h`, `gaussian_k`, `unsphericity_m`, `difference_e`, `kmin`, `kmax`,
/// `kh` (horizontal), `kv` (vertical), `khe` (horizontal excess), `kve`
/// (vertical excess), `ka` (accumulation), `kr` (ring), `rotor`, `laplacian`.
///
/// SurtGIS is the only library that exposes the complete 14-curvature
/// system from a single binary, and the only one that does it in the
/// browser via WebAssembly. See paper §2.3.2 for definitions and
/// references.
#[wasm_bindgen]
pub fn advanced_curvature(tiff_bytes: &[u8], ctype: &str) -> Result<Vec<u8>, JsValue> {
    let ct = match ctype.to_lowercase().as_str() {
        "mean_h" | "mean" | "h" => AdvancedCurvatureType::MeanH,
        "gaussian_k" | "gaussian" | "k" => AdvancedCurvatureType::GaussianK,
        "unsphericity_m" | "unsphericity" | "m" => AdvancedCurvatureType::UnsphericitytM,
        "difference_e" | "difference" | "e" => AdvancedCurvatureType::DifferenceE,
        "kmin" | "minimal" => AdvancedCurvatureType::MinimalKmin,
        "kmax" | "maximal" => AdvancedCurvatureType::MaximalKmax,
        "kh" | "horizontal" => AdvancedCurvatureType::HorizontalKh,
        "kv" | "vertical" => AdvancedCurvatureType::VerticalKv,
        "khe" | "horizontal_excess" => AdvancedCurvatureType::HorizontalExcessKhe,
        "kve" | "vertical_excess" => AdvancedCurvatureType::VerticalExcessKve,
        "ka" | "accumulation" => AdvancedCurvatureType::AccumulationKa,
        "kr" | "ring" => AdvancedCurvatureType::RingKr,
        "rotor" => AdvancedCurvatureType::Rotor,
        "laplacian" => AdvancedCurvatureType::Laplacian,
        other => {
            return Err(JsValue::from_str(&format!(
                "unknown advanced curvature type: '{}'. Valid: mean_h, gaussian_k, unsphericity_m, difference_e, kmin, kmax, kh, kv, khe, kve, ka, kr, rotor, laplacian",
                other
            )));
        }
    };
    dem_op!(tiff_bytes, |dem: &_| advanced_curvatures(dem, ct))
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

/// Positive openness — visibility of the sky above each cell across `n_dirs`
/// directions out to `radius` cells. High values = exposed ridges/peaks.
#[wasm_bindgen]
pub fn openness_positive(tiff_bytes: &[u8], radius: usize, n_dirs: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_positive_openness(
        dem, OpennessParams { radius, directions: n_dirs }
    ))
}

/// Negative openness — visibility of the ground below each cell. High values
/// = concave hollows / valleys.
#[wasm_bindgen]
pub fn openness_negative(tiff_bytes: &[u8], radius: usize, n_dirs: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| compute_negative_openness(
        dem, OpennessParams { radius, directions: n_dirs }
    ))
}

/// MRVBF — Multi-Resolution Valley Bottom Flatness (Gallant & Dowling 2003).
/// Returns only the MRVBF raster (companion MRRTF dropped); if you need both,
/// expose via the native Rust API.
#[wasm_bindgen]
pub fn mrvbf(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let (mrvbf_raster, _mrrtf) = compute_mrvbf(&dem, MrvbfParams::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&mrvbf_raster, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
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
/// Compute MFD (Multiple Flow Direction) accumulation directly from a DEM.
/// Distributes flow proportionally to all downslope neighbors based on slope,
/// better representing sheet flow on gentle terrain than D8.
#[wasm_bindgen]
pub fn flow_accumulation_mfd_compute(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| flow_accumulation_mfd(
        dem, MfdParams::default()
    ))
}

/// Compute D-infinity flow direction angles (radians, 0 = East, CCW).
/// NaN for nodata/pit cells. Reference implementation matches TauDEM.
#[wasm_bindgen]
pub fn flow_direction_dinf_compute(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| flow_direction_dinf(dem))
}

/// Compute D-infinity flow accumulation (contributing area in cell counts).
/// Internally computes the directions and the accumulation in one pass; if
/// you need the angles separately, use `flow_direction_dinf_compute` first.
#[wasm_bindgen]
pub fn flow_accumulation_dinf_compute(tiff_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    let dem = read_geotiff_from_buffer::<f64>(tiff_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = flow_dinf(&dem)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result.accumulation, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

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

/// Helper: two-band spectral index. Loads the two GeoTIFFs, applies `f`, writes result.
macro_rules! two_band_index {
    ($a:expr, $b:expr, $f:expr) => {{
        let a = read_geotiff_from_buffer::<f64>($a, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let b = read_geotiff_from_buffer::<f64>($b, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let result = ($f)(&a, &b)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        write_geotiff_to_buffer(&result, None)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }};
}

/// MNDWI (Modified NDWI) from Green and SWIR bands.
#[wasm_bindgen]
pub fn mndwi(green_bytes: &[u8], swir_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(green_bytes, swir_bytes, compute_mndwi)
}

/// NBR (Normalized Burn Ratio) from NIR and SWIR bands.
#[wasm_bindgen]
pub fn nbr(nir_bytes: &[u8], swir_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(nir_bytes, swir_bytes, compute_nbr)
}

/// NDRE (Normalized Difference Red Edge) from NIR and Red Edge bands.
#[wasm_bindgen]
pub fn ndre(nir_bytes: &[u8], red_edge_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(nir_bytes, red_edge_bytes, compute_ndre)
}

/// GNDVI (Green NDVI) from NIR and Green bands.
#[wasm_bindgen]
pub fn gndvi(nir_bytes: &[u8], green_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(nir_bytes, green_bytes, compute_gndvi)
}

/// NDBI (Normalized Difference Built-up Index) from SWIR and NIR bands.
#[wasm_bindgen]
pub fn ndbi(swir_bytes: &[u8], nir_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(swir_bytes, nir_bytes, compute_ndbi)
}

/// NDMI (Normalized Difference Moisture Index) from NIR and SWIR bands.
#[wasm_bindgen]
pub fn ndmi(nir_bytes: &[u8], swir_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(nir_bytes, swir_bytes, compute_ndmi)
}

/// MSAVI (Modified SAVI) from NIR and Red bands. Auto-adjusts L factor.
#[wasm_bindgen]
pub fn msavi(nir_bytes: &[u8], red_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(nir_bytes, red_bytes, compute_msavi)
}

/// EVI2 (Enhanced Vegetation Index, 2-band variant) from NIR and Red bands.
#[wasm_bindgen]
pub fn evi2(nir_bytes: &[u8], red_bytes: &[u8]) -> Result<Vec<u8>, JsValue> {
    two_band_index!(nir_bytes, red_bytes, compute_evi2)
}

/// EVI (Enhanced Vegetation Index) from NIR, Red, and Blue bands.
/// Uses default EVI coefficients (G=2.5, L=1, C1=6, C2=7.5).
#[wasm_bindgen]
pub fn evi(
    nir_bytes: &[u8], red_bytes: &[u8], blue_bytes: &[u8],
) -> Result<Vec<u8>, JsValue> {
    let nir = read_geotiff_from_buffer::<f64>(nir_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let red = read_geotiff_from_buffer::<f64>(red_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let blue = read_geotiff_from_buffer::<f64>(blue_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_evi(&nir, &red, &blue, EviParams::default())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    write_geotiff_to_buffer(&result, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// BSI (Bare Soil Index) from SWIR, Red, NIR, and Blue bands.
#[wasm_bindgen]
pub fn bsi(
    swir_bytes: &[u8], red_bytes: &[u8], nir_bytes: &[u8], blue_bytes: &[u8],
) -> Result<Vec<u8>, JsValue> {
    let swir = read_geotiff_from_buffer::<f64>(swir_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let red = read_geotiff_from_buffer::<f64>(red_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let nir = read_geotiff_from_buffer::<f64>(nir_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let blue = read_geotiff_from_buffer::<f64>(blue_bytes, None)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let result = compute_bsi(&swir, &red, &nir, &blue)
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

/// Focal minimum.
#[wasm_bindgen]
pub fn focal_min(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Min, radius, circular: false }
    ))
}

/// Focal maximum.
#[wasm_bindgen]
pub fn focal_max(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Max, radius, circular: false }
    ))
}

/// Focal sum.
#[wasm_bindgen]
pub fn focal_sum(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Sum, radius, circular: false }
    ))
}

/// Focal median.
#[wasm_bindgen]
pub fn focal_median(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Median, radius, circular: false }
    ))
}

/// Focal majority (mode). Useful for categorical rasters.
#[wasm_bindgen]
pub fn focal_majority(tiff_bytes: &[u8], radius: usize) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Majority, radius, circular: false }
    ))
}

/// Focal percentile. `percentile` in [0, 100]; e.g. 50 = median, 25 = Q1.
#[wasm_bindgen]
pub fn focal_percentile(
    tiff_bytes: &[u8], radius: usize, percentile: f64,
) -> Result<Vec<u8>, JsValue> {
    dem_op!(tiff_bytes, |dem: &_| focal_statistics(
        dem, FocalParams { statistic: FocalStatistic::Percentile(percentile), radius, circular: false }
    ))
}
