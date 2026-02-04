//! Algorithm execution in background threads.
//!
//! Each algorithm runs in a separate `std::thread` and sends results back
//! via `crossbeam_channel`.

use std::collections::HashMap;
use std::time::Instant;

use crossbeam_channel::Sender;

use surtgis_algorithms::hydrology::{
    breach_depressions, fill_sinks, flow_accumulation, flow_accumulation_mfd,
    flow_accumulation_mfd_adaptive, flow_accumulation_tfga, flow_dinf, flow_direction,
    flow_direction_dinf, hand, nested_depressions, priority_flood_flat, stream_network, watershed,
    AdaptiveMfdParams, BreachParams, FillSinksParams, HandParams, MfdParams,
    NestedDepressionParams, StreamNetworkParams, TfgaParams, WatershedParams,
};
use surtgis_algorithms::imagery::{
    band_math_binary, bsi, evi, evi2, gndvi, mndwi, msavi, nbr, ndbi, ndmi, ndre, ndsi,
    ndvi, ndwi, ngrdi, normalized_difference, reci, reclassify, savi, BandMathOp, EviParams,
    ReclassEntry, ReclassifyParams, SaviParams,
};
use surtgis_algorithms::landscape::{
    shannon_diversity, simpson_diversity, patch_density, DiversityParams,
};
use surtgis_algorithms::interpolation::{
    idw, natural_neighbor, nearest_neighbor, ordinary_kriging, tin_interpolation,
    tps_interpolation, IdwParams, NaturalNeighborParams, NearestNeighborParams,
    OrdinaryKrigingParams, SamplePoint, TinParams, TpsParams,
};
use surtgis_algorithms::morphology::{
    black_hat, closing, dilate, erode, gradient, opening, top_hat, StructuringElement,
};
use surtgis_algorithms::statistics::{
    focal_statistics, global_morans_i, local_getis_ord, FocalParams, FocalStatistic,
};
use surtgis_algorithms::terrain::{
    advanced_curvatures, aspect, contour_lines, convergence_index, cost_distance, curvature,
    curvedness, dev, eastness, geomorphons, hillshade, landform_classification,
    lineament_detection, multidirectional_hillshade, mrvbf, negative_openness, northness,
    positive_openness, shape_index, sky_view_factor, slope, solar_radiation, spi, sti, tpi,
    tri, viewshed, vrm, wind_exposure, AdvancedCurvatureType, AspectOutput, ContourParams,
    ConvergenceParams, CostDistanceParams, CurvatureParams, CurvatureType, DevParams,
    GeomorphonParams, HillshadeParams, LandformParams, LineamentParams, MultiHillshadeParams,
    MrvbfParams, OpennessParams, SlopeParams, SlopeUnits, SolarParams, StiParams, SvfParams,
    TpiParams, TriParams, ViewshedParams, VrmParams, WindExposureParams,
};
use surtgis_core::raster::{GeoTransform, Raster};

use crate::registry::ParamValue;
use crate::state::{AppMessage, LogEntry};

/// Dispatch an algorithm by ID, running it in a background thread.
pub fn dispatch_algorithm(
    algo_id: &str,
    input: Raster<f64>,
    params: &HashMap<String, ParamValue>,
    extra_inputs: HashMap<String, Raster<f64>>,
    tx: Sender<AppMessage>,
) {
    let algo_id = algo_id.to_string();
    let params = params.clone();

    std::thread::spawn(move || {
        let _ = tx.send(AppMessage::Log(LogEntry::info(format!(
            "Running {}...",
            algo_id
        ))));

        let start = Instant::now();

        match algo_id.as_str() {
            // ═══════════════════════════════════════════════════
            // TERRAIN
            // ═══════════════════════════════════════════════════
            "slope" => {
                let units = match get_choice(&params, "units") {
                    1 => SlopeUnits::Percent,
                    2 => SlopeUnits::Radians,
                    _ => SlopeUnits::Degrees,
                };
                let z_factor = get_f64(&params, "z_factor", 1.0);
                dispatch_f64(&tx, "Slope", start, slope(&input, SlopeParams { units, z_factor }));
            }

            "aspect" => {
                let fmt = match get_choice(&params, "format") {
                    1 => AspectOutput::Radians,
                    _ => AspectOutput::Degrees,
                };
                dispatch_f64(&tx, "Aspect", start, aspect(&input, fmt));
            }

            "hillshade" => {
                dispatch_f64(&tx, "Hillshade", start, hillshade(&input, HillshadeParams {
                    azimuth: get_f64(&params, "azimuth", 315.0),
                    altitude: get_f64(&params, "altitude", 45.0),
                    z_factor: get_f64(&params, "z_factor", 1.0),
                    normalized: false,
                }));
            }

            "multidirectional_hillshade" => {
                dispatch_f64(&tx, "Multidirectional Hillshade", start,
                    multidirectional_hillshade(&input, MultiHillshadeParams::default()));
            }

            "curvature" => {
                let ct = match get_choice(&params, "type") {
                    1 => CurvatureType::Profile,
                    2 => CurvatureType::Plan,
                    _ => CurvatureType::General,
                };
                dispatch_f64(&tx, "Curvature", start, curvature(&input, CurvatureParams {
                    curvature_type: ct,
                    z_factor: get_f64(&params, "z_factor", 1.0),
                    ..Default::default()
                }));
            }

            "tpi" => {
                dispatch_f64(&tx, "TPI", start, tpi(&input, TpiParams {
                    radius: get_usize(&params, "radius", 3),
                }));
            }

            "tri" => {
                dispatch_f64(&tx, "TRI", start, tri(&input, TriParams {
                    radius: get_usize(&params, "radius", 1),
                }));
            }

            "twi" => {
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow + slope for TWI...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    let fdir = flow_direction(&filled)?;
                    let facc = flow_accumulation(&fdir)?;
                    let slope_rad = slope(&filled, SlopeParams {
                        units: SlopeUnits::Radians, z_factor: 1.0,
                    })?;
                    Ok(surtgis_algorithms::terrain::twi(&facc, &slope_rad)?)
                })();
                dispatch_f64(&tx, "TWI", start, res);
            }

            "geomorphons" => {
                let result = geomorphons(&input, GeomorphonParams {
                    radius: get_usize(&params, "radius", 10),
                    flatness_threshold: get_f64(&params, "flatness", 1.0),
                });
                dispatch_u8(&tx, "Geomorphons", start, result);
            }

            "dev" => {
                dispatch_f64(&tx, "DEV", start, dev(&input, DevParams {
                    radius: get_usize(&params, "radius", 10),
                }));
            }

            "landform" => {
                dispatch_f64(&tx, "Landform", start, landform_classification(&input, LandformParams {
                    small_radius: get_usize(&params, "small_radius", 3),
                    large_radius: get_usize(&params, "large_radius", 10),
                    tpi_threshold: get_f64(&params, "threshold", 1.0),
                    slope_threshold: get_f64(&params, "slope_threshold", 6.0),
                }));
            }

            "sky_view_factor" => {
                dispatch_f64(&tx, "Sky View Factor", start, sky_view_factor(&input, SvfParams {
                    radius: get_usize(&params, "radius", 10),
                    directions: get_usize(&params, "directions", 16),
                }));
            }

            "positive_openness" => {
                dispatch_f64(&tx, "Positive Openness", start, positive_openness(&input, OpennessParams {
                    radius: get_usize(&params, "radius", 10),
                    directions: get_usize(&params, "directions", 8),
                }));
            }

            "negative_openness" => {
                dispatch_f64(&tx, "Negative Openness", start, negative_openness(&input, OpennessParams {
                    radius: get_usize(&params, "radius", 10),
                    directions: get_usize(&params, "directions", 8),
                }));
            }

            "convergence_index" => {
                dispatch_f64(&tx, "Convergence Index", start, convergence_index(&input, ConvergenceParams {
                    radius: get_usize(&params, "radius", 1),
                }));
            }

            "vrm" => {
                dispatch_f64(&tx, "VRM", start, vrm(&input, VrmParams {
                    radius: get_usize(&params, "radius", 1),
                }));
            }

            "shape_index" => {
                dispatch_f64(&tx, "Shape Index", start, shape_index(&input));
            }

            "curvedness" => {
                dispatch_f64(&tx, "Curvedness", start, curvedness(&input));
            }

            "northness" => {
                dispatch_f64(&tx, "Northness", start, northness(&input));
            }

            "eastness" => {
                dispatch_f64(&tx, "Eastness", start, eastness(&input));
            }

            // ═══════════════════════════════════════════════════
            // HYDROLOGY
            // ═══════════════════════════════════════════════════
            "fill_sinks" => {
                dispatch_f64(&tx, "Fill Sinks", start,
                    fill_sinks(&input, FillSinksParams { min_slope: 0.01 }));
            }

            "priority_flood" => {
                dispatch_f64(&tx, "Priority Flood", start, priority_flood_flat(&input));
            }

            "flow_direction" => {
                dispatch_u8(&tx, "Flow Direction", start, flow_direction(&input));
            }

            "flow_accumulation" => {
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow direction first...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    let fdir = flow_direction(&filled)?;
                    Ok(flow_accumulation(&fdir)?)
                })();
                dispatch_f64(&tx, "Flow Accumulation", start, res);
            }

            "hand" => {
                let threshold = get_f64(&params, "stream_threshold", 1000.0);
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow + accumulation for HAND...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    let fdir = flow_direction(&filled)?;
                    let facc = flow_accumulation(&fdir)?;
                    Ok(hand(&filled, &fdir, &facc, HandParams { stream_threshold: threshold })?)
                })();
                dispatch_f64(&tx, "HAND", start, res);
            }

            // ═══════════════════════════════════════════════════
            // IMAGERY
            // ═══════════════════════════════════════════════════
            "ndvi" => {
                require_extra(&tx, "NDVI", &extra_inputs, &["nir"], |inputs| {
                    ndvi(inputs["nir"], &input)
                }, start);
            }

            "ndwi" => {
                require_extra(&tx, "NDWI", &extra_inputs, &["green"], |inputs| {
                    ndwi(inputs["green"], &input)
                }, start);
            }

            "mndwi" => {
                require_extra(&tx, "MNDWI", &extra_inputs, &["green", "swir"], |inputs| {
                    mndwi(inputs["green"], inputs["swir"])
                }, start);
            }

            "nbr" => {
                require_extra(&tx, "NBR", &extra_inputs, &["nir", "swir"], |inputs| {
                    nbr(inputs["nir"], inputs["swir"])
                }, start);
            }

            "savi" => {
                let l_factor = get_f64(&params, "l_factor", 0.5);
                require_extra(&tx, "SAVI", &extra_inputs, &["nir", "red"], |inputs| {
                    savi(inputs["nir"], inputs["red"], SaviParams { l_factor })
                }, start);
            }

            "evi" => {
                require_extra(&tx, "EVI", &extra_inputs, &["nir", "red", "blue"], |inputs| {
                    evi(inputs["nir"], inputs["red"], inputs["blue"], EviParams::default())
                }, start);
            }

            "bsi" => {
                require_extra(&tx, "BSI", &extra_inputs, &["swir", "red", "nir", "blue"], |inputs| {
                    bsi(inputs["swir"], inputs["red"], inputs["nir"], inputs["blue"])
                }, start);
            }

            "band_math" => {
                let op = match get_choice(&params, "op") {
                    0 => BandMathOp::Add,
                    1 => BandMathOp::Subtract,
                    2 => BandMathOp::Multiply,
                    3 => BandMathOp::Divide,
                    4 => BandMathOp::Min,
                    5 => BandMathOp::Max,
                    _ => BandMathOp::Add,
                };
                require_extra(&tx, "Band Math", &extra_inputs, &["a", "b"], |inputs| {
                    band_math_binary(inputs["a"], inputs["b"], op)
                }, start);
            }

            // ═══════════════════════════════════════════════════
            // MORPHOLOGY
            // ═══════════════════════════════════════════════════
            "erode" => {
                let se = make_se(&params);
                dispatch_f64(&tx, "Erosion", start, erode(&input, &se));
            }

            "dilate" => {
                let se = make_se(&params);
                dispatch_f64(&tx, "Dilation", start, dilate(&input, &se));
            }

            "morph_opening" => {
                let se = make_se(&params);
                dispatch_f64(&tx, "Opening", start, opening(&input, &se));
            }

            "morph_closing" => {
                let se = make_se(&params);
                dispatch_f64(&tx, "Closing", start, closing(&input, &se));
            }

            "morph_gradient" => {
                let se = make_se(&params);
                dispatch_f64(&tx, "Gradient", start, gradient(&input, &se));
            }

            "top_hat" => {
                let se = make_se(&params);
                dispatch_f64(&tx, "Top Hat", start, top_hat(&input, &se));
            }

            "black_hat" => {
                let se = make_se(&params);
                dispatch_f64(&tx, "Black Hat", start, black_hat(&input, &se));
            }

            // ═══════════════════════════════════════════════════
            // STATISTICS
            // ═══════════════════════════════════════════════════
            "focal_mean" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::Mean, "Focal Mean");
            }

            "focal_std" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::StdDev, "Focal Std Dev");
            }

            "focal_range" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::Range, "Focal Range");
            }

            "focal_median" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::Median, "Focal Median");
            }

            "focal_min" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::Min, "Focal Min");
            }

            "focal_max" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::Max, "Focal Max");
            }

            "focal_sum" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::Sum, "Focal Sum");
            }

            "focal_majority" => {
                dispatch_focal(&tx, start, &input, &params, FocalStatistic::Majority, "Focal Majority");
            }

            "zonal_statistics" => {
                require_extra(&tx, "Zonal Statistics", &extra_inputs, &["zones"], |inputs| {
                    let zones_f64 = inputs["zones"];
                    // Convert f64 zones raster to i32
                    let mut zones_i32 = Raster::<i32>::new(zones_f64.rows(), zones_f64.cols());
                    zones_i32.set_transform(zones_f64.transform().clone());
                    for r in 0..zones_f64.rows() {
                        for c in 0..zones_f64.cols() {
                            if let Ok(v) = zones_f64.get(r, c) {
                                let _ = zones_i32.set(r, c, v as i32);
                            }
                        }
                    }
                    use surtgis_algorithms::statistics::zonal_statistics;
                    let results = zonal_statistics(&input, &zones_i32)?;
                    // Write mean values back to a raster (zone_id -> mean)
                    let mut out = Raster::<f64>::new(input.rows(), input.cols());
                    out.set_transform(input.transform().clone());
                    out.set_crs(input.crs().cloned());
                    for r in 0..input.rows() {
                        for c in 0..input.cols() {
                            if let Ok(zone_val) = zones_i32.get(r, c) {
                                if let Some(stats) = results.get(&zone_val) {
                                    let _ = out.set(r, c, stats.mean);
                                }
                            }
                        }
                    }
                    Ok(out)
                }, start);
            }

            "morans_i" => {
                let result = global_morans_i(&input);
                match result {
                    Ok(mi) => {
                        let elapsed = start.elapsed();
                        let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                            "Moran's I = {:.6}, z = {:.3}, p = {:.6} ({:.2}s)",
                            mi.i, mi.z_score, mi.p_value, elapsed.as_secs_f64()
                        ))));
                        // Return the input raster unchanged — result is in the log
                        let _ = tx.send(AppMessage::AlgoComplete {
                            name: "Moran's I".to_string(),
                            result: input,
                            elapsed,
                        });
                    }
                    Err(e) => send_error(&tx, "Moran's I", e),
                }
            }

            "getis_ord" => {
                let radius = get_usize(&params, "radius", 3);
                match local_getis_ord(&input, radius) {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        let _ = tx.send(AppMessage::AlgoComplete {
                            name: "Getis-Ord Gi* (z-scores)".to_string(),
                            result: result.z_scores,
                            elapsed,
                        });
                        let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                            "Getis-Ord Gi* completed in {:.2}s", elapsed.as_secs_f64()
                        ))));
                    }
                    Err(e) => send_error(&tx, "Getis-Ord Gi*", e),
                }
            }

            // ═══════════════════════════════════════════════════
            // HYDROLOGY — new
            // ═══════════════════════════════════════════════════
            "breach_depressions" => {
                let max_depth = get_f64(&params, "max_depth", f64::INFINITY);
                let max_length = get_usize(&params, "max_length", 0);
                dispatch_f64(&tx, "Breach Depressions", start, breach_depressions(&input, BreachParams {
                    max_depth,
                    max_length,
                    fill_remaining: true,
                }));
            }

            "flow_direction_dinf" => {
                dispatch_f64(&tx, "Flow Direction (D-inf)", start, flow_direction_dinf(&input));
            }

            "flow_accumulation_mfd" => {
                let exponent = get_f64(&params, "exponent", 1.1);
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + MFD accumulation...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    Ok(flow_accumulation_mfd(&filled, MfdParams { exponent })?)
                })();
                dispatch_f64(&tx, "Flow Accumulation (MFD)", start, res);
            }

            "flow_accumulation_mfd_adaptive" => {
                let scale_factor = get_f64(&params, "scale_factor", 1.0);
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + Adaptive MFD accumulation...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    Ok(flow_accumulation_mfd_adaptive(&filled, AdaptiveMfdParams {
                        scale_factor,
                        ..AdaptiveMfdParams::default()
                    })?)
                })();
                dispatch_f64(&tx, "Flow Accumulation (Adaptive MFD)", start, res);
            }

            "flow_accumulation_tfga" => {
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + TFGA accumulation...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    Ok(flow_accumulation_tfga(&filled, TfgaParams::default())?)
                })();
                dispatch_f64(&tx, "Flow Accumulation (TFGA)", start, res);
            }

            "watershed" => {
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow direction for watershed...",
                )));
                let res = (|| -> anyhow::Result<Raster<i32>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    let fdir = flow_direction(&filled)?;
                    Ok(watershed(&fdir, WatershedParams::default())?)
                })();
                dispatch_i32(&tx, "Watershed", start, res);
            }

            "stream_network" => {
                let threshold = get_f64(&params, "stream_threshold", 1000.0);
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow + accumulation for stream network...",
                )));
                let res = (|| -> anyhow::Result<Raster<u8>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    let fdir = flow_direction(&filled)?;
                    let facc = flow_accumulation(&fdir)?;
                    Ok(stream_network(&facc, StreamNetworkParams { threshold })?)
                })();
                dispatch_u8(&tx, "Stream Network", start, res);
            }

            "nested_depressions" => {
                let min_depth = get_f64(&params, "min_depth", 0.1);
                let min_area = get_usize(&params, "min_area", 10);
                match nested_depressions(&input, NestedDepressionParams { min_depth, min_area }) {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        let n = result.depressions.len();
                        let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                            "Found {} nested depressions in {:.2}s", n, elapsed.as_secs_f64()
                        ))));
                        // Return depth raster
                        let _ = tx.send(AppMessage::AlgoComplete {
                            name: "Nested Depressions (depth)".to_string(),
                            result: result.depth,
                            elapsed,
                        });
                    }
                    Err(e) => send_error(&tx, "Nested Depressions", e),
                }
            }

            "flow_dinf" => {
                match flow_dinf(&input) {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        // Send accumulation as primary result
                        let _ = tx.send(AppMessage::AlgoComplete {
                            name: "D-inf Accumulation".to_string(),
                            result: result.accumulation,
                            elapsed,
                        });
                        let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                            "D-inf flow completed in {:.2}s", elapsed.as_secs_f64()
                        ))));
                    }
                    Err(e) => send_error(&tx, "D-inf Flow", e),
                }
            }

            // ═══════════════════════════════════════════════════
            // IMAGERY — new
            // ═══════════════════════════════════════════════════
            "ndre" => {
                require_extra(&tx, "NDRE", &extra_inputs, &["nir", "red_edge"], |inputs| {
                    ndre(inputs["nir"], inputs["red_edge"])
                }, start);
            }

            "gndvi" => {
                require_extra(&tx, "GNDVI", &extra_inputs, &["nir", "green"], |inputs| {
                    gndvi(inputs["nir"], inputs["green"])
                }, start);
            }

            "ngrdi" => {
                require_extra(&tx, "NGRDI", &extra_inputs, &["green", "red"], |inputs| {
                    ngrdi(inputs["green"], inputs["red"])
                }, start);
            }

            "reci" => {
                require_extra(&tx, "RECI", &extra_inputs, &["nir", "red_edge"], |inputs| {
                    reci(inputs["nir"], inputs["red_edge"])
                }, start);
            }

            "normalized_difference" => {
                require_extra(&tx, "Normalized Difference", &extra_inputs, &["a", "b"], |inputs| {
                    normalized_difference(inputs["a"], inputs["b"])
                }, start);
            }

            "reclassify" => {
                let n_classes = get_usize(&params, "n_classes", 5);
                // Auto-generate equal-interval classes from the raster's value range
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let mut vmin = f64::INFINITY;
                    let mut vmax = f64::NEG_INFINITY;
                    for r in 0..input.rows() {
                        for c in 0..input.cols() {
                            if let Ok(v) = input.get(r, c) {
                                if v.is_finite() {
                                    vmin = vmin.min(v);
                                    vmax = vmax.max(v);
                                }
                            }
                        }
                    }
                    let step = (vmax - vmin) / n_classes as f64;
                    let classes: Vec<ReclassEntry> = (0..n_classes)
                        .map(|i| ReclassEntry {
                            min: vmin + i as f64 * step,
                            max: vmin + (i + 1) as f64 * step,
                            value: (i + 1) as f64,
                        })
                        .collect();
                    Ok(reclassify(&input, ReclassifyParams {
                        classes,
                        default_value: 0.0,
                    })?)
                })();
                dispatch_f64(&tx, "Reclassify", start, res);
            }

            "band_math_expr" => {
                let op_idx = get_choice(&params, "operation");
                let res = match op_idx {
                    0 => surtgis_algorithms::imagery::band_math(&input, |v| v.sqrt()),
                    1 => surtgis_algorithms::imagery::band_math(&input, |v| if v > 0.0 { v.ln() } else { f64::NAN }),
                    2 => surtgis_algorithms::imagery::band_math(&input, |v| if v > 0.0 { v.log10() } else { f64::NAN }),
                    3 => surtgis_algorithms::imagery::band_math(&input, |v| v.abs()),
                    4 => surtgis_algorithms::imagery::band_math(&input, |v| -v),
                    5 => surtgis_algorithms::imagery::band_math(&input, |v| v * v),
                    6 => {
                        // normalize 0-1
                        let mut vmin = f64::INFINITY;
                        let mut vmax = f64::NEG_INFINITY;
                        for r in 0..input.rows() {
                            for c in 0..input.cols() {
                                if let Ok(v) = input.get(r, c) {
                                    if v.is_finite() {
                                        vmin = vmin.min(v);
                                        vmax = vmax.max(v);
                                    }
                                }
                            }
                        }
                        let range = vmax - vmin;
                        if range.abs() < 1e-12 {
                            surtgis_algorithms::imagery::band_math(&input, |_| 0.0)
                        } else {
                            surtgis_algorithms::imagery::band_math(&input, move |v| (v - vmin) / range)
                        }
                    }
                    _ => surtgis_algorithms::imagery::band_math(&input, |v| v),
                };
                let name = ["sqrt", "log", "log10", "abs", "negate", "square", "normalize"];
                dispatch_f64(&tx, &format!("Band Math ({})", name.get(op_idx).unwrap_or(&"?")), start, res);
            }

            // ═══════════════════════════════════════════════════
            // IMAGERY — Fase B
            // ═══════════════════════════════════════════════════
            "ndsi" => {
                require_extra(&tx, "NDSI", &extra_inputs, &["green", "swir"], |inputs| {
                    ndsi(inputs["green"], inputs["swir"])
                }, start);
            }

            "ndbi" => {
                require_extra(&tx, "NDBI", &extra_inputs, &["swir", "nir"], |inputs| {
                    ndbi(inputs["swir"], inputs["nir"])
                }, start);
            }

            "ndmi" => {
                require_extra(&tx, "NDMI", &extra_inputs, &["nir", "swir"], |inputs| {
                    ndmi(inputs["nir"], inputs["swir"])
                }, start);
            }

            "msavi" => {
                require_extra(&tx, "MSAVI", &extra_inputs, &["nir", "red"], |inputs| {
                    msavi(inputs["nir"], inputs["red"])
                }, start);
            }

            "evi2" => {
                require_extra(&tx, "EVI2", &extra_inputs, &["nir", "red"], |inputs| {
                    evi2(inputs["nir"], inputs["red"])
                }, start);
            }

            // ═══════════════════════════════════════════════════
            // TERRAIN — new
            // ═══════════════════════════════════════════════════
            "spi" => {
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow + accumulation + slope for SPI...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    let fdir = flow_direction(&filled)?;
                    let facc = flow_accumulation(&fdir)?;
                    let slope_rad = slope(&filled, SlopeParams {
                        units: SlopeUnits::Radians, z_factor: 1.0,
                    })?;
                    Ok(spi(&facc, &slope_rad)?)
                })();
                dispatch_f64(&tx, "SPI", start, res);
            }

            "sti" => {
                let m = get_f64(&params, "m", 0.4);
                let n = get_f64(&params, "n", 1.3);
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow + accumulation + slope for STI...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let filled = fill_sinks(&input, FillSinksParams { min_slope: 0.01 })?;
                    let fdir = flow_direction(&filled)?;
                    let facc = flow_accumulation(&fdir)?;
                    let slope_rad = slope(&filled, SlopeParams {
                        units: SlopeUnits::Radians, z_factor: 1.0,
                    })?;
                    Ok(sti(&facc, &slope_rad, StiParams { m, n })?)
                })();
                dispatch_f64(&tx, "STI", start, res);
            }

            "viewshed" => {
                let observer_row = get_usize(&params, "observer_row", 0);
                let observer_col = get_usize(&params, "observer_col", 0);
                let observer_height = get_f64(&params, "observer_height", 1.7);
                let target_height = get_f64(&params, "target_height", 0.0);
                let max_radius = get_usize(&params, "max_radius", 0);
                dispatch_u8(&tx, "Viewshed", start, viewshed(&input, ViewshedParams {
                    observer_row,
                    observer_col,
                    observer_height,
                    target_height,
                    max_radius,
                }));
            }

            "mrvbf" => {
                let t_slope = get_f64(&params, "t_slope", 16.0);
                let steps = get_usize(&params, "steps", 3);
                match mrvbf(&input, MrvbfParams {
                    initial_slope_threshold: t_slope,
                    steps,
                    ..MrvbfParams::default()
                }) {
                    Ok((mrvbf_raster, _mrrtf_raster)) => {
                        let elapsed = start.elapsed();
                        let _ = tx.send(AppMessage::AlgoComplete {
                            name: "MRVBF".to_string(),
                            result: mrvbf_raster,
                            elapsed,
                        });
                        let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                            "MRVBF completed in {:.2}s", elapsed.as_secs_f64()
                        ))));
                    }
                    Err(e) => send_error(&tx, "MRVBF", e),
                }
            }

            "wind_exposure" => {
                let radius = get_usize(&params, "radius", 30);
                let directions = get_usize(&params, "directions", 8);
                dispatch_f64(&tx, "Wind Exposure", start, wind_exposure(&input, WindExposureParams {
                    radius,
                    directions,
                    wind_direction: None,
                    wind_window: 45.0,
                }));
            }

            "solar_radiation" => {
                let day = get_usize(&params, "day", 172) as u32;
                let latitude = get_f64(&params, "latitude", 40.0);
                let transmittance = get_f64(&params, "transmittance", 0.7);
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing slope + aspect for solar radiation...",
                )));
                let res = (|| -> anyhow::Result<Raster<f64>> {
                    let slope_rad = slope(&input, SlopeParams {
                        units: SlopeUnits::Radians, z_factor: 1.0,
                    })?;
                    let aspect_rad = aspect(&input, AspectOutput::Radians)?;
                    let result = solar_radiation(&slope_rad, &aspect_rad, SolarParams {
                        day,
                        latitude,
                        transmittance,
                        ..SolarParams::default()
                    })?;
                    Ok(result.total)
                })();
                dispatch_f64(&tx, "Solar Radiation (total)", start, res);
            }

            "lineament_detection" => {
                let min_length = get_usize(&params, "min_length", 5);
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing curvatures for lineament detection...",
                )));
                let res = (|| -> anyhow::Result<Raster<u8>> {
                    let plan = curvature(&input, CurvatureParams {
                        curvature_type: CurvatureType::Plan,
                        ..Default::default()
                    })?;
                    let profile = curvature(&input, CurvatureParams {
                        curvature_type: CurvatureType::Profile,
                        ..Default::default()
                    })?;
                    let result = lineament_detection(&plan, &profile, LineamentParams { min_length })?;
                    Ok(result.classified)
                })();
                dispatch_u8(&tx, "Lineament Detection", start, res);
            }

            "advanced_curvatures" => {
                let curv_type = match get_choice(&params, "curv_type") {
                    0 => AdvancedCurvatureType::MeanH,
                    1 => AdvancedCurvatureType::GaussianK,
                    2 => AdvancedCurvatureType::UnsphericitytM,
                    3 => AdvancedCurvatureType::DifferenceE,
                    4 => AdvancedCurvatureType::MinimalKmin,
                    5 => AdvancedCurvatureType::MaximalKmax,
                    6 => AdvancedCurvatureType::HorizontalKh,
                    7 => AdvancedCurvatureType::VerticalKv,
                    8 => AdvancedCurvatureType::HorizontalExcessKhe,
                    9 => AdvancedCurvatureType::VerticalExcessKve,
                    10 => AdvancedCurvatureType::AccumulationKa,
                    11 => AdvancedCurvatureType::RingKr,
                    12 => AdvancedCurvatureType::Rotor,
                    13 => AdvancedCurvatureType::Laplacian,
                    _ => AdvancedCurvatureType::MeanH,
                };
                dispatch_f64(&tx, "Advanced Curvature", start, advanced_curvatures(&input, curv_type));
            }

            // ═══════════════════════════════════════════════════
            // TERRAIN — Fase B
            // ═══════════════════════════════════════════════════
            "contour_lines" => {
                let interval = get_f64(&params, "interval", 10.0);
                let base = get_f64(&params, "base", 0.0);
                dispatch_f64(&tx, "Contour Lines", start, contour_lines(&input, ContourParams {
                    interval,
                    base,
                }));
            }

            "cost_distance" => {
                let source_row = get_usize(&params, "source_row", 0);
                let source_col = get_usize(&params, "source_col", 0);
                dispatch_f64(&tx, "Cost Distance", start, cost_distance(&input, CostDistanceParams {
                    sources: vec![(source_row, source_col)],
                }));
            }

            // ═══════════════════════════════════════════════════
            // LANDSCAPE ECOLOGY — Fase B (new category)
            // ═══════════════════════════════════════════════════
            "shannon_diversity" => {
                let radius = get_usize(&params, "radius", 3);
                dispatch_f64(&tx, "Shannon Diversity", start, shannon_diversity(&input, DiversityParams {
                    radius,
                    circular: false,
                }));
            }

            "simpson_diversity" => {
                let radius = get_usize(&params, "radius", 3);
                dispatch_f64(&tx, "Simpson Diversity", start, simpson_diversity(&input, DiversityParams {
                    radius,
                    circular: false,
                }));
            }

            "patch_density" => {
                let radius = get_usize(&params, "radius", 3);
                dispatch_f64(&tx, "Patch Density", start, patch_density(&input, DiversityParams {
                    radius,
                    circular: false,
                }));
            }

            // ═══════════════════════════════════════════════════
            // INTERPOLATION — new category
            // Extracts valid (non-NaN) pixels as sample points,
            // then interpolates to fill NoData gaps.
            // ═══════════════════════════════════════════════════
            "interp_idw" => {
                let power = get_f64(&params, "power", 2.0);
                let max_points = get_usize(&params, "max_points", 12);
                dispatch_interpolation(&tx, "IDW Interpolation", start, &input, |points, params_grid| {
                    Ok(idw(&points, IdwParams {
                        power,
                        max_points: Some(max_points),
                        rows: params_grid.0,
                        cols: params_grid.1,
                        transform: params_grid.2.clone(),
                        ..IdwParams::default()
                    })?)
                });
            }

            "interp_nearest" => {
                dispatch_interpolation(&tx, "Nearest Neighbor", start, &input, |points, params_grid| {
                    Ok(nearest_neighbor(&points, NearestNeighborParams {
                        max_radius: None,
                        rows: params_grid.0,
                        cols: params_grid.1,
                        transform: params_grid.2.clone(),
                    })?)
                });
            }

            "interp_natural" => {
                dispatch_interpolation(&tx, "Natural Neighbor", start, &input, |points, params_grid| {
                    Ok(natural_neighbor(&points, NaturalNeighborParams {
                        rows: params_grid.0,
                        cols: params_grid.1,
                        transform: params_grid.2.clone(),
                        ..NaturalNeighborParams::default()
                    })?)
                });
            }

            "interp_tin" => {
                dispatch_interpolation(&tx, "TIN Interpolation", start, &input, |points, params_grid| {
                    Ok(tin_interpolation(&points, TinParams {
                        rows: params_grid.0,
                        cols: params_grid.1,
                        transform: params_grid.2.clone(),
                    })?)
                });
            }

            "interp_tps" => {
                let smoothing = get_f64(&params, "smoothing", 0.0);
                dispatch_interpolation(&tx, "Thin Plate Spline", start, &input, |points, params_grid| {
                    Ok(tps_interpolation(&points, TpsParams {
                        rows: params_grid.0,
                        cols: params_grid.1,
                        transform: params_grid.2.clone(),
                        smoothing,
                    })?)
                });
            }

            "interp_kriging" => {
                let max_points = get_usize(&params, "max_points", 16);
                dispatch_interpolation(&tx, "Ordinary Kriging", start, &input, |points, params_grid| {
                    use surtgis_algorithms::interpolation::{
                        empirical_variogram, fit_best_variogram, VariogramParams,
                    };
                    let variogram = empirical_variogram(&points, VariogramParams::default())?;
                    let fitted = fit_best_variogram(&variogram)?;
                    let result = ordinary_kriging(&points, &fitted, OrdinaryKrigingParams {
                        rows: params_grid.0,
                        cols: params_grid.1,
                        transform: params_grid.2.clone(),
                        max_points,
                        ..OrdinaryKrigingParams::default()
                    })?;
                    Ok(result.estimate)
                });
            }

            _ => {
                let _ = tx.send(AppMessage::Error {
                    context: "Executor".to_string(),
                    message: format!("Unknown algorithm: {}", algo_id),
                });
            }
        }
    });
}

// ─── Helpers ───────────────────────────────────────────────────────────

fn get_f64(params: &HashMap<String, ParamValue>, key: &str, default: f64) -> f64 {
    params.get(key).map(|v| v.as_f64()).unwrap_or(default)
}

fn get_usize(params: &HashMap<String, ParamValue>, key: &str, default: usize) -> usize {
    params.get(key).map(|v| v.as_usize()).unwrap_or(default)
}

fn get_choice(params: &HashMap<String, ParamValue>, key: &str) -> usize {
    params.get(key).map(|v| v.as_choice_index()).unwrap_or(0)
}

fn make_se(params: &HashMap<String, ParamValue>) -> StructuringElement {
    let radius = get_usize(params, "radius", 1);
    match get_choice(params, "shape") {
        1 => StructuringElement::Cross(radius),
        2 => StructuringElement::Disk(radius),
        _ => StructuringElement::Square(radius),
    }
}

fn dispatch_f64<E: std::fmt::Display>(
    tx: &Sender<AppMessage>,
    name: &str,
    start: Instant,
    result: Result<Raster<f64>, E>,
) {
    match result {
        Ok(raster) => {
            let elapsed = start.elapsed();
            let _ = tx.send(AppMessage::AlgoComplete {
                name: name.to_string(),
                result: raster,
                elapsed,
            });
            let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                "{} completed in {:.2}s", name, elapsed.as_secs_f64()
            ))));
        }
        Err(e) => send_error(tx, name, e),
    }
}

fn dispatch_u8<E: std::fmt::Display>(
    tx: &Sender<AppMessage>,
    name: &str,
    start: Instant,
    result: Result<Raster<u8>, E>,
) {
    match result {
        Ok(raster) => {
            let elapsed = start.elapsed();
            let _ = tx.send(AppMessage::AlgoCompleteU8 {
                name: name.to_string(),
                result: raster,
                elapsed,
            });
            let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                "{} completed in {:.2}s", name, elapsed.as_secs_f64()
            ))));
        }
        Err(e) => send_error(tx, name, e),
    }
}

fn dispatch_focal(
    tx: &Sender<AppMessage>,
    start: Instant,
    input: &Raster<f64>,
    params: &HashMap<String, ParamValue>,
    statistic: FocalStatistic,
    name: &str,
) {
    dispatch_f64(tx, name, start, focal_statistics(input, FocalParams {
        radius: get_usize(params, "radius", 3),
        statistic,
        circular: false,
    }));
}

/// Helper for multi-input algorithms: checks all required extra inputs exist.
fn require_extra<F>(
    tx: &Sender<AppMessage>,
    name: &str,
    extra_inputs: &HashMap<String, Raster<f64>>,
    required: &[&str],
    f: F,
    start: Instant,
) where
    F: FnOnce(HashMap<&str, &Raster<f64>>) -> surtgis_core::Result<Raster<f64>>,
{
    let mut inputs = HashMap::new();
    for &key in required {
        match extra_inputs.get(key) {
            Some(r) => { inputs.insert(key, r); }
            None => {
                send_error(tx, name, format!("Missing input: {}", key));
                return;
            }
        }
    }
    dispatch_f64(tx, name, start, f(inputs));
}

fn dispatch_i32<E: std::fmt::Display>(
    tx: &Sender<AppMessage>,
    name: &str,
    start: Instant,
    result: Result<Raster<i32>, E>,
) {
    match result {
        Ok(raster) => {
            let elapsed = start.elapsed();
            let _ = tx.send(AppMessage::AlgoCompleteI32 {
                name: name.to_string(),
                result: raster,
                elapsed,
            });
            let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                "{} completed in {:.2}s", name, elapsed.as_secs_f64()
            ))));
        }
        Err(e) => send_error(tx, name, e),
    }
}

/// Helper for interpolation algorithms: extracts valid pixels as sample points,
/// runs the interpolation, and dispatches the result.
fn dispatch_interpolation<F>(
    tx: &Sender<AppMessage>,
    name: &str,
    start: Instant,
    input: &Raster<f64>,
    interp_fn: F,
) where
    F: FnOnce(Vec<SamplePoint>, (usize, usize, GeoTransform)) -> anyhow::Result<Raster<f64>>,
{
    let transform = input.transform().clone();
    let rows = input.rows();
    let cols = input.cols();

    // Extract valid (non-NaN) pixels as sample points
    let mut points = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            if let Ok(v) = input.get(r, c) {
                if v.is_finite() {
                    let x = transform.origin_x + c as f64 * transform.pixel_width + r as f64 * transform.row_rotation;
                    let y = transform.origin_y + c as f64 * transform.col_rotation + r as f64 * transform.pixel_height;
                    points.push(SamplePoint::new(x, y, v));
                }
            }
        }
    }

    if points.is_empty() {
        send_error(tx, name, "No valid (non-NaN) pixels found for interpolation");
        return;
    }

    let _ = tx.send(AppMessage::Log(LogEntry::info(format!(
        "{}: {} sample points from {}x{} raster", name, points.len(), rows, cols
    ))));

    let res: anyhow::Result<Raster<f64>> = interp_fn(points, (rows, cols, transform));
    dispatch_f64(tx, name, start, res);
}

fn send_error(tx: &Sender<AppMessage>, name: &str, err: impl std::fmt::Display) {
    let msg = format!("{}: {}", name, err);
    let _ = tx.send(AppMessage::Error {
        context: name.to_string(),
        message: err.to_string(),
    });
    let _ = tx.send(AppMessage::Log(LogEntry::error(msg)));
}
