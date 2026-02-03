//! Algorithm execution in background threads.
//!
//! Each algorithm runs in a separate `std::thread` and sends results back
//! via `crossbeam_channel`.

use std::collections::HashMap;
use std::time::Instant;

use crossbeam_channel::Sender;

use surtgis_algorithms::hydrology::{
    fill_sinks, flow_accumulation, flow_direction, hand, priority_flood_flat, FillSinksParams,
    HandParams,
};
use surtgis_algorithms::imagery::{
    band_math_binary, bsi, evi, mndwi, nbr, ndvi, ndwi, savi, BandMathOp, EviParams, SaviParams,
};
use surtgis_algorithms::morphology::{
    black_hat, closing, dilate, erode, gradient, opening, top_hat, StructuringElement,
};
use surtgis_algorithms::statistics::{focal_statistics, FocalParams, FocalStatistic};
use surtgis_algorithms::terrain::{
    aspect, convergence_index, curvature, curvedness, dev, eastness, geomorphons, hillshade,
    landform_classification, multidirectional_hillshade, negative_openness, northness,
    positive_openness, shape_index, sky_view_factor, slope, tpi, tri, vrm, AspectOutput,
    ConvergenceParams, CurvatureParams, CurvatureType, DevParams, GeomorphonParams,
    HillshadeParams, LandformParams, MultiHillshadeParams, OpennessParams, SlopeParams,
    SlopeUnits, SvfParams, TpiParams, TriParams, VrmParams,
};
use surtgis_core::raster::Raster;

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

fn send_error(tx: &Sender<AppMessage>, name: &str, err: impl std::fmt::Display) {
    let msg = format!("{}: {}", name, err);
    let _ = tx.send(AppMessage::Error {
        context: name.to_string(),
        message: err.to_string(),
    });
    let _ = tx.send(AppMessage::Log(LogEntry::error(msg)));
}
