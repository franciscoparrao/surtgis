//! Algorithm execution in background threads.
//!
//! Each algorithm runs in a separate `std::thread` and sends results back
//! via `crossbeam_channel`.

use std::collections::HashMap;
use std::time::Instant;

use crossbeam_channel::Sender;

use surtgis_algorithms::hydrology::{fill_sinks, flow_direction, flow_accumulation, FillSinksParams};
use surtgis_algorithms::imagery::{ndvi, ndwi};
use surtgis_algorithms::statistics::{focal_statistics, FocalParams, FocalStatistic};
use surtgis_algorithms::terrain::{
    aspect, curvature, dev, geomorphons, hillshade, multidirectional_hillshade, slope, tpi, tri,
    AspectOutput, CurvatureParams, CurvatureType, DevParams, GeomorphonParams, HillshadeParams,
    MultiHillshadeParams, SlopeParams, SlopeUnits, TpiParams, TriParams,
};
use surtgis_core::raster::Raster;

use crate::registry::ParamValue;
use crate::state::{AppMessage, LogEntry};

/// Dispatch an algorithm by ID, running it in a background thread.
///
/// `input` is the primary input raster. `params` are the user-configured parameter values.
/// `extra_inputs` holds additional input rasters keyed by parameter name (for multi-input algos).
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
            "slope" => {
                let units = match params.get("units").map(|v| v.as_choice_index()).unwrap_or(0) {
                    0 => SlopeUnits::Degrees,
                    1 => SlopeUnits::Percent,
                    2 => SlopeUnits::Radians,
                    _ => SlopeUnits::Degrees,
                };
                let z_factor = params.get("z_factor").map(|v| v.as_f64()).unwrap_or(1.0);
                match slope(&input, SlopeParams { units, z_factor }) {
                    Ok(result) => send_f64(&tx, "Slope", result, start),
                    Err(e) => send_error(&tx, "Slope", e),
                }
            }

            "aspect" => {
                let fmt = match params.get("format").map(|v| v.as_choice_index()).unwrap_or(0) {
                    0 => AspectOutput::Degrees,
                    1 => AspectOutput::Radians,
                    _ => AspectOutput::Degrees,
                };
                match aspect(&input, fmt) {
                    Ok(result) => send_f64(&tx, "Aspect", result, start),
                    Err(e) => send_error(&tx, "Aspect", e),
                }
            }

            "hillshade" => {
                let azimuth = params.get("azimuth").map(|v| v.as_f64()).unwrap_or(315.0);
                let altitude = params.get("altitude").map(|v| v.as_f64()).unwrap_or(45.0);
                let z_factor = params.get("z_factor").map(|v| v.as_f64()).unwrap_or(1.0);
                match hillshade(
                    &input,
                    HillshadeParams {
                        azimuth,
                        altitude,
                        z_factor,
                        normalized: false,
                    },
                ) {
                    Ok(result) => send_f64(&tx, "Hillshade", result, start),
                    Err(e) => send_error(&tx, "Hillshade", e),
                }
            }

            "curvature" => {
                let ct = match params.get("type").map(|v| v.as_choice_index()).unwrap_or(0) {
                    0 => CurvatureType::General,
                    1 => CurvatureType::Profile,
                    2 => CurvatureType::Plan,
                    _ => CurvatureType::General,
                };
                let z_factor = params.get("z_factor").map(|v| v.as_f64()).unwrap_or(1.0);
                match curvature(
                    &input,
                    CurvatureParams {
                        curvature_type: ct,
                        z_factor,
                        ..Default::default()
                    },
                ) {
                    Ok(result) => send_f64(&tx, "Curvature", result, start),
                    Err(e) => send_error(&tx, "Curvature", e),
                }
            }

            "tpi" => {
                let radius = params.get("radius").map(|v| v.as_usize()).unwrap_or(3);
                match tpi(&input, TpiParams { radius }) {
                    Ok(result) => send_f64(&tx, "TPI", result, start),
                    Err(e) => send_error(&tx, "TPI", e),
                }
            }

            "tri" => {
                let radius = params.get("radius").map(|v| v.as_usize()).unwrap_or(1);
                match tri(&input, TriParams { radius }) {
                    Ok(result) => send_f64(&tx, "TRI", result, start),
                    Err(e) => send_error(&tx, "TRI", e),
                }
            }

            "twi" => {
                // TWI requires flow accumulation + slope. Compute both from DEM.
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing flow direction + accumulation + slope for TWI...",
                )));
                let filled = match fill_sinks(&input, FillSinksParams { min_slope: 0.01 }) {
                    Ok(r) => r,
                    Err(e) => {
                        send_error(&tx, "TWI (fill sinks)", e);
                        return;
                    }
                };
                let fdir = match flow_direction(&filled) {
                    Ok(r) => r,
                    Err(e) => {
                        send_error(&tx, "TWI (flow direction)", e);
                        return;
                    }
                };
                let facc = match flow_accumulation(&fdir) {
                    Ok(r) => r,
                    Err(e) => {
                        send_error(&tx, "TWI (flow accumulation)", e);
                        return;
                    }
                };
                let slope_rad = match slope(
                    &filled,
                    SlopeParams {
                        units: SlopeUnits::Radians,
                        z_factor: 1.0,
                    },
                ) {
                    Ok(r) => r,
                    Err(e) => {
                        send_error(&tx, "TWI (slope)", e);
                        return;
                    }
                };
                match surtgis_algorithms::terrain::twi(&facc, &slope_rad) {
                    Ok(result) => send_f64(&tx, "TWI", result, start),
                    Err(e) => send_error(&tx, "TWI", e),
                }
            }

            "geomorphons" => {
                let flatness = params.get("flatness").map(|v| v.as_f64()).unwrap_or(1.0);
                let radius = params.get("radius").map(|v| v.as_usize()).unwrap_or(10);
                match geomorphons(
                    &input,
                    GeomorphonParams {
                        radius,
                        flatness_threshold: flatness,
                    },
                ) {
                    Ok(result) => {
                        let elapsed = start.elapsed();
                        let _ = tx.send(AppMessage::AlgoCompleteU8 {
                            name: "Geomorphons".to_string(),
                            result,
                            elapsed,
                        });
                        let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                            "Geomorphons completed in {:.2}s",
                            elapsed.as_secs_f64()
                        ))));
                    }
                    Err(e) => send_error(&tx, "Geomorphons", e),
                }
            }

            "multidirectional_hillshade" => {
                match multidirectional_hillshade(&input, MultiHillshadeParams::default()) {
                    Ok(result) => send_f64(&tx, "Multidirectional Hillshade", result, start),
                    Err(e) => send_error(&tx, "Multidirectional Hillshade", e),
                }
            }

            "dev" => {
                let radius = params.get("radius").map(|v| v.as_usize()).unwrap_or(10);
                match dev(&input, DevParams { radius }) {
                    Ok(result) => send_f64(&tx, "DEV", result, start),
                    Err(e) => send_error(&tx, "DEV", e),
                }
            }

            "fill_sinks" => {
                match fill_sinks(&input, FillSinksParams { min_slope: 0.01 }) {
                    Ok(result) => send_f64(&tx, "Fill Sinks", result, start),
                    Err(e) => send_error(&tx, "Fill Sinks", e),
                }
            }

            "flow_direction" => match flow_direction(&input) {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let _ = tx.send(AppMessage::AlgoCompleteU8 {
                        name: "Flow Direction".to_string(),
                        result,
                        elapsed,
                    });
                    let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
                        "Flow Direction completed in {:.2}s",
                        elapsed.as_secs_f64()
                    ))));
                }
                Err(e) => send_error(&tx, "Flow Direction", e),
            },

            "flow_accumulation" => {
                // Flow accumulation expects a D8 flow direction as u8 input.
                // For simplicity in MVP, compute from DEM directly.
                let _ = tx.send(AppMessage::Log(LogEntry::info(
                    "Computing fill + flow direction first...",
                )));
                let filled = match fill_sinks(&input, FillSinksParams { min_slope: 0.01 }) {
                    Ok(r) => r,
                    Err(e) => {
                        send_error(&tx, "Flow Accumulation (fill)", e);
                        return;
                    }
                };
                let fdir = match flow_direction(&filled) {
                    Ok(r) => r,
                    Err(e) => {
                        send_error(&tx, "Flow Accumulation (fdir)", e);
                        return;
                    }
                };
                match flow_accumulation(&fdir) {
                    Ok(result) => send_f64(&tx, "Flow Accumulation", result, start),
                    Err(e) => send_error(&tx, "Flow Accumulation", e),
                }
            }

            "ndvi" => {
                if let Some(nir) = extra_inputs.get("nir") {
                    match ndvi(nir, &input) {
                        Ok(result) => send_f64(&tx, "NDVI", result, start),
                        Err(e) => send_error(&tx, "NDVI", e),
                    }
                } else {
                    send_error(
                        &tx,
                        "NDVI",
                        anyhow::anyhow!("NIR band not provided"),
                    );
                }
            }

            "ndwi" => {
                if let Some(green) = extra_inputs.get("green") {
                    match ndwi(green, &input) {
                        Ok(result) => send_f64(&tx, "NDWI", result, start),
                        Err(e) => send_error(&tx, "NDWI", e),
                    }
                } else {
                    send_error(
                        &tx,
                        "NDWI",
                        anyhow::anyhow!("Green band not provided"),
                    );
                }
            }

            "focal_mean" => {
                let radius = params.get("radius").map(|v| v.as_usize()).unwrap_or(3);
                match focal_statistics(
                    &input,
                    FocalParams {
                        radius,
                        statistic: FocalStatistic::Mean,
                        circular: false,
                    },
                ) {
                    Ok(result) => send_f64(&tx, "Focal Mean", result, start),
                    Err(e) => send_error(&tx, "Focal Mean", e),
                }
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

fn send_f64(tx: &Sender<AppMessage>, name: &str, result: Raster<f64>, start: Instant) {
    let elapsed = start.elapsed();
    let _ = tx.send(AppMessage::AlgoComplete {
        name: name.to_string(),
        result,
        elapsed,
    });
    let _ = tx.send(AppMessage::Log(LogEntry::success(format!(
        "{} completed in {:.2}s",
        name,
        elapsed.as_secs_f64()
    ))));
}

fn send_error(tx: &Sender<AppMessage>, name: &str, err: impl std::fmt::Display) {
    let msg = format!("{}: {}", name, err);
    let _ = tx.send(AppMessage::Error {
        context: name.to_string(),
        message: err.to_string(),
    });
    let _ = tx.send(AppMessage::Log(LogEntry::error(msg)));
}
