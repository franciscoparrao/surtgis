//! CLI handlers for geostatistical interpolation commands.

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use geo::Point;

use surtgis_algorithms::interpolation::{
    empirical_variogram, fit_best_variogram, ordinary_kriging,
    universal_kriging, regression_kriging,
    OrdinaryKrigingParams, UniversalKrigingParams,
    RegressionKrigingParams,
    SamplePoint, VariogramParams, FittedVariogram, VariogramModel, DriftOrder,
};
use surtgis_core::io::read_geotiff;
use surtgis_core::raster::Raster;
use surtgis_core::vector::{read_vector, AttributeValue};

use crate::commands::InterpolationCommands;
use crate::helpers::{done, spinner, write_result};

pub fn handle(action: InterpolationCommands, compress: bool) -> Result<()> {
    match action {
        InterpolationCommands::Variogram {
            points, attribute, bins, max_lag, output,
        } => {
            let start = Instant::now();
            let samples = load_samples(&points, &attribute)?;
            println!("Loaded {} sample points", samples.len());

            let pb = spinner("Computing variogram...");
            let params = VariogramParams { n_lags: bins, max_lag, lag_tolerance: 1.0 };
            let empirical = empirical_variogram(&samples, params)
                .context("Failed to compute empirical variogram")?;
            let fitted = fit_best_variogram(&empirical)
                .context("Failed to fit variogram model")?;
            pb.finish_and_clear();

            println!("Model: {:?} — range={:.2}, sill={:.6}, nugget={:.6}",
                fitted.model, fitted.range, fitted.sill, fitted.nugget);

            let result = serde_json::json!({
                "empirical": {
                    "lags": empirical.lags,
                    "semivariance": empirical.semivariance,
                    "pair_counts": empirical.pair_counts,
                },
                "fitted": {
                    "model": format!("{:?}", fitted.model),
                    "range": fitted.range,
                    "sill": fitted.sill,
                    "nugget": fitted.nugget,
                    "partial_sill": fitted.partial_sill,
                },
                "n_points": samples.len(),
            });
            std::fs::write(&output, serde_json::to_string_pretty(&result)?)
                .context("Failed to write variogram JSON")?;
            done("Variogram", &output, start.elapsed());
        }

        InterpolationCommands::Kriging {
            points, attribute, reference, model,
            range, sill, nugget, max_neighbors, output,
        } => {
            let start = Instant::now();
            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();
            println!("Points: {}, Grid: {}x{}", samples.len(), cols, rows);

            let variogram = if let (Some(r), Some(s)) = (range, sill) {
                FittedVariogram {
                    model: parse_variogram_model(&model),
                    range: r, sill: s, nugget,
                    partial_sill: s - nugget, rss: 0.0,
                }
            } else {
                let pb = spinner("Auto-fitting variogram...");
                let emp = empirical_variogram(&samples, VariogramParams { n_lags: 15, max_lag: None, lag_tolerance: 1.0 })?;
                let v = fit_best_variogram(&emp)?;
                pb.finish_and_clear();
                println!("Auto-fit: {:?} (range={:.1}, sill={:.4})", v.model, v.range, v.sill);
                v
            };

            let pb = spinner("Kriging...");
            let params = OrdinaryKrigingParams {
                rows, cols,
                transform: ref_raster.transform().clone(),
                max_points: max_neighbors,
                max_radius: None,
                compute_variance: false,
            };
            let result = ordinary_kriging(&samples, &variogram, params)
                .context("Kriging failed")?;
            pb.finish_and_clear();

            let mut out = result.estimate;
            out.set_transform(ref_raster.transform().clone());
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("Ordinary Kriging", &output, start.elapsed());
        }

        InterpolationCommands::UniversalKriging {
            points, attribute, reference, drift, model, max_neighbors, output,
        } => {
            let start = Instant::now();
            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();

            let drift_order = match drift.to_lowercase().as_str() {
                "quadratic" => DriftOrder::Quadratic,
                _ => DriftOrder::Linear,
            };

            let pb = spinner("Auto-fitting variogram...");
            let emp = empirical_variogram(&samples, VariogramParams { n_lags: 15, max_lag: None, lag_tolerance: 1.0 })?;
            let variogram = fit_best_variogram(&emp)?;
            pb.finish_and_clear();

            let pb = spinner("Universal Kriging...");
            let params = UniversalKrigingParams {
                rows, cols,
                transform: ref_raster.transform().clone(),
                max_points: max_neighbors,
                max_radius: None,
                drift_order,
                compute_variance: false,
            };
            let result = universal_kriging(&samples, &variogram, params)
                .context("Universal Kriging failed")?;
            pb.finish_and_clear();

            let mut out = result.estimate;
            out.set_transform(ref_raster.transform().clone());
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("Universal Kriging", &output, start.elapsed());
        }

        InterpolationCommands::RegressionKriging {
            points, attribute, features: _features, reference, model: _model, output,
        } => {
            let start = Instant::now();
            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();
            println!("Points: {}, Grid: {}x{}", samples.len(), cols, rows);

            let pb = spinner("Regression-Kriging...");
            let params = RegressionKrigingParams {
                rows, cols,
                transform: ref_raster.transform().clone(),
                max_points: 16,
                max_radius: None,
                compute_variance: false,
                variogram_params: VariogramParams { n_lags: 15, max_lag: None, lag_tolerance: 1.0 },
            };
            let result = regression_kriging(&samples, params)
                .context("Regression-Kriging failed")?;
            pb.finish_and_clear();

            let mut out = result.estimate;
            out.set_transform(ref_raster.transform().clone());
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("Regression-Kriging", &output, start.elapsed());
        }

        InterpolationCommands::Idw {
            points, attribute, reference, power, max_radius, max_points, output,
        } => {
            use surtgis_algorithms::interpolation::{idw, IdwParams};

            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();
            println!("Points: {}, Grid: {}x{}", samples.len(), cols, rows);

            let start = Instant::now();
            let pb = spinner(&format!("IDW (power={})...", power));
            let result = idw(&samples, IdwParams {
                power,
                max_radius,
                max_points,
                rows,
                cols,
                transform: ref_raster.transform().clone(),
                ..IdwParams::default()
            }).context("IDW failed")?;
            pb.finish_and_clear();

            let mut out = result;
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("IDW", &output, start.elapsed());
        }

        InterpolationCommands::NearestNeighbor {
            points, attribute, reference, max_radius, output,
        } => {
            use surtgis_algorithms::interpolation::{nearest_neighbor, NearestNeighborParams};

            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();
            println!("Points: {}, Grid: {}x{}", samples.len(), cols, rows);

            let start = Instant::now();
            let pb = spinner("Nearest Neighbor...");
            let result = nearest_neighbor(&samples, NearestNeighborParams {
                max_radius,
                rows,
                cols,
                transform: ref_raster.transform().clone(),
            }).context("Nearest Neighbor failed")?;
            pb.finish_and_clear();

            let mut out = result;
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("Nearest Neighbor", &output, start.elapsed());
        }

        InterpolationCommands::NaturalNeighbor {
            points, attribute, reference, output,
        } => {
            use surtgis_algorithms::interpolation::{natural_neighbor, NaturalNeighborParams};

            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();
            println!("Points: {}, Grid: {}x{}", samples.len(), cols, rows);

            let start = Instant::now();
            let pb = spinner("Natural Neighbor...");
            let result = natural_neighbor(&samples, NaturalNeighborParams {
                max_neighbors: 20,
                sub_resolution: 11,
                rows,
                cols,
                transform: ref_raster.transform().clone(),
            }).context("Natural Neighbor failed")?;
            pb.finish_and_clear();

            let mut out = result;
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("Natural Neighbor", &output, start.elapsed());
        }

        InterpolationCommands::Tps {
            points, attribute, reference, smoothing, output,
        } => {
            use surtgis_algorithms::interpolation::{tps_interpolation, TpsParams};

            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();
            println!("Points: {}, Grid: {}x{}", samples.len(), cols, rows);

            let start = Instant::now();
            let pb = spinner("Thin Plate Spline...");
            let result = tps_interpolation(&samples, TpsParams {
                rows,
                cols,
                transform: ref_raster.transform().clone(),
                smoothing,
            }).context("TPS failed")?;
            pb.finish_and_clear();

            let mut out = result;
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("TPS", &output, start.elapsed());
        }

        InterpolationCommands::Tin {
            points, attribute, reference, output,
        } => {
            use surtgis_algorithms::interpolation::{tin_interpolation, TinParams};

            let samples = load_samples(&points, &attribute)?;
            let ref_raster: Raster<f64> = read_geotiff(&reference, None)?;
            let (rows, cols) = ref_raster.shape();
            println!("Points: {}, Grid: {}x{}", samples.len(), cols, rows);

            let start = Instant::now();
            let pb = spinner("TIN interpolation...");
            let result = tin_interpolation(&samples, TinParams {
                rows,
                cols,
                transform: ref_raster.transform().clone(),
            }).context("TIN failed")?;
            pb.finish_and_clear();

            let mut out = result;
            out.set_crs(ref_raster.crs().cloned());
            write_result(&out, &output, compress)?;
            done("TIN", &output, start.elapsed());
        }
    }
    Ok(())
}

fn load_samples(path: &Path, attribute: &str) -> Result<Vec<SamplePoint>> {
    let fc = read_vector(path).context("Failed to read vector file")?;

    let mut samples = Vec::new();
    for feature in &fc.features {
        let geom = feature.geometry.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Feature missing geometry"))?;

        let (x, y) = match geom {
            geo::Geometry::Point(p) => (p.x(), p.y()),
            _ => continue, // skip non-point geometries
        };

        let value = match feature.properties.get(attribute) {
            Some(AttributeValue::Float(v)) => *v,
            Some(AttributeValue::Int(v)) => *v as f64,
            _ => anyhow::bail!("Missing or non-numeric attribute '{}'", attribute),
        };

        samples.push(SamplePoint { x, y, value });
    }

    anyhow::ensure!(!samples.is_empty(), "No valid point samples found");
    Ok(samples)
}

fn parse_variogram_model(s: &str) -> VariogramModel {
    match s.to_lowercase().as_str() {
        "exponential" => VariogramModel::Exponential,
        "gaussian" => VariogramModel::Gaussian,
        _ => VariogramModel::Spherical,
    }
}
