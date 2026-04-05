//! Handler for temporal analysis subcommands.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::imagery::{raster_difference, RasterDiffParams};
use surtgis_algorithms::temporal::{
    temporal_percentile, temporal_stats,
    linear_trend, mann_kendall, sens_slope,
    temporal_anomaly, AnomalyMethod,
    vegetation_phenology, PhenologyParams,
};

use crate::commands::TemporalCommands;
use crate::helpers::{done, read_dem, write_result};

pub fn handle(cmd: TemporalCommands, compress: bool) -> Result<()> {
    match cmd {
        TemporalCommands::Stats { input, outdir, stats } => handle_stats(input, outdir, stats, compress),
        TemporalCommands::Trend { input, outdir, method, times } => handle_trend(input, outdir, method, times, compress),
        TemporalCommands::Change { before, after, output, decrease_threshold, increase_threshold } => {
            handle_change(before, after, output, decrease_threshold, increase_threshold, compress)
        }
        TemporalCommands::Anomaly { reference, target, outdir, method } => {
            handle_anomaly(reference, target, outdir, method, compress)
        }
        TemporalCommands::Phenology { input, outdir, doys, threshold, smooth } => {
            handle_phenology(input, outdir, doys, threshold, smooth, compress)
        }
    }
}

fn load_raster_stack(paths: &[PathBuf]) -> Result<Vec<surtgis_core::Raster<f64>>> {
    println!("Loading {} rasters...", paths.len());
    let mut rasters = Vec::with_capacity(paths.len());
    for (i, path) in paths.iter().enumerate() {
        let r = read_dem(path).with_context(|| format!("Failed to read raster {}: {}", i, path.display()))?;
        rasters.push(r);
    }
    Ok(rasters)
}

fn parse_times(times_str: &str) -> Result<Vec<f64>> {
    times_str.split(',')
        .map(|s| s.trim().parse::<f64>().with_context(|| format!("invalid time value: '{}'", s)))
        .collect()
}

fn handle_stats(input: Vec<PathBuf>, outdir: PathBuf, stats_str: String, compress: bool) -> Result<()> {
    let rasters = load_raster_stack(&input)?;
    let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();

    std::fs::create_dir_all(&outdir)?;
    let start = Instant::now();

    let requested: Vec<&str> = stats_str.split(',').map(|s| s.trim()).collect();

    // Use single-pass temporal_stats for the common ones
    let has_basic = requested.iter().any(|s| matches!(*s, "mean" | "std" | "min" | "max" | "count"));

    if has_basic {
        println!("Computing temporal statistics...");
        let ts = temporal_stats(&refs).context("temporal_stats failed")?;

        for stat_name in &requested {
            let (raster, name) = match *stat_name {
                "mean" => (Some(&ts.mean), "mean"),
                "std" => (Some(&ts.std), "std"),
                "min" => (Some(&ts.min), "min"),
                "max" => (Some(&ts.max), "max"),
                "count" => (Some(&ts.count), "count"),
                _ => (None, *stat_name),
            };
            if let Some(r) = raster {
                let path = outdir.join(format!("{}.tif", name));
                write_result(r, &path, compress)?;
                println!("  {} → {}", name, path.display());
            }
        }
    }

    // Handle percentiles
    for stat_name in &requested {
        if stat_name.starts_with('p') {
            if let Ok(pct) = stat_name[1..].parse::<f64>() {
                println!("Computing percentile {}...", pct);
                let r = temporal_percentile(&refs, pct)
                    .with_context(|| format!("percentile {} failed", pct))?;
                let path = outdir.join(format!("p{}.tif", pct as u32));
                write_result(&r, &path, compress)?;
                println!("  p{} → {}", pct as u32, path.display());
            }
        }
    }

    done("Temporal statistics", &outdir, start.elapsed());
    Ok(())
}

fn handle_trend(input: Vec<PathBuf>, outdir: PathBuf, method: String, times_str: Option<String>, compress: bool) -> Result<()> {
    let rasters = load_raster_stack(&input)?;
    let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();

    let times = times_str.as_ref().map(|s| parse_times(s)).transpose()?;

    std::fs::create_dir_all(&outdir)?;
    let start = Instant::now();

    match method.as_str() {
        "linear" | "ols" => {
            println!("Computing linear trend (OLS)...");
            let result = linear_trend(&refs, times.as_deref())
                .context("linear_trend failed")?;

            let slope_path = outdir.join("slope.tif");
            let intercept_path = outdir.join("intercept.tif");
            let r2_path = outdir.join("r_squared.tif");
            let pval_path = outdir.join("p_value.tif");

            write_result(&result.slope, &slope_path, compress)?;
            write_result(&result.intercept, &intercept_path, compress)?;
            write_result(&result.r_squared, &r2_path, compress)?;
            write_result(&result.p_value, &pval_path, compress)?;

            println!("  slope     → {}", slope_path.display());
            println!("  intercept → {}", intercept_path.display());
            println!("  R²        → {}", r2_path.display());
            println!("  p-value   → {}", pval_path.display());
        }
        "mann-kendall" | "mk" => {
            println!("Computing Mann-Kendall trend test...");
            let result = mann_kendall(&refs).context("mann_kendall failed")?;

            let tau_path = outdir.join("tau.tif");
            let pval_path = outdir.join("p_value.tif");
            let trend_path = outdir.join("trend.tif");
            let sens_path = outdir.join("sens_slope.tif");

            write_result(&result.tau, &tau_path, compress)?;
            write_result(&result.p_value, &pval_path, compress)?;
            write_result(&result.trend, &trend_path, compress)?;
            write_result(&result.sens_slope, &sens_path, compress)?;

            println!("  tau        → {}", tau_path.display());
            println!("  p-value    → {}", pval_path.display());
            println!("  trend      → {} (1=up, 0=none, -1=down)", trend_path.display());
            println!("  Sen's slope → {}", sens_path.display());
        }
        "sens" | "sens-slope" => {
            println!("Computing Sen's slope...");
            let result = sens_slope(&refs).context("sens_slope failed")?;
            let path = outdir.join("sens_slope.tif");
            write_result(&result, &path, compress)?;
            println!("  Sen's slope → {}", path.display());
        }
        _ => anyhow::bail!("unknown trend method: '{}'. Use: linear, mann-kendall, sens", method),
    }

    done("Trend analysis", &outdir, start.elapsed());
    Ok(())
}

fn handle_change(
    before: PathBuf, after: PathBuf, output: PathBuf,
    decrease_threshold: f64, increase_threshold: f64, compress: bool,
) -> Result<()> {
    let before_r = read_dem(&before)?;
    let after_r = read_dem(&after)?;

    let start = Instant::now();
    println!("Computing change detection...");

    let params = RasterDiffParams {
        decrease_threshold,
        increase_threshold,
    };
    let (diff, cat) = raster_difference(&before_r, &after_r, params)
        .context("raster_difference failed")?;

    // Write difference raster
    write_result(&diff, &output, compress)?;

    // Write categorical raster alongside
    let cat_path = output.with_file_name(
        format!("{}_categorical.tif",
            output.file_stem().unwrap_or_default().to_string_lossy()
        )
    );
    write_result(&cat, &cat_path, compress)?;

    println!("  difference  → {}", output.display());
    println!("  categorical → {} (1=decrease, 2=stable, 3=increase)", cat_path.display());

    done("Change detection", &output, start.elapsed());
    Ok(())
}

fn handle_anomaly(
    reference: Vec<PathBuf>, target: Vec<PathBuf>, outdir: PathBuf,
    method_str: String, compress: bool,
) -> Result<()> {
    let ref_rasters = load_raster_stack(&reference)?;
    let tgt_rasters = load_raster_stack(&target)?;
    let ref_refs: Vec<&surtgis_core::Raster<f64>> = ref_rasters.iter().collect();
    let tgt_refs: Vec<&surtgis_core::Raster<f64>> = tgt_rasters.iter().collect();

    let method = AnomalyMethod::from_str(&method_str)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    std::fs::create_dir_all(&outdir)?;
    let start = Instant::now();

    println!("Computing anomaly ({:?}) with {} reference, {} target rasters...",
        method, ref_rasters.len(), tgt_rasters.len());

    let results = temporal_anomaly(&ref_refs, &tgt_refs, method)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    for (i, result) in results.iter().enumerate() {
        let name = target[i].file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("anomaly_{}", i));
        let path = outdir.join(format!("{}_anomaly.tif", name));
        write_result(result, &path, compress)?;
        println!("  {} → {}", name, path.display());
    }

    done("Anomaly detection", &outdir, start.elapsed());
    Ok(())
}

fn handle_phenology(
    input: Vec<PathBuf>, outdir: PathBuf, doys_str: Option<String>,
    threshold: f64, smooth: usize, compress: bool,
) -> Result<()> {
    let rasters = load_raster_stack(&input)?;
    let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();

    let doys = doys_str.as_ref().map(|s| parse_times(s)).transpose()?;

    let params = PhenologyParams {
        threshold,
        smooth_window: if smooth % 2 == 1 { smooth } else { smooth + 1 },
    };

    std::fs::create_dir_all(&outdir)?;
    let start = Instant::now();

    println!("Extracting vegetation phenology ({} time steps, threshold={:.2})...", rasters.len(), threshold);

    let result = vegetation_phenology(&refs, doys.as_deref(), &params)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    let outputs = [
        (&result.sos, "sos", "Start of Season"),
        (&result.eos, "eos", "End of Season"),
        (&result.peak, "peak", "Peak value"),
        (&result.peak_time, "peak_time", "Peak time"),
        (&result.amplitude, "amplitude", "Amplitude"),
        (&result.season_length, "season_length", "Season length"),
    ];

    for (raster, name, desc) in &outputs {
        let path = outdir.join(format!("{}.tif", name));
        write_result(raster, &path, compress)?;
        println!("  {} ({}) → {}", name, desc, path.display());
    }

    done("Phenology", &outdir, start.elapsed());
    Ok(())
}
