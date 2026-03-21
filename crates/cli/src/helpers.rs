//! Shared utility functions for the CLI: I/O, parsing, progress display.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use surtgis_algorithms::imagery::{BandMathOp, ReclassEntry};
use surtgis_algorithms::landscape::Connectivity;
use surtgis_algorithms::morphology::StructuringElement;
use surtgis_algorithms::terrain::AdvancedCurvatureType;
use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};

#[cfg(feature = "cloud")]
use surtgis_cloud::BBox;

pub fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
}

pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

pub fn read_dem(path: &PathBuf) -> Result<surtgis_core::Raster<f64>> {
    let pb = spinner("Reading raster...");
    let raster: surtgis_core::Raster<f64> =
        read_geotiff(path, None).context("Failed to read raster")?;
    pb.finish_and_clear();
    info!("Input: {} x {}", raster.cols(), raster.rows());
    Ok(raster)
}

pub fn read_u8(path: &PathBuf) -> Result<surtgis_core::Raster<u8>> {
    let pb = spinner("Reading raster...");
    let raster: surtgis_core::Raster<u8> =
        read_geotiff(path, None).context("Failed to read raster")?;
    pb.finish_and_clear();
    Ok(raster)
}

pub fn write_opts(compress: bool) -> GeoTiffOptions {
    GeoTiffOptions {
        compression: if compress {
            "deflate".to_string()
        } else {
            "NONE".to_string()
        },
    }
}

pub fn write_result(raster: &surtgis_core::Raster<f64>, path: &PathBuf, compress: bool) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress)))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

pub fn write_result_u8(
    raster: &surtgis_core::Raster<u8>,
    path: &PathBuf,
    compress: bool,
) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress)))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

pub fn write_result_i32(
    raster: &surtgis_core::Raster<i32>,
    path: &PathBuf,
    compress: bool,
) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress)))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

pub fn done(name: &str, path: &std::path::Path, elapsed: std::time::Duration) {
    println!("{} saved to: {}", name, path.display());
    println!("  Processing time: {:.2?}", elapsed);
}

pub fn parse_se(shape: &str, radius: usize) -> Result<StructuringElement> {
    let se = match shape.to_lowercase().as_str() {
        "square" | "sq" => StructuringElement::Square(radius),
        "cross" | "cr" => StructuringElement::Cross(radius),
        "disk" | "circle" => StructuringElement::Disk(radius),
        _ => anyhow::bail!("Unknown shape: {}. Use square, cross, or disk.", shape),
    };
    se.validate()
        .map_err(|e| anyhow::anyhow!("Invalid structuring element: {}", e))?;
    Ok(se)
}

pub fn parse_connectivity(c: u8) -> Result<Connectivity> {
    match c {
        4 => Ok(Connectivity::Four),
        8 => Ok(Connectivity::Eight),
        _ => anyhow::bail!("Connectivity must be 4 or 8, got: {}", c),
    }
}

pub fn parse_band_math_op(s: &str) -> Result<BandMathOp> {
    match s.to_lowercase().as_str() {
        "add" | "+" => Ok(BandMathOp::Add),
        "subtract" | "sub" | "-" => Ok(BandMathOp::Subtract),
        "multiply" | "mul" | "*" => Ok(BandMathOp::Multiply),
        "divide" | "div" | "/" => Ok(BandMathOp::Divide),
        "power" | "pow" | "^" => Ok(BandMathOp::Power),
        "min" => Ok(BandMathOp::Min),
        "max" => Ok(BandMathOp::Max),
        _ => anyhow::bail!(
            "Unknown operation: {}. Use add, subtract, multiply, divide, power, min, max.",
            s
        ),
    }
}

pub fn parse_band_assignments(bands: &[String]) -> Result<Vec<(String, PathBuf)>> {
    bands
        .iter()
        .map(|s| {
            let parts: Vec<&str> = s.splitn(2, '=').collect();
            if parts.len() != 2 {
                anyhow::bail!("Band must be NAME=path, got: {}", s);
            }
            Ok((parts[0].to_string(), PathBuf::from(parts[1])))
        })
        .collect()
}

pub fn parse_reclass_entry(s: &str) -> Result<ReclassEntry> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        anyhow::bail!("Class must be 'min,max,value', got: {}", s);
    }
    let min: f64 = parts[0].trim().parse().context("Invalid min")?;
    let max: f64 = parts[1].trim().parse().context("Invalid max")?;
    let value: f64 = parts[2].trim().parse().context("Invalid value")?;
    Ok(ReclassEntry { min, max, value })
}

pub fn parse_scl_classes(s: &str) -> Result<Vec<u8>> {
    s.split(',')
        .map(|c| {
            c.trim()
                .parse::<u8>()
                .with_context(|| format!("Invalid SCL class: {}", c))
        })
        .collect()
}

pub fn parse_pour_points(s: &str) -> Result<Vec<(usize, usize)>> {
    s.split(';')
        .map(|pair| {
            let parts: Vec<&str> = pair.trim().split(',').collect();
            if parts.len() != 2 {
                anyhow::bail!("Pour point must be 'row,col', got: {}", pair);
            }
            let row: usize = parts[0].trim().parse().context("Invalid row")?;
            let col: usize = parts[1].trim().parse().context("Invalid col")?;
            Ok((row, col))
        })
        .collect()
}

pub fn parse_advanced_curvature_type(s: &str) -> Result<AdvancedCurvatureType> {
    match s.to_lowercase().as_str() {
        "mean_h" | "mean" | "h" => Ok(AdvancedCurvatureType::MeanH),
        "gaussian_k" | "gaussian" | "k" => Ok(AdvancedCurvatureType::GaussianK),
        "kmin" | "minimal" => Ok(AdvancedCurvatureType::MinimalKmin),
        "kmax" | "maximal" => Ok(AdvancedCurvatureType::MaximalKmax),
        "kh" | "horizontal" => Ok(AdvancedCurvatureType::HorizontalKh),
        "kv" | "vertical" => Ok(AdvancedCurvatureType::VerticalKv),
        "khe" | "horizontal_excess" => Ok(AdvancedCurvatureType::HorizontalExcessKhe),
        "kve" | "vertical_excess" => Ok(AdvancedCurvatureType::VerticalExcessKve),
        "ka" | "accumulation" => Ok(AdvancedCurvatureType::AccumulationKa),
        "kr" | "ring" => Ok(AdvancedCurvatureType::RingKr),
        "rotor" => Ok(AdvancedCurvatureType::Rotor),
        "laplacian" => Ok(AdvancedCurvatureType::Laplacian),
        "unsphericity" | "m" => Ok(AdvancedCurvatureType::UnsphericitytM),
        "difference" | "e" => Ok(AdvancedCurvatureType::DifferenceE),
        _ => anyhow::bail!(
            "Unknown curvature type: {}. Use mean_h, gaussian_k, kmin, kmax, kh, kv, khe, kve, ka, kr, rotor, laplacian, unsphericity, difference.",
            s
        ),
    }
}

#[cfg(feature = "cloud")]
pub fn parse_bbox(s: &str) -> Result<BBox> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        anyhow::bail!(
            "Bbox must be min_x,min_y,max_x,max_y (got {} parts)",
            parts.len()
        );
    }
    let min_x: f64 = parts[0].trim().parse().context("Invalid min_x")?;
    let min_y: f64 = parts[1].trim().parse().context("Invalid min_y")?;
    let max_x: f64 = parts[2].trim().parse().context("Invalid max_x")?;
    let max_y: f64 = parts[3].trim().parse().context("Invalid max_y")?;
    Ok(BBox::new(min_x, min_y, max_x, max_y))
}
