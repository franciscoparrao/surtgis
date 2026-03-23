//! Handlers for clip, rasterize, and resample commands.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::helpers;

pub fn handle_clip(input: PathBuf, polygon: PathBuf, output: PathBuf, compress: bool) -> Result<()> {
    let raster = helpers::read_dem(&input)?;
    let features = surtgis_core::vector::read_vector(&polygon)
        .context("Failed to read vector file")?;

    let start = Instant::now();
    let result = surtgis_core::vector::clip_raster(&raster, &features)
        .context("Failed to clip raster")?;
    let elapsed = start.elapsed();

    helpers::write_result(&result, &output, compress)?;

    // Count valid cells
    let total = result.len();
    let valid = result.data().iter().filter(|v| v.is_finite()).count();
    println!(
        "Clipped: {:.1}% of cells retained",
        100.0 * valid as f64 / total as f64
    );

    helpers::done("Clip", &output, elapsed);
    Ok(())
}

pub fn handle_rasterize(
    input: PathBuf,
    output: PathBuf,
    reference: PathBuf,
    attribute: Option<String>,
    compress: bool,
) -> Result<()> {
    let features = surtgis_core::vector::read_vector(&input)
        .context("Failed to read vector file")?;
    let ref_raster = helpers::read_dem(&reference)?;

    let (rows, cols) = ref_raster.shape();
    let start = Instant::now();
    let result = surtgis_core::vector::rasterize_polygons(
        &features,
        ref_raster.transform(),
        rows,
        cols,
        attribute.as_deref(),
    )
    .context("Failed to rasterize")?;
    let elapsed = start.elapsed();

    helpers::write_result(&result, &output, compress)?;
    println!("{} features rasterized", features.len());
    helpers::done("Rasterize", &output, elapsed);
    Ok(())
}

pub fn handle_resample(
    input: PathBuf,
    output: PathBuf,
    reference: PathBuf,
    method: String,
    compress: bool,
) -> Result<()> {
    let method = match method.to_lowercase().as_str() {
        "nearest" | "nn" => surtgis_core::ResampleMethod::NearestNeighbor,
        "bilinear" | "linear" => surtgis_core::ResampleMethod::Bilinear,
        _ => {
            eprintln!("Unknown method '{}', using bilinear", method);
            surtgis_core::ResampleMethod::Bilinear
        }
    };

    let source = helpers::read_dem(&input)?;
    let ref_raster = helpers::read_dem(&reference)?;

    let (src_rows, src_cols) = source.shape();
    let (ref_rows, ref_cols) = ref_raster.shape();

    let start = Instant::now();
    let result = surtgis_core::resample_to_grid(&source, &ref_raster, method)
        .context("Failed to resample")?;
    let elapsed = start.elapsed();

    helpers::write_result(&result, &output, compress)?;
    println!(
        "Resampled: {} x {} → {} x {}",
        src_cols, src_rows, ref_cols, ref_rows
    );
    helpers::done("Resample", &output, elapsed);
    Ok(())
}
