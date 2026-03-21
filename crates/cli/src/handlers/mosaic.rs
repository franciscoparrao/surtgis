//! Handler for the mosaic subcommand.

use anyhow::{Context, Result};
use std::path::PathBuf;
use std::time::Instant;

use crate::helpers::{done, read_dem, write_result};

pub fn handle(input: Vec<PathBuf>, output: PathBuf, compress: bool) -> Result<()> {
    if input.len() < 2 {
        anyhow::bail!("mosaic requires at least 2 input rasters");
    }
    let rasters: Vec<surtgis_core::Raster<f64>> = input
        .iter()
        .map(|p| read_dem(p))
        .collect::<Result<Vec<_>>>()?;
    let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
    let start = Instant::now();
    let result = surtgis_core::mosaic(&refs, None)
        .context("Failed to mosaic rasters")?;
    let elapsed = start.elapsed();
    let (rows, cols) = result.shape();
    write_result(&result, &output, compress)?;
    println!(
        "Mosaic: {} tiles -> {} x {} ({} cells)",
        input.len(),
        cols,
        rows,
        result.len()
    );
    done("Mosaic", &output, elapsed);

    Ok(())
}
