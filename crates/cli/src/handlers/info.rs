//! Handler for the `info` subcommand.

use anyhow::Result;
use std::path::PathBuf;

use crate::helpers::read_dem;

pub fn handle(input: PathBuf) -> Result<()> {
    let raster = read_dem(&input)?;
    let (rows, cols) = raster.shape();
    let bounds = raster.bounds();
    let stats = raster.statistics();

    println!("File: {}", input.display());
    println!(
        "Dimensions: {} x {} ({} cells)",
        cols,
        rows,
        raster.len()
    );
    println!("Cell size: {}", raster.cell_size());
    println!(
        "Bounds: ({:.6}, {:.6}) - ({:.6}, {:.6})",
        bounds.0, bounds.1, bounds.2, bounds.3
    );
    if let Some(crs) = raster.crs() {
        println!("CRS: {}", crs);
    }
    if let Some(nodata) = raster.nodata() {
        println!("NoData: {}", nodata);
    }
    println!("\nStatistics:");
    if let Some(min) = stats.min {
        println!("  Min: {:.4}", min);
    }
    if let Some(max) = stats.max {
        println!("  Max: {:.4}", max);
    }
    if let Some(mean) = stats.mean {
        println!("  Mean: {:.4}", mean);
    }
    println!(
        "  Valid cells: {} ({:.1}%)",
        stats.valid_count,
        100.0 * stats.valid_count as f64 / raster.len() as f64
    );

    Ok(())
}
