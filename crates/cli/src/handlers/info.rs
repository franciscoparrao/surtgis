//! Handler for the `info` subcommand.

use anyhow::{Context, Result};
use std::path::PathBuf;
use surtgis_core::RasterElement;
use surtgis_core::dispatch_any;

pub fn handle(input: PathBuf) -> Result<()> {
    // `info` is read-only, display-only — a good low-risk fit for
    // `read_geotiff_any`: no need to force the file to `f64` (4x a u16
    // DEM's memory) just to print its shape/bounds/stats. `dispatch_any!`
    // runs the same body against whichever concrete `Raster<T>` the file
    // actually decoded to.
    let any = surtgis_core::io::read_geotiff_any(&input, None).context("Failed to read raster")?;
    let dtype = any.dtype();
    let (rows, cols) = any.shape();

    println!("File: {}", input.display());
    println!("Data type: {}", dtype);
    println!("Dimensions: {} x {} ({} cells)", cols, rows, rows * cols);
    dispatch_any!(&any, r => {
        println!("Cell size: {}", r.cell_size());
        let bounds = r.bounds();
        println!(
            "Bounds: ({:.6}, {:.6}) - ({:.6}, {:.6})",
            bounds.0, bounds.1, bounds.2, bounds.3
        );
        if let Some(crs) = r.crs() {
            println!("CRS: {}", crs);
        }
        if let Some(nodata) = r.nodata() {
            println!("NoData: {}", nodata);
        }
        let stats = r.statistics();
        println!("\nStatistics:");
        if let Some(min) = stats.min {
            println!("  Min: {:.4}", min.to_f64().unwrap_or(f64::NAN));
        }
        if let Some(max) = stats.max {
            println!("  Max: {:.4}", max.to_f64().unwrap_or(f64::NAN));
        }
        if let Some(mean) = stats.mean {
            println!("  Mean: {:.4}", mean);
        }
        println!(
            "  Valid cells: {} ({:.1}%)",
            stats.valid_count,
            100.0 * stats.valid_count as f64 / (rows * cols) as f64
        );
    });

    Ok(())
}
