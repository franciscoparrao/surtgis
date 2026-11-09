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
    if let Ok(Some(p)) = surtgis_core::io::read_provenance(&input) {
        println!(
            "Provenance: {} {} on {}, {} input(s) — `surtgis provenance` for details",
            p.engine.name,
            p.engine.version,
            p.created,
            p.inputs.len()
        );
    }
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
        // Floats are normalised to NaN in memory; say what the file declares
        // as well, so "NoData: NaN" is not mistaken for "no nodata tag".
        let declared = surtgis_core::io::window::geotiff_info(&input)
            .ok()
            .and_then(|i| i.nodata);
        match (r.nodata(), declared) {
            (Some(nd), Some(tag)) if nd.to_f64().is_some_and(|v| v.is_nan()) && !tag.is_nan() => {
                println!("NoData: {tag} (file tag; NaN in memory)");
            }
            (Some(nd), _) => println!("NoData: {}", nd),
            (None, Some(tag)) => println!("NoData: {tag} (file tag)"),
            (None, None) => {}
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
