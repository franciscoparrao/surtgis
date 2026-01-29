//! Example: Basic terrain analysis workflow
//!
//! This example demonstrates how to use SurtGis for terrain analysis:
//! 1. Create or load a DEM
//! 2. Calculate slope, aspect, and hillshade
//! 3. Combine results

use surtgis_algorithms::terrain::{aspect, hillshade, slope, HillshadeParams, SlopeParams};
use surtgis_core::{GeoTransform, Raster};

fn main() {
    // Create a synthetic DEM (in real use, load from file)
    let dem = create_synthetic_dem(100, 100);

    println!("DEM created: {} x {}", dem.cols(), dem.rows());
    println!("Cell size: {}", dem.cell_size());

    // Calculate slope
    let slope_result = slope(&dem, SlopeParams::default()).unwrap();
    let slope_stats = slope_result.statistics();
    println!(
        "\nSlope (degrees):\n  Min: {:.2}°\n  Max: {:.2}°\n  Mean: {:.2}°",
        slope_stats.min.unwrap_or(0.0),
        slope_stats.max.unwrap_or(0.0),
        slope_stats.mean.unwrap_or(0.0)
    );

    // Calculate aspect
    let aspect_result = aspect(&dem, Default::default()).unwrap();
    let aspect_stats = aspect_result.statistics();
    println!(
        "\nAspect (degrees):\n  Min: {:.2}°\n  Max: {:.2}°",
        aspect_stats.min.unwrap_or(0.0),
        aspect_stats.max.unwrap_or(0.0),
    );

    // Calculate hillshade
    let hillshade_result = hillshade(&dem, HillshadeParams::default()).unwrap();
    let hs_stats = hillshade_result.statistics();
    println!(
        "\nHillshade (0-255):\n  Min: {:.0}\n  Max: {:.0}\n  Mean: {:.0}",
        hs_stats.min.unwrap_or(0.0),
        hs_stats.max.unwrap_or(0.0),
        hs_stats.mean.unwrap_or(0.0)
    );

    println!("\n✓ Terrain analysis complete!");
}

/// Create a synthetic DEM with interesting terrain features
fn create_synthetic_dem(rows: usize, cols: usize) -> Raster<f64> {
    let mut dem = Raster::new(rows, cols);
    dem.set_transform(GeoTransform::new(0.0, rows as f64, 10.0, -10.0));

    let center_row = rows as f64 / 2.0;
    let center_col = cols as f64 / 2.0;

    for row in 0..rows {
        for col in 0..cols {
            // Create a conical hill in the center
            let dr = row as f64 - center_row;
            let dc = col as f64 - center_col;
            let dist = (dr * dr + dc * dc).sqrt();

            // Base elevation + conical hill + some ridges
            let hill = 500.0 - dist * 5.0;
            let ridge = ((row as f64 * 0.1).sin() * 20.0).max(0.0);
            let valley = ((col as f64 * 0.15).cos() * 15.0).max(0.0);

            let elevation = hill.max(100.0) + ridge + valley;
            dem.set(row, col, elevation).unwrap();
        }
    }

    dem
}
