//! Handler for extracting raster values at point locations.

use anyhow::{Context, Result};
use std::path::Path;
use std::time::Instant;

/// Read feature rasters and extract pixel values at point locations.
/// Writes a CSV with one row per point, columns = feature names + target.
pub fn handle(
    features_dir: &Path,
    points_path: &Path,
    target_attr: &str,
    output: &Path,
) -> Result<()> {
    let start = Instant::now();

    println!("SurtGIS Extract");
    println!("=========================================");
    println!("  Features: {}", features_dir.display());
    println!("  Points:   {}", points_path.display());
    println!("  Target:   {}", target_attr);
    println!("  Output:   {}", output.display());
    println!();

    // 1. Read features.json
    let features_json_path = features_dir.join("features.json");
    let features_json_str = std::fs::read_to_string(&features_json_path)
        .with_context(|| format!("Failed to read {}", features_json_path.display()))?;
    let features_meta: serde_json::Value = serde_json::from_str(&features_json_str)
        .context("Failed to parse features.json")?;

    let feature_entries = features_meta["features"]
        .as_array()
        .context("features.json missing 'features' array")?;

    // 2. Load all feature rasters
    println!("Loading {} feature rasters...", feature_entries.len());
    let mut feature_names: Vec<String> = Vec::new();
    let mut rasters: Vec<surtgis_core::Raster<f64>> = Vec::new();

    for entry in feature_entries {
        let name = entry["name"]
            .as_str()
            .context("Feature entry missing 'name'")?;
        let file = entry["file"]
            .as_str()
            .context("Feature entry missing 'file'")?;

        let raster_path = features_dir.join(file);
        if !raster_path.exists() {
            eprintln!("  WARNING: skipping missing raster: {}", raster_path.display());
            continue;
        }

        let raster = surtgis_core::io::read_geotiff::<f64, _>(&raster_path, None)
            .with_context(|| format!("Failed to read raster: {}", raster_path.display()))?;
        println!("  Loaded: {} ({}x{})", name, raster.cols(), raster.rows());

        feature_names.push(name.to_string());
        rasters.push(raster);
    }

    if rasters.is_empty() {
        anyhow::bail!("No feature rasters were loaded");
    }

    // Use the first raster as reference for coordinate transform
    let ref_raster = &rasters[0];

    // 3. Read vector points
    println!("\nReading point locations...");
    let fc = surtgis_core::vector::read_vector(points_path)
        .context("Failed to read vector points")?;
    println!("  {} features read", fc.len());

    // 4. Extract pixel values at each point
    println!("Extracting pixel values...");

    let mut csv_writer = csv::Writer::from_path(output)
        .with_context(|| format!("Failed to create CSV: {}", output.display()))?;

    // Write header: feature_names... + target
    let mut header: Vec<String> = feature_names.clone();
    header.push(target_attr.to_string());
    csv_writer
        .write_record(&header)
        .context("Failed to write CSV header")?;

    let mut extracted = 0usize;
    let mut skipped = 0usize;

    for feature in fc.iter() {
        // Get point geometry
        let geom = match &feature.geometry {
            Some(g) => g,
            None => {
                skipped += 1;
                continue;
            }
        };

        // Extract point coordinates
        let (x, y) = match geom {
            geo::Geometry::Point(p) => (p.x(), p.y()),
            _ => {
                skipped += 1;
                continue;
            }
        };

        // Convert geographic to pixel coordinates
        let (col_f, row_f) = ref_raster.geo_to_pixel(x, y);
        let col = col_f.floor() as isize;
        let row = row_f.floor() as isize;

        // Check bounds
        if row < 0
            || col < 0
            || row as usize >= ref_raster.rows()
            || col as usize >= ref_raster.cols()
        {
            skipped += 1;
            continue;
        }

        let row = row as usize;
        let col = col as usize;

        // Extract values from all rasters
        let mut values: Vec<String> = Vec::with_capacity(rasters.len() + 1);
        let mut has_nan = false;

        for raster in &rasters {
            match raster.get(row, col) {
                Ok(v) if v.is_finite() => values.push(format!("{}", v)),
                _ => {
                    has_nan = true;
                    break;
                }
            }
        }

        if has_nan {
            skipped += 1;
            continue;
        }

        // Extract target value from feature properties
        let target_val = match feature.get_property(target_attr) {
            Some(surtgis_core::vector::AttributeValue::Int(v)) => format!("{}", v),
            Some(surtgis_core::vector::AttributeValue::Float(v)) => format!("{}", v),
            Some(surtgis_core::vector::AttributeValue::String(v)) => v.clone(),
            Some(surtgis_core::vector::AttributeValue::Bool(v)) => {
                format!("{}", if *v { 1 } else { 0 })
            }
            _ => {
                skipped += 1;
                continue;
            }
        };

        values.push(target_val);
        csv_writer
            .write_record(&values)
            .context("Failed to write CSV row")?;
        extracted += 1;
    }

    csv_writer.flush().context("Failed to flush CSV")?;

    println!();
    println!("=========================================");
    println!("EXTRACTION COMPLETE");
    println!("=========================================");
    println!("  Extracted: {} samples", extracted);
    println!("  Skipped:   {} (out of bounds, NaN, missing target)", skipped);
    println!("  Features:  {}", feature_names.len());
    println!("  Output:    {}", output.display());
    println!("  Time:      {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}
