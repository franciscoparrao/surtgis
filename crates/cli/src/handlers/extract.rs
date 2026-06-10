//! Handler for extracting raster values at point locations.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

/// Recursively find all .tif files under a directory.
fn find_tifs(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut tifs = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                tifs.extend(find_tifs(&path));
            } else if let Some(ext) = path.extension() {
                if ext.eq_ignore_ascii_case("tif") || ext.eq_ignore_ascii_case("tiff") {
                    tifs.push(path);
                }
            }
        }
    }
    tifs.sort();
    tifs
}

/// Read feature rasters and extract pixel values at point locations.
/// Writes a CSV with one row per point, columns = feature names + target.
///
/// Discovery strategy:
/// 1. If features.json exists, load all rasters listed there (preserving order/names).
/// 2. Scan the directory recursively for .tif files not already listed in features.json.
/// 3. If features.json does not exist, scan all .tif files (file stem = feature name).
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

    let mut feature_names: Vec<String> = Vec::new();
    let mut rasters: Vec<surtgis_core::Raster<f64>> = Vec::new();
    let mut loaded_paths: HashSet<std::path::PathBuf> = HashSet::new();

    // 1. Load rasters from features.json (if it exists)
    let features_json_path = features_dir.join("features.json");
    if features_json_path.exists() {
        let features_json_str = std::fs::read_to_string(&features_json_path)
            .with_context(|| format!("Failed to read {}", features_json_path.display()))?;
        let features_meta: serde_json::Value =
            serde_json::from_str(&features_json_str).context("Failed to parse features.json")?;

        if let Some(entries) = features_meta["features"].as_array() {
            println!("From features.json ({} entries):", entries.len());
            for entry in entries {
                let name = entry["name"]
                    .as_str()
                    .context("Feature entry missing 'name'")?;
                let file = entry["file"]
                    .as_str()
                    .context("Feature entry missing 'file'")?;

                let raster_path = features_dir.join(file);
                if !raster_path.exists() {
                    eprintln!(
                        "  WARNING: skipping missing raster: {}",
                        raster_path.display()
                    );
                    continue;
                }

                let canonical = raster_path
                    .canonicalize()
                    .unwrap_or_else(|_| raster_path.clone());
                let raster = surtgis_core::io::read_geotiff::<f64, _>(&raster_path, None)
                    .with_context(|| format!("Failed to read raster: {}", raster_path.display()))?;
                println!("  Loaded: {} ({}x{})", name, raster.cols(), raster.rows());

                feature_names.push(name.to_string());
                rasters.push(raster);
                loaded_paths.insert(canonical);
            }
        }
    }

    // 2. Scan for additional .tif files not in features.json
    let all_tifs = find_tifs(features_dir);
    let mut extra_count = 0;

    for tif_path in &all_tifs {
        let canonical = tif_path.canonicalize().unwrap_or_else(|_| tif_path.clone());
        if loaded_paths.contains(&canonical) {
            continue;
        }

        // Use file stem as feature name, with subdirectory prefix for disambiguation
        let rel = tif_path.strip_prefix(features_dir).unwrap_or(tif_path);
        let name = rel
            .with_extension("")
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");

        match surtgis_core::io::read_geotiff::<f64, _>(tif_path, None) {
            Ok(raster) => {
                if extra_count == 0 {
                    println!("\nAuto-discovered rasters:");
                }
                println!("  Loaded: {} ({}x{})", name, raster.cols(), raster.rows());
                feature_names.push(name);
                rasters.push(raster);
                loaded_paths.insert(canonical);
                extra_count += 1;
            }
            Err(e) => {
                eprintln!("  WARNING: skipping {}: {}", tif_path.display(), e);
            }
        }
    }

    if extra_count > 0 {
        println!("  {} additional rasters discovered", extra_count);
    }

    if rasters.is_empty() {
        anyhow::bail!("No feature rasters found in {}", features_dir.display());
    }

    println!("\nTotal features: {}", rasters.len());

    // 3. Read vector points
    println!("Reading point locations...");
    let fc =
        surtgis_core::vector::read_vector(points_path).context("Failed to read vector points")?;
    println!("  {} features read", fc.len());

    // 4. Collect point coordinates + target values
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut targets: Vec<String> = Vec::new();
    let mut skipped = 0usize;

    for feature in fc.iter() {
        let Some(geo::Geometry::Point(p)) = &feature.geometry else {
            skipped += 1;
            continue;
        };

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

        points.push((p.x(), p.y()));
        targets.push(target_val);
    }

    // 5. Sample all rasters at the collected points
    println!("Extracting pixel values...");
    let raster_refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
    let samples = surtgis_algorithms::sampling::sample_at_points(&raster_refs, &points)
        .context("Failed to sample rasters at points")?;

    let mut csv_writer = csv::Writer::from_path(output)
        .with_context(|| format!("Failed to create CSV: {}", output.display()))?;

    // Write header: feature_names... + target
    let mut header: Vec<String> = feature_names.clone();
    header.push(target_attr.to_string());
    csv_writer
        .write_record(&header)
        .context("Failed to write CSV header")?;

    let mut extracted = 0usize;

    for (sample, target_val) in samples.into_iter().zip(targets) {
        let Some(values) = sample else {
            skipped += 1;
            continue;
        };

        let mut record: Vec<String> = values.iter().map(|v| format!("{}", v)).collect();
        record.push(target_val);
        csv_writer
            .write_record(&record)
            .context("Failed to write CSV row")?;
        extracted += 1;
    }

    csv_writer.flush().context("Failed to flush CSV")?;

    println!();
    println!("=========================================");
    println!("EXTRACTION COMPLETE");
    println!("=========================================");
    println!("  Extracted: {} samples", extracted);
    println!(
        "  Skipped:   {} (out of bounds, NaN, missing target)",
        skipped
    );
    println!("  Features:  {}", feature_names.len());
    println!("  Output:    {}", output.display());
    println!("  Time:      {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}
