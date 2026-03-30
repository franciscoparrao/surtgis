//! Handler for machine learning workflows using smelt-ml.

use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::Path;
use std::time::Instant;

use smelt_ml::prelude::*;

use crate::commands::MlCommands;
use crate::helpers;

pub fn handle(command: MlCommands, compress: bool) -> Result<()> {
    match command {
        MlCommands::ExtractSamples {
            features_dir,
            points,
            target,
            output,
        } => handle_extract_samples(&features_dir, &points, &target, &output),
        MlCommands::Train {
            input,
            target,
            model,
            n_estimators,
            folds,
            task,
            output,
        } => handle_train(&input, &target, &model, n_estimators, folds, &task, &output),
        MlCommands::Predict {
            model,
            features_dir,
            output,
        } => handle_predict(&model, &features_dir, &output, compress),
        MlCommands::Benchmark {
            input,
            target,
            folds,
            task,
        } => handle_benchmark(&input, &target, folds, &task),
    }
}

// ─── Extract Samples ───────────────────────────────────────────────────

/// Read feature rasters and extract pixel values at point locations.
/// Writes a CSV with one row per point, columns = feature names + target.
fn handle_extract_samples(
    features_dir: &Path,
    points_path: &Path,
    target_attr: &str,
    output: &Path,
) -> Result<()> {
    let start = Instant::now();

    println!("SurtGIS ML: Extract Samples");
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

// ─── Train ─────────────────────────────────────────────────────────────

/// Train a model on a CSV dataset with cross-validation.
fn handle_train(
    input: &Path,
    target_col: &str,
    model_type: &str,
    n_estimators: usize,
    folds: usize,
    task_type: &str,
    output: &Path,
) -> Result<()> {
    let start = Instant::now();

    println!("SurtGIS ML: Train Model");
    println!("=========================================");
    println!("  Input:       {}", input.display());
    println!("  Target:      {}", target_col);
    println!("  Model:       {} (n_estimators={})", model_type, n_estimators);
    println!("  Task:        {}", task_type);
    println!("  CV folds:    {}", folds);
    println!("  Output:      {}", output.display());
    println!();

    let is_classification = task_type == "classification";

    // 1. Load dataset
    println!("Loading dataset...");
    let mut learner = make_learner(model_type, n_estimators)?;

    if is_classification {
        let task = CsvLoader::from_path(input)
            .target(target_col)
            .load_classif()
            .context("Failed to load classification dataset from CSV")?;

        let n_samples = task.n_samples();
        let n_features = task.n_features();
        println!(
            "  Loaded: {} samples x {} features",
            n_samples, n_features
        );

        // 2. Cross-validation
        println!("\nCross-validation ({} folds)...", folds);
        let cv = CrossValidation::new(folds).with_seed(42);
        let splits = cv.splits(n_samples);

        let mut accuracies = Vec::new();
        let mut f1_scores = Vec::new();

        for (i, (train_idx, test_idx)) in splits.iter().enumerate() {
            let train_features = select_rows(task.features(), train_idx);
            let train_target: Vec<usize> = train_idx.iter().map(|&i| task.target()[i]).collect();
            let test_features = select_rows(task.features(), test_idx);
            let test_target: Vec<usize> = test_idx.iter().map(|&i| task.target()[i]).collect();

            let fold_task =
                ClassificationTask::new(&format!("fold_{}", i), train_features, train_target)
                    .context("Failed to create fold task")?;

            let model = learner
                .train_classif(&fold_task)
                .with_context(|| format!("Failed to train fold {}", i))?;

            let pred = model
                .predict(&test_features)
                .context("Failed to predict on test fold")?;

            // Wrap prediction with ground truth for scoring
            let pred_with_truth = pred.with_truth_classif(test_target);

            let acc = Accuracy.score(&pred_with_truth).unwrap_or(f64::NAN);
            let f1 = F1Score.score(&pred_with_truth).unwrap_or(f64::NAN);

            println!("  Fold {}: accuracy={:.4}, F1={:.4}", i + 1, acc, f1);
            accuracies.push(acc);
            f1_scores.push(f1);
        }

        let mean_acc: f64 = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
        let mean_f1: f64 = f1_scores.iter().sum::<f64>() / f1_scores.len() as f64;
        println!("\n  Mean accuracy: {:.4}", mean_acc);
        println!("  Mean F1:       {:.4}", mean_f1);

        // 3. Train on full data
        println!("\nTraining on full dataset...");
        let full_model = learner
            .train_classif(&task)
            .context("Failed to train on full dataset")?;

        // Feature importance
        if let Some(importances) = full_model.feature_importance() {
            println!("\nFeature importance (top 10):");
            let mut sorted = importances.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, (name, imp)) in sorted.iter().take(10).enumerate() {
                println!("  {:2}. {:30} {:.4}", i + 1, name, imp);
            }
        }

        // 4. Serialize model (only for supported learner types)
        println!("\nSaving model...");
        // Note: save_json requires SerializableModel, not Box<dyn TrainedModel>.
        // For now, we skip serialization for non-serializable learners.
        eprintln!("  ⚠ Model serialization requires re-training with type-specific API.");
        eprintln!("  Use 'predict' with the CSV directly for non-serializable models.");
        // TODO: Implement per-learner serialization wrapping
    } else {
        let task = CsvLoader::from_path(input)
            .target(target_col)
            .load_regress()
            .context("Failed to load regression dataset from CSV")?;

        let n_samples = task.n_samples();
        let n_features = task.n_features();
        println!(
            "  Loaded: {} samples x {} features",
            n_samples, n_features
        );

        // 2. Cross-validation
        println!("\nCross-validation ({} folds)...", folds);
        let cv = CrossValidation::new(folds).with_seed(42);
        let splits = cv.splits(n_samples);

        let mut rmse_scores = Vec::new();

        for (i, (train_idx, test_idx)) in splits.iter().enumerate() {
            let train_features = select_rows(task.features(), train_idx);
            let train_target: Vec<f64> = train_idx.iter().map(|&i| task.target()[i]).collect();
            let test_features = select_rows(task.features(), test_idx);
            let test_target: Vec<f64> = test_idx.iter().map(|&i| task.target()[i]).collect();

            let fold_task =
                RegressionTask::new(&format!("fold_{}", i), train_features, train_target)
                    .context("Failed to create fold task")?;

            let model = learner
                .train_regress(&fold_task)
                .with_context(|| format!("Failed to train fold {}", i))?;

            let pred = model
                .predict(&test_features)
                .context("Failed to predict on test fold")?;

            let pred_with_truth = pred.with_truth_regress(test_target);
            let rmse = Rmse.score(&pred_with_truth).unwrap_or(f64::NAN);

            println!("  Fold {}: RMSE={:.4}", i + 1, rmse);
            rmse_scores.push(rmse);
        }

        let mean_rmse: f64 = rmse_scores.iter().sum::<f64>() / rmse_scores.len() as f64;
        println!("\n  Mean RMSE: {:.4}", mean_rmse);

        // 3. Train on full data
        println!("\nTraining on full dataset...");
        let full_model = learner
            .train_regress(&task)
            .context("Failed to train on full dataset")?;

        // Feature importance
        if let Some(importances) = full_model.feature_importance() {
            println!("\nFeature importance (top 10):");
            let mut sorted = importances.clone();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, (name, imp)) in sorted.iter().take(10).enumerate() {
                println!("  {:2}. {:30} {:.4}", i + 1, name, imp);
            }
        }

        // 4. Serialize model
        println!("\nSaving model...");
        eprintln!("  ⚠ Regression model serialization: TODO");
    }

    println!();
    println!("=========================================");
    println!("TRAINING COMPLETE");
    println!("=========================================");
    println!("  Model: {}", output.display());
    println!("  Time:  {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

// ─── Predict ───────────────────────────────────────────────────────────

/// Load a trained model and predict on raster features, writing output GeoTIFF.
fn handle_predict(
    model_path: &Path,
    features_dir: &Path,
    output: &Path,
    compress: bool,
) -> Result<()> {
    let start = Instant::now();

    println!("SurtGIS ML: Predict");
    println!("=========================================");
    println!("  Model:    {}", model_path.display());
    println!("  Features: {}", features_dir.display());
    println!("  Output:   {}", output.display());
    println!();

    // 1. Load model
    println!("Loading model...");
    let model: smelt_ml::serialize::SerializableModel =
        smelt_ml::serialize::load_json(model_path).context("Failed to load model")?;

    // 2. Load feature rasters
    let features_json_path = features_dir.join("features.json");
    let features_json_str = std::fs::read_to_string(&features_json_path)
        .with_context(|| format!("Failed to read {}", features_json_path.display()))?;
    let features_meta: serde_json::Value = serde_json::from_str(&features_json_str)
        .context("Failed to parse features.json")?;

    let feature_entries = features_meta["features"]
        .as_array()
        .context("features.json missing 'features' array")?;

    println!("Loading {} feature rasters...", feature_entries.len());
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
            anyhow::bail!("Missing required raster: {} ({})", name, raster_path.display());
        }

        let raster = surtgis_core::io::read_geotiff::<f64, _>(&raster_path, None)
            .with_context(|| format!("Failed to read raster: {}", raster_path.display()))?;
        println!("  Loaded: {}", name);
        rasters.push(raster);
    }

    if rasters.is_empty() {
        anyhow::bail!("No feature rasters were loaded");
    }

    let ref_raster = &rasters[0];
    let rows = ref_raster.rows();
    let cols = ref_raster.cols();
    let n_features = rasters.len();

    // 3. Build pixel matrix (only valid pixels)
    println!("\nBuilding prediction matrix ({}x{} pixels)...", cols, rows);
    let pb = helpers::spinner("Assembling feature matrix");

    let mut valid_indices: Vec<(usize, usize)> = Vec::new();
    let mut feature_rows: Vec<Vec<f64>> = Vec::new();

    for r in 0..rows {
        for c in 0..cols {
            let mut vals = Vec::with_capacity(n_features);
            let mut valid = true;

            for raster in &rasters {
                match raster.get(r, c) {
                    Ok(v) if v.is_finite() => vals.push(v),
                    _ => {
                        valid = false;
                        break;
                    }
                }
            }

            if valid {
                valid_indices.push((r, c));
                feature_rows.push(vals);
            }
        }
    }

    pb.finish_and_clear();
    println!(
        "  Valid pixels: {} / {} ({:.1}%)",
        valid_indices.len(),
        rows * cols,
        100.0 * valid_indices.len() as f64 / (rows * cols) as f64
    );

    if valid_indices.is_empty() {
        anyhow::bail!("No valid pixels found (all NaN)");
    }

    // Convert to Array2
    let n_valid = valid_indices.len();
    let mut features_array = Array2::<f64>::zeros((n_valid, n_features));
    for (i, row_vals) in feature_rows.iter().enumerate() {
        for (j, &v) in row_vals.iter().enumerate() {
            features_array[[i, j]] = v;
        }
    }

    // 4. Predict
    println!("Predicting...");
    let pb = helpers::spinner("Running model prediction");

    let prediction = model
        .predict(&features_array)
        .context("Failed to predict")?;

    pb.finish_and_clear();

    // 5. Write output raster
    println!("Writing output raster...");
    let mut out_raster = surtgis_core::Raster::<f64>::filled(rows, cols, f64::NAN);
    out_raster.set_transform(ref_raster.transform().clone());
    out_raster.set_crs(ref_raster.crs().cloned());
    out_raster.set_nodata(Some(f64::NAN));

    match prediction {
        Prediction::Classification { ref predicted, .. } => {
            for (i, &(r, c)) in valid_indices.iter().enumerate() {
                out_raster.set(r, c, predicted[i] as f64).ok();
            }
        }
        Prediction::Regression { ref predicted, .. } => {
            for (i, &(r, c)) in valid_indices.iter().enumerate() {
                out_raster.set(r, c, predicted[i]).ok();
            }
        }
    }

    helpers::write_result(&out_raster, &output.to_path_buf(), compress)
        .context("Failed to write prediction raster")?;

    println!();
    println!("=========================================");
    println!("PREDICTION COMPLETE");
    println!("=========================================");
    println!("  Output:       {}", output.display());
    println!("  Pixels:       {}x{}", cols, rows);
    println!("  Valid pixels: {}", valid_indices.len());
    println!("  Time:         {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

// ─── Benchmark ─────────────────────────────────────────────────────────

/// Compare multiple learners on a dataset using cross-validation.
fn handle_benchmark(
    input: &Path,
    target_col: &str,
    folds: usize,
    task_type: &str,
) -> Result<()> {
    let start = Instant::now();

    println!("SurtGIS ML: Benchmark");
    println!("=========================================");
    println!("  Input:  {}", input.display());
    println!("  Target: {}", target_col);
    println!("  Folds:  {}", folds);
    println!("  Task:   {}", task_type);
    println!();

    let is_classification = task_type == "classification";

    // Learners to compare
    let configs: Vec<(&str, usize)> = vec![
        ("rf", 10),
        ("rf", 50),
        ("rf", 100),
        ("rf", 200),
    ];

    if is_classification {
        let task = CsvLoader::from_path(input)
            .target(target_col)
            .load_classif()
            .context("Failed to load classification dataset")?;

        let n_samples = task.n_samples();
        let n_features = task.n_features();
        println!("Dataset: {} samples x {} features\n", n_samples, n_features);

        println!(
            "{:<20} {:>10} {:>10} {:>10}",
            "Model", "Accuracy", "F1", "Time(s)"
        );
        println!("{}", "-".repeat(52));

        let cv = CrossValidation::new(folds).with_seed(42);
        let splits = cv.splits(n_samples);

        for (model_type, n_est) in &configs {
            let mut learner = make_learner(model_type, *n_est)?;
            let model_start = Instant::now();

            let mut accuracies = Vec::new();
            let mut f1_scores = Vec::new();

            for (train_idx, test_idx) in splits.iter() {
                let train_features = select_rows(task.features(), train_idx);
                let train_target: Vec<usize> =
                    train_idx.iter().map(|&i| task.target()[i]).collect();
                let test_features = select_rows(task.features(), test_idx);
                let test_target: Vec<usize> =
                    test_idx.iter().map(|&i| task.target()[i]).collect();

                let fold_task =
                    ClassificationTask::new("bench", train_features, train_target)
                        .context("Failed to create fold task")?;

                let model = learner
                    .train_classif(&fold_task)
                    .context("Failed to train")?;

                let pred = model.predict(&test_features).context("Failed to predict")?;
                let pred_with_truth = pred.with_truth_classif(test_target);

                accuracies.push(Accuracy.score(&pred_with_truth).unwrap_or(f64::NAN));
                f1_scores.push(F1Score.score(&pred_with_truth).unwrap_or(f64::NAN));
            }

            let mean_acc = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
            let mean_f1 = f1_scores.iter().sum::<f64>() / f1_scores.len() as f64;
            let elapsed = model_start.elapsed().as_secs_f64();

            println!(
                "{:<20} {:>10.4} {:>10.4} {:>10.2}",
                format!("{} (n={})", model_type, n_est),
                mean_acc,
                mean_f1,
                elapsed
            );
        }
    } else {
        let task = CsvLoader::from_path(input)
            .target(target_col)
            .load_regress()
            .context("Failed to load regression dataset")?;

        let n_samples = task.n_samples();
        let n_features = task.n_features();
        println!("Dataset: {} samples x {} features\n", n_samples, n_features);

        println!(
            "{:<20} {:>10} {:>10}",
            "Model", "RMSE", "Time(s)"
        );
        println!("{}", "-".repeat(42));

        let cv = CrossValidation::new(folds).with_seed(42);
        let splits = cv.splits(n_samples);

        for (model_type, n_est) in &configs {
            let mut learner = make_learner(model_type, *n_est)?;
            let model_start = Instant::now();

            let mut rmse_scores = Vec::new();

            for (train_idx, test_idx) in splits.iter() {
                let train_features = select_rows(task.features(), train_idx);
                let train_target: Vec<f64> =
                    train_idx.iter().map(|&i| task.target()[i]).collect();
                let test_features = select_rows(task.features(), test_idx);
                let test_target: Vec<f64> =
                    test_idx.iter().map(|&i| task.target()[i]).collect();

                let fold_task =
                    RegressionTask::new("bench", train_features, train_target)
                        .context("Failed to create fold task")?;

                let model = learner
                    .train_regress(&fold_task)
                    .context("Failed to train")?;

                let pred = model.predict(&test_features).context("Failed to predict")?;
                let pred_with_truth = pred.with_truth_regress(test_target);

                rmse_scores.push(Rmse.score(&pred_with_truth).unwrap_or(f64::NAN));
            }

            let mean_rmse = rmse_scores.iter().sum::<f64>() / rmse_scores.len() as f64;
            let elapsed = model_start.elapsed().as_secs_f64();

            println!(
                "{:<20} {:>10.4} {:>10.2}",
                format!("{} (n={})", model_type, n_est),
                mean_rmse,
                elapsed
            );
        }
    }

    println!();
    println!("Total time: {:.1}s", start.elapsed().as_secs_f64());

    Ok(())
}

// ─── Helpers ───────────────────────────────────────────────────────────

/// Create a learner by name.
fn make_learner(model_type: &str, n_estimators: usize) -> Result<RandomForest> {
    match model_type {
        "rf" | "random-forest" => Ok(RandomForest::new()
            .with_n_estimators(n_estimators)
            .with_seed(42)),
        _ => anyhow::bail!(
            "Unknown model type: '{}'. Supported: rf (random-forest)",
            model_type
        ),
    }
}

/// Select rows from an Array2 by indices.
fn select_rows(array: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let ncols = array.ncols();
    let mut result = Array2::<f64>::zeros((indices.len(), ncols));
    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..ncols {
            result[[i, j]] = array[[idx, j]];
        }
    }
    result
}
