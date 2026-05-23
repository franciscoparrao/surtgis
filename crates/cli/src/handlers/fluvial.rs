//! Handler for `surtgis fluvial <subcmd>` commands.
//!
//! Subcommand surface tracks docs/SPEC_morfometria_fluvial_tectonica.md.
//! Sprint 2 ships `chi` only; ksn / knickpoints / concavity /
//! divide-migration arrive in subsequent sprints.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ndarray::Array2;
use surtgis_algorithms::fluvial::{
    channel_steepness, chi_transform, concavity_index, divide_migration, knickpoint_detection,
    ChiParams, ConcavityParams, ConcavityResult, DivideMigrationParams, DivideSegment, Knickpoint,
    KnickpointParams, KnickpointPolarity, KsnParams, KsnSegment,
};
use surtgis_core::io::{read_geotiff, write_geotiff};

use crate::commands::FluvialCommands;
use crate::helpers::write_opts;

pub fn handle(cmd: FluvialCommands, compress: bool) -> Result<()> {
    match cmd {
        FluvialCommands::Chi {
            stream,
            flow_dir,
            flow_acc,
            output,
            theta_ref,
            a_0_m2,
            cell_size_m,
        } => handle_chi(
            &stream,
            &flow_dir,
            &flow_acc,
            &output,
            theta_ref,
            a_0_m2,
            cell_size_m,
            compress,
        ),
        FluvialCommands::Ksn {
            stream,
            flow_dir,
            flow_acc,
            dem,
            output,
            theta_ref,
            segment_length_m,
            min_drainage_area_m2,
            cell_size_m,
            segments,
        } => handle_ksn(
            &stream,
            &flow_dir,
            &flow_acc,
            &dem,
            &output,
            theta_ref,
            segment_length_m,
            min_drainage_area_m2,
            cell_size_m,
            segments.as_deref(),
            compress,
        ),
        FluvialCommands::Knickpoints {
            stream,
            flow_dir,
            flow_acc,
            dem,
            output,
            theta_ref,
            tvd_lambda,
            curvature_threshold,
            min_magnitude_m,
            confluence_buffer_cells,
            cell_size_m,
            raster,
        } => handle_knickpoints(
            &stream,
            &flow_dir,
            &flow_acc,
            &dem,
            &output,
            theta_ref,
            tvd_lambda,
            curvature_threshold,
            min_magnitude_m,
            confluence_buffer_cells,
            cell_size_m,
            raster.as_deref(),
            compress,
        ),
        FluvialCommands::Concavity {
            stream,
            flow_dir,
            flow_acc,
            dem,
            basins,
            output,
            theta_range,
            theta_step,
            bootstrap_n,
            min_basin_cells,
            seed,
            cell_size_m,
        } => handle_concavity(
            &stream,
            &flow_dir,
            &flow_acc,
            &dem,
            &basins,
            &output,
            &theta_range,
            theta_step,
            bootstrap_n,
            min_basin_cells,
            seed,
            cell_size_m,
        ),
        FluvialCommands::DivideMigration {
            basins,
            dem,
            flow_acc,
            output,
            chi,
            min_divide_length_m,
            cell_size_m,
        } => handle_divide_migration(
            &basins,
            &dem,
            &flow_acc,
            &output,
            chi.as_deref(),
            min_divide_length_m,
            cell_size_m,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_chi(
    stream_path: &Path,
    flow_dir_path: &Path,
    flow_acc_path: &Path,
    output_path: &Path,
    theta_ref: f64,
    a_0_m2: f64,
    cell_size_m_override: Option<f64>,
    compress: bool,
) -> Result<()> {
    let start = std::time::Instant::now();

    println!("SurtGIS Fluvial — χ (chi) transform");
    println!("====================================");
    println!("  Stream:        {}", stream_path.display());
    println!("  Flow dir:      {}", flow_dir_path.display());
    println!("  Flow acc:      {}", flow_acc_path.display());
    println!("  Output:        {}", output_path.display());
    println!("  θref:          {}", theta_ref);
    println!("  A0 (m²):       {:e}", a_0_m2);

    let stream: surtgis_core::Raster<u8> = read_geotiff(stream_path, None)
        .with_context(|| format!("read stream from {}", stream_path.display()))?;
    let flow_dir: surtgis_core::Raster<u8> = read_geotiff(flow_dir_path, None)
        .with_context(|| format!("read flow_dir from {}", flow_dir_path.display()))?;
    let flow_acc: surtgis_core::Raster<f64> = read_geotiff(flow_acc_path, None)
        .with_context(|| format!("read flow_acc from {}", flow_acc_path.display()))?;

    // Cell size resolution. Priority: explicit --cell-size-m > raster
    // transform pixel_width. Per spec §8 pitfall #8, the CRS must be
    // projected (units = metres). We can't introspect the CRS unit string
    // from here without proj4rs heavy lifting, so we use a heuristic: if
    // pixel_width is < 1.0 in absolute value, it's almost certainly
    // degrees → reject unless the user supplied --cell-size-m.
    let pixel_width = stream.transform().pixel_width.abs();
    let cell_size_m = match cell_size_m_override {
        Some(v) if v > 0.0 => v,
        Some(v) => return Err(anyhow!("--cell-size-m must be > 0, got {}", v)),
        None => {
            if pixel_width < 1.0 {
                return Err(anyhow!(
                    "Raster pixel size is {} (looks like degrees, not metres). \
                     Reproject your inputs to a projected CRS first \
                     (e.g. `surtgis reproject --to EPSG:32719 ...`) \
                     or pass `--cell-size-m <metres>` explicitly.",
                    pixel_width,
                ));
            }
            pixel_width
        }
    };
    println!("  Cell size (m): {:.2}", cell_size_m);

    // Cell-size > 30 m generates the warning from spec §8 pitfall #1: ksn
    // / chi sensitivity drops. Chi is less sensitive than ksn but we
    // surface the warning for consistency with the spec.
    if cell_size_m > 30.0 {
        eprintln!(
            "  WARNING: cell size {:.1} m > 30 m — χ and downstream ksn \
             will be noisier than the literature standard (~10–30 m).",
            cell_size_m,
        );
    }

    let params = ChiParams {
        theta_ref,
        a_0_m2,
        cell_size_m,
        base_outlets: None,
    };

    let chi = chi_transform(&stream, &flow_dir, &flow_acc, params)
        .context("χ transform failed")?;

    // Quick stats so the user knows the run produced something sensible.
    let stats = chi.statistics();
    println!();
    println!("  χ stats:");
    if let (Some(mn), Some(mx), Some(mean)) = (stats.min, stats.max, stats.mean) {
        println!("    min   {:.2}", mn);
        println!("    max   {:.2}", mx);
        println!("    mean  {:.2}", mean);
    } else {
        println!("    (all NaN — no stream cells reached an outlet?)");
    }
    println!("    valid {} / {} cells ({:.1}%)",
        stats.valid_count,
        chi.len(),
        100.0 * stats.valid_count as f64 / chi.len() as f64,
    );

    write_geotiff(&chi, output_path, Some(write_opts(compress)))
        .with_context(|| format!("write chi to {}", output_path.display()))?;

    println!();
    println!("Wrote {} in {:.1}s",
        output_path.display(),
        start.elapsed().as_secs_f64(),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_ksn(
    stream_path: &Path,
    flow_dir_path: &Path,
    flow_acc_path: &Path,
    dem_path: &Path,
    output_path: &Path,
    theta_ref: f64,
    segment_length_m: f64,
    min_drainage_area_m2: f64,
    cell_size_m_override: Option<f64>,
    segments_path: Option<&Path>,
    compress: bool,
) -> Result<()> {
    let start = std::time::Instant::now();

    println!("SurtGIS Fluvial — ksn (channel steepness)");
    println!("=========================================");
    println!("  Stream:        {}", stream_path.display());
    println!("  Flow dir:      {}", flow_dir_path.display());
    println!("  Flow acc:      {}", flow_acc_path.display());
    println!("  DEM:           {}", dem_path.display());
    println!("  Output:        {}", output_path.display());
    println!("  θref:          {}", theta_ref);
    println!("  Segment len:   {} m", segment_length_m);
    println!("  Min A:         {:e} m²", min_drainage_area_m2);

    let stream: surtgis_core::Raster<u8> = read_geotiff(stream_path, None)
        .with_context(|| format!("read stream from {}", stream_path.display()))?;
    let flow_dir: surtgis_core::Raster<u8> = read_geotiff(flow_dir_path, None)
        .with_context(|| format!("read flow_dir from {}", flow_dir_path.display()))?;
    let flow_acc: surtgis_core::Raster<f64> = read_geotiff(flow_acc_path, None)
        .with_context(|| format!("read flow_acc from {}", flow_acc_path.display()))?;
    let dem: surtgis_core::Raster<f64> = read_geotiff(dem_path, None)
        .with_context(|| format!("read dem from {}", dem_path.display()))?;

    // Cell-size resolution. Same heuristic as chi: refuse degree-like
    // pixel sizes (< 1.0) unless the user supplies --cell-size-m. Spec
    // §8 pitfall #8 — fluvial algorithms require a projected CRS.
    let pixel_width = stream.transform().pixel_width.abs();
    let cell_size_m = match cell_size_m_override {
        Some(v) if v > 0.0 => v,
        Some(v) => return Err(anyhow!("--cell-size-m must be > 0, got {}", v)),
        None => {
            if pixel_width < 1.0 {
                return Err(anyhow!(
                    "Raster pixel size is {} (looks like degrees, not metres). \
                     Reproject your inputs to a projected CRS first \
                     (e.g. `surtgis reproject --to EPSG:32719 ...`) \
                     or pass `--cell-size-m <metres>` explicitly.",
                    pixel_width,
                ));
            }
            pixel_width
        }
    };
    println!("  Cell size (m): {:.2}", cell_size_m);
    if cell_size_m > 30.0 {
        eprintln!(
            "  WARNING: cell size {:.1} m > 30 m — ksn is noisier than the \
             literature standard (~10–30 m). Consider a finer DEM.",
            cell_size_m,
        );
    }

    let params = KsnParams {
        theta_ref,
        segment_length_m,
        cell_size_m,
        min_drainage_area_m2,
    };
    let emit_segments = segments_path.is_some();

    let result = channel_steepness(&stream, &flow_dir, &flow_acc, &dem, params, emit_segments)
        .context("ksn computation failed")?;

    let stats = result.ksn_raster.statistics();
    println!();
    println!("  ksn stats:");
    if let (Some(mn), Some(mx), Some(mean)) = (stats.min, stats.max, stats.mean) {
        println!("    min   {:.4}", mn);
        println!("    max   {:.4}", mx);
        println!("    mean  {:.4}", mean);
    } else {
        println!("    (all NaN — no qualifying stream cells?)");
    }
    println!(
        "    valid {} / {} cells ({:.1}%)",
        stats.valid_count,
        result.ksn_raster.len(),
        100.0 * stats.valid_count as f64 / result.ksn_raster.len() as f64,
    );

    write_geotiff(&result.ksn_raster, output_path, Some(write_opts(compress)))
        .with_context(|| format!("write ksn to {}", output_path.display()))?;

    if let Some(p) = segments_path {
        let segs = result.segments.as_deref().unwrap_or(&[]);
        write_segments_geojson(p, segs).with_context(|| {
            format!("write ksn segments to {}", p.display())
        })?;
        println!("  Segments:    {} features → {}", segs.len(), p.display());
    }

    println!();
    println!(
        "Wrote {} in {:.1}s",
        output_path.display(),
        start.elapsed().as_secs_f64(),
    );
    Ok(())
}

/// Write a list of [`KsnSegment`] as a GeoJSON LineString
/// FeatureCollection. Coordinates are the cell-centre `(x, y)` values
/// in the source raster's CRS (NOT WGS84 — callers needing lon/lat must
/// reproject downstream). Each feature carries `ksn_mean` and `n_cells`.
fn write_segments_geojson(path: &Path, segments: &[KsnSegment]) -> Result<()> {
    let features: Vec<serde_json::Value> = segments
        .iter()
        .filter(|s| s.coordinates.len() >= 2)
        .map(|s| {
            let coords: Vec<[f64; 2]> = s.coordinates.iter().map(|&(x, y)| [x, y]).collect();
            serde_json::json!({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "ksn_mean": if s.ksn_mean.is_finite() {
                        serde_json::json!(s.ksn_mean)
                    } else {
                        serde_json::Value::Null
                    },
                    "n_cells": s.n_cells,
                },
            })
        })
        .collect();
    let fc = serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
    });
    std::fs::write(path, serde_json::to_string_pretty(&fc)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_concavity(
    stream_path: &Path,
    flow_dir_path: &Path,
    flow_acc_path: &Path,
    dem_path: &Path,
    basins_path: &Path,
    output_path: &Path,
    theta_range_str: &str,
    theta_step: f64,
    bootstrap_n: usize,
    min_basin_cells: usize,
    seed: u64,
    cell_size_m_override: Option<f64>,
) -> Result<()> {
    let start = std::time::Instant::now();

    // Parse "low,high" → (f64, f64).
    let parts: Vec<&str> = theta_range_str.split(',').collect();
    if parts.len() != 2 {
        return Err(anyhow!(
            "--theta-range must be `low,high` (e.g. \"0.1,0.9\"); got {:?}",
            theta_range_str
        ));
    }
    let lo: f64 = parts[0].trim().parse().context("parse --theta-range low")?;
    let hi: f64 = parts[1].trim().parse().context("parse --theta-range high")?;

    println!("SurtGIS Fluvial — concavity index (per basin)");
    println!("==============================================");
    println!("  Stream:        {}", stream_path.display());
    println!("  Flow dir:      {}", flow_dir_path.display());
    println!("  Flow acc:      {}", flow_acc_path.display());
    println!("  DEM:           {}", dem_path.display());
    println!("  Basins:        {}", basins_path.display());
    println!("  Output CSV:    {}", output_path.display());
    println!("  θ range:       [{}, {}] step {}", lo, hi, theta_step);
    println!("  Bootstrap n:   {}", bootstrap_n);
    println!("  Min cells:     {}", min_basin_cells);

    let stream: surtgis_core::Raster<u8> = read_geotiff(stream_path, None)
        .with_context(|| format!("read stream from {}", stream_path.display()))?;
    let flow_dir: surtgis_core::Raster<u8> = read_geotiff(flow_dir_path, None)
        .with_context(|| format!("read flow_dir from {}", flow_dir_path.display()))?;
    let flow_acc: surtgis_core::Raster<f64> = read_geotiff(flow_acc_path, None)
        .with_context(|| format!("read flow_acc from {}", flow_acc_path.display()))?;
    let dem: surtgis_core::Raster<f64> = read_geotiff(dem_path, None)
        .with_context(|| format!("read dem from {}", dem_path.display()))?;
    let basins: surtgis_core::Raster<i32> = read_geotiff(basins_path, None)
        .with_context(|| format!("read basins from {}", basins_path.display()))?;

    let pixel_width = stream.transform().pixel_width.abs();
    let cell_size_m = match cell_size_m_override {
        Some(v) if v > 0.0 => v,
        Some(v) => return Err(anyhow!("--cell-size-m must be > 0, got {}", v)),
        None => {
            if pixel_width < 1.0 {
                return Err(anyhow!(
                    "Raster pixel size is {} (looks like degrees, not metres). \
                     Reproject your inputs to a projected CRS first \
                     or pass `--cell-size-m <metres>` explicitly.",
                    pixel_width,
                ));
            }
            pixel_width
        }
    };
    println!("  Cell size (m): {:.2}", cell_size_m);

    let params = ConcavityParams {
        theta_range: (lo, hi),
        theta_step,
        bootstrap_n,
        cell_size_m,
        seed,
        min_basin_cells,
    };

    let results = concavity_index(&stream, &flow_dir, &flow_acc, &dem, &basins, params)
        .context("concavity_index failed")?;

    write_concavity_csv(output_path, &results)
        .with_context(|| format!("write concavity csv to {}", output_path.display()))?;

    println!();
    println!("  {} basins reported", results.len());
    if !results.is_empty() {
        let mean_t: f64 =
            results.iter().map(|r| r.theta_opt).sum::<f64>() / results.len() as f64;
        let min_t = results
            .iter()
            .map(|r| r.theta_opt)
            .fold(f64::INFINITY, f64::min);
        let max_t = results
            .iter()
            .map(|r| r.theta_opt)
            .fold(f64::NEG_INFINITY, f64::max);
        println!("  θ_opt range:   [{:.3}, {:.3}], mean {:.3}", min_t, max_t, mean_t);
    }

    println!();
    println!(
        "Wrote {} in {:.1}s",
        output_path.display(),
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn write_concavity_csv(path: &Path, results: &[ConcavityResult]) -> Result<()> {
    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "basin_id",
        "theta_opt",
        "theta_ci_low",
        "theta_ci_high",
        "n_cells",
        "rmse",
    ])?;
    for r in results {
        w.write_record([
            r.basin_id.to_string(),
            format!("{:.4}", r.theta_opt),
            format!("{:.4}", r.theta_ci.0),
            format!("{:.4}", r.theta_ci.1),
            r.n_cells.to_string(),
            format!("{:.6}", r.rmse),
        ])?;
    }
    w.flush()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_knickpoints(
    stream_path: &Path,
    flow_dir_path: &Path,
    flow_acc_path: &Path,
    dem_path: &Path,
    output_path: &Path,
    theta_ref: f64,
    tvd_lambda: f64,
    curvature_threshold: f64,
    min_magnitude_m: f64,
    confluence_buffer_cells: usize,
    cell_size_m_override: Option<f64>,
    raster_path: Option<&Path>,
    compress: bool,
) -> Result<()> {
    let start = std::time::Instant::now();

    println!("SurtGIS Fluvial — knickpoint detection");
    println!("======================================");
    println!("  Stream:        {}", stream_path.display());
    println!("  Flow dir:      {}", flow_dir_path.display());
    println!("  Flow acc:      {}", flow_acc_path.display());
    println!("  DEM:           {}", dem_path.display());
    println!("  Output GeoJSON:{}", output_path.display());
    println!("  θref:          {}", theta_ref);
    println!("  TVD λ:         {}", tvd_lambda);
    println!("  |κ| threshold: {}", curvature_threshold);
    println!("  min magnitude: {} m", min_magnitude_m);
    println!("  confluence buf:{} cells", confluence_buffer_cells);

    let stream: surtgis_core::Raster<u8> = read_geotiff(stream_path, None)
        .with_context(|| format!("read stream from {}", stream_path.display()))?;
    let flow_dir: surtgis_core::Raster<u8> = read_geotiff(flow_dir_path, None)
        .with_context(|| format!("read flow_dir from {}", flow_dir_path.display()))?;
    let flow_acc: surtgis_core::Raster<f64> = read_geotiff(flow_acc_path, None)
        .with_context(|| format!("read flow_acc from {}", flow_acc_path.display()))?;
    let dem: surtgis_core::Raster<f64> = read_geotiff(dem_path, None)
        .with_context(|| format!("read dem from {}", dem_path.display()))?;

    let pixel_width = stream.transform().pixel_width.abs();
    let cell_size_m = match cell_size_m_override {
        Some(v) if v > 0.0 => v,
        Some(v) => return Err(anyhow!("--cell-size-m must be > 0, got {}", v)),
        None => {
            if pixel_width < 1.0 {
                return Err(anyhow!(
                    "Raster pixel size is {} (looks like degrees, not metres). \
                     Reproject your inputs to a projected CRS first \
                     or pass `--cell-size-m <metres>` explicitly.",
                    pixel_width,
                ));
            }
            pixel_width
        }
    };
    println!("  Cell size (m): {:.2}", cell_size_m);
    if cell_size_m > 30.0 {
        eprintln!(
            "  WARNING: cell size {:.1} m > 30 m — knickpoint detection \
             is noisier than the literature standard (~10–30 m).",
            cell_size_m,
        );
    }

    let params = KnickpointParams {
        theta_ref,
        tvd_lambda,
        curvature_threshold,
        min_magnitude_m,
        cell_size_m,
        confluence_buffer_cells,
    };

    let knicks = knickpoint_detection(&stream, &flow_dir, &flow_acc, &dem, params)
        .context("knickpoint detection failed")?;

    let n_concave = knicks
        .iter()
        .filter(|k| k.polarity == KnickpointPolarity::Concave)
        .count();
    let n_convex = knicks.len() - n_concave;
    println!();
    println!(
        "  {} knickpoints detected ({} concave, {} convex)",
        knicks.len(),
        n_concave,
        n_convex,
    );

    write_knickpoints_geojson(output_path, &knicks, &stream).with_context(|| {
        format!("write knickpoints geojson to {}", output_path.display())
    })?;

    if let Some(rp) = raster_path {
        let (rows, cols) = stream.shape();
        let mut data = Array2::<u8>::zeros((rows, cols));
        for k in &knicks {
            data[(k.row, k.col)] = match k.polarity {
                KnickpointPolarity::Concave => 1,
                KnickpointPolarity::Convex => 2,
            };
        }
        let mut out = stream.with_same_meta::<u8>(rows, cols);
        out.set_nodata(Some(0));
        *out.data_mut() = data;
        write_geotiff(&out, rp, Some(write_opts(compress)))
            .with_context(|| format!("write knickpoint raster to {}", rp.display()))?;
        println!("  Raster:        {}", rp.display());
    }

    println!();
    println!(
        "Wrote {} in {:.1}s",
        output_path.display(),
        start.elapsed().as_secs_f64(),
    );
    Ok(())
}

/// Write knickpoints as a GeoJSON Point FeatureCollection. Coordinates
/// are cell-centre `(x, y)` in the source raster's CRS. Each feature
/// carries `elevation_m`, `magnitude_m`, `chi`, and `polarity`
/// (string: "concave" or "convex").
fn write_knickpoints_geojson(
    path: &Path,
    knicks: &[Knickpoint],
    stream: &surtgis_core::Raster<u8>,
) -> Result<()> {
    let gt = stream.transform();
    let features: Vec<serde_json::Value> = knicks
        .iter()
        .map(|k| {
            let (x0, y0) = stream.pixel_to_geo(k.col, k.row);
            let x = x0 + 0.5 * gt.pixel_width;
            let y = y0 + 0.5 * gt.pixel_height;
            let pol_str = match k.polarity {
                KnickpointPolarity::Concave => "concave",
                KnickpointPolarity::Convex => "convex",
            };
            serde_json::json!({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [x, y],
                },
                "properties": {
                    "elevation_m": k.elevation_m,
                    "magnitude_m": k.magnitude_m,
                    "chi": k.chi,
                    "polarity": pol_str,
                },
            })
        })
        .collect();
    let fc = serde_json::json!({
        "type": "FeatureCollection",
        "features": features,
    });
    std::fs::write(path, serde_json::to_string_pretty(&fc)?)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_divide_migration(
    basins_path: &Path,
    dem_path: &Path,
    flow_acc_path: &Path,
    output_path: &Path,
    chi_path: Option<&Path>,
    min_divide_length_m: f64,
    cell_size_m_override: Option<f64>,
) -> Result<()> {
    let start = std::time::Instant::now();

    println!("SurtGIS Fluvial — divide migration");
    println!("==================================");
    println!("  Basins:        {}", basins_path.display());
    println!("  DEM:           {}", dem_path.display());
    println!("  Flow acc:      {}", flow_acc_path.display());
    if let Some(p) = chi_path {
        println!("  χ raster:      {}", p.display());
    } else {
        println!("  χ raster:      (none — median_chi_diff will be NaN)");
    }
    println!("  Output GeoJSON:{}", output_path.display());
    println!("  Min length:    {} m", min_divide_length_m);

    let basins: surtgis_core::Raster<i32> = read_geotiff(basins_path, None)
        .with_context(|| format!("read basins from {}", basins_path.display()))?;
    let dem: surtgis_core::Raster<f64> = read_geotiff(dem_path, None)
        .with_context(|| format!("read dem from {}", dem_path.display()))?;
    let flow_acc: surtgis_core::Raster<f64> = read_geotiff(flow_acc_path, None)
        .with_context(|| format!("read flow_acc from {}", flow_acc_path.display()))?;
    let chi: Option<surtgis_core::Raster<f64>> = match chi_path {
        Some(p) => Some(
            read_geotiff(p, None).with_context(|| format!("read chi from {}", p.display()))?,
        ),
        None => None,
    };

    let pixel_width = basins.transform().pixel_width.abs();
    let cell_size_m = match cell_size_m_override {
        Some(v) if v > 0.0 => v,
        Some(v) => return Err(anyhow!("--cell-size-m must be > 0, got {}", v)),
        None => {
            if pixel_width < 1.0 {
                return Err(anyhow!(
                    "Raster pixel size is {} (looks like degrees, not metres). \
                     Reproject your inputs to a projected CRS first \
                     or pass `--cell-size-m <metres>` explicitly.",
                    pixel_width,
                ));
            }
            pixel_width
        }
    };
    println!("  Cell size (m): {:.2}", cell_size_m);

    let params = DivideMigrationParams {
        cell_size_m,
        min_divide_length_m,
    };

    let result = divide_migration(&basins, &dem, chi.as_ref(), &flow_acc, params)
        .context("divide_migration failed")?;

    write_divide_geojson(output_path, &result)
        .with_context(|| format!("write divides geojson to {}", output_path.display()))?;

    println!();
    println!("  {} divides reported", result.len());
    if !result.is_empty() {
        let n_with_chi = result.iter().filter(|d| d.median_chi_diff.is_finite()).count();
        println!("  with finite χ diff: {}", n_with_chi);
        let max_elev = result
            .iter()
            .map(|d| d.median_elev_diff.abs())
            .fold(0.0_f64, f64::max);
        println!("  max |median Δelev|: {:.2} m", max_elev);
    }

    println!();
    println!("Wrote {} in {:.1}s", output_path.display(), start.elapsed().as_secs_f64());
    Ok(())
}

fn write_divide_geojson(path: &Path, divides: &[DivideSegment]) -> Result<()> {
    let features: Vec<serde_json::Value> = divides
        .iter()
        .filter(|d| d.coordinates.len() >= 2)
        .map(|d| {
            let coords: Vec<[f64; 2]> = d.coordinates.iter().map(|&(x, y)| [x, y]).collect();
            serde_json::json!({
                "type": "Feature",
                "geometry": { "type": "LineString", "coordinates": coords },
                "properties": {
                    "basin_a": d.basin_a,
                    "basin_b": d.basin_b,
                    "median_chi_diff": if d.median_chi_diff.is_finite() {
                        serde_json::json!(d.median_chi_diff)
                    } else { serde_json::Value::Null },
                    "median_elev_diff": if d.median_elev_diff.is_finite() {
                        serde_json::json!(d.median_elev_diff)
                    } else { serde_json::Value::Null },
                    "median_relief_diff": if d.median_relief_diff.is_finite() {
                        serde_json::json!(d.median_relief_diff)
                    } else { serde_json::Value::Null },
                    "n_pairs": d.n_pairs,
                },
            })
        })
        .collect();
    let fc = serde_json::json!({ "type": "FeatureCollection", "features": features });
    std::fs::write(path, serde_json::to_string_pretty(&fc)?)?;
    Ok(())
}
