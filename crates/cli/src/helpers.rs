//! Shared utility functions for the CLI: I/O, parsing, progress display.

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use tracing::{Level, info};
use tracing_subscriber::FmtSubscriber;

use surtgis_algorithms::imagery::{BandMathOp, ReclassEntry};
use surtgis_algorithms::landscape::Connectivity;
use surtgis_algorithms::morphology::StructuringElement;
use surtgis_algorithms::terrain::AdvancedCurvatureType;
use surtgis_core::io::{GeoTiffOptions, read_geotiff, read_geotiff_bands, write_geotiff};

#[cfg(feature = "cloud")]
use surtgis_cloud::BBox;

pub fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
}

pub fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

pub fn read_dem(path: &PathBuf) -> Result<surtgis_core::Raster<f64>> {
    let pb = spinner("Reading raster...");
    let raster: surtgis_core::Raster<f64> =
        read_geotiff(path, None).context("Failed to read raster")?;
    pb.finish_and_clear();
    info!("Input: {} x {}", raster.cols(), raster.rows());
    Ok(raster)
}

pub fn read_u8(path: &PathBuf) -> Result<surtgis_core::Raster<u8>> {
    let pb = spinner("Reading raster...");
    let raster: surtgis_core::Raster<u8> =
        read_geotiff(path, None).context("Failed to read raster")?;
    pb.finish_and_clear();
    Ok(raster)
}

/// Read a feature raster as one or more named features. A single-band
/// file is the feature `name`; a multi-band file (an embedding stack such
/// as AlphaEarth's 64 bands, or any band stack) becomes `name:b1`,
/// `name:b2`, … one feature per band, in band order.
pub fn read_feature_bands(
    path: &std::path::Path,
    name: &str,
) -> Result<Vec<(String, surtgis_core::Raster<f64>)>> {
    let bands: Vec<surtgis_core::Raster<f64>> = read_geotiff_bands(path)
        .with_context(|| format!("Failed to read raster: {}", path.display()))?;
    match bands.len() {
        0 => anyhow::bail!("{}: no bands", path.display()),
        1 => Ok(vec![(name.to_string(), bands.into_iter().next().unwrap())]),
        _ => Ok(bands
            .into_iter()
            .enumerate()
            .map(|(i, b)| (format!("{name}:b{}", i + 1), b))
            .collect()),
    }
}

/// Process one or more aligned GeoTIFFs strip by strip and stream the
/// result to `output` as Float32 (nodata NaN): the inputs are never loaded
/// whole. Every input must share the first one's grid (size and
/// geotransform); band 0 of each is used. `f` receives, per strip, the
/// strip index and one `Raster<f64>` per input (nodata already NaN,
/// georeferenced to the strip) and returns the strip's values.
///
/// Strips are sized so that one input strip is about 64 MB of f64, so
/// memory is roughly `(n_inputs + 1) × 64 MB` whatever the raster height —
/// the fix for `imagery calc` at 293 M cells (issues_surtgis.md #8).
pub fn stream_aligned<F>(
    inputs: &[std::path::PathBuf],
    output: &std::path::Path,
    compress: bool,
    label: &str,
    mut f: F,
) -> Result<(usize, usize, usize)>
where
    F: FnMut(usize, &[surtgis_core::Raster<f64>]) -> Result<ndarray::Array2<f64>>,
{
    use surtgis_core::io::window::{PixelWindow, geotiff_info, read_geotiff_window_bands};
    use surtgis_core::io::{StripWriterConfig, write_geotiff_streaming};
    if inputs.is_empty() {
        anyhow::bail!("no input rasters");
    }
    let infos: Vec<_> = inputs
        .iter()
        .map(|p| geotiff_info(p).with_context(|| format!("Failed to read {}", p.display())))
        .collect::<Result<_>>()?;
    let first = &infos[0];
    let (rows, cols) = (first.height as usize, first.width as usize);
    for (p, i) in inputs.iter().zip(&infos).skip(1) {
        let same_grid = i.width == first.width
            && i.height == first.height
            && (i.transform.origin_x - first.transform.origin_x).abs() < 1e-6
            && (i.transform.origin_y - first.transform.origin_y).abs() < 1e-6
            && (i.transform.pixel_width - first.transform.pixel_width).abs() < 1e-9
            && (i.transform.pixel_height - first.transform.pixel_height).abs() < 1e-9;
        if !same_grid {
            anyhow::bail!(
                "{} is not on the same grid as {} ({}x{} vs {}x{}); resample it first",
                p.display(),
                inputs[0].display(),
                i.width,
                i.height,
                first.width,
                first.height
            );
        }
    }
    let rows_per_strip = std::env::var("SURTGIS_STRIP_ROWS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| {
            // ~64 MB of f64 across all inputs per strip; the expression
            // evaluator allocates temporaries of the same size, so peak RSS
            // stays a small multiple of this regardless of raster size.
            ((64usize << 20) / (inputs.len() * cols.max(1) * 8)).clamp(16, 8192)
        })
        .min(rows.max(1));
    let config = StripWriterConfig {
        rows,
        cols,
        transform: first.transform,
        crs: first.crs.clone(),
        nodata: Some(f64::NAN),
        compress,
        rows_per_strip: rows_per_strip as u32,
    };
    let pb = spinner(&format!("{label} (streaming)..."));
    write_geotiff_streaming(output, &config, |strip_idx, strip_rows| {
        let start = strip_idx * rows_per_strip;
        let pw = PixelWindow {
            col: 0,
            row: start as u32,
            width: cols as u32,
            height: strip_rows as u32,
        };
        let mut strips = Vec::with_capacity(inputs.len());
        for (p, info) in inputs.iter().zip(&infos) {
            let band = read_geotiff_window_bands::<f64, _>(p, info, 0, &pw)?
                .into_iter()
                .next()
                .ok_or_else(|| surtgis_core::Error::Other(format!("{}: no bands", p.display())))?;
            strips.push(band);
        }
        f(strip_idx, &strips).map_err(|e| surtgis_core::Error::Other(e.to_string()))
    })
    .with_context(|| format!("{label} failed"))?;
    pb.finish_and_clear();
    Ok((rows, cols, rows.div_ceil(rows_per_strip)))
}

pub fn write_opts(compress: bool) -> GeoTiffOptions {
    // GeoTiffOptions is shared by both I/O backends; only `compression` is
    // relevant here, so start from Default and override just that field.
    GeoTiffOptions {
        compression: if compress {
            "deflate".to_string()
        } else {
            "NONE".to_string()
        },
        ..Default::default()
    }
}

/// Sample type of Float64 outputs, set once from `--output-dtype`.
static OUTPUT_F32: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Make every Float64 output written through [`write_result`] Float32.
pub fn set_output_f32(on: bool) {
    OUTPUT_F32.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn write_result(
    raster: &surtgis_core::Raster<f64>,
    path: &PathBuf,
    compress: bool,
) -> Result<()> {
    let pb = spinner("Writing output...");
    if OUTPUT_F32.load(std::sync::atomic::Ordering::Relaxed) {
        // Half the file: nodata sentinels cast with the data (NaN stays NaN).
        let (rows, cols) = raster.shape();
        let mut out = raster.with_same_meta::<f32>(rows, cols);
        *out.data_mut() = raster.data().mapv(|v| v as f32);
        out.set_nodata(raster.nodata().map(|nd| nd as f32));
        write_geotiff(&out, path, Some(write_opts(compress))).context("Failed to write output")?;
    } else {
        write_geotiff(raster, path, Some(write_opts(compress)))
            .context("Failed to write output")?;
    }
    pb.finish_and_clear();
    Ok(())
}

pub fn write_result_u8(
    raster: &surtgis_core::Raster<u8>,
    path: &PathBuf,
    compress: bool,
) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress))).context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

pub fn write_result_i32(
    raster: &surtgis_core::Raster<i32>,
    path: &PathBuf,
    compress: bool,
) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress))).context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

pub fn done(name: &str, path: &std::path::Path, elapsed: std::time::Duration) {
    println!("{} saved to: {}", name, path.display());
    println!("  Processing time: {:.2?}", elapsed);
}

pub fn parse_se(shape: &str, radius: usize) -> Result<StructuringElement> {
    let se = match shape.to_lowercase().as_str() {
        "square" | "sq" => StructuringElement::Square(radius),
        "cross" | "cr" => StructuringElement::Cross(radius),
        "disk" | "circle" => StructuringElement::Disk(radius),
        _ => anyhow::bail!("Unknown shape: {}. Use square, cross, or disk.", shape),
    };
    se.validate()
        .map_err(|e| anyhow::anyhow!("Invalid structuring element: {}", e))?;
    Ok(se)
}

pub fn parse_connectivity(c: u8) -> Result<Connectivity> {
    match c {
        4 => Ok(Connectivity::Four),
        8 => Ok(Connectivity::Eight),
        _ => anyhow::bail!("Connectivity must be 4 or 8, got: {}", c),
    }
}

pub fn parse_band_math_op(s: &str) -> Result<BandMathOp> {
    match s.to_lowercase().as_str() {
        "add" | "+" => Ok(BandMathOp::Add),
        "subtract" | "sub" | "-" => Ok(BandMathOp::Subtract),
        "multiply" | "mul" | "*" => Ok(BandMathOp::Multiply),
        "divide" | "div" | "/" => Ok(BandMathOp::Divide),
        "power" | "pow" | "^" => Ok(BandMathOp::Power),
        "min" => Ok(BandMathOp::Min),
        "max" => Ok(BandMathOp::Max),
        _ => anyhow::bail!(
            "Unknown operation: {}. Use add, subtract, multiply, divide, power, min, max.",
            s
        ),
    }
}

pub fn parse_band_assignments(bands: &[String]) -> Result<Vec<(String, PathBuf)>> {
    bands
        .iter()
        .map(|s| {
            let parts: Vec<&str> = s.splitn(2, '=').collect();
            if parts.len() != 2 {
                anyhow::bail!("Band must be NAME=path, got: {}", s);
            }
            Ok((parts[0].to_string(), PathBuf::from(parts[1])))
        })
        .collect()
}

pub fn parse_reclass_entry(s: &str) -> Result<ReclassEntry> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        anyhow::bail!("Class must be 'min,max,value', got: {}", s);
    }
    let min: f64 = parts[0].trim().parse().context("Invalid min")?;
    let max: f64 = parts[1].trim().parse().context("Invalid max")?;
    let value: f64 = parts[2].trim().parse().context("Invalid value")?;
    Ok(ReclassEntry { min, max, value })
}

pub fn parse_scl_classes(s: &str) -> Result<Vec<u8>> {
    s.split(',')
        .map(|c| {
            c.trim()
                .parse::<u8>()
                .with_context(|| format!("Invalid SCL class: {}", c))
        })
        .collect()
}

pub fn parse_pour_points(s: &str) -> Result<Vec<(usize, usize)>> {
    s.split(';')
        .map(|pair| {
            let parts: Vec<&str> = pair.trim().split(',').collect();
            if parts.len() != 2 {
                anyhow::bail!("Pour point must be 'row,col', got: {}", pair);
            }
            let row: usize = parts[0].trim().parse().context("Invalid row")?;
            let col: usize = parts[1].trim().parse().context("Invalid col")?;
            Ok((row, col))
        })
        .collect()
}

pub fn parse_advanced_curvature_type(s: &str) -> Result<AdvancedCurvatureType> {
    match s.to_lowercase().as_str() {
        "mean_h" | "mean" | "h" => Ok(AdvancedCurvatureType::MeanH),
        "gaussian_k" | "gaussian" | "k" => Ok(AdvancedCurvatureType::GaussianK),
        "kmin" | "minimal" => Ok(AdvancedCurvatureType::MinimalKmin),
        "kmax" | "maximal" => Ok(AdvancedCurvatureType::MaximalKmax),
        "kh" | "horizontal" => Ok(AdvancedCurvatureType::HorizontalKh),
        "kv" | "vertical" => Ok(AdvancedCurvatureType::VerticalKv),
        "khe" | "horizontal_excess" => Ok(AdvancedCurvatureType::HorizontalExcessKhe),
        "kve" | "vertical_excess" => Ok(AdvancedCurvatureType::VerticalExcessKve),
        "ka" | "accumulation" => Ok(AdvancedCurvatureType::AccumulationKa),
        "kr" | "ring" => Ok(AdvancedCurvatureType::RingKr),
        "rotor" => Ok(AdvancedCurvatureType::Rotor),
        "laplacian" => Ok(AdvancedCurvatureType::Laplacian),
        "unsphericity" | "m" => Ok(AdvancedCurvatureType::UnsphericitytM),
        "difference" | "e" => Ok(AdvancedCurvatureType::DifferenceE),
        _ => anyhow::bail!(
            "Unknown curvature type: {}. Use mean_h, gaussian_k, kmin, kmax, kh, kv, khe, kve, ka, kr, rotor, laplacian, unsphericity, difference.",
            s
        ),
    }
}

#[cfg(feature = "cloud")]
pub fn parse_bbox(s: &str) -> Result<BBox> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        anyhow::bail!(
            "Bbox must be min_x,min_y,max_x,max_y (got {} parts)",
            parts.len()
        );
    }
    let min_x: f64 = parts[0].trim().parse().context("Invalid min_x")?;
    let min_y: f64 = parts[1].trim().parse().context("Invalid min_y")?;
    let max_x: f64 = parts[2].trim().parse().context("Invalid max_x")?;
    let max_y: f64 = parts[3].trim().parse().context("Invalid max_y")?;
    Ok(BBox::new(min_x, min_y, max_x, max_y))
}
