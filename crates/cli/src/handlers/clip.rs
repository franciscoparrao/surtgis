//! Handlers for clip, rasterize, and resample commands.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use ndarray;

use crate::helpers;
use crate::memory;

pub fn handle_clip(
    input: PathBuf,
    polygon: Option<PathBuf>,
    bbox: Option<String>,
    output: PathBuf,
    compress: bool,
    mem_limit_bytes: Option<u64>,
) -> Result<()> {
    if polygon.is_none() && bbox.is_none() {
        anyhow::bail!("Either --polygon or --bbox is required");
    }

    // Validate that DEM doesn't exceed memory limit (clip requires in-memory processing)
    if let Some(limit) = mem_limit_bytes {
        if let Ok(est_size) = memory::estimate_decompressed_size(&input) {
            if est_size > limit {
                eprintln!(
                    "Warning: DEM size ({:.2} GB) exceeds --max-memory limit ({:.2} GB)",
                    est_size as f64 / 1e9,
                    limit as f64 / 1e9
                );
                eprintln!("Note: Clip operation requires in-memory processing and cannot stream");
            }
        }
    }

    let raster = helpers::read_dem(&input)?;
    let start = Instant::now();

    let result = if let Some(polygon_path) = polygon {
        let features = surtgis_core::vector::read_vector(&polygon_path)
            .context("Failed to read vector file")?;
        surtgis_core::vector::clip_raster(&raster, &features).context("Failed to clip raster")?
    } else {
        // Parse bbox: xmin,ymin,xmax,ymax
        let bbox_str = bbox.unwrap();
        let parts: Vec<f64> = bbox_str
            .split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Invalid bbox format. Expected: xmin,ymin,xmax,ymax")?;
        if parts.len() != 4 {
            anyhow::bail!(
                "bbox must have exactly 4 values: xmin,ymin,xmax,ymax (got {})",
                parts.len()
            );
        }
        let (xmin, ymin, xmax, ymax) = (parts[0], parts[1], parts[2], parts[3]);

        // Convert geographic bounds to pixel coordinates
        let (col_min_f, row_max_f) = raster.geo_to_pixel(xmin, ymin);
        let (col_max_f, row_min_f) = raster.geo_to_pixel(xmax, ymax);

        let row_start = (row_min_f.floor() as isize).max(0) as usize;
        let row_end = (row_max_f.ceil() as isize).max(0) as usize;
        let col_start = (col_min_f.floor() as isize).max(0) as usize;
        let col_end = (col_max_f.ceil() as isize).max(0) as usize;

        let (rows, cols) = raster.shape();
        let row_end = row_end.min(rows);
        let col_end = col_end.min(cols);

        if row_start >= row_end || col_start >= col_end {
            anyhow::bail!("Bounding box does not overlap with raster");
        }

        let out_rows = row_end - row_start;
        let out_cols = col_end - col_start;
        let data = raster.data();
        let sub = data.slice(ndarray::s![row_start..row_end, col_start..col_end]);

        let (ox, oy) = raster.pixel_to_geo(col_start, row_start);
        let gt = raster.transform();
        let new_gt = surtgis_core::GeoTransform::new(ox, oy, gt.pixel_width, gt.pixel_height);

        let mut out =
            surtgis_core::Raster::from_vec(sub.iter().copied().collect(), out_rows, out_cols)?;
        out.set_transform(new_gt);
        out.set_crs(raster.crs().cloned());
        out.set_nodata(raster.nodata());
        out
    };

    let elapsed = start.elapsed();
    helpers::write_result(&result, &output, compress)?;

    let total = result.len();
    let valid = result.data().iter().filter(|v| v.is_finite()).count();
    println!(
        "Clipped: {} x {} ({:.1}% valid cells)",
        result.cols(),
        result.rows(),
        100.0 * valid as f64 / total as f64,
    );

    helpers::done("Clip", &output, elapsed);
    Ok(())
}

/// Rows per strip of the streaming rasterizer: a 20k-column grid is
/// ~80 MB of f64 per strip, whatever the grid's height.
const RASTERIZE_STRIP_ROWS: u32 = 512;

pub fn handle_rasterize(
    input: PathBuf,
    output: PathBuf,
    reference: PathBuf,
    attribute: Option<String>,
    compress: bool,
) -> Result<()> {
    use surtgis_core::io::window::geotiff_info;
    use surtgis_core::io::{StripWriterConfig, write_geotiff_streaming};

    let features =
        surtgis_core::vector::read_vector(&input).context("Failed to read vector file")?;
    // Only the reference's grid is needed (transform, size, CRS): never its
    // pixels. Reading a 293 M-cell reference as f64 alone was 2.3 GB.
    let info = geotiff_info(&reference)
        .with_context(|| format!("Failed to read reference {}", reference.display()))?;
    let (rows, cols) = (info.height as usize, info.width as usize);
    let gt = info.transform;
    let crs = info.crs.clone();

    // CRS check up front (rasterize_polygons repeats it per strip).
    if let (Some(v), Some(r)) = (features.crs(), crs.as_ref())
        && !v.is_equivalent(r)
    {
        anyhow::bail!(
            "vector CRS ({}) does not match raster CRS ({}); reproject the vector data before rasterizing",
            v.identifier(),
            r.identifier()
        );
    }

    let start = Instant::now();
    // Streaming: the output is written strip by strip (Float32, NaN outside
    // every polygon), so memory is one strip, not the whole grid. Each strip
    // is rasterized on its own sub-grid (same columns, shifted origin).
    let config = StripWriterConfig {
        rows,
        cols,
        transform: gt,
        crs: crs.clone(),
        nodata: Some(f64::NAN),
        compress,
        rows_per_strip: RASTERIZE_STRIP_ROWS,
    };
    let rps = RASTERIZE_STRIP_ROWS as usize;
    let pb = helpers::spinner("Rasterizing...");
    write_geotiff_streaming(&output, &config, |strip_idx, strip_rows| {
        let start_row = strip_idx * rps;
        let (ox, oy) = gt.pixel_to_geo_corner(0, start_row);
        let strip_gt = surtgis_core::GeoTransform::new(ox, oy, gt.pixel_width, gt.pixel_height);
        let strip = surtgis_core::vector::rasterize_polygons(
            &features,
            &strip_gt,
            strip_rows,
            cols,
            attribute.as_deref(),
            crs.as_ref(),
        )?;
        Ok(strip.data().to_owned())
    })
    .context("Failed to rasterize")?;
    pb.finish_and_clear();
    let elapsed = start.elapsed();

    println!(
        "{} features rasterized ({}x{} grid, {} strips)",
        features.len(),
        cols,
        rows,
        rows.div_ceil(rps)
    );
    helpers::done("Rasterize", &output, elapsed);
    Ok(())
}

pub fn handle_resample(
    input: PathBuf,
    output: PathBuf,
    reference: PathBuf,
    method: String,
    compress: bool,
) -> Result<()> {
    let method = match method.to_lowercase().as_str() {
        "nearest" | "nn" => surtgis_core::ResampleMethod::NearestNeighbor,
        "bilinear" | "linear" => surtgis_core::ResampleMethod::Bilinear,
        _ => {
            eprintln!("Unknown method '{}', using bilinear", method);
            surtgis_core::ResampleMethod::Bilinear
        }
    };

    let source = helpers::read_dem(&input)?;
    let ref_raster = helpers::read_dem(&reference)?;

    let (src_rows, src_cols) = source.shape();
    let (ref_rows, ref_cols) = ref_raster.shape();

    let start = Instant::now();
    let result = surtgis_core::resample_to_grid(&source, &ref_raster, method)
        .context("Failed to resample")?;
    let elapsed = start.elapsed();

    helpers::write_result(&result, &output, compress)?;
    println!(
        "Resampled: {} x {} → {} x {}",
        src_cols, src_rows, ref_cols, ref_rows
    );
    helpers::done("Resample", &output, elapsed);
    Ok(())
}
