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
    keep_extent: bool,
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
        let masked = surtgis_core::vector::clip_raster(&raster, &features)
            .context("Failed to clip raster")?;
        if keep_extent {
            masked
        } else {
            // Crop to the polygons' extent, as gdalwarp -crop_to_cutline
            // and rasterio.mask(crop=True) do.
            use geo::BoundingRect;
            let rect = features
                .iter()
                .filter_map(|f| f.geometry.as_ref().and_then(|g| g.bounding_rect()))
                .reduce(|a, b| {
                    geo::Rect::new(
                        geo::coord! { x: a.min().x.min(b.min().x), y: a.min().y.min(b.min().y) },
                        geo::coord! { x: a.max().x.max(b.max().x), y: a.max().y.max(b.max().y) },
                    )
                })
                .ok_or_else(|| anyhow::anyhow!("the vector file has no geometry to clip by"))?;
            crop_to_bounds(
                &masked,
                rect.min().x,
                rect.min().y,
                rect.max().x,
                rect.max().y,
            )
            .context("The polygons do not overlap the raster (check that both use the same CRS)")?
        }
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
        crop_to_bounds(&raster, parts[0], parts[1], parts[2], parts[3])?
    };

    let valid_cells = result.data().iter().filter(|v| v.is_finite()).count();
    if valid_cells == 0 {
        anyhow::bail!(
            "the clip leaves no valid cell (the polygon or bbox misses the raster's data; \
             check that both use the same CRS). Nothing was written"
        );
    }

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

/// Crop `raster` to the cells intersecting `xmin,ymin,xmax,ymax` (map
/// units of the raster's CRS), snapped outwards to the grid.
fn crop_to_bounds(
    raster: &surtgis_core::Raster<f64>,
    xmin: f64,
    ymin: f64,
    xmax: f64,
    ymax: f64,
) -> Result<surtgis_core::Raster<f64>> {
    let (col_a, row_a) = raster.geo_to_pixel(xmin, ymin);
    let (col_b, row_b) = raster.geo_to_pixel(xmax, ymax);
    let (rows, cols) = raster.shape();
    let clamp = |v: f64, hi: usize| (v.max(0.0) as usize).min(hi);
    let row_start = clamp(row_a.min(row_b).floor(), rows);
    let row_end = clamp(row_a.max(row_b).ceil(), rows);
    let col_start = clamp(col_a.min(col_b).floor(), cols);
    let col_end = clamp(col_a.max(col_b).ceil(), cols);
    if row_start >= row_end || col_start >= col_end {
        anyhow::bail!("Bounding box does not overlap with raster");
    }
    let (out_rows, out_cols) = (row_end - row_start, col_end - col_start);
    let sub = raster
        .data()
        .slice(ndarray::s![row_start..row_end, col_start..col_end]);
    // Corner, not centre: `pixel_to_geo` returns the cell centre, which
    // shifted every cropped raster by half a cell (v1.5.2 and earlier).
    let gt = raster.transform();
    let (ox, oy) = gt.pixel_to_geo_corner(col_start, row_start);
    let mut out =
        surtgis_core::Raster::from_vec(sub.iter().copied().collect(), out_rows, out_cols)?;
    out.set_transform(surtgis_core::GeoTransform::new(
        ox,
        oy,
        gt.pixel_width,
        gt.pixel_height,
    ));
    out.set_crs(raster.crs().cloned());
    out.set_nodata(raster.nodata());
    Ok(out)
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

    use surtgis_core::io::window::{geotiff_info, read_geotiff_window_bands};
    use surtgis_core::io::{StripWriterConfig, write_geotiff_streaming};

    // Neither raster is loaded whole: the reference gives the grid, the
    // source is read window by window (issues_surtgis.md #7 — reading the
    // whole source was ~12 B/cell, 3.4 GB for a 293 M-cell LOCC).
    let src_info =
        geotiff_info(&input).with_context(|| format!("Failed to read {}", input.display()))?;
    let ref_info = geotiff_info(&reference)
        .with_context(|| format!("Failed to read reference {}", reference.display()))?;
    let (src_rows, src_cols) = (src_info.height as usize, src_info.width as usize);
    let (ref_rows, ref_cols) = (ref_info.height as usize, ref_info.width as usize);
    let ref_gt = ref_info.transform;

    // CRS check: sampling the source with reference coordinates of another
    // CRS reads the wrong window (or none) and silently yields an all-NaN
    // raster. Differing EPSG codes are reprojected on the fly.
    let reproject = resample_crs_plan(&input, &reference, &src_info.crs, &ref_info.crs)?;
    let src_px = src_info
        .transform
        .pixel_width
        .abs()
        .max(src_info.transform.pixel_height.abs());
    let ref_px = ref_gt.pixel_width.abs().max(ref_gt.pixel_height.abs());

    // Strips sized so that the source window behind each one is about 512
    // source rows: downsampling 10 m → 121 m gives ~42 output rows per strip.
    // (Pixel sizes in different CRSs are not comparable: fixed 256 rows.)
    let rows_per_strip = if reproject.is_some() {
        256.min(ref_rows.max(1))
    } else {
        ((512.0 * src_px / ref_px).floor() as usize).clamp(1, 4096)
    };
    #[cfg(feature = "projections")]
    let transformer = match reproject {
        Some((s_epsg, r_epsg)) => Some(
            surtgis_core::warp::Transformer::new(s_epsg, r_epsg)
                .context("Failed to set up the reprojection")?,
        ),
        None => None,
    };
    let warp_method = match method {
        surtgis_core::ResampleMethod::NearestNeighbor => "nearest",
        _ => "bilinear",
    };
    let valid_cells = std::sync::atomic::AtomicUsize::new(0);
    let margin = 2.0 * src_px.max(ref_px);

    let start = Instant::now();
    let config = StripWriterConfig {
        rows: ref_rows,
        cols: ref_cols,
        transform: ref_gt,
        crs: ref_info.crs.clone(),
        nodata: Some(f64::NAN),
        compress,
        rows_per_strip: rows_per_strip as u32,
    };
    let pb = helpers::spinner("Resampling...");
    write_geotiff_streaming(&output, &config, |strip_idx, strip_rows| {
        let start_row = strip_idx * rows_per_strip;
        let (ox, oy) = ref_gt.pixel_to_geo_corner(0, start_row);
        let strip_gt =
            surtgis_core::GeoTransform::new(ox, oy, ref_gt.pixel_width, ref_gt.pixel_height);
        let count = |a: ndarray::Array2<f64>| {
            let n = a.iter().filter(|v| v.is_finite()).count();
            valid_cells.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
            a
        };
        #[cfg(feature = "projections")]
        if let (Some(tf), Some((_, r_epsg))) = (transformer.as_ref(), reproject) {
            use surtgis_core::warp;
            let grid = warp::GridSpec {
                transform: strip_gt,
                rows: strip_rows,
                cols: ref_cols,
                epsg: r_epsg,
            };
            let nan = || ndarray::Array2::from_elem((strip_rows, ref_cols), f64::NAN);
            let Ok(sb) = warp::source_window(&grid, tf, 2) else {
                return Ok(nan());
            };
            let Some(pw) = src_info.window_for_bounds(0, sb.min_x, sb.min_y, sb.max_x, sb.max_y)
            else {
                return Ok(nan());
            };
            let win = read_geotiff_window_bands::<f64, _>(&input, &src_info, 0, &pw)?
                .into_iter()
                .next()
                .ok_or_else(|| surtgis_core::Error::Other("source has no bands".into()))?;
            let win_gt = *win.transform();
            let out = warp::warp(
                std::slice::from_ref(&win),
                &win_gt,
                tf,
                &grid,
                warp::Resampling::parse(warp_method)?,
            )?;
            return Ok(count(out[0].data().to_owned()));
        }
        // Geographic footprint of the strip plus a margin for the kernel.
        let (bx0, by0, bx1, by1) = strip_gt.bounds(ref_cols, strip_rows);
        let Some(pw) =
            src_info.window_for_bounds(0, bx0 - margin, by0 - margin, bx1 + margin, by1 + margin)
        else {
            return Ok(ndarray::Array2::from_elem((strip_rows, ref_cols), f64::NAN));
        };
        let src_window = read_geotiff_window_bands::<f64, _>(&input, &src_info, 0, &pw)?
            .into_iter()
            .next()
            .ok_or_else(|| surtgis_core::Error::Other("source has no bands".into()))?;
        let mut template = surtgis_core::Raster::<f64>::new(strip_rows, ref_cols);
        template.set_transform(strip_gt);
        let out = surtgis_core::resample_to_grid(&src_window, &template, method)?;
        Ok(count(out.data().to_owned()))
    })
    .context("Failed to resample")?;
    pb.finish_and_clear();
    let elapsed = start.elapsed();
    let _ = warp_method;

    let valid = valid_cells.into_inner();
    if valid == 0 {
        let _ = std::fs::remove_file(&output);
        let (sx0, sy0, sx1, sy1) = src_info.transform.bounds(src_cols, src_rows);
        let (rx0, ry0, rx1, ry1) = ref_gt.bounds(ref_cols, ref_rows);
        anyhow::bail!(
            "no output cell received data: the input does not overlap the reference grid \
             (input bounds {sx0:.4},{sy0:.4},{sx1:.4},{sy1:.4}; reference bounds \
             {rx0:.4},{ry0:.4},{rx1:.4},{ry1:.4}{}). Nothing was written",
            match reproject {
                Some((a, b)) => format!("; input reprojected EPSG:{a} -> EPSG:{b}"),
                None => String::new(),
            }
        );
    }
    if let Some((a, b)) = reproject {
        println!("Reprojected on the fly: EPSG:{a} -> EPSG:{b}");
    }

    println!(
        "Resampled: {} x {} → {} x {} ({} strips of {} rows, {:.1}% valid cells)",
        src_cols,
        src_rows,
        ref_cols,
        ref_rows,
        ref_rows.div_ceil(rows_per_strip),
        rows_per_strip,
        100.0 * valid as f64 / (ref_rows * ref_cols).max(1) as f64
    );
    helpers::done("Resample", &output, elapsed);
    Ok(())
}

/// Decide how `resample` treats the two CRSs: `None` when they match (or
/// one is unknown, with a warning), `Some((src_epsg, ref_epsg))` when the
/// input must be reprojected on the fly. Fails when they differ and cannot
/// be reprojected (no EPSG code, or a build without `projections`).
fn resample_crs_plan(
    input: &std::path::Path,
    reference: &std::path::Path,
    src: &Option<surtgis_core::CRS>,
    dst: &Option<surtgis_core::CRS>,
) -> Result<Option<(u32, u32)>> {
    let (Some(s), Some(d)) = (src, dst) else {
        if src.is_some() != dst.is_some() {
            eprintln!(
                "Warning: {} has no CRS; assuming input and reference share one",
                if src.is_none() {
                    input.display()
                } else {
                    reference.display()
                }
            );
        }
        return Ok(None);
    };
    match (s.epsg(), d.epsg()) {
        (Some(a), Some(b)) if a == b => Ok(None),
        (Some(a), Some(b)) => {
            if cfg!(feature = "projections") {
                Ok(Some((a, b)))
            } else {
                anyhow::bail!(
                    "input CRS (EPSG:{a}) differs from reference CRS (EPSG:{b}) and this build \
                     has no reprojection support; run `surtgis reproject --to {b}` first"
                )
            }
        }
        _ if s.is_equivalent(d) => Ok(None),
        (a, b) => anyhow::bail!(
            "input CRS ({}) differs from reference CRS ({}) and no EPSG code is known for both; \
             run `surtgis reproject` first",
            a.map_or("unknown EPSG".to_string(), |e| format!("EPSG:{e}")),
            b.map_or("unknown EPSG".to_string(), |e| format!("EPSG:{e}"))
        ),
    }
}
