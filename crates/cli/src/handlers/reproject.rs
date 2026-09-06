//! Native raster reprojection between arbitrary CRSs via proj4rs.
//!
//! Closes the last operational gap that forced users to fall back to
//! `gdalwarp`. With this command, the full SurtGIS pipeline
//! (STAC composite → reproject → terrain analysis → output) runs without
//! a system GDAL dependency.
//!
//! Multi-band aware (issue #123): every band of the input is reprojected
//! onto one shared output grid — the per-pixel coordinate transform is
//! computed once, not once per band — and the result is written as a
//! single multi-band GeoTIFF (RGB/RGBA photometric for 3/4 bands, plain
//! band stack otherwise). The output preserves the input's sample type, so
//! a u8 RGB orthophoto comes back as u8 RGB, not as float greyscale.
//!
//! Performance: a 10 000 × 10 000 raster takes ~30–60 s to reproject in
//! release mode on the i7-1270P benchmark machine. Reprojection is
//! parallelised across rows via Rayon. proj4rs handles the coordinate
//! transformation; we handle the per-pixel inverse-mapping and
//! interpolation.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;

use surtgis_core::io::{
    GeoTiffOptions, read_geotiff_any, read_geotiff_bands, write_geotiff, write_geotiff_multiband,
    write_geotiff_stack,
};
use surtgis_core::raster::DataType;
use surtgis_core::{Raster, crs::CRS, raster::GeoTransform};

/// Resampling method for the inverse-mapping interpolation.
#[derive(Debug, Clone, Copy)]
enum Method {
    Nearest,
    Bilinear,
}

impl Method {
    fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "nearest" | "nn" => Ok(Method::Nearest),
            "bilinear" | "bl" => Ok(Method::Bilinear),
            other => Err(anyhow!(
                "unknown resampling method: '{}'. Supported: nearest, bilinear",
                other
            )),
        }
    }
}

/// Parse `"EPSG:32719"` or `"32719"` into a u32 EPSG code.
fn parse_epsg(s: &str) -> Result<u32> {
    let trimmed = s.trim();
    let stripped = trimmed
        .strip_prefix("EPSG:")
        .or_else(|| trimmed.strip_prefix("epsg:"))
        .unwrap_or(trimmed);
    stripped
        .parse::<u32>()
        .with_context(|| format!("invalid EPSG code: '{}'. Expected e.g. EPSG:32719", s))
}

pub fn handle(
    input: PathBuf,
    output: PathBuf,
    to: String,
    from: Option<String>,
    method: String,
    pixel_size: Option<f64>,
    compress: bool,
) -> Result<()> {
    let dst_epsg = parse_epsg(&to)?;
    let method = Method::parse(&method)?;

    // The resampling maths runs in f64 regardless of the stored sample
    // type; the input dtype is probed separately so the output can be
    // written back with the same type (u8 RGB in, u8 RGB out).
    let bands: Vec<Raster<f64>> = read_geotiff_bands(&input)
        .with_context(|| format!("failed to read input GeoTIFF: {}", input.display()))?;
    if bands.is_empty() {
        return Err(anyhow!("input has no bands"));
    }
    let dtype = read_geotiff_any(&input, Some(0))
        .map(|r| r.dtype())
        .unwrap_or(DataType::F64);

    let src_epsg = match from {
        Some(f) => parse_epsg(&f)?,
        None => bands[0].crs().and_then(|c| c.epsg()).ok_or_else(|| {
            anyhow!("source CRS not embedded in input GeoTIFF; pass --from EPSG:XXXX to override")
        })?,
    };

    let start = Instant::now();

    if src_epsg == dst_epsg {
        eprintln!(
            "source and target CRS are the same (EPSG:{}); copying input to output",
            src_epsg
        );
        write_output(&bands, dtype, &output, compress)?;
        eprintln!(
            "✓ wrote {} in {:.2} s",
            output.display(),
            start.elapsed().as_secs_f64()
        );
        return Ok(());
    }

    eprintln!(
        "Reprojecting EPSG:{} → EPSG:{} ({} band{}, {} rows, {} method)",
        src_epsg,
        dst_epsg,
        bands.len(),
        if bands.len() == 1 { "" } else { "s" },
        bands[0].shape().0,
        match method {
            Method::Nearest => "nearest",
            Method::Bilinear => "bilinear",
        }
    );

    let out = reproject(&bands, src_epsg, dst_epsg, method, pixel_size)?;

    write_output(&out, dtype, &output, compress)?;
    eprintln!(
        "✓ wrote {} ({}×{}, {} band{}) in {:.2} s",
        output.display(),
        out[0].shape().0,
        out[0].shape().1,
        out.len(),
        if out.len() == 1 { "" } else { "s" },
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Convert one reprojected f64 band back to an integer sample type,
/// rounding and saturating. Cells that fell outside the source footprint
/// (NaN) become 0, the conventional fill for integer imagery.
fn quantize_band<T>(band: &Raster<f64>, min: f64, max: f64, cast: impl Fn(f64) -> T) -> Raster<T>
where
    T: surtgis_core::raster::RasterElement,
{
    let (rows, cols) = band.shape();
    let data: Vec<T> = band
        .data()
        .iter()
        .map(|&v| {
            if v.is_finite() {
                cast(v.round().clamp(min, max))
            } else {
                cast(0.0)
            }
        })
        .collect();
    let mut out = Raster::from_vec(data, rows, cols).expect("shape preserved");
    out.set_transform(*band.transform());
    out.set_crs(band.crs().cloned());
    out
}

/// Dispatch a typed band group to the right writer: one band through
/// `write_geotiff`, 3 or 4 through `write_geotiff_multiband` (RGB/RGBA
/// photometric, sample type preserved), any other count through
/// `write_geotiff_stack` (BlackIsZero, sample type preserved). A macro
/// instead of a generic fn because the writers' trait bounds are private
/// to core; instantiating at concrete types sidesteps them.
macro_rules! write_group {
    ($bands:expr, $path:expr, $opts:expr) => {{
        let refs: Vec<_> = $bands.iter().collect();
        let path: &Path = $path;
        let result = match refs.len() {
            1 => write_geotiff(refs[0], path, $opts),
            3 | 4 => write_geotiff_multiband(&refs, path, $opts),
            _ => write_geotiff_stack(&refs, None, path, &$opts.unwrap_or_default()),
        };
        result.with_context(|| format!("failed to write {}", path.display()))
    }};
}

/// Write the reprojected bands as one GeoTIFF, restoring the input's
/// sample type (u8 RGB in, u8 RGB out; u16 and i16 likewise; f32 stays
/// f32; everything else is written as f64).
fn write_output(
    bands: &[Raster<f64>],
    dtype: DataType,
    output: &Path,
    compress: bool,
) -> Result<()> {
    let opts = if compress {
        Some(GeoTiffOptions {
            compression: "DEFLATE".into(),
            ..Default::default()
        })
    } else {
        None
    };

    match dtype {
        DataType::U8 => {
            let typed: Vec<Raster<u8>> = bands
                .iter()
                .map(|b| quantize_band(b, 0.0, 255.0, |v| v as u8))
                .collect();
            write_group!(typed, output, opts)?;
        }
        DataType::U16 => {
            let typed: Vec<Raster<u16>> = bands
                .iter()
                .map(|b| quantize_band(b, 0.0, 65535.0, |v| v as u16))
                .collect();
            write_group!(typed, output, opts)?;
        }
        DataType::I16 => {
            let typed: Vec<Raster<i16>> = bands
                .iter()
                .map(|b| quantize_band(b, -32768.0, 32767.0, |v| v as i16))
                .collect();
            write_group!(typed, output, opts)?;
        }
        DataType::F32 => {
            let typed: Vec<Raster<f32>> = bands
                .iter()
                .map(|b| {
                    let (rows, cols) = b.shape();
                    let data: Vec<f32> = b.data().iter().map(|&v| v as f32).collect();
                    let mut out = Raster::from_vec(data, rows, cols).expect("shape preserved");
                    out.set_transform(*b.transform());
                    out.set_crs(b.crs().cloned());
                    out.set_nodata(Some(f32::NAN));
                    out
                })
                .collect();
            write_group!(typed, output, opts)?;
        }
        // F64 and the exotic integer widths stay in f64, which every
        // downstream SurtGIS tool consumes natively.
        _ => {
            write_group!(bands, output, opts)?;
        }
    }
    Ok(())
}

/// Reproject a stack of co-registered bands from src_epsg to dst_epsg.
/// The output grid and the per-pixel inverse coordinate mapping are
/// computed once from the first band and shared by all of them.
fn reproject(
    src_bands: &[Raster<f64>],
    src_epsg: u32,
    dst_epsg: u32,
    method: Method,
    pixel_size_override: Option<f64>,
) -> Result<Vec<Raster<f64>>> {
    use proj4rs::Proj;

    let src = &src_bands[0];
    let n_bands = src_bands.len();
    for (i, b) in src_bands.iter().enumerate().skip(1) {
        if b.shape() != src.shape() {
            return Err(anyhow!(
                "band {} has shape {:?}, band 1 has {:?}; bands must be co-registered",
                i + 1,
                b.shape(),
                src.shape()
            ));
        }
    }

    let src_proj = Proj::from_epsg_code(src_epsg as u16)
        .map_err(|e| anyhow!("proj4rs failed to load EPSG:{}: {:?}", src_epsg, e))?;
    let dst_proj = Proj::from_epsg_code(dst_epsg as u16)
        .map_err(|e| anyhow!("proj4rs failed to load EPSG:{}: {:?}", dst_epsg, e))?;

    let src_gt = *src.transform();
    let (src_rows, src_cols) = src.shape();

    // Compute output extent by transforming the source raster's four corners
    // into the target CRS, plus a sampling of edge midpoints for non-affine
    // projections that bow the bounding box.
    let mut samples: Vec<(f64, f64)> = Vec::new();
    for &r in &[0usize, src_rows / 2, src_rows] {
        for &c in &[0usize, src_cols / 2, src_cols] {
            let x = src_gt.origin_x + c as f64 * src_gt.pixel_width;
            let y = src_gt.origin_y + r as f64 * src_gt.pixel_height;
            samples.push((x, y));
        }
    }

    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for &(x, y) in &samples {
        let (in_x, in_y) = if src_proj.is_latlong() {
            (x.to_radians(), y.to_radians())
        } else {
            (x, y)
        };
        let (tx, ty) = proj4rs::adaptors::transform_xy(&src_proj, &dst_proj, in_x, in_y)
            .map_err(|e| anyhow!("proj4rs transform failed at ({}, {}): {:?}", x, y, e))?;
        let (out_x, out_y) = if dst_proj.is_latlong() {
            (tx.to_degrees(), ty.to_degrees())
        } else {
            (tx, ty)
        };
        min_x = min_x.min(out_x);
        min_y = min_y.min(out_y);
        max_x = max_x.max(out_x);
        max_y = max_y.max(out_y);
    }

    // Default pixel size: keep approximate ground resolution. If source is
    // metric (e.g. UTM), reuse source pixel width. If source is geographic
    // (degrees) and target is metric, approximate using the latitude-weighted
    // metres-per-degree at the centre of the source extent.
    let px = match pixel_size_override {
        Some(p) => p,
        None => infer_default_pixel_size(
            &src_gt, &src_proj, &dst_proj, &samples, max_x, max_y, min_x, min_y,
        )?,
    };

    let out_cols = ((max_x - min_x) / px).ceil() as usize;
    let out_rows = ((max_y - min_y) / px).ceil() as usize;

    if out_rows == 0 || out_cols == 0 {
        return Err(anyhow!(
            "output dimensions are zero (extent: {:.3}..{:.3}, {:.3}..{:.3}; pixel {}). Use --pixel-size",
            min_x,
            max_x,
            min_y,
            max_y,
            px
        ));
    }
    if out_rows > 200_000 || out_cols > 200_000 {
        return Err(anyhow!(
            "output dimensions too large ({}x{}). Refine --pixel-size",
            out_rows,
            out_cols
        ));
    }

    let out_gt = GeoTransform::new(min_x, max_y, px, -px);

    // Inverse-mapping per output pixel, parallelised across rows with
    // Rayon. The expensive part — the dst→src coordinate transform — runs
    // once per pixel and its result samples every band.
    let src_datas: Vec<&ndarray::Array2<f64>> = src_bands.iter().map(|b| b.data()).collect();
    let row_results: Vec<Vec<Vec<f64>>> = (0..out_rows)
        .into_par_iter()
        .map(|out_r| {
            let mut rows: Vec<Vec<f64>> = (0..n_bands).map(|_| vec![f64::NAN; out_cols]).collect();
            let dst_y = max_y - (out_r as f64 + 0.5) * px;
            for out_c in 0..out_cols {
                let dst_x = min_x + (out_c as f64 + 0.5) * px;
                let (in_x, in_y) = if dst_proj.is_latlong() {
                    (dst_x.to_radians(), dst_y.to_radians())
                } else {
                    (dst_x, dst_y)
                };
                let (sx, sy) =
                    match proj4rs::adaptors::transform_xy(&dst_proj, &src_proj, in_x, in_y) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                let (src_x, src_y) = if src_proj.is_latlong() {
                    (sx.to_degrees(), sy.to_degrees())
                } else {
                    (sx, sy)
                };

                let src_c_f = (src_x - src_gt.origin_x) / src_gt.pixel_width - 0.5;
                let src_r_f = (src_y - src_gt.origin_y) / src_gt.pixel_height - 0.5;

                for (band, data) in src_datas.iter().enumerate() {
                    let val = match method {
                        Method::Nearest => {
                            sample_nearest(data, src_rows, src_cols, src_r_f, src_c_f)
                        }
                        Method::Bilinear => {
                            sample_bilinear(data, src_rows, src_cols, src_r_f, src_c_f)
                        }
                    };
                    if let Some(v) = val {
                        rows[band][out_c] = v;
                    }
                }
            }
            rows
        })
        .collect();

    let mut outputs: Vec<Raster<f64>> = (0..n_bands)
        .map(|band| {
            let mut out = Raster::<f64>::new(out_rows, out_cols);
            out.set_transform(out_gt);
            out.set_crs(Some(CRS::from_epsg(dst_epsg)));
            if let Some(nd) = src_bands[band].nodata() {
                out.set_nodata(Some(nd));
            }
            out
        })
        .collect();

    for (out_r, rows) in row_results.into_iter().enumerate() {
        for (band, row) in rows.into_iter().enumerate() {
            for (out_c, v) in row.into_iter().enumerate() {
                let _ = outputs[band].set(out_r, out_c, v);
            }
        }
    }

    Ok(outputs)
}

fn sample_nearest(
    data: &ndarray::Array2<f64>,
    rows: usize,
    cols: usize,
    src_r_f: f64,
    src_c_f: f64,
) -> Option<f64> {
    let r = src_r_f.round();
    let c = src_c_f.round();
    if r < 0.0 || c < 0.0 || r >= rows as f64 || c >= cols as f64 {
        return None;
    }
    let v = data[[r as usize, c as usize]];
    if v.is_finite() { Some(v) } else { None }
}

fn sample_bilinear(
    data: &ndarray::Array2<f64>,
    rows: usize,
    cols: usize,
    src_r_f: f64,
    src_c_f: f64,
) -> Option<f64> {
    let c0 = src_c_f.floor() as isize;
    let r0 = src_r_f.floor() as isize;
    let fc = src_c_f - c0 as f64;
    let fr = src_r_f - r0 as f64;

    if c0 < 0 || r0 < 0 || (c0 + 1) >= cols as isize || (r0 + 1) >= rows as isize {
        return None;
    }

    let r0u = r0 as usize;
    let c0u = c0 as usize;
    let v00 = data[[r0u, c0u]];
    let v01 = data[[r0u, c0u + 1]];
    let v10 = data[[r0u + 1, c0u]];
    let v11 = data[[r0u + 1, c0u + 1]];

    if !(v00.is_finite() && v01.is_finite() && v10.is_finite() && v11.is_finite()) {
        return None;
    }

    Some(
        v00 * (1.0 - fc) * (1.0 - fr)
            + v01 * fc * (1.0 - fr)
            + v10 * (1.0 - fc) * fr
            + v11 * fc * fr,
    )
}

fn infer_default_pixel_size(
    src_gt: &GeoTransform,
    src_proj: &proj4rs::Proj,
    dst_proj: &proj4rs::Proj,
    samples: &[(f64, f64)],
    max_x: f64,
    max_y: f64,
    min_x: f64,
    min_y: f64,
) -> Result<f64> {
    // If both CRS are in the same kind of units (both metric or both geographic),
    // reuse source pixel width.
    if src_proj.is_latlong() == dst_proj.is_latlong() {
        return Ok(src_gt.pixel_width.abs());
    }

    // Otherwise approximate: use the ratio of (output extent width / source extent width)
    // times the source pixel width. This produces an output pixel size that roughly
    // preserves the column count, which is a sensible default for ad-hoc reprojection.
    let src_x_min = samples.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let src_x_max = samples
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let src_width = src_x_max - src_x_min;
    if src_width <= 0.0 {
        return Err(anyhow!("could not infer pixel size; use --pixel-size"));
    }
    let dst_width = max_x - min_x;
    let _ = (max_y, min_y);
    Ok(dst_width / src_width * src_gt.pixel_width.abs())
}
