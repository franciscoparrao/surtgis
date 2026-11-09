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

use surtgis_core::Raster;
use surtgis_core::io::{
    GeoTiffOptions, read_geotiff_any, read_geotiff_bands, write_geotiff, write_geotiff_multiband,
    write_geotiff_stack,
};
use surtgis_core::raster::DataType;
use surtgis_core::warp::{self, Resampling, Transformer};

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
    let method = Resampling::parse(&method)?;

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
            Resampling::Nearest => "nearest",
            Resampling::Bilinear => "bilinear",
            _ => "other",
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
/// Whole-raster warp: derive the target grid from the transformed extent
/// (pixel size kept or inferred unless overridden) and fill it by inverse
/// mapping. The kernel lives in `surtgis_core::warp`, shared with the tile
/// server.
fn reproject(
    src_bands: &[Raster<f64>],
    src_epsg: u32,
    dst_epsg: u32,
    method: Resampling,
    pixel_size_override: Option<f64>,
) -> Result<Vec<Raster<f64>>> {
    let src = &src_bands[0];
    let (src_rows, src_cols) = src.shape();
    let src_gt = *src.transform();
    let tf = Transformer::new(src_epsg, dst_epsg)?;
    let grid = warp::grid_for(&src_gt, src_rows, src_cols, &tf, pixel_size_override)
        .map_err(|e| anyhow!("{e}. Use --pixel-size"))?;
    if grid.rows > 200_000 || grid.cols > 200_000 {
        return Err(anyhow!(
            "output dimensions too large ({}x{}). Refine --pixel-size",
            grid.rows,
            grid.cols
        ));
    }
    Ok(warp::warp(src_bands, &src_gt, &tf, &grid, method)?)
}
