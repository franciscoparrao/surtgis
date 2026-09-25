//! Windowed GeoTIFF reads: decode only the strips or tiles that intersect
//! a pixel window, at any overview level, without loading the file.
//!
//! [`geotiff_info`] describes the file once (size, bands, georeferencing,
//! the chunking of every IFD); [`read_geotiff_window`] then decodes the
//! chunks a window touches and assembles them. A tile server uses it to
//! serve gigapixel local COGs with memory bounded by the window, and a
//! streaming pipeline can use it to pull arbitrary blocks.
//!
//! Chunk semantics follow the TIFF layout the `tiff` crate exposes: a
//! stripped image is a column of chunks `image_width × rows_per_strip`, a
//! tiled image a grid of `tile_width × tile_height` chunks; edge chunks
//! are padded on disk but decoded to their data size.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;

use crate::crs::CRS;
use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster, RasterElement};

use super::native::{cast_and_normalize, read_crs, read_geotransform, read_nodata};

/// A rectangle of pixels at some level: origin column/row and size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelWindow {
    /// First column.
    pub col: u32,
    /// First row.
    pub row: u32,
    /// Number of columns.
    pub width: u32,
    /// Number of rows.
    pub height: u32,
}

/// One IFD of the file: full resolution (index 0) or a reduced level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeoTiffLevel {
    /// IFD index.
    pub index: usize,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Chunk width (tile width, or the image width for strips).
    pub chunk_width: u32,
    /// Chunk height (tile height, or rows per strip).
    pub chunk_height: u32,
    /// Whether the level is tiled (else stripped).
    pub tiled: bool,
}

impl GeoTiffLevel {
    /// Downsampling factor relative to the full-resolution level.
    pub fn factor(&self, full_width: u32) -> f64 {
        full_width as f64 / self.width as f64
    }
}

/// Everything about a GeoTIFF that windowed reads need.
#[derive(Debug, Clone)]
pub struct GeoTiffInfo {
    /// Full-resolution width.
    pub width: u32,
    /// Full-resolution height.
    pub height: u32,
    /// Bands (samples per pixel, pixel-interleaved).
    pub bands: usize,
    /// Georeferencing of the full-resolution level.
    pub transform: GeoTransform,
    /// CRS, if the GeoKeys resolved to one.
    pub crs: Option<CRS>,
    /// Declared nodata.
    pub nodata: Option<f64>,
    /// Levels, index 0 = full resolution, then reduced levels in file order.
    pub levels: Vec<GeoTiffLevel>,
}

impl GeoTiffInfo {
    /// Georeferencing of level `level` (pixel size scaled by its factor).
    pub fn level_transform(&self, level: usize) -> GeoTransform {
        let l = &self.levels[level];
        let fx = self.width as f64 / l.width as f64;
        let fy = self.height as f64 / l.height as f64;
        let t = &self.transform;
        GeoTransform::new(
            t.origin_x,
            t.origin_y,
            t.pixel_width * fx,
            t.pixel_height * fy,
        )
    }

    /// The coarsest level whose downsampling factor does not exceed
    /// `scale` (output pixels per full-resolution pixel wanted), so the
    /// read stays at or above the requested resolution.
    pub fn level_for_scale(&self, scale: f64) -> usize {
        let mut best = 0;
        for (i, l) in self.levels.iter().enumerate() {
            if l.factor(self.width) <= scale.max(1.0) + 1e-9 {
                best = i;
            }
        }
        best
    }

    /// Pixel window of level `level` covering the world-coordinate box
    /// `[min_x, max_x] × [min_y, max_y]`, clamped to the level; `None`
    /// when they do not intersect.
    pub fn window_for_bounds(
        &self,
        level: usize,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Option<PixelWindow> {
        let l = &self.levels[level];
        let t = self.level_transform(level);
        let pw = t.pixel_width;
        let ph = t.pixel_height.abs();
        let c0 = ((min_x - t.origin_x) / pw).floor().max(0.0);
        let c1 = ((max_x - t.origin_x) / pw).ceil().min(l.width as f64);
        let r0 = ((t.origin_y - max_y) / ph).floor().max(0.0);
        let r1 = ((t.origin_y - min_y) / ph).ceil().min(l.height as f64);
        if c0 >= c1 || r0 >= r1 {
            return None;
        }
        Some(PixelWindow {
            col: c0 as u32,
            row: r0 as u32,
            width: (c1 - c0) as u32,
            height: (r1 - r0) as u32,
        })
    }
}

fn open_decoder(path: &Path) -> Result<Decoder<BufReader<File>>> {
    let file = File::open(path)
        .map_err(|e| Error::Other(format!("cannot open {}: {e}", path.display())))?;
    let source_len = file.metadata().ok().map(|m| m.len());
    const FLOOR: usize = 1024 * 1024 * 1024;
    const RATIO: u64 = 4096;
    let buf_cap = source_len
        .map(|n| n.saturating_mul(RATIO).min(usize::MAX as u64) as usize)
        .unwrap_or(usize::MAX)
        .max(FLOOR);
    let mut limits = Limits::unlimited();
    limits.decoding_buffer_size = buf_cap;
    limits.intermediate_buffer_size = buf_cap;
    limits.ifd_value_size = (64 * 1024 * 1024).min(buf_cap);
    Decoder::new(BufReader::new(file))
        .map(|d| d.with_limits(limits))
        .map_err(|e| Error::Other(format!("TIFF decode error: {e}")))
}

fn level_of<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
    index: usize,
) -> Result<GeoTiffLevel> {
    let (width, height) = decoder
        .dimensions()
        .map_err(|e| Error::Other(format!("cannot read dimensions: {e}")))?;
    let (chunk_width, chunk_height) = decoder.chunk_dimensions();
    let tiled = decoder.get_tag_u32(Tag::TileWidth).is_ok();
    Ok(GeoTiffLevel {
        index,
        width,
        height,
        chunk_width,
        chunk_height,
        tiled,
    })
}

/// Describe a GeoTIFF: size, bands, georeferencing and the chunking of
/// its full-resolution IFD and every reduced-resolution IFD after it.
pub fn geotiff_info<P: AsRef<Path>>(path: P) -> Result<GeoTiffInfo> {
    let path = path.as_ref();
    let mut decoder = open_decoder(path)?;
    let spp = decoder
        .get_tag_u32(Tag::SamplesPerPixel)
        .map(|v| v as usize)
        .unwrap_or(1)
        .max(1);
    if spp > 1
        && decoder
            .get_tag_u32(Tag::PlanarConfiguration)
            .map(|v| v == 2)
            .unwrap_or(false)
    {
        return Err(Error::Other(
            "planar (PlanarConfiguration=2) multi-band TIFFs are not supported".into(),
        ));
    }
    let transform = read_geotransform(&mut decoder)?;
    let crs = read_crs(&mut decoder);
    let nodata = read_nodata(&mut decoder);
    let full = level_of(&mut decoder, 0)?;
    let mut levels = vec![full.clone()];
    let mut index = 0;
    while decoder.more_images() {
        index += 1;
        if decoder.next_image().is_err() {
            break;
        }
        let Ok(l) = level_of(&mut decoder, index) else {
            break;
        };
        // Reduced-resolution IFDs shrink; anything else (masks, unrelated
        // pages) is skipped.
        if l.width < full.width && l.height < full.height {
            levels.push(l);
        }
    }
    Ok(GeoTiffInfo {
        width: full.width,
        height: full.height,
        bands: spp,
        transform,
        crs,
        nodata,
        levels,
    })
}

fn chunk_to_vec<T: RasterElement>(result: DecodingResult, nodata: Option<f64>) -> Result<Vec<T>> {
    Ok(match result {
        DecodingResult::F32(b) => cast_and_normalize::<T, f32>(b, nodata),
        DecodingResult::F64(b) => cast_and_normalize::<T, f64>(b, nodata),
        DecodingResult::U8(b) => cast_and_normalize::<T, u8>(b, nodata),
        DecodingResult::U16(b) => cast_and_normalize::<T, u16>(b, nodata),
        DecodingResult::U32(b) => cast_and_normalize::<T, u32>(b, nodata),
        DecodingResult::I8(b) => cast_and_normalize::<T, i8>(b, nodata),
        DecodingResult::I16(b) => cast_and_normalize::<T, i16>(b, nodata),
        DecodingResult::I32(b) => cast_and_normalize::<T, i32>(b, nodata),
        _ => {
            return Err(Error::UnsupportedDataType(
                "Unsupported TIFF pixel format".to_string(),
            ));
        }
    })
}

/// Read every band of `window` at `level`, decoding only the chunks the
/// window intersects. Each band comes back georeferenced to the window,
/// with the file's CRS and nodata (float nodata normalised to NaN).
pub fn read_geotiff_window_bands<T, P>(
    path: P,
    info: &GeoTiffInfo,
    level: usize,
    window: &PixelWindow,
) -> Result<Vec<Raster<T>>>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    let Some(lvl) = info.levels.get(level) else {
        return Err(Error::Other(format!(
            "level {level} out of range (file has {})",
            info.levels.len()
        )));
    };
    if window.width == 0
        || window.height == 0
        || window.col + window.width > lvl.width
        || window.row + window.height > lvl.height
    {
        return Err(Error::Other(format!(
            "window {window:?} outside level {level} ({}×{})",
            lvl.width, lvl.height
        )));
    }
    let mut decoder = open_decoder(path.as_ref())?;
    decoder
        .seek_to_image(lvl.index)
        .map_err(|e| Error::Other(format!("cannot seek to IFD {}: {e}", lvl.index)))?;

    let spp = info.bands;
    let (cw, ch) = (lvl.chunk_width as usize, lvl.chunk_height as usize);
    let per_row = (lvl.width as usize).div_ceil(cw);
    let (ww, wh) = (window.width as usize, window.height as usize);
    let mut planes: Vec<Vec<T>> = (0..spp)
        .map(|_| vec![T::default_nodata(); ww * wh])
        .collect();

    let cx0 = window.col as usize / cw;
    let cx1 = (window.col as usize + ww - 1) / cw;
    let cy0 = window.row as usize / ch;
    let cy1 = (window.row as usize + wh - 1) / ch;
    for cy in cy0..=cy1 {
        for cx in cx0..=cx1 {
            let idx = (cy * per_row + cx) as u32;
            let (dw, dh) = decoder.chunk_data_dimensions(idx);
            let (dw, dh) = (dw as usize, dh as usize);
            let data: Vec<T> = chunk_to_vec(
                decoder
                    .read_chunk(idx)
                    .map_err(|e| Error::Other(format!("cannot read chunk {idx}: {e}")))?,
                info.nodata,
            )?;
            if data.len() < dw * dh * spp {
                return Err(Error::Other(format!(
                    "chunk {idx} decoded to {} values, expected {}",
                    data.len(),
                    dw * dh * spp
                )));
            }
            // Intersection of this chunk with the window, in level pixels.
            let (chunk_x0, chunk_y0) = (cx * cw, cy * ch);
            let x0 = chunk_x0.max(window.col as usize);
            let x1 = (chunk_x0 + dw).min(window.col as usize + ww);
            let y0 = chunk_y0.max(window.row as usize);
            let y1 = (chunk_y0 + dh).min(window.row as usize + wh);
            for y in y0..y1 {
                let src_row = (y - chunk_y0) * dw;
                let dst_row = (y - window.row as usize) * ww;
                for x in x0..x1 {
                    let src = (src_row + (x - chunk_x0)) * spp;
                    let dst = dst_row + (x - window.col as usize);
                    for (band, plane) in planes.iter_mut().enumerate() {
                        plane[dst] = data[src + band];
                    }
                }
            }
        }
    }

    let lt = info.level_transform(level);
    let transform = GeoTransform::new(
        lt.origin_x + window.col as f64 * lt.pixel_width,
        lt.origin_y + window.row as f64 * lt.pixel_height,
        lt.pixel_width,
        lt.pixel_height,
    );
    planes
        .into_iter()
        .map(|plane| {
            let mut r = Raster::from_vec(plane, wh, ww)?;
            r.set_transform(transform);
            r.set_crs(info.crs.clone());
            r.set_nodata(info.nodata.and_then(|nd| {
                if T::is_float() {
                    Some(T::default_nodata())
                } else {
                    num_traits::cast::<f64, T>(nd)
                }
            }));
            Ok(r)
        })
        .collect()
}

/// Read one band (0-based, default 0) of `window` at `level`.
pub fn read_geotiff_window<T, P>(
    path: P,
    info: &GeoTiffInfo,
    level: usize,
    window: &PixelWindow,
    band: Option<usize>,
) -> Result<Raster<T>>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    let band = band.unwrap_or(0);
    let mut bands = read_geotiff_window_bands::<T, P>(path, info, level, window)?;
    if band >= bands.len() {
        return Err(Error::Other(format!(
            "band {band} out of range (file has {})",
            bands.len()
        )));
    }
    Ok(bands.swap_remove(band))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{CogOptions, read_geotiff, write_cog, write_geotiff};

    fn ramp(rows: usize, cols: usize) -> Raster<f32> {
        let mut r = Raster::<f32>::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                r.data_mut()[[i, j]] = (i * cols + j) as f32;
            }
        }
        r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
        r.set_crs(Some(CRS::from_epsg(32719)));
        r.set_nodata(Some(-1.0));
        r
    }

    fn check_window_equals_slice(path: &Path, info: &GeoTiffInfo, w: PixelWindow) {
        let full: Raster<f32> = read_geotiff(path, None).unwrap();
        let win: Raster<f32> = read_geotiff_window(path, info, 0, &w, None).unwrap();
        assert_eq!(win.shape(), (w.height as usize, w.width as usize));
        for i in 0..w.height as usize {
            for j in 0..w.width as usize {
                let (a, b) = (
                    win.data()[[i, j]],
                    full.data()[[w.row as usize + i, w.col as usize + j]],
                );
                assert!(
                    a == b || (a.is_nan() && b.is_nan()),
                    "({i},{j}) of {w:?}: {a} vs {b}"
                );
            }
        }
        let t = win.transform();
        assert_eq!(t.origin_x, 500_000.0 + w.col as f64 * 10.0);
        assert_eq!(t.origin_y, 6_300_000.0 - w.row as f64 * 10.0);
        assert_eq!(win.crs().and_then(|c| c.epsg()), Some(32719));
    }

    /// Tiled COG with overviews: windows crossing tile borders equal the
    /// slice of the full read; the levels shrink by 2 and a level-1
    /// window is readable and georeferenced with the doubled pixel.
    #[test]
    fn tiled_cog_windows_and_levels() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cog.tif");
        let r = ramp(300, 500);
        let opts = CogOptions {
            tile_size: 128,
            ..CogOptions::default()
        };
        write_cog(&r, &path, &opts).unwrap();
        let info = geotiff_info(&path).unwrap();
        assert_eq!((info.width, info.height, info.bands), (500, 300, 1));
        assert!(info.levels[0].tiled);
        assert_eq!(
            (info.levels[0].chunk_width, info.levels[0].chunk_height),
            (128, 128)
        );
        assert!(info.levels.len() >= 2, "levels {:?}", info.levels);
        assert_eq!(info.levels[1].width, 250);
        assert_eq!(info.nodata, Some(-1.0));

        for w in [
            PixelWindow {
                col: 0,
                row: 0,
                width: 10,
                height: 10,
            },
            PixelWindow {
                col: 120,
                row: 120,
                width: 20,
                height: 20,
            }, // crosses 4 tiles
            PixelWindow {
                col: 490,
                row: 290,
                width: 10,
                height: 10,
            }, // edge tiles
            PixelWindow {
                col: 0,
                row: 0,
                width: 500,
                height: 300,
            }, // everything
        ] {
            check_window_equals_slice(&path, &info, w);
        }

        let lvl1 = PixelWindow {
            col: 60,
            row: 60,
            width: 30,
            height: 20,
        };
        let win: Raster<f32> = read_geotiff_window(&path, &info, 1, &lvl1, None).unwrap();
        assert_eq!(win.shape(), (20, 30));
        assert_eq!(win.transform().pixel_width, 20.0);
        assert_eq!(win.transform().origin_x, 500_000.0 + 60.0 * 20.0);
        assert!(win.data().iter().all(|v| v.is_finite()));

        assert_eq!(info.level_for_scale(1.0), 0);
        assert_eq!(info.level_for_scale(2.5), 1);
        let ww = info
            .window_for_bounds(
                0,
                500_000.0 + 1_195.0,
                6_300_000.0 - 3_000.0,
                500_000.0 + 5_005.0,
                6_300_000.0 - 1_195.0,
            )
            .unwrap();
        assert_eq!(
            ww,
            PixelWindow {
                col: 119,
                row: 119,
                width: 381, // cols 119..500: 5005 m → 500.5 → ceil 501, clamped to 500
                height: 181
            }
        );
        assert!(
            info.window_for_bounds(0, 900_000.0, 0.0, 900_100.0, 100.0)
                .is_none()
        );
        assert!(
            read_geotiff_window::<f32, _>(
                &path,
                &info,
                0,
                &PixelWindow {
                    col: 495,
                    row: 0,
                    width: 10,
                    height: 10
                },
                None
            )
            .is_err()
        );
    }

    /// Stripped GeoTIFF (the default writer output): the same window
    /// arithmetic works with strips as chunks; nodata becomes NaN.
    #[test]
    fn stripped_geotiff_windows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("strips.tif");
        let mut r = ramp(64, 40);
        r.data_mut()[[5, 5]] = -1.0;
        write_geotiff(&r, &path, None).unwrap();
        let info = geotiff_info(&path).unwrap();
        assert!(!info.levels[0].tiled);
        assert_eq!(info.levels.len(), 1);
        assert_eq!(info.levels[0].chunk_width, 40);
        check_window_equals_slice(
            &path,
            &info,
            PixelWindow {
                col: 3,
                row: 2,
                width: 30,
                height: 60,
            },
        );
        let win: Raster<f32> = read_geotiff_window(
            &path,
            &info,
            0,
            &PixelWindow {
                col: 0,
                row: 0,
                width: 10,
                height: 10,
            },
            None,
        )
        .unwrap();
        assert!(win.data()[[5, 5]].is_nan());
        assert!(win.nodata().unwrap().is_nan());
    }
}
