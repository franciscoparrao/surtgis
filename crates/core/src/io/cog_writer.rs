//! Native Cloud-Optimized GeoTIFF (COG) writer.
//!
//! Closes SurtGIS's #1 engine-audit capability gap versus GDAL: the
//! engine can *read* remote Cloud-Optimized GeoTIFFs (tiled, range-request
//! friendly, with resolution pyramids — see `surtgis-cloud`'s `CogReader`)
//! but had no way to *write* one, so no pipeline could go
//! `DEM -> terrain algorithm -> publish COG` without shelling out to GDAL.
//!
//! A COG is, structurally, just a GeoTIFF with two properties:
//!
//! 1. **Tiled**, not stripped: pixels are organised into fixed-size square
//!    tiles (`TileWidth`/`TileLength`/`TileOffsets`/`TileByteCounts`, tags
//!    322-325) instead of `RowsPerStrip`/`StripOffsets`/`StripByteCounts`.
//! 2. **Internal overviews**: a reduced-resolution image pyramid stored as
//!    additional IFDs chained after the main (full-resolution) IFD via the
//!    standard TIFF "next IFD" pointer, each one flagged
//!    `NewSubfileType = 1` (reduced-resolution image) and each roughly
//!    half the width/height of the previous level.
//!
//! # Implementation notes
//!
//! The `tiff` crate (0.10) has no high-level API for either of these —
//! `ImageEncoder` only knows how to write strip-organised single images.
//! This writer instead drives the lower-level `DirectoryEncoder` API by
//! hand (the same approach `write_geotiff_stack` in `native.rs` uses for
//! N-band stacks), writing every tag itself and assembling/compressing
//! tile bytes before handing them to the encoder. Calling
//! `TiffEncoder::image_directory()` once per pyramid level automatically
//! chains each directory into the main IFD sequence via the encoder's
//! internal `last_ifd_chain` bookkeeping, which is exactly the layout a
//! COG reader expects (main IFD first, followed by overviews from
//! coarsest full-res down to smallest).
//!
//! `native.rs`'s own tag-writing helpers (`geokeys_for_crs`,
//! `deflate_compress_bytes`, ...) are private to that module and this
//! writer must not modify `native.rs`, so small equivalents are
//! reimplemented here rather than exported across module boundaries.
//!
//! # Completeness (honest status)
//!
//! Implemented and empirically validated against `gdalinfo`/`rasterio`
//! (see the `interop_*` tests below):
//!
//! - Tiled main image (arbitrary tile size, must be a multiple of 16 per
//!   the TIFF spec), with correct partial-tile padding at the right/bottom
//!   edges.
//! - Resolution pyramid (2x2 box-filter downsampling, nodata-aware),
//!   written as internal overview IFDs chained after the main IFD.
//! - BigTIFF (`CogOptions::big_tiff`).
//! - Georeferencing (pixel scale, tiepoint, GeoKeyDirectory, GDAL_NODATA)
//!   on the main IFD, using the same tag payload shape `native.rs` writes
//!   (only EPSG-code CRSs are round-tripped, matching the rest of the
//!   native backend).
//! - DEFLATE compression, per-tile (each tile is its own compressed
//!   stream, i.e. an independent random-access unit — not one deflate
//!   stream for the whole level).
//!
//! Not implemented (documented as future work rather than silently
//! skipped):
//!
//! - **Byte-order-optimised ("ghost area") ifd/tile layout for HTTP range
//!   requests.** This writer prioritises structural correctness (valid
//!   tile/overview placement) over the physical byte ordering the COG
//!   spec recommends for optimal streaming reads — see
//!   <https://github.com/cogeotiff/cog-spec> for that optimisation. A
//!   GDAL/rasterio/any-TIFF-reader client reads the file correctly either
//!   way; only the number of HTTP round-trips for a *remote* read would
//!   differ.
//! - Overview-only georeferencing tags: overview IFDs intentionally carry
//!   no `ModelPixelScaleTag`/`ModelTiepointTag`/`GeoKeyDirectoryTag`,
//!   matching what GDAL's own GTiff/COG driver emits — they are
//!   identified as overviews structurally (`NewSubfileType = 1`, chained
//!   after the main IFD), not via duplicated georeferencing.
//! - Round-trip validation through SurtGIS's own `CogReader`
//!   (`surtgis-cloud`): that reader has no local-file backend today (only
//!   HTTP range requests), and adding one is out of scope for this
//!   writer. `gdalinfo`/`rasterio` (both installed on real GDAL/OGR
//!   toolchains) are the primary empirical validation instead.

use super::native::NativeGraySample;
use crate::error::{Error, Result};
use crate::raster::{Raster, RasterElement};
use std::fs::File;
use std::path::Path;
use tiff::encoder::colortype::ColorType;
use tiff::encoder::{TiffEncoder, TiffKind, TiffKindBig, TiffKindStandard};
use tiff::tags::{CompressionMethod, PhotometricInterpretation, Tag};

/// Compression codec for COG tile data.
///
/// Mirrors the native GeoTIFF writer's supported set (see
/// `native::resolve_compression`): only what the `tiff` crate can
/// actually produce without the `gdal` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CogCompression {
    /// No compression.
    None,
    /// Zlib/DEFLATE ("Adobe Deflate"), applied independently per tile so
    /// every tile stays an independently-decodable random-access unit.
    Deflate,
}

/// Resampling method used to build each overview (pyramid) level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverviewResampling {
    /// 2x2 box-filter mean, excluding nodata/NaN source pixels. Correct
    /// for a power-of-two-aligned pyramid (unlike the general-purpose
    /// `resample::ResampleMethod::Average`, which targets arbitrary,
    /// possibly non-aligned target grids and is unnecessarily expensive
    /// for this case).
    Average,
    /// Nearest-neighbor: the top-left source pixel of each 2x2 block.
    /// Cheaper, appropriate for categorical/classified rasters where
    /// averaging would invent nonsense class values.
    Nearest,
}

/// Options controlling [`write_cog`].
#[derive(Debug, Clone)]
pub struct CogOptions {
    /// Tile edge length in pixels. Must be a positive multiple of 16 (a
    /// hard TIFF-spec requirement for tiled images — libtiff/GDAL will
    /// not read a tiled TIFF whose `TileWidth`/`TileLength` violate it).
    /// Default: 512.
    pub tile_size: u32,
    /// Tile compression. Default: `None`.
    pub compression: CogCompression,
    /// Resampling method for building overview levels. Default:
    /// `Average`.
    pub overview_resampling: OverviewResampling,
    /// Write BigTIFF (64-bit offsets) instead of classic 32-bit-offset
    /// TIFF. Default: `false`.
    pub big_tiff: bool,
    /// Cap the number of overview levels built. `None` (default) builds
    /// levels until the reduced image fits within a single tile in both
    /// dimensions.
    pub max_overview_level: Option<u32>,
}

impl Default for CogOptions {
    fn default() -> Self {
        Self {
            tile_size: 512,
            compression: CogCompression::None,
            overview_resampling: OverviewResampling::Average,
            big_tiff: false,
            max_overview_level: None,
        }
    }
}

/// One level of the resolution pyramid: full single-band, row-major
/// pixel data at this level's resolution.
#[derive(Debug)]
struct Level<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

/// Write a [`Raster`] as a Cloud-Optimized GeoTIFF: tiled, with an
/// internal resolution pyramid (overviews), optionally BigTIFF.
///
/// See the module docs for exactly what "Cloud-Optimized" means here and
/// what is/isn't implemented.
pub fn write_cog<T, P>(raster: &Raster<T>, path: P, options: &CogOptions) -> Result<()>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
    P: AsRef<Path>,
{
    let tile_size = options.tile_size as usize;
    if tile_size == 0 || !tile_size.is_multiple_of(16) {
        return Err(Error::Other(format!(
            "COG tile_size must be a positive multiple of 16 (TIFF spec requirement for \
             tiled images); got {}",
            options.tile_size
        )));
    }

    let levels = build_pyramid(raster, tile_size, options)?;

    let final_path = path.as_ref();
    let tmp_path = final_path.with_extension("tmp");
    let file = File::create(&tmp_path)?;

    let fill: T = raster.nodata().unwrap_or_else(T::zero);

    if options.big_tiff {
        let mut encoder = TiffEncoder::new_big(file)
            .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?;
        for (idx, level) in levels.iter().enumerate() {
            write_level_ifd::<T, _, TiffKindBig>(
                &mut encoder,
                level,
                tile_size,
                fill,
                options.compression,
                idx == 0,
                raster,
            )?;
        }
    } else {
        let mut encoder = TiffEncoder::new(file)
            .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?;
        for (idx, level) in levels.iter().enumerate() {
            write_level_ifd::<T, _, TiffKindStandard>(
                &mut encoder,
                level,
                tile_size,
                fill,
                options.compression,
                idx == 0,
                raster,
            )?;
        }
    }

    std::fs::rename(&tmp_path, final_path)?;
    Ok(())
}

/// Build the full resolution pyramid: `levels[0]` is the raster
/// unchanged, each subsequent level is a 2:1 downsample of the previous
/// one. Stops once a level fits within a single tile in both dimensions
/// (or `options.max_overview_level` overview levels have been built,
/// whichever comes first).
fn build_pyramid<T: RasterElement>(
    raster: &Raster<T>,
    tile_size: usize,
    options: &CogOptions,
) -> Result<Vec<Level<T>>> {
    let (rows, cols) = raster.shape();
    if rows == 0 || cols == 0 {
        return Err(Error::Other("cannot write an empty raster as a COG".into()));
    }
    let data: Vec<T> = raster.data().iter().copied().collect();
    let mut levels = vec![Level { rows, cols, data }];

    let nodata = raster.nodata();
    let mut n_overviews: u32 = 0;
    loop {
        let needs_more = {
            let last = levels.last().expect("levels is never empty");
            last.rows > tile_size || last.cols > tile_size
        };
        if !needs_more {
            break;
        }
        if let Some(max) = options.max_overview_level
            && n_overviews >= max
        {
            break;
        }
        let last = levels.last().expect("levels is never empty");
        let next = downsample_2x2(last, nodata, options.overview_resampling);
        levels.push(next);
        n_overviews += 1;
    }
    Ok(levels)
}

/// Downsample one pyramid level 2:1 in both dimensions.
///
/// `Average`: mean of up to 4 contributing source pixels, excluding
/// nodata/NaN; if all 4 (or the fewer that exist at a right/bottom
/// boundary with an odd source dimension) are nodata, the output pixel is
/// nodata too. `Nearest`: the top-left contributing pixel, verbatim.
fn downsample_2x2<T: RasterElement>(
    level: &Level<T>,
    nodata: Option<T>,
    method: OverviewResampling,
) -> Level<T> {
    let out_rows = level.rows.div_ceil(2).max(1);
    let out_cols = level.cols.div_ceil(2).max(1);
    let mut data = vec![T::zero(); out_rows * out_cols];

    for orow in 0..out_rows {
        for ocol in 0..out_cols {
            let r0 = orow * 2;
            let c0 = ocol * 2;
            data[orow * out_cols + ocol] = match method {
                OverviewResampling::Nearest => level.data[r0 * level.cols + c0],
                OverviewResampling::Average => {
                    let mut sum = 0.0_f64;
                    let mut count = 0u32;
                    for dr in 0..2 {
                        for dc in 0..2 {
                            let r = r0 + dr;
                            let c = c0 + dc;
                            if r < level.rows && c < level.cols {
                                let v = level.data[r * level.cols + c];
                                if !v.is_nodata(nodata)
                                    && let Some(f) = v.to_f64()
                                {
                                    sum += f;
                                    count += 1;
                                }
                            }
                        }
                    }
                    if count == 0 {
                        // Every contributing source pixel was nodata: the
                        // output must say so too, using the raster's own
                        // sentinel if it has one, or the type's default
                        // nodata convention (NaN for floats) otherwise --
                        // never silently invent a `0.0` that looks like
                        // real data.
                        nodata.unwrap_or_else(T::default_nodata)
                    } else {
                        num_traits::cast::<f64, T>(sum / f64::from(count))
                            .unwrap_or_else(|| nodata.unwrap_or_else(T::default_nodata))
                    }
                }
            };
        }
    }

    Level {
        rows: out_rows,
        cols: out_cols,
        data,
    }
}

/// Extract one `tile_size`x`tile_size` tile (row-major) from `level`,
/// padding with `fill` where the tile extends past the level's actual
/// bounds (the standard behaviour for a partial edge tile in a tiled
/// TIFF).
fn extract_tile<T: RasterElement>(
    level: &Level<T>,
    tile_size: usize,
    trow: usize,
    tcol: usize,
    fill: T,
) -> Vec<T> {
    let mut buf = vec![fill; tile_size * tile_size];
    let row0 = trow * tile_size;
    let col0 = tcol * tile_size;
    if row0 >= level.rows || col0 >= level.cols {
        return buf;
    }
    let rows_avail = (level.rows - row0).min(tile_size);
    let cols_avail = (level.cols - col0).min(tile_size);
    for r in 0..rows_avail {
        let src_start = (row0 + r) * level.cols + col0;
        let dst_start = r * tile_size;
        buf[dst_start..dst_start + cols_avail]
            .copy_from_slice(&level.data[src_start..src_start + cols_avail]);
    }
    buf
}

/// Native-endian byte representation of `data`, matching exactly what
/// the `tiff` crate itself writes for a `[T]` sample buffer (see
/// `NativeGraySample::write_ne_bytes` / `native.rs`'s
/// `flatten_native_bytes`, reimplemented here because that helper is
/// private to `native.rs`).
fn flatten_native_bytes<T: NativeGraySample>(data: &[T]) -> Vec<u8>
where
    [T]: tiff::encoder::TiffValue,
{
    let mut buf = Vec::with_capacity(std::mem::size_of_val(data));
    for &v in data {
        v.write_ne_bytes(&mut buf);
    }
    buf
}

/// Zlib/DEFLATE-compress `data` via `flate2`, the same wrapper the
/// `tiff` crate's own `Compression::Deflate` uses internally (so the
/// resulting stream is the "Adobe Deflate" TIFF readers expect).
/// Reimplemented locally (see module docs: `native.rs`'s equivalent is
/// private to that module).
fn deflate_compress_bytes(data: &[u8]) -> Result<Vec<u8>> {
    use std::io::Write as _;
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    encoder
        .write_all(data)
        .map_err(|e| Error::Other(format!("DEFLATE compression failed: {}", e)))?;
    encoder
        .finish()
        .map_err(|e| Error::Other(format!("DEFLATE compression failed: {}", e)))
}

/// GeoKeyDirectoryTag (34735) payload for a raster's CRS. Reimplemented
/// from `native.rs::geokeys_for_crs` (private there) — same payload
/// shape, so the same CRSs (EPSG-code only) round-trip the same way.
fn geokeys_for_crs(crs: Option<&crate::crs::CRS>) -> Vec<u16> {
    if let Some(crs) = crs {
        if let Some(epsg) = crs.epsg() {
            if epsg == 4326 {
                vec![
                    1, 1, 0, 3, // Version 1.1.0, 3 keys
                    1024, 0, 1, 2, // GTModelTypeGeoKey = ModelTypeGeographic
                    1025, 0, 1, 1, // GTRasterTypeGeoKey = RasterPixelIsArea
                    2048, 0, 1, epsg as u16, // GeographicTypeGeoKey
                ]
            } else {
                vec![
                    1, 1, 0, 3, // Version 1.1.0, 3 keys
                    1024, 0, 1, 1, // GTModelTypeGeoKey = ModelTypeProjected
                    1025, 0, 1, 1, // GTRasterTypeGeoKey = RasterPixelIsArea
                    3072, 0, 1, epsg as u16, // ProjectedCSTypeGeoKey
                ]
            }
        } else {
            vec![1, 1, 0, 2, 1024, 0, 1, 1, 1025, 0, 1, 1]
        }
    } else {
        vec![1, 1, 0, 2, 1024, 0, 1, 1, 1025, 0, 1, 1]
    }
}

/// Write one pyramid level as one IFD, chained into `encoder`'s ongoing
/// IFD sequence (`encoder.image_directory()` handles the chaining — see
/// module docs). `is_main` is `true` only for `levels[0]` (the
/// full-resolution image): only it gets `NewSubfileType = 0` and
/// georeferencing tags; every other level is `NewSubfileType = 1`
/// (reduced-resolution / overview) with no georeferencing, matching
/// GDAL's own COG output.
#[allow(clippy::too_many_arguments)]
fn write_level_ifd<T, W, K>(
    encoder: &mut TiffEncoder<W, K>,
    level: &Level<T>,
    tile_size: usize,
    fill: T,
    compression: CogCompression,
    is_main: bool,
    raster: &Raster<T>,
) -> Result<()>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
    W: std::io::Write + std::io::Seek,
    K: TiffKind,
{
    let tiles_across = level.cols.div_ceil(tile_size).max(1);
    let tiles_down = level.rows.div_ceil(tile_size).max(1);

    // Build every (optionally compressed) tile buffer up front, in
    // row-major order -- the order TileOffsets/TileByteCounts must list
    // them in per the TIFF spec.
    let mut tile_bytes: Vec<Vec<u8>> = Vec::with_capacity(tiles_across * tiles_down);
    for trow in 0..tiles_down {
        for tcol in 0..tiles_across {
            let px = extract_tile(level, tile_size, trow, tcol, fill);
            let raw = flatten_native_bytes(&px);
            let bytes = match compression {
                CogCompression::None => raw,
                CogCompression::Deflate => deflate_compress_bytes(&raw)?,
            };
            tile_bytes.push(bytes);
        }
    }

    let mut dir = encoder
        .image_directory()
        .map_err(|e| Error::Other(format!("Cannot create TIFF directory: {}", e)))?;

    dir.write_tag(Tag::ImageWidth, level.cols as u32)
        .map_err(|e| Error::Other(format!("Cannot write ImageWidth: {}", e)))?;
    dir.write_tag(Tag::ImageLength, level.rows as u32)
        .map_err(|e| Error::Other(format!("Cannot write ImageLength: {}", e)))?;
    dir.write_tag(Tag::BitsPerSample, <T::Gray as ColorType>::BITS_PER_SAMPLE[0])
        .map_err(|e| Error::Other(format!("Cannot write BitsPerSample: {}", e)))?;
    dir.write_tag(
        Tag::SampleFormat,
        <T::Gray as ColorType>::SAMPLE_FORMAT[0].to_u16(),
    )
    .map_err(|e| Error::Other(format!("Cannot write SampleFormat: {}", e)))?;
    dir.write_tag(
        Tag::PhotometricInterpretation,
        PhotometricInterpretation::BlackIsZero.to_u16(),
    )
    .map_err(|e| Error::Other(format!("Cannot write PhotometricInterpretation: {}", e)))?;
    dir.write_tag(Tag::SamplesPerPixel, 1u16)
        .map_err(|e| Error::Other(format!("Cannot write SamplesPerPixel: {}", e)))?;
    let compression_code = match compression {
        CogCompression::None => CompressionMethod::None.to_u16(),
        CogCompression::Deflate => CompressionMethod::Deflate.to_u16(),
    };
    dir.write_tag(Tag::Compression, compression_code)
        .map_err(|e| Error::Other(format!("Cannot write Compression: {}", e)))?;
    dir.write_tag(Tag::TileWidth, tile_size as u32)
        .map_err(|e| Error::Other(format!("Cannot write TileWidth: {}", e)))?;
    dir.write_tag(Tag::TileLength, tile_size as u32)
        .map_err(|e| Error::Other(format!("Cannot write TileLength: {}", e)))?;
    dir.write_tag(Tag::NewSubfileType, if is_main { 0u32 } else { 1u32 })
        .map_err(|e| Error::Other(format!("Cannot write NewSubfileType: {}", e)))?;

    // Write tile data, gathering offsets/byte counts as we go.
    let mut offsets_u64 = Vec::with_capacity(tile_bytes.len());
    let mut byte_counts_u64 = Vec::with_capacity(tile_bytes.len());
    for bytes in &tile_bytes {
        let offset = dir
            .write_data(bytes.as_slice())
            .map_err(|e| Error::Other(format!("Cannot write tile data: {}", e)))?;
        offsets_u64.push(offset);
        byte_counts_u64.push(bytes.len() as u64);
    }

    let mut offsets: Vec<K::OffsetType> = Vec::with_capacity(offsets_u64.len());
    for &o in &offsets_u64 {
        offsets.push(
            K::convert_offset(o)
                .map_err(|e| Error::Other(format!("Tile offset overflow: {}", e)))?,
        );
    }
    let mut byte_counts: Vec<K::OffsetType> = Vec::with_capacity(byte_counts_u64.len());
    for &c in &byte_counts_u64 {
        byte_counts.push(
            K::convert_offset(c)
                .map_err(|e| Error::Other(format!("Tile byte count overflow: {}", e)))?,
        );
    }
    dir.write_tag(Tag::TileOffsets, K::convert_slice(&offsets))
        .map_err(|e| Error::Other(format!("Cannot write TileOffsets: {}", e)))?;
    dir.write_tag(Tag::TileByteCounts, K::convert_slice(&byte_counts))
        .map_err(|e| Error::Other(format!("Cannot write TileByteCounts: {}", e)))?;

    if is_main {
        let gt = raster.transform();
        let scale = vec![gt.pixel_width, gt.pixel_height.abs(), 0.0];
        dir.write_tag(Tag::ModelPixelScaleTag, scale.as_slice())
            .map_err(|e| Error::Other(format!("Cannot write scale tag: {}", e)))?;
        let tiepoint = vec![0.0, 0.0, 0.0, gt.origin_x, gt.origin_y, 0.0];
        dir.write_tag(Tag::ModelTiepointTag, tiepoint.as_slice())
            .map_err(|e| Error::Other(format!("Cannot write tiepoint tag: {}", e)))?;
        let geokeys = geokeys_for_crs(raster.crs());
        dir.write_tag(Tag::GeoKeyDirectoryTag, geokeys.as_slice())
            .map_err(|e| Error::Other(format!("Cannot write geokey tag: {}", e)))?;
        if let Some(nd) = raster.nodata()
            && let Some(nd_f64) = nd.to_f64()
        {
            let nodata_str = if nd_f64.is_nan() {
                "nan".to_string()
            } else {
                format!("{}", nd_f64)
            };
            dir.write_tag(Tag::GdalNodata, nodata_str.as_str())
                .map_err(|e| Error::Other(format!("Cannot write nodata tag: {}", e)))?;
        }
    }

    dir.finish()
        .map_err(|e| Error::Other(format!("Cannot finish TIFF directory: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::GeoTransform;
    use tempfile::TempDir;

    fn ramp_raster(rows: usize, cols: usize) -> Raster<f32> {
        let mut r: Raster<f32> = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 30.0, -30.0));
        r.set_crs(Some(crate::crs::CRS::from_epsg(32719)));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, (row * cols + col) as f32).unwrap();
            }
        }
        r
    }

    // -----------------------------------------------------------------
    // Pyramid level calculation
    // -----------------------------------------------------------------

    #[test]
    fn pyramid_stops_once_level_fits_a_single_tile() {
        let r = ramp_raster(2000, 2000);
        let levels = build_pyramid(&r, 512, &CogOptions::default()).unwrap();
        // 2000 -> 1000 -> 500 (500 <= 512, stop). 3 levels total.
        assert_eq!(levels.len(), 3);
        assert_eq!((levels[0].rows, levels[0].cols), (2000, 2000));
        assert_eq!((levels[1].rows, levels[1].cols), (1000, 1000));
        assert_eq!((levels[2].rows, levels[2].cols), (500, 500));
    }

    #[test]
    fn pyramid_handles_odd_dimensions_with_ceil_division() {
        let r = ramp_raster(1025, 513);
        let levels = build_pyramid(&r, 512, &CogOptions::default()).unwrap();
        // 1025 -> 513 -> 257; 513 -> 257 -> 129. Stop once both <= 512.
        assert_eq!((levels[0].rows, levels[0].cols), (1025, 513));
        assert_eq!((levels[1].rows, levels[1].cols), (513, 257));
        assert_eq!((levels[2].rows, levels[2].cols), (257, 129));
        assert!(levels[2].rows <= 512 && levels[2].cols <= 512);
    }

    #[test]
    fn pyramid_no_overviews_when_raster_fits_one_tile() {
        let r = ramp_raster(100, 100);
        let levels = build_pyramid(&r, 512, &CogOptions::default()).unwrap();
        assert_eq!(levels.len(), 1, "a raster smaller than one tile needs no overviews");
    }

    #[test]
    fn pyramid_respects_max_overview_level_cap() {
        let r = ramp_raster(2000, 2000);
        let opts = CogOptions {
            max_overview_level: Some(1),
            ..CogOptions::default()
        };
        let levels = build_pyramid(&r, 512, &opts).unwrap();
        // Main level + exactly 1 capped overview, even though more would
        // normally be built (2000 -> 1000 needs another halving to 500).
        assert_eq!(levels.len(), 2);
        assert_eq!((levels[1].rows, levels[1].cols), (1000, 1000));
    }

    #[test]
    fn pyramid_rejects_empty_raster() {
        let r: Raster<f32> = Raster::new(0, 0);
        let err = build_pyramid(&r, 512, &CogOptions::default()).unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    // -----------------------------------------------------------------
    // Downsampling correctness
    // -----------------------------------------------------------------

    #[test]
    fn downsample_average_matches_hand_computed_mean() {
        // 4x4 raster, values 0..16 row-major. 2x2 average -> 2x2 level
        // with each output cell the mean of its four inputs.
        let mut r: Raster<f64> = Raster::new(4, 4);
        for row in 0..4 {
            for col in 0..4 {
                r.set(row, col, (row * 4 + col) as f64).unwrap();
            }
        }
        let level0 = Level {
            rows: 4,
            cols: 4,
            data: r.data().iter().copied().collect(),
        };
        let level1 = downsample_2x2(&level0, None, OverviewResampling::Average);
        assert_eq!((level1.rows, level1.cols), (2, 2));
        // Top-left 2x2 block: 0,1,4,5 -> mean 2.5
        assert_eq!(level1.data[0], 2.5);
        // Top-right block: 2,3,6,7 -> mean 4.5
        assert_eq!(level1.data[1], 4.5);
        // Bottom-left block: 8,9,12,13 -> mean 10.5
        assert_eq!(level1.data[2], 10.5);
        // Bottom-right block: 10,11,14,15 -> mean 12.5
        assert_eq!(level1.data[3], 12.5);
    }

    #[test]
    fn downsample_average_excludes_nodata() {
        let mut r: Raster<f64> = Raster::new(2, 2);
        r.set(0, 0, 10.0).unwrap();
        r.set(0, 1, f64::NAN).unwrap(); // nodata (NaN sentinel for floats)
        r.set(1, 0, 20.0).unwrap();
        r.set(1, 1, 30.0).unwrap();
        let level0 = Level {
            rows: 2,
            cols: 2,
            data: r.data().iter().copied().collect(),
        };
        let level1 = downsample_2x2(&level0, None, OverviewResampling::Average);
        assert_eq!(level1.data[0], 20.0, "mean of {{10,20,30}}, NaN excluded");
    }

    #[test]
    fn downsample_average_all_nodata_stays_nodata() {
        let level0 = Level {
            rows: 2,
            cols: 2,
            data: vec![f64::NAN; 4],
        };
        let level1 = downsample_2x2(&level0, None, OverviewResampling::Average);
        assert!(level1.data[0].is_nan());
    }

    #[test]
    fn downsample_nearest_takes_top_left_source_pixel() {
        let level0 = Level {
            rows: 2,
            cols: 2,
            data: vec![1.0_f64, 2.0, 3.0, 4.0],
        };
        let level1 = downsample_2x2(&level0, None, OverviewResampling::Nearest);
        assert_eq!(level1.data[0], 1.0);
    }

    // -----------------------------------------------------------------
    // Tile partitioning (incl. partial edge tiles)
    // -----------------------------------------------------------------

    #[test]
    fn extract_tile_full_interior_tile_has_no_padding() {
        let level: Level<u8> = Level {
            rows: 4,
            cols: 4,
            data: (0..16).collect(),
        };
        let tile = extract_tile(&level, 4, 0, 0, 255);
        assert_eq!(tile, (0..16).collect::<Vec<u8>>());
    }

    #[test]
    fn extract_tile_partial_edge_tile_pads_with_fill() {
        // 3x3 level, tile_size 2: tiles_across = tiles_down = 2.
        // Tile (0,0): rows 0-1, cols 0-1 -- full.
        // Tile (0,1): rows 0-1, col 2 only -- right edge padded.
        // Tile (1,0): row 2 only, cols 0-1 -- bottom edge padded.
        // Tile (1,1): row 2, col 2 only -- corner, mostly padded.
        let level: Level<i32> = Level {
            rows: 3,
            cols: 3,
            data: (0..9).collect(), // 0 1 2 / 3 4 5 / 6 7 8
        };
        let fill = -1;

        let t00 = extract_tile(&level, 2, 0, 0, fill);
        assert_eq!(t00, vec![0, 1, 3, 4]);

        let t01 = extract_tile(&level, 2, 0, 1, fill);
        // col 2 of rows 0-1: values 2, 5; second column of tile is padding.
        assert_eq!(t01, vec![2, fill, 5, fill]);

        let t10 = extract_tile(&level, 2, 1, 0, fill);
        // row 2, cols 0-1: values 6, 7; second row of tile is padding.
        assert_eq!(t10, vec![6, 7, fill, fill]);

        let t11 = extract_tile(&level, 2, 1, 1, fill);
        // row 2, col 2: value 8; rest padding.
        assert_eq!(t11, vec![8, fill, fill, fill]);
    }

    #[test]
    fn extract_tile_entirely_outside_level_is_all_fill() {
        let level: Level<u8> = Level {
            rows: 2,
            cols: 2,
            data: vec![1, 2, 3, 4],
        };
        // Tile (5,5) at tile_size 2 starts at row/col 10, fully outside.
        let tile = extract_tile(&level, 2, 5, 5, 9);
        assert_eq!(tile, vec![9, 9, 9, 9]);
    }

    // -----------------------------------------------------------------
    // Compression round-trip
    // -----------------------------------------------------------------

    #[test]
    fn deflate_compress_then_inflate_roundtrips_tile_bytes() {
        use std::io::Read as _;
        let raw: Vec<u8> = (0..=255u8).cycle().take(4096).collect();
        let compressed = deflate_compress_bytes(&raw).unwrap();
        assert!(
            compressed.len() < raw.len(),
            "expected DEFLATE to shrink a repetitive 4KB buffer"
        );
        let mut decoder = flate2::read::ZlibDecoder::new(compressed.as_slice());
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).unwrap();
        assert_eq!(out, raw);
    }

    #[test]
    fn rejects_tile_size_not_multiple_of_16() {
        let r = ramp_raster(64, 64);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad_tile.tif");
        let opts = CogOptions {
            tile_size: 100,
            ..CogOptions::default()
        };
        let err = write_cog(&r, &path, &opts).unwrap_err();
        assert!(format!("{err}").contains("multiple of 16"));
    }

    // -----------------------------------------------------------------
    // End-to-end: write_cog + read back the main level via SurtGIS's
    // own native reader (the `tiff` crate's decoder handles tiled
    // images transparently, same code path as strip-organised files).
    // -----------------------------------------------------------------

    #[test]
    fn write_cog_main_level_roundtrips_through_native_reader() {
        let r = ramp_raster(600, 600); // > one 512 tile -> 1 overview level
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("roundtrip.tif");
        write_cog(&r, &path, &CogOptions::default()).unwrap();

        let back: Raster<f32> = crate::io::native::read_geotiff(&path, None).unwrap();
        assert_eq!(back.shape(), (600, 600));
        for (row, col) in [(0, 0), (5, 5), (599, 599), (300, 17)] {
            assert_eq!(
                back.get(row, col).unwrap(),
                r.get(row, col).unwrap(),
                "mismatch at ({row},{col})"
            );
        }
        // Georeferencing survives too.
        assert_eq!(back.transform().pixel_width, 30.0);
        assert_eq!(back.crs().and_then(|c| c.epsg()), Some(32719));
    }

    #[test]
    fn write_cog_with_deflate_roundtrips_through_native_reader() {
        let r = ramp_raster(300, 400);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("roundtrip_deflate.tif");
        let opts = CogOptions {
            compression: CogCompression::Deflate,
            tile_size: 128,
            ..CogOptions::default()
        };
        write_cog(&r, &path, &opts).unwrap();

        let back: Raster<f32> = crate::io::native::read_geotiff(&path, None).unwrap();
        assert_eq!(back.shape(), (300, 400));
        for (row, col) in [(0, 0), (299, 399), (150, 200)] {
            assert_eq!(back.get(row, col).unwrap(), r.get(row, col).unwrap());
        }
    }

    #[test]
    fn write_cog_bigtiff_roundtrips_through_native_reader() {
        let r = ramp_raster(600, 600);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("roundtrip_big.tif");
        let opts = CogOptions {
            big_tiff: true,
            ..CogOptions::default()
        };
        write_cog(&r, &path, &opts).unwrap();

        let back: Raster<f32> = crate::io::native::read_geotiff(&path, None).unwrap();
        assert_eq!(back.get(0, 0).unwrap(), r.get(0, 0).unwrap());
        assert_eq!(back.get(599, 599).unwrap(), r.get(599, 599).unwrap());
    }

    // -----------------------------------------------------------------
    // GDAL/rasterio interop (empirical validation, not just "in theory")
    // -----------------------------------------------------------------
    //
    // Same pattern as native.rs's `interop_*` tests: skip (not fail) when
    // the tool isn't on PATH, so `cargo test` stays green without a GDAL
    // install, but are the actual proof of interoperability on a machine
    // that has one (this one does).

    fn command_available(cmd: &str, arg: &str) -> bool {
        std::process::Command::new(cmd)
            .arg(arg)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn interop_gdalinfo_reports_tiles_and_overviews() {
        if !command_available("gdalinfo", "--version") {
            eprintln!("skipping interop_gdalinfo_reports_tiles_and_overviews: gdalinfo not on PATH");
            return;
        }
        let r = ramp_raster(2000, 2000);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("interop_cog.tif");
        write_cog(&r, &path, &CogOptions::default()).unwrap();

        let output = std::process::Command::new("gdalinfo")
            .arg(path.to_str().unwrap())
            .output()
            .expect("failed to run gdalinfo");
        assert!(
            output.status.success(),
            "gdalinfo failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = String::from_utf8_lossy(&output.stdout);
        eprintln!("--- gdalinfo output ---\n{text}");
        assert!(
            text.contains("Block=512x512"),
            "expected 512x512 tiles, got:\n{text}"
        );
        assert!(
            text.contains("Overviews"),
            "expected an Overviews section, got:\n{text}"
        );
        // 2000 -> 1000 -> 500: both overview sizes should be listed.
        assert!(text.contains("1000x1000"), "missing 1000x1000 overview:\n{text}");
        assert!(text.contains("500x500"), "missing 500x500 overview:\n{text}");
    }

    #[test]
    fn interop_rasterio_reads_overviews_and_pixel_values() {
        if !command_available("python3", "--version") {
            eprintln!("skipping interop_rasterio_reads_overviews_and_pixel_values: python3 not on PATH");
            return;
        }
        let r = ramp_raster(2000, 2000);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("interop_cog_rasterio.tif");
        write_cog(&r, &path, &CogOptions::default()).unwrap();

        // Note: every logical line below must be a top-level (unindented)
        // Python statement -- the `\` line-continuation in this Rust
        // string literal strips each following line's leading whitespace,
        // so any Python block requiring indentation (e.g. `with:`/`if:`)
        // cannot be expressed this way.
        let script = format!(
            "import rasterio\n\
             import numpy as np\n\
             ds = rasterio.open('{p}')\n\
             ovr = ds.overviews(1)\n\
             assert len(ovr) == 2, ovr\n\
             assert set(ovr) == {{2, 4}}, ovr\n\
             full = ds.read(1)\n\
             assert full.shape == (2000, 2000), full.shape\n\
             assert full[5, 5] == 5 * 2000 + 5, full[5, 5]\n\
             ov = ds.read(1, out_shape=(1, 500, 500))\n\
             region = full[0:4, 0:4].astype(np.float64)\n\
             assert abs(float(ov[0, 0]) - region.mean()) < region.mean() * 0.05 + 1.0, (ov[0,0], region.mean())\n\
             print('OK', ovr, full[5,5], ov[0,0], region.mean())\n",
            p = path.to_str().unwrap()
        );
        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(&script)
            .output()
            .expect("failed to run python3");
        eprintln!(
            "--- rasterio stdout ---\n{}\n--- rasterio stderr ---\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("ModuleNotFoundError") || stderr.contains("No module named") {
                eprintln!(
                    "skipping interop_rasterio_reads_overviews_and_pixel_values: rasterio not installed"
                );
                return;
            }
            panic!("rasterio check failed: {}", stderr);
        }
    }
}
