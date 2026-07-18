//! Native GeoTIFF reading/writing (without GDAL dependency)
//!
//! Uses the `tiff` crate for basic TIFF I/O.
//! For full GeoTIFF support (projections, advanced types), enable the `gdal` feature.

use super::GeoTiffOptions;
use crate::error::{Error, Result};
use crate::raster::{AnyRaster, GeoTransform, Raster, RasterElement};
use std::any::{Any, TypeId};
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::encoder::colortype::{
    ColorType, Gray8, Gray16, Gray32, Gray32Float, Gray64, Gray64Float, GrayI8, GrayI16, GrayI32,
    GrayI64, RGB32Float, RGBA32Float,
};
use tiff::encoder::compression::DeflateLevel;
use tiff::encoder::{
    Compression, DirectoryEncoder, TiffEncoder, TiffKind, TiffKindBig, TiffKindStandard,
};
use tiff::tags::{CompressionMethod, PhotometricInterpretation, Tag};

/// Read a GeoTIFF file into a Raster
///
/// Native reader with limited GeoTIFF metadata support.
/// For full support, enable the `gdal` feature.
pub fn read_geotiff<T, P>(path: P, band: Option<usize>) -> Result<Raster<T>>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref())?;
    let len = file.metadata().ok().map(|m| m.len());
    decode_geotiff(file, band, len)
}

/// Read a GeoTIFF from an in-memory buffer into a Raster
///
/// Same as `read_geotiff` but operates on a byte slice instead of a file path.
/// Useful for WASM environments where filesystem access is not available.
pub fn read_geotiff_from_buffer<T>(data: &[u8], band: Option<usize>) -> Result<Raster<T>>
where
    T: RasterElement,
{
    let cursor = Cursor::new(data);
    decode_geotiff(cursor, band, Some(data.len() as u64))
}

/// Read a GeoTIFF file into an [`AnyRaster`], preserving its native
/// pixel type instead of forcing a cast to a caller-chosen `T`.
///
/// Where `read_geotiff::<T, _>` always returns `Raster<T>` — typically
/// `T = f64` when the caller doesn't know the file's type ahead of
/// time, which quadruples the memory footprint of a `u16` DEM for
/// callers that only need to inspect or pass the data through — this
/// lets the TIFF's own sample format pick the variant. Use
/// [`AnyRaster::to_f64`] to opt back into the old always-`f64`
/// behavior once you need to hand the raster to code that is
/// concretely `Raster<f64>`-typed (e.g. `algorithms`).
///
/// A native TIFF sample type without an exact `AnyRaster` variant
/// (`i8`, `u64`, `i64`, half-precision `f16`) is widened losslessly
/// rather than failing the read — see [`decode_geotiff_any`] for the
/// mapping.
pub fn read_geotiff_any<P: AsRef<Path>>(path: P, band: Option<usize>) -> Result<AnyRaster> {
    let file = File::open(path.as_ref())?;
    let len = file.metadata().ok().map(|m| m.len());
    decode_geotiff_any(file, band, len)
}

/// Read a GeoTIFF from an in-memory buffer into an [`AnyRaster`].
///
/// Same as [`read_geotiff_any`] but for buffers (WASM / network
/// fetches) — see there for the dtype-preservation rationale.
pub fn read_geotiff_any_from_buffer(data: &[u8], band: Option<usize>) -> Result<AnyRaster> {
    let cursor = Cursor::new(data);
    decode_geotiff_any(cursor, band, Some(data.len() as u64))
}

/// Read every band of a (multi-band) GeoTIFF as separate rasters.
///
/// Returns one [`Raster`] per sample/band, each carrying the file's
/// geotransform, CRS and nodata. For a single-band file this yields a
/// one-element vector. Bands are de-interleaved from the pixel-interleaved
/// TIFF buffer.
pub fn read_geotiff_bands<T, P>(path: P) -> Result<Vec<Raster<T>>>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref())?;
    let len = file.metadata().ok().map(|m| m.len());
    decode_geotiff_bands(file, len)
}

/// Read every band of a GeoTIFF from an in-memory buffer.
pub fn read_geotiff_bands_from_buffer<T>(data: &[u8]) -> Result<Vec<Raster<T>>>
where
    T: RasterElement,
{
    decode_geotiff_bands(Cursor::new(data), Some(data.len() as u64))
}

/// A decoded GeoTIFF: interleaved sample buffer plus geo-metadata, shared
/// by the single-band and multi-band entry points.
struct DecodedImage<T> {
    /// Pixel-interleaved samples (`spp` per pixel, row-major).
    data: Vec<T>,
    rows: usize,
    cols: usize,
    /// Samples per pixel (bands).
    spp: usize,
    transform: Option<GeoTransform>,
    crs: Option<crate::crs::CRS>,
    /// Raw GDAL_NODATA value (cast per band when building the raster).
    nodata: Option<f64>,
}

/// Raw decode result before any cast to a caller-chosen element type:
/// the TIFF's native sample buffer (whichever `DecodingResult` variant
/// the file actually stores) plus the shared geo-metadata.
///
/// Factored out of `decode_image` so it can be shared by two different
/// consumers: `decode_image` (casts every native type down to a single
/// fixed `T`) and `decode_geotiff_any` (dispatches to the matching
/// [`AnyRaster`] variant instead of casting at all).
struct RawDecoded {
    result: DecodingResult,
    rows: usize,
    cols: usize,
    /// Samples per pixel (bands).
    spp: usize,
    transform: Option<GeoTransform>,
    crs: Option<crate::crs::CRS>,
    /// Raw GDAL_NODATA value (cast per band when building the raster).
    nodata: Option<f64>,
}

/// Internal: open the TIFF decoder with fuzz-safe limits, read the geo
/// tags and the raw (native-type) pixel buffer. Does not know or care
/// what element type the caller eventually wants — see `decode_image`
/// and `decode_geotiff_any` for the two ways the result gets turned
/// into a `Raster`.
fn decode_raw<R>(reader: R, source_len: Option<u64>) -> Result<RawDecoded>
where
    R: std::io::Read + std::io::Seek,
{
    // A hostile TIFF can declare huge ImageWidth/ImageLength or tag counts;
    // `Limits::unlimited()` honours them with a single multi-GB allocation
    // (a 338-byte file forced ~32 GB — found by fuzzing). We keep the
    // limits far above any real DEM but bound them to the source size:
    // decoded pixels cannot legitimately exceed the input by more than the
    // best-case compression ratio. Floor at 1 GiB so uncompressed/large
    // DEMs (the reason `unlimited` was originally used) still read in one
    // shot; ceiling scales with the input so a tiny file can't demand GBs.
    const FLOOR: usize = 1024 * 1024 * 1024; // 1 GiB
    const RATIO: u64 = 4096; // generous decompression headroom
    let buf_cap = source_len
        .map(|n| (n.saturating_mul(RATIO)).min(usize::MAX as u64) as usize)
        .unwrap_or(usize::MAX)
        .max(FLOOR);
    let mut limits = Limits::unlimited();
    limits.decoding_buffer_size = buf_cap;
    limits.intermediate_buffer_size = buf_cap;
    // No legitimate GeoTIFF metadata tag exceeds a few MB.
    limits.ifd_value_size = (64 * 1024 * 1024).min(buf_cap);
    let mut decoder = Decoder::new(reader)
        .map_err(|e| Error::Other(format!("TIFF decode error: {}", e)))?
        .with_limits(limits);

    let (width, height) = decoder
        .dimensions()
        .map_err(|e| Error::Other(format!("Cannot read dimensions: {}", e)))?;

    let rows = height as usize;
    let cols = width as usize;
    // Samples per pixel (tag 277); absent or 0 means single band.
    let spp = decoder
        .get_tag_u32(Tag::SamplesPerPixel)
        .map(|v| v as usize)
        .unwrap_or(1)
        .max(1);

    // PlanarConfiguration (tag 284): 2 = separate planes (GDAL's
    // INTERLEAVE=BAND). The `tiff` crate's `read_image()` returns only the
    // first plane for planar multi-band TIFFs, so `spp` stays > 1 while the
    // buffer holds just `rows*cols` samples. The interleaved band selection
    // downstream (`buf[i*spp + b]`) would then index out of bounds and
    // panic. A planar TIFF is perfectly valid GDAL output, so reject it with
    // an actionable message rather than crashing or mis-sizing.
    if spp > 1
        && decoder
            .get_tag_u32(Tag::PlanarConfiguration)
            .map(|v| v == 2)
            .unwrap_or(false)
    {
        return Err(Error::Other(
            "planar (PlanarConfiguration=2, INTERLEAVE=BAND) multi-band TIFFs are not \
             supported; convert with `gdal_translate -co INTERLEAVE=PIXEL in.tif out.tif`"
                .to_string(),
        ));
    }

    // Read geo-tags before the pixel body: tag lookups are independent of
    // `read_image()` (the decoder parsed the IFD at construction), and
    // having `nodata` in hand lets the cast below fuse the nodata->NaN
    // normalization into the same pass instead of a second full-buffer
    // scan in `finish_raster`.
    let transform = read_geotransform(&mut decoder).ok();
    let crs = read_crs(&mut decoder);
    let nodata = read_nodata(&mut decoder);

    // Read image data (pixel-interleaved for multi-band TIFFs)
    let result = decoder
        .read_image()
        .map_err(|e| Error::Other(format!("Cannot read image data: {}", e)))?;

    Ok(RawDecoded {
        result,
        rows,
        cols,
        spp,
        transform,
        crs,
        nodata,
    })
}

/// Internal: decode a GeoTIFF (all bands, interleaved) plus its geo-tags.
fn decode_image<T, R>(reader: R, source_len: Option<u64>) -> Result<DecodedImage<T>>
where
    T: RasterElement,
    R: std::io::Read + std::io::Seek,
{
    let raw = decode_raw(reader, source_len)?;

    let data: Vec<T> = match raw.result {
        DecodingResult::F32(buf) => cast_and_normalize::<T, f32>(buf, raw.nodata),
        DecodingResult::F64(buf) => cast_and_normalize::<T, f64>(buf, raw.nodata),
        DecodingResult::U8(buf) => cast_and_normalize::<T, u8>(buf, raw.nodata),
        DecodingResult::U16(buf) => cast_and_normalize::<T, u16>(buf, raw.nodata),
        DecodingResult::U32(buf) => cast_and_normalize::<T, u32>(buf, raw.nodata),
        DecodingResult::I8(buf) => cast_and_normalize::<T, i8>(buf, raw.nodata),
        DecodingResult::I16(buf) => cast_and_normalize::<T, i16>(buf, raw.nodata),
        DecodingResult::I32(buf) => cast_and_normalize::<T, i32>(buf, raw.nodata),
        _ => {
            return Err(Error::UnsupportedDataType(
                "Unsupported TIFF pixel format".to_string(),
            ));
        }
    };

    if data.len() != raw.rows * raw.cols * raw.spp {
        return Err(Error::InvalidDimensions {
            width: raw.cols,
            height: raw.rows,
        });
    }

    Ok(DecodedImage {
        data,
        rows: raw.rows,
        cols: raw.cols,
        spp: raw.spp,
        transform: raw.transform,
        crs: raw.crs,
        nodata: raw.nodata,
    })
}

/// Internal: decode a GeoTIFF into an [`AnyRaster`], selecting one band
/// (0-based; default: first) — the dtype-preserving counterpart of
/// `decode_geotiff`.
///
/// Where `decode_geotiff`/`decode_image` cast every native TIFF sample
/// type down to a single caller-chosen `T`, this keeps the file's own
/// sample format and only widens the handful of native types without
/// an exact `AnyRaster` variant:
/// - `i8` → `I16` (exact: `i8`'s range is a strict subset of `i16`'s)
/// - `f16` → `F32` (exact: `f32` has strictly more range/precision)
/// - `u64`, `i64` → `F64` (lossless for any value up to 2^53 ≈ 9e15,
///   far beyond any realistic DEM/raster cell value; `f64` is the
///   widest numeric variant available so this is the best available
///   fallback rather than a failure)
fn decode_geotiff_any<R>(
    reader: R,
    band: Option<usize>,
    source_len: Option<u64>,
) -> Result<AnyRaster>
where
    R: std::io::Read + std::io::Seek,
{
    let raw = decode_raw(reader, source_len)?;
    let b = band.unwrap_or(0);
    if b >= raw.spp {
        return Err(Error::Other(format!(
            "band {b} out of range; image has {} band(s)",
            raw.spp
        )));
    }

    // Select band `b` out of an interleaved native-type buffer (or pass
    // it straight through on the spp==1 fast path), then wrap it as a
    // `Raster<T>` via the same metadata-attachment helper used by the
    // fixed-`T` path.
    macro_rules! band_raster {
        ($buf:expr) => {{
            let buf = $buf;
            let selected = if raw.spp == 1 {
                buf
            } else {
                let n = raw.rows * raw.cols;
                (0..n).map(|i| buf[i * raw.spp + b]).collect()
            };
            finish_raster(
                selected,
                raw.rows,
                raw.cols,
                raw.transform,
                raw.crs,
                raw.nodata,
            )?
        }};
    }

    Ok(match raw.result {
        DecodingResult::U8(buf) => AnyRaster::U8(band_raster!(buf)),
        DecodingResult::U16(buf) => AnyRaster::U16(band_raster!(buf)),
        DecodingResult::I16(buf) => AnyRaster::I16(band_raster!(buf)),
        DecodingResult::U32(buf) => AnyRaster::U32(band_raster!(buf)),
        DecodingResult::I32(buf) => AnyRaster::I32(band_raster!(buf)),
        DecodingResult::F32(buf) => AnyRaster::F32(band_raster!(buf)),
        DecodingResult::F64(buf) => AnyRaster::F64(band_raster!(buf)),
        // No exact `AnyRaster` variant for these native TIFF sample
        // types — widen losslessly rather than fail the read.
        DecodingResult::I8(buf) => {
            let widened: Vec<i16> = buf.iter().map(|&v| v as i16).collect();
            AnyRaster::I16(band_raster!(widened))
        }
        DecodingResult::F16(buf) => {
            let widened: Vec<f32> = buf.iter().map(|&v| v.to_f32()).collect();
            AnyRaster::F32(band_raster!(widened))
        }
        DecodingResult::U64(buf) => {
            let widened: Vec<f64> = buf.iter().map(|&v| v as f64).collect();
            AnyRaster::F64(band_raster!(widened))
        }
        DecodingResult::I64(buf) => {
            let widened: Vec<f64> = buf.iter().map(|&v| v as f64).collect();
            AnyRaster::F64(band_raster!(widened))
        }
    })
}

/// If `T` is exactly `U` at the type level, reinterprets `v: Vec<U>` as
/// `Vec<T>` at zero cost (a safe [`Any`] downcast — no `mem::transmute`).
/// Returns the original `Vec<U>` back in `Err` when the types differ, so
/// callers can fall through to the per-element cast.
fn same_type_vec<T: 'static, U: 'static>(v: Vec<U>) -> std::result::Result<Vec<T>, Vec<U>> {
    if TypeId::of::<T>() == TypeId::of::<U>() {
        let boxed: Box<dyn Any> = Box::new(v);
        match boxed.downcast::<Vec<T>>() {
            Ok(same) => Ok(*same),
            // TypeId matched so this can't actually happen, but recover
            // the original buffer rather than unwrap-panicking on it.
            Err(boxed) => Err(*boxed.downcast::<Vec<U>>().expect("TypeId checked above")),
        }
    } else {
        Err(v)
    }
}

/// Convert a decoded native-type sample buffer (`U`, one of the `tiff`
/// crate's `DecodingResult` element types) into the raster's requested
/// `T`, normalizing the GDAL_NODATA sentinel to `T`'s NaN convention
/// along the way.
///
/// Two passes become (at most) one:
/// - **Fast path** (`T == U`, e.g. reading a Float32 GeoTIFF into
///   `Raster<f32>`): the identity "cast" is skipped entirely via
///   [`same_type_vec`]; only a nodata normalization pass runs, and only
///   when nodata metadata is actually present.
/// - **Slow path** (`T != U`): the per-element `num_traits::cast` and the
///   nodata->NaN check happen in the same `map`, instead of casting here
///   and normalizing again in a second full-buffer pass in
///   `finish_raster`.
fn cast_and_normalize<T, U>(buf: Vec<U>, nodata: Option<f64>) -> Vec<T>
where
    T: RasterElement,
    U: RasterElement,
{
    match same_type_vec::<T, U>(buf) {
        Ok(mut same) => {
            if T::is_float()
                && let Some(nd_f64) = nodata
                && let Some(nd) = num_traits::cast::<f64, T>(nd_f64)
            {
                for v in same.iter_mut() {
                    if v.is_nodata(Some(nd)) {
                        *v = T::default_nodata();
                    }
                }
            }
            same
        }
        Err(buf) => {
            let nd_t: Option<T> = nodata.and_then(|nd| num_traits::cast::<f64, T>(nd));
            buf.iter()
                .map(|&v| {
                    let casted: T = num_traits::cast(v).unwrap_or(T::default_nodata());
                    if T::is_float()
                        && let Some(nd) = nd_t
                        && casted.is_nodata(Some(nd))
                    {
                        return T::default_nodata();
                    }
                    casted
                })
                .collect()
        }
    }
}

/// Build a single-band raster from owned data and the shared geo-metadata.
fn finish_raster<T: RasterElement>(
    data: Vec<T>,
    rows: usize,
    cols: usize,
    transform: Option<GeoTransform>,
    crs: Option<crate::crs::CRS>,
    nodata: Option<f64>,
) -> Result<Raster<T>> {
    let mut raster = Raster::from_vec(data, rows, cols)?;

    if let Some(t) = transform {
        raster.set_transform(t);
    }
    if let Some(crs) = crs {
        raster.set_crs(Some(crs));
    }
    // Cast GDAL_NODATA to T for the metadata sentinel. The pixel-level
    // nodata->NaN normalization itself already happened in
    // `cast_and_normalize` while `data` was being built (fused with the
    // type cast, or as the sole pass on the no-cast-needed fast path) —
    // this only sets the raster's `nodata` field to match, it does not
    // re-scan the buffer.
    if let Some(nodata_f64) = nodata
        && let Some(nd) = num_traits::cast::<f64, T>(nodata_f64)
    {
        if T::is_float() {
            // Pixels were already normalized to NaN, so the metadata must
            // say NaN too. Keeping the original sentinel here would make
            // a subsequent write emit GDAL_NODATA = <sentinel> over NaN
            // pixels — external tools (GDAL/QGIS) would then treat the
            // NaNs as valid data.
            raster.set_nodata(Some(T::default_nodata()));
        } else {
            raster.set_nodata(Some(nd));
        }
    }
    Ok(raster)
}

/// Extract one band (0-based) from a decoded image into its own raster.
fn select_band<T: RasterElement>(img: &DecodedImage<T>, band: usize) -> Result<Raster<T>> {
    let n = img.rows * img.cols;
    let data: Vec<T> = (0..n).map(|i| img.data[i * img.spp + band]).collect();
    finish_raster(
        data,
        img.rows,
        img.cols,
        img.transform,
        img.crs.clone(),
        img.nodata,
    )
}

/// Internal: decode a GeoTIFF, selecting one band (0-based; default: first).
fn decode_geotiff<T, R>(
    reader: R,
    band: Option<usize>,
    source_len: Option<u64>,
) -> Result<Raster<T>>
where
    T: RasterElement,
    R: std::io::Read + std::io::Seek,
{
    let img = decode_image::<T, R>(reader, source_len)?;
    let b = band.unwrap_or(0);
    if b >= img.spp {
        return Err(Error::Other(format!(
            "band {b} out of range; image has {} band(s)",
            img.spp
        )));
    }
    if img.spp == 1 {
        // Fast path: no de-interleave, move the buffer straight through.
        finish_raster(
            img.data,
            img.rows,
            img.cols,
            img.transform,
            img.crs,
            img.nodata,
        )
    } else {
        select_band(&img, b)
    }
}

/// Internal: decode every band of a GeoTIFF as separate rasters.
fn decode_geotiff_bands<T, R>(reader: R, source_len: Option<u64>) -> Result<Vec<Raster<T>>>
where
    T: RasterElement,
    R: std::io::Read + std::io::Seek,
{
    let img = decode_image::<T, R>(reader, source_len)?;
    (0..img.spp).map(|b| select_band(&img, b)).collect()
}

/// Attempt to read CRS EPSG code from GeoKeyDirectory tag
fn read_crs<R: std::io::Read + std::io::Seek>(decoder: &mut Decoder<R>) -> Option<crate::crs::CRS> {
    let geokeys = decoder.get_tag_u16_vec(Tag::Unknown(34735)).ok()?;
    if geokeys.len() < 4 {
        return None;
    }
    let num_keys = geokeys[3] as usize;
    for i in 0..num_keys {
        let base = 4 + i * 4;
        if base + 3 >= geokeys.len() {
            break;
        }
        let key_id = geokeys[base];
        let value = geokeys[base + 3];
        // ProjectedCSTypeGeoKey (3072) or GeographicTypeGeoKey (2048)
        if (key_id == 3072 || key_id == 2048) && value > 0 {
            return Some(crate::crs::CRS::from_epsg(value as u32));
        }
    }
    None
}

/// Attempt to read GeoTransform from TIFF tags
fn read_geotransform<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> Result<GeoTransform> {
    // ModelPixelScaleTag = 33550
    // ModelTiepointTag = 33922
    // ModelTransformationTag = 34264 (alternative)

    // Try ModelPixelScaleTag + ModelTiepointTag first
    let scale_tag = Tag::Unknown(33550);
    let tiepoint_tag = Tag::Unknown(33922);

    let scale = decoder
        .get_tag_f64_vec(scale_tag)
        .map_err(|_| Error::Other("No pixel scale tag".into()))?;

    let tiepoint = decoder
        .get_tag_f64_vec(tiepoint_tag)
        .map_err(|_| Error::Other("No tiepoint tag".into()))?;

    if scale.len() >= 2 && tiepoint.len() >= 6 {
        // tiepoint: [I, J, K, X, Y, Z]
        // scale: [ScaleX, ScaleY, ScaleZ]
        let origin_x = tiepoint[3] - tiepoint[0] * scale[0];
        let origin_y = tiepoint[4] + tiepoint[1] * scale[1];
        let pixel_width = scale[0];
        let pixel_height = -scale[1]; // Negative for north-up

        return Ok(GeoTransform::new(
            origin_x,
            origin_y,
            pixel_width,
            pixel_height,
        ));
    }

    Err(Error::Other("Cannot determine geotransform".into()))
}

/// Read GDAL_NODATA tag (42113) — stored as ASCII string, parsed to f64.
fn read_nodata<R: std::io::Read + std::io::Seek>(decoder: &mut Decoder<R>) -> Option<f64> {
    let s = match decoder.get_tag_ascii_string(Tag::Unknown(42113)) {
        Ok(s) => s,
        // Fallback: SurtGIS <= 0.16.3 wrote this tag via as_bytes(),
        // producing type BYTE, which get_tag_ascii_string rejects.
        Err(_) => {
            let bytes = decoder.get_tag_u8_vec(Tag::Unknown(42113)).ok()?;
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };
    s.trim().trim_end_matches('\0').parse::<f64>().ok()
}

/// Maps a [`RasterElement`] to the native `tiff`-crate `ColorType` that
/// stores it losslessly. The writer used to always cast to `f32`
/// (`Gray32Float`) regardless of `T`, so a `u16` DEM or a `u8` mask was
/// written 2-4x larger than necessary and lost its integer sample
/// semantics (`SampleFormat` claimed `IEEEFP` for integer data).
///
/// `tiff` 0.10's `colortype` module happens to expose an exact
/// single-sample grayscale type for every numeric type
/// [`RasterElement`] is implemented for (`u8`, `i8`, `u16`, `i16`,
/// `u32`, `i32`, `u64`, `i64`, `f32`, `f64` -- see
/// `crates/core/src/raster/element.rs`), so every impl below is a
/// lossless 1:1 mapping; no fallback casting to a wider/narrower type
/// is needed for any of the ten.
pub trait NativeGraySample: RasterElement
where
    [Self]: tiff::encoder::TiffValue,
{
    /// The `tiff` colortype with `Inner = Self`.
    type Gray: ColorType<Inner = Self>;

    /// Append this value's native-endian bytes (matching how the `tiff`
    /// crate itself serializes samples -- see `encoder::writer::TiffWriter`,
    /// which always writes `to_ne_bytes()`) to `buf`. Used by
    /// `write_geotiff_stack`, which drives the low-level
    /// `DirectoryEncoder` API directly and therefore has to assemble the
    /// strip bytes (and, when compressing, deflate them) itself instead
    /// of going through `ImageEncoder`.
    fn write_ne_bytes(self, buf: &mut Vec<u8>);
}

macro_rules! impl_native_gray_sample {
    ($t:ty, $ct:ty) => {
        impl NativeGraySample for $t {
            type Gray = $ct;

            fn write_ne_bytes(self, buf: &mut Vec<u8>) {
                buf.extend_from_slice(&self.to_ne_bytes());
            }
        }
    };
}

impl_native_gray_sample!(u8, Gray8);
impl_native_gray_sample!(i8, GrayI8);
impl_native_gray_sample!(u16, Gray16);
impl_native_gray_sample!(i16, GrayI16);
impl_native_gray_sample!(u32, Gray32);
impl_native_gray_sample!(i32, GrayI32);
impl_native_gray_sample!(u64, Gray64);
impl_native_gray_sample!(i64, GrayI64);
impl_native_gray_sample!(f32, Gray32Float);
impl_native_gray_sample!(f64, Gray64Float);

/// Resolve `options.compression` into a concrete `tiff::encoder::Compression`.
///
/// Previously any non-`"NONE"` string silently fell back to DEFLATE --
/// requesting `"LZW"` or `"ZSTD"` (both valid for the `gdal` backend)
/// produced a DEFLATE-compressed file with no indication the requested
/// codec was ignored. Now only `"NONE"`/`"DEFLATE"` (case-insensitive)
/// resolve; anything else the native backend can't actually produce is
/// a hard error naming the unsupported codec.
fn resolve_compression(options: Option<&GeoTiffOptions>) -> Result<Compression> {
    let Some(options) = options else {
        return Ok(Compression::Uncompressed);
    };
    match options.compression.to_lowercase().as_str() {
        "none" | "" => Ok(Compression::Uncompressed),
        "deflate" => Ok(Compression::Deflate(DeflateLevel::Balanced)),
        other => Err(Error::Other(format!(
            "compression '{other}' is not supported by the native GeoTIFF writer (only NONE/DEFLATE); use the gdal feature for other codecs"
        ))),
    }
}

/// Estimated uncompressed on-disk size of a GeoTIFF, for the BigTIFF
/// heads-up in [`warn_if_bigtiff_recommended`]. Compression may shrink
/// the real file, but strip/tile *offsets* -- the thing that actually
/// overflows in a classic (non-Big) TIFF -- are driven by the
/// uncompressed layout geometry, not the compressed size, so this stays
/// a meaningful (if conservative) trigger either way.
fn estimate_geotiff_bytes(
    rows: usize,
    cols: usize,
    n_bands: usize,
    bytes_per_sample: usize,
) -> u64 {
    (rows as u64) * (cols as u64) * (n_bands as u64) * (bytes_per_sample as u64)
}

/// Comfortably under the 4 GiB (2^32 byte) classic-TIFF offset ceiling,
/// so the warning fires with headroom to actually switch to BigTIFF
/// before hitting the real limit.
const BIGTIFF_WARN_THRESHOLD_BYTES: u64 = 3_800_000_000;

/// Emit a heads-up when writing a large classic (non-Big) TIFF.
///
/// The `tiff` crate itself already returns a clear `TryFromIntError`-
/// wrapped error if a strip offset genuinely overflows `u32::MAX` (see
/// `TiffKindStandard::convert_offset`) -- this warning fires earlier, at
/// the point the file is *likely* to approach that ceiling, so callers
/// get a chance to opt into BigTIFF instead of hitting the failure only
/// after most of the (potentially expensive) write has happened.
fn warn_if_bigtiff_recommended(bigtiff: bool, estimated_bytes: u64) {
    if !bigtiff && estimated_bytes > BIGTIFF_WARN_THRESHOLD_BYTES {
        eprintln!(
            "surtgis: WARNING - estimated GeoTIFF size is {:.2} GiB, close to the 4 GiB \
             classic-TIFF limit. Pass GeoTiffOptions {{ bigtiff: true, .. }} to write BigTIFF \
             instead of risking a write failure once strip offsets exceed u32::MAX.",
            estimated_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
}

/// Write the GeoTIFF geo-referencing tags (pixel scale, tiepoint,
/// geokeys, nodata) shared by every native write path. Factored out of
/// the old per-function copies so `write_geotiff_stack` (which drives
/// `DirectoryEncoder` directly rather than through `ImageEncoder`) can
/// reuse the same tag payloads without duplicating the CRS -> GeoKey
/// logic a third time — and `pub(crate)` so `strip_writer.rs`'s
/// compressed streaming path is the fourth.
pub(crate) fn write_geo_metadata_tags<W, K>(
    dir: &mut DirectoryEncoder<'_, W, K>,
    transform: &GeoTransform,
    crs: Option<&crate::crs::CRS>,
    nodata_f64: Option<f64>,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    K: TiffKind,
{
    // ModelPixelScaleTag
    let scale = vec![transform.pixel_width, transform.pixel_height.abs(), 0.0];
    dir.write_tag(Tag::Unknown(33550), scale.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write scale tag: {}", e)))?;

    // ModelTiepointTag
    let tiepoint = vec![0.0, 0.0, 0.0, transform.origin_x, transform.origin_y, 0.0];
    dir.write_tag(Tag::Unknown(33922), tiepoint.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write tiepoint tag: {}", e)))?;

    // GeoKeyDirectoryTag (34735)
    let geokeys = geokeys_for_crs(crs);
    dir.write_tag(Tag::Unknown(34735), geokeys.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write geokey tag: {}", e)))?;

    // GDAL_NODATA tag (42113) — write as ASCII string
    if let Some(nd_f64) = nodata_f64 {
        let nodata_str = if nd_f64.is_nan() {
            "nan".to_string()
        } else {
            format!("{}", nd_f64)
        };
        // Write as a proper ASCII tag (the str impl appends the NUL);
        // writing via as_bytes() produced a BYTE-typed tag that
        // get_tag_ascii_string rejects on read-back.
        dir.write_tag(Tag::Unknown(42113), nodata_str.as_str())
            .map_err(|e| Error::Other(format!("Cannot write nodata tag: {}", e)))?;
    }

    Ok(())
}

/// GeoKeyDirectoryTag (34735) payload for a raster's CRS.
///
/// GeoKey structure: `[KeyDirectoryVersion, KeyRevision, MinorRevision,
/// NumberOfKeys, KeyID, TIFFTagLocation, Count, Value_Offset, ...]`.
fn geokeys_for_crs(crs: Option<&crate::crs::CRS>) -> Vec<u16> {
    if let Some(crs) = crs {
        if let Some(epsg) = crs.epsg() {
            if epsg == 4326 {
                // Geographic CRS (WGS84)
                vec![
                    1,
                    1,
                    0,
                    3, // Version 1.1.0, 3 keys
                    1024,
                    0,
                    1,
                    2, // GTModelTypeGeoKey = ModelTypeGeographic
                    1025,
                    0,
                    1,
                    1, // GTRasterTypeGeoKey = RasterPixelIsArea
                    2048,
                    0,
                    1,
                    epsg as u16, // GeographicTypeGeoKey = EPSG code
                ]
            } else {
                // Projected CRS (e.g., UTM zones EPSG:326xx/327xx)
                vec![
                    1,
                    1,
                    0,
                    3, // Version 1.1.0, 3 keys
                    1024,
                    0,
                    1,
                    1, // GTModelTypeGeoKey = ModelTypeProjected
                    1025,
                    0,
                    1,
                    1, // GTRasterTypeGeoKey = RasterPixelIsArea
                    3072,
                    0,
                    1,
                    epsg as u16, // ProjectedCSTypeGeoKey = EPSG code
                ]
            }
        } else {
            // CRS without EPSG code — write generic projected
            vec![
                1, 1, 0, 2, 1024, 0, 1, 1, // ModelTypeProjected
                1025, 0, 1, 1, // RasterPixelIsArea
            ]
        }
    } else {
        // No CRS — write generic projected (backward compatible)
        vec![1, 1, 0, 2, 1024, 0, 1, 1, 1025, 0, 1, 1]
    }
}

/// Write a Raster to a GeoTIFF file
///
/// Native writer with limited GeoTIFF metadata support. Writes using
/// `T`'s own native TIFF sample type (`u8` -> Gray8, `u16` -> Gray16,
/// `f32` -> Gray32Float, etc. -- see [`NativeGraySample`]), not always
/// Float32. For full support, enable the `gdal` feature.
pub fn write_geotiff<T, P>(
    raster: &Raster<T>,
    path: P,
    options: Option<GeoTiffOptions>,
) -> Result<()>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
    P: AsRef<Path>,
{
    // Write to temp file first, then atomic rename to prevent corrupt partial files
    let final_path = path.as_ref();
    let tmp_path = final_path.with_extension("tmp");
    let file = File::create(&tmp_path)?;
    encode_geotiff(raster, file, options.as_ref())?;
    std::fs::rename(&tmp_path, final_path)?;
    Ok(())
}

/// Write a Raster to an in-memory GeoTIFF buffer
///
/// Same as `write_geotiff` but returns a `Vec<u8>` instead of writing to a file.
/// Useful for WASM environments where filesystem access is not available.
pub fn write_geotiff_to_buffer<T>(
    raster: &Raster<T>,
    options: Option<GeoTiffOptions>,
) -> Result<Vec<u8>>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
{
    let mut buf = Vec::new();
    encode_geotiff(raster, Cursor::new(&mut buf), options.as_ref())?;
    Ok(buf)
}

/// Internal: encode a Raster as GeoTIFF into any `Write + Seek` sink,
/// using `T`'s native TIFF sample type and honouring `options.bigtiff`.
fn encode_geotiff<T, W>(
    raster: &Raster<T>,
    writer: W,
    options: Option<&GeoTiffOptions>,
) -> Result<()>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
    W: std::io::Write + std::io::Seek,
{
    let compression = resolve_compression(options)?;
    let bigtiff = options.map(|o| o.bigtiff).unwrap_or(false);
    let (rows, cols) = raster.shape();
    warn_if_bigtiff_recommended(
        bigtiff,
        estimate_geotiff_bytes(rows, cols, 1, std::mem::size_of::<T>()),
    );

    if bigtiff {
        let mut encoder = TiffEncoder::new_big(writer)
            .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?
            .with_compression(compression);
        let image = encoder
            .new_image::<T::Gray>(cols as u32, rows as u32)
            .map_err(|e| Error::Other(format!("Cannot create TIFF image: {}", e)))?;
        write_single_band_image(image, raster)
    } else {
        let mut encoder = TiffEncoder::new(writer)
            .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?
            .with_compression(compression);
        let image = encoder
            .new_image::<T::Gray>(cols as u32, rows as u32)
            .map_err(|e| Error::Other(format!("Cannot create TIFF image: {}", e)))?;
        write_single_band_image(image, raster)
    }
}

/// Write geo-tags + pixel data into an already-created single-band
/// `ImageEncoder`, generic over both TIFF/BigTIFF (`K`) so the two
/// branches of `encode_geotiff` share one code path.
fn write_single_band_image<T, W, K>(
    mut image: tiff::encoder::ImageEncoder<'_, W, T::Gray, K>,
    raster: &Raster<T>,
) -> Result<()>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
    W: std::io::Write + std::io::Seek,
    K: TiffKind,
{
    write_geo_metadata_tags(
        image.encoder(),
        raster.transform(),
        raster.crs(),
        raster.nodata().and_then(|nd| nd.to_f64()),
    )?;

    // No cast needed: T::Gray::Inner == T, so the raw sample buffer is
    // written verbatim instead of through the old lossy f32 conversion.
    let data: Vec<T> = raster.data().iter().copied().collect();
    image
        .write_data(&data)
        .map_err(|e| Error::Other(format!("Cannot write image data: {}", e)))?;

    Ok(())
}

/// Write a stack of single-band rasters as a multi-band GeoTIFF.
///
/// Supports 1, 3 or 4 bands (`Gray32Float`, `RGB32Float`,
/// `RGBA32Float`). All band rasters must share the same shape and
/// geotransform. GeoTIFF metadata (CRS, transform, nodata) is
/// inherited from `bands[0]`. Per-band values are cast to `f32`.
///
/// For arbitrary `N > 4`, enable the `gdal` feature and use
/// `gdal_io::write_geotiff_multiband`.
pub fn write_geotiff_multiband<T, P>(
    bands: &[&Raster<T>],
    path: P,
    options: Option<GeoTiffOptions>,
) -> Result<()>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    if bands.is_empty() {
        return Err(Error::Other("stack needs at least one band".into()));
    }
    let (rows, cols) = bands[0].shape();
    for b in bands.iter().skip(1) {
        if b.shape() != (rows, cols) {
            return Err(Error::Other("stack bands must share shape".into()));
        }
    }
    let n_bands = bands.len();
    if !matches!(n_bands, 1 | 3 | 4) {
        return Err(Error::Other(format!(
            "native stack supports 1, 3 or 4 bands; got {} (use --features gdal for N>4)",
            n_bands
        )));
    }

    // Explicit resolve: a compression string the native backend can't
    // actually produce (e.g. "LZW"/"ZSTD", valid for the `gdal` backend)
    // is now a hard error instead of a silent fallback to DEFLATE.
    let compression = resolve_compression(options.as_ref())?;

    let final_path = path.as_ref();
    let tmp_path = final_path.with_extension("tmp");
    let file = File::create(&tmp_path)?;

    // Build the interleaved (chunky) buffer expected by the tiff
    // crate for pixel-interleaved multi-sample images:
    //   [s0_px0, s1_px0, ..., sK_px0,
    //    s0_px1, s1_px1, ..., sK_px1, ...]
    let n_px = rows * cols;
    let mut interleaved: Vec<f32> = vec![0.0; n_px * n_bands];
    for (b, raster) in bands.iter().enumerate() {
        for (i, &v) in raster.data().iter().enumerate() {
            let f: f32 = num_traits::cast(v).unwrap_or(f32::NAN);
            interleaved[i * n_bands + b] = f;
        }
    }

    match n_bands {
        1 => encode_multiband_image::<Gray32Float, _>(file, bands[0], &interleaved, compression)?,
        3 => encode_multiband_image::<RGB32Float, _>(file, bands[0], &interleaved, compression)?,
        4 => encode_multiband_image::<RGBA32Float, _>(file, bands[0], &interleaved, compression)?,
        _ => unreachable!(),
    }
    std::fs::rename(&tmp_path, final_path)?;
    Ok(())
}

/// Generic multi-band writer parameterised over the tiff
/// `ColorType` so the same GeoTIFF-tag plumbing serves Gray /
/// RGB / RGBA. `meta` provides geotransform, CRS and nodata;
/// `interleaved` is the chunky f32 buffer.
fn encode_multiband_image<CT, W>(
    writer: W,
    meta: &Raster<impl RasterElement>,
    interleaved: &[f32],
    compression: Compression,
) -> Result<()>
where
    CT: ColorType<Inner = f32>,
    W: std::io::Write + std::io::Seek,
{
    let mut encoder = TiffEncoder::new(writer)
        .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?
        .with_compression(compression);

    let (rows, cols) = meta.shape();
    let mut image = encoder
        .new_image::<CT>(cols as u32, rows as u32)
        .map_err(|e| Error::Other(format!("Cannot create TIFF image: {}", e)))?;

    let gt = meta.transform();
    let scale = vec![gt.pixel_width, gt.pixel_height.abs(), 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33550), scale.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write scale tag: {}", e)))?;
    let tiepoint = vec![0.0, 0.0, 0.0, gt.origin_x, gt.origin_y, 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33922), tiepoint.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write tiepoint tag: {}", e)))?;

    let geokeys: Vec<u16> = if let Some(crs) = meta.crs() {
        if let Some(epsg) = crs.epsg() {
            if epsg == 4326 {
                vec![
                    1,
                    1,
                    0,
                    3,
                    1024,
                    0,
                    1,
                    2,
                    1025,
                    0,
                    1,
                    1,
                    2048,
                    0,
                    1,
                    epsg as u16,
                ]
            } else {
                vec![
                    1,
                    1,
                    0,
                    3,
                    1024,
                    0,
                    1,
                    1,
                    1025,
                    0,
                    1,
                    1,
                    3072,
                    0,
                    1,
                    epsg as u16,
                ]
            }
        } else {
            vec![1, 1, 0, 2, 1024, 0, 1, 1, 1025, 0, 1, 1]
        }
    } else {
        vec![1, 1, 0, 2, 1024, 0, 1, 1, 1025, 0, 1, 1]
    };
    image
        .encoder()
        .write_tag(Tag::Unknown(34735), geokeys.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write geokey tag: {}", e)))?;

    if let Some(nd) = meta.nodata()
        && let Some(nd_f64) = nd.to_f64()
    {
        let nodata_str = if nd_f64.is_nan() {
            "nan".to_string()
        } else {
            format!("{}", nd_f64)
        };
        // ASCII tag; the str impl appends the NUL (see write_geotiff).
        image
            .encoder()
            .write_tag(Tag::Unknown(42113), nodata_str.as_str())
            .map_err(|e| Error::Other(format!("Cannot write nodata tag: {}", e)))?;
    }

    image
        .write_data(interleaved)
        .map_err(|e| Error::Other(format!("Cannot write image data: {}", e)))?;
    Ok(())
}

/// Write an arbitrary number (1..=N, not just 1/3/4) of single-band
/// rasters as one multi-band GeoTIFF, using `T`'s *native* TIFF sample
/// type (not always Float32 like [`write_geotiff_multiband`]) and
/// `PhotometricInterpretation = BlackIsZero` — never RGB/RGBA, which the
/// engine audit flagged as misleading for stacks of unrelated bands
/// (e.g. spectral indices) that happen to number 3 or 4.
///
/// If `names` is provided, band descriptions are written to the
/// GDAL_METADATA tag (42112) as `<Item name="DESCRIPTION" sample="i">`
/// entries, readable by GDAL/rasterio as per-band descriptions.
///
/// # Implementation note
///
/// `tiff`'s `ImageEncoder`/`ColorType` API requires the sample layout
/// (`BITS_PER_SAMPLE`, `SAMPLE_FORMAT`) to be a `const` fixed at compile
/// time, which is incompatible with `bands.len()` only being known at
/// runtime. This function instead drives the lower-level
/// `DirectoryEncoder` API directly and writes the tags by hand.
///
/// That lower-level API does not run the crate's built-in compressors
/// (`Compression::Deflate`'s zlib pass lives behind a private
/// `TiffWriter` field that only `ImageEncoder`, in the same module, can
/// reach) — so when DEFLATE is requested this function performs its own
/// zlib/DEFLATE pass via `flate2`, using the exact same "Adobe Deflate"
/// wrapper (`ZlibEncoder`) the `tiff` crate's own `Deflate` compressor
/// uses internally, and writes the resulting bytes as one raw strip.
pub fn write_geotiff_stack<T, P>(
    bands: &[&Raster<T>],
    names: Option<&[&str]>,
    path: P,
    options: &GeoTiffOptions,
) -> Result<()>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
    P: AsRef<Path>,
{
    if bands.is_empty() {
        return Err(Error::Other("stack needs at least one band".into()));
    }
    let (rows, cols) = bands[0].shape();
    for b in bands.iter().skip(1) {
        if b.shape() != (rows, cols) {
            return Err(Error::Other("stack bands must share shape".into()));
        }
    }
    if let Some(names) = names
        && names.len() != bands.len()
    {
        return Err(Error::Other(format!(
            "names length ({}) must match band count ({})",
            names.len(),
            bands.len()
        )));
    }

    let compression = resolve_compression(Some(options))?;
    let n_bands = bands.len();
    let n_px = rows * cols;

    // Build the interleaved (chunky) buffer, native dtype (no f32 cast):
    //   [b0_px0, b1_px0, ..., bK_px0, b0_px1, b1_px1, ..., bK_px1, ...]
    let mut interleaved: Vec<T> = vec![T::zero(); n_px * n_bands];
    for (b, raster) in bands.iter().enumerate() {
        for (i, &v) in raster.data().iter().enumerate() {
            interleaved[i * n_bands + b] = v;
        }
    }

    warn_if_bigtiff_recommended(
        options.bigtiff,
        estimate_geotiff_bytes(rows, cols, n_bands, std::mem::size_of::<T>()),
    );

    let final_path = path.as_ref();
    let tmp_path = final_path.with_extension("tmp");
    let file = File::create(&tmp_path)?;

    if options.bigtiff {
        let mut encoder = TiffEncoder::new_big(file)
            .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?;
        write_stack_ifd::<T, _, TiffKindBig>(
            &mut encoder,
            rows,
            cols,
            n_bands,
            &interleaved,
            compression,
            bands[0],
            names,
        )?;
    } else {
        let mut encoder = TiffEncoder::new(file)
            .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?;
        write_stack_ifd::<T, _, TiffKindStandard>(
            &mut encoder,
            rows,
            cols,
            n_bands,
            &interleaved,
            compression,
            bands[0],
            names,
        )?;
    }

    std::fs::rename(&tmp_path, final_path)?;
    Ok(())
}

/// Flatten `data` into its native-endian byte representation, matching
/// exactly what the `tiff` crate itself writes for a `[T]` sample
/// buffer (see `encoder::writer::TiffWriter`, which always serializes
/// via `to_ne_bytes()`).
pub(crate) fn flatten_native_bytes<T: NativeGraySample>(data: &[T]) -> Vec<u8>
where
    [T]: tiff::encoder::TiffValue,
{
    let mut buf = Vec::with_capacity(data.len() * std::mem::size_of::<T>());
    for &v in data {
        v.write_ne_bytes(&mut buf);
    }
    buf
}

/// Zlib/DEFLATE-compress `data`, using the same wrapper (`flate2`'s
/// `ZlibEncoder`) and level mapping the `tiff` crate's own `Deflate`
/// compressor uses (`encoder::compression::deflate`), so the resulting
/// stream is the same "Adobe Deflate" TIFF readers expect.
pub(crate) fn deflate_compress_bytes(data: &[u8], level: DeflateLevel) -> Result<Vec<u8>> {
    use std::io::Write as _;
    let mut encoder =
        flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(level as u32));
    encoder
        .write_all(data)
        .map_err(|e| Error::Other(format!("DEFLATE compression failed: {}", e)))?;
    encoder
        .finish()
        .map_err(|e| Error::Other(format!("DEFLATE compression failed: {}", e)))
}

/// Escape the handful of characters that are meaningful in XML text
/// content/attribute values.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Build the GDAL_METADATA (tag 42112) XML payload carrying per-band
/// descriptions, in the format GDAL's GTiff driver writes/reads:
/// `<Item name="DESCRIPTION" sample="i">...</Item>` per band.
fn gdal_metadata_xml(names: &[&str]) -> String {
    let mut xml = String::from("<GDALMetadata>");
    for (i, name) in names.iter().enumerate() {
        xml.push_str(&format!(
            "<Item name=\"DESCRIPTION\" sample=\"{}\" role=\"description\">{}</Item>",
            i,
            xml_escape(name)
        ));
    }
    xml.push_str("</GDALMetadata>");
    xml
}

/// Low-level IFD writer behind [`write_geotiff_stack`]: writes every
/// tag by hand via `DirectoryEncoder` (rather than `ImageEncoder`)
/// because the sample layout (band count) is only known at runtime.
/// Generic over `K` so the same code serves both classic TIFF and
/// BigTIFF.
fn write_stack_ifd<T, W, K>(
    encoder: &mut TiffEncoder<W, K>,
    rows: usize,
    cols: usize,
    n_bands: usize,
    interleaved: &[T],
    compression: Compression,
    meta: &Raster<T>,
    names: Option<&[&str]>,
) -> Result<()>
where
    T: RasterElement + NativeGraySample,
    [T]: tiff::encoder::TiffValue,
    W: std::io::Write + std::io::Seek,
    K: TiffKind,
{
    let mut dir = encoder
        .image_directory()
        .map_err(|e| Error::Other(format!("Cannot create TIFF directory: {}", e)))?;

    let bits_per_sample: Vec<u16> = vec![<T::Gray as ColorType>::BITS_PER_SAMPLE[0]; n_bands];
    let sample_format: Vec<u16> = vec![<T::Gray as ColorType>::SAMPLE_FORMAT[0].to_u16(); n_bands];

    dir.write_tag(Tag::ImageWidth, cols as u32)
        .map_err(|e| Error::Other(format!("Cannot write ImageWidth: {}", e)))?;
    dir.write_tag(Tag::ImageLength, rows as u32)
        .map_err(|e| Error::Other(format!("Cannot write ImageLength: {}", e)))?;
    dir.write_tag(Tag::BitsPerSample, bits_per_sample.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write BitsPerSample: {}", e)))?;
    dir.write_tag(Tag::SampleFormat, sample_format.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write SampleFormat: {}", e)))?;
    // Never RGB/RGBA: an N-band stack of arbitrary (possibly unrelated)
    // bands should not imply a color interpretation.
    dir.write_tag(
        Tag::PhotometricInterpretation,
        PhotometricInterpretation::BlackIsZero.to_u16(),
    )
    .map_err(|e| Error::Other(format!("Cannot write PhotometricInterpretation: {}", e)))?;
    dir.write_tag(Tag::SamplesPerPixel, n_bands as u16)
        .map_err(|e| Error::Other(format!("Cannot write SamplesPerPixel: {}", e)))?;
    if n_bands > 1 {
        // PhotometricInterpretation::BlackIsZero only accounts for one
        // "color" channel; without ExtraSamples, libtiff/GDAL emit a
        // (harmless, but noisy) "Sum of Photometric type-related color
        // channels and ExtraSamples doesn't match SamplesPerPixel"
        // warning for every band beyond the first. Declaring the rest
        // as "0 = Unspecified data" (not alpha) is the correct tag for
        // an arbitrary stack of unrelated bands and silences it.
        let extra_samples: Vec<u16> = vec![0; n_bands - 1];
        dir.write_tag(Tag::ExtraSamples, extra_samples.as_slice())
            .map_err(|e| Error::Other(format!("Cannot write ExtraSamples: {}", e)))?;
    }
    dir.write_tag(Tag::RowsPerStrip, rows as u32)
        .map_err(|e| Error::Other(format!("Cannot write RowsPerStrip: {}", e)))?;

    // Single strip covering the whole image — simple and spec-legal;
    // `write_geotiff_streaming` (strip_writer.rs) is the path for
    // memory-bounded chunked writes.
    let raw_bytes = flatten_native_bytes(interleaved);
    let (compression_code, strip_bytes) = match compression {
        Compression::Uncompressed => (CompressionMethod::None.to_u16(), raw_bytes),
        Compression::Deflate(level) => (
            CompressionMethod::Deflate.to_u16(),
            deflate_compress_bytes(&raw_bytes, level)?,
        ),
        // `resolve_compression` only ever returns Uncompressed or
        // Deflate for the native backend (see its doc comment).
        _ => unreachable!("resolve_compression only returns Uncompressed/Deflate"),
    };
    dir.write_tag(Tag::Compression, compression_code)
        .map_err(|e| Error::Other(format!("Cannot write Compression: {}", e)))?;

    let offset = dir
        .write_data(strip_bytes.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write strip data: {}", e)))?;
    let strip_offset = K::convert_offset(offset)
        .map_err(|e| Error::Other(format!("Strip offset overflow: {}", e)))?;
    let strip_byte_count = K::convert_offset(strip_bytes.len() as u64)
        .map_err(|e| Error::Other(format!("Strip byte count overflow: {}", e)))?;
    dir.write_tag(Tag::StripOffsets, strip_offset)
        .map_err(|e| Error::Other(format!("Cannot write StripOffsets: {}", e)))?;
    dir.write_tag(Tag::StripByteCounts, strip_byte_count)
        .map_err(|e| Error::Other(format!("Cannot write StripByteCounts: {}", e)))?;

    write_geo_metadata_tags(
        &mut dir,
        meta.transform(),
        meta.crs(),
        meta.nodata().and_then(|nd| nd.to_f64()),
    )?;

    if let Some(names) = names {
        let xml = gdal_metadata_xml(names);
        dir.write_tag(Tag::Unknown(42112), xml.as_str())
            .map_err(|e| Error::Other(format!("Cannot write GDAL_METADATA tag: {}", e)))?;
    }

    dir.finish()
        .map_err(|e| Error::Other(format!("Cannot finish TIFF directory: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::{DataType, Raster};
    use tempfile::TempDir;

    fn ramp_band(rows: usize, cols: usize, base: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, base + (row + col) as f64).unwrap();
            }
        }
        r
    }

    /// Regression: a file with a sentinel GDAL_NODATA (e.g. -9999) has its
    /// pixels normalized to NaN on read, so the in-memory metadata must be
    /// NaN too. Otherwise a subsequent write emits GDAL_NODATA=-9999 over
    /// NaN pixels and external tools treat the NaNs as valid data.
    #[test]
    fn sentinel_nodata_roundtrip_stays_coherent() {
        let mut r = ramp_band(6, 6, 0.0);
        // Write file 1 with a sentinel nodata and one nodata pixel
        r.set(2, 3, -9999.0).unwrap();
        r.set_nodata(Some(-9999.0));
        let dir = TempDir::new().unwrap();
        let p1 = dir.path().join("sentinel.tif");
        write_geotiff(&r, &p1, None).unwrap();

        // Read: pixel must be NaN AND metadata must be NaN
        let back: Raster<f64> = read_geotiff(&p1, None).unwrap();
        assert!(back.get(2, 3).unwrap().is_nan(), "pixel not normalized");
        let nd = back.nodata().expect("nodata metadata lost");
        assert!(
            nd.is_nan(),
            "metadata nodata should be NaN after normalization, got {}",
            nd
        );

        // Write file 2 and re-read: still coherent (GDAL_NODATA=nan)
        let p2 = dir.path().join("rewritten.tif");
        write_geotiff(&back, &p2, None).unwrap();
        let again: Raster<f64> = read_geotiff(&p2, None).unwrap();
        assert!(again.get(2, 3).unwrap().is_nan());
        assert!(again.nodata().expect("nodata lost on rewrite").is_nan());
        // Valid data untouched
        assert_eq!(again.get(0, 0).unwrap(), 0.0);
    }

    #[test]
    fn multiband_rgb_roundtrip() {
        // Write three single-band rasters as an RGB stack, read it
        // back, verify per-band values match.
        let r0 = ramp_band(10, 12, 0.0);
        let r1 = ramp_band(10, 12, 100.0);
        let r2 = ramp_band(10, 12, 200.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rgb.tif");
        write_geotiff_multiband::<f64, _>(&[&r0, &r1, &r2], &path, None).unwrap();

        // Roundtrip read: the native reader returns band 1 only. We
        // verify the file is a valid 3-band TIFF by reopening with
        // the tiff crate directly and inspecting BitsPerSample.
        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        let bps = dec
            .get_tag(Tag::Unknown(258)) // BitsPerSample
            .unwrap();
        // 3 samples × 32 bits.
        match bps {
            tiff::decoder::ifd::Value::List(list) => {
                assert_eq!(list.len(), 3, "expected 3 samples per pixel");
            }
            _ => panic!("BitsPerSample not a list"),
        }
        let spp = dec.get_tag_u32(Tag::Unknown(277)).unwrap(); // SamplesPerPixel
        assert_eq!(spp, 3);
    }

    #[test]
    fn multiband_rgba_writes_four_samples() {
        let r0 = ramp_band(8, 8, 0.0);
        let r1 = ramp_band(8, 8, 1.0);
        let r2 = ramp_band(8, 8, 2.0);
        let r3 = ramp_band(8, 8, 3.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rgba.tif");
        write_geotiff_multiband::<f64, _>(&[&r0, &r1, &r2, &r3], &path, None).unwrap();
        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        let spp = dec.get_tag_u32(Tag::Unknown(277)).unwrap();
        assert_eq!(spp, 4);
    }

    #[test]
    fn multiband_rejects_unsupported_band_count() {
        let r0 = ramp_band(4, 4, 0.0);
        let r1 = ramp_band(4, 4, 1.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("two_bands.tif");
        let err = write_geotiff_multiband::<f64, _>(&[&r0, &r1], &path, None).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("1, 3 or 4"), "got: {}", msg);
    }

    #[test]
    fn read_bands_deinterleaves_per_band_values() {
        let r0 = ramp_band(10, 12, 0.0);
        let r1 = ramp_band(10, 12, 100.0);
        let r2 = ramp_band(10, 12, 200.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rgb.tif");
        write_geotiff_multiband::<f64, _>(&[&r0, &r1, &r2], &path, None).unwrap();

        let bands = read_geotiff_bands::<f64, _>(&path).unwrap();
        assert_eq!(bands.len(), 3);
        for b in &bands {
            assert_eq!(b.shape(), (10, 12));
        }
        // Each band carries its own ramp; cell (3,4) = base + 7.
        assert_eq!(bands[0].get(3, 4).unwrap(), 7.0);
        assert_eq!(bands[1].get(3, 4).unwrap(), 107.0);
        assert_eq!(bands[2].get(3, 4).unwrap(), 207.0);
        // Geotransform survives on every band.
        assert_eq!(bands[2].transform().cell_size(), 1.0);
    }

    #[test]
    fn read_geotiff_selects_requested_band() {
        let r0 = ramp_band(6, 6, 0.0);
        let r1 = ramp_band(6, 6, 50.0);
        let r2 = ramp_band(6, 6, 70.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rgb.tif");
        write_geotiff_multiband::<f64, _>(&[&r0, &r1, &r2], &path, None).unwrap();

        // None → first band; Some(b) → band b (0-based).
        assert_eq!(
            read_geotiff::<f64, _>(&path, None)
                .unwrap()
                .get(1, 1)
                .unwrap(),
            2.0
        );
        assert_eq!(
            read_geotiff::<f64, _>(&path, Some(1))
                .unwrap()
                .get(1, 1)
                .unwrap(),
            52.0
        );
        assert_eq!(
            read_geotiff::<f64, _>(&path, Some(2))
                .unwrap()
                .get(1, 1)
                .unwrap(),
            72.0
        );
        // Out-of-range band is rejected.
        assert!(read_geotiff::<f64, _>(&path, Some(3)).is_err());
    }

    #[test]
    fn read_bands_single_band_yields_one() {
        let r0 = ramp_band(5, 5, 0.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("gray.tif");
        write_geotiff_multiband::<f64, _>(&[&r0], &path, None).unwrap();
        let bands = read_geotiff_bands::<f64, _>(&path).unwrap();
        assert_eq!(bands.len(), 1);
        assert_eq!(bands[0].get(2, 2).unwrap(), 4.0);
    }

    #[test]
    fn multiband_rejects_mismatched_shapes() {
        let r0 = ramp_band(4, 4, 0.0);
        let r1 = ramp_band(4, 5, 0.0);
        let r2 = ramp_band(4, 4, 0.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.tif");
        let err = write_geotiff_multiband::<f64, _>(&[&r0, &r1, &r2], &path, None).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("share shape"), "got: {}", msg);
    }

    #[test]
    fn multiband_preserves_crs_and_transform() {
        // `read_geotiff` is single-band only; we inspect the GeoTIFF
        // tags directly with the tiff decoder to verify the multiband
        // writer emits the same scale / tiepoint / geokey payload.
        use crate::crs::CRS;
        let mut r0 = ramp_band(6, 6, 0.0);
        let mut r1 = ramp_band(6, 6, 1.0);
        let mut r2 = ramp_band(6, 6, 2.0);
        // GeoTransform::new(origin_x, origin_y, pixel_width, pixel_height).
        let gt = GeoTransform::new(100000.0, 6300000.0, 10.0, -10.0);
        let crs = CRS::from_epsg(32719);
        for r in [&mut r0, &mut r1, &mut r2] {
            r.set_transform(gt.clone());
            r.set_crs(Some(crs.clone()));
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("crs_check.tif");
        write_geotiff_multiband::<f64, _>(&[&r0, &r1, &r2], &path, None).unwrap();

        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        // ModelPixelScaleTag (33550) -> [pix_w, pix_h.abs(), 0].
        let scale = dec.get_tag_f64_vec(Tag::Unknown(33550)).unwrap();
        assert!((scale[0] - 10.0).abs() < 1e-9);
        assert!((scale[1] - 10.0).abs() < 1e-9);
        // ModelTiepointTag (33922) -> [0,0,0, ox, oy, 0].
        let tp = dec.get_tag_f64_vec(Tag::Unknown(33922)).unwrap();
        assert!((tp[3] - 100000.0).abs() < 1e-6);
        assert!((tp[4] - 6300000.0).abs() < 1e-6);
        // GeoKeyDirectory (34735) -> embeds EPSG:32719 under ProjectedCSTypeGeoKey (3072).
        let gk = dec.get_tag_u32_vec(Tag::Unknown(34735)).unwrap();
        assert!(
            gk.windows(2).any(|w| w[0] == 3072 && w[1] == 0) && gk.contains(&32719),
            "EPSG:32719 missing from GeoKeyDirectory: {:?}",
            gk
        );
    }

    // ---------------------------------------------------------------
    // Native dtype (mejora 1): `write_geotiff<T>` now writes T's own
    // TIFF sample type instead of always casting to Gray32Float.
    // ---------------------------------------------------------------

    /// Roundtrip `write_geotiff` -> `read_geotiff` for a native dtype,
    /// plus a direct tag inspection confirming BitsPerSample/SampleFormat
    /// actually match `T` on disk (not always 32/IEEEFP).
    macro_rules! native_dtype_roundtrip_test {
        ($name:ident, $t:ty, $bits:expr, $fmt:expr, $v0:expr, $v1:expr) => {
            #[test]
            fn $name() {
                let mut r: Raster<$t> = Raster::new(3, 3);
                r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
                r.set(0, 0, $v0).unwrap();
                r.set(1, 1, $v1).unwrap();
                let dir = TempDir::new().unwrap();
                let path = dir.path().join(concat!(stringify!($name), ".tif"));
                write_geotiff(&r, &path, None).unwrap();

                // Native dtype on disk: BitsPerSample/SampleFormat match T,
                // not the old hardcoded 32/IEEEFP.
                let file = File::open(&path).unwrap();
                let mut dec = Decoder::new(file).unwrap();
                let bps = dec.get_tag_u32(Tag::BitsPerSample).unwrap();
                assert_eq!(bps, $bits, "BitsPerSample mismatch for {}", stringify!($t));
                let fmt = dec.get_tag_u32(Tag::SampleFormat).unwrap_or(1);
                assert_eq!(fmt, $fmt, "SampleFormat mismatch for {}", stringify!($t));

                // Roundtrip through the native reader.
                let back: Raster<$t> = read_geotiff(&path, None).unwrap();
                assert_eq!(back.get(0, 0).unwrap(), $v0);
                assert_eq!(back.get(1, 1).unwrap(), $v1);
            }
        };
    }

    // SampleFormat codes: 1 = Uint, 2 = Int, 3 = IEEEFP (tiff::tags::SampleFormat).
    native_dtype_roundtrip_test!(native_dtype_roundtrip_u8, u8, 8, 1, 7u8, 200u8);
    native_dtype_roundtrip_test!(native_dtype_roundtrip_i8, i8, 8, 2, -5i8, 100i8);
    native_dtype_roundtrip_test!(native_dtype_roundtrip_u16, u16, 16, 1, 500u16, 60000u16);
    native_dtype_roundtrip_test!(native_dtype_roundtrip_i16, i16, 16, 2, -500i16, 30000i16);
    native_dtype_roundtrip_test!(
        native_dtype_roundtrip_u32,
        u32,
        32,
        1,
        70_000u32,
        4_000_000_000u32
    );
    native_dtype_roundtrip_test!(
        native_dtype_roundtrip_i32,
        i32,
        32,
        2,
        -70_000i32,
        2_000_000_000i32
    );
    native_dtype_roundtrip_test!(native_dtype_roundtrip_f32, f32, 32, 3, 1.5f32, -2.25f32);
    native_dtype_roundtrip_test!(native_dtype_roundtrip_f64, f64, 64, 3, 1.5f64, -2.25f64);

    /// u64/i64: the native *writer* supports them via `Gray64`/`GrayI64`
    /// (mejora 1 maps every `RasterElement` type losslessly), but the
    /// native *reader*'s `decode_image` match (which this task was
    /// explicitly told not to touch) only handles
    /// `DecodingResult::{U8,U16,U32,I8,I16,I32,F32,F64}` — not `U64`/
    /// `I64` — so `read_geotiff::<u64/i64>` can't round-trip these yet.
    /// This test verifies the writer side directly against the `tiff`
    /// decoder (bypassing our reader) instead of asserting a roundtrip
    /// that the reader doesn't support.
    #[test]
    fn native_dtype_u64_writer_produces_correct_tags_and_bytes() {
        let mut r: Raster<u64> = Raster::new(2, 2);
        r.set_transform(GeoTransform::new(0.0, 2.0, 1.0, -1.0));
        r.set(0, 0, 42u64).unwrap();
        r.set(1, 1, 9_000_000_000u64).unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("u64.tif");
        write_geotiff(&r, &path, None).unwrap();

        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        assert_eq!(dec.get_tag_u32(Tag::BitsPerSample).unwrap(), 64);
        assert_eq!(dec.get_tag_u32(Tag::SampleFormat).unwrap(), 1); // Uint
        let img = dec.read_image().unwrap();
        match img {
            DecodingResult::U64(buf) => {
                assert_eq!(buf[0], 42u64);
                assert_eq!(buf[3], 9_000_000_000u64);
            }
            other => panic!("expected DecodingResult::U64, got {:?}", other),
        }
    }

    // --- read_geotiff_any / AnyRaster -------------------------------

    /// Write a single-band GeoTIFF with a genuinely native `CT` sample
    /// type (not the `f32`-only path `write_geotiff` uses), so
    /// `read_geotiff_any` has something other-than-`f32` to detect.
    /// Includes scale/tiepoint tags so the transform round-trips too.
    fn write_native_typed_tiff<CT>(
        path: &std::path::Path,
        rows: usize,
        cols: usize,
        data: &[CT::Inner],
    ) where
        CT: ColorType,
        [CT::Inner]: tiff::encoder::TiffValue,
    {
        let file = File::create(path).unwrap();
        let mut encoder = TiffEncoder::new(file).unwrap();
        let mut image = encoder.new_image::<CT>(cols as u32, rows as u32).unwrap();
        let scale = vec![2.0_f64, 3.0, 0.0];
        image
            .encoder()
            .write_tag(Tag::Unknown(33550), scale.as_slice())
            .unwrap();
        let tiepoint = vec![0.0_f64, 0.0, 0.0, 500.0, 900.0, 0.0];
        image
            .encoder()
            .write_tag(Tag::Unknown(33922), tiepoint.as_slice())
            .unwrap();
        image.write_data(data).unwrap();
    }

    #[test]
    fn read_geotiff_any_detects_native_u16_without_upcasting() {
        use tiff::encoder::colortype::Gray16;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("native_u16.tif");
        let data: Vec<u16> = vec![0, 1, 1000, u16::MAX, 42, 7, 65535, 12];
        write_native_typed_tiff::<Gray16>(&path, 2, 4, &data);

        let any = read_geotiff_any(&path, None).unwrap();
        assert_eq!(
            any.dtype(),
            DataType::U16,
            "must stay u16, not upcast to f64"
        );
        match &any {
            AnyRaster::U16(r) => {
                assert_eq!(r.shape(), (2, 4));
                assert_eq!(r.get(0, 0).unwrap(), 0);
                assert_eq!(r.get(0, 2).unwrap(), 1000);
                assert_eq!(r.get(0, 3).unwrap(), u16::MAX);
                assert_eq!(r.get(1, 2).unwrap(), 65535);
                // Transform round-trips from the tags we wrote.
                assert_eq!(r.transform().origin_x, 500.0);
                assert_eq!(r.transform().origin_y, 900.0);
                assert_eq!(r.transform().pixel_width, 2.0);
                assert_eq!(r.transform().pixel_height, -3.0);
            }
            other => panic!("expected AnyRaster::U16, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn native_dtype_i64_writer_produces_correct_tags_and_bytes() {
        let mut r: Raster<i64> = Raster::new(2, 2);
        r.set_transform(GeoTransform::new(0.0, 2.0, 1.0, -1.0));
        r.set(0, 0, -42i64).unwrap();
        r.set(1, 1, 5_000_000_000i64).unwrap();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("i64.tif");
        write_geotiff(&r, &path, None).unwrap();

        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        assert_eq!(dec.get_tag_u32(Tag::BitsPerSample).unwrap(), 64);
        assert_eq!(dec.get_tag_u32(Tag::SampleFormat).unwrap(), 2); // Int
        let img = dec.read_image().unwrap();
        match img {
            DecodingResult::I64(buf) => {
                assert_eq!(buf[0], -42i64);
                assert_eq!(buf[3], 5_000_000_000i64);
            }
            other => panic!("expected DecodingResult::I64, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // BigTIFF (mejora 2)
    // ---------------------------------------------------------------

    #[test]
    fn bigtiff_option_writes_bigtiff_header_and_roundtrips() {
        let mut r: Raster<f32> = Raster::new(5, 5);
        r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for i in 0..5 {
            for j in 0..5 {
                r.set(i, j, (i * 5 + j) as f32).unwrap();
            }
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("big.tif");
        let opts = GeoTiffOptions {
            bigtiff: true,
            ..GeoTiffOptions::default()
        };
        write_geotiff(&r, &path, Some(opts)).unwrap();

        // BigTIFF header: byte-order mark + version 43 (vs. 42 for
        // classic TIFF) — see `tiff::encoder::writer::write_bigtiff_header`.
        let bytes = std::fs::read(&path).unwrap();
        let version = u16::from_ne_bytes([bytes[2], bytes[3]]);
        assert_eq!(version, 43, "expected BigTIFF version marker (43)");

        // The `tiff` decoder itself can open and read it back fine.
        let back: Raster<f32> = read_geotiff(&path, None).unwrap();
        assert_eq!(back.get(3, 4).unwrap(), 19.0);
    }

    #[test]
    fn non_bigtiff_option_writes_classic_tiff_header() {
        let r = ramp_band(4, 4, 0.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("classic.tif");
        write_geotiff(&r, &path, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let version = u16::from_ne_bytes([bytes[2], bytes[3]]);
        assert_eq!(version, 42, "expected classic TIFF version marker (42)");
    }

    // ---------------------------------------------------------------
    // Explicit compression errors (mejora 4)
    // ---------------------------------------------------------------

    #[test]
    fn write_geotiff_rejects_unsupported_compression() {
        let r = ramp_band(3, 3, 0.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("zstd.tif");
        let opts = GeoTiffOptions {
            compression: "zstd".to_string(),
            ..GeoTiffOptions::default()
        };
        let err = write_geotiff(&r, &path, Some(opts)).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("zstd") && msg.contains("not supported"),
            "got: {}",
            msg
        );
        // No partial/corrupt file left behind.
        assert!(!path.exists());
    }

    #[test]
    fn write_geotiff_multiband_rejects_unsupported_compression() {
        let r0 = ramp_band(3, 3, 0.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("lzw.tif");
        let opts = GeoTiffOptions {
            compression: "lzw".to_string(),
            ..GeoTiffOptions::default()
        };
        let err = write_geotiff_multiband(&[&r0], &path, Some(opts)).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("lzw") && msg.contains("not supported"),
            "got: {}",
            msg
        );
    }

    #[test]
    fn write_geotiff_stack_rejects_unsupported_compression() {
        let r0 = ramp_band(3, 3, 0.0);
        let r1 = ramp_band(3, 3, 1.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("stack_lzw.tif");
        let opts = GeoTiffOptions {
            compression: "LZW".to_string(),
            ..GeoTiffOptions::default()
        };
        let err = write_geotiff_stack(&[&r0, &r1], None, &path, &opts).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.to_lowercase().contains("lzw") && msg.contains("not supported"),
            "got: {}",
            msg
        );
    }

    #[test]
    fn compression_none_and_deflate_both_still_work() {
        let r = ramp_band(4, 4, 0.0);
        let dir = TempDir::new().unwrap();

        let none_opts = GeoTiffOptions {
            compression: "NONE".to_string(),
            ..GeoTiffOptions::default()
        };
        let p1 = dir.path().join("none.tif");
        write_geotiff(&r, &p1, Some(none_opts)).unwrap();
        let back1: Raster<f64> = read_geotiff(&p1, None).unwrap();
        assert_eq!(back1.get(2, 2).unwrap(), 4.0);

        let deflate_opts = GeoTiffOptions {
            compression: "DEFLATE".to_string(),
            ..GeoTiffOptions::default()
        };
        let p2 = dir.path().join("deflate.tif");
        write_geotiff(&r, &p2, Some(deflate_opts)).unwrap();
        let back2: Raster<f64> = read_geotiff(&p2, None).unwrap();
        assert_eq!(back2.get(2, 2).unwrap(), 4.0);
    }

    // ---------------------------------------------------------------
    // write_geotiff_stack: arbitrary N bands (mejora 3)
    // ---------------------------------------------------------------

    #[test]
    fn write_geotiff_stack_roundtrips_arbitrary_band_count() {
        // 5 bands — outside the 1/3/4 the legacy multiband writer allows.
        let bands_owned: Vec<Raster<u16>> = (0..5)
            .map(|k| {
                let mut r: Raster<u16> = Raster::new(4, 4);
                r.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
                for i in 0..4 {
                    for j in 0..4 {
                        r.set(i, j, (k * 100 + i * 4 + j) as u16).unwrap();
                    }
                }
                r
            })
            .collect();
        let bands: Vec<&Raster<u16>> = bands_owned.iter().collect();

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("stack5.tif");
        write_geotiff_stack(&bands, None, &path, &GeoTiffOptions::default()).unwrap();

        // Tags: SamplesPerPixel=5, BlackIsZero (never RGB), native u16 dtype.
        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        assert_eq!(dec.get_tag_u32(Tag::SamplesPerPixel).unwrap(), 5);
        assert_eq!(
            dec.get_tag_u32(Tag::PhotometricInterpretation).unwrap(),
            1,
            "expected BlackIsZero (1), not RGB (2)"
        );
        // SampleFormat/BitsPerSample are per-sample arrays once n_bands > 1.
        let sample_format = dec.get_tag_u32_vec(Tag::SampleFormat).unwrap();
        assert_eq!(sample_format, vec![1u32; 5]); // Uint x5
        let bits_per_sample = dec.get_tag_u32_vec(Tag::BitsPerSample).unwrap();
        assert_eq!(bits_per_sample, vec![16u32; 5]);

        // Readable back through the existing multi-band reader.
        let read_back = read_geotiff_bands::<u16, _>(&path).unwrap();
        assert_eq!(read_back.len(), 5);
        for (k, band) in read_back.iter().enumerate() {
            assert_eq!(band.get(2, 3).unwrap(), (k * 100 + 2 * 4 + 3) as u16);
        }
    }

    #[test]
    fn read_geotiff_any_detects_native_u8_without_upcasting() {
        use tiff::encoder::colortype::Gray8;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("native_u8.tif");
        let data: Vec<u8> = vec![0, 1, 2, 3, 254, 255];
        write_native_typed_tiff::<Gray8>(&path, 2, 3, &data);

        let any = read_geotiff_any(&path, None).unwrap();
        assert_eq!(any.dtype(), DataType::U8);
        match &any {
            AnyRaster::U8(r) => {
                assert_eq!(r.shape(), (2, 3));
                assert_eq!(r.get(0, 0).unwrap(), 0);
                assert_eq!(r.get(1, 1).unwrap(), 254);
                assert_eq!(r.get(1, 2).unwrap(), 255);
            }
            other => panic!("expected AnyRaster::U8, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn read_geotiff_any_detects_native_f32_without_forcing_f64() {
        // Gray32Float is what `write_geotiff` always emits, so this
        // also doubles as "the common case still returns F32, not F64".
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("native_f32.tif");
        let data: Vec<f32> = vec![0.0, -1.5, 3.25, 1e30, -1e-10, f32::NAN];
        write_native_typed_tiff::<Gray32Float>(&path, 2, 3, &data);

        let any = read_geotiff_any(&path, None).unwrap();
        assert_eq!(any.dtype(), DataType::F32);
        match &any {
            AnyRaster::F32(r) => {
                assert_eq!(r.get(0, 1).unwrap(), -1.5);
                assert_eq!(r.get(0, 2).unwrap(), 3.25);
                assert!(r.get(1, 2).unwrap().is_nan());
            }
            other => panic!("expected AnyRaster::F32, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn write_geotiff_stack_writes_gdal_metadata_band_names() {
        let r0 = ramp_band(3, 3, 0.0);
        let r1 = ramp_band(3, 3, 10.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("named.tif");
        write_geotiff_stack(
            &[&r0, &r1],
            Some(&["ndvi", "ndwi"]),
            &path,
            &GeoTiffOptions::default(),
        )
        .unwrap();

        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        let xml = dec.get_tag_ascii_string(Tag::Unknown(42112)).unwrap();
        assert!(xml.contains("ndvi"), "GDAL_METADATA missing 'ndvi': {xml}");
        assert!(xml.contains("ndwi"), "GDAL_METADATA missing 'ndwi': {xml}");
        assert!(xml.contains("sample=\"0\""));
        assert!(xml.contains("sample=\"1\""));
    }

    #[test]
    fn write_geotiff_stack_rejects_name_count_mismatch() {
        let r0 = ramp_band(3, 3, 0.0);
        let r1 = ramp_band(3, 3, 1.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad_names.tif");
        let err = write_geotiff_stack(
            &[&r0, &r1],
            Some(&["only_one"]),
            &path,
            &GeoTiffOptions::default(),
        )
        .unwrap_err();
        assert!(format!("{}", err).contains("names length"));
    }

    #[test]
    fn write_geotiff_stack_rejects_mismatched_shapes() {
        let r0 = ramp_band(4, 4, 0.0);
        let r1 = ramp_band(4, 5, 0.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad_shape.tif");
        let err =
            write_geotiff_stack(&[&r0, &r1], None, &path, &GeoTiffOptions::default()).unwrap_err();
        assert!(format!("{}", err).contains("share shape"));
    }

    #[test]
    fn write_geotiff_stack_rejects_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.tif");
        let bands: Vec<&Raster<f64>> = vec![];
        let err = write_geotiff_stack(&bands, None, &path, &GeoTiffOptions::default()).unwrap_err();
        assert!(format!("{}", err).contains("at least one band"));
    }

    #[test]
    fn write_geotiff_stack_supports_bigtiff_and_f64() {
        let r0 = ramp_band(6, 6, 0.0);
        let r1 = ramp_band(6, 6, 50.0);
        let r2 = ramp_band(6, 6, 100.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("stack_big.tif");
        let opts = GeoTiffOptions {
            bigtiff: true,
            ..GeoTiffOptions::default()
        };
        write_geotiff_stack(&[&r0, &r1, &r2], None, &path, &opts).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let version = u16::from_ne_bytes([bytes[2], bytes[3]]);
        assert_eq!(version, 43, "expected BigTIFF version marker (43)");

        let read_back = read_geotiff_bands::<f64, _>(&path).unwrap();
        assert_eq!(read_back.len(), 3);
        assert_eq!(read_back[1].get(1, 1).unwrap(), 52.0);
    }

    #[test]
    fn write_geotiff_stack_single_band_matches_write_geotiff_dtype() {
        // n_bands == 1 is a valid stack too (not just the 1/3/4 special
        // case the legacy multiband writer hardcodes).
        let r0 = ramp_band(3, 3, 0.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("stack1.tif");
        write_geotiff_stack(&[&r0], None, &path, &GeoTiffOptions::default()).unwrap();
        let file = File::open(&path).unwrap();
        let mut dec = Decoder::new(file).unwrap();
        assert_eq!(dec.get_tag_u32(Tag::SamplesPerPixel).unwrap(), 1);
        assert_eq!(dec.get_tag_u32(Tag::BitsPerSample).unwrap(), 64); // f64
        let read_back = read_geotiff_bands::<f64, _>(&path).unwrap();
        assert_eq!(read_back[0].get(1, 1).unwrap(), 2.0);
    }

    // ---------------------------------------------------------------
    // GDAL/rasterio interop (empirical validation, not just "in theory")
    // ---------------------------------------------------------------
    //
    // These invoke the real `gdalinfo` / `python3 -c "import rasterio..."`
    // binaries on this machine. They skip (rather than fail) when the
    // tool isn't on PATH so `cargo test` stays green in environments
    // without a GDAL install (e.g. some CI images) — but on a machine
    // that has them (this one does), they are the actual proof the
    // written files interoperate with the GDAL ecosystem, not just with
    // SurtGIS's own reader.

    fn command_available(cmd: &str, arg: &str) -> bool {
        std::process::Command::new(cmd)
            .arg(arg)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    #[test]
    fn interop_gdalinfo_reports_native_dtype_bands_and_bigtiff() {
        if !command_available("gdalinfo", "--version") {
            eprintln!(
                "skipping interop_gdalinfo_reports_native_dtype_bands_and_bigtiff: gdalinfo not on PATH"
            );
            return;
        }
        let bands_owned: Vec<Raster<u16>> = (0..3)
            .map(|k| {
                let mut r: Raster<u16> = Raster::new(4, 4);
                r.set_transform(GeoTransform::new(500000.0, 6300000.0, 10.0, -10.0));
                r.set_crs(Some(crate::crs::CRS::from_epsg(32719)));
                for i in 0..4 {
                    for j in 0..4 {
                        r.set(i, j, (k * 1000 + i * 4 + j) as u16).unwrap();
                    }
                }
                r
            })
            .collect();
        let bands: Vec<&Raster<u16>> = bands_owned.iter().collect();
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("interop_stack.tif");
        let opts = GeoTiffOptions {
            bigtiff: true,
            ..GeoTiffOptions::default()
        };
        write_geotiff_stack(&bands, Some(&["b1", "b2", "b3"]), &path, &opts).unwrap();

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
            text.contains("Type=UInt16"),
            "expected UInt16, got:\n{text}"
        );
        assert!(text.contains("Band 3"), "expected 3 bands, got:\n{text}");
    }

    #[test]
    fn interop_rasterio_reads_values_and_dtype() {
        if !command_available("python3", "--version") {
            eprintln!("skipping interop_rasterio_reads_values_and_dtype: python3 not on PATH");
            return;
        }
        let mut r: Raster<i16> = Raster::new(4, 4);
        r.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        for i in 0..4 {
            for j in 0..4 {
                r.set(i, j, (i * 4 + j) as i16 - 5).unwrap();
            }
        }
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("interop_i16.tif");
        write_geotiff(&r, &path, None).unwrap();

        let script = format!(
            "import rasterio\n\
             ds = rasterio.open('{p}')\n\
             assert ds.dtypes[0] == 'int16', ds.dtypes\n\
             arr = ds.read(1)\n\
             assert arr[1, 1] == 0, arr[1,1]\n\
             print('OK', ds.dtypes, arr[1,1])\n",
            // Forward slashes: on Windows, `path.to_str()` yields
            // backslashes, and interpolating those raw into a Python
            // single-quoted string turns e.g. `\U` into the start of a
            // unicode escape, breaking the script with a SyntaxError.
            // Forward slashes are accepted as path separators by both
            // Python and the underlying GDAL/rasterio on all platforms.
            p = path.to_str().unwrap().replace('\\', "/")
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
            // rasterio itself may not be installed even though python3 is;
            // treat that as "skip", not "fail".
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("ModuleNotFoundError") || stderr.contains("No module named") {
                eprintln!(
                    "skipping interop_rasterio_reads_values_and_dtype: rasterio not installed"
                );
                return;
            }
            panic!("rasterio check failed: {}", stderr);
        }
    }

    #[test]
    fn read_geotiff_any_selects_band_on_multiband_file() {
        // Multiband files still go through the native f32 stack writer;
        // read_geotiff_any must still honour the band selection.
        let r0 = ramp_band(4, 4, 0.0);
        let r1 = ramp_band(4, 4, 100.0);
        let r2 = ramp_band(4, 4, 200.0);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("rgb_any.tif");
        write_geotiff_multiband::<f64, _>(&[&r0, &r1, &r2], &path, None).unwrap();

        let band0 = read_geotiff_any(&path, None).unwrap();
        let band2 = read_geotiff_any(&path, Some(2)).unwrap();
        assert_eq!(band0.dtype(), DataType::F32);
        assert_eq!(band2.dtype(), DataType::F32);
        if let (AnyRaster::F32(b0), AnyRaster::F32(b2)) = (&band0, &band2) {
            assert_eq!(b0.get(1, 1).unwrap(), 2.0);
            assert_eq!(b2.get(1, 1).unwrap(), 202.0);
        } else {
            panic!("expected AnyRaster::F32 for both bands");
        }

        // Out-of-range band is still rejected, same as read_geotiff.
        assert!(read_geotiff_any(&path, Some(3)).is_err());
    }

    #[test]
    fn any_raster_to_f64_matches_read_geotiff_f64() {
        use tiff::encoder::colortype::Gray16;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("native_u16_f64_check.tif");
        let data: Vec<u16> = vec![0, 1, 1000, u16::MAX];
        write_native_typed_tiff::<Gray16>(&path, 2, 2, &data);

        let any = read_geotiff_any(&path, None).unwrap();
        assert_eq!(any.dtype(), DataType::U16);
        let as_f64 = any.to_f64();
        // Same file, forced straight to f64 via the existing API.
        let direct: Raster<f64> = read_geotiff(&path, None).unwrap();
        assert_eq!(as_f64.shape(), direct.shape());
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(as_f64.get(row, col).unwrap(), direct.get(row, col).unwrap());
            }
        }
    }

    #[test]
    fn read_geotiff_any_from_buffer_matches_path_based_read() {
        use tiff::encoder::colortype::Gray8;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("native_u8_buf.tif");
        let data: Vec<u8> = vec![10, 20, 30, 40];
        write_native_typed_tiff::<Gray8>(&path, 2, 2, &data);

        let bytes = std::fs::read(&path).unwrap();
        let any = read_geotiff_any_from_buffer(&bytes, None).unwrap();
        assert_eq!(any.dtype(), DataType::U8);
        if let AnyRaster::U8(r) = &any {
            assert_eq!(r.get(0, 0).unwrap(), 10);
            assert_eq!(r.get(1, 1).unwrap(), 40);
        } else {
            unreachable!();
        }
    }
}
