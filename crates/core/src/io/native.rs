//! Native GeoTIFF reading/writing (without GDAL dependency)
//!
//! Uses the `tiff` crate for basic TIFF I/O.
//! For full GeoTIFF support (projections, advanced types), enable the `gdal` feature.

use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster, RasterElement};
use std::fs::File;
use std::io::Cursor;
use std::path::Path;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::encoder::colortype::{ColorType, Gray32Float, RGB32Float, RGBA32Float};
use tiff::encoder::compression::DeflateLevel;
use tiff::encoder::{Compression, TiffEncoder};
use tiff::tags::Tag;

/// Options for writing GeoTIFF files
#[derive(Debug, Clone)]
pub struct GeoTiffOptions {
    /// Compression (not fully supported in native mode)
    pub compression: String,
}

impl Default for GeoTiffOptions {
    fn default() -> Self {
        Self {
            compression: "NONE".to_string(),
        }
    }
}

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

/// Internal: decode a GeoTIFF (all bands, interleaved) plus its geo-tags.
fn decode_image<T, R>(reader: R, source_len: Option<u64>) -> Result<DecodedImage<T>>
where
    T: RasterElement,
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

    // Read image data (pixel-interleaved for multi-band TIFFs)
    let result = decoder
        .read_image()
        .map_err(|e| Error::Other(format!("Cannot read image data: {}", e)))?;

    let data: Vec<T> = match result {
        DecodingResult::F32(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::F64(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::U8(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::U16(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::U32(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::I8(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::I16(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        DecodingResult::I32(buf) => buf
            .iter()
            .map(|&v| num_traits::cast(v).unwrap_or(T::default_nodata()))
            .collect(),
        _ => {
            return Err(Error::UnsupportedDataType(
                "Unsupported TIFF pixel format".to_string(),
            ));
        }
    };

    if data.len() != rows * cols * spp {
        return Err(Error::InvalidDimensions {
            width: cols,
            height: rows,
        });
    }

    Ok(DecodedImage {
        data,
        rows,
        cols,
        spp,
        transform: read_geotransform(&mut decoder).ok(),
        crs: read_crs(&mut decoder),
        nodata: read_nodata(&mut decoder),
    })
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
    // Cast GDAL_NODATA to T; for float types, normalize nodata to NaN so
    // all algorithms (which check .is_nan()) handle them correctly.
    if let Some(nodata_f64) = nodata
        && let Some(nd) = num_traits::cast::<f64, T>(nodata_f64)
    {
        if T::is_float() {
            let nan_val = T::default_nodata();
            for val in raster.data_mut().iter_mut() {
                if val.is_nodata(Some(nd)) {
                    *val = nan_val;
                }
            }
            // Pixels were rewritten to NaN, so the metadata must say NaN
            // too. Keeping the original sentinel here would make a
            // subsequent write emit GDAL_NODATA = <sentinel> over NaN
            // pixels — external tools (GDAL/QGIS) would then treat the
            // NaNs as valid data.
            raster.set_nodata(Some(nan_val));
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

/// Write a Raster to a GeoTIFF file
///
/// Native writer with limited GeoTIFF metadata support.
/// Writes as 32-bit float. For full support, enable the `gdal` feature.
pub fn write_geotiff<T, P>(
    raster: &Raster<T>,
    path: P,
    options: Option<GeoTiffOptions>,
) -> Result<()>
where
    T: RasterElement,
    P: AsRef<Path>,
{
    let compress = options
        .as_ref()
        .map(|o| o.compression.to_lowercase() != "none")
        .unwrap_or(false);

    // Write to temp file first, then atomic rename to prevent corrupt partial files
    let final_path = path.as_ref();
    let tmp_path = final_path.with_extension("tmp");
    let file = File::create(&tmp_path)?;
    encode_geotiff(raster, file, compress)?;
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
    T: RasterElement,
{
    let compress = options
        .as_ref()
        .map(|o| o.compression.to_lowercase() != "none")
        .unwrap_or(false);
    let mut buf = Vec::new();
    encode_geotiff(raster, Cursor::new(&mut buf), compress)?;
    Ok(buf)
}

/// Internal: encode a Raster as GeoTIFF into any `Write + Seek` sink
fn encode_geotiff<T, W>(raster: &Raster<T>, writer: W, compress: bool) -> Result<()>
where
    T: RasterElement,
    W: std::io::Write + std::io::Seek,
{
    let compression = if compress {
        Compression::Deflate(DeflateLevel::Balanced)
    } else {
        Compression::Uncompressed
    };

    let mut encoder = TiffEncoder::new(writer)
        .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?
        .with_compression(compression);

    let (rows, cols) = raster.shape();

    // Convert data to f32
    let data: Vec<f32> = raster
        .data()
        .iter()
        .map(|&v| num_traits::cast(v).unwrap_or(f32::NAN))
        .collect();

    let mut image = encoder
        .new_image::<Gray32Float>(cols as u32, rows as u32)
        .map_err(|e| Error::Other(format!("Cannot create TIFF image: {}", e)))?;

    // Write GeoTIFF tags
    let gt = raster.transform();

    // ModelPixelScaleTag
    let scale = vec![gt.pixel_width, gt.pixel_height.abs(), 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33550), scale.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write scale tag: {}", e)))?;

    // ModelTiepointTag
    let tiepoint = vec![0.0, 0.0, 0.0, gt.origin_x, gt.origin_y, 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33922), tiepoint.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write tiepoint tag: {}", e)))?;

    // GeoKeyDirectoryTag (34735) — embed actual CRS from raster metadata.
    // GeoKey structure: [KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys,
    //                    KeyID, TIFFTagLocation, Count, Value_Offset, ...]
    let geokeys: Vec<u16> = if let Some(crs) = raster.crs() {
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
    };
    image
        .encoder()
        .write_tag(Tag::Unknown(34735), geokeys.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write geokey tag: {}", e)))?;

    // GDAL_NODATA tag (42113) — write as ASCII string
    if let Some(nd) = raster.nodata() {
        if let Some(nd_f64) = nd.to_f64() {
            let nodata_str = if nd_f64.is_nan() {
                "nan".to_string()
            } else {
                format!("{}", nd_f64)
            };
            // Write as a proper ASCII tag (the str impl appends the NUL);
            // writing via as_bytes() produced a BYTE-typed tag that
            // get_tag_ascii_string rejects on read-back.
            image
                .encoder()
                .write_tag(Tag::Unknown(42113), nodata_str.as_str())
                .map_err(|e| Error::Other(format!("Cannot write nodata tag: {}", e)))?;
        }
    }

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

    let compress = options
        .as_ref()
        .map(|o| o.compression.to_lowercase() != "none")
        .unwrap_or(false);

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
        1 => encode_multiband_image::<Gray32Float, _>(file, bands[0], &interleaved, compress)?,
        3 => encode_multiband_image::<RGB32Float, _>(file, bands[0], &interleaved, compress)?,
        4 => encode_multiband_image::<RGBA32Float, _>(file, bands[0], &interleaved, compress)?,
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
    compress: bool,
) -> Result<()>
where
    CT: ColorType<Inner = f32>,
    W: std::io::Write + std::io::Seek,
{
    let compression = if compress {
        Compression::Deflate(DeflateLevel::Balanced)
    } else {
        Compression::Uncompressed
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::Raster;
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
}
