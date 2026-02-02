//! Tile decompression for COG tiles.
//!
//! Supports DEFLATE (via `flate2`), LZW (via `weezl`), and uncompressed.

use crate::error::{CloudError, Result};
use num_traits::NumCast;
use surtgis_core::RasterElement;

/// TIFF compression codes.
pub mod compression {
    pub const NONE: u16 = 1;
    pub const LZW: u16 = 5;
    pub const DEFLATE: u16 = 8;
    pub const ADOBE_DEFLATE: u16 = 32946;
}

/// TIFF sample format codes.
pub mod sample_format {
    pub const UNSIGNED_INT: u16 = 1;
    pub const SIGNED_INT: u16 = 2;
    pub const FLOAT: u16 = 3;
}

/// Decompress raw tile bytes according to the compression method.
pub fn decompress_tile(
    data: &[u8],
    compression_code: u16,
    expected_raw_size: usize,
) -> Result<Vec<u8>> {
    match compression_code {
        compression::NONE => {
            Ok(data.to_vec())
        }

        #[cfg(feature = "deflate")]
        compression::DEFLATE | compression::ADOBE_DEFLATE => {
            use std::io::Read;
            // TIFF DEFLATE tiles use zlib format (with 2-byte header),
            // not raw deflate. Try zlib first, fall back to raw deflate.
            let mut decoder = flate2::read::ZlibDecoder::new(data);
            let mut out = Vec::with_capacity(expected_raw_size);
            match decoder.read_to_end(&mut out) {
                Ok(_) => Ok(out),
                Err(_) => {
                    // Fallback: raw deflate (no zlib header)
                    out.clear();
                    let mut decoder = flate2::read::DeflateDecoder::new(data);
                    decoder
                        .read_to_end(&mut out)
                        .map_err(|e| CloudError::Decompress(format!("DEFLATE: {}", e)))?;
                    Ok(out)
                }
            }
        }

        #[cfg(not(feature = "deflate"))]
        compression::DEFLATE | compression::ADOBE_DEFLATE => {
            Err(CloudError::UnsupportedCompression(compression_code))
        }

        #[cfg(feature = "lzw")]
        compression::LZW => {
            let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
            let out = decoder
                .decode(data)
                .map_err(|e| CloudError::Decompress(format!("LZW: {}", e)))?;
            Ok(out)
        }

        #[cfg(not(feature = "lzw"))]
        compression::LZW => {
            Err(CloudError::UnsupportedCompression(compression_code))
        }

        _ => Err(CloudError::UnsupportedCompression(compression_code)),
    }
}

/// Convert raw decompressed bytes into typed values.
///
/// Interprets `raw` as an array of the appropriate type based on
/// `bits_per_sample` and `sample_format`, then casts each value to `T`.
pub fn bytes_to_typed<T: RasterElement>(
    raw: &[u8],
    bits_per_sample: u16,
    sample_format: u16,
) -> Result<Vec<T>> {
    let bps = bits_per_sample;
    let sf = sample_format;

    match (bps, sf) {
        (8, sample_format::UNSIGNED_INT) => cast_slice::<u8, T>(raw),
        (8, sample_format::SIGNED_INT) => cast_from_bytes::<i8, T>(raw),
        (16, sample_format::UNSIGNED_INT) => cast_from_le::<u16, T>(raw),
        (16, sample_format::SIGNED_INT) => cast_from_le::<i16, T>(raw),
        (32, sample_format::UNSIGNED_INT) => cast_from_le::<u32, T>(raw),
        (32, sample_format::SIGNED_INT) => cast_from_le::<i32, T>(raw),
        (32, sample_format::FLOAT) => cast_from_le::<f32, T>(raw),
        (64, sample_format::FLOAT) => cast_from_le::<f64, T>(raw),
        _ => Err(CloudError::UnsupportedDataType { bps, sf }),
    }
}

/// Cast a byte slice where each byte is one element.
fn cast_slice<S, T: RasterElement>(raw: &[u8]) -> Result<Vec<T>>
where
    S: Copy + Into<f64>,
{
    Ok(raw
        .iter()
        .map(|&b| {
            let src: S = unsafe { std::mem::transmute_copy(&b) };
            let f: f64 = src.into();
            NumCast::from(f).unwrap_or(T::default_nodata())
        })
        .collect())
}

/// Cast from i8 bytes.
fn cast_from_bytes<S, T: RasterElement>(raw: &[u8]) -> Result<Vec<T>>
where
    S: Copy + 'static,
{
    Ok(raw
        .iter()
        .map(|&b| {
            let v: S = unsafe { std::mem::transmute_copy(&b) };
            // Use NumCast to go through f64
            let f = unsafe { *(&v as *const S as *const i8) } as f64;
            NumCast::from(f).unwrap_or(T::default_nodata())
        })
        .collect())
}

/// Cast from little-endian multibyte types.
fn cast_from_le<S, T: RasterElement>(raw: &[u8]) -> Result<Vec<T>>
where
    S: Copy + 'static + LeRead,
{
    let elem_size = std::mem::size_of::<S>();
    if raw.len() % elem_size != 0 {
        return Err(CloudError::Decompress(format!(
            "raw data length {} not aligned to element size {}",
            raw.len(),
            elem_size
        )));
    }

    let count = raw.len() / elem_size;
    let mut result = Vec::with_capacity(count);

    for i in 0..count {
        let offset = i * elem_size;
        let value = S::read_le(&raw[offset..offset + elem_size]);
        let f64_val = value.to_f64_val();
        result.push(NumCast::from(f64_val).unwrap_or(T::default_nodata()));
    }

    Ok(result)
}

/// Trait for reading little-endian values from byte slices.
trait LeRead: Sized {
    fn read_le(data: &[u8]) -> Self;
    fn to_f64_val(&self) -> f64;
}

macro_rules! impl_le_read_int {
    ($t:ty) => {
        impl LeRead for $t {
            fn read_le(data: &[u8]) -> Self {
                let mut bytes = [0u8; std::mem::size_of::<$t>()];
                bytes.copy_from_slice(&data[..std::mem::size_of::<$t>()]);
                <$t>::from_le_bytes(bytes)
            }
            fn to_f64_val(&self) -> f64 {
                *self as f64
            }
        }
    };
}

impl_le_read_int!(u16);
impl_le_read_int!(i16);
impl_le_read_int!(u32);
impl_le_read_int!(i32);

impl LeRead for f32 {
    fn read_le(data: &[u8]) -> Self {
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[..4]);
        f32::from_le_bytes(bytes)
    }
    fn to_f64_val(&self) -> f64 {
        *self as f64
    }
}

impl LeRead for f64 {
    fn read_le(data: &[u8]) -> Self {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[..8]);
        f64::from_le_bytes(bytes)
    }
    fn to_f64_val(&self) -> f64 {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompress_none() {
        let data = vec![1, 2, 3, 4];
        let out = decompress_tile(&data, compression::NONE, 4).unwrap();
        assert_eq!(out, data);
    }

    #[cfg(feature = "deflate")]
    #[test]
    fn test_decompress_deflate() {
        use std::io::Write;
        let original = vec![42u8; 256];
        let mut encoder = flate2::write::DeflateEncoder::new(
            Vec::new(),
            flate2::Compression::default(),
        );
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decompressed =
            decompress_tile(&compressed, compression::DEFLATE, original.len()).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_bytes_to_typed_f32() {
        let values: Vec<f32> = vec![1.0, 2.5, 3.0];
        let mut raw = Vec::new();
        for v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }

        let result: Vec<f64> =
            bytes_to_typed(&raw, 32, sample_format::FLOAT).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.5).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_bytes_to_typed_u16() {
        let values: Vec<u16> = vec![100, 200, 300];
        let mut raw = Vec::new();
        for v in &values {
            raw.extend_from_slice(&v.to_le_bytes());
        }

        let result: Vec<f64> =
            bytes_to_typed(&raw, 16, sample_format::UNSIGNED_INT).unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 100.0).abs() < 1e-6);
        assert!((result[1] - 200.0).abs() < 1e-6);
    }
}
