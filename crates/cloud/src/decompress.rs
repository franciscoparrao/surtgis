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

/// Undo horizontal differencing predictor (TIFF Predictor=2).
///
/// For each row, the first sample is stored as-is. Subsequent samples
/// store the difference from the previous sample as a WHOLE SAMPLE
/// (not individual bytes). For uint16, accumulate as u16 with wrapping.
pub fn undo_horizontal_differencing(
    data: &mut [u8],
    tile_width: usize,
    bytes_per_sample: usize,
) {
    if tile_width == 0 || bytes_per_sample == 0 { return; }
    let row_bytes = tile_width * bytes_per_sample;
    for row_start in (0..data.len()).step_by(row_bytes) {
        let row_end = (row_start + row_bytes).min(data.len());
        let row = &mut data[row_start..row_end];
        match bytes_per_sample {
            1 => {
                for i in 1..row.len() {
                    row[i] = row[i].wrapping_add(row[i - 1]);
                }
            }
            2 => {
                // Accumulate as u16 to propagate carry between bytes
                let samples = row.len() / 2;
                for i in 1..samples {
                    let prev = u16::from_le_bytes([row[(i-1)*2], row[(i-1)*2+1]]);
                    let diff = u16::from_le_bytes([row[i*2], row[i*2+1]]);
                    let val = prev.wrapping_add(diff);
                    let bytes = val.to_le_bytes();
                    row[i*2] = bytes[0];
                    row[i*2+1] = bytes[1];
                }
            }
            4 => {
                let samples = row.len() / 4;
                for i in 1..samples {
                    let off = i * 4;
                    let prev = u32::from_le_bytes([row[off-4], row[off-3], row[off-2], row[off-1]]);
                    let diff = u32::from_le_bytes([row[off], row[off+1], row[off+2], row[off+3]]);
                    let val = prev.wrapping_add(diff);
                    let bytes = val.to_le_bytes();
                    row[off..off+4].copy_from_slice(&bytes);
                }
            }
            _ => {
                // Fallback: byte-level accumulation (may not be correct for >1 bps)
                for i in bytes_per_sample..row.len() {
                    row[i] = row[i].wrapping_add(row[i - bytes_per_sample]);
                }
            }
        }
    }
}

/// Undo floating-point predictor (TIFF Predictor=3).
///
/// Per TIFF Technical Note 3 — applies per row, in two steps:
///   1. Byte-wise cumulative sum with stride = `samples_per_pixel` (=1 for
///      grayscale DEM). Undoes the differencing applied during encoding.
///   2. Un-shuffle from planar (big-endian layout) to native-endian interleaved:
///      plane 0 = MSB of each sample, plane 1 = next byte, ..., last = LSB.
///
/// Matches the Rust `tiff` crate implementation (predict_f32) and libtiff.
/// Used by Copernicus DEM GLO-30/90, NASADEM, and other f32 COGs.
pub fn undo_floating_point_predictor(
    data: &mut [u8],
    tile_width: usize,
    bytes_per_sample: usize,
) {
    undo_floating_point_predictor_multi(data, tile_width, bytes_per_sample, 1)
}

/// Same as [`undo_floating_point_predictor`] but with explicit `samples_per_pixel`.
///
/// For multi-band (RGB etc.) use samples_per_pixel = number of bands.
/// For grayscale (DEM) use samples_per_pixel = 1.
pub fn undo_floating_point_predictor_multi(
    data: &mut [u8],
    tile_width: usize,
    bytes_per_sample: usize,
    samples_per_pixel: usize,
) {
    if tile_width == 0 || bytes_per_sample <= 1 || samples_per_pixel == 0 { return; }
    let row_bytes = tile_width * bytes_per_sample * samples_per_pixel;
    let mut tmp = vec![0u8; row_bytes];
    let n_samples_in_row = tile_width * samples_per_pixel;

    for row_start in (0..data.len()).step_by(row_bytes) {
        let row_end = (row_start + row_bytes).min(data.len());
        let row = &mut data[row_start..row_end];
        if row.len() != row_bytes { continue; }

        // Step 1: byte-wise cumsum with stride = samples_per_pixel.
        // (For grayscale, stride=1, so it's continuous byte-wise cumsum.)
        for i in samples_per_pixel..row_bytes {
            row[i] = row[i].wrapping_add(row[i - samples_per_pixel]);
        }

        // Step 2: un-shuffle planes → interleaved samples, big-endian to native.
        // Planar layout: [plane0 (MSB): sample 0, sample 1, ..., sample N-1,
        //                 plane1:        sample 0, sample 1, ..., sample N-1,
        //                 ...
        //                 plane{B-1} (LSB): sample 0, sample 1, ..., sample N-1]
        // where N = tile_width * samples_per_pixel, B = bytes_per_sample.
        match bytes_per_sample {
            4 => {
                for i in 0..n_samples_in_row {
                    let val = u32::from_be_bytes([
                        row[i],
                        row[n_samples_in_row + i],
                        row[2 * n_samples_in_row + i],
                        row[3 * n_samples_in_row + i],
                    ]);
                    let bytes = val.to_ne_bytes();
                    tmp[i * 4..i * 4 + 4].copy_from_slice(&bytes);
                }
            }
            8 => {
                for i in 0..n_samples_in_row {
                    let val = u64::from_be_bytes([
                        row[i],
                        row[n_samples_in_row + i],
                        row[2 * n_samples_in_row + i],
                        row[3 * n_samples_in_row + i],
                        row[4 * n_samples_in_row + i],
                        row[5 * n_samples_in_row + i],
                        row[6 * n_samples_in_row + i],
                        row[7 * n_samples_in_row + i],
                    ]);
                    let bytes = val.to_ne_bytes();
                    tmp[i * 8..i * 8 + 8].copy_from_slice(&bytes);
                }
            }
            2 => {
                for i in 0..n_samples_in_row {
                    let val = u16::from_be_bytes([row[i], row[n_samples_in_row + i]]);
                    let bytes = val.to_ne_bytes();
                    tmp[i * 2..i * 2 + 2].copy_from_slice(&bytes);
                }
            }
            _ => {
                for i in 0..n_samples_in_row {
                    for b in 0..bytes_per_sample {
                        tmp[i * bytes_per_sample + b] = row[b * n_samples_in_row + i];
                    }
                }
            }
        }
        row.copy_from_slice(&tmp);
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
        // 15-bit unsigned: tightly bit-packed (Sentinel-2 L2A on Planetary
        // Computer uses this for some bands to save 6.25 % on disk). See
        // BUG_TILE_DECODE_BPS15_STRIPING.md — previously this branch
        // mis-treated the buffer as padded u16, dropping 6.25 % of the
        // samples per tile and producing silent striping in composites.
        (15, sample_format::UNSIGNED_INT) => unpack_packed_to_u16::<T>(raw, 15),
        (16, sample_format::UNSIGNED_INT) => cast_from_le::<u16, T>(raw),
        (16, sample_format::SIGNED_INT) => cast_from_le::<i16, T>(raw),
        (32, sample_format::UNSIGNED_INT) => cast_from_le::<u32, T>(raw),
        (32, sample_format::SIGNED_INT) => cast_from_le::<i32, T>(raw),
        (32, sample_format::FLOAT) => cast_from_le::<f32, T>(raw),
        (64, sample_format::FLOAT) => cast_from_le::<f64, T>(raw),
        _ => Err(CloudError::UnsupportedDataType { bps, sf }),
    }
}

/// Unpack a TIFF buffer of unsigned integer samples that are stored at a
/// non-byte-aligned bit width (`bps` ∈ {9..=15}), packed MSB-first into the
/// byte stream as per the TIFF 6.0 spec § "FillOrder = 1". Returns one value
/// per sample, cast to `T`.
///
/// This matches GDAL and libtiff's behaviour on Planetary Computer Sentinel-2
/// L2A COGs, which store some bands at `bps=15` packed (491520 bytes for a
/// 512×512 tile rather than 524288 padded bytes).
fn unpack_packed_to_u16<T: RasterElement>(raw: &[u8], bps: u32) -> Result<Vec<T>> {
    if !(9..=16).contains(&bps) {
        return Err(CloudError::Decompress(format!(
            "unpack_packed_to_u16 called with unsupported bps={}", bps
        )));
    }
    let total_bits = (raw.len() as u64) * 8;
    if total_bits % (bps as u64) != 0 {
        return Err(CloudError::Decompress(format!(
            "raw data of {} bytes ({} bits) is not a clean multiple of {} bits per sample; \
             tile is likely truncated or the encoder padded samples in an unsupported way",
            raw.len(), total_bits, bps,
        )));
    }
    let n_samples = (total_bits / bps as u64) as usize;
    let mask: u32 = (1u32 << bps) - 1;

    let mut result = Vec::with_capacity(n_samples);
    let mut bit_pos: u64 = 0;
    for _ in 0..n_samples {
        let byte_offset = (bit_pos / 8) as usize;
        let bit_in_byte = (bit_pos % 8) as u32;

        // Read up to 3 bytes starting at byte_offset; for bps ≤ 16 plus any
        // 0..7 bit offset, the sample fits within 24 consecutive bits.
        let b0 = raw[byte_offset] as u32;
        let b1 = if byte_offset + 1 < raw.len() { raw[byte_offset + 1] as u32 } else { 0 };
        let b2 = if byte_offset + 2 < raw.len() { raw[byte_offset + 2] as u32 } else { 0 };
        let window: u32 = (b0 << 16) | (b1 << 8) | b2;
        let shift = 24 - bit_in_byte - bps;
        let sample = ((window >> shift) & mask) as u16;

        let f64_val = sample as f64;
        result.push(NumCast::from(f64_val).unwrap_or(T::default_nodata()));
        bit_pos += bps as u64;
    }
    Ok(result)
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

    /// Pack 16 samples of 15 bits each into 30 bytes (240 bits) MSB-first,
    /// then verify our unpacker recovers them. Regression test for
    /// BUG_TILE_DECODE_BPS15_STRIPING.md.
    #[test]
    fn test_bytes_to_typed_packed_15bit() {
        // 16 known samples, each fits in 15 bits (max 32767).
        let samples: [u16; 16] = [
            0, 1, 2, 0x7FFF, 0x4000, 0x2AAA, 0x5555, 0x1234,
            0x4321, 0x6789, 0x0FFF, 0x7000, 0x3CCC, 0x1248, 0x5A5A, 0x2727,
        ];

        // Pack MSB-first: total bits = 16 × 15 = 240 → 30 bytes.
        let mut bit_pos: u32 = 0;
        let mut packed = vec![0u8; 30];
        for &s in &samples {
            for b in 0..15 {
                // bit `b` of sample (MSB first within the 15 bits)
                let bit_in_sample = 14 - b;
                let bit_val = (s >> bit_in_sample) & 1;
                let byte_off = (bit_pos / 8) as usize;
                let bit_in_byte = 7 - (bit_pos % 8);
                packed[byte_off] |= (bit_val as u8) << bit_in_byte;
                bit_pos += 1;
            }
        }

        let unpacked: Vec<f64> =
            bytes_to_typed(&packed, 15, sample_format::UNSIGNED_INT).unwrap();
        assert_eq!(unpacked.len(), 16, "unpacked length mismatch");
        for (i, &expected) in samples.iter().enumerate() {
            assert_eq!(
                unpacked[i] as u16, expected,
                "sample {i}: expected {expected}, got {}", unpacked[i] as u16,
            );
        }
    }

    /// 512×512 tile at bps=15 should produce exactly 262144 samples from
    /// 491520 bytes (the value the postdoc's logs reported as `raw_len`).
    /// Pre-fix, the (15, UNSIGNED_INT) branch cast as u16 and produced
    /// 245760 samples — 6.25 % short.
    #[test]
    fn test_bytes_to_typed_15bit_pc_tile_size() {
        let n_samples = 512 * 512;
        let total_bits = n_samples * 15;
        let raw_len = total_bits / 8;
        assert_eq!(raw_len, 491520, "expected sentinel-2 tile raw_len");

        let raw = vec![0xAAu8; raw_len];
        let result: Vec<f64> =
            bytes_to_typed(&raw, 15, sample_format::UNSIGNED_INT).unwrap();
        assert_eq!(result.len(), n_samples, "must produce 262144 samples (got {})", result.len());
    }

    /// Truncated tile (raw_len not a clean multiple of bps bits) must fail
    /// rather than silently returning a shorter buffer.
    #[test]
    fn test_bytes_to_typed_15bit_truncated_errors() {
        // 100 bytes = 800 bits, not divisible by 15
        let raw = vec![0u8; 100];
        let result: Result<Vec<f64>> = bytes_to_typed(&raw, 15, sample_format::UNSIGNED_INT);
        assert!(result.is_err(), "truncated 15-bit buffer should fail decode");
    }
}
