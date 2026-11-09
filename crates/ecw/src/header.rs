//! ECW v2 file header and block-offset table parsing.
//!
//! The header layout was established from the libecwj2-3.3 reference
//! sources (used as a format specification only — see `docs/ecw_format.md`
//! at the repository root for the full write-up and licensing review).
//!
//! Byte-order warning, quoting the reference implementation verbatim:
//! *"DUE TO A COMPLETE COCKUP THE INTS IN THE HEADER ARE STORED ON DISC AS
//! MSB WHILST FLOATS, THE BLOCK TABLE AND THE REST OF THE DATA IS LSB!"*
//! Header integers are big-endian; IEEE8 doubles, the block-offset table
//! and all block payloads are little-endian.

use crate::{EcwError, Result};
use std::io::{Read, Seek, SeekFrom};

/// First byte of every ECW file: ASCII `'e'`.
pub const ECW_HEADER_TAG: u8 = 0x65;
/// Highest format version this decoder understands (the classic ECW v2).
pub const MAX_SUPPORTED_VERSION: u8 = 2;

const MAX_DATUM_LEN: usize = 16;
const MAX_PROJECTION_LEN: usize = 16;
/// Same bound as the reference implementation (`MAX_LEVELS`).
const MAX_LEVELS: usize = 20;

/// Colour-space / band organisation of the compressed data
/// (`CompressFormat` in the reference sources; values are the
/// `NCSFileColorSpace` enum).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressFormat {
    /// Single-band greyscale, reconstructed straight to 0..255.
    Uint8,
    /// 3 bands stored as JPEG-style YUV; decoded back to RGB.
    Yuv,
    /// N independent bands (no colour transform), 0..255 each.
    Multiband,
    /// Any other value found in the header (kept for error messages).
    Other(u8),
}

impl CompressFormat {
    fn from_raw(v: u8) -> Self {
        match v {
            1 => CompressFormat::Uint8,
            2 => CompressFormat::Yuv,
            3 => CompressFormat::Multiband,
            other => CompressFormat::Other(other),
        }
    }
}

/// Units of the cell increments (`CellSizeUnits` in the reference sources).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellUnits {
    /// Metres (projected CRS).
    Meters,
    /// Decimal degrees (geographic CRS).
    Degrees,
    /// Feet.
    Feet,
    /// Unknown/invalid marker.
    Other(u8),
}

impl CellUnits {
    fn from_raw(v: u8) -> Self {
        match v {
            1 => CellUnits::Meters,
            2 => CellUnits::Degrees,
            3 => CellUnits::Feet,
            other => CellUnits::Other(other),
        }
    }
}

/// One QMF pyramid level, from smallest (level 0) to largest.
#[derive(Debug, Clone)]
pub struct LevelInfo {
    /// Level width in cells.
    pub x_size: u32,
    /// Level height in cells.
    pub y_size: u32,
    /// Quantization bin size per band (dequantized value = bin × binsize).
    pub bin_sizes: Vec<u32>,
    /// Index of this level's first block in the global block-offset table.
    pub first_block: u32,
    /// Number of blocks across.
    pub nr_x_blocks: u32,
    /// Number of blocks down.
    pub nr_y_blocks: u32,
}

/// Parsed ECW file header (fixed part + per-level chain).
#[derive(Debug, Clone)]
pub struct EcwHeader {
    /// Format version (1 or 2; only ≤ 2 is supported).
    pub version: u8,
    /// Colour-space organisation of the stored bands.
    pub compress_format: CompressFormat,
    /// Number of QMF levels (excluding the virtual file level).
    pub num_levels: u8,
    /// Sidebands per level (always 4: LL, LH, HL, HH).
    pub nr_sidebands: u8,
    /// Full-resolution image width in cells.
    pub x_size: u32,
    /// Full-resolution image height in cells.
    pub y_size: u32,
    /// Number of bands (3 for YUV, N for multiband, 1 for greyscale).
    pub nr_bands: u16,
    /// v1 quantization scale factor (always 1 in v2 files).
    pub scale_factor: u16,
    /// Block width in cells (typically 64).
    pub x_block_size: u16,
    /// Block height in cells (typically 64).
    pub y_block_size: u16,
    /// Target compression ratio recorded at compression time (v2).
    pub compression_rate: u16,
    /// Units of the cell increments (v2).
    pub cell_units: CellUnits,
    /// Cell size in X (world units per cell, v2).
    pub cell_increment_x: f64,
    /// Cell size in Y (usually negative: north-up, v2).
    pub cell_increment_y: f64,
    /// World X of the raster origin (v2).
    pub origin_x: f64,
    /// World Y of the raster origin (v2).
    pub origin_y: f64,
    /// ER Mapper datum name, e.g. `"WGS84"` (v2).
    pub datum: String,
    /// ER Mapper projection name, e.g. `"SUTM19"` (v2).
    pub projection: String,
    /// Pyramid levels, smallest first.
    pub levels: Vec<LevelInfo>,
    /// Byte offset of the first data block (end of the block table).
    pub blocks_start: u64,
    /// Total number of data blocks over all levels.
    pub total_blocks: u32,
}

impl EcwHeader {
    /// EPSG code inferred from the ER Mapper datum/projection pair, when the
    /// pair is one of the common well-known combinations. `None` otherwise —
    /// the caller still has the raw strings.
    pub fn epsg(&self) -> Option<u32> {
        let proj = self.projection.trim();
        let datum = self.datum.trim();
        if datum.eq_ignore_ascii_case("WGS84") {
            if let Some(zone) = proj
                .strip_prefix("NUTM")
                .or_else(|| proj.strip_prefix("nutm"))
                && let Ok(z) = zone.parse::<u32>()
                && (1..=60).contains(&z)
            {
                return Some(32600 + z);
            }
            if let Some(zone) = proj
                .strip_prefix("SUTM")
                .or_else(|| proj.strip_prefix("sutm"))
                && let Ok(z) = zone.parse::<u32>()
                && (1..=60).contains(&z)
            {
                return Some(32700 + z);
            }
            if proj.eq_ignore_ascii_case("GEODETIC") {
                return Some(4326);
            }
        }
        None
    }
}

fn read_exact<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<()> {
    r.read_exact(buf)
        .map_err(|e| EcwError::Io(format!("unexpected end of header: {e}")))
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    read_exact(r, &mut b)?;
    Ok(b[0])
}

/// Header integers are big-endian (see module docs).
fn read_u16_be<R: Read>(r: &mut R) -> Result<u16> {
    let mut b = [0u8; 2];
    read_exact(r, &mut b)?;
    Ok(u16::from_be_bytes(b))
}

fn read_u32_be<R: Read>(r: &mut R) -> Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_be_bytes(b))
}

/// Doubles, unlike the header integers, are little-endian.
fn read_f64_le<R: Read>(r: &mut R) -> Result<f64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_fixed_str<R: Read>(r: &mut R, len: usize) -> Result<String> {
    let mut buf = vec![0u8; len];
    read_exact(r, &mut buf)?;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(len);
    Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
}

/// Parse the ECW header, level chain and block-offset table.
///
/// On return the reader is positioned at the first data block and the
/// returned table has `total_blocks + 1` entries (the sentinel points one
/// past the last block, so `table[i+1] - table[i]` is always a length).
/// Table offsets are relative to [`EcwHeader::blocks_start`].
pub fn parse_header<R: Read + Seek>(r: &mut R) -> Result<(EcwHeader, Vec<u64>)> {
    if read_u8(r)? != ECW_HEADER_TAG {
        return Err(EcwError::NotEcw);
    }
    let version = read_u8(r)?;
    if version == 0 || version > MAX_SUPPORTED_VERSION {
        return Err(EcwError::UnsupportedVersion(version));
    }
    let blocking_format = read_u8(r)?;
    if blocking_format != 1 {
        return Err(EcwError::Malformed(format!(
            "unknown blocking format {blocking_format} (expected 1 = BLOCKING_LEVEL)"
        )));
    }
    let compress_format = CompressFormat::from_raw(read_u8(r)?);
    let num_levels = read_u8(r)?;
    let nr_sidebands = read_u8(r)?;
    let x_size = read_u32_be(r)?;
    let y_size = read_u32_be(r)?;
    let nr_bands = read_u16_be(r)?;
    let scale_factor = read_u16_be(r)?;
    let x_block_size = read_u16_be(r)?;
    let y_block_size = read_u16_be(r)?;

    if num_levels == 0 || num_levels as usize > MAX_LEVELS {
        return Err(EcwError::Malformed(format!(
            "invalid level count {num_levels}"
        )));
    }
    if nr_sidebands != 4 {
        return Err(EcwError::Malformed(format!(
            "invalid sideband count {nr_sidebands} (expected 4)"
        )));
    }
    if nr_bands == 0 || x_size == 0 || y_size == 0 || scale_factor == 0 {
        return Err(EcwError::Malformed(
            "zero bands, dimensions or scale factor".into(),
        ));
    }
    if x_block_size == 0 || y_block_size == 0 {
        return Err(EcwError::Malformed("zero block size".into()));
    }

    // Version 2 extension: georeferencing.
    let (compression_rate, cell_units, inc_x, inc_y, org_x, org_y, datum, projection) =
        if version > 1 {
            let rate = read_u16_be(r)?;
            let units = CellUnits::from_raw(read_u8(r)?);
            let inc_x = read_f64_le(r)?;
            let inc_y = read_f64_le(r)?;
            let org_x = read_f64_le(r)?;
            let org_y = read_f64_le(r)?;
            let datum = read_fixed_str(r, MAX_DATUM_LEN)?;
            let projection = read_fixed_str(r, MAX_PROJECTION_LEN)?;
            (rate, units, inc_x, inc_y, org_x, org_y, datum, projection)
        } else {
            (
                1,
                CellUnits::Meters,
                1.0,
                1.0,
                0.0,
                0.0,
                "RAW".to_string(),
                "RAW".to_string(),
            )
        };

    // Level chain, smallest level first. Each level records its own
    // dimensions plus one u32 bin size per band.
    let mut levels = Vec::with_capacity(num_levels as usize);
    let mut total_blocks: u64 = 0;
    for level in 0..num_levels {
        let lvl = read_u8(r)?;
        if lvl != level {
            return Err(EcwError::Malformed(format!(
                "level chain out of order: expected {level}, found {lvl}"
            )));
        }
        let lx = read_u32_be(r)?;
        let ly = read_u32_be(r)?;
        if lx == 0 || ly == 0 || lx >= x_size || ly >= y_size {
            return Err(EcwError::Malformed(format!(
                "level {level} has invalid size {lx}x{ly}"
            )));
        }
        let mut bin_sizes = Vec::with_capacity(nr_bands as usize);
        for _ in 0..nr_bands {
            bin_sizes.push(read_u32_be(r)?);
        }
        let nr_x_blocks = lx.div_ceil(x_block_size as u32);
        let nr_y_blocks = ly.div_ceil(y_block_size as u32);
        levels.push(LevelInfo {
            x_size: lx,
            y_size: ly,
            bin_sizes,
            first_block: u32::try_from(total_blocks)
                .map_err(|_| EcwError::Malformed("block count overflow".into()))?,
            nr_x_blocks,
            nr_y_blocks,
        });
        total_blocks += u64::from(nr_x_blocks) * u64::from(nr_y_blocks);
    }
    let total_blocks = u32::try_from(total_blocks)
        .map_err(|_| EcwError::Malformed("block count overflow".into()))?;

    // Block-offset table: u32 BE packed length (which includes the 1-byte
    // encode format that follows), u8 encode format, then the payload.
    // Table entries are u64 little-endian; there are total_blocks + 1 of
    // them (the sentinel gives the length of the last block).
    let packed_length = read_u32_be(r)?;
    let table_format = read_u8(r)?;
    let expected = u64::from(total_blocks + 1) * 8;
    let payload_len = u64::from(packed_length)
        .checked_sub(1)
        .ok_or_else(|| EcwError::Malformed("empty block table".into()))?;

    let table: Vec<u64> = match table_format {
        // ENCODE_RAW: the table is stored verbatim.
        1 => {
            if payload_len != expected {
                return Err(EcwError::Malformed(format!(
                    "block table is {payload_len} bytes, expected {expected} \
                     for {total_blocks} blocks"
                )));
            }
            let mut raw = vec![0u8; payload_len as usize];
            read_exact(r, &mut raw)?;
            raw.chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        other => {
            // Compressed block tables (Huffman/range) exist in the format
            // but have not been observed in v2 files from current encoders;
            // fail loudly rather than guess.
            return Err(EcwError::Unsupported(format!(
                "block table encode format {other} not supported yet (only RAW)"
            )));
        }
    };

    // Sanity: offsets must be non-decreasing and start at zero.
    if table.first() != Some(&0) {
        return Err(EcwError::Malformed(
            "block table does not start at 0".into(),
        ));
    }
    if table.windows(2).any(|w| w[1] < w[0]) {
        return Err(EcwError::Malformed("block table offsets decrease".into()));
    }

    let blocks_start = r
        .stream_position()
        .map_err(|e| EcwError::Io(e.to_string()))?;

    // Cross-check the sentinel against the physical file size.
    let file_len = r
        .seek(SeekFrom::End(0))
        .map_err(|e| EcwError::Io(e.to_string()))?;
    let sentinel = *table.last().unwrap();
    if blocks_start + sentinel > file_len {
        return Err(EcwError::Malformed(format!(
            "block table sentinel {sentinel} points past end of file"
        )));
    }

    Ok((
        EcwHeader {
            version,
            compress_format,
            num_levels,
            nr_sidebands,
            x_size,
            y_size,
            nr_bands,
            scale_factor,
            x_block_size,
            y_block_size,
            compression_rate,
            cell_units,
            cell_increment_x: inc_x,
            cell_increment_y: inc_y,
            origin_x: org_x,
            origin_y: org_y,
            datum,
            projection,
            levels,
            blocks_start,
            total_blocks,
        },
        table,
    ))
}
