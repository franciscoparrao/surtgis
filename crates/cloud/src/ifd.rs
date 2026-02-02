//! Custom TIFF IFD (Image File Directory) parser for COG files.
//!
//! Parses IFD entries from raw bytes fetched via HTTP Range requests,
//! without requiring `Read + Seek`. Supports IFD chains (overviews).

use byteorder::{BigEndian, ByteOrder, LittleEndian};

use crate::error::{CloudError, Result};

/// Byte order of the TIFF file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffByteOrder {
    LittleEndian,
    BigEndian,
}

/// Well-known TIFF tag IDs.
#[allow(dead_code)]
pub mod tags {
    pub const IMAGE_WIDTH: u16 = 256;
    pub const IMAGE_LENGTH: u16 = 257;
    pub const BITS_PER_SAMPLE: u16 = 258;
    pub const COMPRESSION: u16 = 259;
    pub const PHOTOMETRIC: u16 = 262;
    pub const SAMPLES_PER_PIXEL: u16 = 277;
    pub const PLANAR_CONFIG: u16 = 284;
    pub const TILE_WIDTH: u16 = 322;
    pub const TILE_LENGTH: u16 = 323;
    pub const TILE_OFFSETS: u16 = 324;
    pub const TILE_BYTE_COUNTS: u16 = 325;
    pub const SAMPLE_FORMAT: u16 = 339;
    pub const MODEL_PIXEL_SCALE: u16 = 33550;
    pub const MODEL_TIEPOINT: u16 = 33922;
    pub const MODEL_TRANSFORMATION: u16 = 34264;
    pub const GEO_KEY_DIRECTORY: u16 = 34735;
    pub const GEO_DOUBLE_PARAMS: u16 = 34736;
    pub const GEO_ASCII_PARAMS: u16 = 34737;
    pub const GDAL_NODATA: u16 = 42113;
}

/// TIFF data type IDs and their byte sizes.
fn type_byte_size(type_id: u16) -> Option<usize> {
    match type_id {
        1 => Some(1),   // BYTE
        2 => Some(1),   // ASCII
        3 => Some(2),   // SHORT
        4 => Some(4),   // LONG
        5 => Some(8),   // RATIONAL
        6 => Some(1),   // SBYTE
        7 => Some(1),   // UNDEFINED
        8 => Some(2),   // SSHORT
        9 => Some(4),   // SLONG
        10 => Some(8),  // SRATIONAL
        11 => Some(4),  // FLOAT
        12 => Some(8),  // DOUBLE
        16 => Some(8),  // LONG8 (BigTIFF)
        _ => None,
    }
}

/// A raw IFD tag entry before value resolution.
#[derive(Debug, Clone)]
pub struct RawTagEntry {
    pub tag: u16,
    pub type_id: u16,
    pub count: u32,
    /// If the value fits in 4 bytes, it's inline; otherwise this is the file offset.
    pub value_or_offset: u32,
    /// True if the value data is inline (fits in 4 bytes).
    pub inline: bool,
}

/// Parsed TIFF header.
#[derive(Debug, Clone)]
pub struct TiffHeader {
    pub byte_order: TiffByteOrder,
    pub first_ifd_offset: u32,
}

/// A single parsed IFD with all tag entries and the offset to the next IFD.
#[derive(Debug, Clone)]
pub struct RawIfd {
    pub entries: Vec<RawTagEntry>,
    pub next_ifd_offset: u32,
}

/// Information extracted from one IFD relevant to COG reading.
#[derive(Debug, Clone)]
pub struct IfdInfo {
    pub width: u32,
    pub height: u32,
    pub tile_width: u32,
    pub tile_height: u32,
    pub tile_offsets: Vec<u64>,
    pub tile_byte_counts: Vec<u64>,
    pub bits_per_sample: u16,
    pub sample_format: u16,
    pub compression: u16,
    pub samples_per_pixel: u16,
    pub planar_config: u16,
    /// Raw IFD entries for GeoTIFF key extraction.
    pub raw_entries: Vec<RawTagEntry>,
}

/// Parse the 8-byte TIFF header.
pub fn parse_header(data: &[u8]) -> Result<TiffHeader> {
    if data.len() < 8 {
        return Err(CloudError::InvalidTiff {
            reason: "header too short".into(),
        });
    }

    let byte_order = match (data[0], data[1]) {
        (b'I', b'I') => TiffByteOrder::LittleEndian,
        (b'M', b'M') => TiffByteOrder::BigEndian,
        _ => {
            return Err(CloudError::InvalidTiff {
                reason: "invalid byte order marker".into(),
            });
        }
    };

    let magic = read_u16(byte_order, &data[2..4]);
    if magic != 42 {
        return Err(CloudError::InvalidTiff {
            reason: format!("expected magic 42, got {}", magic),
        });
    }

    let first_ifd_offset = read_u32(byte_order, &data[4..8]);

    Ok(TiffHeader {
        byte_order,
        first_ifd_offset,
    })
}

/// Parse one IFD from raw bytes.
///
/// `data` must start at the IFD offset and contain enough bytes to parse
/// all entries plus the 4-byte next-IFD pointer.
pub fn parse_ifd(byte_order: TiffByteOrder, data: &[u8]) -> Result<RawIfd> {
    if data.len() < 2 {
        return Err(CloudError::InvalidTiff {
            reason: "IFD too short".into(),
        });
    }

    let entry_count = read_u16(byte_order, &data[0..2]) as usize;
    let needed = 2 + entry_count * 12 + 4;

    if data.len() < needed {
        return Err(CloudError::InvalidTiff {
            reason: format!(
                "IFD needs {} bytes but only {} available",
                needed,
                data.len()
            ),
        });
    }

    let mut entries = Vec::with_capacity(entry_count);
    for i in 0..entry_count {
        let offset = 2 + i * 12;
        let tag = read_u16(byte_order, &data[offset..offset + 2]);
        let type_id = read_u16(byte_order, &data[offset + 2..offset + 4]);
        let count = read_u32(byte_order, &data[offset + 4..offset + 8]);
        let value_or_offset = read_u32(byte_order, &data[offset + 8..offset + 12]);

        let total_bytes = type_byte_size(type_id).unwrap_or(1) as u64 * count as u64;
        let inline = total_bytes <= 4;

        entries.push(RawTagEntry {
            tag,
            type_id,
            count,
            value_or_offset,
            inline,
        });
    }

    let next_offset_pos = 2 + entry_count * 12;
    let next_ifd_offset = read_u32(byte_order, &data[next_offset_pos..next_offset_pos + 4]);

    Ok(RawIfd {
        entries,
        next_ifd_offset,
    })
}

/// Extract a single u16 value from an inline tag entry.
pub fn inline_u16(byte_order: TiffByteOrder, entry: &RawTagEntry) -> u16 {
    let bytes = match byte_order {
        TiffByteOrder::LittleEndian => entry.value_or_offset.to_le_bytes(),
        TiffByteOrder::BigEndian => entry.value_or_offset.to_be_bytes(),
    };
    read_u16(byte_order, &bytes[0..2])
}

/// Extract a single u32 value from an inline tag entry.
pub fn inline_u32(_byte_order: TiffByteOrder, entry: &RawTagEntry) -> u32 {
    entry.value_or_offset
}

/// Read an array of u32 or u16 values from external data at the offset indicated by a tag.
///
/// `file_data` is the raw bytes starting at `entry.value_or_offset`, of length
/// at least `entry.count * type_size`.
pub fn read_offset_values_u64(
    byte_order: TiffByteOrder,
    entry: &RawTagEntry,
    data: &[u8],
) -> Vec<u64> {
    let count = entry.count as usize;
    let mut values = Vec::with_capacity(count);

    match entry.type_id {
        3 => {
            // SHORT
            for i in 0..count {
                let off = i * 2;
                if off + 2 <= data.len() {
                    values.push(read_u16(byte_order, &data[off..off + 2]) as u64);
                }
            }
        }
        4 => {
            // LONG
            for i in 0..count {
                let off = i * 4;
                if off + 4 <= data.len() {
                    values.push(read_u32(byte_order, &data[off..off + 4]) as u64);
                }
            }
        }
        16 => {
            // LONG8
            for i in 0..count {
                let off = i * 8;
                if off + 8 <= data.len() {
                    values.push(read_u64(byte_order, &data[off..off + 8]));
                }
            }
        }
        _ => {}
    }

    values
}

/// Read an array of f64 values from data at an offset.
pub fn read_offset_values_f64(
    byte_order: TiffByteOrder,
    entry: &RawTagEntry,
    data: &[u8],
) -> Vec<f64> {
    let count = entry.count as usize;
    let mut values = Vec::with_capacity(count);

    match entry.type_id {
        12 => {
            // DOUBLE
            for i in 0..count {
                let off = i * 8;
                if off + 8 <= data.len() {
                    values.push(read_f64(byte_order, &data[off..off + 8]));
                }
            }
        }
        11 => {
            // FLOAT
            for i in 0..count {
                let off = i * 4;
                if off + 4 <= data.len() {
                    values.push(read_f32(byte_order, &data[off..off + 4]) as f64);
                }
            }
        }
        _ => {}
    }

    values
}

/// Read ASCII string from data at an offset.
pub fn read_offset_ascii(entry: &RawTagEntry, data: &[u8]) -> String {
    let len = entry.count as usize;
    let bytes = &data[..len.min(data.len())];
    // TIFF ASCII is null-terminated
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).to_string()
}

/// Calculate the total byte size needed to read a tag's out-of-line value.
pub fn external_value_size(entry: &RawTagEntry) -> u64 {
    type_byte_size(entry.type_id).unwrap_or(1) as u64 * entry.count as u64
}

// ---- Byte order helpers ----

fn read_u16(order: TiffByteOrder, data: &[u8]) -> u16 {
    match order {
        TiffByteOrder::LittleEndian => LittleEndian::read_u16(data),
        TiffByteOrder::BigEndian => BigEndian::read_u16(data),
    }
}

fn read_u32(order: TiffByteOrder, data: &[u8]) -> u32 {
    match order {
        TiffByteOrder::LittleEndian => LittleEndian::read_u32(data),
        TiffByteOrder::BigEndian => BigEndian::read_u32(data),
    }
}

fn read_u64(order: TiffByteOrder, data: &[u8]) -> u64 {
    match order {
        TiffByteOrder::LittleEndian => LittleEndian::read_u64(data),
        TiffByteOrder::BigEndian => BigEndian::read_u64(data),
    }
}

fn read_f32(order: TiffByteOrder, data: &[u8]) -> f32 {
    match order {
        TiffByteOrder::LittleEndian => LittleEndian::read_f32(data),
        TiffByteOrder::BigEndian => BigEndian::read_f32(data),
    }
}

fn read_f64(order: TiffByteOrder, data: &[u8]) -> f64 {
    match order {
        TiffByteOrder::LittleEndian => LittleEndian::read_f64(data),
        TiffByteOrder::BigEndian => BigEndian::read_f64(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_header_le() {
        // II (little-endian), magic 42, first IFD at offset 8
        let mut data = vec![b'I', b'I', 42, 0, 8, 0, 0, 0];
        let header = parse_header(&data).unwrap();
        assert_eq!(header.byte_order, TiffByteOrder::LittleEndian);
        assert_eq!(header.first_ifd_offset, 8);

        // Big-endian
        data = vec![b'M', b'M', 0, 42, 0, 0, 0, 8];
        let header = parse_header(&data).unwrap();
        assert_eq!(header.byte_order, TiffByteOrder::BigEndian);
        assert_eq!(header.first_ifd_offset, 8);
    }

    #[test]
    fn test_parse_ifd_empty() {
        // 0 entries, next IFD = 0
        let data = vec![0, 0, 0, 0, 0, 0];
        let ifd = parse_ifd(TiffByteOrder::LittleEndian, &data).unwrap();
        assert_eq!(ifd.entries.len(), 0);
        assert_eq!(ifd.next_ifd_offset, 0);
    }

    #[test]
    fn test_parse_ifd_one_entry() {
        let mut data = Vec::new();
        // entry count = 1 (LE)
        data.extend_from_slice(&1u16.to_le_bytes());
        // tag = 256 (ImageWidth)
        data.extend_from_slice(&256u16.to_le_bytes());
        // type = 3 (SHORT)
        data.extend_from_slice(&3u16.to_le_bytes());
        // count = 1
        data.extend_from_slice(&1u32.to_le_bytes());
        // value = 512
        data.extend_from_slice(&512u32.to_le_bytes());
        // next IFD = 0
        data.extend_from_slice(&0u32.to_le_bytes());

        let ifd = parse_ifd(TiffByteOrder::LittleEndian, &data).unwrap();
        assert_eq!(ifd.entries.len(), 1);
        assert_eq!(ifd.entries[0].tag, 256);
        assert_eq!(ifd.entries[0].type_id, 3);
        assert_eq!(ifd.entries[0].count, 1);
        assert!(ifd.entries[0].inline);
        assert_eq!(inline_u16(TiffByteOrder::LittleEndian, &ifd.entries[0]), 512);
    }
}
