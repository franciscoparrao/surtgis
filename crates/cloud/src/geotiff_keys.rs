//! GeoTIFF key extraction from IFD entries.
//!
//! Reads tags 33550 (ModelPixelScale), 33922 (ModelTiepoint),
//! 34264 (ModelTransformation), 34735 (GeoKeyDirectory), and
//! 42113 (GDAL_NODATA) to produce a `GeoTransform`, optional `CRS`,
//! and optional nodata value.

use surtgis_core::raster::GeoTransform;
use surtgis_core::crs::CRS;

use crate::ifd::{self, RawTagEntry, TiffByteOrder, tags};

/// GeoTIFF metadata extracted from IFD entries.
#[derive(Debug, Clone)]
pub struct GeoTiffMeta {
    pub geo_transform: GeoTransform,
    pub crs: Option<CRS>,
    pub nodata: Option<f64>,
}

/// Extract GeoTIFF metadata from resolved IFD tag values.
///
/// `resolved` is a list of `(tag_id, raw_bytes)` for tags whose values
/// were fetched from external offsets. `entries` are the raw tag entries
/// for inline value extraction.
pub fn extract_geotiff_meta(
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
) -> GeoTiffMeta {
    let geo_transform = extract_geotransform(byte_order, entries, resolved);
    let crs = extract_crs(byte_order, entries, resolved);
    let nodata = extract_nodata(entries, resolved);

    GeoTiffMeta {
        geo_transform,
        crs,
        nodata,
    }
}

/// Extract GeoTransform from ModelPixelScale + ModelTiepoint, or from
/// ModelTransformation matrix.
fn extract_geotransform(
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
) -> GeoTransform {
    // Try ModelPixelScale (33550) + ModelTiepoint (33922)
    let scale = find_resolved_f64(byte_order, entries, resolved, tags::MODEL_PIXEL_SCALE);
    let tiepoint = find_resolved_f64(byte_order, entries, resolved, tags::MODEL_TIEPOINT);

    if let (Some(scale), Some(tiepoint)) = (&scale, &tiepoint) {
        if scale.len() >= 2 && tiepoint.len() >= 6 {
            let origin_x = tiepoint[3] - tiepoint[0] * scale[0];
            let origin_y = tiepoint[4] + tiepoint[1] * scale[1];
            let pixel_width = scale[0];
            let pixel_height = -scale[1];
            return GeoTransform::new(origin_x, origin_y, pixel_width, pixel_height);
        }
    }

    // Try ModelTransformation (34264) — 4x4 matrix, row-major
    let transform = find_resolved_f64(byte_order, entries, resolved, tags::MODEL_TRANSFORMATION);
    if let Some(t) = &transform {
        if t.len() >= 16 {
            // Row-major 4x4: [a b c d / e f g h / ...]
            // GeoTransform from first two rows:
            // x = t[3] + col * t[0] + row * t[1]
            // y = t[7] + col * t[4] + row * t[5]
            return GeoTransform {
                origin_x: t[3],
                origin_y: t[7],
                pixel_width: t[0],
                pixel_height: t[5],
                row_rotation: t[1],
                col_rotation: t[4],
            };
        }
    }

    GeoTransform::default()
}

/// Extract CRS from GeoKeyDirectory (34735).
fn extract_crs(
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
) -> Option<CRS> {
    let entry = entries.iter().find(|e| e.tag == tags::GEO_KEY_DIRECTORY)?;
    let data = if entry.inline {
        return None; // GeoKeyDirectory is always external
    } else {
        resolved.iter().find(|(tag, _)| *tag == tags::GEO_KEY_DIRECTORY)?.1.as_slice()
    };

    // GeoKeyDirectory: [version, revision, minor, count, key1_id, key1_loc, key1_count, key1_value, ...]
    if data.len() < 8 {
        return None;
    }

    let count = ifd::read_offset_values_u64(byte_order, entry, data);
    if count.len() < 4 {
        return None;
    }

    let num_keys = count[3] as usize;

    // Parse key entries looking for ProjectedCSTypeGeoKey (3072) or
    // GeographicTypeGeoKey (2048)
    for i in 0..num_keys {
        let base = 4 + i * 4;
        if base + 4 > count.len() {
            break;
        }
        let key_id = count[base] as u16;
        let _tiff_tag_location = count[base + 1] as u16;
        let _key_count = count[base + 2];
        let value_or_index = count[base + 3] as u32;

        match key_id {
            // ProjectedCSTypeGeoKey
            3072 if value_or_index > 0 => {
                return Some(CRS::from_epsg(value_or_index));
            }
            // GeographicTypeGeoKey
            2048 if value_or_index > 0 => {
                return Some(CRS::from_epsg(value_or_index));
            }
            _ => {}
        }
    }

    None
}

/// Extract nodata from GDAL_NODATA tag (42113) — ASCII string.
fn extract_nodata(
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
) -> Option<f64> {
    let entry = entries.iter().find(|e| e.tag == tags::GDAL_NODATA)?;

    if entry.inline {
        // Unlikely for ASCII, but handle 4-byte inline case
        let bytes = entry.value_or_offset.to_le_bytes();
        let s = std::str::from_utf8(&bytes).ok()?;
        return s.trim_end_matches('\0').trim().parse::<f64>().ok();
    }

    let data = resolved.iter().find(|(tag, _)| *tag == tags::GDAL_NODATA)?.1.as_slice();
    let s = ifd::read_offset_ascii(entry, data);
    s.trim().parse::<f64>().ok()
}

/// Helper: find resolved f64 values for a specific tag.
fn find_resolved_f64(
    byte_order: TiffByteOrder,
    entries: &[RawTagEntry],
    resolved: &[(u16, Vec<u8>)],
    tag_id: u16,
) -> Option<Vec<f64>> {
    let entry = entries.iter().find(|e| e.tag == tag_id)?;
    let data = resolved.iter().find(|(tag, _)| *tag == tag_id)?.1.as_slice();
    let values = ifd::read_offset_values_f64(byte_order, entry, data);
    if values.is_empty() {
        None
    } else {
        Some(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geotransform_from_scale_tiepoint() {
        // Simulate scale = [10.0, 10.0, 0.0], tiepoint = [0, 0, 0, 100.0, 200.0, 0]
        let bo = TiffByteOrder::LittleEndian;

        let scale_entry = RawTagEntry {
            tag: tags::MODEL_PIXEL_SCALE,
            type_id: 12, // DOUBLE
            count: 3,
            value_or_offset: 0,
            inline: false,
        };
        let tiepoint_entry = RawTagEntry {
            tag: tags::MODEL_TIEPOINT,
            type_id: 12,
            count: 6,
            value_or_offset: 0,
            inline: false,
        };

        let mut scale_data = Vec::new();
        for v in &[10.0f64, 10.0, 0.0] {
            scale_data.extend_from_slice(&v.to_le_bytes());
        }

        let mut tiepoint_data = Vec::new();
        for v in &[0.0f64, 0.0, 0.0, 100.0, 200.0, 0.0] {
            tiepoint_data.extend_from_slice(&v.to_le_bytes());
        }

        let entries = vec![scale_entry, tiepoint_entry];
        let resolved = vec![
            (tags::MODEL_PIXEL_SCALE, scale_data),
            (tags::MODEL_TIEPOINT, tiepoint_data),
        ];

        let meta = extract_geotiff_meta(bo, &entries, &resolved);
        assert_eq!(meta.geo_transform.origin_x, 100.0);
        assert_eq!(meta.geo_transform.origin_y, 200.0);
        assert_eq!(meta.geo_transform.pixel_width, 10.0);
        assert_eq!(meta.geo_transform.pixel_height, -10.0);
    }
}
