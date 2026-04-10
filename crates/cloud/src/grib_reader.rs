//! GRIB2 reader for meteorological data (HRRR, ECMWF, MRMS).
//!
//! Reads local GRIB2 files containing weather forecast and observation data.
//! Each GRIB2 file contains multiple messages (one per variable/level combination).

use std::path::{Path, PathBuf};

use ndarray::Array2;

use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::CRS;

use crate::error::{CloudError, Result};
use crate::tile_index::BBox;

/// Information about a single GRIB2 message.
#[derive(Debug, Clone)]
pub struct GribMessageInfo {
    pub index: usize,
    pub message_index: (usize, usize),
    pub description: String,
}

/// Metadata about an opened GRIB2 file.
#[derive(Debug, Clone)]
pub struct GribMetadata {
    pub path: PathBuf,
    pub message_count: usize,
    pub messages: Vec<GribMessageInfo>,
}

/// Reader for local GRIB2 files.
pub struct GribReader {
    data: Vec<u8>,
    metadata: GribMetadata,
}

impl GribReader {
    /// Open a GRIB2 file and index all messages.
    pub fn open(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| CloudError::Grib(format!("failed to read {}: {e}", path.display())))?;
        Self::from_bytes(data, &path.display().to_string())
    }

    /// Open from bytes already in memory.
    pub fn from_bytes(data: Vec<u8>, name: &str) -> Result<Self> {
        let messages = index_messages(&data)?;
        let metadata = GribMetadata {
            path: PathBuf::from(name),
            message_count: messages.len(),
            messages,
        };
        Ok(Self { data, metadata })
    }

    /// Read a specific message by sequential index as a georeferenced raster.
    pub fn read_message(&self, index: usize) -> Result<Raster<f64>> {
        use grib::LatLons;

        let grib2 = grib::from_bytes(&self.data)
            .map_err(|e| CloudError::Grib(format!("parse: {e}")))?;

        let (_msg_idx, submessage) = grib2
            .iter()
            .nth(index)
            .ok_or_else(|| CloudError::Grib(format!("message {index} out of range")))?;

        // Get lat/lon coordinates
        let latlons: Vec<(f64, f64)> = submessage
            .latlons()
            .map_err(|e| CloudError::Grib(format!("grid: {e}")))?
            .map(|(lat, lon)| (lat as f64, lon as f64))
            .collect();

        if latlons.is_empty() {
            return Err(CloudError::Grib("empty grid".into()));
        }

        // Decode values
        let decoder = grib::Grib2SubmessageDecoder::from(submessage)
            .map_err(|e| CloudError::Grib(format!("decoder: {e}")))?;
        let values: Vec<f64> = decoder
            .dispatch()
            .map_err(|e| CloudError::Grib(format!("decode: {e}")))?
            .map(|v| v as f64)
            .collect();

        // Infer grid shape and build raster
        let (nrows, ncols) = infer_grid_shape(&latlons)?;

        if values.len() != nrows * ncols {
            return Err(CloudError::Grib(format!(
                "shape mismatch: {}x{}={} vs {} values",
                nrows, ncols, nrows * ncols, values.len()
            )));
        }

        let gt = build_geotransform(&latlons, nrows, ncols);
        let data = Array2::from_shape_vec((nrows, ncols), values)
            .map_err(|e| CloudError::Grib(format!("reshape: {e}")))?;

        let mut raster = Raster::from_array(data);
        raster.set_transform(gt);
        raster.set_crs(Some(CRS::wgs84()));
        Ok(raster)
    }

    /// Read the first message matching a description (case-insensitive substring).
    pub fn read_by_parameter(&self, param: &str) -> Result<Raster<f64>> {
        let lower = param.to_lowercase();
        let idx = self
            .metadata
            .messages
            .iter()
            .find(|m| m.description.to_lowercase().contains(&lower))
            .map(|m| m.index)
            .ok_or_else(|| {
                let avail: Vec<_> = self.metadata.messages.iter()
                    .take(20)
                    .map(|m| m.description.as_str())
                    .collect();
                CloudError::Grib(format!(
                    "no message matching '{}'. First {}:\n  {}",
                    param, avail.len(), avail.join("\n  ")
                ))
            })?;
        self.read_message(idx)
    }

    /// Read a message cropped to a bounding box.
    pub fn read_bbox(&self, index: usize, bbox: &BBox) -> Result<Raster<f64>> {
        let full = self.read_message(index)?;
        crop_raster(&full, bbox)
    }

    /// Read by parameter name, cropped to bbox.
    pub fn read_bbox_by_parameter(&self, param: &str, bbox: &BBox) -> Result<Raster<f64>> {
        let full = self.read_by_parameter(param)?;
        crop_raster(&full, bbox)
    }

    pub fn metadata(&self) -> &GribMetadata { &self.metadata }
    pub fn list_messages(&self) -> &[GribMessageInfo] { &self.metadata.messages }
}

// ─── Internal ────────────────────────────────────────────────────────

fn index_messages(data: &[u8]) -> Result<Vec<GribMessageInfo>> {
    let grib2 = grib::from_bytes(data)
        .map_err(|e| CloudError::Grib(format!("parse: {e}")))?;

    let mut messages = Vec::new();
    for (index, (msg_idx, submessage)) in grib2.iter().enumerate() {
        messages.push(GribMessageInfo {
            index,
            message_index: msg_idx,
            description: submessage.describe(),
        });
    }
    Ok(messages)
}

fn infer_grid_shape(latlons: &[(f64, f64)]) -> Result<(usize, usize)> {
    if latlons.len() < 2 {
        return Err(CloudError::Grib("too few grid points".into()));
    }

    // Determine scan direction from the first two points
    let (lat0, lon0) = latlons[0];
    let (lat1, lon1) = latlons[1];

    if (lon1 - lon0).abs() > (lat1 - lat0).abs() {
        // Scan direction: longitude changes first (standard i+ scan)
        let ncols = latlons
            .iter()
            .take_while(|(lat, _)| (*lat - lat0).abs() < 1e-4)
            .count();
        if ncols == 0 {
            return Err(CloudError::Grib("cannot determine grid columns".into()));
        }
        let nrows = latlons.len() / ncols;
        if nrows * ncols != latlons.len() {
            return Err(CloudError::Grib(format!(
                "non-regular grid: {} pts ≠ {}×{}", latlons.len(), nrows, ncols
            )));
        }
        Ok((nrows, ncols))
    } else {
        // Scan direction: latitude changes first (j+ scan)
        let nrows = latlons
            .iter()
            .take_while(|(_, lon)| (*lon - lon0).abs() < 1e-4)
            .count();
        if nrows == 0 {
            return Err(CloudError::Grib("cannot determine grid rows".into()));
        }
        let ncols = latlons.len() / nrows;
        if nrows * ncols != latlons.len() {
            return Err(CloudError::Grib(format!(
                "non-regular grid: {} pts ≠ {}×{}", latlons.len(), nrows, ncols
            )));
        }
        Ok((nrows, ncols))
    }
}

fn build_geotransform(latlons: &[(f64, f64)], nrows: usize, ncols: usize) -> GeoTransform {
    // Use step from first two adjacent points (avoids wrap-around issues)
    let lat_first = latlons[0].0;
    let lon_first = latlons[0].1;

    // Pixel width: from first two points in the same row
    let pw = if ncols > 1 {
        latlons[1].1 - latlons[0].1
    } else {
        1.0
    };

    // Pixel height: from first two rows
    let ph = if nrows > 1 {
        latlons[ncols].0 - latlons[0].0
    } else {
        -1.0
    };

    GeoTransform::new(lon_first - pw / 2.0, lat_first - ph / 2.0, pw, ph)
}

fn crop_raster(raster: &Raster<f64>, bbox: &BBox) -> Result<Raster<f64>> {
    let t = raster.transform();
    let (rows, cols) = raster.shape();

    // Detect 0-360 convention and adjust bbox
    let (min_x, max_x) = if t.origin_x >= 0.0 && bbox.min_x < 0.0 {
        // Raster uses 0-360, bbox uses -180..180
        let mx = if bbox.min_x < 0.0 { bbox.min_x + 360.0 } else { bbox.min_x };
        let mxx = if bbox.max_x < 0.0 { bbox.max_x + 360.0 } else { bbox.max_x };
        (mx, mxx)
    } else {
        (bbox.min_x, bbox.max_x)
    };

    let (c0f, r0f) = t.geo_to_pixel(min_x, bbox.max_y);
    let (c1f, r1f) = t.geo_to_pixel(max_x, bbox.min_y);

    let c0 = (c0f.floor() as isize).max(0) as usize;
    let r0 = (r0f.floor() as isize).max(0) as usize;
    let c1 = (c1f.ceil() as usize).min(cols);
    let r1 = (r1f.ceil() as usize).min(rows);

    if c0 >= c1 || r0 >= r1 {
        return Err(CloudError::BBoxOutside);
    }

    let sub = raster.data().slice(ndarray::s![r0..r1, c0..c1]).to_owned();
    let (ox, oy) = t.pixel_to_geo(c0, r0);

    let mut out = Raster::from_array(sub);
    out.set_transform(GeoTransform::new(ox, oy, t.pixel_width, t.pixel_height));
    out.set_crs(raster.crs().cloned());
    out.set_nodata(raster.nodata());
    Ok(out)
}
