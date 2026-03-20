//! Strip-based GeoTIFF reader for streaming I/O.
//!
//! Reads GeoTIFF files one strip at a time, enabling bounded-memory
//! processing of arbitrarily large rasters.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use ndarray::Array2;
use tiff::decoder::{Decoder, DecodingResult, Limits};
use tiff::tags::Tag;

use crate::crs::CRS;
use crate::error::{Error, Result};
use crate::raster::GeoTransform;

/// Strip-based GeoTIFF reader.
///
/// Opens a GeoTIFF and reads strips sequentially, keeping memory usage
/// proportional to strip size rather than total image size.
pub struct StripReader {
    decoder: Decoder<BufReader<File>>,
    rows: usize,
    cols: usize,
    rows_per_strip: usize,
    num_strips: usize,
    transform: GeoTransform,
    crs: Option<CRS>,
    nodata: Option<f64>,
}

impl StripReader {
    /// Open a GeoTIFF file for strip-based reading.
    pub fn open(path: &Path) -> Result<Self> {
        let file = BufReader::new(File::open(path)?);
        let mut decoder = Decoder::new(file)
            .map_err(|e| Error::Other(format!("TIFF decode error: {}", e)))?
            .with_limits(Limits::unlimited());

        let (width, height) = decoder
            .dimensions()
            .map_err(|e| Error::Other(format!("Cannot read dimensions: {}", e)))?;

        let rows = height as usize;
        let cols = width as usize;

        // Get rows per strip from chunk dimensions
        let (_, strip_height) = decoder.chunk_dimensions();
        let rows_per_strip = strip_height as usize;
        let num_strips = decoder
            .strip_count()
            .map_err(|e| Error::Other(format!("Cannot get strip count: {}", e)))?
            as usize;

        // Read GeoTransform from tags
        let transform = read_geotransform_from_decoder(&mut decoder).unwrap_or_default();

        // Read CRS from GeoKeys
        let crs = read_crs_from_decoder(&mut decoder);

        Ok(Self {
            decoder,
            rows,
            cols,
            rows_per_strip,
            num_strips,
            transform,
            crs,
            nodata: None,
        })
    }

    /// Total number of rows in the image.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Total number of columns in the image.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of strips in the image.
    pub fn num_strips(&self) -> usize {
        self.num_strips
    }

    /// Default number of rows per strip.
    pub fn rows_per_strip(&self) -> usize {
        self.rows_per_strip
    }

    /// GeoTransform read from the file.
    pub fn transform(&self) -> &GeoTransform {
        &self.transform
    }

    /// CRS read from the file, if present.
    pub fn crs(&self) -> Option<&CRS> {
        self.crs.as_ref()
    }

    /// NoData value, if set.
    pub fn nodata(&self) -> Option<f64> {
        self.nodata
    }

    /// How many rows in a specific strip (last strip may be shorter).
    pub fn rows_in_strip(&self, strip_idx: usize) -> usize {
        if strip_idx >= self.num_strips {
            return 0;
        }
        if strip_idx == self.num_strips - 1 {
            // Last strip: remaining rows
            let remaining = self.rows - strip_idx * self.rows_per_strip;
            remaining.min(self.rows_per_strip)
        } else {
            self.rows_per_strip
        }
    }

    /// Read a single strip as `Array2<f64>`.
    pub fn read_strip(&mut self, strip_idx: usize) -> Result<Array2<f64>> {
        if strip_idx >= self.num_strips {
            return Err(Error::Other(format!(
                "Strip index {} out of range ({})",
                strip_idx, self.num_strips
            )));
        }

        let result = self
            .decoder
            .read_chunk(strip_idx as u32)
            .map_err(|e| Error::Other(format!("Cannot read strip {}: {}", strip_idx, e)))?;

        let data: Vec<f64> = decode_to_f64(result);
        let strip_rows = self.rows_in_strip(strip_idx);
        let expected = strip_rows * self.cols;

        if data.len() < expected {
            return Err(Error::Other(format!(
                "Strip {} has {} values, expected {}",
                strip_idx,
                data.len(),
                expected
            )));
        }

        // Truncate to actual rows (strip may have padding)
        let arr = Array2::from_shape_vec((strip_rows, self.cols), data[..expected].to_vec())
            .map_err(|e| Error::Other(format!("Array shape error: {}", e)))?;

        Ok(arr)
    }

    /// Read a range of rows (may span multiple strips).
    pub fn read_rows(&mut self, start_row: usize, count: usize) -> Result<Array2<f64>> {
        if count == 0 {
            return Ok(Array2::<f64>::zeros((0, self.cols)));
        }

        if start_row + count > self.rows {
            return Err(Error::Other(format!(
                "Row range {}..{} exceeds image height {}",
                start_row,
                start_row + count,
                self.rows
            )));
        }

        let mut output = Array2::<f64>::zeros((count, self.cols));
        let mut out_row = 0;

        let first_strip = start_row / self.rows_per_strip;
        let last_strip = (start_row + count - 1) / self.rows_per_strip;

        for si in first_strip..=last_strip {
            let strip_data = self.read_strip(si)?;
            let strip_start_row = si * self.rows_per_strip;
            let strip_rows = strip_data.nrows();

            // Which rows within this strip do we need?
            let local_start = if si == first_strip {
                start_row - strip_start_row
            } else {
                0
            };
            let local_end = if si == last_strip {
                (start_row + count) - strip_start_row
            } else {
                strip_rows
            };

            let rows_to_copy = local_end - local_start;
            output
                .slice_mut(ndarray::s![out_row..out_row + rows_to_copy, ..])
                .assign(&strip_data.slice(ndarray::s![local_start..local_end, ..]));
            out_row += rows_to_copy;
        }

        Ok(output)
    }
}

/// Convert a `DecodingResult` to `Vec<f64>`.
fn decode_to_f64(result: DecodingResult) -> Vec<f64> {
    match result {
        DecodingResult::F32(buf) => buf.iter().map(|&v| v as f64).collect(),
        DecodingResult::F64(buf) => buf,
        DecodingResult::U8(buf) => buf.iter().map(|&v| v as f64).collect(),
        DecodingResult::U16(buf) => buf.iter().map(|&v| v as f64).collect(),
        DecodingResult::U32(buf) => buf.iter().map(|&v| v as f64).collect(),
        DecodingResult::I8(buf) => buf.iter().map(|&v| v as f64).collect(),
        DecodingResult::I16(buf) => buf.iter().map(|&v| v as f64).collect(),
        DecodingResult::I32(buf) => buf.iter().map(|&v| v as f64).collect(),
        _ => vec![],
    }
}

/// Read GeoTransform from TIFF tags (ModelPixelScaleTag + ModelTiepointTag).
fn read_geotransform_from_decoder<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> Result<GeoTransform> {
    let scale_tag = Tag::Unknown(33550);
    let tiepoint_tag = Tag::Unknown(33922);

    let scale = decoder
        .get_tag_f64_vec(scale_tag)
        .map_err(|_| Error::Other("No pixel scale tag".into()))?;

    let tiepoint = decoder
        .get_tag_f64_vec(tiepoint_tag)
        .map_err(|_| Error::Other("No tiepoint tag".into()))?;

    if scale.len() >= 2 && tiepoint.len() >= 6 {
        let origin_x = tiepoint[3] - tiepoint[0] * scale[0];
        let origin_y = tiepoint[4] + tiepoint[1] * scale[1];
        let pixel_width = scale[0];
        let pixel_height = -scale[1];
        return Ok(GeoTransform::new(
            origin_x,
            origin_y,
            pixel_width,
            pixel_height,
        ));
    }

    Err(Error::Other("Cannot determine geotransform".into()))
}

/// Read CRS EPSG from GeoKeyDirectory tag (34735).
fn read_crs_from_decoder<R: std::io::Read + std::io::Seek>(
    decoder: &mut Decoder<R>,
) -> Option<CRS> {
    let geokeys = decoder.get_tag_u16_vec(Tag::Unknown(34735)).ok()?;
    // GeoKeyDirectory: [version, revision, minor, num_keys, key_id, loc, count, value, ...]
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
            return Some(CRS::from_epsg(value as u32));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_reader_open() {
        let path = Path::new("../../benchmarks/results/dems/fbm_1000_raw.tif");
        if !path.exists() {
            return;
        } // skip if test data not available

        let reader = StripReader::open(path).unwrap();
        assert_eq!(reader.rows(), 1000);
        assert_eq!(reader.cols(), 1000);
        assert!(reader.num_strips() > 0);
    }

    #[test]
    fn test_strip_reader_read_strip() {
        let path = Path::new("../../benchmarks/results/dems/fbm_1000_raw.tif");
        if !path.exists() {
            return;
        }

        let mut reader = StripReader::open(path).unwrap();
        let strip = reader.read_strip(0).unwrap();
        assert_eq!(strip.ncols(), 1000);
        assert!(strip.nrows() > 0);
    }

    #[test]
    fn test_strip_reader_read_rows() {
        let path = Path::new("../../benchmarks/results/dems/fbm_1000_raw.tif");
        if !path.exists() {
            return;
        }

        let mut reader = StripReader::open(path).unwrap();
        let rows = reader.read_rows(10, 50).unwrap();
        assert_eq!(rows.dim(), (50, 1000));
    }

    #[test]
    fn test_strip_reader_full_equals_standard() {
        // Read all rows via strip reader and compare to standard reader
        let path = Path::new("../../benchmarks/results/dems/fbm_1000_raw.tif");
        if !path.exists() {
            return;
        }

        let mut reader = StripReader::open(path).unwrap();
        let streaming = reader.read_rows(0, reader.rows()).unwrap();

        let standard: crate::raster::Raster<f64> = crate::io::read_geotiff(path, None).unwrap();
        let standard_data = standard.data();

        assert_eq!(streaming.dim(), standard_data.dim());
        for r in 0..streaming.nrows() {
            for c in 0..streaming.ncols() {
                let s = streaming[[r, c]];
                let t = standard_data[[r, c]];
                assert!(
                    (s - t).abs() < 1e-6 || (s.is_nan() && t.is_nan()),
                    "Mismatch at ({}, {}): {} vs {}",
                    r,
                    c,
                    s,
                    t
                );
            }
        }
    }
}
