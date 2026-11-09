//! Native Rust decoder for ER Mapper ECW version 2 imagery.
//!
//! Reads the classic wavelet-compressed `.ecw` orthomosaic format without
//! GDAL and without the Hexagon/ERDAS SDK: pure safe Rust, no FFI. The
//! format knowledge was extracted from the publicly released libecwj2-3.3
//! sources, used strictly as a *format specification* — no code was ported.
//! `docs/ecw_format.md` at the repository root documents the format, that
//! decision, and the licensing/patent review that backs it.
//!
//! Because ECW is a multiresolution wavelet pyramid, reads at reduced
//! resolution and windowed reads are first-class: the decoder only touches
//! the levels and blocks a request needs, so gigapixel mosaics open
//! instantly at overview scale. Reconstruction is line-streamed — memory
//! scales with image width, not area.
//!
//! ```no_run
//! use surtgis_ecw::EcwReader;
//!
//! let mut reader = EcwReader::open("ortho.ecw")?;
//! let h = reader.header();
//! println!("{}x{} px, {} bands, {:?}", h.x_size, h.y_size, h.nr_bands, h.compress_format);
//!
//! // 1:16 overview of the whole mosaic, one Raster<u8> per band (RGB for YUV files)
//! let overview = reader.read_reduced(4)?;
//! # Ok::<(), surtgis_ecw::EcwError>(())
//! ```

#![deny(missing_docs)]

mod block;
mod header;
mod huffman;
mod range;
mod synthesis;

pub use header::{CellUnits, CompressFormat, EcwHeader, LevelInfo};
pub use synthesis::RegionParams;

use std::fs::File;
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use surtgis_core::raster::{AnyRaster, GeoTransform, Raster};

/// Errors produced while opening or decoding an ECW file.
#[derive(Debug, thiserror::Error)]
pub enum EcwError {
    /// The file does not begin with the ECW magic byte.
    #[error("not an ECW file (missing 'e' header tag)")]
    NotEcw,
    /// ECW v3+ (SDK 5.x era) or an unknown version byte.
    #[error("ECW version {0} is not supported (this decoder reads v1-v2)")]
    UnsupportedVersion(u8),
    /// Structurally valid ECW using a feature this decoder does not cover.
    #[error("unsupported ECW feature: {0}")]
    Unsupported(String),
    /// The file contradicts the format specification.
    #[error("malformed ECW file: {0}")]
    Malformed(String),
    /// A read request outside the image or over-sampled.
    #[error("invalid region: {0}")]
    InvalidRegion(String),
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, EcwError>;

impl From<EcwError> for surtgis_core::Error {
    fn from(e: EcwError) -> Self {
        surtgis_core::Error::Other(e.to_string())
    }
}

/// An open ECW file: parsed header plus the block-offset table.
pub struct EcwReader<R: Read + Seek = BufReader<File>> {
    file: R,
    header: EcwHeader,
    table: Vec<u64>,
}

impl EcwReader<BufReader<File>> {
    /// Open an ECW file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| EcwError::Io(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_reader(BufReader::new(file))
    }
}

impl<R: Read + Seek> EcwReader<R> {
    /// Open an ECW image from any seekable byte source.
    pub fn from_reader(mut file: R) -> Result<Self> {
        let (header, table) = header::parse_header(&mut file)?;
        Ok(Self {
            file,
            header,
            table,
        })
    }

    /// Parsed file header (dimensions, bands, georeferencing, pyramid).
    pub fn header(&self) -> &EcwHeader {
        &self.header
    }

    /// Decode a region of the image. Bounds are inclusive file-space cell
    /// coordinates; `number_x × number_y` is the output size, which may be
    /// smaller than the window (the decoder then reads from the shallowest
    /// pyramid level that covers it — this is how overviews work).
    ///
    /// Returns one `Raster<u8>` per band, georeferenced to the window.
    /// YUV files come back as R, G, B; multiband files in stored order.
    pub fn read_region(&mut self, params: RegionParams) -> Result<Vec<Raster<u8>>> {
        let bands = self.header.nr_bands as usize;
        if self.header.compress_format == CompressFormat::Yuv && bands != 3 {
            return Err(EcwError::Malformed(format!(
                "YUV file with {bands} bands (must be 3)"
            )));
        }
        if let CompressFormat::Other(v) = self.header.compress_format {
            return Err(EcwError::Unsupported(format!("compress format {v}")));
        }

        let mut region = synthesis::Region::new(&mut self.file, &self.header, &self.table, params)?;
        let (out_w, out_h) = region.output_size();
        let (rows, cols) = (out_h as usize, out_w as usize);

        let mut planes: Vec<Vec<u8>> = (0..bands).map(|_| vec![0u8; rows * cols]).collect();
        let mut line: Vec<Vec<f32>> = (0..bands).map(|_| vec![0.0f32; cols]).collect();
        let yuv = self.header.compress_format == CompressFormat::Yuv;

        for row in 0..rows {
            region.next_line(&mut line)?;
            let base = row * cols;
            if yuv {
                // JPEG-standard YCbCr, with the chroma channels already
                // centred on zero (they are signed wavelet output).
                for col in 0..cols {
                    let y = line[0][col];
                    let u = line[1][col];
                    let v = line[2][col];
                    planes[0][base + col] = clamp_u8(v.mul_add(1.402, y));
                    planes[1][base + col] = clamp_u8(u.mul_add(-0.344_14, v.mul_add(-0.714_14, y)));
                    planes[2][base + col] = clamp_u8(u.mul_add(1.772, y));
                }
            } else {
                for (plane, src) in planes.iter_mut().zip(line.iter()) {
                    for (dst, &val) in plane[base..base + cols].iter_mut().zip(src.iter()) {
                        *dst = clamp_u8(val);
                    }
                }
            }
        }

        // Georeference the window: origin shifted by the window start,
        // cells scaled by the sampling ratio.
        let (scale_x, scale_y) = region.output_cell_scale();
        drop(region);
        let h = &self.header;
        let transform = GeoTransform::new(
            h.origin_x + f64::from(params.start_x) * h.cell_increment_x,
            h.origin_y + f64::from(params.start_y) * h.cell_increment_y,
            h.cell_increment_x * scale_x,
            h.cell_increment_y * scale_y,
        );
        let crs = h.epsg().map(surtgis_core::crs::CRS::from_epsg);

        planes
            .into_iter()
            .map(|plane| {
                let mut raster = Raster::from_vec(plane, rows, cols)
                    .map_err(|e| EcwError::Malformed(e.to_string()))?;
                raster.set_transform(transform);
                raster.set_crs(crs.clone());
                Ok(raster)
            })
            .collect()
    }

    /// Decode the full image at `1 : 2^reduction` scale — `read_reduced(0)`
    /// is full resolution, `read_reduced(3)` one-eighth in each axis.
    pub fn read_reduced(&mut self, reduction: u32) -> Result<Vec<Raster<u8>>> {
        let h = self.header();
        let number_x = (h.x_size >> reduction).max(1);
        let number_y = (h.y_size >> reduction).max(1);
        let params = RegionParams {
            start_x: 0,
            start_y: 0,
            end_x: h.x_size - 1,
            end_y: h.y_size - 1,
            number_x,
            number_y,
        };
        self.read_region(params)
    }

    /// Decode the full image at full resolution. For gigapixel mosaics
    /// prefer [`EcwReader::read_region`] or [`EcwReader::read_reduced`]:
    /// this materializes every band in memory.
    pub fn read_all(&mut self) -> Result<Vec<Raster<u8>>> {
        self.read_reduced(0)
    }
}

#[inline]
fn clamp_u8(v: f32) -> u8 {
    // Round-to-nearest first (the reference decoder converts through the
    // FPU's round-to-nearest-even mode), then clamp to the byte range.
    let r = v.round_ties_even();
    if r < 0.0 {
        0
    } else if r > 255.0 {
        255
    } else {
        r as u8
    }
}

/// Read an ECW file with the same shape as
/// [`surtgis_core::io::read_geotiff_any`]: `band` selects a single band
/// (0-based), `None` returns the first. ECW v2 is always 8-bit, so the
/// result is [`AnyRaster::U8`].
pub fn read_ecw_any<P: AsRef<Path>>(
    path: P,
    band: Option<usize>,
) -> surtgis_core::Result<AnyRaster> {
    let mut reader = EcwReader::open(path)?;
    let mut bands = reader.read_all()?;
    let index = band.unwrap_or(0);
    if index >= bands.len() {
        return Err(surtgis_core::Error::Other(format!(
            "band {index} out of range: file has {} bands",
            bands.len()
        )));
    }
    Ok(AnyRaster::U8(bands.swap_remove(index)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_rounds_and_saturates() {
        assert_eq!(clamp_u8(-3.0), 0);
        assert_eq!(clamp_u8(0.4), 0);
        assert_eq!(clamp_u8(0.6), 1);
        assert_eq!(clamp_u8(254.9), 255);
        assert_eq!(clamp_u8(300.0), 255);
        // ties to even, like the reference FPU conversion
        assert_eq!(clamp_u8(2.5), 2);
        assert_eq!(clamp_u8(3.5), 4);
    }

    #[test]
    fn rejects_non_ecw() {
        let data = b"not an ecw file at all".to_vec();
        let Err(err) = EcwReader::from_reader(std::io::Cursor::new(data)) else {
            panic!("garbage input must not parse as ECW");
        };
        assert!(matches!(err, EcwError::NotEcw));
    }
}
