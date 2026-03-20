//! Strip-based GeoTIFF writer for streaming I/O.
//!
//! Writes GeoTIFF files one strip at a time using a callback-based
//! approach, enabling bounded-memory output of large rasters.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use ndarray::Array2;
use tiff::encoder::colortype::Gray32Float;
use tiff::encoder::compression::DeflateLevel;
use tiff::encoder::{Compression, TiffEncoder};
use tiff::tags::Tag;

use crate::crs::CRS;
use crate::error::{Error, Result};
use crate::raster::GeoTransform;

/// Configuration for streaming GeoTIFF writing.
pub struct StripWriterConfig {
    /// Total number of rows in the output image.
    pub rows: usize,
    /// Total number of columns in the output image.
    pub cols: usize,
    /// GeoTransform for georeferencing.
    pub transform: GeoTransform,
    /// Optional CRS to embed.
    pub crs: Option<CRS>,
    /// Whether to use DEFLATE compression.
    pub compress: bool,
    /// Number of rows per output strip.
    pub rows_per_strip: u32,
}

/// Write a GeoTIFF strip by strip using a callback.
///
/// The callback `produce_strip` is called for each strip with `(strip_index,
/// expected_rows)`. It must return an `Array2<f64>` with exactly
/// `expected_rows x cols` elements.
///
/// # Example
///
/// ```no_run
/// use surtgis_core::io::strip_writer::{write_geotiff_streaming, StripWriterConfig};
/// use surtgis_core::raster::GeoTransform;
/// use ndarray::Array2;
/// use std::path::Path;
///
/// let config = StripWriterConfig {
///     rows: 1000,
///     cols: 1000,
///     transform: GeoTransform::new(0.0, 1000.0, 10.0, -10.0),
///     crs: None,
///     compress: false,
///     rows_per_strip: 256,
/// };
///
/// write_geotiff_streaming(Path::new("output.tif"), &config, |strip_idx, strip_rows| {
///     Ok(Array2::<f64>::zeros((strip_rows, 1000)))
/// }).unwrap();
/// ```
pub fn write_geotiff_streaming<F>(
    path: &Path,
    config: &StripWriterConfig,
    mut produce_strip: F,
) -> Result<()>
where
    F: FnMut(usize, usize) -> Result<Array2<f64>>,
{
    let file = BufWriter::new(File::create(path)?);

    let compression = if config.compress {
        Compression::Deflate(DeflateLevel::Balanced)
    } else {
        Compression::Uncompressed
    };

    let mut encoder = TiffEncoder::new(file)
        .map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?
        .with_compression(compression);

    let mut image = encoder
        .new_image::<Gray32Float>(config.cols as u32, config.rows as u32)
        .map_err(|e| Error::Other(format!("Cannot create TIFF image: {}", e)))?;

    // Set rows per strip (must be called before any write_strip)
    image
        .rows_per_strip(config.rows_per_strip)
        .map_err(|e| Error::Other(format!("Cannot set rows_per_strip: {}", e)))?;

    // Write GeoTIFF tags via the underlying DirectoryEncoder
    let gt = &config.transform;

    // ModelPixelScaleTag (33550)
    let scale = vec![gt.pixel_width, gt.pixel_height.abs(), 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33550), scale.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write scale tag: {}", e)))?;

    // ModelTiepointTag (33922)
    let tiepoint = vec![0.0, 0.0, 0.0, gt.origin_x, gt.origin_y, 0.0];
    image
        .encoder()
        .write_tag(Tag::Unknown(33922), tiepoint.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write tiepoint tag: {}", e)))?;

    // GeoKeyDirectoryTag (34735)
    let geokeys: Vec<u16> = if let Some(ref crs) = config.crs {
        if let Some(epsg) = crs.epsg() {
            if epsg == 4326 {
                // Geographic CRS (WGS84)
                vec![
                    1, 1, 0, 3, // Version 1.1.0, 3 keys
                    1024, 0, 1, 2, // GTModelTypeGeoKey = ModelTypeGeographic
                    1025, 0, 1, 1, // GTRasterTypeGeoKey = RasterPixelIsArea
                    2048, 0, 1, epsg as u16, // GeographicTypeGeoKey
                ]
            } else {
                // Projected CRS (e.g., UTM)
                vec![
                    1, 1, 0, 3, // Version 1.1.0, 3 keys
                    1024, 0, 1, 1, // GTModelTypeGeoKey = ModelTypeProjected
                    1025, 0, 1, 1, // GTRasterTypeGeoKey = RasterPixelIsArea
                    3072, 0, 1, epsg as u16, // ProjectedCSTypeGeoKey
                ]
            }
        } else {
            // CRS without EPSG code
            vec![
                1, 1, 0, 2, // Version 1.1.0, 2 keys
                1024, 0, 1, 1, // ModelTypeProjected
                1025, 0, 1, 1, // RasterPixelIsArea
            ]
        }
    } else {
        // No CRS
        vec![
            1, 1, 0, 2, // Version 1.1.0, 2 keys
            1024, 0, 1, 1, // ModelTypeProjected
            1025, 0, 1, 1, // RasterPixelIsArea
        ]
    };
    image
        .encoder()
        .write_tag(Tag::Unknown(34735), geokeys.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write geokey tag: {}", e)))?;

    // Write strips via callback
    let rps = config.rows_per_strip as usize;
    let num_strips = (config.rows + rps - 1) / rps;

    for strip_idx in 0..num_strips {
        let strip_rows = if strip_idx == num_strips - 1 {
            config.rows - strip_idx * rps
        } else {
            rps
        };

        let data = produce_strip(strip_idx, strip_rows)?;

        // Convert f64 -> f32 for writing
        let f32_data: Vec<f32> = data.iter().map(|&v| v as f32).collect();

        image
            .write_strip(&f32_data)
            .map_err(|e| Error::Other(format!("Cannot write strip {}: {}", strip_idx, e)))?;
    }

    image
        .finish()
        .map_err(|e| Error::Other(format!("Cannot finalize TIFF: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::strip_reader::StripReader;

    #[test]
    fn test_write_and_read_back() {
        let path = std::path::Path::new("/tmp/surtgis_strip_writer_test.tif");
        let rows = 100;
        let cols = 50;
        let config = StripWriterConfig {
            rows,
            cols,
            transform: GeoTransform::new(100.0, 200.0, 10.0, -10.0),
            crs: Some(CRS::from_epsg(32719)),
            compress: false,
            rows_per_strip: 32,
        };

        // Write with known values
        write_geotiff_streaming(path, &config, |_strip_idx, strip_rows| {
            let mut data = Array2::<f64>::zeros((strip_rows, cols));
            for r in 0..strip_rows {
                for c in 0..cols {
                    data[[r, c]] = (r * 1000 + c) as f64;
                }
            }
            Ok(data)
        })
        .unwrap();

        // Read back and verify
        let mut reader = StripReader::open(path).unwrap();
        assert_eq!(reader.rows(), rows);
        assert_eq!(reader.cols(), cols);

        // Check transform
        let gt = reader.transform();
        assert!((gt.origin_x - 100.0).abs() < 1e-6);
        assert!((gt.origin_y - 200.0).abs() < 1e-6);

        // Check CRS
        assert!(reader.crs().is_some());
        assert_eq!(reader.crs().unwrap().epsg(), Some(32719));

        // Clean up
        let _ = std::fs::remove_file(path);
    }
}
