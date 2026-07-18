//! Strip-based GeoTIFF writer for streaming I/O.
//!
//! Writes GeoTIFF files one strip at a time using a callback-based
//! approach, enabling bounded-memory output of large rasters — both
//! uncompressed and DEFLATE-compressed (see
//! [`write_compressed_streaming`] for why the compressed path needs its
//! own manual `DirectoryEncoder`-driven implementation instead of the
//! `tiff` crate's higher-level `ImageEncoder`).

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use ndarray::Array2;
use tiff::encoder::colortype::{ColorType, Gray32Float};
use tiff::encoder::compression::DeflateLevel;
use tiff::encoder::{TiffEncoder, TiffKindStandard};
use tiff::tags::{CompressionMethod, PhotometricInterpretation, Tag};

use super::native::{deflate_compress_bytes, flatten_native_bytes, write_geo_metadata_tags};
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
    /// Optional NoData value to embed as GDAL_NODATA tag (42113).
    pub nodata: Option<f64>,
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
///     nodata: Some(f64::NAN),
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
    // Write to temp file first, then atomic rename to prevent corrupt partial files
    let tmp_path = path.with_extension("tmp");
    let file = BufWriter::new(File::create(&tmp_path)?);

    let mut encoder =
        TiffEncoder::new(file).map_err(|e| Error::Other(format!("TIFF encoder error: {}", e)))?;

    if config.compress {
        write_compressed_streaming(&mut encoder, config, &mut produce_strip)?;
    } else {
        write_uncompressed_streaming(&mut encoder, config, &mut produce_strip)?;
    }

    // Atomic rename: only appears at final path if write completed successfully
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Genuinely bounded-memory uncompressed path: the encoder's compressor
/// defaults to `Uncompressed` and is never touched unless `ImageEncoder`'s
/// own `write_data()` runs, so calling `write_strip()` directly, one strip
/// at a time, is safe here and never holds more than one strip's worth of
/// pixels in memory (plus whatever `produce_strip` itself allocates).
fn write_uncompressed_streaming<W, F>(
    encoder: &mut TiffEncoder<W, TiffKindStandard>,
    config: &StripWriterConfig,
    produce_strip: &mut F,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    F: FnMut(usize, usize) -> Result<Array2<f64>>,
{
    let mut image = encoder
        .new_image::<Gray32Float>(config.cols as u32, config.rows as u32)
        .map_err(|e| Error::Other(format!("Cannot create TIFF image: {}", e)))?;

    image
        .rows_per_strip(config.rows_per_strip)
        .map_err(|e| Error::Other(format!("Cannot set rows_per_strip: {}", e)))?;

    write_geo_metadata_tags(
        image.encoder(),
        &config.transform,
        config.crs.as_ref(),
        config.nodata,
    )?;

    let rps = config.rows_per_strip as usize;
    let num_strips = (config.rows + rps - 1) / rps;
    for strip_idx in 0..num_strips {
        let strip_rows = if strip_idx == num_strips - 1 {
            config.rows - strip_idx * rps
        } else {
            rps
        };
        let data = produce_strip(strip_idx, strip_rows)?;
        let strip_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        image
            .write_strip(&strip_f32)
            .map_err(|e| Error::Other(format!("Cannot write strip {}: {}", strip_idx, e)))?;
    }
    image
        .finish()
        .map_err(|e| Error::Other(format!("Cannot finish TIFF image: {}", e)))?;
    Ok(())
}

/// Genuinely bounded-memory *compressed* streaming path.
///
/// `ImageEncoder::write_data()` is the only way to get `write_strip()` to
/// actually engage DEFLATE (it toggles the encoder's compressor around
/// its own strip loop), but it needs the whole image as one contiguous
/// `&[f32]` slice — that's the one real `tiff` 0.10.3 API limitation
/// here: there is no public hook to toggle the encoder's own compressor
/// before a manual `write_strip()` loop. This function sidesteps
/// `ImageEncoder` entirely instead of buffering: it drives the
/// lower-level `DirectoryEncoder` API by hand —
/// the same approach `write_geotiff_stack` (`native.rs`) and the COG
/// writer (`cog_writer.rs`) use — compressing each strip's bytes itself
/// with `flate2` (the identical wrapper the `tiff` crate uses internally,
/// so the stream shape readers expect is unchanged) and writing the
/// already-compressed bytes as opaque data. Since `DirectoryEncoder`'s
/// low-level `write_data()` just writes bytes at the current offset
/// without touching any compressor state, this never depends on the
/// private toggle at all — only one strip's raw + compressed bytes are
/// ever in memory at a time, on top of whatever `produce_strip` itself
/// allocates for that strip.
fn write_compressed_streaming<W, F>(
    encoder: &mut TiffEncoder<W, TiffKindStandard>,
    config: &StripWriterConfig,
    produce_strip: &mut F,
) -> Result<()>
where
    W: std::io::Write + std::io::Seek,
    F: FnMut(usize, usize) -> Result<Array2<f64>>,
{
    let mut dir = encoder
        .image_directory()
        .map_err(|e| Error::Other(format!("Cannot create TIFF directory: {}", e)))?;

    dir.write_tag(Tag::ImageWidth, config.cols as u32)
        .map_err(|e| Error::Other(format!("Cannot write ImageWidth: {}", e)))?;
    dir.write_tag(Tag::ImageLength, config.rows as u32)
        .map_err(|e| Error::Other(format!("Cannot write ImageLength: {}", e)))?;
    dir.write_tag(
        Tag::BitsPerSample,
        <Gray32Float as ColorType>::BITS_PER_SAMPLE[0],
    )
    .map_err(|e| Error::Other(format!("Cannot write BitsPerSample: {}", e)))?;
    dir.write_tag(
        Tag::SampleFormat,
        <Gray32Float as ColorType>::SAMPLE_FORMAT[0].to_u16(),
    )
    .map_err(|e| Error::Other(format!("Cannot write SampleFormat: {}", e)))?;
    dir.write_tag(
        Tag::PhotometricInterpretation,
        PhotometricInterpretation::BlackIsZero.to_u16(),
    )
    .map_err(|e| Error::Other(format!("Cannot write PhotometricInterpretation: {}", e)))?;
    dir.write_tag(Tag::SamplesPerPixel, 1u16)
        .map_err(|e| Error::Other(format!("Cannot write SamplesPerPixel: {}", e)))?;
    dir.write_tag(Tag::Compression, CompressionMethod::Deflate.to_u16())
        .map_err(|e| Error::Other(format!("Cannot write Compression: {}", e)))?;
    dir.write_tag(Tag::RowsPerStrip, config.rows_per_strip)
        .map_err(|e| Error::Other(format!("Cannot write RowsPerStrip: {}", e)))?;

    let rps = config.rows_per_strip as usize;
    let num_strips = (config.rows + rps - 1) / rps;

    // Each strip is compressed as its own independent Deflate/zlib stream
    // (matching what a multi-strip TIFF reader expects to decode
    // per-strip) — never one stream spanning the whole image.
    let mut strip_offsets: Vec<u32> = Vec::with_capacity(num_strips);
    let mut strip_byte_counts: Vec<u32> = Vec::with_capacity(num_strips);
    for strip_idx in 0..num_strips {
        let strip_rows = if strip_idx == num_strips - 1 {
            config.rows - strip_idx * rps
        } else {
            rps
        };
        let data = produce_strip(strip_idx, strip_rows)?;
        let strip_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let raw_bytes = flatten_native_bytes(&strip_f32);
        let compressed = deflate_compress_bytes(&raw_bytes, DeflateLevel::Balanced)?;

        let offset = dir
            .write_data(compressed.as_slice())
            .map_err(|e| Error::Other(format!("Cannot write strip {}: {}", strip_idx, e)))?;
        strip_offsets.push(
            u32::try_from(offset)
                .map_err(|_| Error::Other(format!("Strip {strip_idx} offset overflow")))?,
        );
        strip_byte_counts.push(
            u32::try_from(compressed.len())
                .map_err(|_| Error::Other(format!("Strip {strip_idx} byte count overflow")))?,
        );
    }

    dir.write_tag(Tag::StripOffsets, strip_offsets.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write StripOffsets: {}", e)))?;
    dir.write_tag(Tag::StripByteCounts, strip_byte_counts.as_slice())
        .map_err(|e| Error::Other(format!("Cannot write StripByteCounts: {}", e)))?;

    write_geo_metadata_tags(
        &mut dir,
        &config.transform,
        config.crs.as_ref(),
        config.nodata,
    )?;

    dir.finish()
        .map_err(|e| Error::Other(format!("Cannot finish TIFF directory: {}", e)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::strip_reader::StripReader;

    #[test]
    fn test_write_and_read_back() {
        // A hardcoded /tmp path fails on Windows (found by the OS matrix).
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("surtgis_strip_writer_test.tif");
        let path = path.as_path();
        let rows = 100;
        let cols = 50;
        let config = StripWriterConfig {
            rows,
            cols,
            transform: GeoTransform::new(100.0, 200.0, 10.0, -10.0),
            crs: Some(CRS::from_epsg(32719)),
            nodata: Some(-9999.0),
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
        let reader = StripReader::open(path).unwrap();
        assert_eq!(reader.rows(), rows);
        assert_eq!(reader.cols(), cols);

        // Check transform
        let gt = reader.transform();
        assert!((gt.origin_x - 100.0).abs() < 1e-6);
        assert!((gt.origin_y - 200.0).abs() < 1e-6);

        // Check CRS
        assert!(reader.crs().is_some());
        assert_eq!(reader.crs().unwrap().epsg(), Some(32719));
        // TempDir cleans up on drop.
    }

    /// Regression for the `compress: true` branch: it now streams
    /// strip-by-strip (hand-driven `DirectoryEncoder`, one compressed
    /// strip in memory at a time) and must remain byte-for-byte correct
    /// after being split out of the uncompressed streaming path.
    #[test]
    fn test_write_and_read_back_compressed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("surtgis_strip_writer_compressed.tif");
        let path = path.as_path();
        let rows = 80;
        let cols = 40;
        let config = StripWriterConfig {
            rows,
            cols,
            transform: GeoTransform::new(0.0, 500.0, 5.0, -5.0),
            crs: Some(CRS::from_epsg(4326)),
            nodata: Some(f64::NAN),
            compress: true,
            rows_per_strip: rows as u32,
        };

        write_geotiff_streaming(path, &config, |_strip_idx, strip_rows| {
            let mut data = Array2::<f64>::zeros((strip_rows, cols));
            for r in 0..strip_rows {
                for c in 0..cols {
                    data[[r, c]] = (r * 100 + c) as f64 * 0.5;
                }
            }
            Ok(data)
        })
        .unwrap();

        let mut reader = StripReader::open(path).unwrap();
        assert_eq!(reader.rows(), rows);
        assert_eq!(reader.cols(), cols);
        assert_eq!(reader.crs().unwrap().epsg(), Some(4326));

        let block = reader.read_rows(0, rows).unwrap();
        for r in 0..rows {
            for c in 0..cols {
                let expected = (r * 100 + c) as f64 * 0.5;
                assert_eq!(block[[r, c]], expected, "mismatch at ({}, {})", r, c);
            }
        }
    }

    /// Was filed as a "known bug" (multi-strip DEFLATE corrupts rows past
    /// the first strip) but turned out to be a bug in *this test*, not in
    /// `write_geotiff_streaming`, the native reader, or the `tiff` crate:
    /// the original callback ignored `strip_idx` and always synthesized
    /// values from a strip-local `r` starting at 0, so strip 1's "row 0"
    /// (global row 40) collided with strip 0's row 0 pattern. Cross-checked
    /// against `rasterio`/GDAL (an independent libtiff-based reader, not
    /// just this crate's own `StripReader`): it read back exactly what the
    /// buggy callback actually wrote — proof the file itself was
    /// well-formed and the mismatch was purely in the test's expected
    /// values. Fixed by offsetting `r` with `strip_idx * rows_per_strip`
    /// so the callback produces (and the assertion expects) the same
    /// global row numbering multi-strip or single-strip alike.
    #[test]
    fn multi_strip_deflate_roundtrips_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("surtgis_strip_writer_multistrip.tif");
        let path = path.as_path();
        let rows = 80;
        let cols = 40;
        let rows_per_strip = 40u32;
        let config = StripWriterConfig {
            rows,
            cols,
            transform: GeoTransform::new(0.0, 500.0, 5.0, -5.0),
            crs: Some(CRS::from_epsg(4326)),
            nodata: Some(f64::NAN),
            compress: true,
            rows_per_strip, // 2 strips over 80 rows
        };

        write_geotiff_streaming(path, &config, |strip_idx, strip_rows| {
            let mut data = Array2::<f64>::zeros((strip_rows, cols));
            let row_offset = strip_idx * rows_per_strip as usize;
            for r in 0..strip_rows {
                let global_r = row_offset + r;
                for c in 0..cols {
                    data[[r, c]] = (global_r * 100 + c) as f64 * 0.5;
                }
            }
            Ok(data)
        })
        .unwrap();

        let mut reader = StripReader::open(path).unwrap();
        let block = reader.read_rows(0, rows).unwrap();
        for r in 0..rows {
            for c in 0..cols {
                let expected = (r * 100 + c) as f64 * 0.5;
                assert_eq!(block[[r, c]], expected, "mismatch at ({}, {})", r, c);
            }
        }
    }

    /// 5 strips (not 2), with a non-uniform last strip (97 rows / 20 per
    /// strip = 4 full strips + 1 partial of 17) — the general case, not
    /// just the smallest multi-strip example. Cross-checked with
    /// `gdalinfo`/`rasterio` during development (external, independent of
    /// this crate's own reader); this test uses `StripReader` since that's
    /// what the rest of this file's tests use and gdal/rasterio aren't
    /// guaranteed available in every CI job.
    #[test]
    fn multi_strip_deflate_five_strips_uneven_last() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("surtgis_strip_writer_5strips.tif");
        let path = path.as_path();
        let rows = 97;
        let cols = 33;
        let rows_per_strip = 20u32;
        let config = StripWriterConfig {
            rows,
            cols,
            transform: GeoTransform::new(10.0, 300.0, 3.0, -3.0),
            crs: Some(CRS::from_epsg(32719)),
            nodata: Some(-9999.0),
            compress: true,
            rows_per_strip,
        };

        write_geotiff_streaming(path, &config, |strip_idx, strip_rows| {
            let mut data = Array2::<f64>::zeros((strip_rows, cols));
            let row_offset = strip_idx * rows_per_strip as usize;
            for r in 0..strip_rows {
                let global_r = row_offset + r;
                for c in 0..cols {
                    data[[r, c]] = (global_r * 1000 + c) as f64 * 0.25;
                }
            }
            Ok(data)
        })
        .unwrap();

        let mut reader = StripReader::open(path).unwrap();
        let block = reader.read_rows(0, rows).unwrap();
        for r in 0..rows {
            for c in 0..cols {
                let expected = (r * 1000 + c) as f64 * 0.25;
                assert_eq!(block[[r, c]], expected, "mismatch at ({}, {})", r, c);
            }
        }
    }
}
