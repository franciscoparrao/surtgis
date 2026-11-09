//! Streaming output sink for the multiband composite.
//!
//! [`StreamingTiffSink`] retires the composite's persistent output buffers
//! (`n_bands × rows × cols × 8` bytes held in RAM until the end — the
//! dominant memory term for large outputs). Instead, each composited band
//! strip is streamed to a per-band raw scratch file on arrival (only one
//! strip is ever in memory), and the final GeoTIFFs are assembled at
//! [`finish`](StreamingTiffSink::finish) with the bounded-memory
//! [`write_geotiff_streaming`], which reads the scratch back one strip at a
//! time. Peak RAM is then independent of the output size — the composite can
//! produce rasters larger than RAM, trading the in-memory buffers for
//! transient scratch on disk.
//!
//! The engine pushes strips top-to-bottom within each band (outer-strip,
//! inner-band loop), so appending each band's strip to its scratch file
//! yields exactly the row-major image.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ndarray::Array2;

use surtgis_cloud::composite::{OutputGrid, StripSink};
use surtgis_core::CRS;
use surtgis_core::io::{StripWriterConfig, write_geotiff_streaming};
use surtgis_core::raster::GeoTransform;

/// Rows per strip in the assembled GeoTIFF (output TIFF strip structure;
/// independent of the composite's own strip height).
const OUTPUT_ROWS_PER_STRIP: u32 = 256;

/// A [`StripSink`] that streams each band strip to disk instead of holding
/// the whole output in RAM.
pub struct StreamingTiffSink {
    band_paths: Vec<PathBuf>,
    scratch_paths: Vec<PathBuf>,
    scratch: Vec<BufWriter<File>>,
    rows: usize,
    cols: usize,
    transform: GeoTransform,
    crs: Option<CRS>,
    compress: bool,
    /// Rows appended to each band's scratch so far; guards the top-to-bottom
    /// append assumption against an out-of-order strip.
    rows_written: Vec<usize>,
}

impl StreamingTiffSink {
    /// Create a sink writing one GeoTIFF per band at `band_paths` (in band
    /// order). Scratch files live alongside each output.
    pub fn new(band_paths: Vec<PathBuf>, compress: bool) -> Self {
        Self {
            band_paths,
            scratch_paths: Vec::new(),
            scratch: Vec::new(),
            rows: 0,
            cols: 0,
            transform: GeoTransform::new(0.0, 0.0, 1.0, -1.0),
            crs: None,
            compress,
            rows_written: Vec::new(),
        }
    }

    /// Assemble the final GeoTIFFs from the scratch files (bounded memory),
    /// then remove the scratch. Returns each output's size in bytes.
    pub fn finish(mut self) -> Result<Vec<u64>> {
        // Flush all scratch writers before reading them back.
        for w in &mut self.scratch {
            w.flush().context("flushing composite scratch")?;
        }
        let mut sizes = Vec::with_capacity(self.band_paths.len());
        let config_base = StripWriterConfig {
            rows: self.rows,
            cols: self.cols,
            transform: self.transform,
            crs: self.crs.clone(),
            nodata: Some(f64::NAN),
            compress: self.compress,
            rows_per_strip: OUTPUT_ROWS_PER_STRIP.min(self.rows.max(1) as u32).max(1),
        };
        for (bi, band_path) in self.band_paths.iter().enumerate() {
            let scratch = &self.scratch_paths[bi];
            let file = File::open(scratch)
                .with_context(|| format!("reopening scratch {}", scratch.display()))?;
            let mut reader = BufReader::new(file);
            let cols = self.cols;
            write_geotiff_streaming(band_path, &config_base, |_strip_idx, strip_rows| {
                read_strip(&mut reader, strip_rows, cols)
            })
            .map_err(|e| anyhow::anyhow!("streaming write {}: {e}", band_path.display()))?;
            let size = std::fs::metadata(band_path).map(|m| m.len()).unwrap_or(0);
            sizes.push(size);
        }
        // Best-effort scratch cleanup.
        for scratch in &self.scratch_paths {
            let _ = std::fs::remove_file(scratch);
        }
        Ok(sizes)
    }
}

impl StripSink for StreamingTiffSink {
    fn begin(&mut self, grid: &OutputGrid) -> surtgis_cloud::Result<()> {
        self.rows = grid.rows;
        self.cols = grid.cols;
        self.transform = grid.transform;
        self.crs = grid.crs.clone();
        self.scratch_paths = self.band_paths.iter().map(|p| scratch_path(p)).collect();
        self.rows_written = vec![0; self.band_paths.len()];
        self.scratch = Vec::with_capacity(self.scratch_paths.len());
        for sp in &self.scratch_paths {
            let f = File::create(sp).map_err(|e| {
                surtgis_cloud::CloudError::Composite(format!(
                    "creating composite scratch {}: {e}",
                    sp.display()
                ))
            })?;
            self.scratch.push(BufWriter::new(f));
        }
        Ok(())
    }

    fn holds_output_in_ram(&self) -> bool {
        false
    }

    fn accept(
        &mut self,
        band_idx: usize,
        row_start: usize,
        strip: Array2<f64>,
    ) -> surtgis_cloud::Result<()> {
        // Strips arrive top-to-bottom per band, so appending row-major f32
        // reproduces the full image. An out-of-order strip would silently
        // corrupt the file, so assert the contiguous top-to-bottom order the
        // scratch append relies on.
        if row_start != self.rows_written[band_idx] {
            return Err(surtgis_cloud::CloudError::Composite(format!(
                "composite strip for band {band_idx} arrived at row {row_start}, expected {} \
                 (strips must be appended top-to-bottom)",
                self.rows_written[band_idx]
            )));
        }
        self.rows_written[band_idx] += strip.nrows();
        // Cast to f32 (the output dtype).
        let w = &mut self.scratch[band_idx];
        let mut bytes: Vec<u8> = Vec::with_capacity(strip.len() * 4);
        for &v in strip.iter() {
            bytes.extend_from_slice(&(v as f32).to_ne_bytes());
        }
        w.write_all(&bytes).map_err(|e| {
            surtgis_cloud::CloudError::Composite(format!("writing composite scratch: {e}"))
        })?;
        Ok(())
    }
}

/// Scratch file path next to `band_path` (`<name>.scratch`).
fn scratch_path(band_path: &Path) -> PathBuf {
    band_path.with_extension("scratch")
}

/// Read `rows × cols` native-endian f32 values from `reader` into an
/// `Array2<f64>` (the pull callback for [`write_geotiff_streaming`]).
fn read_strip<R: Read>(
    reader: &mut R,
    rows: usize,
    cols: usize,
) -> surtgis_core::Result<Array2<f64>> {
    let n = rows * cols;
    let mut buf = vec![0u8; n * 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| surtgis_core::Error::Other(format!("reading composite scratch: {e}")))?;
    let data: Vec<f64> = buf
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect();
    Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| surtgis_core::Error::Other(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::io::read_geotiff;

    /// Push two bands' strips through the streaming sink and read the
    /// assembled GeoTIFFs back: every pixel must round-trip, and no scratch
    /// files may survive `finish`.
    #[test]
    fn streams_strips_and_assembles_geotiffs() {
        let dir = tempfile::tempdir().unwrap();
        let paths = vec![dir.path().join("b0.tif"), dir.path().join("b1.tif")];
        let (rows, cols) = (10usize, 4usize);

        let mut sink = StreamingTiffSink::new(paths.clone(), false);
        let grid = OutputGrid {
            cols,
            rows,
            transform: GeoTransform::new(100.0, 500.0, 10.0, -10.0),
            crs: Some(CRS::from_epsg(32719)),
            bbox: surtgis_cloud::BBox::new(100.0, 400.0, 140.0, 500.0),
        };
        sink.begin(&grid).unwrap();

        // Two strips of 5 rows each, per band, top-to-bottom (engine order:
        // strip 0 all bands, then strip 1 all bands).
        let value = |band: usize, r: usize, c: usize| (band * 1000 + r * 10 + c) as f64;
        for strip_idx in 0..2 {
            let row_start = strip_idx * 5;
            for band in 0..2 {
                let mut strip = Array2::<f64>::zeros((5, cols));
                for r in 0..5 {
                    for c in 0..cols {
                        strip[[r, c]] = value(band, row_start + r, c);
                    }
                }
                sink.accept(band, row_start, strip).unwrap();
            }
        }
        let sizes = sink.finish().unwrap();
        assert_eq!(sizes.len(), 2);
        assert!(sizes.iter().all(|&s| s > 0));

        for (band, path) in paths.iter().enumerate() {
            let raster: surtgis_core::Raster<f64> = read_geotiff(path, None).unwrap();
            assert_eq!(raster.shape(), (rows, cols));
            assert_eq!(raster.crs().and_then(|c| c.epsg()), Some(32719));
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(
                        raster.get(r, c).unwrap(),
                        value(band, r, c),
                        "band {band} pixel ({r},{c})"
                    );
                }
            }
        }
        // Scratch cleaned up.
        for path in &paths {
            assert!(!scratch_path(path).exists(), "scratch left behind");
        }
    }
}
