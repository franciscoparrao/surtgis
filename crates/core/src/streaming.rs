//! Streaming processor for window-based raster algorithms.
//!
//! Processes rasters strip-by-strip with bounded memory, enabling
//! arbitrarily large DEMs for algorithms that only need a local window.

use std::path::Path;

use ndarray::Array2;

use crate::error::Result;
use crate::io::strip_reader::StripReader;
use crate::io::strip_writer::{StripWriterConfig, write_geotiff_streaming};

/// Per-chunk geographic context for latitude-dependent cell-size
/// corrections, passed to [`WindowAlgorithm::process_chunk_geo`].
///
/// Geographic (lon/lat) rasters need a per-row metric cell size because a
/// degree of longitude shrinks with latitude (the batch path applies this
/// via `CellSizes`/`cell_dimensions` in the `algorithms` crate). `core` sits
/// below `algorithms` in the dependency graph and does not know that
/// ellipsoidal formula, so [`StripProcessor`] only supplies the raw
/// ingredients an implementation needs to derive each row's latitude:
///
/// - `row_offset`: absolute row index (within the full raster) of local
///   output row 0 in the current chunk. The absolute row for local row
///   `r` is `row_offset + r`.
/// - `origin_y`, `pixel_height`: from the raster's [`crate::GeoTransform`],
///   so the latitude of absolute row `ar` is
///   `origin_y + (ar as f64 + 0.5) * pixel_height`.
#[derive(Debug, Clone, Copy)]
pub struct GeoRowContext {
    /// Absolute row index (in the full raster) of local output row 0.
    pub row_offset: usize,
    /// Y coordinate of the raster's top-left corner (degrees).
    pub origin_y: f64,
    /// Pixel height in degrees (usually negative).
    pub pixel_height: f64,
}

/// Trait for algorithms that operate on a moving window.
///
/// Implementations define the kernel radius and how to process a chunk
/// of rows. The [`StripProcessor`] handles I/O and buffer management.
pub trait WindowAlgorithm: Send + Sync {
    /// Kernel radius (1 for 3x3, 10 for 21x21).
    fn kernel_radius(&self) -> usize;

    /// Process a chunk of input rows, producing output rows.
    ///
    /// `input` has `(chunk_rows + top_pad + bottom_pad)` rows x cols columns,
    /// where the padding is the halo clamped to image bounds.
    /// `output` has `chunk_rows` rows x cols columns.
    /// The algorithm should write results for the center rows only.
    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        cell_size_y: f64,
    );

    /// Like [`process_chunk`](Self::process_chunk), but also receives
    /// per-chunk geographic context so implementations can apply a
    /// latitude-dependent cell-size correction on geographic (lon/lat)
    /// rasters — mirroring the batch path's `CellSizes::for_dem` behavior.
    ///
    /// `geo_ctx` is `Some` only when the raster's CRS is geographic
    /// ([`crate::crs::CRS::is_geographic`]); it is `None` for projected
    /// rasters and for rasters with no CRS at all.
    ///
    /// The default implementation ignores `geo_ctx` and forwards to
    /// [`process_chunk`](Self::process_chunk) unchanged, so existing
    /// implementations that don't override this method keep their
    /// current (uncorrected) behavior on geographic rasters. Algorithms
    /// that need correct results on geographic CRSs (slope, aspect,
    /// hillshade, multidirectional hillshade) override it.
    fn process_chunk_geo(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        cell_size_y: f64,
        _geo_ctx: Option<GeoRowContext>,
    ) {
        self.process_chunk(input, output, nodata, cell_size_x, cell_size_y);
    }
}

/// Streaming processor that reads and writes GeoTIFF strip-by-strip.
///
/// Memory usage is bounded to approximately
/// `(chunk_rows + 2 * radius) * cols * 8` bytes for input plus
/// `chunk_rows * cols * 8` bytes for output.
pub struct StripProcessor {
    /// Number of output rows per chunk (default: 256).
    pub chunk_rows: usize,
}

impl StripProcessor {
    /// Create a new `StripProcessor` with the given chunk size.
    pub fn new(chunk_rows: usize) -> Self {
        Self { chunk_rows }
    }

    /// Process an entire raster file using streaming I/O.
    ///
    /// Returns `(rows, cols)` of the processed raster.
    pub fn process<A: WindowAlgorithm>(
        &self,
        input_path: &Path,
        output_path: &Path,
        algorithm: &A,
        compress: bool,
    ) -> Result<(usize, usize)> {
        let reader = StripReader::open(input_path)?;
        let rows = reader.rows();
        let cols = reader.cols();
        let radius = algorithm.kernel_radius();

        let nodata = reader.nodata();

        let config = StripWriterConfig {
            rows,
            cols,
            transform: reader.transform().clone(),
            crs: reader.crs().cloned(),
            nodata: nodata.or(Some(f64::NAN)),
            compress,
            rows_per_strip: self.chunk_rows as u32,
        };
        let cell_size_x = reader.transform().pixel_width.abs();
        let cell_size_y = reader.transform().pixel_height.abs();

        // REG-1: geographic (lon/lat) CRS needs a per-row latitude
        // correction, same as the batch path (`CellSizes::for_dem`).
        // Captured before `reader` moves into the RefCell below.
        let is_geographic = reader.crs().is_some_and(|c| c.is_geographic());
        let origin_y = reader.transform().origin_y;
        let pixel_height = reader.transform().pixel_height;

        // Use RefCell to share the reader with the sequential write callback.
        let reader_cell = std::cell::RefCell::new(reader);

        let mut current_out_row = 0usize;

        write_geotiff_streaming(output_path, &config, |_strip_idx, out_strip_rows| {
            let mut reader = reader_cell.borrow_mut();

            // Determine output row range
            let out_start = current_out_row;
            let out_end = (current_out_row + out_strip_rows).min(rows);
            let actual_out_rows = out_end - out_start;

            // Input range with halo (clamped to image bounds)
            let in_start = out_start.saturating_sub(radius);
            let in_end = (out_end + radius).min(rows);

            // Read the available input rows
            let raw_input = reader.read_rows(in_start, in_end - in_start)?;

            // Build a padded input buffer that always has exactly
            // (actual_out_rows + 2*radius) rows with the output centered.
            // Rows outside the image are filled with NaN.
            let padded_rows = actual_out_rows + 2 * radius;
            let mut input = Array2::<f64>::from_elem((padded_rows, cols), f64::NAN);

            // How many halo rows are missing at the top?
            let top_pad = radius.saturating_sub(out_start);
            // Copy raw data into the padded buffer at the right offset
            let copy_rows = raw_input.nrows().min(padded_rows - top_pad);
            input
                .slice_mut(ndarray::s![top_pad..top_pad + copy_rows, ..])
                .assign(&raw_input.slice(ndarray::s![..copy_rows, ..]));

            // Prepare output buffer
            let mut output = Array2::<f64>::from_elem((actual_out_rows, cols), f64::NAN);

            // Process: input[radius..radius+actual_out_rows] are the center rows
            let geo_ctx = is_geographic.then_some(GeoRowContext {
                row_offset: out_start,
                origin_y,
                pixel_height,
            });
            algorithm.process_chunk_geo(
                &input,
                &mut output,
                nodata,
                cell_size_x,
                cell_size_y,
                geo_ctx,
            );

            current_out_row = out_end;
            Ok(output)
        })?;

        Ok((rows, cols))
    }
}
