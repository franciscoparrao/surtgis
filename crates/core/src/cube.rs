//! Aligned (time, band) raster stacks for data-cube workflows.
//!
//! [`Cube`] is the **container** half of the data-cube story: it
//! holds `n_times × n_bands` single-band [`Raster`]s verified to
//! share one grid (shape, transform, CRS), and exposes aligned
//! iteration — per-pixel time series and row-chunk streaming.
//! Temporal *analysis* (regressions, break detection, compositing
//! algebra) deliberately lives in consumer engines (datacube-rs,
//! unmix-rs), not here.
//!
//! Timestamps are Unix epoch seconds (`i64`) — cheap, sortable, and
//! convertible from any datetime library without pulling one into
//! core.
//!
//! ```
//! use surtgis_core::cube::Cube;
//! use surtgis_core::{GeoTransform, Raster};
//!
//! let mut slices = Vec::new();
//! for t in 0..3 {
//!     let mut r: `Raster<f64>` = Raster::new(4, 4);
//!     r.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
//!     for row in 0..4 {
//!         for col in 0..4 {
//!             r.set(row, col, (t * 100 + row * 4 + col) as f64).unwrap();
//!         }
//!     }
//!     slices.push(r);
//! }
//! let cube = Cube::from_slices(vec![0, 86_400, 172_800], vec!["ndvi".into()], slices)?;
//! let series: Vec<f64> = cube.pixel_series(1, 2, 0)?.collect();
//! assert_eq!(series, vec![6.0, 106.0, 206.0]);
//! # Ok::<(), surtgis_core::Error>(())
//! ```

use ndarray::ArrayView2;

use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster, RasterElement};

/// An aligned stack of rasters over time and band axes.
///
/// Slices are stored in `(time, band)` row-major order:
/// `slice index = t * n_bands + b`. Every slice shares the same
/// shape, [`GeoTransform`] and CRS — enforced at construction, so
/// downstream code can index without re-checking.
#[derive(Debug, Clone)]
pub struct Cube<T: RasterElement> {
    times: Vec<i64>,
    bands: Vec<String>,
    slices: Vec<Raster<T>>,
}

/// One row-chunk of a [`Cube`]: aligned 2-D views (same rows) into
/// every `(time, band)` slice, in slice order.
pub struct CubeChunk<'a, T: RasterElement> {
    /// First grid row covered by this chunk.
    pub row0: usize,
    /// Number of rows in this chunk.
    pub rows: usize,
    /// One `rows × cols` view per slice, ordered `t * n_bands + b`.
    pub views: Vec<ArrayView2<'a, T>>,
}

impl<T: RasterElement> Cube<T> {
    /// Build a cube from `n_times × n_bands` rasters in
    /// `(time, band)` order.
    ///
    /// Validates that `slices.len() == times.len() * bands.len()`,
    /// that times are strictly increasing, that band names are
    /// unique, and that every raster shares the first one's shape,
    /// transform and CRS.
    pub fn from_slices(
        times: Vec<i64>,
        bands: Vec<String>,
        slices: Vec<Raster<T>>,
    ) -> Result<Self> {
        if times.is_empty() || bands.is_empty() {
            return Err(Error::Other(
                "cube: times and bands must be non-empty".into(),
            ));
        }
        if slices.len() != times.len() * bands.len() {
            return Err(Error::Other(format!(
                "cube: expected {} slices ({} times × {} bands), got {}",
                times.len() * bands.len(),
                times.len(),
                bands.len(),
                slices.len()
            )));
        }
        for pair in times.windows(2) {
            if pair[1] <= pair[0] {
                return Err(Error::Other(format!(
                    "cube: times must be strictly increasing ({} then {})",
                    pair[0], pair[1]
                )));
            }
        }
        for (i, name) in bands.iter().enumerate() {
            if bands[..i].contains(name) {
                return Err(Error::Other(format!("cube: duplicate band '{}'", name)));
            }
        }

        let shape = slices[0].shape();
        let transform = *slices[0].transform();
        let crs = slices[0].crs().cloned();
        for (i, slice) in slices.iter().enumerate().skip(1) {
            if slice.shape() != shape {
                return Err(Error::Other(format!(
                    "cube: slice {} shape {:?} != slice 0 shape {:?}",
                    i,
                    slice.shape(),
                    shape
                )));
            }
            if *slice.transform() != transform {
                return Err(Error::Other(format!(
                    "cube: slice {} transform differs from slice 0",
                    i
                )));
            }
            if slice.crs() != crs.as_ref() {
                return Err(Error::Other(format!(
                    "cube: slice {} CRS differs from slice 0",
                    i
                )));
            }
        }

        Ok(Self {
            times,
            bands,
            slices,
        })
    }

    /// Number of timestamps.
    pub fn n_times(&self) -> usize {
        self.times.len()
    }

    /// Number of bands.
    pub fn n_bands(&self) -> usize {
        self.bands.len()
    }

    /// Grid shape `(rows, cols)` shared by every slice.
    pub fn shape(&self) -> (usize, usize) {
        self.slices[0].shape()
    }

    /// Shared geotransform.
    pub fn transform(&self) -> &GeoTransform {
        self.slices[0].transform()
    }

    /// Timestamps (Unix epoch seconds), strictly increasing.
    pub fn times(&self) -> &[i64] {
        &self.times
    }

    /// Band names, in band-axis order.
    pub fn bands(&self) -> &[String] {
        &self.bands
    }

    /// Index of a band by name.
    pub fn band_index(&self, name: &str) -> Option<usize> {
        self.bands.iter().position(|b| b == name)
    }

    /// The raster at `(time t, band b)`.
    pub fn slice(&self, t: usize, b: usize) -> Result<&Raster<T>> {
        if t >= self.n_times() || b >= self.n_bands() {
            return Err(Error::Other(format!(
                "cube: slice ({}, {}) out of range ({} times, {} bands)",
                t,
                b,
                self.n_times(),
                self.n_bands()
            )));
        }
        Ok(&self.slices[t * self.n_bands() + b])
    }

    /// Consume the cube, returning the slices in `(time, band)` order.
    pub fn into_slices(self) -> Vec<Raster<T>> {
        self.slices
    }

    /// Time series of one band at one pixel, in time order.
    ///
    /// Yields one value per timestamp; nodata cells come through as
    /// stored (NaN for float rasters).
    pub fn pixel_series(
        &self,
        row: usize,
        col: usize,
        band: usize,
    ) -> Result<impl Iterator<Item = T> + '_> {
        let (rows, cols) = self.shape();
        if row >= rows || col >= cols {
            return Err(Error::IndexOutOfBounds {
                row,
                col,
                rows,
                cols,
            });
        }
        if band >= self.n_bands() {
            return Err(Error::Other(format!(
                "cube: band {} out of range ({} bands)",
                band,
                self.n_bands()
            )));
        }
        let n_bands = self.n_bands();
        Ok((0..self.n_times())
            .map(move |t| unsafe { self.slices[t * n_bands + band].get_unchecked(row, col) }))
    }

    /// All bands of one timestamp at one pixel, in band order.
    pub fn band_values(&self, row: usize, col: usize, t: usize) -> Result<Vec<T>> {
        let (rows, cols) = self.shape();
        if row >= rows || col >= cols {
            return Err(Error::IndexOutOfBounds {
                row,
                col,
                rows,
                cols,
            });
        }
        if t >= self.n_times() {
            return Err(Error::Other(format!(
                "cube: time {} out of range ({} times)",
                t,
                self.n_times()
            )));
        }
        let n_bands = self.n_bands();
        Ok((0..n_bands)
            .map(|b| unsafe { self.slices[t * n_bands + b].get_unchecked(row, col) })
            .collect())
    }

    /// Iterate the cube in row chunks of at most `chunk_rows` rows.
    ///
    /// Each [`CubeChunk`] holds one aligned view per slice covering
    /// the same grid rows — the bounded-memory access pattern for
    /// per-pixel temporal algebra over cubes larger than cache.
    /// `chunk_rows` is clamped to at least 1.
    pub fn chunks(&self, chunk_rows: usize) -> impl Iterator<Item = CubeChunk<'_, T>> {
        let (rows, _) = self.shape();
        let chunk_rows = chunk_rows.max(1);
        let n_chunks = rows.div_ceil(chunk_rows);
        (0..n_chunks).map(move |i| {
            let row0 = i * chunk_rows;
            let n = chunk_rows.min(rows - row0);
            CubeChunk {
                row0,
                rows: n,
                views: self
                    .slices
                    .iter()
                    .map(|s| s.data().slice(ndarray::s![row0..row0 + n, ..]))
                    .collect(),
            }
        })
    }
}

impl<'a, T: RasterElement> CubeChunk<'a, T> {
    /// View for `(time t, band b)`, given the cube's band count.
    pub fn view(&self, t: usize, b: usize, n_bands: usize) -> &ArrayView2<'a, T> {
        &self.views[t * n_bands + b]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GeoTransform;

    fn slice(rows: usize, cols: usize, fill_base: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, fill_base + (row * cols + col) as f64)
                    .unwrap();
            }
        }
        r
    }

    /// DoD test: 3 dated rasters, aligned iteration.
    #[test]
    fn three_dated_rasters_pixel_series() {
        let times = vec![1_700_000_000, 1_702_592_000, 1_705_184_000];
        let slices = vec![slice(5, 5, 0.0), slice(5, 5, 100.0), slice(5, 5, 200.0)];
        let cube = Cube::from_slices(times.clone(), vec!["b1".into()], slices).unwrap();

        assert_eq!(cube.n_times(), 3);
        assert_eq!(cube.n_bands(), 1);
        assert_eq!(cube.times(), &times[..]);

        let series: Vec<f64> = cube.pixel_series(2, 3, 0).unwrap().collect();
        assert_eq!(series, vec![13.0, 113.0, 213.0]);
    }

    #[test]
    fn multiband_layout_and_band_values() {
        // 2 times × 2 bands, distinguishable fill values
        let slices = vec![
            slice(4, 4, 0.0),    // t0 b0
            slice(4, 4, 1000.0), // t0 b1
            slice(4, 4, 100.0),  // t1 b0
            slice(4, 4, 1100.0), // t1 b1
        ];
        let cube =
            Cube::from_slices(vec![0, 86_400], vec!["red".into(), "nir".into()], slices).unwrap();

        assert_eq!(cube.band_index("nir"), Some(1));
        assert_eq!(cube.slice(1, 0).unwrap().get(0, 0).unwrap(), 100.0);
        assert_eq!(cube.band_values(0, 0, 1).unwrap(), vec![100.0, 1100.0]);

        let nir_series: Vec<f64> = cube.pixel_series(0, 1, 1).unwrap().collect();
        assert_eq!(nir_series, vec![1001.0, 1101.0]);
    }

    #[test]
    fn chunks_cover_all_rows_aligned() {
        let slices = vec![slice(10, 6, 0.0), slice(10, 6, 100.0), slice(10, 6, 200.0)];
        let cube = Cube::from_slices(vec![0, 1, 2], vec!["b".into()], slices).unwrap();

        let mut covered = 0usize;
        for chunk in cube.chunks(4) {
            assert_eq!(chunk.views.len(), 3);
            for v in &chunk.views {
                assert_eq!(v.nrows(), chunk.rows);
                assert_eq!(v.ncols(), 6);
            }
            // Alignment: same cell across slices differs by the fill base
            let a = chunk.views[0][[0, 2]];
            let b = chunk.views[1][[0, 2]];
            let c = chunk.views[2][[0, 2]];
            assert_eq!(b - a, 100.0);
            assert_eq!(c - b, 100.0);
            covered += chunk.rows;
        }
        assert_eq!(covered, 10);
    }

    #[test]
    fn rejects_misaligned_grids() {
        // Different shape
        let r = Cube::from_slices(
            vec![0, 1],
            vec!["b".into()],
            vec![slice(5, 5, 0.0), slice(4, 5, 0.0)],
        );
        assert!(r.is_err());

        // Different transform
        let mut shifted = slice(5, 5, 0.0);
        shifted.set_transform(GeoTransform::new(10.0, 5.0, 1.0, -1.0));
        let r = Cube::from_slices(
            vec![0, 1],
            vec!["b".into()],
            vec![slice(5, 5, 0.0), shifted],
        );
        assert!(r.is_err());
    }

    #[test]
    fn rejects_bad_metadata() {
        // Wrong slice count
        assert!(Cube::from_slices(vec![0, 1], vec!["b".into()], vec![slice(3, 3, 0.0)]).is_err());
        // Non-increasing times
        assert!(
            Cube::from_slices(
                vec![5, 5],
                vec!["b".into()],
                vec![slice(3, 3, 0.0), slice(3, 3, 1.0)]
            )
            .is_err()
        );
        // Duplicate band names
        assert!(
            Cube::from_slices(
                vec![0],
                vec!["b".into(), "b".into()],
                vec![slice(3, 3, 0.0), slice(3, 3, 1.0)]
            )
            .is_err()
        );
    }

    #[test]
    fn nan_passes_through_series() {
        let mut s1 = slice(3, 3, 0.0);
        s1.set(1, 1, f64::NAN).unwrap();
        let cube =
            Cube::from_slices(vec![0, 1], vec!["b".into()], vec![s1, slice(3, 3, 100.0)]).unwrap();
        let series: Vec<f64> = cube.pixel_series(1, 1, 0).unwrap().collect();
        assert!(series[0].is_nan());
        assert_eq!(series[1], 104.0);
    }
}
