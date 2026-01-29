//! Main Raster type

use crate::crs::CRS;
use crate::error::{Error, Result};
use crate::raster::{GeoTransform, RasterElement};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

/// A georeferenced 2D raster grid.
///
/// `Raster<T>` stores values of type `T` in a 2D grid with associated
/// geographic metadata (transform and CRS).
///
/// # Type Parameters
///
/// - `T`: The cell value type, must implement [`RasterElement`]
///
/// # Example
///
/// ```ignore
/// use surtgis_core::Raster;
///
/// // Create a 100x100 raster filled with zeros
/// let mut raster: Raster<f32> = Raster::new(100, 100);
///
/// // Set a value
/// raster.set(10, 20, 42.0)?;
///
/// // Get a value
/// let value = raster.get(10, 20)?;
/// ```
#[derive(Debug, Clone)]
pub struct Raster<T: RasterElement> {
    /// Raster data stored in row-major order (row, col)
    data: Array2<T>,
    /// Affine transformation
    transform: GeoTransform,
    /// Coordinate reference system
    crs: Option<CRS>,
    /// No-data value
    nodata: Option<T>,
}

impl<T: RasterElement> Raster<T> {
    /// Create a new raster filled with zeros
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: Array2::zeros((rows, cols)),
            transform: GeoTransform::default(),
            crs: None,
            nodata: None,
        }
    }

    /// Create a new raster filled with a specific value
    pub fn filled(rows: usize, cols: usize, value: T) -> Self {
        Self {
            data: Array2::from_elem((rows, cols), value),
            transform: GeoTransform::default(),
            crs: None,
            nodata: None,
        }
    }

    /// Create a raster from existing data
    pub fn from_vec(data: Vec<T>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(Error::InvalidDimensions {
                width: cols,
                height: rows,
            });
        }

        let array = Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| Error::Other(e.to_string()))?;

        Ok(Self {
            data: array,
            transform: GeoTransform::default(),
            crs: None,
            nodata: None,
        })
    }

    /// Create a raster from an ndarray
    pub fn from_array(data: Array2<T>) -> Self {
        Self {
            data,
            transform: GeoTransform::default(),
            crs: None,
            nodata: None,
        }
    }

    /// Create a raster with the same metadata but different data type
    pub fn with_same_meta<U: RasterElement>(&self, rows: usize, cols: usize) -> Raster<U> {
        Raster {
            data: Array2::zeros((rows, cols)),
            transform: self.transform,
            crs: self.crs.clone(),
            nodata: None,
        }
    }

    /// Create a raster with the same dimensions and metadata, filled with a value
    pub fn like(&self, fill_value: T) -> Self {
        Self {
            data: Array2::from_elem(self.data.dim(), fill_value),
            transform: self.transform,
            crs: self.crs.clone(),
            nodata: self.nodata,
        }
    }

    // Dimensions

    /// Number of rows
    pub fn rows(&self) -> usize {
        self.data.nrows()
    }

    /// Number of columns
    pub fn cols(&self) -> usize {
        self.data.ncols()
    }

    /// Dimensions as (rows, cols)
    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Total number of cells
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the raster is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    // Data access

    /// Get value at (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<T> {
        self.data
            .get((row, col))
            .copied()
            .ok_or(Error::IndexOutOfBounds {
                row,
                col,
                rows: self.rows(),
                cols: self.cols(),
            })
    }

    /// Get value at (row, col) without bounds checking
    ///
    /// # Safety
    /// Caller must ensure row < self.rows() and col < self.cols()
    pub unsafe fn get_unchecked(&self, row: usize, col: usize) -> T {
        unsafe { *self.data.uget((row, col)) }
    }

    /// Set value at (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: T) -> Result<()> {
        if row >= self.rows() || col >= self.cols() {
            return Err(Error::IndexOutOfBounds {
                row,
                col,
                rows: self.rows(),
                cols: self.cols(),
            });
        }
        self.data[(row, col)] = value;
        Ok(())
    }

    /// Set value at (row, col) without bounds checking
    ///
    /// # Safety
    /// Caller must ensure row < self.rows() and col < self.cols()
    pub unsafe fn set_unchecked(&mut self, row: usize, col: usize, value: T) {
        unsafe { *self.data.uget_mut((row, col)) = value; }
    }

    /// Get a view of the underlying data
    pub fn view(&self) -> ArrayView2<'_, T> {
        self.data.view()
    }

    /// Get a mutable view of the underlying data
    pub fn view_mut(&mut self) -> ArrayViewMut2<'_, T> {
        self.data.view_mut()
    }

    /// Get a reference to the underlying array
    pub fn data(&self) -> &Array2<T> {
        &self.data
    }

    /// Get a mutable reference to the underlying array
    pub fn data_mut(&mut self) -> &mut Array2<T> {
        &mut self.data
    }

    /// Consume the raster and return the underlying array
    pub fn into_array(self) -> Array2<T> {
        self.data
    }

    /// Get a row slice
    pub fn row(&self, row: usize) -> Result<ndarray::ArrayView1<'_, T>> {
        if row >= self.rows() {
            return Err(Error::IndexOutOfBounds {
                row,
                col: 0,
                rows: self.rows(),
                cols: self.cols(),
            });
        }
        Ok(self.data.row(row))
    }

    // Metadata

    /// Get the geotransform
    pub fn transform(&self) -> &GeoTransform {
        &self.transform
    }

    /// Set the geotransform
    pub fn set_transform(&mut self, transform: GeoTransform) {
        self.transform = transform;
    }

    /// Get the CRS
    pub fn crs(&self) -> Option<&CRS> {
        self.crs.as_ref()
    }

    /// Set the CRS
    pub fn set_crs(&mut self, crs: Option<CRS>) {
        self.crs = crs;
    }

    /// Get the no-data value
    pub fn nodata(&self) -> Option<T> {
        self.nodata
    }

    /// Set the no-data value
    pub fn set_nodata(&mut self, nodata: Option<T>) {
        self.nodata = nodata;
    }

    /// Cell size (assumes square cells)
    pub fn cell_size(&self) -> f64 {
        self.transform.cell_size()
    }

    /// Geographic bounds (min_x, min_y, max_x, max_y)
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        self.transform.bounds(self.cols(), self.rows())
    }

    // Coordinate conversion

    /// Convert pixel coordinates to geographic coordinates
    pub fn pixel_to_geo(&self, col: usize, row: usize) -> (f64, f64) {
        self.transform.pixel_to_geo(col, row)
    }

    /// Convert geographic coordinates to pixel coordinates
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        self.transform.geo_to_pixel(x, y)
    }

    // Value checks

    /// Check if a value is no-data
    pub fn is_nodata(&self, value: T) -> bool {
        value.is_nodata(self.nodata)
    }

    /// Check if cell at (row, col) contains no-data
    pub fn is_nodata_at(&self, row: usize, col: usize) -> Result<bool> {
        let value = self.get(row, col)?;
        Ok(self.is_nodata(value))
    }

    // Statistics

    /// Calculate basic statistics (min, max, mean, count of valid cells)
    pub fn statistics(&self) -> RasterStatistics<T>
    where
        T: PartialOrd,
    {
        let mut min = None;
        let mut max = None;
        let mut sum: f64 = 0.0;
        let mut count: usize = 0;

        for &value in self.data.iter() {
            if self.is_nodata(value) {
                continue;
            }

            if min.is_none() || value < min.unwrap() {
                min = Some(value);
            }
            if max.is_none() || value > max.unwrap() {
                max = Some(value);
            }

            if let Some(v) = value.to_f64() {
                sum += v;
                count += 1;
            }
        }

        let mean = if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        };

        RasterStatistics {
            min,
            max,
            mean,
            valid_count: count,
            nodata_count: self.len() - count,
        }
    }
}

/// Basic statistics for a raster
#[derive(Debug, Clone)]
pub struct RasterStatistics<T> {
    pub min: Option<T>,
    pub max: Option<T>,
    pub mean: Option<f64>,
    pub valid_count: usize,
    pub nodata_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raster_creation() {
        let raster: Raster<f32> = Raster::new(100, 200);
        assert_eq!(raster.rows(), 100);
        assert_eq!(raster.cols(), 200);
        assert_eq!(raster.shape(), (100, 200));
    }

    #[test]
    fn test_raster_access() {
        let mut raster: Raster<f32> = Raster::new(10, 10);
        raster.set(5, 5, 42.0).unwrap();
        assert_eq!(raster.get(5, 5).unwrap(), 42.0);
    }

    #[test]
    fn test_raster_statistics() {
        let mut raster: Raster<f32> = Raster::new(10, 10);
        for i in 0..10 {
            for j in 0..10 {
                raster.set(i, j, (i * 10 + j) as f32).unwrap();
            }
        }

        let stats = raster.statistics();
        assert_eq!(stats.min, Some(0.0));
        assert_eq!(stats.max, Some(99.0));
        assert_eq!(stats.valid_count, 100);
    }
}
