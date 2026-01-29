//! Affine geotransformation for rasters

use serde::{Deserialize, Serialize};

/// Affine transformation coefficients for georeferencing rasters.
///
/// Converts between pixel coordinates (col, row) and geographic coordinates (x, y):
/// ```text
/// x = origin_x + col * pixel_width + row * row_rotation
/// y = origin_y + col * col_rotation + row * pixel_height
/// ```
///
/// For north-up images, `row_rotation` and `col_rotation` are typically 0,
/// and `pixel_height` is negative.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoTransform {
    /// X coordinate of the upper-left corner
    pub origin_x: f64,
    /// Y coordinate of the upper-left corner
    pub origin_y: f64,
    /// Pixel width (cell size in X direction)
    pub pixel_width: f64,
    /// Pixel height (cell size in Y direction, usually negative)
    pub pixel_height: f64,
    /// Rotation about X axis (usually 0)
    pub row_rotation: f64,
    /// Rotation about Y axis (usually 0)
    pub col_rotation: f64,
}

impl GeoTransform {
    /// Create a new GeoTransform with no rotation (north-up image)
    pub fn new(origin_x: f64, origin_y: f64, pixel_width: f64, pixel_height: f64) -> Self {
        Self {
            origin_x,
            origin_y,
            pixel_width,
            pixel_height,
            row_rotation: 0.0,
            col_rotation: 0.0,
        }
    }

    /// Create from GDAL-style array [origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height]
    pub fn from_gdal(coeffs: [f64; 6]) -> Self {
        Self {
            origin_x: coeffs[0],
            pixel_width: coeffs[1],
            row_rotation: coeffs[2],
            origin_y: coeffs[3],
            col_rotation: coeffs[4],
            pixel_height: coeffs[5],
        }
    }

    /// Convert to GDAL-style array
    pub fn to_gdal(&self) -> [f64; 6] {
        [
            self.origin_x,
            self.pixel_width,
            self.row_rotation,
            self.origin_y,
            self.col_rotation,
            self.pixel_height,
        ]
    }

    /// Convert pixel coordinates to geographic coordinates
    ///
    /// Returns the coordinates of the pixel center
    pub fn pixel_to_geo(&self, col: usize, row: usize) -> (f64, f64) {
        let col_f = col as f64 + 0.5;
        let row_f = row as f64 + 0.5;

        let x = self.origin_x + col_f * self.pixel_width + row_f * self.row_rotation;
        let y = self.origin_y + col_f * self.col_rotation + row_f * self.pixel_height;

        (x, y)
    }

    /// Convert pixel coordinates to geographic coordinates (top-left corner)
    pub fn pixel_to_geo_corner(&self, col: usize, row: usize) -> (f64, f64) {
        let col_f = col as f64;
        let row_f = row as f64;

        let x = self.origin_x + col_f * self.pixel_width + row_f * self.row_rotation;
        let y = self.origin_y + col_f * self.col_rotation + row_f * self.pixel_height;

        (x, y)
    }

    /// Convert geographic coordinates to pixel coordinates
    ///
    /// Returns fractional pixel coordinates; use `.floor()` to get integer indices
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        // Solve the inverse transformation
        let det = self.pixel_width * self.pixel_height - self.row_rotation * self.col_rotation;

        if det.abs() < 1e-10 {
            // Degenerate transformation
            return (f64::NAN, f64::NAN);
        }

        let dx = x - self.origin_x;
        let dy = y - self.origin_y;

        let col = (self.pixel_height * dx - self.row_rotation * dy) / det;
        let row = (-self.col_rotation * dx + self.pixel_width * dy) / det;

        (col, row)
    }

    /// Get the cell size (assumes square pixels and no rotation)
    pub fn cell_size(&self) -> f64 {
        self.pixel_width.abs()
    }

    /// Check if this is a north-up image (no rotation)
    pub fn is_north_up(&self) -> bool {
        self.row_rotation.abs() < 1e-10
            && self.col_rotation.abs() < 1e-10
            && self.pixel_height < 0.0
    }

    /// Calculate the bounding box for a raster of given dimensions
    pub fn bounds(&self, width: usize, height: usize) -> (f64, f64, f64, f64) {
        let (x0, y0) = self.pixel_to_geo_corner(0, 0);
        let (x1, y1) = self.pixel_to_geo_corner(width, 0);
        let (x2, y2) = self.pixel_to_geo_corner(0, height);
        let (x3, y3) = self.pixel_to_geo_corner(width, height);

        let min_x = x0.min(x1).min(x2).min(x3);
        let max_x = x0.max(x1).max(x2).max(x3);
        let min_y = y0.min(y1).min(y2).min(y3);
        let max_y = y0.max(y1).max(y2).max(y3);

        (min_x, min_y, max_x, max_y)
    }
}

impl Default for GeoTransform {
    fn default() -> Self {
        Self::new(0.0, 0.0, 1.0, -1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pixel_to_geo_roundtrip() {
        let gt = GeoTransform::new(100.0, 200.0, 10.0, -10.0);

        let (x, y) = gt.pixel_to_geo(5, 10);
        let (col, row) = gt.geo_to_pixel(x, y);

        assert_relative_eq!(col, 5.5, epsilon = 1e-10);
        assert_relative_eq!(row, 10.5, epsilon = 1e-10);
    }

    #[test]
    fn test_bounds() {
        let gt = GeoTransform::new(0.0, 100.0, 1.0, -1.0);
        let (min_x, min_y, max_x, max_y) = gt.bounds(100, 100);

        assert_relative_eq!(min_x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(min_y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(max_x, 100.0, epsilon = 1e-10);
        assert_relative_eq!(max_y, 100.0, epsilon = 1e-10);
    }
}
