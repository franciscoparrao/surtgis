//! Contour line generation via marching squares
//!
//! Generates a raster where cells on contour lines are marked with the contour
//! value and all other cells are NaN. This is a raster-based approximation of
//! vector contour extraction, useful for visualization overlays.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for contour generation
#[derive(Debug, Clone)]
pub struct ContourParams {
    /// Contour interval (elevation difference between successive contour lines)
    pub interval: f64,
    /// Base contour value (contours are generated at base + n*interval)
    pub base: f64,
}

impl Default for ContourParams {
    fn default() -> Self {
        Self {
            interval: 10.0,
            base: 0.0,
        }
    }
}

/// Generate contour lines as a raster.
///
/// Uses a simplified marching-squares approach: for each cell, checks whether
/// any contour level crosses between this cell and its right/bottom neighbor.
/// Cells that straddle a contour level are assigned that contour value.
///
/// # Arguments
/// * `dem` - Input elevation raster
/// * `params` - Contour parameters (interval, base)
///
/// # Returns
/// Raster where contour cells contain the contour elevation value and
/// non-contour cells are NaN.
pub fn contour_lines(dem: &Raster<f64>, params: ContourParams) -> Result<Raster<f64>> {
    if params.interval <= 0.0 {
        return Err(Error::Algorithm("Contour interval must be > 0".into()));
    }

    let (rows, cols) = dem.shape();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let v = unsafe { dem.get_unchecked(row, col) };
                if v.is_nan() {
                    continue;
                }

                // Check right neighbor
                if col + 1 < cols {
                    let vr = unsafe { dem.get_unchecked(row, col + 1) };
                    if !vr.is_nan() {
                        if let Some(level) = contour_crossing(v, vr, params.interval, params.base) {
                            row_data[col] = level;
                            continue;
                        }
                    }
                }

                // Check bottom neighbor
                if row + 1 < rows {
                    let vb = unsafe { dem.get_unchecked(row + 1, col) };
                    if !vb.is_nan() {
                        if let Some(level) = contour_crossing(v, vb, params.interval, params.base) {
                            row_data[col] = level;
                        }
                    }
                }
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Check if a contour level crosses between two elevation values.
/// Returns the nearest contour level if a crossing exists.
fn contour_crossing(a: f64, b: f64, interval: f64, base: f64) -> Option<f64> {
    let lo = a.min(b);
    let hi = a.max(b);

    // Find the lowest contour level >= lo
    let first_level = ((lo - base) / interval).ceil() * interval + base;

    if first_level >= lo && first_level <= hi && (hi - lo) > 1e-15 {
        Some(first_level)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn slope_raster(rows: usize, cols: usize, step: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, row as f64 * step).unwrap();
            }
        }
        r
    }

    #[test]
    fn test_contour_basic() {
        // Raster where each row = row_index (0, 1, 2, ... 19)
        let r = slope_raster(20, 10, 1.0);
        let result = contour_lines(&r, ContourParams { interval: 5.0, base: 0.0 }).unwrap();

        // Contours at 5, 10, 15. Cell at row 4 has value 4, row 5 has 5.
        // Crossing between row 4->5 means row 4 is marked.
        let v4 = result.get(4, 0).unwrap();
        assert!((v4 - 5.0).abs() < 1e-10, "Row 4 should have contour 5.0, got {}", v4);

        let v9 = result.get(9, 0).unwrap();
        assert!((v9 - 10.0).abs() < 1e-10, "Row 9 should have contour 10.0, got {}", v9);

        // Row 2 should NOT be on a contour
        let v2 = result.get(2, 0).unwrap();
        assert!(v2.is_nan(), "Row 2 should be NaN, got {}", v2);
    }

    #[test]
    fn test_contour_interval_error() {
        let r = slope_raster(5, 5, 1.0);
        let result = contour_lines(&r, ContourParams { interval: 0.0, base: 0.0 });
        assert!(result.is_err());
    }

    #[test]
    fn test_contour_with_base() {
        let r = slope_raster(20, 10, 1.0);
        let result = contour_lines(&r, ContourParams { interval: 10.0, base: 3.0 }).unwrap();

        // Contours at 3, 13. Row 2->3 crossing at 3.0.
        let v2 = result.get(2, 0).unwrap();
        assert!((v2 - 3.0).abs() < 1e-10, "Row 2 should have contour 3.0, got {}", v2);
    }
}
