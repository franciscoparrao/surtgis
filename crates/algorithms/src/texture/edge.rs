//! Edge detection filters
//!
//! Classic convolution-based edge detectors:
//! - Sobel: first-derivative gradient magnitude
//! - Laplacian: second-derivative zero-crossing detector

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Sobel edge detection.
///
/// Computes gradient magnitude using 3x3 Sobel operators:
/// `G = sqrt(Gx² + Gy²)`
///
/// Where Gx and Gy are the horizontal and vertical Sobel kernels.
///
/// # Arguments
/// * `raster` - Input raster
///
/// # Returns
/// Raster with gradient magnitude values (edges = high values)
pub fn sobel_edge(raster: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = raster.shape();
    if rows < 3 || cols < 3 {
        return Err(Error::Algorithm("Sobel requires at least 3x3 raster".into()));
    }

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            if row == 0 || row == rows - 1 {
                return row_data;
            }

            for col in 1..(cols - 1) {
                // 3x3 window
                let z = |r: usize, c: usize| -> f64 {
                    let v = unsafe { raster.get_unchecked(r, c) };
                    if v.is_nan() { 0.0 } else { v }
                };

                let z1 = z(row - 1, col - 1);
                let z2 = z(row - 1, col);
                let z3 = z(row - 1, col + 1);
                let z4 = z(row, col - 1);
                // z5 = center (not used)
                let z6 = z(row, col + 1);
                let z7 = z(row + 1, col - 1);
                let z8 = z(row + 1, col);
                let z9 = z(row + 1, col + 1);

                // Sobel Gx: horizontal gradient
                let gx = (z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7);
                // Sobel Gy: vertical gradient
                let gy = (z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3);

                row_data[col] = (gx * gx + gy * gy).sqrt();
            }

            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Laplacian filter (second-derivative edge detection).
///
/// Applies the 3x3 Laplacian kernel:
/// ```text
///  0  1  0
///  1 -4  1
///  0  1  0
/// ```
///
/// Detects edges as zero-crossings. High absolute values indicate edges.
///
/// # Arguments
/// * `raster` - Input raster
///
/// # Returns
/// Raster with Laplacian values (positive and negative around edges)
pub fn laplacian(raster: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = raster.shape();
    if rows < 3 || cols < 3 {
        return Err(Error::Algorithm("Laplacian requires at least 3x3 raster".into()));
    }

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            if row == 0 || row == rows - 1 {
                return row_data;
            }

            for col in 1..(cols - 1) {
                let center = unsafe { raster.get_unchecked(row, col) };
                if center.is_nan() {
                    continue;
                }

                let top = unsafe { raster.get_unchecked(row - 1, col) };
                let bottom = unsafe { raster.get_unchecked(row + 1, col) };
                let left = unsafe { raster.get_unchecked(row, col - 1) };
                let right = unsafe { raster.get_unchecked(row, col + 1) };

                if top.is_nan() || bottom.is_nan() || left.is_nan() || right.is_nan() {
                    continue;
                }

                row_data[col] = top + bottom + left + right - 4.0 * center;
            }

            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn gradient_raster(rows: usize, cols: usize) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, col as f64).unwrap(); // Horizontal gradient
            }
        }
        r
    }

    fn flat_raster(rows: usize, cols: usize, val: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, val);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn test_sobel_gradient() {
        let r = gradient_raster(10, 10);
        let result = sobel_edge(&r).unwrap();

        // Interior cell of horizontal gradient should have positive edge value
        let v = result.get(5, 5).unwrap();
        assert!(v > 0.0, "Horizontal gradient should produce edge, got {}", v);
    }

    #[test]
    fn test_sobel_flat() {
        let r = flat_raster(10, 10, 5.0);
        let result = sobel_edge(&r).unwrap();

        let v = result.get(5, 5).unwrap();
        assert!(v.abs() < 1e-10, "Flat surface should have zero Sobel, got {}", v);
    }

    #[test]
    fn test_laplacian_flat() {
        let r = flat_raster(10, 10, 5.0);
        let result = laplacian(&r).unwrap();

        let v = result.get(5, 5).unwrap();
        assert!(v.abs() < 1e-10, "Flat surface should have zero Laplacian, got {}", v);
    }

    #[test]
    fn test_laplacian_step() {
        // Step function: left half = 0, right half = 100
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                r.set(row, col, if col < 5 { 0.0 } else { 100.0 }).unwrap();
            }
        }

        let result = laplacian(&r).unwrap();

        // At the edge (col=4→5 boundary), Laplacian should be non-zero
        let v4 = result.get(5, 4).unwrap();
        let v5 = result.get(5, 5).unwrap();
        assert!(v4.abs() > 0.0 || v5.abs() > 0.0,
            "Step edge should produce non-zero Laplacian");
    }

    #[test]
    fn test_sobel_too_small() {
        let r = flat_raster(2, 2, 1.0);
        assert!(sobel_edge(&r).is_err());
    }
}
