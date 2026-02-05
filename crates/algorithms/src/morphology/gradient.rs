//! Morphological gradient (dilation minus erosion)
//!
//! Highlights edges and boundaries by computing the difference between
//! the dilation and erosion of the input. The result is always non-negative.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

use super::dilate::dilate;
use super::element::StructuringElement;
use super::erode::erode;

/// Parameters for morphological gradient
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct GradientParams {
    /// Structuring element shape
    pub element: StructuringElement,
}


/// Morphological gradient algorithm
#[derive(Debug, Clone, Default)]
pub struct Gradient;

impl Algorithm for Gradient {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = GradientParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "MorphologicalGradient"
    }

    fn description(&self) -> &'static str {
        "Morphological gradient (dilation minus erosion) for edge detection"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        gradient(&input, &params.element)
    }
}

/// Compute the morphological gradient of a raster
///
/// Gradient = dilate - erode. Both operations produce NaN at the
/// same edge positions (radius-wide border), so the subtraction
/// propagates NaN there. Interior values are always >= 0.
///
/// # Arguments
/// * `raster` - Input raster
/// * `element` - Structuring element defining the neighborhood shape
pub fn gradient(raster: &Raster<f64>, element: &StructuringElement) -> Result<Raster<f64>> {
    let dilated = dilate(raster, element)?;
    let eroded = erode(raster, element)?;

    let (rows, cols) = raster.shape();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let d = unsafe { dilated.get_unchecked(row, col) };
                let e = unsafe { eroded.get_unchecked(row, col) };
                if d.is_nan() || e.is_nan() {
                    continue;
                }
                *row_data_col = d - e;
            }
            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), output_data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_raster(rows: usize, cols: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, value);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn test_gradient_uniform_is_zero() {
        let raster = make_raster(7, 7, 5.0);
        let result = gradient(&raster, &StructuringElement::Square(1)).unwrap();
        let val = result.get(3, 3).unwrap();
        assert!(
            val.abs() < 1e-10,
            "Gradient of uniform raster should be 0, got {}",
            val
        );
    }

    #[test]
    fn test_gradient_detects_edge() {
        let mut raster = make_raster(9, 9, 5.0);
        // Create a sharp step: left half = 5, right half = 15
        for row in 0..9 {
            for col in 5..9 {
                raster.set(row, col, 15.0).unwrap();
            }
        }

        let result = gradient(&raster, &StructuringElement::Square(1)).unwrap();
        // At the boundary (col=4), dilation picks 15, erosion picks 5 → gradient = 10
        let val = result.get(4, 4).unwrap();
        assert!(
            (val - 10.0).abs() < 1e-10,
            "Gradient at edge should be 10, got {}",
            val
        );
        // Away from boundary, gradient should be 0
        let val_flat = result.get(4, 2).unwrap();
        assert!(
            val_flat.abs() < 1e-10,
            "Gradient in flat area should be 0, got {}",
            val_flat
        );
    }

    #[test]
    fn test_gradient_non_negative() {
        let mut raster = make_raster(9, 9, 0.0);
        // Random-ish values
        for row in 0..9 {
            for col in 0..9 {
                raster.set(row, col, ((row * 7 + col * 3) % 20) as f64).unwrap();
            }
        }

        let result = gradient(&raster, &StructuringElement::Square(1)).unwrap();
        for row in 0..9 {
            for col in 0..9 {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    assert!(
                        val >= -1e-10,
                        "Gradient should be non-negative, got {} at ({}, {})",
                        val,
                        row,
                        col
                    );
                }
            }
        }
    }
}
