//! Top-hat and black-hat morphological transforms
//!
//! - **Top-hat** (white top-hat): original - opening. Extracts small bright
//!   features on a dark background.
//! - **Black-hat**: closing - original. Extracts small dark features on a
//!   bright background.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

use super::closing::closing;
use super::element::StructuringElement;
use super::opening::opening;

/// Parameters for top-hat transform
#[derive(Debug, Clone)]
pub struct TopHatParams {
    /// Structuring element shape
    pub element: StructuringElement,
}

impl Default for TopHatParams {
    fn default() -> Self {
        Self {
            element: StructuringElement::default(),
        }
    }
}

/// Top-hat (white top-hat) algorithm
#[derive(Debug, Clone, Default)]
pub struct TopHat;

impl Algorithm for TopHat {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = TopHatParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "TopHat"
    }

    fn description(&self) -> &'static str {
        "Top-hat transform (original minus opening) to extract bright features"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        top_hat(&input, &params.element)
    }
}

/// Parameters for black-hat transform
#[derive(Debug, Clone)]
pub struct BlackHatParams {
    /// Structuring element shape
    pub element: StructuringElement,
}

impl Default for BlackHatParams {
    fn default() -> Self {
        Self {
            element: StructuringElement::default(),
        }
    }
}

/// Black-hat algorithm
#[derive(Debug, Clone, Default)]
pub struct BlackHat;

impl Algorithm for BlackHat {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = BlackHatParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "BlackHat"
    }

    fn description(&self) -> &'static str {
        "Black-hat transform (closing minus original) to extract dark features"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        black_hat(&input, &params.element)
    }
}

/// Compute the top-hat (white top-hat) transform
///
/// Top-hat = original - opening. Extracts bright features smaller than
/// the structuring element. The NaN border comes from the opening side
/// (width = 2 * radius); where opening is NaN, the output is NaN.
///
/// # Arguments
/// * `raster` - Input raster
/// * `element` - Structuring element defining the neighborhood shape
pub fn top_hat(raster: &Raster<f64>, element: &StructuringElement) -> Result<Raster<f64>> {
    let opened = opening(raster, element)?;

    let (rows, cols) = raster.shape();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let orig = unsafe { raster.get_unchecked(row, col) };
                let op = unsafe { opened.get_unchecked(row, col) };
                if orig.is_nan() || op.is_nan() {
                    continue;
                }
                row_data[col] = orig - op;
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

/// Compute the black-hat transform
///
/// Black-hat = closing - original. Extracts dark features smaller than
/// the structuring element. The NaN border comes from the closing side
/// (width = 2 * radius); where closing is NaN, the output is NaN.
///
/// # Arguments
/// * `raster` - Input raster
/// * `element` - Structuring element defining the neighborhood shape
pub fn black_hat(raster: &Raster<f64>, element: &StructuringElement) -> Result<Raster<f64>> {
    let closed = closing(raster, element)?;

    let (rows, cols) = raster.shape();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let orig = unsafe { raster.get_unchecked(row, col) };
                let cl = unsafe { closed.get_unchecked(row, col) };
                if orig.is_nan() || cl.is_nan() {
                    continue;
                }
                row_data[col] = cl - orig;
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

    // --- Top-hat tests ---

    #[test]
    fn test_top_hat_uniform_is_zero() {
        let raster = make_raster(11, 11, 5.0);
        let result = top_hat(&raster, &StructuringElement::Square(1)).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(
            val.abs() < 1e-10,
            "Top-hat of uniform raster should be 0, got {}",
            val
        );
    }

    #[test]
    fn test_top_hat_detects_bright_spot() {
        let mut raster = make_raster(11, 11, 5.0);
        raster.set(5, 5, 100.0).unwrap();

        let result = top_hat(&raster, &StructuringElement::Square(1)).unwrap();
        // Opening removes the bright spot, so top_hat = 100 - 5 = 95
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 95.0).abs() < 1e-10,
            "Top-hat should detect bright spot (expected 95), got {}",
            val
        );
    }

    #[test]
    fn test_top_hat_non_negative() {
        let mut raster = make_raster(11, 11, 0.0);
        for row in 0..11 {
            for col in 0..11 {
                raster.set(row, col, ((row * 7 + col * 3) % 20) as f64).unwrap();
            }
        }

        let result = top_hat(&raster, &StructuringElement::Square(1)).unwrap();
        for row in 0..11 {
            for col in 0..11 {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    assert!(
                        val >= -1e-10,
                        "Top-hat should be non-negative, got {} at ({}, {})",
                        val,
                        row,
                        col
                    );
                }
            }
        }
    }

    // --- Black-hat tests ---

    #[test]
    fn test_black_hat_uniform_is_zero() {
        let raster = make_raster(11, 11, 5.0);
        let result = black_hat(&raster, &StructuringElement::Square(1)).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(
            val.abs() < 1e-10,
            "Black-hat of uniform raster should be 0, got {}",
            val
        );
    }

    #[test]
    fn test_black_hat_detects_dark_spot() {
        let mut raster = make_raster(11, 11, 100.0);
        raster.set(5, 5, 1.0).unwrap();

        let result = black_hat(&raster, &StructuringElement::Square(1)).unwrap();
        // Closing fills the dark spot, so black_hat = 100 - 1 = 99
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 99.0).abs() < 1e-10,
            "Black-hat should detect dark spot (expected 99), got {}",
            val
        );
    }

    #[test]
    fn test_black_hat_non_negative() {
        let mut raster = make_raster(11, 11, 0.0);
        for row in 0..11 {
            for col in 0..11 {
                raster.set(row, col, ((row * 7 + col * 3) % 20) as f64).unwrap();
            }
        }

        let result = black_hat(&raster, &StructuringElement::Square(1)).unwrap();
        for row in 0..11 {
            for col in 0..11 {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    assert!(
                        val >= -1e-10,
                        "Black-hat should be non-negative, got {} at ({}, {})",
                        val,
                        row,
                        col
                    );
                }
            }
        }
    }
}
