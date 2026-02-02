//! Band math operations
//!
//! Raster algebra operations: apply mathematical functions to one or
//! two rasters element-wise.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Binary operations for band math
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandMathOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Min,
    Max,
}

/// Apply a unary function to every cell in a raster.
///
/// Nodata cells (NaN) are preserved.
///
/// # Example
/// ```ignore
/// let log_raster = band_math(&input, |v| v.ln())?;
/// let scaled = band_math(&input, |v| v * 0.001)?;
/// ```
pub fn band_math<F>(raster: &Raster<f64>, f: F) -> Result<Raster<f64>>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let val = unsafe { raster.get_unchecked(row, col) };

                if val.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata {
                    if (val - nd).abs() < f64::EPSILON {
                        continue;
                    }
                }

                row_data[col] = f(val);
            }
            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Apply a binary operation between two rasters element-wise.
///
/// Both rasters must have the same dimensions. Nodata in either input
/// produces nodata in the output.
///
/// # Arguments
/// * `a` - First raster
/// * `b` - Second raster
/// * `op` - Operation to apply
pub fn band_math_binary(
    a: &Raster<f64>,
    b: &Raster<f64>,
    op: BandMathOp,
) -> Result<Raster<f64>> {
    if a.shape() != b.shape() {
        return Err(Error::SizeMismatch {
            er: a.rows(),
            ec: a.cols(),
            ar: b.rows(),
            ac: b.cols(),
        });
    }

    let (rows, cols) = a.shape();
    let nodata_a = a.nodata();
    let nodata_b = b.nodata();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let va = unsafe { a.get_unchecked(row, col) };
                let vb = unsafe { b.get_unchecked(row, col) };

                if va.is_nan() || vb.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata_a {
                    if (va - nd).abs() < f64::EPSILON {
                        continue;
                    }
                }
                if let Some(nd) = nodata_b {
                    if (vb - nd).abs() < f64::EPSILON {
                        continue;
                    }
                }

                row_data[col] = match op {
                    BandMathOp::Add => va + vb,
                    BandMathOp::Subtract => va - vb,
                    BandMathOp::Multiply => va * vb,
                    BandMathOp::Divide => {
                        if vb.abs() < 1e-10 {
                            f64::NAN
                        } else {
                            va / vb
                        }
                    }
                    BandMathOp::Power => va.powf(vb),
                    BandMathOp::Min => va.min(vb),
                    BandMathOp::Max => va.max(vb),
                };
            }
            row_data
        })
        .collect();

    let mut output = a.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_band(value: f64) -> Raster<f64> {
        let mut r = Raster::filled(5, 5, value);
        r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        r
    }

    #[test]
    fn test_band_math_unary() {
        let input = make_band(100.0);
        let result = band_math(&input, |v| v.sqrt()).unwrap();
        let val = result.get(2, 2).unwrap();
        assert!((val - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_band_math_scale() {
        let input = make_band(5000.0);
        let result = band_math(&input, |v| v * 0.0001).unwrap();
        let val = result.get(2, 2).unwrap();
        assert!((val - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_band_math_preserves_nan() {
        let mut input = make_band(100.0);
        input.set(2, 2, f64::NAN).unwrap();

        let result = band_math(&input, |v| v * 2.0).unwrap();
        assert!(result.get(2, 2).unwrap().is_nan());
    }

    #[test]
    fn test_band_math_binary_add() {
        let a = make_band(3.0);
        let b = make_band(7.0);

        let result = band_math_binary(&a, &b, BandMathOp::Add).unwrap();
        let val = result.get(2, 2).unwrap();
        assert!((val - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_band_math_binary_divide() {
        let a = make_band(10.0);
        let b = make_band(4.0);

        let result = band_math_binary(&a, &b, BandMathOp::Divide).unwrap();
        let val = result.get(2, 2).unwrap();
        assert!((val - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_band_math_binary_divide_by_zero() {
        let a = make_band(10.0);
        let b = make_band(0.0);

        let result = band_math_binary(&a, &b, BandMathOp::Divide).unwrap();
        let val = result.get(2, 2).unwrap();
        assert!(val.is_nan(), "Division by zero should produce NaN");
    }

    #[test]
    fn test_band_math_binary_min_max() {
        let a = make_band(3.0);
        let b = make_band(7.0);

        let min_result = band_math_binary(&a, &b, BandMathOp::Min).unwrap();
        assert!((min_result.get(2, 2).unwrap() - 3.0).abs() < 1e-10);

        let max_result = band_math_binary(&a, &b, BandMathOp::Max).unwrap();
        assert!((max_result.get(2, 2).unwrap() - 7.0).abs() < 1e-10);
    }
}
