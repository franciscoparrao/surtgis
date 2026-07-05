//! Band math operations
//!
//! Raster algebra operations: apply mathematical functions to one or
//! two rasters element-wise.

use crate::maybe_rayon::par_map_rows;
use surtgis_core::Result;
use surtgis_core::raster::Raster;

/// Binary operations for band math
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandMathOp {
    /// Per-pixel sum of the two bands.
    Add,
    /// Per-pixel difference (first minus second band).
    Subtract,
    /// Per-pixel product of the two bands.
    Multiply,
    /// Per-pixel ratio (first divided by second band).
    Divide,
    /// Per-pixel exponentiation (first band raised to the second).
    Power,
    /// Per-pixel minimum of the two bands.
    Min,
    /// Per-pixel maximum of the two bands.
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

    let output_data = par_map_rows(rows, cols, |row, out_row| {
        for (col, out_val) in out_row.iter_mut().enumerate() {
            let val = unsafe { raster.get_unchecked(row, col) };

            if val.is_nan() {
                continue;
            }
            if let Some(nd) = nodata
                && val == nd
            {
                continue;
            }

            *out_val = f(val);
        }
    });

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

/// Apply a binary operation between two rasters element-wise.
///
/// Both rasters must live on the same georeferenced grid (shape,
/// geotransform and EPSG-comparable CRS — see
/// [`surtgis_core::raster::check_aligned`]). Nodata in either input
/// produces nodata in the output.
///
/// # Arguments
/// * `a` - First raster
/// * `b` - Second raster
/// * `op` - Operation to apply
pub fn band_math_binary(a: &Raster<f64>, b: &Raster<f64>, op: BandMathOp) -> Result<Raster<f64>> {
    surtgis_core::raster::check_aligned(&[a, b])?;

    let (rows, cols) = a.shape();
    let nodata_a = a.nodata();
    let nodata_b = b.nodata();

    let output_data = par_map_rows(rows, cols, |row, out_row| {
        for (col, out_val) in out_row.iter_mut().enumerate() {
            let va = unsafe { a.get_unchecked(row, col) };
            let vb = unsafe { b.get_unchecked(row, col) };

            if va.is_nan() || vb.is_nan() {
                continue;
            }
            if let Some(nd) = nodata_a
                && va == nd
            {
                continue;
            }
            if let Some(nd) = nodata_b
                && vb == nd
            {
                continue;
            }

            *out_val = match op {
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
    });

    let mut output = a.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

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
