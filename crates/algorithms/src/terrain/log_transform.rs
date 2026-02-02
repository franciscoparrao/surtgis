//! Log transform for curvature visualization
//!
//! Applies a sign-preserving logarithmic transformation to compress
//! the dynamic range of curvature values:
//!
//!   f(x) = sign(x) * ln(1 + |x|)
//!
//! This is essential for visualizing curvatures, which typically span
//! several orders of magnitude with extreme outliers at ridges/valleys.
//!
//! Reference: Florinsky, I.V. (2025) "Digital Terrain Analysis" 3rd ed.,
//! Chapter 8, Eq. 8.1

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Apply sign-preserving log transform to a raster
///
/// `f(x) = sign(x) * ln(1 + |x|)`
///
/// Preserves the sign of values while compressing dynamic range.
/// Useful for visualizing curvature, DEV, and other quantities
/// with heavy-tailed distributions.
///
/// # Arguments
/// * `raster` - Input raster (e.g., curvature output)
///
/// # Returns
/// Raster with log-transformed values
pub fn log_transform(raster: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let v = unsafe { raster.get_unchecked(row, col) };
                if v.is_nan() || nodata.map_or(false, |nd| (v - nd).abs() < f64::EPSILON) {
                    continue;
                }
                row_data[col] = v.signum() * (1.0 + v.abs()).ln();
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
    fn test_log_transform_zero() {
        let r = make_raster(5, 5, 0.0);
        let result = log_transform(&r).unwrap();
        let val = result.get(2, 2).unwrap();
        // sign(0)*ln(1+0) = 0
        assert!(val.abs() < 1e-10, "Expected 0, got {}", val);
    }

    #[test]
    fn test_log_transform_positive() {
        let r = make_raster(5, 5, 10.0);
        let result = log_transform(&r).unwrap();
        let val = result.get(2, 2).unwrap();
        let expected = (11.0_f64).ln();
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_log_transform_negative() {
        let r = make_raster(5, 5, -10.0);
        let result = log_transform(&r).unwrap();
        let val = result.get(2, 2).unwrap();
        let expected = -(11.0_f64).ln();
        assert!(
            (val - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_log_transform_preserves_sign() {
        let mut r = Raster::new(5, 5);
        r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        r.set(2, 2, 100.0).unwrap();
        r.set(2, 3, -100.0).unwrap();

        let result = log_transform(&r).unwrap();
        let pos = result.get(2, 2).unwrap();
        let neg = result.get(2, 3).unwrap();

        assert!(pos > 0.0, "Positive input → positive output");
        assert!(neg < 0.0, "Negative input → negative output");
        assert!(
            (pos.abs() - neg.abs()).abs() < 1e-10,
            "Symmetric: |f(x)| == |f(-x)|"
        );
    }

    #[test]
    fn test_log_transform_compresses_range() {
        let mut r = Raster::new(5, 5);
        r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        r.set(2, 2, 1000.0).unwrap();
        r.set(2, 3, 1.0).unwrap();

        let result = log_transform(&r).unwrap();
        let big = result.get(2, 2).unwrap();
        let small = result.get(2, 3).unwrap();

        // Input ratio: 1000:1. Output ratio should be much smaller.
        let input_ratio = 1000.0;
        let output_ratio = big / small;
        assert!(
            output_ratio < input_ratio,
            "Log should compress: input ratio {}, output ratio {}",
            input_ratio,
            output_ratio
        );
    }
}
