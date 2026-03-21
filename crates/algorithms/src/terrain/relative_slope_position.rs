//! Relative Slope Position (Normalized Height)
//!
//! RSP normalizes the position of each cell along the hillslope profile,
//! ranging from 0 (valley bottom) to 1 (ridge top).
//!
//! RSP = HAND / (HAND + valley_depth)
//!
//! where HAND is the Height Above Nearest Drainage and valley_depth is
//! the vertical distance from each cell to the interpolated ridge surface.
//!
//! Reference:
//! Gallant, J.C. & Dowling, T.I. (2003). A multiresolution index of valley
//! bottom flatness for mapping depositional areas.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Compute Relative Slope Position from HAND and valley depth.
///
/// RSP = HAND / (HAND + valley_depth)
///
/// Range [0, 1]: 0 = valley bottom, 1 = ridge top.
/// When both HAND and valley_depth are 0 (flat area), returns 0.5.
///
/// # Arguments
/// * `hand` - Height Above Nearest Drainage raster
/// * `valley_depth` - Valley depth raster
///
/// # Returns
/// Raster with RSP values in [0, 1]
pub fn relative_slope_position(
    hand: &Raster<f64>,
    valley_depth: &Raster<f64>,
) -> Result<Raster<f64>> {
    let (rows, cols) = hand.shape();
    let (vr, vc) = valley_depth.shape();

    if rows != vr || cols != vc {
        return Err(Error::SizeMismatch {
            er: rows,
            ec: cols,
            ar: vr,
            ac: vc,
        });
    }

    let nodata_h = hand.nodata();
    let nodata_v = valley_depth.nodata();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let h = unsafe { hand.get_unchecked(row, col) };
                let v = unsafe { valley_depth.get_unchecked(row, col) };

                // Skip nodata
                if h.is_nan()
                    || v.is_nan()
                    || nodata_h.is_some_and(|nd| (h - nd).abs() < f64::EPSILON)
                    || nodata_v.is_some_and(|nd| (v - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                let denom = h + v;
                if denom.abs() < 1e-10 {
                    // Both HAND and valley_depth are ~0 → flat area
                    row_data[col] = 0.5;
                } else {
                    row_data[col] = h / denom;
                }
            }
            row_data
        })
        .collect();

    let mut output = hand.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn make_raster(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_rsp_valley_bottom() {
        // HAND = 0 (at drainage), valley_depth > 0 → RSP = 0
        let hand = make_raster(vec![vec![0.0, 0.0, 0.0]]);
        let vd = make_raster(vec![vec![10.0, 20.0, 5.0]]);

        let result = relative_slope_position(&hand, &vd).unwrap();
        for col in 0..3 {
            let val = result.get(0, col).unwrap();
            assert!(
                val.abs() < 1e-10,
                "Valley bottom should have RSP=0, got {} at col {}",
                val,
                col
            );
        }
    }

    #[test]
    fn test_rsp_ridge_top() {
        // HAND > 0, valley_depth = 0 → RSP = 1
        let hand = make_raster(vec![vec![10.0, 20.0, 5.0]]);
        let vd = make_raster(vec![vec![0.0, 0.0, 0.0]]);

        let result = relative_slope_position(&hand, &vd).unwrap();
        for col in 0..3 {
            let val = result.get(0, col).unwrap();
            assert!(
                (val - 1.0).abs() < 1e-10,
                "Ridge top should have RSP=1, got {} at col {}",
                val,
                col
            );
        }
    }

    #[test]
    fn test_rsp_mid_slope() {
        // HAND = valley_depth → RSP = 0.5
        let hand = make_raster(vec![vec![5.0, 10.0, 15.0]]);
        let vd = make_raster(vec![vec![5.0, 10.0, 15.0]]);

        let result = relative_slope_position(&hand, &vd).unwrap();
        for col in 0..3 {
            let val = result.get(0, col).unwrap();
            assert!(
                (val - 0.5).abs() < 1e-10,
                "Mid-slope should have RSP=0.5, got {} at col {}",
                val,
                col
            );
        }
    }

    #[test]
    fn test_rsp_both_zero() {
        // Both HAND and valley_depth are 0 → RSP = 0.5
        let hand = make_raster(vec![vec![0.0]]);
        let vd = make_raster(vec![vec![0.0]]);

        let result = relative_slope_position(&hand, &vd).unwrap();
        let val = result.get(0, 0).unwrap();
        assert!(
            (val - 0.5).abs() < 1e-10,
            "Both zero should give RSP=0.5, got {}",
            val
        );
    }

    #[test]
    fn test_rsp_dimension_mismatch() {
        let hand = Raster::<f64>::new(5, 5);
        let vd = Raster::<f64>::new(3, 3);
        let result = relative_slope_position(&hand, &vd);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}
