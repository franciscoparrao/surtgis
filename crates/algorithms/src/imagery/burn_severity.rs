//! Burn severity assessment using dNBR
//!
//! The differenced Normalized Burn Ratio (dNBR) quantifies fire-induced
//! changes in vegetation and soil by comparing pre-fire and post-fire
//! NBR values. Higher dNBR indicates more severe burning.
//!
//! dNBR = NBR_pre - NBR_post
//!
//! Severity classification follows USGS thresholds:
//! - Enhanced regrowth: dNBR < -0.25
//! - Unburned: -0.25 to 0.1
//! - Low severity: 0.1 to 0.27
//! - Moderate-low severity: 0.27 to 0.44
//! - Moderate-high severity: 0.44 to 0.66
//! - High severity: > 0.66
//!
//! Reference:
//! Key, C.H. & Benson, N.C. (2006). Landscape assessment: ground measure
//! of severity, the Composite Burn Index; and remote sensing of severity,
//! the Normalized Burn Ratio. FIREMON: Fire Effects Monitoring and
//! Inventory System.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use crate::imagery::indices::nbr;

/// Compute differenced Normalized Burn Ratio (dNBR).
///
/// dNBR = NBR_pre - NBR_post
///
/// where NBR = (NIR - SWIR) / (NIR + SWIR)
///
/// # Arguments
/// * `pre_nir` - Pre-fire Near-Infrared band
/// * `pre_swir` - Pre-fire Shortwave Infrared band
/// * `post_nir` - Post-fire Near-Infrared band
/// * `post_swir` - Post-fire Shortwave Infrared band
///
/// # Returns
/// Raster with dNBR values (typically in range [-2, 2])
pub fn dnbr(
    pre_nir: &Raster<f64>,
    pre_swir: &Raster<f64>,
    post_nir: &Raster<f64>,
    post_swir: &Raster<f64>,
) -> Result<Raster<f64>> {
    let (rows, cols) = pre_nir.shape();

    // Check all dimensions match
    for (name, raster) in [
        ("pre_swir", pre_swir),
        ("post_nir", post_nir),
        ("post_swir", post_swir),
    ] {
        let (r, c) = raster.shape();
        if rows != r || cols != c {
            return Err(Error::Other(format!(
                "{} dimensions {}x{} don't match pre_nir {}x{}",
                name, r, c, rows, cols
            )));
        }
    }

    // Compute NBR for pre and post
    let nbr_pre = nbr(pre_nir, pre_swir)?;
    let nbr_post = nbr(post_nir, post_swir)?;

    // dNBR = NBR_pre - NBR_post
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let pre = unsafe { nbr_pre.get_unchecked(row, col) };
                let post = unsafe { nbr_post.get_unchecked(row, col) };

                if pre.is_nan() || post.is_nan() {
                    continue;
                }

                row_data[col] = pre - post;
            }
            row_data
        })
        .collect();

    let mut output = pre_nir.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Classify dNBR into burn severity classes (USGS thresholds).
///
/// Classes:
/// - 1: Enhanced regrowth (dNBR < -0.25)
/// - 2: Unburned (-0.25 to 0.1)
/// - 3: Low severity (0.1 to 0.27)
/// - 4: Moderate-low severity (0.27 to 0.44)
/// - 5: Moderate-high severity (0.44 to 0.66)
/// - 6: High severity (> 0.66)
///
/// # Arguments
/// * `dnbr` - Input dNBR raster
///
/// # Returns
/// Raster with severity class values (1-6), NaN for nodata
pub fn burn_severity_classify(dnbr_raster: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = dnbr_raster.shape();
    let nodata = dnbr_raster.nodata();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let val = unsafe { dnbr_raster.get_unchecked(row, col) };

                if val.is_nan() || nodata.is_some_and(|nd| (val - nd).abs() < f64::EPSILON) {
                    continue;
                }

                row_data[col] = if val < -0.25 {
                    1.0 // Enhanced regrowth
                } else if val < 0.1 {
                    2.0 // Unburned
                } else if val < 0.27 {
                    3.0 // Low severity
                } else if val < 0.44 {
                    4.0 // Moderate-low severity
                } else if val < 0.66 {
                    5.0 // Moderate-high severity
                } else {
                    6.0 // High severity
                };
            }
            row_data
        })
        .collect();

    let mut output = dnbr_raster.with_same_meta::<f64>(rows, cols);
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

    fn make_band(data: Vec<f64>, rows: usize, cols: usize) -> Raster<f64> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 10.0, -10.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_dnbr_no_change() {
        // Same pre and post → dNBR = 0
        let nir = make_band(vec![0.5, 0.6, 0.7, 0.8], 2, 2);
        let swir = make_band(vec![0.2, 0.3, 0.3, 0.4], 2, 2);

        let result = dnbr(&nir, &swir, &nir, &swir).unwrap();

        for row in 0..2 {
            for col in 0..2 {
                let val = result.get(row, col).unwrap();
                assert!(
                    val.abs() < 1e-10,
                    "No change should give dNBR=0, got {} at ({},{})",
                    val,
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_dnbr_high_severity() {
        // Pre: healthy vegetation (high NIR, low SWIR) → high NBR
        // Post: burned (low NIR, high SWIR) → low/negative NBR
        // dNBR should be large positive
        let pre_nir = make_band(vec![0.8], 1, 1);
        let pre_swir = make_band(vec![0.1], 1, 1);
        let post_nir = make_band(vec![0.1], 1, 1);
        let post_swir = make_band(vec![0.4], 1, 1);

        let result = dnbr(&pre_nir, &pre_swir, &post_nir, &post_swir).unwrap();
        let val = result.get(0, 0).unwrap();

        // NBR_pre = (0.8-0.1)/(0.8+0.1) = 0.778
        // NBR_post = (0.1-0.4)/(0.1+0.4) = -0.6
        // dNBR = 0.778 - (-0.6) = 1.378
        assert!(
            val > 0.66,
            "High severity burn should have dNBR > 0.66, got {}",
            val
        );
    }

    #[test]
    fn test_burn_severity_classification() {
        let dnbr_data = vec![-0.5, 0.0, 0.15, 0.35, 0.5, 0.8];
        let dnbr_raster = make_band(dnbr_data, 1, 6);

        let result = burn_severity_classify(&dnbr_raster).unwrap();

        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for col in 0..6 {
            let val = result.get(0, col).unwrap();
            assert!(
                (val - expected[col]).abs() < 1e-10,
                "Class at col {} should be {}, got {}",
                col,
                expected[col],
                val
            );
        }
    }

    #[test]
    fn test_dnbr_dimension_mismatch() {
        let nir = Raster::<f64>::new(5, 5);
        let swir = Raster::<f64>::new(3, 3);
        let result = dnbr(&nir, &swir, &nir, &swir);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}
