//! Change detection algorithms
//!
//! Methods for detecting changes between two temporal rasters:
//! - Raster Difference with categorical thresholds
//! - Change Vector Analysis (CVA) for multi-band data

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for raster difference classification
#[derive(Debug, Clone)]
pub struct RasterDiffParams {
    /// Threshold for significant decrease (negative change)
    pub decrease_threshold: f64,
    /// Threshold for significant increase (positive change)
    pub increase_threshold: f64,
}

impl Default for RasterDiffParams {
    fn default() -> Self {
        Self {
            decrease_threshold: -1.0,
            increase_threshold: 1.0,
        }
    }
}

/// Change categories from raster difference
pub const CHANGE_DECREASE: f64 = 1.0;
pub const CHANGE_NO_CHANGE: f64 = 2.0;
pub const CHANGE_INCREASE: f64 = 3.0;

/// Compute raster difference with change categories.
///
/// `diff = after - before`
///
/// Output categories:
/// - 1.0 = Significant decrease (diff < decrease_threshold)
/// - 2.0 = No significant change
/// - 3.0 = Significant increase (diff > increase_threshold)
///
/// # Arguments
/// * `before` - Raster at time T1
/// * `after` - Raster at time T2
/// * `params` - Threshold parameters
///
/// # Returns
/// Tuple of (difference raster, categorical change raster)
pub fn raster_difference(
    before: &Raster<f64>,
    after: &Raster<f64>,
    params: RasterDiffParams,
) -> Result<(Raster<f64>, Raster<f64>)> {
    let (rows, cols) = before.shape();
    if after.shape() != (rows, cols) {
        return Err(Error::SizeMismatch {
            er: rows, ec: cols, ar: after.rows(), ac: after.cols(),
        });
    }

    let (diff_data, cat_data): (Vec<f64>, Vec<f64>) = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut diffs = Vec::with_capacity(cols);
            let mut cats = Vec::with_capacity(cols);
            for col in 0..cols {
                let b = unsafe { before.get_unchecked(row, col) };
                let a = unsafe { after.get_unchecked(row, col) };

                if b.is_nan() || a.is_nan() {
                    diffs.push(f64::NAN);
                    cats.push(f64::NAN);
                } else {
                    let d = a - b;
                    diffs.push(d);
                    cats.push(if d < params.decrease_threshold {
                        CHANGE_DECREASE
                    } else if d > params.increase_threshold {
                        CHANGE_INCREASE
                    } else {
                        CHANGE_NO_CHANGE
                    });
                }
            }
            diffs.into_iter().zip(cats).collect::<Vec<_>>()
        })
        .unzip();

    let mut diff_raster = before.with_same_meta::<f64>(rows, cols);
    diff_raster.set_nodata(Some(f64::NAN));
    *diff_raster.data_mut() = Array2::from_shape_vec((rows, cols), diff_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    let mut cat_raster = before.with_same_meta::<f64>(rows, cols);
    cat_raster.set_nodata(Some(f64::NAN));
    *cat_raster.data_mut() = Array2::from_shape_vec((rows, cols), cat_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok((diff_raster, cat_raster))
}

/// Change Vector Analysis (CVA) for two-band pairs.
///
/// Computes the magnitude and direction of change between two dates
/// in a 2D feature space (e.g., brightness/greenness, NDVI/NDWI).
///
/// `magnitude = sqrt((b1_after - b1_before)² + (b2_after - b2_before)²)`
/// `direction = atan2(b2_after - b2_before, b1_after - b1_before)`
///
/// # Arguments
/// * `band1_before` - Band 1 at time T1
/// * `band1_after` - Band 1 at time T2
/// * `band2_before` - Band 2 at time T1
/// * `band2_after` - Band 2 at time T2
///
/// # Returns
/// Tuple of (magnitude raster, direction raster in radians)
pub fn change_vector_analysis(
    band1_before: &Raster<f64>,
    band1_after: &Raster<f64>,
    band2_before: &Raster<f64>,
    band2_after: &Raster<f64>,
) -> Result<(Raster<f64>, Raster<f64>)> {
    let (rows, cols) = band1_before.shape();

    // Verify dimensions
    for (name, r) in [("band1_after", band1_after), ("band2_before", band2_before), ("band2_after", band2_after)] {
        if r.shape() != (rows, cols) {
            return Err(Error::Algorithm(format!(
                "{} dimensions {}x{} don't match {}x{}", name, r.rows(), r.cols(), rows, cols
            )));
        }
    }

    let (mag_data, dir_data): (Vec<f64>, Vec<f64>) = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut mags = Vec::with_capacity(cols);
            let mut dirs = Vec::with_capacity(cols);
            for col in 0..cols {
                let b1b = unsafe { band1_before.get_unchecked(row, col) };
                let b1a = unsafe { band1_after.get_unchecked(row, col) };
                let b2b = unsafe { band2_before.get_unchecked(row, col) };
                let b2a = unsafe { band2_after.get_unchecked(row, col) };

                if b1b.is_nan() || b1a.is_nan() || b2b.is_nan() || b2a.is_nan() {
                    mags.push(f64::NAN);
                    dirs.push(f64::NAN);
                } else {
                    let d1 = b1a - b1b;
                    let d2 = b2a - b2b;
                    mags.push((d1 * d1 + d2 * d2).sqrt());
                    dirs.push(d2.atan2(d1));
                }
            }
            mags.into_iter().zip(dirs).collect::<Vec<_>>()
        })
        .unzip();

    let mut mag_raster = band1_before.with_same_meta::<f64>(rows, cols);
    mag_raster.set_nodata(Some(f64::NAN));
    *mag_raster.data_mut() = Array2::from_shape_vec((rows, cols), mag_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    let mut dir_raster = band1_before.with_same_meta::<f64>(rows, cols);
    dir_raster.set_nodata(Some(f64::NAN));
    *dir_raster.data_mut() = Array2::from_shape_vec((rows, cols), dir_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok((mag_raster, dir_raster))
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_band(rows: usize, cols: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, value);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn test_raster_difference() {
        let before = make_band(5, 5, 10.0);
        let after = make_band(5, 5, 15.0);

        let (diff, cat) = raster_difference(&before, &after, RasterDiffParams::default()).unwrap();

        let d = diff.get(2, 2).unwrap();
        assert!((d - 5.0).abs() < 1e-10, "Diff should be 5, got {}", d);

        let c = cat.get(2, 2).unwrap();
        assert!((c - CHANGE_INCREASE).abs() < 1e-10, "Should be increase");
    }

    #[test]
    fn test_raster_difference_decrease() {
        let before = make_band(5, 5, 20.0);
        let after = make_band(5, 5, 10.0);

        let (_, cat) = raster_difference(&before, &after, RasterDiffParams {
            decrease_threshold: -5.0,
            increase_threshold: 5.0,
        }).unwrap();

        let c = cat.get(2, 2).unwrap();
        assert!((c - CHANGE_DECREASE).abs() < 1e-10, "Should be decrease");
    }

    #[test]
    fn test_cva() {
        let b1_before = make_band(5, 5, 0.2);
        let b1_after = make_band(5, 5, 0.5);   // +0.3 in band 1
        let b2_before = make_band(5, 5, 0.3);
        let b2_after = make_band(5, 5, 0.7);   // +0.4 in band 2

        let (mag, dir) = change_vector_analysis(&b1_before, &b1_after, &b2_before, &b2_after).unwrap();

        let m = mag.get(2, 2).unwrap();
        let expected_mag = (0.3_f64.powi(2) + 0.4_f64.powi(2)).sqrt(); // 0.5
        assert!((m - expected_mag).abs() < 1e-10, "Magnitude should be {}, got {}", expected_mag, m);

        let d = dir.get(2, 2).unwrap();
        let expected_dir = (0.4_f64).atan2(0.3);
        assert!((d - expected_dir).abs() < 1e-10, "Direction should be {}, got {}", expected_dir, d);
    }
}
