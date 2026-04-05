//! Temporal anomaly detection.
//!
//! Computes per-pixel deviations from a reference period (baseline).
//! Useful for identifying drought, deforestation, or unusual conditions.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Method for computing anomaly values.
#[derive(Debug, Clone, Copy)]
pub enum AnomalyMethod {
    /// Z-score: (value - mean) / std
    ZScore,
    /// Absolute difference: value - mean
    Difference,
    /// Relative difference: (value - mean) / mean × 100
    PercentDifference,
}

impl AnomalyMethod {
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "zscore" | "z-score" | "z" => Ok(Self::ZScore),
            "difference" | "diff" | "absolute" => Ok(Self::Difference),
            "percent" | "percent_difference" | "relative" => Ok(Self::PercentDifference),
            _ => Err(Error::Other(format!("unknown anomaly method: '{}'. Use zscore, difference, or percent", s))),
        }
    }
}

/// Compute temporal anomaly: how each target raster deviates from a reference period.
///
/// # Arguments
/// * `reference` - Rasters forming the baseline/reference period
/// * `targets` - Rasters to evaluate against the baseline
/// * `method` - How to express the deviation
///
/// # Returns
/// One anomaly raster per target, in the same order.
pub fn temporal_anomaly(
    reference: &[&Raster<f64>],
    targets: &[&Raster<f64>],
    method: AnomalyMethod,
) -> Result<Vec<Raster<f64>>> {
    if reference.len() < 2 {
        return Err(Error::Other("anomaly requires at least 2 reference rasters".into()));
    }
    if targets.is_empty() {
        return Err(Error::Other("anomaly requires at least 1 target raster".into()));
    }

    let (rows, cols) = reference[0].shape();
    for r in reference.iter().chain(targets.iter()) {
        if r.shape() != (rows, cols) {
            return Err(Error::SizeMismatch {
                er: rows, ec: cols, ar: r.rows(), ac: r.cols(),
            });
        }
    }

    // Compute reference mean and std per pixel
    let n_ref = reference.len();
    let total = rows * cols;
    let mut ref_mean = vec![f64::NAN; total];
    let mut ref_std = vec![f64::NAN; total];

    ref_mean.par_chunks_mut(cols)
        .zip(ref_std.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(row, (mean_row, std_row))| {
            let mut vals = Vec::with_capacity(n_ref);
            for col in 0..cols {
                vals.clear();
                for r in reference {
                    let v = unsafe { r.get_unchecked(row, col) };
                    if v.is_finite() {
                        vals.push(v);
                    }
                }
                if !vals.is_empty() {
                    let m = vals.iter().sum::<f64>() / vals.len() as f64;
                    mean_row[col] = m;
                    if vals.len() >= 2 {
                        let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
                        std_row[col] = var.sqrt();
                    }
                }
            }
        });

    // Compute anomaly for each target
    let mut results = Vec::with_capacity(targets.len());

    for target in targets {
        let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);

        out.as_slice_mut().unwrap()
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let base = row * cols;
                for col in 0..cols {
                    let v = unsafe { target.get_unchecked(row, col) };
                    if !v.is_finite() { continue; }
                    let m = ref_mean[base + col];
                    if !m.is_finite() { continue; }

                    out_row[col] = match method {
                        AnomalyMethod::Difference => v - m,
                        AnomalyMethod::PercentDifference => {
                            if m.abs() > 1e-10 {
                                (v - m) / m * 100.0
                            } else {
                                f64::NAN
                            }
                        }
                        AnomalyMethod::ZScore => {
                            let s = ref_std[base + col];
                            if s.is_finite() && s > 1e-10 {
                                (v - m) / s
                            } else {
                                f64::NAN
                            }
                        }
                    };
                }
            });

        let mut result = Raster::from_array(out);
        result.set_transform(reference[0].transform().clone());
        result.set_nodata(Some(f64::NAN));
        if let Some(crs) = reference[0].crs() {
            result.set_crs(Some(crs.clone()));
        }
        results.push(result);
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_raster(val: f64) -> Raster<f64> {
        let arr = Array2::from_shape_vec((1, 1), vec![val]).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_zscore_anomaly() {
        // Reference: [10, 20, 30] → mean=20, std=10
        let r1 = make_raster(10.0);
        let r2 = make_raster(20.0);
        let r3 = make_raster(30.0);
        let target = make_raster(40.0); // z = (40-20)/10 = 2.0

        let result = temporal_anomaly(
            &[&r1, &r2, &r3], &[&target], AnomalyMethod::ZScore,
        ).unwrap();

        assert_eq!(result.len(), 1);
        assert!((result[0].data()[[0, 0]] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_difference_anomaly() {
        let r1 = make_raster(10.0);
        let r2 = make_raster(20.0);
        let r3 = make_raster(30.0);
        let target = make_raster(15.0); // diff = 15 - 20 = -5

        let result = temporal_anomaly(
            &[&r1, &r2, &r3], &[&target], AnomalyMethod::Difference,
        ).unwrap();

        assert!((result[0].data()[[0, 0]] - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_percent_anomaly() {
        let r1 = make_raster(100.0);
        let r2 = make_raster(200.0);
        // mean=150, target=225 → (225-150)/150*100 = 50%
        let target = make_raster(225.0);

        let result = temporal_anomaly(
            &[&r1, &r2], &[&target], AnomalyMethod::PercentDifference,
        ).unwrap();

        assert!((result[0].data()[[0, 0]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_multiple_targets() {
        let r1 = make_raster(10.0);
        let r2 = make_raster(20.0);
        let t1 = make_raster(30.0); // diff = 30-15 = 15
        let t2 = make_raster(5.0);  // diff = 5-15 = -10

        let result = temporal_anomaly(
            &[&r1, &r2], &[&t1, &t2], AnomalyMethod::Difference,
        ).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0].data()[[0, 0]] - 15.0).abs() < 1e-10);
        assert!((result[1].data()[[0, 0]] - (-10.0)).abs() < 1e-10);
    }
}
