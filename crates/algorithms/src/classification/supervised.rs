//! Supervised classification algorithms
//!
//! Minimum distance and maximum likelihood classifiers.
//! Both use class signatures (mean and optional covariance) derived
//! from training samples.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// A class signature derived from training samples (single-band).
#[derive(Debug, Clone)]
pub struct ClassSignature {
    /// Class label (output value)
    pub label: f64,
    /// Mean value of the class
    pub mean: f64,
    /// Standard deviation of the class (used by maximum likelihood)
    pub std_dev: f64,
}

/// Minimum Distance classification.
///
/// Assigns each pixel to the class with the nearest centroid (mean).
/// Simple and fast but does not account for class variance.
///
/// # Arguments
/// * `raster` - Input raster
/// * `signatures` - Class signatures (at least 2)
///
/// # Returns
/// Raster with class labels. NaN pixels remain NaN.
pub fn minimum_distance(
    raster: &Raster<f64>,
    signatures: &[ClassSignature],
) -> Result<Raster<f64>> {
    if signatures.len() < 2 {
        return Err(Error::Algorithm("Minimum distance requires at least 2 class signatures".into()));
    }

    let (rows, cols) = raster.shape();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let v = unsafe { raster.get_unchecked(row, col) };
                if !v.is_finite() {
                    continue;
                }

                let mut best_dist = f64::INFINITY;
                let mut best_label = f64::NAN;
                for sig in signatures {
                    let dist = (v - sig.mean).abs();
                    if dist < best_dist {
                        best_dist = dist;
                        best_label = sig.label;
                    }
                }
                *out = best_label;
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

/// Maximum Likelihood classification.
///
/// Assigns each pixel to the class with the highest Gaussian probability
/// density, assuming each class follows a normal distribution.
///
/// `P(x|c) = (1 / (σ√(2π))) * exp(-(x-μ)² / (2σ²))`
///
/// # Arguments
/// * `raster` - Input raster
/// * `signatures` - Class signatures with mean and std_dev (at least 2)
///
/// # Returns
/// Raster with class labels. NaN pixels remain NaN.
pub fn maximum_likelihood(
    raster: &Raster<f64>,
    signatures: &[ClassSignature],
) -> Result<Raster<f64>> {
    if signatures.len() < 2 {
        return Err(Error::Algorithm("Maximum likelihood requires at least 2 class signatures".into()));
    }

    // Verify all std_devs are positive
    for sig in signatures {
        if sig.std_dev <= 0.0 {
            return Err(Error::Algorithm(format!(
                "Class {} has non-positive std_dev: {}", sig.label, sig.std_dev
            )));
        }
    }

    let (rows, cols) = raster.shape();

    // Precompute log-likelihood constants: -ln(σ) - 0.5*ln(2π)
    let log_consts: Vec<f64> = signatures.iter()
        .map(|sig| -sig.std_dev.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln())
        .collect();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let v = unsafe { raster.get_unchecked(row, col) };
                if !v.is_finite() {
                    continue;
                }

                let mut best_ll = f64::NEG_INFINITY;
                let mut best_label = f64::NAN;

                for (i, sig) in signatures.iter().enumerate() {
                    let z = (v - sig.mean) / sig.std_dev;
                    let ll = log_consts[i] - 0.5 * z * z;
                    if ll > best_ll {
                        best_ll = ll;
                        best_label = sig.label;
                    }
                }
                *out = best_label;
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

/// Auto-generate class signatures from a classified raster and a value raster.
///
/// # Arguments
/// * `classified` - Raster with known class labels (e.g., from field data)
/// * `values` - Raster with spectral/measurement values
///
/// # Returns
/// Vector of ClassSignature derived from the data
pub fn signatures_from_training(
    classified: &Raster<f64>,
    values: &Raster<f64>,
) -> Result<Vec<ClassSignature>> {
    let (rows, cols) = classified.shape();
    if values.shape() != (rows, cols) {
        return Err(Error::SizeMismatch {
            er: rows, ec: cols, ar: values.rows(), ac: values.cols(),
        });
    }

    // Collect values per class
    let mut class_data: std::collections::HashMap<i64, Vec<f64>> = std::collections::HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let class_val = unsafe { classified.get_unchecked(r, c) };
            let data_val = unsafe { values.get_unchecked(r, c) };
            if class_val.is_finite() && data_val.is_finite() {
                class_data.entry(class_val.round() as i64).or_default().push(data_val);
            }
        }
    }

    let mut signatures = Vec::new();
    for (label, vals) in &class_data {
        if vals.len() < 2 {
            continue;
        }
        let n = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / n;
        let variance = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt().max(1e-10); // Guard against zero variance

        signatures.push(ClassSignature {
            label: *label as f64,
            mean,
            std_dev,
        });
    }

    signatures.sort_by(|a, b| a.label.partial_cmp(&b.label).unwrap_or(std::cmp::Ordering::Equal));
    Ok(signatures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_signatures() -> Vec<ClassSignature> {
        vec![
            ClassSignature { label: 1.0, mean: 10.0, std_dev: 2.0 },
            ClassSignature { label: 2.0, mean: 50.0, std_dev: 5.0 },
            ClassSignature { label: 3.0, mean: 90.0, std_dev: 3.0 },
        ]
    }

    fn make_test_raster() -> Raster<f64> {
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                let val = match row / 3 {
                    0 => 12.0,  // Near class 1
                    1 => 48.0,  // Near class 2
                    _ => 88.0,  // Near class 3
                };
                r.set(row, col, val).unwrap();
            }
        }
        r
    }

    #[test]
    fn test_minimum_distance() {
        let r = make_test_raster();
        let sigs = make_signatures();

        let result = minimum_distance(&r, &sigs).unwrap();

        assert!((result.get(0, 0).unwrap() - 1.0).abs() < 1e-10);  // Near class 1
        assert!((result.get(4, 0).unwrap() - 2.0).abs() < 1e-10);  // Near class 2
        assert!((result.get(9, 0).unwrap() - 3.0).abs() < 1e-10);  // Near class 3
    }

    #[test]
    fn test_maximum_likelihood() {
        let r = make_test_raster();
        let sigs = make_signatures();

        let result = maximum_likelihood(&r, &sigs).unwrap();

        assert!((result.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((result.get(4, 0).unwrap() - 2.0).abs() < 1e-10);
        assert!((result.get(9, 0).unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_signatures_from_training() {
        let mut classified = Raster::new(10, 10);
        classified.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let mut values = Raster::new(10, 10);
        values.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                let class = if row < 5 { 1.0 } else { 2.0 };
                let val = if row < 5 { 10.0 + col as f64 } else { 50.0 + col as f64 };
                classified.set(row, col, class).unwrap();
                values.set(row, col, val).unwrap();
            }
        }

        let sigs = signatures_from_training(&classified, &values).unwrap();
        assert_eq!(sigs.len(), 2);
        assert!((sigs[0].label - 1.0).abs() < 1e-10);
        assert!((sigs[1].label - 2.0).abs() < 1e-10);
        assert!((sigs[0].mean - 14.5).abs() < 1e-10); // mean of 10..19
        assert!((sigs[1].mean - 54.5).abs() < 1e-10); // mean of 50..59
    }

    #[test]
    fn test_too_few_signatures() {
        let r = make_test_raster();
        assert!(minimum_distance(&r, &[make_signatures()[0].clone()]).is_err());
        assert!(maximum_likelihood(&r, &[make_signatures()[0].clone()]).is_err());
    }
}
