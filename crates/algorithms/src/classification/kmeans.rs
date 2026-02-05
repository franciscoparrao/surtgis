//! K-means clustering for raster data
//!
//! Unsupervised classification by iteratively partitioning pixels
//! into k clusters based on spectral distance.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for K-means clustering
#[derive(Debug, Clone)]
pub struct KmeansParams {
    /// Number of clusters
    pub k: usize,
    /// Maximum iterations (default: 100)
    pub max_iterations: usize,
    /// Convergence threshold — stop when centroids move less than this (default: 0.001)
    pub convergence: f64,
    /// Random seed for initial centroid selection
    pub seed: u64,
}

impl Default for KmeansParams {
    fn default() -> Self {
        Self {
            k: 5,
            max_iterations: 100,
            convergence: 0.001,
            seed: 42,
        }
    }
}

/// K-means clustering on a single raster.
///
/// Treats each pixel as a 1D feature vector. For multi-band analysis,
/// run PCA first and use the first principal component, or call this
/// on each band independently.
///
/// # Arguments
/// * `raster` - Input raster (single band)
/// * `params` - K-means parameters
///
/// # Returns
/// Raster with cluster labels (1..k). NaN pixels remain NaN.
pub fn kmeans_raster(raster: &Raster<f64>, params: KmeansParams) -> Result<Raster<f64>> {
    if params.k < 2 {
        return Err(Error::Algorithm("K-means requires k >= 2".into()));
    }

    let (rows, cols) = raster.shape();

    // Collect valid pixels with their indices
    let mut values: Vec<(usize, f64)> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = unsafe { raster.get_unchecked(r, c) };
            if v.is_finite() {
                values.push((r * cols + c, v));
            }
        }
    }

    if values.len() < params.k {
        return Err(Error::Algorithm(format!(
            "Not enough valid pixels ({}) for {} clusters", values.len(), params.k
        )));
    }

    // Initialize centroids using deterministic spacing (k-means++ simplified)
    let mut centroids = initialize_centroids(&values, params.k, params.seed);

    // Iterative refinement
    let mut labels = vec![0usize; values.len()];

    for _iter in 0..params.max_iterations {
        // Assignment step: assign each pixel to nearest centroid
        labels.par_iter_mut().enumerate().for_each(|(i, label)| {
            let val = values[i].1;
            let mut best_dist = f64::INFINITY;
            let mut best_k = 0;
            for (k, &centroid) in centroids.iter().enumerate() {
                let dist = (val - centroid).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = k;
                }
            }
            *label = best_k;
        });

        // Update step: recompute centroids
        let mut new_centroids = vec![0.0; params.k];
        let mut counts = vec![0usize; params.k];

        for (i, &(_, val)) in values.iter().enumerate() {
            let k = labels[i];
            new_centroids[k] += val;
            counts[k] += 1;
        }

        let mut max_shift = 0.0_f64;
        for k in 0..params.k {
            if counts[k] > 0 {
                new_centroids[k] /= counts[k] as f64;
                max_shift = max_shift.max((new_centroids[k] - centroids[k]).abs());
            } else {
                new_centroids[k] = centroids[k]; // Keep empty cluster centroid
            }
        }

        centroids = new_centroids;

        if max_shift < params.convergence {
            break;
        }
    }

    // Build output raster
    let mut data = vec![f64::NAN; rows * cols];
    for (i, &(idx, _)) in values.iter().enumerate() {
        data[idx] = (labels[i] + 1) as f64; // 1-indexed classes
    }

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Initialize centroids by evenly spacing in value range (deterministic)
fn initialize_centroids(values: &[(usize, f64)], k: usize, _seed: u64) -> Vec<f64> {
    let mut sorted: Vec<f64> = values.iter().map(|&(_, v)| v).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    (0..k)
        .map(|i| {
            let idx = (i * n / k) + n / (2 * k);
            sorted[idx.min(n - 1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_kmeans_basic() {
        // Create raster with two distinct value groups
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                let val = if row < 5 { 10.0 } else { 100.0 };
                r.set(row, col, val).unwrap();
            }
        }

        let result = kmeans_raster(&r, KmeansParams { k: 2, ..Default::default() }).unwrap();

        // Top and bottom should have different classes
        let top = result.get(0, 0).unwrap();
        let bottom = result.get(9, 0).unwrap();
        assert!(top != bottom, "Different value groups should get different clusters");
        assert!(top >= 1.0 && top <= 2.0);
        assert!(bottom >= 1.0 && bottom <= 2.0);
    }

    #[test]
    fn test_kmeans_k_too_large() {
        let mut r = Raster::filled(2, 2, 1.0);
        r.set_transform(GeoTransform::new(0.0, 2.0, 1.0, -1.0));
        let result = kmeans_raster(&r, KmeansParams { k: 10, ..Default::default() });
        assert!(result.is_err());
    }

    #[test]
    fn test_kmeans_k_one() {
        let r = Raster::filled(5, 5, 1.0);
        let result = kmeans_raster(&r, KmeansParams { k: 1, ..Default::default() });
        assert!(result.is_err(), "k=1 should error");
    }
}
