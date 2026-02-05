//! ISODATA clustering algorithm
//!
//! Iterative Self-Organizing Data Analysis Technique. Extends K-means
//! with automatic split and merge of clusters based on statistical criteria.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for ISODATA
#[derive(Debug, Clone)]
pub struct IsodataParams {
    /// Initial number of clusters
    pub initial_k: usize,
    /// Minimum number of clusters
    pub min_k: usize,
    /// Maximum number of clusters
    pub max_k: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Minimum number of pixels per cluster (below this → merge or discard)
    pub min_samples: usize,
    /// Maximum standard deviation before splitting a cluster
    pub max_std_dev: f64,
    /// Minimum distance between centroids before merging
    pub min_merge_distance: f64,
    /// Convergence threshold
    pub convergence: f64,
}

impl Default for IsodataParams {
    fn default() -> Self {
        Self {
            initial_k: 5,
            min_k: 2,
            max_k: 10,
            max_iterations: 50,
            min_samples: 10,
            max_std_dev: 10.0,
            min_merge_distance: 5.0,
            convergence: 0.001,
        }
    }
}

/// ISODATA unsupervised classification on a single raster.
///
/// # Arguments
/// * `raster` - Input raster
/// * `params` - ISODATA parameters
///
/// # Returns
/// Raster with cluster labels (1..k). NaN pixels remain NaN.
pub fn isodata(raster: &Raster<f64>, params: IsodataParams) -> Result<Raster<f64>> {
    if params.initial_k < 2 {
        return Err(Error::Algorithm("ISODATA requires initial_k >= 2".into()));
    }

    let (rows, cols) = raster.shape();

    // Collect valid pixels
    let mut values: Vec<(usize, f64)> = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let v = unsafe { raster.get_unchecked(r, c) };
            if v.is_finite() {
                values.push((r * cols + c, v));
            }
        }
    }

    if values.len() < params.initial_k {
        return Err(Error::Algorithm("Not enough valid pixels for ISODATA".into()));
    }

    // Initialize centroids evenly
    let mut sorted_vals: Vec<f64> = values.iter().map(|&(_, v)| v).collect();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_vals.len();
    let mut centroids: Vec<f64> = (0..params.initial_k)
        .map(|i| sorted_vals[(i * n / params.initial_k) + n / (2 * params.initial_k)])
        .collect();

    let mut labels = vec![0usize; values.len()];

    for _iter in 0..params.max_iterations {
        let k = centroids.len();

        // Assignment
        labels.par_iter_mut().enumerate().for_each(|(i, label)| {
            let val = values[i].1;
            let mut best_dist = f64::INFINITY;
            let mut best_k = 0;
            for (ki, &c) in centroids.iter().enumerate() {
                let dist = (val - c).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_k = ki;
                }
            }
            *label = best_k;
        });

        // Compute cluster statistics
        let mut sums = vec![0.0; k];
        let mut sq_sums = vec![0.0; k];
        let mut counts = vec![0usize; k];

        for (i, &(_, val)) in values.iter().enumerate() {
            let ki = labels[i];
            sums[ki] += val;
            sq_sums[ki] += val * val;
            counts[ki] += 1;
        }

        // Update centroids
        let mut new_centroids = Vec::new();
        let mut remap = vec![0usize; k];
        let mut new_idx = 0;

        for ki in 0..k {
            if counts[ki] >= params.min_samples {
                new_centroids.push(sums[ki] / counts[ki] as f64);
                remap[ki] = new_idx;
                new_idx += 1;
            }
            // Clusters below min_samples are dropped
        }

        // Remap labels for dropped clusters
        if new_centroids.len() < k {
            for i in 0..labels.len() {
                if counts[labels[i]] < params.min_samples {
                    // Assign to nearest remaining centroid
                    let val = values[i].1;
                    let mut best = 0;
                    let mut best_d = f64::INFINITY;
                    for (ki, &c) in new_centroids.iter().enumerate() {
                        let d = (val - c).abs();
                        if d < best_d {
                            best_d = d;
                            best = ki;
                        }
                    }
                    labels[i] = best;
                } else {
                    labels[i] = remap[labels[i]];
                }
            }
        }

        // Split: clusters with high std dev
        if new_centroids.len() < params.max_k {
            let mut to_split = Vec::new();
            for ki in 0..new_centroids.len() {
                let c = new_centroids[ki];
                let mut count = 0usize;
                let mut sq_sum = 0.0;
                for (i, &(_, val)) in values.iter().enumerate() {
                    if labels[i] == ki {
                        sq_sum += (val - c).powi(2);
                        count += 1;
                    }
                }
                if count > 2 * params.min_samples {
                    let std_dev = (sq_sum / count as f64).sqrt();
                    if std_dev > params.max_std_dev && new_centroids.len() + to_split.len() < params.max_k {
                        to_split.push((ki, std_dev));
                    }
                }
            }
            for (ki, std_dev) in to_split {
                let c = new_centroids[ki];
                new_centroids[ki] = c - std_dev * 0.5;
                new_centroids.push(c + std_dev * 0.5);
            }
        }

        // Merge: clusters that are too close
        if new_centroids.len() > params.min_k {
            let mut merged = vec![false; new_centroids.len()];
            let mut final_centroids = Vec::new();

            for i in 0..new_centroids.len() {
                if merged[i] { continue; }
                let mut merge_with = None;
                for j in (i + 1)..new_centroids.len() {
                    if merged[j] { continue; }
                    if (new_centroids[i] - new_centroids[j]).abs() < params.min_merge_distance {
                        merge_with = Some(j);
                        break;
                    }
                }
                if let Some(j) = merge_with {
                    final_centroids.push((new_centroids[i] + new_centroids[j]) / 2.0);
                    merged[j] = true;
                } else {
                    final_centroids.push(new_centroids[i]);
                }
            }
            new_centroids = final_centroids;
        }

        // Check convergence
        let max_shift = if centroids.len() == new_centroids.len() {
            centroids.iter().zip(&new_centroids)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        } else {
            f64::INFINITY
        };

        centroids = new_centroids;

        if max_shift < params.convergence {
            break;
        }
    }

    // Final assignment
    let k = centroids.len();
    labels.par_iter_mut().enumerate().for_each(|(i, label)| {
        let val = values[i].1;
        let mut best_dist = f64::INFINITY;
        let mut best_k = 0;
        for (ki, &c) in centroids.iter().enumerate() {
            let dist = (val - c).abs();
            if dist < best_dist {
                best_dist = dist;
                best_k = ki;
            }
        }
        *label = best_k;
    });

    // Build output
    let mut data = vec![f64::NAN; rows * cols];
    for (i, &(idx, _)) in values.iter().enumerate() {
        data[idx] = (labels[i] + 1) as f64;
    }

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| Error::Other(e.to_string()))?;

    let _ = k; // suppress unused warning
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_isodata_basic() {
        let mut r = Raster::new(20, 20);
        r.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                let val = if row < 10 { 10.0 } else { 100.0 };
                r.set(row, col, val).unwrap();
            }
        }

        let result = isodata(&r, IsodataParams {
            initial_k: 3,
            min_k: 2,
            max_k: 5,
            ..Default::default()
        }).unwrap();

        let top = result.get(0, 0).unwrap();
        let bottom = result.get(19, 0).unwrap();
        assert!(top.is_finite());
        assert!(bottom.is_finite());
        assert!(top != bottom, "Distinct groups should have different labels");
    }
}
