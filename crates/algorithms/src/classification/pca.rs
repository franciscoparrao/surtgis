//! Principal Component Analysis for multi-band rasters
//!
//! Computes PCA by building the covariance matrix of valid pixels across bands,
//! then extracting eigenvalues/eigenvectors via Jacobi iteration.
//! Returns the first N principal component bands.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for PCA
#[derive(Debug, Clone)]
pub struct PcaParams {
    /// Number of principal components to return (default: all)
    pub n_components: Option<usize>,
}

impl Default for PcaParams {
    fn default() -> Self {
        Self { n_components: None }
    }
}

/// Result of PCA
#[derive(Debug)]
pub struct PcaResult {
    /// Principal component rasters (PC1, PC2, ...)
    pub components: Vec<Raster<f64>>,
    /// Eigenvalues (variance explained by each component)
    pub eigenvalues: Vec<f64>,
    /// Proportion of variance explained by each component
    pub variance_explained: Vec<f64>,
}

/// Compute PCA on a stack of bands.
///
/// # Arguments
/// * `bands` - Slice of rasters (one per spectral band, must have same dimensions)
/// * `params` - PCA parameters
///
/// # Returns
/// PcaResult with principal component rasters and eigenvalues
pub fn pca(bands: &[&Raster<f64>], params: PcaParams) -> Result<PcaResult> {
    if bands.is_empty() {
        return Err(Error::Algorithm("PCA requires at least 1 band".into()));
    }

    let n_bands = bands.len();
    let (rows, cols) = bands[0].shape();

    // Verify all bands have same dimensions
    for (_i, band) in bands.iter().enumerate().skip(1) {
        if band.shape() != (rows, cols) {
            return Err(Error::SizeMismatch {
                er: rows, ec: cols, ar: band.rows(), ac: band.cols(),
            });
        }
    }

    // Collect valid pixel vectors (pixels where ALL bands are finite)
    let mut pixels: Vec<Vec<f64>> = Vec::new();
    let mut valid_mask = vec![false; rows * cols];

    for r in 0..rows {
        for c in 0..cols {
            let mut vals = Vec::with_capacity(n_bands);
            let mut all_valid = true;
            for band in bands {
                let v = unsafe { band.get_unchecked(r, c) };
                if !v.is_finite() {
                    all_valid = false;
                    break;
                }
                vals.push(v);
            }
            if all_valid {
                valid_mask[r * cols + c] = true;
                pixels.push(vals);
            }
        }
    }

    if pixels.is_empty() {
        return Err(Error::Algorithm("No valid pixels found across all bands".into()));
    }

    let n_pixels = pixels.len();

    // Compute mean of each band
    let mut means = vec![0.0; n_bands];
    for pixel in &pixels {
        for (i, v) in pixel.iter().enumerate() {
            means[i] += v;
        }
    }
    for m in &mut means {
        *m /= n_pixels as f64;
    }

    // Compute covariance matrix
    let mut cov = vec![vec![0.0; n_bands]; n_bands];
    for pixel in &pixels {
        for i in 0..n_bands {
            let di = pixel[i] - means[i];
            for j in i..n_bands {
                let dj = pixel[j] - means[j];
                cov[i][j] += di * dj;
            }
        }
    }
    for i in 0..n_bands {
        for j in i..n_bands {
            cov[i][j] /= (n_pixels - 1).max(1) as f64;
            if j > i {
                cov[j][i] = cov[i][j]; // Symmetric
            }
        }
    }

    // Eigen decomposition via Jacobi iteration
    let (eigenvalues, eigenvectors) = jacobi_eigen(&cov, n_bands)?;

    // Sort by eigenvalue descending
    let mut indices: Vec<usize> = (0..n_bands).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap_or(std::cmp::Ordering::Equal));

    let n_components = params.n_components.unwrap_or(n_bands).min(n_bands);
    let total_var: f64 = eigenvalues.iter().sum();

    let sorted_eigenvalues: Vec<f64> = indices.iter().take(n_components).map(|&i| eigenvalues[i]).collect();
    let variance_explained: Vec<f64> = sorted_eigenvalues.iter().map(|ev| {
        if total_var > 0.0 { ev / total_var } else { 0.0 }
    }).collect();

    // Project pixels onto principal components
    let mut components = Vec::with_capacity(n_components);
    for comp_idx in 0..n_components {
        let eigvec_idx = indices[comp_idx];
        let mut data = vec![f64::NAN; rows * cols];

        let mut pixel_idx = 0;
        for rc in 0..(rows * cols) {
            if valid_mask[rc] {
                let pixel = &pixels[pixel_idx];
                let mut projected = 0.0;
                for b in 0..n_bands {
                    projected += (pixel[b] - means[b]) * eigenvectors[b][eigvec_idx];
                }
                data[rc] = projected;
                pixel_idx += 1;
            }
        }

        let mut raster = bands[0].with_same_meta::<f64>(rows, cols);
        raster.set_nodata(Some(f64::NAN));
        *raster.data_mut() = Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| Error::Other(e.to_string()))?;
        components.push(raster);
    }

    Ok(PcaResult {
        components,
        eigenvalues: sorted_eigenvalues,
        variance_explained,
    })
}

/// Jacobi eigenvalue algorithm for symmetric matrices
fn jacobi_eigen(matrix: &[Vec<f64>], n: usize) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
    let max_iter = 100 * n * n;
    let eps = 1e-12;

    // Copy matrix
    let mut a: Vec<Vec<f64>> = matrix.to_vec();

    // Initialize eigenvectors as identity
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < eps {
            break; // Converged
        }

        // Compute rotation
        let theta = if (a[p][p] - a[q][q]).abs() < eps {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply rotation to matrix
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = cos_t * cos_t * a[p][p] + 2.0 * sin_t * cos_t * a[p][q] + sin_t * sin_t * a[q][q];
        new_a[q][q] = sin_t * sin_t * a[p][p] - 2.0 * sin_t * cos_t * a[p][q] + cos_t * cos_t * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = cos_t * vip + sin_t * viq;
            v[i][q] = -sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    Ok((eigenvalues, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_band(rows: usize, cols: usize, base: f64, step: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, base + (row * cols + col) as f64 * step).unwrap();
            }
        }
        r
    }

    #[test]
    fn test_pca_two_bands() {
        let b1 = make_band(10, 10, 0.0, 1.0);
        let b2 = make_band(10, 10, 0.0, 2.0); // Perfectly correlated with b1

        let result = pca(&[&b1, &b2], PcaParams::default()).unwrap();

        assert_eq!(result.components.len(), 2);
        assert_eq!(result.eigenvalues.len(), 2);
        // First component should explain most variance
        assert!(result.variance_explained[0] > 0.9,
            "PC1 should explain >90% variance, got {}", result.variance_explained[0]);
    }

    #[test]
    fn test_pca_n_components() {
        let b1 = make_band(10, 10, 0.0, 1.0);
        let b2 = make_band(10, 10, 100.0, 0.5);
        let b3 = make_band(10, 10, 50.0, 0.3);

        let result = pca(&[&b1, &b2, &b3], PcaParams { n_components: Some(2) }).unwrap();
        assert_eq!(result.components.len(), 2);
    }

    #[test]
    fn test_pca_empty() {
        let result = pca(&[], PcaParams::default());
        assert!(result.is_err());
    }
}
