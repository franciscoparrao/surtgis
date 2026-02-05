//! Gaussian Scale-Space (fGSS) framework for multiscale terrain analysis
//!
//! Implements the filtered Gaussian Scale-Space approach where the DEM is
//! progressively smoothed using Gaussian kernels of increasing width (σ).
//! At each scale, terrain derivatives and morphometric parameters are computed.
//!
//! This implements the approach described by Newman (2022, 2023) which
//! identifies fGSS as the optimal scaling method compared to aggregate
//! (resampling) and window-based approaches.
//!
//! ## Algorithm
//!
//! 1. Construct Gaussian kernel of size (6σ+1) × (6σ+1) truncated at 3σ
//! 2. Apply separable convolution (row-pass then col-pass) for efficiency
//! 3. Compute derivatives at the smoothed scale using Evans-Young
//! 4. Return scale-space pyramid: smoothed DEM + derivatives at each σ
//!
//! Reference:
//! Newman, D.R. et al. (2022). Evaluating scaling methodologies for
//!   landform classification. Geomorphology.
//! Newman, D.R. et al. (2023). Gaussian scale-space for curvature.
//! Lindeberg, T. (1994). Scale-Space Theory in Computer Vision. Springer.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use super::derivatives::evans_young;

/// Parameters for Gaussian Scale-Space computation
#[derive(Debug, Clone)]
pub struct GssParams {
    /// List of sigma values (standard deviations of Gaussian kernels).
    /// Each sigma defines a scale level. Measured in cell units.
    /// Example: vec![1.0, 2.0, 4.0, 8.0] for octave spacing.
    pub sigmas: Vec<f64>,
}

impl Default for GssParams {
    fn default() -> Self {
        Self {
            sigmas: vec![1.0, 2.0, 4.0, 8.0],
        }
    }
}

/// A single scale level in the scale-space pyramid
pub struct ScaleLevel {
    /// The sigma (smoothing width) for this level
    pub sigma: f64,
    /// The smoothed DEM at this scale
    pub smoothed: Raster<f64>,
}

/// Result of Gaussian Scale-Space computation
pub struct GssResult {
    /// Scale levels: smoothed DEMs at each sigma
    pub levels: Vec<ScaleLevel>,
}

/// Compute the Gaussian Scale-Space pyramid of a DEM.
///
/// Applies progressive Gaussian smoothing at each sigma level.
/// Uses separable convolution for O(n·k) per pixel instead of O(n·k²).
///
/// # Arguments
/// * `dem` — Input DEM raster
/// * `params` — Scale-space parameters (list of sigma values)
///
/// # Returns
/// [`GssResult`] containing smoothed DEMs at each scale level.
pub fn gaussian_scale_space(
    dem: &Raster<f64>,
    params: GssParams,
) -> Result<GssResult> {
    if params.sigmas.is_empty() {
        return Err(Error::Algorithm("At least one sigma value required".into()));
    }

    let rows = dem.rows();
    let cols = dem.cols();
    let data = dem.data();

    let mut levels = Vec::with_capacity(params.sigmas.len());

    for &sigma in &params.sigmas {
        if sigma <= 0.0 {
            return Err(Error::Algorithm(format!(
                "Sigma must be positive, got {}",
                sigma
            )));
        }

        let smoothed_data = gaussian_smooth_2d(data, rows, cols, sigma);

        let mut raster = Raster::new(rows, cols);
        raster.set_transform(*dem.transform());
        raster.set_nodata(dem.nodata());
        *raster.data_mut() = Array2::from_shape_vec((rows, cols), smoothed_data)
            .map_err(|e| Error::Other(e.to_string()))?;

        levels.push(ScaleLevel {
            sigma,
            smoothed: raster,
        });
    }

    Ok(GssResult { levels })
}

/// Compute derivatives at a specific scale level.
///
/// Applies Gaussian smoothing at the given sigma, then computes Evans-Young
/// derivatives on the smoothed DEM.
///
/// # Arguments
/// * `dem` — Input DEM raster
/// * `sigma` — Gaussian smoothing width in cell units
/// * `cellsize` — Grid cell size for derivative computation
///
/// # Returns
/// Raster-sized arrays of derivative values at each interior pixel.
pub fn scale_space_derivatives(
    dem: &Raster<f64>,
    sigma: f64,
    cellsize: f64,
) -> Result<Raster<f64>> {
    if sigma <= 0.0 {
        return Err(Error::Algorithm("Sigma must be positive".into()));
    }

    let rows = dem.rows();
    let cols = dem.cols();
    let data = dem.data();

    // Step 1: Smooth
    let smooth_vec = gaussian_smooth_2d(data, rows, cols, sigma);
    let smooth = Array2::from_shape_vec((rows, cols), smooth_vec)
        .map_err(|e| Error::Other(e.to_string()))?;

    // Step 2: Compute slope magnitude from derivatives at each interior cell
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            if row == 0 || row >= rows - 1 {
                return row_data;
            }
            for col in 1..cols - 1 {
                let z = [
                    smooth[[row - 1, col - 1]], smooth[[row - 1, col]], smooth[[row - 1, col + 1]],
                    smooth[[row, col - 1]], smooth[[row, col]], smooth[[row, col + 1]],
                    smooth[[row + 1, col - 1]], smooth[[row + 1, col]], smooth[[row + 1, col + 1]],
                ];

                if z.iter().any(|v| v.is_nan()) {
                    continue;
                }

                let d = evans_young(z, cellsize);
                // Return slope angle at this scale
                row_data[col] = d.slope_angle().to_degrees();
            }
            row_data
        })
        .collect();

    let mut output = Raster::new(rows, cols);
    output.set_transform(*dem.transform());
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Apply 2D Gaussian smoothing using separable convolution.
///
/// Splits the 2D Gaussian into row and column passes for efficiency.
/// Kernel truncated at 3σ.
fn gaussian_smooth_2d(
    data: &Array2<f64>,
    rows: usize,
    cols: usize,
    sigma: f64,
) -> Vec<f64> {
    let kernel = make_gaussian_kernel(sigma);
    let half = kernel.len() / 2;

    // Row pass
    let row_smoothed: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut out = vec![f64::NAN; cols];
            for col in 0..cols {
                let center = data[[row, col]];
                if center.is_nan() {
                    continue;
                }

                let mut sum = 0.0;
                let mut wsum = 0.0;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let c = col as isize + ki as isize - half as isize;
                    if c >= 0 && c < cols as isize {
                        let v = data[[row, c as usize]];
                        if !v.is_nan() {
                            sum += kw * v;
                            wsum += kw;
                        }
                    }
                }
                if wsum > 0.0 {
                    out[col] = sum / wsum;
                }
            }
            out
        })
        .collect();

    let row_arr = Array2::from_shape_vec((rows, cols), row_smoothed).unwrap();

    // Column pass
    let col_smoothed: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut out = vec![f64::NAN; cols];
            for col in 0..cols {
                let center = row_arr[[row, col]];
                if center.is_nan() {
                    continue;
                }

                let mut sum = 0.0;
                let mut wsum = 0.0;
                for (ki, &kw) in kernel.iter().enumerate() {
                    let r = row as isize + ki as isize - half as isize;
                    if r >= 0 && r < rows as isize {
                        let v = row_arr[[r as usize, col]];
                        if !v.is_nan() {
                            sum += kw * v;
                            wsum += kw;
                        }
                    }
                }
                if wsum > 0.0 {
                    out[col] = sum / wsum;
                }
            }
            out
        })
        .collect();

    col_smoothed
}

/// Create a 1D Gaussian kernel truncated at 3σ.
fn make_gaussian_kernel(sigma: f64) -> Vec<f64> {
    let half = (3.0 * sigma).ceil() as usize;
    let size = 2 * half + 1;
    let mut kernel = Vec::with_capacity(size);
    let denom = 2.0 * sigma * sigma;

    for i in 0..size {
        let x = i as f64 - half as f64;
        kernel.push((-x * x / denom).exp());
    }

    // Normalize
    let sum: f64 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::GeoTransform;

    fn make_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut data = Array2::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                // Simple surface: z = x + y with a bump in the center
                let x = c as f64;
                let y = r as f64;
                let cx = cols as f64 / 2.0;
                let cy = rows as f64 / 2.0;
                let dist = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                data[[r, c]] = x + y + 50.0 * (-dist * dist / 100.0).exp();
            }
        }
        let mut raster = Raster::new(rows, cols);
        raster.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        *raster.data_mut() = data;
        raster
    }

    #[test]
    fn test_gaussian_kernel() {
        let k = make_gaussian_kernel(1.0);
        // Should be symmetric
        let n = k.len();
        for i in 0..n / 2 {
            assert!(
                (k[i] - k[n - 1 - i]).abs() < 1e-10,
                "Kernel should be symmetric"
            );
        }
        // Should sum to ~1
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Kernel sum should be 1.0, got {}", sum);
        // Center should be the largest
        assert!(k[n / 2] >= k[0]);
    }

    #[test]
    fn test_gss_basic() {
        let dem = make_dem(30, 30);
        let params = GssParams {
            sigmas: vec![1.0, 2.0, 4.0],
        };

        let result = gaussian_scale_space(&dem, params).unwrap();
        assert_eq!(result.levels.len(), 3);
        assert!((result.levels[0].sigma - 1.0).abs() < 1e-10);
        assert!((result.levels[1].sigma - 2.0).abs() < 1e-10);
        assert!((result.levels[2].sigma - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gss_increasing_smoothness() {
        let dem = make_dem(30, 30);
        let params = GssParams {
            sigmas: vec![0.5, 2.0, 8.0],
        };

        let result = gaussian_scale_space(&dem, params).unwrap();

        // Higher sigma should produce smoother surfaces
        // Measure "roughness" as variance of Laplacian (r + t)
        let roughness: Vec<f64> = result.levels.iter().map(|level| {
            let d = level.smoothed.data();
            let rows = level.smoothed.rows();
            let cols = level.smoothed.cols();
            let mut sum = 0.0;
            let mut count = 0;
            for r in 1..rows - 1 {
                for c in 1..cols - 1 {
                    let lap = d[[r - 1, c]] + d[[r + 1, c]] + d[[r, c - 1]] + d[[r, c + 1]] - 4.0 * d[[r, c]];
                    sum += lap * lap;
                    count += 1;
                }
            }
            sum / count as f64
        }).collect();

        // Roughness should decrease with increasing sigma
        for i in 1..roughness.len() {
            assert!(
                roughness[i] < roughness[i - 1] + 1e-6,
                "σ={} roughness ({:.4}) should be <= σ={} roughness ({:.4})",
                result.levels[i].sigma, roughness[i],
                result.levels[i - 1].sigma, roughness[i - 1]
            );
        }
    }

    #[test]
    fn test_gss_preserves_mean() {
        // Gaussian smoothing should approximately preserve the mean
        let dem = make_dem(30, 30);
        let original_mean = {
            let d = dem.data();
            let sum: f64 = d.iter().sum();
            sum / d.len() as f64
        };

        let params = GssParams { sigmas: vec![2.0] };
        let result = gaussian_scale_space(&dem, params).unwrap();
        let smoothed_mean = {
            let d = result.levels[0].smoothed.data();
            let valid: Vec<f64> = d.iter().copied().filter(|v| !v.is_nan()).collect();
            valid.iter().sum::<f64>() / valid.len() as f64
        };

        assert!(
            (original_mean - smoothed_mean).abs() / original_mean.abs() < 0.05,
            "Mean should be approximately preserved: {:.2} vs {:.2}",
            original_mean, smoothed_mean
        );
    }

    #[test]
    fn test_gss_empty_sigmas() {
        let dem = make_dem(10, 10);
        let params = GssParams { sigmas: vec![] };
        assert!(gaussian_scale_space(&dem, params).is_err());
    }

    #[test]
    fn test_gss_negative_sigma() {
        let dem = make_dem(10, 10);
        let params = GssParams { sigmas: vec![-1.0] };
        assert!(gaussian_scale_space(&dem, params).is_err());
    }

    #[test]
    fn test_scale_space_derivatives() {
        let dem = make_dem(30, 30);
        let result = scale_space_derivatives(&dem, 2.0, 1.0).unwrap();

        // Interior cells should have valid slope values
        let center = result.get(15, 15).unwrap();
        assert!(!center.is_nan(), "Center slope should be valid");
        assert!(center >= 0.0, "Slope should be non-negative");

        // Edge cells should be NaN
        assert!(result.get(0, 0).unwrap().is_nan());
    }

    #[test]
    fn test_flat_surface_smoothing() {
        // Smoothing a flat surface should keep it flat
        let mut dem = Raster::new(20, 20);
        let data = Array2::from_elem((20, 20), 100.0);
        *dem.data_mut() = data;
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let params = GssParams { sigmas: vec![3.0] };
        let result = gaussian_scale_space(&dem, params).unwrap();

        for r in 0..20 {
            for c in 0..20 {
                let v = result.levels[0].smoothed.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(
                        (v - 100.0).abs() < 0.01,
                        "Flat surface should stay flat after smoothing, got {:.4} at ({},{})",
                        v, r, c
                    );
                }
            }
        }
    }
}
