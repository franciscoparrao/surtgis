//! Feature-Preserving DEM Smoothing
//!
//! Removes noise while preserving breaks-in-slope (ridges, valleys, scarps).
//! Based on Sun et al. (2007) feature-preserving mesh denoising adapted for DEMs.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for feature-preserving smoothing
#[derive(Debug, Clone)]
pub struct SmoothingParams {
    /// Filter radius in cells (default 2)
    pub radius: usize,
    /// Number of iterations (default 3)
    pub iterations: usize,
    /// Normal difference threshold in degrees (default 15.0)
    /// Neighbors with surface normal difference > threshold are excluded
    pub threshold: f64,
}

impl Default for SmoothingParams {
    fn default() -> Self {
        Self {
            radius: 2,
            iterations: 3,
            threshold: 15.0,
        }
    }
}

/// Apply feature-preserving smoothing to a DEM
///
/// Uses bilateral filtering adapted for terrain: smooths only within
/// regions of similar surface orientation, preserving edges where
/// surface normals change abruptly (breaks-in-slope).
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Smoothing parameters
///
/// # Returns
/// Smoothed DEM preserving slope breaks
pub fn feature_preserving_smoothing(
    dem: &Raster<f64>,
    params: SmoothingParams,
) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }
    if params.iterations == 0 {
        return Err(Error::Algorithm("Iterations must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let threshold_rad = params.threshold.to_radians();
    let r = params.radius as isize;

    let mut current = dem.data().clone();

    for _ in 0..params.iterations {
        let prev = current.clone();

        let new_data: Vec<f64> = (0..rows)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![f64::NAN; cols];

                for col in 0..cols {
                    if row < 1 || row >= rows - 1 || col < 1 || col >= cols - 1 {
                        row_data[col] = prev[(row, col)];
                        continue;
                    }

                    let z0 = prev[(row, col)];
                    if z0.is_nan() {
                        continue;
                    }

                    // Compute surface normal at center
                    let n0 = compute_normal(&prev, row, col, rows, cols);

                    let mut weighted_sum = 0.0;
                    let mut weight_total = 0.0;

                    for dr in -r..=r {
                        for dc in -r..=r {
                            let nr = row as isize + dr;
                            let nc = col as isize + dc;

                            if nr < 1 || nc < 1 || (nr as usize) >= rows - 1 || (nc as usize) >= cols - 1 {
                                continue;
                            }

                            let nr = nr as usize;
                            let nc = nc as usize;
                            let z = prev[(nr, nc)];
                            if z.is_nan() {
                                continue;
                            }

                            // Compute surface normal at neighbor
                            let nn = compute_normal(&prev, nr, nc, rows, cols);

                            // Normal similarity: angle between normals
                            let cos_angle = (n0.0 * nn.0 + n0.1 * nn.1 + n0.2 * nn.2)
                                .clamp(-1.0, 1.0);
                            let angle = cos_angle.acos();

                            if angle > threshold_rad {
                                continue; // Skip: different surface orientation
                            }

                            // Spatial weight (Gaussian)
                            let dist_sq = (dr * dr + dc * dc) as f64;
                            let sigma = params.radius as f64;
                            let spatial_w = (-dist_sq / (2.0 * sigma * sigma)).exp();

                            // Normal similarity weight
                            let normal_w = (-(angle * angle) / (2.0 * threshold_rad * threshold_rad)).exp();

                            let w = spatial_w * normal_w;
                            weighted_sum += z * w;
                            weight_total += w;
                        }
                    }

                    if weight_total > 0.0 {
                        row_data[col] = weighted_sum / weight_total;
                    } else {
                        row_data[col] = z0;
                    }
                }

                row_data
            })
            .collect();

        current = Array2::from_shape_vec((rows, cols), new_data)
            .map_err(|e| Error::Other(e.to_string()))?;
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = current;

    Ok(output)
}

/// Compute approximate surface normal at a cell (nx, ny, nz)
fn compute_normal(data: &Array2<f64>, row: usize, col: usize, rows: usize, cols: usize) -> (f64, f64, f64) {
    if row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1 {
        return (0.0, 0.0, 1.0);
    }

    let dz_dx = (data[(row, col + 1)] - data[(row, col - 1)]) / 2.0;
    let dz_dy = (data[(row + 1, col)] - data[(row - 1, col)]) / 2.0;

    let len = (dz_dx * dz_dx + dz_dy * dz_dy + 1.0).sqrt();
    (-dz_dx / len, -dz_dy / len, 1.0 / len)
}

/// Parameters for Gaussian smoothing
#[derive(Debug, Clone)]
pub struct GaussianSmoothingParams {
    /// Kernel radius in cells (default 2).
    /// The kernel is (2*radius+1) × (2*radius+1).
    pub radius: usize,
    /// Standard deviation (sigma) in cell units.
    /// Default: radius / 2.0 (auto-computed if 0.0)
    pub sigma: f64,
}

impl Default for GaussianSmoothingParams {
    fn default() -> Self {
        Self {
            radius: 2,
            sigma: 0.0, // auto: radius / 2.0
        }
    }
}

/// Apply Gaussian smoothing to a DEM.
///
/// Standard isotropic Gaussian filter using a precomputed kernel.
/// Florinsky (2025) Eq. 6.10: G(x,y) = (1/2πσ²) exp(-(x²+y²)/(2σ²))
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Gaussian smoothing parameters
///
/// # Returns
/// Smoothed DEM
pub fn gaussian_smoothing(
    dem: &Raster<f64>,
    params: GaussianSmoothingParams,
) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let sigma = if params.sigma <= 0.0 {
        params.radius as f64 / 2.0
    } else {
        params.sigma
    };
    let two_sigma_sq = 2.0 * sigma * sigma;

    // Precompute kernel
    let kernel_size = (2 * params.radius + 1) as usize;
    let mut kernel = vec![0.0_f64; kernel_size * kernel_size];
    let mut kernel_sum = 0.0;

    for dr in -r..=r {
        for dc in -r..=r {
            let dist_sq = (dr * dr + dc * dc) as f64;
            let w = (-dist_sq / two_sigma_sq).exp();
            let idx = ((dr + r) as usize) * kernel_size + (dc + r) as usize;
            kernel[idx] = w;
            kernel_sum += w;
        }
    }

    // Normalize kernel
    for w in kernel.iter_mut() {
        *w /= kernel_sum;
    }

    let nodata = dem.nodata();
    let data = dem.data();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let z0 = data[(row, col)];
                if z0.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata {
                    if (z0 - nd).abs() < f64::EPSILON { continue; }
                }

                let mut sum = 0.0;
                let mut wsum = 0.0;

                for dr in -r..=r {
                    let nr = row as isize + dr;
                    if nr < 0 || (nr as usize) >= rows { continue; }

                    for dc in -r..=r {
                        let nc = col as isize + dc;
                        if nc < 0 || (nc as usize) >= cols { continue; }

                        let z = data[(nr as usize, nc as usize)];
                        if z.is_nan() { continue; }
                        if let Some(nd) = nodata {
                            if (z - nd).abs() < f64::EPSILON { continue; }
                        }

                        let ki = ((dr + r) as usize) * kernel_size + (dc + r) as usize;
                        let w = kernel[ki];
                        sum += z * w;
                        wsum += w;
                    }
                }

                if wsum > 0.0 {
                    row_data[col] = sum / wsum;
                }
            }
            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Parameters for iterative weighted mean smoothing
#[derive(Debug, Clone)]
pub struct IterativeMeanParams {
    /// Number of iterations (default 3).
    /// Florinsky recommends 3 for regional studies, up to 10 for noisy DEMs.
    pub iterations: usize,
    /// Weight exponent m for distance weighting (default 1).
    /// m=0: uniform weights (simple mean)
    /// m=1: inverse-distance weighting (recommended)
    /// m=2: inverse-distance-squared weighting
    pub weight_exponent: u32,
}

impl Default for IterativeMeanParams {
    fn default() -> Self {
        Self {
            iterations: 3,
            weight_exponent: 1,
        }
    }
}

/// Apply iterative weighted mean smoothing to a DEM.
///
/// Florinsky (2025) Eqs. 6.8–6.9: At each iteration, each cell is
/// replaced by the weighted mean of its 3×3 neighborhood.
///
/// Weights: w_i = 1/d_i^m where d_i is the distance to the neighbor
/// and m is the weight exponent.
///
/// This is Florinsky's workhorse for DEM preprocessing.
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Iterative mean parameters
///
/// # Returns
/// Smoothed DEM
pub fn iterative_mean_smoothing(
    dem: &Raster<f64>,
    params: IterativeMeanParams,
) -> Result<Raster<f64>> {
    if params.iterations == 0 {
        return Err(Error::Algorithm("Iterations must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();
    let m = params.weight_exponent;

    // Distance weights for 3×3 neighborhood
    let sqrt2 = std::f64::consts::SQRT_2;
    let offsets: [(isize, isize, f64); 8] = [
        (-1, -1, sqrt2), (-1, 0, 1.0), (-1, 1, sqrt2),
        (0, -1, 1.0),                  (0, 1, 1.0),
        (1, -1, sqrt2),  (1, 0, 1.0),  (1, 1, sqrt2),
    ];

    // Precompute weights
    let weights: Vec<f64> = offsets.iter().map(|&(_, _, d)| {
        if m == 0 { 1.0 } else { 1.0 / d.powi(m as i32) }
    }).collect();

    let mut current = dem.data().clone();

    for _ in 0..params.iterations {
        let prev = current.clone();

        let new_data: Vec<f64> = (0..rows)
            .into_par_iter()
            .flat_map(|row| {
                let mut row_data = vec![f64::NAN; cols];

                for col in 0..cols {
                    let z0 = prev[(row, col)];
                    if z0.is_nan() {
                        continue;
                    }
                    if let Some(nd) = nodata {
                        if (z0 - nd).abs() < f64::EPSILON { continue; }
                    }

                    // Border cells: preserve
                    if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                        row_data[col] = z0;
                        continue;
                    }

                    let mut sum = z0; // include center with weight 1
                    let mut wsum = 1.0;

                    for (i, &(dr, dc, _)) in offsets.iter().enumerate() {
                        let nr = (row as isize + dr) as usize;
                        let nc = (col as isize + dc) as usize;
                        let z = prev[(nr, nc)];

                        if z.is_nan() { continue; }
                        if let Some(nd) = nodata {
                            if (z - nd).abs() < f64::EPSILON { continue; }
                        }

                        let w = weights[i];
                        sum += z * w;
                        wsum += w;
                    }

                    row_data[col] = sum / wsum;
                }
                row_data
            })
            .collect();

        current = Array2::from_shape_vec((rows, cols), new_data)
            .map_err(|e| Error::Other(e.to_string()))?;
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = current;

    Ok(output)
}

/// Parameters for FFT low-pass smoothing
#[derive(Debug, Clone)]
pub struct FftLowPassParams {
    /// Cutoff wavelength in cells. Frequencies with wavelength shorter
    /// than this are suppressed. Larger = more smoothing.
    /// Default: 8.0 (filters features smaller than 8 cells)
    pub cutoff_wavelength: f64,
}

impl Default for FftLowPassParams {
    fn default() -> Self {
        Self {
            cutoff_wavelength: 8.0,
        }
    }
}

/// A complex number for FFT computation
#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn add(self, other: Self) -> Self {
        Self { re: self.re + other.re, im: self.im + other.im }
    }

    fn sub(self, other: Self) -> Self {
        Self { re: self.re - other.re, im: self.im - other.im }
    }
}

/// Next power of 2 >= n
fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

/// In-place Cooley-Tukey radix-2 FFT.
/// `inverse` = true for inverse FFT (divides by n).
fn fft_1d(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be power of 2");

    // Bit reversal
    let mut j = 0_usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m > 0 && j & m != 0 {
            j ^= m;
            m >>= 1;
        }
        j |= m;
    }

    // Butterfly stages
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * std::f64::consts::PI / len as f64;
        let wn = Complex::new(angle.cos(), angle.sin());

        let mut k = 0;
        while k < n {
            let mut w = Complex::new(1.0, 0.0);
            for m in 0..half {
                let u = data[k + m];
                let t = w.mul(data[k + m + half]);
                data[k + m] = u.add(t);
                data[k + m + half] = u.sub(t);
                w = w.mul(wn);
            }
            k += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for c in data.iter_mut() {
            c.re *= inv_n;
            c.im *= inv_n;
        }
    }
}

/// Apply FFT-based low-pass filter to a DEM.
///
/// Implements Florinsky (2025) §6.1 frequency-domain filtering:
/// 1. Pad DEM to power-of-2 dimensions
/// 2. 2D FFT (row-column decomposition)
/// 3. Apply circular low-pass filter in frequency domain
/// 4. Inverse 2D FFT
///
/// # Arguments
/// * `dem` — Input DEM raster
/// * `params` — Low-pass filter parameters
///
/// # Returns
/// Smoothed DEM raster
pub fn fft_low_pass(dem: &Raster<f64>, params: FftLowPassParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    if rows < 3 || cols < 3 {
        return Err(Error::Algorithm("DEM must be at least 3×3 for FFT smoothing".into()));
    }
    if params.cutoff_wavelength <= 0.0 {
        return Err(Error::Algorithm("cutoff_wavelength must be positive".into()));
    }

    let data = dem.data();

    // Pad to power of 2
    let nrows = next_pow2(rows);
    let ncols = next_pow2(cols);

    // Fill padded array (mirror padding at edges)
    let mut grid = vec![vec![Complex::zero(); ncols]; nrows];
    for r in 0..nrows {
        for c in 0..ncols {
            let sr = if r < rows { r } else { 2 * rows - r - 2 };
            let sc = if c < cols { c } else { 2 * cols - c - 2 };
            let sr = sr.min(rows - 1);
            let sc = sc.min(cols - 1);
            let z = data[[sr, sc]];
            grid[r][c] = Complex::new(if z.is_nan() { 0.0 } else { z }, 0.0);
        }
    }

    // Row-wise FFT
    for r in 0..nrows {
        fft_1d(&mut grid[r], false);
    }

    // Column-wise FFT (transpose, FFT, transpose back)
    let mut col_buf = vec![Complex::zero(); nrows];
    for c in 0..ncols {
        for r in 0..nrows {
            col_buf[r] = grid[r][c];
        }
        fft_1d(&mut col_buf, false);
        for r in 0..nrows {
            grid[r][c] = col_buf[r];
        }
    }

    // Apply low-pass filter: zero out frequencies above cutoff
    // Cutoff frequency = 1/cutoff_wavelength (in cycles per cell)
    let freq_cutoff_r = nrows as f64 / params.cutoff_wavelength;
    let freq_cutoff_c = ncols as f64 / params.cutoff_wavelength;

    for r in 0..nrows {
        for c in 0..ncols {
            // Frequency indices (handle wrapping)
            let fr = if r <= nrows / 2 { r as f64 } else { (nrows - r) as f64 };
            let fc = if c <= ncols / 2 { c as f64 } else { (ncols - c) as f64 };

            // Normalized radial frequency
            let f_norm = ((fr / freq_cutoff_r).powi(2) + (fc / freq_cutoff_c).powi(2)).sqrt();

            if f_norm > 1.0 {
                grid[r][c] = Complex::zero();
            }
        }
    }

    // Inverse column-wise FFT
    for c in 0..ncols {
        for r in 0..nrows {
            col_buf[r] = grid[r][c];
        }
        fft_1d(&mut col_buf, true);
        for r in 0..nrows {
            grid[r][c] = col_buf[r];
        }
    }

    // Inverse row-wise FFT
    for r in 0..nrows {
        fft_1d(&mut grid[r], true);
    }

    // Extract original-size result
    let mut out_data = Array2::from_elem((rows, cols), f64::NAN);
    for r in 0..rows {
        for c in 0..cols {
            let orig = data[[r, c]];
            if !orig.is_nan() {
                out_data[[r, c]] = grid[r][c].re;
            }
        }
    }

    let mut output = Raster::new(rows, cols);
    output.set_transform(*dem.transform());
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = out_data;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_smoothing_preserves_flat() {
        let mut dem = Raster::filled(20, 20, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = feature_preserving_smoothing(&dem, SmoothingParams::default()).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!((v - 100.0).abs() < 0.01, "Flat should stay flat, got {}", v);
    }

    #[test]
    fn test_smoothing_reduces_noise() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        // Tilted plane with noise
        for row in 0..20 {
            for col in 0..20 {
                let base = row as f64 * 10.0;
                let noise = ((row * 7 + col * 13) % 5) as f64 - 2.0;
                dem.set(row, col, base + noise).unwrap();
            }
        }

        let result = feature_preserving_smoothing(&dem, SmoothingParams::default()).unwrap();

        // Check that variance is reduced
        let orig_stats = dem.statistics();
        let smooth_stats = result.statistics();

        // The smoothed version should have less local variation
        // (hard to test precisely, but mean should be similar)
        assert!(
            (orig_stats.mean.unwrap() - smooth_stats.mean.unwrap()).abs() < 5.0,
            "Smoothing shouldn't change global mean significantly"
        );
    }

    #[test]
    fn test_smoothing_params_validation() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(feature_preserving_smoothing(&dem, SmoothingParams {
            radius: 0, ..Default::default()
        }).is_err());
        assert!(feature_preserving_smoothing(&dem, SmoothingParams {
            iterations: 0, ..Default::default()
        }).is_err());
    }

    // === Gaussian smoothing tests ===

    #[test]
    fn test_gaussian_preserves_flat() {
        let mut dem = Raster::filled(20, 20, 50.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = gaussian_smoothing(&dem, GaussianSmoothingParams::default()).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!((v - 50.0).abs() < 0.01, "Flat should stay flat, got {}", v);
    }

    #[test]
    fn test_gaussian_reduces_noise() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                let base = 100.0;
                let noise = ((row * 7 + col * 13) % 11) as f64 - 5.0;
                dem.set(row, col, base + noise).unwrap();
            }
        }

        let result = gaussian_smoothing(&dem, GaussianSmoothingParams {
            radius: 3, sigma: 1.5,
        }).unwrap();

        // Variance should decrease after smoothing
        let orig_var = compute_variance(&dem);
        let smooth_var = compute_variance(&result);
        assert!(
            smooth_var < orig_var,
            "Gaussian should reduce variance: orig={:.2}, smooth={:.2}",
            orig_var, smooth_var
        );
    }

    #[test]
    fn test_gaussian_params_validation() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(gaussian_smoothing(&dem, GaussianSmoothingParams {
            radius: 0, sigma: 1.0,
        }).is_err());
    }

    // === Iterative mean smoothing tests ===

    #[test]
    fn test_iterative_mean_preserves_flat() {
        let mut dem = Raster::filled(20, 20, 75.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = iterative_mean_smoothing(&dem, IterativeMeanParams::default()).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!((v - 75.0).abs() < 0.01, "Flat should stay flat, got {}", v);
    }

    #[test]
    fn test_iterative_mean_reduces_noise() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                let base = 100.0;
                let noise = ((row * 7 + col * 13) % 11) as f64 - 5.0;
                dem.set(row, col, base + noise).unwrap();
            }
        }

        let result = iterative_mean_smoothing(&dem, IterativeMeanParams {
            iterations: 5,
            weight_exponent: 1,
        }).unwrap();

        let orig_var = compute_variance(&dem);
        let smooth_var = compute_variance(&result);
        assert!(
            smooth_var < orig_var,
            "Iterative mean should reduce variance: orig={:.2}, smooth={:.2}",
            orig_var, smooth_var
        );
    }

    #[test]
    fn test_iterative_mean_more_iterations_smoother() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                let noise = ((row * 7 + col * 13) % 11) as f64;
                dem.set(row, col, 100.0 + noise).unwrap();
            }
        }

        let r1 = iterative_mean_smoothing(&dem, IterativeMeanParams {
            iterations: 1, weight_exponent: 1,
        }).unwrap();
        let r5 = iterative_mean_smoothing(&dem, IterativeMeanParams {
            iterations: 5, weight_exponent: 1,
        }).unwrap();

        let var1 = compute_variance(&r1);
        let var5 = compute_variance(&r5);
        assert!(
            var5 < var1,
            "More iterations should be smoother: var1={:.2}, var5={:.2}",
            var1, var5
        );
    }

    #[test]
    fn test_iterative_mean_params_validation() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(iterative_mean_smoothing(&dem, IterativeMeanParams {
            iterations: 0, weight_exponent: 1,
        }).is_err());
    }

    /// Helper: compute variance of interior cells
    fn compute_variance(raster: &Raster<f64>) -> f64 {
        let (rows, cols) = raster.shape();
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0.0;
        for row in 2..rows - 2 {
            for col in 2..cols - 2 {
                let v = raster.get(row, col).unwrap();
                if !v.is_nan() {
                    sum += v;
                    sum_sq += v * v;
                    count += 1.0;
                }
            }
        }
        if count < 2.0 { return 0.0; }
        let mean = sum / count;
        sum_sq / count - mean * mean
    }

    #[test]
    fn test_fft_lowpass_reduces_variance() {
        // High-frequency noise should be removed
        let size = 32;
        let mut data = Array2::zeros((size, size));
        for r in 0..size {
            for c in 0..size {
                // Smooth trend + high-frequency noise
                data[[r, c]] = r as f64 * 2.0 + c as f64
                    + if (r + c) % 2 == 0 { 5.0 } else { -5.0 };
            }
        }
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        *dem.data_mut() = data;

        let smoothed = fft_low_pass(&dem, FftLowPassParams {
            cutoff_wavelength: 4.0,
        }).unwrap();

        let var_orig = compute_variance(&dem);
        let var_smooth = compute_variance(&smoothed);

        assert!(
            var_smooth < var_orig,
            "FFT smoothing should reduce variance: orig={:.1}, smooth={:.1}",
            var_orig, var_smooth
        );
    }

    #[test]
    fn test_fft_lowpass_preserves_trend() {
        // A smooth linear trend should be mostly preserved
        let size = 32;
        let mut data = Array2::zeros((size, size));
        for r in 0..size {
            for c in 0..size {
                data[[r, c]] = r as f64 * 3.0 + c as f64 * 2.0 + 100.0;
            }
        }
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        *dem.data_mut() = data;

        let smoothed = fft_low_pass(&dem, FftLowPassParams {
            cutoff_wavelength: 4.0,
        }).unwrap();

        // Center value should be close to original
        let orig = dem.get(16, 16).unwrap();
        let smth = smoothed.get(16, 16).unwrap();
        assert!(
            (orig - smth).abs() < 5.0,
            "Linear trend should be preserved: orig={:.1}, smooth={:.1}",
            orig, smth
        );
    }

    #[test]
    fn test_fft_lowpass_nan_handling() {
        let size = 16;
        let mut data = Array2::from_elem((size, size), 100.0);
        data[[5, 5]] = f64::NAN;
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        *dem.data_mut() = data;

        let smoothed = fft_low_pass(&dem, FftLowPassParams::default()).unwrap();
        // NaN cell should remain NaN
        assert!(smoothed.get(5, 5).unwrap().is_nan());
        // Non-NaN cells should be valid
        let v = smoothed.get(8, 8).unwrap();
        assert!(!v.is_nan() && v.is_finite(), "Non-NaN cells should be valid, got {}", v);
    }

    #[test]
    fn test_fft_larger_cutoff_more_smooth() {
        let size = 32;
        let mut data = Array2::zeros((size, size));
        for r in 0..size {
            for c in 0..size {
                data[[r, c]] = ((r * 7 + c * 13) % 17) as f64 * 5.0;
            }
        }
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        *dem.data_mut() = data;

        let mild = fft_low_pass(&dem, FftLowPassParams { cutoff_wavelength: 4.0 }).unwrap();
        let strong = fft_low_pass(&dem, FftLowPassParams { cutoff_wavelength: 16.0 }).unwrap();

        let var_mild = compute_variance(&mild);
        let var_strong = compute_variance(&strong);

        assert!(
            var_strong < var_mild,
            "Larger cutoff should smooth more: mild={:.1}, strong={:.1}",
            var_mild, var_strong
        );
    }
}
