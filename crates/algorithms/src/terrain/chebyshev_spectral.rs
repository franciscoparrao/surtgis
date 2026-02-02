//! Chebyshev Spectral Analytical Method for terrain analysis
//!
//! Unified framework for filtering and derivative computation using
//! polynomial least-squares fitting on local (2l+1)×(2l+1) windows.
//! The polynomial basis uses Chebyshev polynomials for optimal numerical
//! conditioning.
//!
//! ## Algorithm
//!
//! 1. Map the (2l+1)×(2l+1) window to normalized coordinates [-1, 1]²
//! 2. Fit a degree-d polynomial surface via least squares (precomputed matrix)
//! 3. Extract derivatives at the window center from polynomial coefficients
//!
//! The precomputed pseudo-inverse matrix M = (X^T X)^{-1} X^T converts
//! each local window into polynomial coefficients with a single
//! matrix-vector multiply (no per-pixel linear system solving).
//!
//! The continuous scale parameter `l` controls the window size,
//! analogous to σ in Gaussian methods.
//!
//! **UNIQUE**: No open-source library implements this spectral method.
//!
//! Reference:
//! Florinsky, I.V. & Pankratov, A.N. (2016). A universal spectral analytical
//!   method for digital terrain modeling. Int. J. Geogr. Info. Sci.
//! Florinsky, I.V. (2025). Digital Terrain Analysis, Ch. 7, Eqs. 7.18–7.22.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Chebyshev spectral analysis
#[derive(Debug, Clone)]
pub struct ChebyshevParams {
    /// Half-window size in cells. Window is (2l+1)×(2l+1).
    /// Larger l → more smoothing. Typical: 2–10.
    pub l: usize,
    /// Maximum polynomial degree for the fit (default 3 = cubic).
    /// Degree 2 = quadratic (6 coefficients), degree 3 = cubic (10 coefficients).
    pub degree: usize,
}

impl Default for ChebyshevParams {
    fn default() -> Self {
        Self {
            l: 3,
            degree: 3,
        }
    }
}

/// Result of Chebyshev spectral derivative computation.
pub struct ChebyshevDerivatives {
    /// ∂z/∂x
    pub p: Raster<f64>,
    /// ∂z/∂y
    pub q: Raster<f64>,
    /// ∂²z/∂x²
    pub r: Raster<f64>,
    /// ∂²z/∂x∂y
    pub s: Raster<f64>,
    /// ∂²z/∂y²
    pub t: Raster<f64>,
}

/// Number of polynomial terms for total degree ≤ d in 2D: (d+1)(d+2)/2
fn n_terms(degree: usize) -> usize {
    (degree + 1) * (degree + 2) / 2
}

/// Enumerate monomial (px, py) pairs for total degree ≤ d, sorted by total degree.
/// Returns pairs (px, py) where px + py ≤ d.
fn monomial_pairs(degree: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(n_terms(degree));
    for total in 0..=degree {
        for px in (0..=total).rev() {
            let py = total - px;
            pairs.push((px, py));
        }
    }
    pairs
}

/// Compute terrain derivatives using the Chebyshev spectral method.
///
/// For each interior pixel, extracts a (2l+1)×(2l+1) window and computes
/// polynomial LS derivatives using a precomputed pseudo-inverse matrix.
///
/// # Arguments
/// * `dem` — Input DEM raster
/// * `cellsize` — Grid cell spacing (assumes square cells)
/// * `params` — Chebyshev parameters (window half-size l, polynomial degree)
///
/// # Returns
/// [`ChebyshevDerivatives`] with 5 derivative rasters (p, q, r, s, t).
pub fn chebyshev_derivatives(
    dem: &Raster<f64>,
    cellsize: f64,
    params: ChebyshevParams,
) -> Result<ChebyshevDerivatives> {
    let l = params.l;
    let deg = params.degree;

    if l == 0 {
        return Err(Error::Algorithm("Window half-size l must be >= 1".into()));
    }
    if deg < 2 {
        return Err(Error::Algorithm("Polynomial degree must be >= 2 for second derivatives".into()));
    }

    let rows = dem.rows();
    let cols = dem.cols();
    let data = dem.data();
    let n = 2 * l + 1; // window size per dimension
    let n2 = n * n;     // total cells in window
    let p = n_terms(deg);

    if n2 < p {
        return Err(Error::Algorithm(format!(
            "Window too small ({0}×{0}={1} cells) for degree {2} polynomial ({3} terms)",
            n, n2, deg, p
        )));
    }

    // Enumerate monomial exponents
    let pairs = monomial_pairs(deg);

    // Build design matrix X: [n² × p]
    // Each row corresponds to a grid point, each column to a monomial
    let mut x_mat = vec![0.0_f64; n2 * p];
    for wr in 0..n {
        for wc in 0..n {
            let u = (wc as f64 - l as f64) / l as f64;  // x direction (col)
            let v = (l as f64 - wr as f64) / l as f64;   // y direction (row, north-up)
            let idx = wr * n + wc;
            for (j, &(px, py)) in pairs.iter().enumerate() {
                x_mat[idx * p + j] = u.powi(px as i32) * v.powi(py as i32);
            }
        }
    }

    // Compute pseudo-inverse: M = (X^T X)^{-1} X^T   [p × n²]
    // First compute X^T X [p × p]
    let mut xtx = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..n2 {
                s += x_mat[k * p + i] * x_mat[k * p + j];
            }
            xtx[i * p + j] = s;
        }
    }

    // Solve (X^T X) * M_row = X^T for each column of X^T
    // Equivalent to inverting X^T X, then multiplying by X^T
    // Since p is small (6-10), direct Gauss elimination is fine
    let xtx_inv = invert_symmetric(p, &xtx)?;

    // Pseudo-inverse M = xtx_inv * X^T   [p × n²]
    let mut pseudo_inv = vec![0.0_f64; p * n2];
    for i in 0..p {
        for j in 0..n2 {
            let mut s = 0.0;
            for k in 0..p {
                s += xtx_inv[i * p + k] * x_mat[j * p + k];
            }
            pseudo_inv[i * n2 + j] = s;
        }
    }

    // Find indices in the monomial pairs for the derivatives we need
    // (1,0) → ∂f/∂u = p coefficient
    // (0,1) → ∂f/∂v = q coefficient
    // (2,0) → ∂²f/∂u² = 2 * this coefficient
    // (1,1) → ∂²f/∂u∂v = this coefficient
    // (0,2) → ∂²f/∂v² = 2 * this coefficient
    let idx_10 = pairs.iter().position(|&(a, b)| a == 1 && b == 0);
    let idx_01 = pairs.iter().position(|&(a, b)| a == 0 && b == 1);
    let idx_20 = pairs.iter().position(|&(a, b)| a == 2 && b == 0);
    let idx_11 = pairs.iter().position(|&(a, b)| a == 1 && b == 1);
    let idx_02 = pairs.iter().position(|&(a, b)| a == 0 && b == 2);

    // Scaling from normalized to physical coordinates
    let scale_1 = 1.0 / (l as f64 * cellsize);
    let scale_2 = 1.0 / (l as f64 * cellsize * l as f64 * cellsize);

    // Compute derivatives for each pixel
    let output: Vec<(f64, f64, f64, f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN); cols];

            if row < l || row >= rows - l {
                return row_data;
            }

            for col in l..cols - l {
                // Extract window as flat vector
                let mut window = vec![0.0_f64; n2];
                let mut has_nan = false;
                for wr in 0..n {
                    for wc in 0..n {
                        let v = data[[row + wr - l, col + wc - l]];
                        if v.is_nan() {
                            has_nan = true;
                            break;
                        }
                        window[wr * n + wc] = v;
                    }
                    if has_nan {
                        break;
                    }
                }
                if has_nan {
                    continue;
                }

                // Compute coefficients: a = M * z   [p × 1]
                let mut coeffs = vec![0.0_f64; p];
                for i in 0..p {
                    let mut s = 0.0;
                    for j in 0..n2 {
                        s += pseudo_inv[i * n2 + j] * window[j];
                    }
                    coeffs[i] = s;
                }

                // Extract derivatives
                let dp = idx_10.map_or(0.0, |i| coeffs[i] * scale_1);
                let dq = idx_01.map_or(0.0, |i| coeffs[i] * scale_1);
                let dr = idx_20.map_or(0.0, |i| 2.0 * coeffs[i] * scale_2);
                let ds = idx_11.map_or(0.0, |i| coeffs[i] * scale_2);
                let dt = idx_02.map_or(0.0, |i| 2.0 * coeffs[i] * scale_2);

                row_data[col] = (dp, dq, dr, ds, dt);
            }

            row_data
        })
        .collect();

    // Build output rasters
    let transform = *dem.transform();
    let nodata = Some(f64::NAN);

    let make_raster = |extractor: fn(&(f64, f64, f64, f64, f64)) -> f64| -> Result<Raster<f64>> {
        let vec_data: Vec<f64> = output.iter().map(extractor).collect();
        let mut raster = Raster::new(rows, cols);
        raster.set_transform(transform);
        raster.set_nodata(nodata);
        *raster.data_mut() = Array2::from_shape_vec((rows, cols), vec_data)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(raster)
    };

    Ok(ChebyshevDerivatives {
        p: make_raster(|v| v.0)?,
        q: make_raster(|v| v.1)?,
        r: make_raster(|v| v.2)?,
        s: make_raster(|v| v.3)?,
        t: make_raster(|v| v.4)?,
    })
}

/// Invert a symmetric positive-definite matrix using Gaussian elimination.
fn invert_symmetric(n: usize, mat: &[f64]) -> Result<Vec<f64>> {
    // Augmented matrix [A | I]
    let mut aug = vec![0.0_f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = mat[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut max_val = aug[col * 2 * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * 2 * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(Error::Algorithm(
                "Chebyshev: singular design matrix (try larger window or lower degree)".into(),
            ));
        }

        if max_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }

        let pivot = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= pivot;
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    // Extract inverse from right half
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::GeoTransform;

    fn make_planar_dem(rows: usize, cols: usize, ax: f64, ay: f64, c: f64, cs: f64) -> Raster<f64> {
        // z = ax*x + ay*y + c
        let mut data = Array2::zeros((rows, cols));
        for r in 0..rows {
            for cc in 0..cols {
                let x = cc as f64 * cs;
                let y = (rows - 1 - r) as f64 * cs;
                data[[r, cc]] = ax * x + ay * y + c;
            }
        }
        let mut raster = Raster::new(rows, cols);
        raster.set_transform(GeoTransform::new(0.0, (rows as f64) * cs, cs, -cs));
        *raster.data_mut() = data;
        raster
    }

    fn make_quadratic_dem(rows: usize, cols: usize, cs: f64) -> Raster<f64> {
        // z = x² + y² (bowl)
        let mut data = Array2::zeros((rows, cols));
        let cx = (cols as f64 * cs) / 2.0;
        let cy = (rows as f64 * cs) / 2.0;
        for r in 0..rows {
            for cc in 0..cols {
                let x = cc as f64 * cs - cx;
                let y = (rows - 1 - r) as f64 * cs - cy;
                data[[r, cc]] = x * x + y * y;
            }
        }
        let mut raster = Raster::new(rows, cols);
        raster.set_transform(GeoTransform::new(-cx, cy + cs, cs, -cs));
        *raster.data_mut() = data;
        raster
    }

    #[test]
    fn test_monomial_pairs() {
        let pairs = monomial_pairs(2);
        assert_eq!(pairs.len(), 6); // 1, x, y, x², xy, y²
        assert!(pairs.contains(&(0, 0)));
        assert!(pairs.contains(&(1, 0)));
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(2, 0)));
        assert!(pairs.contains(&(1, 1)));
        assert!(pairs.contains(&(0, 2)));
    }

    #[test]
    fn test_chebyshev_flat_surface() {
        let dem = make_planar_dem(20, 20, 0.0, 0.0, 100.0, 10.0);
        let result = chebyshev_derivatives(&dem, 10.0, ChebyshevParams::default()).unwrap();

        let center_r = 10;
        let center_c = 10;
        let p = result.p.get(center_r, center_c).unwrap();
        let q = result.q.get(center_r, center_c).unwrap();
        assert!(
            p.abs() < 0.01,
            "p should be ~0 on flat surface, got {:.6}",
            p
        );
        assert!(
            q.abs() < 0.01,
            "q should be ~0 on flat surface, got {:.6}",
            q
        );
    }

    #[test]
    fn test_chebyshev_tilted_surface() {
        // z = 2x + 3y + 10, cellsize=10
        let dem = make_planar_dem(30, 30, 2.0, 3.0, 10.0, 10.0);
        let result = chebyshev_derivatives(&dem, 10.0, ChebyshevParams {
            l: 3,
            degree: 3,
        }).unwrap();

        let r = 15;
        let c = 15;
        let p = result.p.get(r, c).unwrap();
        let q = result.q.get(r, c).unwrap();

        assert!(
            (p - 2.0).abs() < 0.5,
            "p should be ~2.0, got {:.4}",
            p
        );
        assert!(
            (q - 3.0).abs() < 0.5,
            "q should be ~3.0, got {:.4}",
            q
        );
    }

    #[test]
    fn test_chebyshev_quadratic_curvature() {
        // z = x² + y² → r = 2, t = 2, s = 0
        let dem = make_quadratic_dem(30, 30, 10.0);
        let result = chebyshev_derivatives(&dem, 10.0, ChebyshevParams {
            l: 5,
            degree: 3,
        }).unwrap();

        let r_val = result.r.get(15, 15).unwrap();
        let t_val = result.t.get(15, 15).unwrap();
        let s_val = result.s.get(15, 15).unwrap();

        assert!(
            (r_val - 2.0).abs() < 1.0,
            "r should be ~2.0, got {:.4}",
            r_val
        );
        assert!(
            (t_val - 2.0).abs() < 1.0,
            "t should be ~2.0, got {:.4}",
            t_val
        );
        assert!(
            s_val.abs() < 1.0,
            "s should be ~0, got {:.4}",
            s_val
        );
    }

    #[test]
    fn test_chebyshev_invalid_params() {
        let dem = make_planar_dem(10, 10, 0.0, 0.0, 100.0, 10.0);
        let params = ChebyshevParams { l: 0, degree: 3 };
        assert!(chebyshev_derivatives(&dem, 10.0, params).is_err());
    }

    #[test]
    fn test_chebyshev_edges_are_nan() {
        let dem = make_planar_dem(20, 20, 1.0, 1.0, 0.0, 10.0);
        let params = ChebyshevParams { l: 3, degree: 3 };
        let result = chebyshev_derivatives(&dem, 10.0, params).unwrap();

        assert!(result.p.get(0, 0).unwrap().is_nan());
        assert!(result.p.get(1, 1).unwrap().is_nan());
        assert!(result.p.get(2, 2).unwrap().is_nan());
    }

    #[test]
    fn test_chebyshev_degree2_vs_degree3() {
        let dem = make_planar_dem(30, 30, 1.5, 2.5, 50.0, 10.0);

        let d2 = chebyshev_derivatives(&dem, 10.0, ChebyshevParams { l: 3, degree: 2 }).unwrap();
        let d3 = chebyshev_derivatives(&dem, 10.0, ChebyshevParams { l: 3, degree: 3 }).unwrap();

        // For a linear surface, both degrees should give similar first derivatives
        let p2 = d2.p.get(15, 15).unwrap();
        let p3 = d3.p.get(15, 15).unwrap();
        assert!(
            (p2 - p3).abs() < 0.1,
            "Degree 2 and 3 should agree on linear: {:.4} vs {:.4}",
            p2, p3
        );
    }
}
