//! Multiscale Curvatures (Florinsky 2025, Ch. 4)
//!
//! Uses 3rd-order bivariate polynomial fitting on a 5×5 neighborhood
//! to compute curvatures more robust against noise than standard 3×3 methods.
//!
//! The cubic fit ensures unbiased first-derivative estimates (p, q)
//! even when the terrain has cubic variation — unlike a quadratic fit
//! which leaks cubic trend into the gradient estimates.
//!
//! Closed-form LS solution: Florinsky 2025, Eqs. 4.14–4.22.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Type of curvature to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiscaleCurvatureType {
    /// Mean curvature: (kmax + kmin) / 2
    Mean,
    /// Gaussian curvature: kmax × kmin
    Gaussian,
    /// Maximum curvature
    Maximal,
    /// Minimum curvature
    Minimal,
    /// Profile curvature (in direction of steepest slope)
    Profile,
    /// Plan (tangential) curvature
    Plan,
}

/// Parameters for multiscale curvature
#[derive(Debug, Clone)]
pub struct MultiscaleCurvatureParams {
    /// Type of curvature to compute
    pub curvature_type: MultiscaleCurvatureType,
}

impl Default for MultiscaleCurvatureParams {
    fn default() -> Self {
        Self {
            curvature_type: MultiscaleCurvatureType::Mean,
        }
    }
}

/// Compute multiscale curvature using Florinsky's (2025) method
///
/// Fits a 3rd-order (cubic) bivariate polynomial to the 5×5 neighborhood
/// around each cell, then computes curvatures from the partial derivatives.
///
/// The cubic model on a symmetric 5×5 grid yields closed-form solutions
/// for all five partial derivatives (p, q, r, s, t) at the center.
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Curvature type
///
/// # Returns
/// Raster with the requested curvature
pub fn multiscale_curvatures(
    dem: &Raster<f64>,
    params: MultiscaleCurvatureParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cs = dem.cell_size();
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                // Need 5×5 neighborhood (margin of 2)
                if row < 2 || row >= rows - 2 || col < 2 || col >= cols - 2 {
                    continue;
                }

                // Check all 25 cells for NaN
                let mut has_nan = false;
                for dr in -2_isize..=2 {
                    for dc in -2_isize..=2 {
                        let v = unsafe { dem.get_unchecked(
                            (row as isize + dr) as usize,
                            (col as isize + dc) as usize,
                        )};
                        if v.is_nan() {
                            has_nan = true;
                            break;
                        }
                    }
                    if has_nan { break; }
                }
                if has_nan { continue; }

                // Fit cubic surface and get partial derivatives
                let (p, q, r, s, t) = fit_cubic_5x5(dem, row, col, cs);

                let g2 = p * p + q * q; // gradient magnitude squared

                *row_data_col = match params.curvature_type {
                    MultiscaleCurvatureType::Mean => {
                        // H = -((1+q²)r + (1+p²)t - 2pqs) / (2(1+p²+q²)^1.5)
                        if g2 < f64::EPSILON && r.abs() < f64::EPSILON && t.abs() < f64::EPSILON {
                            0.0
                        } else {
                            -((1.0 + q * q) * r + (1.0 + p * p) * t - 2.0 * p * q * s)
                                / (2.0 * (1.0 + g2).powf(1.5))
                        }
                    }
                    MultiscaleCurvatureType::Gaussian => {
                        // K = (rt - s²) / (1 + p² + q²)²
                        (r * t - s * s) / (1.0 + g2).powi(2)
                    }
                    MultiscaleCurvatureType::Profile => {
                        // kp = -(p²r + 2pqs + q²t) / ((p²+q²)(1+p²+q²)^1.5)
                        if g2 < f64::EPSILON { 0.0 }
                        else {
                            -(p * p * r + 2.0 * p * q * s + q * q * t)
                                / (g2 * (1.0 + g2).powf(1.5))
                        }
                    }
                    MultiscaleCurvatureType::Plan => {
                        // kh = -(q²r - 2pqs + p²t) / ((p²+q²)√(1+p²+q²))
                        if g2 < f64::EPSILON { 0.0 }
                        else {
                            -(q * q * r - 2.0 * p * q * s + p * p * t)
                                / (g2 * (1.0 + g2).sqrt())
                        }
                    }
                    MultiscaleCurvatureType::Maximal => {
                        let h = -((1.0 + q * q) * r + (1.0 + p * p) * t - 2.0 * p * q * s)
                            / (2.0 * (1.0 + g2).powf(1.5));
                        let k = (r * t - s * s) / (1.0 + g2).powi(2);
                        let disc = (h * h - k).max(0.0).sqrt();
                        h + disc
                    }
                    MultiscaleCurvatureType::Minimal => {
                        let h = -((1.0 + q * q) * r + (1.0 + p * p) * t - 2.0 * p * q * s)
                            / (2.0 * (1.0 + g2).powf(1.5));
                        let k = (r * t - s * s) / (1.0 + g2).powi(2);
                        let disc = (h * h - k).max(0.0).sqrt();
                        h - disc
                    }
                };
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

/// Fit a 3rd-order (cubic) bivariate polynomial to the 5×5 neighborhood
/// using the closed-form least-squares solution (Florinsky 2025, Eqs. 4.14–4.22).
///
/// Returns (p, q, r, s, t) — the actual partial derivatives at the center:
///   p = ∂z/∂x, q = ∂z/∂y, r = ∂²z/∂x², s = ∂²z/∂x∂y, t = ∂²z/∂y²
///
/// The cubic model on the symmetric 5×5 grid decouples into four independent
/// blocks due to parity symmetry, yielding analytical formulas:
///
///   Block A (even-even): r, t from {1, x², y²}
///   Block B (odd-x):     p from {x, x³, xy²}
///   Block C (odd-y):     q from {y, y³, x²y}
///   Block D (odd-odd):   s from {xy}
fn fit_cubic_5x5(dem: &Raster<f64>, row: usize, col: usize, cs: f64) -> (f64, f64, f64, f64, f64) {
    let h = cs;
    let h2 = h * h;

    // Accumulate weighted sums in grid coordinates (integer offsets)
    let mut sz = 0.0;     // Σ z
    let mut sx_z = 0.0;   // Σ x·z
    let mut sy_z = 0.0;   // Σ y·z
    let mut sx2_z = 0.0;  // Σ x²·z
    let mut sy2_z = 0.0;  // Σ y²·z
    let mut sxy_z = 0.0;  // Σ xy·z
    let mut sx3_z = 0.0;  // Σ x³·z
    let mut sy3_z = 0.0;  // Σ y³·z
    let mut sxy2_z = 0.0; // Σ xy²·z
    let mut sx2y_z = 0.0; // Σ x²y·z

    for di in -2_isize..=2 {
        let y = di as f64;
        for dj in -2_isize..=2 {
            let x = dj as f64;
            let v = unsafe { dem.get_unchecked(
                (row as isize + di) as usize,
                (col as isize + dj) as usize,
            )};
            sz += v;
            sx_z += x * v;
            sy_z += y * v;
            sx2_z += x * x * v;
            sy2_z += y * y * v;
            sxy_z += x * y * v;
            sx3_z += x * x * x * v;
            sy3_z += y * y * y * v;
            sxy2_z += x * y * y * v;
            sx2y_z += x * x * y * v;
        }
    }

    // -------------------------------------------------------------------
    // Block B: solve for a₁₀ (→ p) from the 3×3 system:
    //   M_B · [a₁₀, a₃₀, a₁₂]ᵀ = [sx_z, sx3_z, sxy2_z]ᵀ
    //
    //   M_B = [[50, 170, 100],
    //          [170, 650, 340],
    //          [100, 340, 340]]
    //
    //   det(M_B) = 504000
    //   First row of M_B⁻¹ = [105400, -23800, -7200] / 504000
    //                       = [527, -119, -36] / 2520
    // -------------------------------------------------------------------
    let p = (527.0 * sx_z - 119.0 * sx3_z - 36.0 * sxy2_z) / (2520.0 * h);

    // -------------------------------------------------------------------
    // Block C: identical structure by x↔y symmetry
    // -------------------------------------------------------------------
    let q = (527.0 * sy_z - 119.0 * sy3_z - 36.0 * sx2y_z) / (2520.0 * h);

    // -------------------------------------------------------------------
    // Block A: solve for a₂₀, a₀₂ from the 3×3 system:
    //   [[25, 50, 50], [50, 170, 100], [50, 100, 170]]
    //
    //   Row reduction gives: a₂₀ = (sx2_z - 2·sz) / 70
    //                        a₀₂ = (sy2_z - 2·sz) / 70
    //
    //   r = ∂²z/∂x² = 2·a₂₀,  t = ∂²z/∂y² = 2·a₀₂
    // -------------------------------------------------------------------
    let r = (sx2_z - 2.0 * sz) / (35.0 * h2);
    let t = (sy2_z - 2.0 * sz) / (35.0 * h2);

    // -------------------------------------------------------------------
    // Block D: single equation for a₁₁ = s
    //   a₁₁ = sxy_z / Σ(x²y²) = sxy_z / 100
    //   s = ∂²z/∂x∂y = a₁₁
    // -------------------------------------------------------------------
    let s = sxy_z / (100.0 * h2);

    (p, q, r, s, t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn bowl_dem(size: usize) -> Raster<f64> {
        let center = size as f64 / 2.0;
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        for row in 0..size {
            for col in 0..size {
                let dx = col as f64 - center;
                let dy = row as f64 - center;
                dem.set(row, col, dx * dx + dy * dy).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_multiscale_mean_curvature_bowl() {
        let dem = bowl_dem(20);
        let result = multiscale_curvatures(&dem, MultiscaleCurvatureParams {
            curvature_type: MultiscaleCurvatureType::Mean,
        }).unwrap();

        let v = result.get(10, 10).unwrap();
        // Bowl z=x²+y² → negative mean curvature (concave up in terrain convention)
        assert!(v < 0.0, "Bowl should have negative mean curvature (concave), got {}", v);
    }

    #[test]
    fn test_multiscale_gaussian_bowl() {
        let dem = bowl_dem(20);
        let result = multiscale_curvatures(&dem, MultiscaleCurvatureParams {
            curvature_type: MultiscaleCurvatureType::Gaussian,
        }).unwrap();

        let v = result.get(10, 10).unwrap();
        assert!(v > 0.0, "Bowl should have positive Gaussian curvature, got {}", v);
    }

    #[test]
    fn test_multiscale_flat() {
        let mut dem = Raster::filled(20, 20, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = multiscale_curvatures(&dem, MultiscaleCurvatureParams {
            curvature_type: MultiscaleCurvatureType::Mean,
        }).unwrap();

        let v = result.get(10, 10).unwrap();
        assert!(v.abs() < 0.01, "Flat should have ~0 curvature, got {}", v);
    }

    #[test]
    fn test_multiscale_profile_plan() {
        let dem = bowl_dem(20);

        let profile = multiscale_curvatures(&dem, MultiscaleCurvatureParams {
            curvature_type: MultiscaleCurvatureType::Profile,
        }).unwrap();

        let plan = multiscale_curvatures(&dem, MultiscaleCurvatureParams {
            curvature_type: MultiscaleCurvatureType::Plan,
        }).unwrap();

        let pf = profile.get(10, 10).unwrap();
        let pl = plan.get(10, 10).unwrap();

        // Bowl has both profile and plan curvature
        assert!(!pf.is_nan() && !pl.is_nan());
    }

    #[test]
    fn test_cubic_fit_linear_surface() {
        // Linear surface z = 3x + 2y: p=3, q=2, r=s=t=0
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                dem.set(row, col, 3.0 * col as f64 + 2.0 * row as f64).unwrap();
            }
        }

        let (p, q, r, s, t) = fit_cubic_5x5(&dem, 10, 10, 1.0);
        assert!((p - 3.0).abs() < 1e-10, "p should be 3.0, got {}", p);
        assert!((q - 2.0).abs() < 1e-10, "q should be 2.0, got {}", q);
        assert!(r.abs() < 1e-10, "r should be 0, got {}", r);
        assert!(s.abs() < 1e-10, "s should be 0, got {}", s);
        assert!(t.abs() < 1e-10, "t should be 0, got {}", t);
    }

    #[test]
    fn test_cubic_fit_quadratic_surface() {
        // Quadratic z = x² + y²: p=0 at center, q=0 at center, r=2, t=2, s=0
        let dem = bowl_dem(20);
        let (p, q, r, s, t) = fit_cubic_5x5(&dem, 10, 10, 1.0);
        assert!(p.abs() < 1e-6, "p should be ~0 at bowl center, got {}", p);
        assert!(q.abs() < 1e-6, "q should be ~0 at bowl center, got {}", q);
        assert!((r - 2.0).abs() < 1e-6, "r should be 2.0, got {}", r);
        assert!((t - 2.0).abs() < 1e-6, "t should be 2.0, got {}", t);
        assert!(s.abs() < 1e-6, "s should be 0, got {}", s);
    }

    #[test]
    fn test_cubic_fit_unbiased_on_cubic_surface() {
        // Cubic surface z = x³: at center p=0, q=0 (∂(x³)/∂x = 3x² = 0 at x=0)
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                let x = col as f64 - 10.0;
                dem.set(row, col, x * x * x).unwrap();
            }
        }

        let (p, _q, _r, _s, _t) = fit_cubic_5x5(&dem, 10, 10, 1.0);
        // Cubic fit should give p = 0 at center of z = x³
        // (quadratic fit would give biased nonzero value)
        assert!(p.abs() < 1e-6, "Cubic fit should give p=0 for z=x³ at center, got {}", p);
    }
}
