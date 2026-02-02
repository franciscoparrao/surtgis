//! Ordinary Kriging interpolation
//!
//! Best Linear Unbiased Estimator (BLUE) for spatial data. Uses a fitted
//! variogram model to compute optimal interpolation weights that minimize
//! estimation variance while satisfying an unbiasedness constraint.
//!
//! The kriging system for n sample points:
//! ```text
//! [γ(x₁,x₁) ... γ(x₁,xₙ) 1] [w₁]   [γ(x₁,x₀)]
//! [   ...     ...    ...    .]  [. ] = [   ...    ]
//! [γ(xₙ,x₁) ... γ(xₙ,xₙ) 1] [wₙ]   [γ(xₙ,x₀)]
//! [  1       ...    1       0] [μ ]   [    1     ]
//! ```
//! where γ is the semivariance from the fitted variogram, x₀ is the
//! target location, and μ is the Lagrange multiplier ensuring Σwᵢ = 1.
//!
//! Reference:
//! Matheron, G. (1963). Principles of geostatistics. Economic Geology.
//! Cressie, N. (1993). Statistics for Spatial Data. Wiley.
//! Florinsky, I.V. (2025). Digital Terrain Analysis, §3.3.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::variogram::FittedVariogram;
use super::SamplePoint;

/// Parameters for Ordinary Kriging interpolation
#[derive(Debug, Clone)]
pub struct OrdinaryKrigingParams {
    /// Output raster rows
    pub rows: usize,
    /// Output raster columns
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
    /// Maximum number of nearest points to use per estimation (default 16).
    /// Using fewer points is faster but may be less accurate.
    pub max_points: usize,
    /// Maximum search radius. Points beyond this are ignored.
    /// `None` means use global search.
    pub max_radius: Option<f64>,
    /// Whether to produce a kriging variance raster alongside the estimate.
    pub compute_variance: bool,
}

impl Default for OrdinaryKrigingParams {
    fn default() -> Self {
        Self {
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
            max_points: 16,
            max_radius: None,
            compute_variance: false,
        }
    }
}

/// Result of Ordinary Kriging interpolation
pub struct KrigingResult {
    /// Interpolated values
    pub estimate: Raster<f64>,
    /// Kriging variance (estimation uncertainty). `None` if not requested.
    pub variance: Option<Raster<f64>>,
}

/// Perform Ordinary Kriging interpolation from scattered points to a raster grid.
///
/// # Arguments
/// * `points` — Sample points with (x, y, value)
/// * `variogram` — Fitted variogram model (from [`fit_variogram`] or [`fit_best_variogram`])
/// * `params` — Output grid specification and search parameters
///
/// # Returns
/// [`KrigingResult`] with interpolated raster and optionally kriging variance.
///
/// # Errors
/// - If fewer than 2 points are provided
/// - If the kriging system is singular at any location
pub fn ordinary_kriging(
    points: &[SamplePoint],
    variogram: &FittedVariogram,
    params: OrdinaryKrigingParams,
) -> Result<KrigingResult> {
    let n = points.len();
    if n < 2 {
        return Err(Error::Algorithm(
            "Kriging requires at least 2 sample points".into(),
        ));
    }

    let rows = params.rows;
    let cols = params.cols;
    let transform = params.transform;
    let max_pts = params.max_points.min(n);
    let max_radius_sq = params.max_radius.map(|r| r * r);
    let compute_var = params.compute_variance;

    // Pre-compute all pairwise distances for the sample points
    // (used to select neighbors and build the kriging matrix)
    let output: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN); cols];

            for col in 0..cols {
                let (x0, y0) = transform.pixel_to_geo(col, row);

                // Find nearest points
                let mut dists: Vec<(usize, f64)> = points
                    .iter()
                    .enumerate()
                    .map(|(i, pt)| {
                        let dx = pt.x - x0;
                        let dy = pt.y - y0;
                        (i, (dx * dx + dy * dy).sqrt())
                    })
                    .collect();

                // Filter by radius
                if let Some(max_sq) = max_radius_sq {
                    let max_r = max_sq.sqrt();
                    dists.retain(|(_, d)| *d <= max_r);
                }

                if dists.is_empty() {
                    continue;
                }

                // Sort by distance, take nearest max_pts
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let k = dists.len().min(max_pts);
                let neighbors = &dists[..k];

                // Check if target is very close to a sample point
                if neighbors[0].1 < 1e-12 {
                    let idx = neighbors[0].0;
                    row_data[col] = (points[idx].value, 0.0);
                    continue;
                }

                // Build kriging system (k+1) × (k+1)
                let m = k + 1;
                let mut mat = vec![0.0_f64; m * m];
                let mut rhs = vec![0.0_f64; m];

                // Fill K matrix: γ(xᵢ, xⱼ) for neighbors
                for i in 0..k {
                    let pi = &points[neighbors[i].0];
                    for j in 0..k {
                        if i == j {
                            mat[i * m + j] = 0.0; // γ(0) = 0 by convention
                        } else {
                            let pj = &points[neighbors[j].0];
                            let dx = pi.x - pj.x;
                            let dy = pi.y - pj.y;
                            let h = (dx * dx + dy * dy).sqrt();
                            mat[i * m + j] = variogram.evaluate(h);
                        }
                    }
                    // Lagrange constraint column
                    mat[i * m + k] = 1.0;
                    // Lagrange constraint row
                    mat[k * m + i] = 1.0;
                }
                // mat[k*m + k] = 0.0 already

                // RHS: γ(xᵢ, x₀) for each neighbor, plus 1.0 for constraint
                for i in 0..k {
                    rhs[i] = variogram.evaluate(neighbors[i].1);
                }
                rhs[k] = 1.0;

                // Solve the kriging system
                match kriging_solve(m, &mut mat, &mut rhs) {
                    Ok(solution) => {
                        // Compute estimate: z₀ = Σ wᵢ · zᵢ
                        let mut estimate = 0.0;
                        for i in 0..k {
                            estimate += solution[i] * points[neighbors[i].0].value;
                        }

                        // Kriging variance: σ² = Σ wᵢ·γ(xᵢ,x₀) + μ
                        let variance = if compute_var {
                            let mut var = solution[k]; // Lagrange multiplier μ
                            for i in 0..k {
                                var += solution[i] * variogram.evaluate(neighbors[i].1);
                            }
                            var.max(0.0) // Variance should be non-negative
                        } else {
                            0.0
                        };

                        row_data[col] = (estimate, variance);
                    }
                    Err(_) => {
                        // Singular system — fall back to IDW-like behavior
                        let mut sum_w = 0.0;
                        let mut sum_wz = 0.0;
                        for (idx, dist) in neighbors {
                            let w = 1.0 / (dist * dist);
                            sum_w += w;
                            sum_wz += w * points[*idx].value;
                        }
                        if sum_w > 0.0 {
                            row_data[col] = (sum_wz / sum_w, f64::NAN);
                        }
                    }
                }
            }

            row_data
        })
        .collect();

    // Separate estimate and variance
    let est_data: Vec<f64> = output.iter().map(|(e, _)| *e).collect();
    let mut estimate = Raster::new(rows, cols);
    estimate.set_transform(transform);
    estimate.set_nodata(Some(f64::NAN));
    *estimate.data_mut() = Array2::from_shape_vec((rows, cols), est_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    let variance = if compute_var {
        let var_data: Vec<f64> = output.iter().map(|(_, v)| *v).collect();
        let mut var_raster = Raster::new(rows, cols);
        var_raster.set_transform(transform);
        var_raster.set_nodata(Some(f64::NAN));
        *var_raster.data_mut() = Array2::from_shape_vec((rows, cols), var_data)
            .map_err(|e| Error::Other(e.to_string()))?;
        Some(var_raster)
    } else {
        None
    };

    Ok(KrigingResult { estimate, variance })
}

/// Solve Ax = b using Gaussian elimination with partial pivoting.
/// Specialized for the small kriging systems (typically 5–20 unknowns).
fn kriging_solve(n: usize, mat: &mut [f64], rhs: &mut [f64]) -> Result<Vec<f64>> {
    // Forward elimination
    for col in 0..n {
        let mut max_val = mat[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = mat[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(Error::Algorithm("Kriging: singular matrix".into()));
        }

        if max_row != col {
            for j in 0..n {
                let a = col * n + j;
                let b = max_row * n + j;
                mat.swap(a, b);
            }
            rhs.swap(col, max_row);
        }

        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            mat[row * n + col] = 0.0;
            for j in (col + 1)..n {
                mat[row * n + j] -= factor * mat[col * n + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for col in (0..n).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..n {
            sum -= mat[col * n + j] * x[j];
        }
        x[col] = sum / mat[col * n + col];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::variogram::{
        empirical_variogram, fit_best_variogram, VariogramParams,
    };

    fn make_params(rows: usize, cols: usize, extent: (f64, f64, f64, f64)) -> OrdinaryKrigingParams {
        let (x_min, y_min, x_max, y_max) = extent;
        let x_res = (x_max - x_min) / cols as f64;
        let y_res = -(y_max - y_min) / rows as f64;
        OrdinaryKrigingParams {
            rows,
            cols,
            transform: GeoTransform::new(x_min, y_max, x_res, y_res),
            ..Default::default()
        }
    }

    fn generate_points(n: usize, seed: u64) -> Vec<SamplePoint> {
        let mut points = Vec::with_capacity(n);
        let mut rng = seed;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            let value = 0.5 * x + 0.3 * y
                + 10.0 * ((x / 20.0).sin() + (y / 20.0).sin());
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (rng >> 33) as f64 / (1u64 << 31) as f64 * 2.0 - 1.0;
            points.push(SamplePoint::new(x, y, value + noise));
        }
        points
    }

    fn fit_for_points(points: &[SamplePoint]) -> FittedVariogram {
        let emp = empirical_variogram(points, VariogramParams::default()).unwrap();
        fit_best_variogram(&emp).unwrap()
    }

    #[test]
    fn test_ok_basic() {
        let points = generate_points(50, 42);
        let variogram = fit_for_points(&points);
        let params = make_params(20, 20, (0.0, 0.0, 100.0, 100.0));

        let result = ordinary_kriging(&points, &variogram, params).unwrap();

        // No NaN in interior
        let mut nan_count = 0;
        for row in 2..18 {
            for col in 2..18 {
                if result.estimate.get(row, col).unwrap().is_nan() {
                    nan_count += 1;
                }
            }
        }
        assert!(nan_count == 0, "Interior should have no NaN, got {}", nan_count);
    }

    #[test]
    fn test_ok_near_sample_point() {
        let points = vec![
            SamplePoint::new(10.0, 10.0, 100.0),
            SamplePoint::new(90.0, 10.0, 200.0),
            SamplePoint::new(10.0, 90.0, 300.0),
            SamplePoint::new(90.0, 90.0, 400.0),
            SamplePoint::new(50.0, 50.0, 250.0),
        ];
        // Use manual variogram (too few points for reliable fitting)
        let variogram = FittedVariogram {
            model: super::super::variogram::VariogramModel::Spherical,
            nugget: 0.0,
            sill: 5000.0,
            range: 80.0,
            partial_sill: 5000.0,
            rss: 0.0,
        };

        let params = make_params(10, 10, (0.0, 0.0, 100.0, 100.0));
        let result = ordinary_kriging(&points, &variogram, params).unwrap();

        let center = result.estimate.get(5, 5).unwrap();
        assert!(
            !center.is_nan() && center > 50.0 && center < 450.0,
            "Center should be reasonable, got {:.2}",
            center
        );
    }

    #[test]
    fn test_ok_with_variance() {
        let points = generate_points(30, 99);
        let variogram = fit_for_points(&points);
        let params = OrdinaryKrigingParams {
            compute_variance: true,
            ..make_params(10, 10, (0.0, 0.0, 100.0, 100.0))
        };

        let result = ordinary_kriging(&points, &variogram, params).unwrap();
        assert!(result.variance.is_some(), "Variance raster should be present");

        let var = result.variance.unwrap();
        // Variance should be non-negative
        for row in 0..10 {
            for col in 0..10 {
                let v = var.get(row, col).unwrap();
                if !v.is_nan() {
                    assert!(v >= 0.0, "Variance should be >= 0, got {:.4} at ({},{})", v, row, col);
                }
            }
        }
    }

    #[test]
    fn test_ok_with_search_radius() {
        let points = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(1.0, 0.0, 20.0),
            SamplePoint::new(0.0, 1.0, 30.0),
            SamplePoint::new(1.0, 1.0, 40.0),
        ];
        // Use manual variogram (too few points for reliable fitting)
        let variogram = FittedVariogram {
            model: super::super::variogram::VariogramModel::Spherical,
            nugget: 0.0,
            sill: 100.0,
            range: 1.0,
            partial_sill: 100.0,
            rss: 0.0,
        };

        let params = OrdinaryKrigingParams {
            max_radius: Some(0.5),
            ..make_params(10, 10, (0.0, 0.0, 10.0, 10.0))
        };

        let result = ordinary_kriging(&points, &variogram, params).unwrap();

        // Far cells should be NaN
        let far = result.estimate.get(9, 9).unwrap();
        assert!(far.is_nan(), "Far cell should be NaN with small radius, got {:.2}", far);
    }

    #[test]
    fn test_ok_too_few_points() {
        let points = vec![SamplePoint::new(0.0, 0.0, 10.0)];
        let variogram = FittedVariogram {
            model: super::super::variogram::VariogramModel::Spherical,
            nugget: 0.0,
            sill: 10.0,
            range: 50.0,
            partial_sill: 10.0,
            rss: 0.0,
        };
        let params = make_params(5, 5, (0.0, 0.0, 10.0, 10.0));
        assert!(ordinary_kriging(&points, &variogram, params).is_err());
    }

    #[test]
    fn test_ok_constant_field() {
        // Constant field: kriging should return the constant
        let points = vec![
            SamplePoint::new(0.0, 0.0, 42.0),
            SamplePoint::new(100.0, 0.0, 42.0),
            SamplePoint::new(0.0, 100.0, 42.0),
            SamplePoint::new(100.0, 100.0, 42.0),
            SamplePoint::new(50.0, 50.0, 42.0),
        ];

        // For constant field, empirical variogram will be ~0
        // Use a manual variogram since fitting on constant data may fail
        let variogram = FittedVariogram {
            model: super::super::variogram::VariogramModel::Spherical,
            nugget: 0.001,
            sill: 0.002,
            range: 50.0,
            partial_sill: 0.001,
            rss: 0.0,
        };

        let params = make_params(5, 5, (0.0, 0.0, 100.0, 100.0));
        let result = ordinary_kriging(&points, &variogram, params).unwrap();

        for row in 0..5 {
            for col in 0..5 {
                let v = result.estimate.get(row, col).unwrap();
                if !v.is_nan() {
                    assert!(
                        (v - 42.0).abs() < 1.0,
                        "Constant field should give ~42.0, got {:.2} at ({},{})",
                        v, row, col
                    );
                }
            }
        }
    }

    #[test]
    fn test_kriging_solve_basic() {
        // Simple 2×2 system
        let mut mat = vec![2.0, 1.0, 1.0, 3.0];
        let mut rhs = vec![5.0, 7.0];
        let x = kriging_solve(2, &mut mat, &mut rhs).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10, "x[0] = {}", x[0]);
        assert!((x[1] - 1.8).abs() < 1e-10, "x[1] = {}", x[1]);
    }
}
