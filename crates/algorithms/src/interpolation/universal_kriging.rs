//! Universal Kriging (UK) interpolation
//!
//! Extends Ordinary Kriging by incorporating a polynomial drift model
//! (spatial trend) into the kriging system. The trend is modeled as a
//! linear combination of spatial coordinate functions (monomials).
//!
//! The UK system for n sample points with p drift functions:
//! ```text
//! [γ(xᵢ,xⱼ) | fₖ(xᵢ)] [wᵢ]   [γ(xᵢ,x₀)]
//! [-----------+--------] [  ] = [----------]
//! [fₖ(xᵢ)ᵀ  |    0   ] [μₖ]   [fₖ(x₀)   ]
//! ```
//! where fₖ are the drift functions: {1, x, y} for linear trend,
//! {1, x, y, x², xy, y²} for quadratic trend.
//!
//! Reference:
//! Matheron, G. (1969). Le Krigeage Universel. Cahiers du CMMM.
//! Cressie, N. (1993). Statistics for Spatial Data. Wiley.
//! Florinsky, I.V. (2025). Digital Terrain Analysis, §3.3.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::variogram::FittedVariogram;
use super::SamplePoint;

/// Polynomial drift order for Universal Kriging
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriftOrder {
    /// Linear trend: f = {1, x, y} → 3 drift functions
    /// Use when there is a clear directional trend in the data.
    Linear,
    /// Quadratic trend: f = {1, x, y, x², xy, y²} → 6 drift functions
    /// Use for more complex spatial trends.
    Quadratic,
}

/// Parameters for Universal Kriging interpolation
#[derive(Debug, Clone)]
pub struct UniversalKrigingParams {
    /// Output raster rows
    pub rows: usize,
    /// Output raster columns
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
    /// Maximum number of nearest points to use per estimation (default 16)
    pub max_points: usize,
    /// Maximum search radius. Points beyond this are ignored.
    pub max_radius: Option<f64>,
    /// Polynomial drift order (default: Linear)
    pub drift_order: DriftOrder,
    /// Whether to produce a kriging variance raster
    pub compute_variance: bool,
}

impl Default for UniversalKrigingParams {
    fn default() -> Self {
        Self {
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
            max_points: 16,
            max_radius: None,
            drift_order: DriftOrder::Linear,
            compute_variance: false,
        }
    }
}

/// Result of Universal Kriging interpolation
pub struct UniversalKrigingResult {
    /// Interpolated values
    pub estimate: Raster<f64>,
    /// Kriging variance (estimation uncertainty). `None` if not requested.
    pub variance: Option<Raster<f64>>,
}

/// Compute drift function values at a point.
/// Returns a vector of drift basis function values.
#[inline]
fn drift_values(x: f64, y: f64, order: DriftOrder) -> Vec<f64> {
    match order {
        DriftOrder::Linear => vec![1.0, x, y],
        DriftOrder::Quadratic => vec![1.0, x, y, x * x, x * y, y * y],
    }
}

/// Number of drift functions for a given order
#[inline]
fn n_drift(order: DriftOrder) -> usize {
    match order {
        DriftOrder::Linear => 3,
        DriftOrder::Quadratic => 6,
    }
}

/// Perform Universal Kriging interpolation from scattered points to a raster grid.
///
/// # Arguments
/// * `points` — Sample points with (x, y, value)
/// * `variogram` — Fitted variogram model (should be fitted on detrended residuals)
/// * `params` — Output grid specification, drift order, and search parameters
///
/// # Returns
/// [`UniversalKrigingResult`] with interpolated raster and optionally kriging variance.
///
/// # Errors
/// - If fewer than (n_drift + 1) points are provided
pub fn universal_kriging(
    points: &[SamplePoint],
    variogram: &FittedVariogram,
    params: UniversalKrigingParams,
) -> Result<UniversalKrigingResult> {
    let n = points.len();
    let p = n_drift(params.drift_order);

    if n < p + 1 {
        return Err(Error::Algorithm(format!(
            "Universal Kriging with {:?} drift requires at least {} points, got {}",
            params.drift_order,
            p + 1,
            n
        )));
    }

    let rows = params.rows;
    let cols = params.cols;
    let transform = params.transform;
    let max_pts = params.max_points.min(n);
    let max_radius_sq = params.max_radius.map(|r| r * r);
    let compute_var = params.compute_variance;
    let drift_order = params.drift_order;

    let output: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN); cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
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

                if let Some(max_sq) = max_radius_sq {
                    let max_r = max_sq.sqrt();
                    dists.retain(|(_, d)| *d <= max_r);
                }

                if dists.is_empty() {
                    continue;
                }

                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                let k = dists.len().min(max_pts);

                // Need at least p+1 neighbors for UK
                if k < p + 1 {
                    continue;
                }

                let neighbors = &dists[..k];

                // Check snap distance
                if neighbors[0].1 < 1e-12 {
                    let idx = neighbors[0].0;
                    *row_data_col = (points[idx].value, 0.0);
                    continue;
                }

                // Build UK system: (k + p) × (k + p)
                let m = k + p;
                let mut mat = vec![0.0_f64; m * m];
                let mut rhs = vec![0.0_f64; m];

                // Upper-left: variogram matrix (k × k)
                for i in 0..k {
                    let pi = &points[neighbors[i].0];
                    for j in 0..k {
                        if i == j {
                            mat[i * m + j] = 0.0;
                        } else {
                            let pj = &points[neighbors[j].0];
                            let dx = pi.x - pj.x;
                            let dy = pi.y - pj.y;
                            let h = (dx * dx + dy * dy).sqrt();
                            mat[i * m + j] = variogram.evaluate(h);
                        }
                    }
                }

                // Upper-right and lower-left: drift functions
                for i in 0..k {
                    let pt = &points[neighbors[i].0];
                    let fvals = drift_values(pt.x, pt.y, drift_order);
                    for (l, fv) in fvals.iter().enumerate() {
                        mat[i * m + k + l] = *fv;       // upper-right
                        mat[(k + l) * m + i] = *fv;     // lower-left
                    }
                }
                // Lower-right (p × p) is already zero

                // RHS: variogram to target + drift at target
                for i in 0..k {
                    rhs[i] = variogram.evaluate(neighbors[i].1);
                }
                let f0 = drift_values(x0, y0, drift_order);
                for (l, fv) in f0.iter().enumerate() {
                    rhs[k + l] = *fv;
                }

                match uk_solve(m, &mut mat, &mut rhs) {
                    Ok(solution) => {
                        let mut estimate = 0.0;
                        for i in 0..k {
                            estimate += solution[i] * points[neighbors[i].0].value;
                        }

                        let variance = if compute_var {
                            let mut var = 0.0;
                            for i in 0..k {
                                var += solution[i] * variogram.evaluate(neighbors[i].1);
                            }
                            for l in 0..p {
                                var += solution[k + l] * f0[l];
                            }
                            var.max(0.0)
                        } else {
                            0.0
                        };

                        *row_data_col = (estimate, variance);
                    }
                    Err(_) => {
                        // Fallback: IDW
                        let mut sum_w = 0.0;
                        let mut sum_wz = 0.0;
                        for (idx, dist) in neighbors {
                            let w = 1.0 / (dist * dist);
                            sum_w += w;
                            sum_wz += w * points[*idx].value;
                        }
                        if sum_w > 0.0 {
                            *row_data_col = (sum_wz / sum_w, f64::NAN);
                        }
                    }
                }
            }

            row_data
        })
        .collect();

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

    Ok(UniversalKrigingResult { estimate, variance })
}

/// Gaussian elimination with partial pivoting (same as OK solver)
fn uk_solve(n: usize, mat: &mut [f64], rhs: &mut [f64]) -> Result<Vec<f64>> {
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
            return Err(Error::Algorithm("UK: singular matrix".into()));
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
        empirical_variogram, fit_best_variogram, VariogramModel, VariogramParams,
    };

    fn make_params(rows: usize, cols: usize, extent: (f64, f64, f64, f64)) -> UniversalKrigingParams {
        let (x_min, y_min, x_max, y_max) = extent;
        let x_res = (x_max - x_min) / cols as f64;
        let y_res = -(y_max - y_min) / rows as f64;
        UniversalKrigingParams {
            rows,
            cols,
            transform: GeoTransform::new(x_min, y_max, x_res, y_res),
            ..Default::default()
        }
    }

    fn generate_trended_points(n: usize, seed: u64) -> Vec<SamplePoint> {
        let mut points = Vec::with_capacity(n);
        let mut rng = seed;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            // Strong linear trend + local variation
            let value = 2.0 * x + 1.5 * y + 10.0 * ((x / 15.0).sin() + (y / 15.0).sin());
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (rng >> 33) as f64 / (1u64 << 31) as f64 * 4.0 - 2.0;
            points.push(SamplePoint::new(x, y, value + noise));
        }
        points
    }

    fn manual_variogram() -> FittedVariogram {
        FittedVariogram {
            model: VariogramModel::Spherical,
            nugget: 1.0,
            sill: 50.0,
            range: 40.0,
            partial_sill: 49.0,
            rss: 0.0,
        }
    }

    #[test]
    fn test_uk_basic_linear() {
        let points = generate_trended_points(60, 42);
        let emp = empirical_variogram(&points, VariogramParams::default()).unwrap();
        let variogram = fit_best_variogram(&emp).unwrap();

        let params = make_params(10, 10, (0.0, 0.0, 100.0, 100.0));
        let result = universal_kriging(&points, &variogram, params).unwrap();

        let mut nan_count = 0;
        for row in 0..10 {
            for col in 0..10 {
                if result.estimate.get(row, col).unwrap().is_nan() {
                    nan_count += 1;
                }
            }
        }
        assert!(nan_count == 0, "Should have no NaN in output, got {}", nan_count);
    }

    #[test]
    fn test_uk_quadratic_drift() {
        let points = generate_trended_points(60, 99);
        let variogram = manual_variogram();

        let params = UniversalKrigingParams {
            drift_order: DriftOrder::Quadratic,
            ..make_params(10, 10, (0.0, 0.0, 100.0, 100.0))
        };
        let result = universal_kriging(&points, &variogram, params).unwrap();

        let center = result.estimate.get(5, 5).unwrap();
        assert!(
            !center.is_nan(),
            "Center should not be NaN, got {:.2}",
            center
        );
    }

    #[test]
    fn test_uk_with_variance() {
        let points = generate_trended_points(40, 77);
        let variogram = manual_variogram();

        let params = UniversalKrigingParams {
            compute_variance: true,
            ..make_params(8, 8, (0.0, 0.0, 100.0, 100.0))
        };

        let result = universal_kriging(&points, &variogram, params).unwrap();
        assert!(result.variance.is_some());

        let var = result.variance.unwrap();
        for row in 0..8 {
            for col in 0..8 {
                let v = var.get(row, col).unwrap();
                if !v.is_nan() {
                    assert!(v >= 0.0, "Variance >= 0, got {:.4} at ({},{})", v, row, col);
                }
            }
        }
    }

    #[test]
    fn test_uk_too_few_points_linear() {
        // Linear drift needs at least 4 points (3 drift + 1)
        let points = vec![
            SamplePoint::new(0.0, 0.0, 1.0),
            SamplePoint::new(1.0, 0.0, 2.0),
            SamplePoint::new(0.0, 1.0, 3.0),
        ];
        let variogram = manual_variogram();
        let params = make_params(5, 5, (0.0, 0.0, 10.0, 10.0));
        assert!(universal_kriging(&points, &variogram, params).is_err());
    }

    #[test]
    fn test_uk_too_few_points_quadratic() {
        // Quadratic drift needs at least 7 points (6 drift + 1)
        let points: Vec<SamplePoint> = (0..6)
            .map(|i| SamplePoint::new(i as f64, 0.0, i as f64))
            .collect();
        let variogram = manual_variogram();
        let params = UniversalKrigingParams {
            drift_order: DriftOrder::Quadratic,
            ..make_params(5, 5, (0.0, 0.0, 10.0, 10.0))
        };
        assert!(universal_kriging(&points, &variogram, params).is_err());
    }

    #[test]
    fn test_drift_values_linear() {
        let dv = drift_values(3.0, 5.0, DriftOrder::Linear);
        assert_eq!(dv.len(), 3);
        assert!((dv[0] - 1.0).abs() < 1e-10);
        assert!((dv[1] - 3.0).abs() < 1e-10);
        assert!((dv[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_drift_values_quadratic() {
        let dv = drift_values(2.0, 3.0, DriftOrder::Quadratic);
        assert_eq!(dv.len(), 6);
        assert!((dv[0] - 1.0).abs() < 1e-10);  // 1
        assert!((dv[1] - 2.0).abs() < 1e-10);  // x
        assert!((dv[2] - 3.0).abs() < 1e-10);  // y
        assert!((dv[3] - 4.0).abs() < 1e-10);  // x²
        assert!((dv[4] - 6.0).abs() < 1e-10);  // xy
        assert!((dv[5] - 9.0).abs() < 1e-10);  // y²
    }
}
