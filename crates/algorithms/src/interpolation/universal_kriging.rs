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

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::SamplePoint;
use super::dedupe_points;
use super::variogram::FittedVariogram;

/// Coordinate tolerance (CRS units) used to merge near-duplicate sample
/// points before building the UK system. See A-8 in the correctness
/// audit and [`dedupe_points`].
const DEDUP_TOLERANCE: f64 = 1e-6;

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
    /// Number of output cells where the UK system was singular (after
    /// duplicate-coordinate merging and drift centering) and the
    /// estimate silently fell back to an IDW value instead. Zero for
    /// well-posed inputs. See A-8 in the correctness audit.
    pub n_fallback_cells: usize,
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
    // Merge near-duplicate sample coordinates (A-8): duplicate locations
    // make the UK matrix singular.
    let points = dedupe_points(points, DEDUP_TOLERANCE);
    let points = points.as_slice();

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

    // A-9: centre coordinates at the sample centroid before evaluating
    // drift functions. With raw projected/UTM-scale coordinates
    // (x ~ 4e5, y ~ 6.2e6) and a quadratic drift (x², xy, y² terms),
    // the condition number of the UK system can reach ~1e28. Kriging is
    // invariant to this translation — it is only a change of basis for
    // the drift functions — so centering strictly improves numerical
    // stability without changing the mathematical result.
    let cx0 = points.iter().map(|pt| pt.x).sum::<f64>() / n as f64;
    let cy0 = points.iter().map(|pt| pt.y).sum::<f64>() / n as f64;

    let rows = params.rows;
    let cols = params.cols;
    let transform = params.transform;
    let max_pts = params.max_points.min(n);
    let max_radius_sq = params.max_radius.map(|r| r * r);
    let compute_var = params.compute_variance;
    let drift_order = params.drift_order;
    let fallback_count = AtomicUsize::new(0);

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

                // Upper-right and lower-left: drift functions.
                // Evaluated in centroid-centered coordinates (A-9) to
                // keep the system well-conditioned; the estimator is
                // translation-invariant so this doesn't change results.
                for i in 0..k {
                    let pt = &points[neighbors[i].0];
                    let fvals = drift_values(pt.x - cx0, pt.y - cy0, drift_order);
                    for (l, fv) in fvals.iter().enumerate() {
                        mat[i * m + k + l] = *fv; // upper-right
                        mat[(k + l) * m + i] = *fv; // lower-left
                    }
                }
                // Lower-right (p × p) is already zero

                // RHS: variogram to target + drift at target
                for i in 0..k {
                    rhs[i] = variogram.evaluate(neighbors[i].1);
                }
                let f0 = drift_values(x0 - cx0, y0 - cy0, drift_order);
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
                        // Singular system — fall back to IDW-like
                        // behavior, but record it (n_fallback_cells).
                        fallback_count.fetch_add(1, Ordering::Relaxed);
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
    *estimate.data_mut() =
        Array2::from_shape_vec((rows, cols), est_data).map_err(|e| Error::Other(e.to_string()))?;

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

    let n_fallback_cells = fallback_count.load(Ordering::Relaxed);
    if n_fallback_cells > 0 {
        eprintln!(
            "surtgis: universal_kriging: {n_fallback_cells} of {} cells had a singular \
             UK system and fell back to IDW; see UniversalKrigingResult::n_fallback_cells",
            rows * cols
        );
    }

    Ok(UniversalKrigingResult {
        estimate,
        variance,
        n_fallback_cells,
    })
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
    use super::super::variogram::{
        VariogramModel, VariogramParams, empirical_variogram, fit_best_variogram,
    };
    use super::*;

    fn make_params(
        rows: usize,
        cols: usize,
        extent: (f64, f64, f64, f64),
    ) -> UniversalKrigingParams {
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
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            // Strong linear trend + local variation
            let value = 2.0 * x + 1.5 * y + 10.0 * ((x / 15.0).sin() + (y / 15.0).sin());
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
        assert!(
            nan_count == 0,
            "Should have no NaN in output, got {}",
            nan_count
        );
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

    /// Build the raw (k+p)×(k+p) UK matrix for `pts`, evaluating drift
    /// functions at `(pt.x - ox, pt.y - oy)`. Mirrors the matrix layout
    /// built inside `universal_kriging` (minus the RHS), so it can be
    /// used to probe conditioning under different coordinate origins.
    fn build_uk_matrix(
        pts: &[SamplePoint],
        variogram: &FittedVariogram,
        order: DriftOrder,
        ox: f64,
        oy: f64,
    ) -> (usize, Vec<f64>) {
        let k = pts.len();
        let p = n_drift(order);
        let m = k + p;
        let mut mat = vec![0.0_f64; m * m];
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    let dx = pts[i].x - pts[j].x;
                    let dy = pts[i].y - pts[j].y;
                    let h = (dx * dx + dy * dy).sqrt();
                    mat[i * m + j] = variogram.evaluate(h);
                }
            }
            let fvals = drift_values(pts[i].x - ox, pts[i].y - oy, order);
            for (l, fv) in fvals.iter().enumerate() {
                mat[i * m + k + l] = *fv;
                mat[(k + l) * m + i] = *fv;
            }
        }
        (m, mat)
    }

    /// Coarse condition-number proxy: ratio of the largest to smallest
    /// |pivot| encountered during Gaussian elimination with partial
    /// pivoting (same elimination strategy as `uk_solve`). Not the exact
    /// SVD-based condition number, but it reliably exposes the dynamic
    /// range blow-up caused by unscaled absolute coordinates in the
    /// drift columns.
    fn pivot_ratio_condition_estimate(n: usize, mat: &[f64]) -> f64 {
        let mut a = mat.to_vec();
        let mut max_pivot = 0.0_f64;
        let mut min_pivot = f64::INFINITY;
        for col in 0..n {
            let mut max_val = a[col * n + col].abs();
            let mut max_row = col;
            for row in (col + 1)..n {
                let val = a[row * n + col].abs();
                if val > max_val {
                    max_val = val;
                    max_row = row;
                }
            }
            if max_row != col {
                for j in 0..n {
                    a.swap(col * n + j, max_row * n + j);
                }
            }
            let pivot = a[col * n + col].abs();
            if pivot > 0.0 {
                max_pivot = max_pivot.max(pivot);
                min_pivot = min_pivot.min(pivot);
            }
            for row in (col + 1)..n {
                if a[col * n + col].abs() < 1e-300 {
                    continue;
                }
                let factor = a[row * n + col] / a[col * n + col];
                for j in col..n {
                    a[row * n + j] -= factor * a[col * n + j];
                }
            }
        }
        if min_pivot <= 0.0 || !min_pivot.is_finite() {
            f64::INFINITY
        } else {
            max_pivot / min_pivot
        }
    }

    #[test]
    fn test_uk_drift_centering_improves_conditioning_and_matches_prediction() {
        // Simulate real UTM-scale coordinates (x ~ 4e5, y ~ 6.2e6), the
        // regime that triggers severe ill-conditioning with a quadratic
        // drift evaluated at absolute coordinates (A-9, condition number
        // verified ~3e28 in the correctness audit).
        let local_points = generate_trended_points(30, 7);
        let ox_utm = 400_000.0;
        let oy_utm = 6_200_000.0;
        let utm_points: Vec<SamplePoint> = local_points
            .iter()
            .map(|p| SamplePoint::new(p.x + ox_utm, p.y + oy_utm, p.value))
            .collect();

        let variogram = manual_variogram();

        // Condition proxy: absolute coordinates (ox=oy=0, i.e. the old
        // un-centered behavior) vs sample-centroid-centered coordinates
        // (the new behavior).
        let (m, mat_absolute) =
            build_uk_matrix(&utm_points, &variogram, DriftOrder::Quadratic, 0.0, 0.0);
        let cx0 = utm_points.iter().map(|p| p.x).sum::<f64>() / utm_points.len() as f64;
        let cy0 = utm_points.iter().map(|p| p.y).sum::<f64>() / utm_points.len() as f64;
        let (_, mat_centered) =
            build_uk_matrix(&utm_points, &variogram, DriftOrder::Quadratic, cx0, cy0);

        let cond_absolute = pivot_ratio_condition_estimate(m, &mat_absolute);
        let cond_centered = pivot_ratio_condition_estimate(m, &mat_centered);

        println!("UK condition proxy: absolute={cond_absolute:.3e}, centered={cond_centered:.3e}");

        assert!(
            cond_centered < 1e12,
            "centered system should be well-conditioned, got {cond_centered:.3e}"
        );
        assert!(
            cond_absolute > cond_centered * 1e6,
            "absolute-coordinate system should be several orders of magnitude worse: \
             absolute={cond_absolute:.3e}, centered={cond_centered:.3e}"
        );

        // Prediction should match (within relative tolerance) whether the
        // caller supplies raw UTM-scale coordinates or locally-translated
        // ones, because `universal_kriging` centers internally (A-9) —
        // kriging is invariant to translation of the drift basis.
        let params_local = UniversalKrigingParams {
            drift_order: DriftOrder::Quadratic,
            ..make_params(10, 10, (0.0, 0.0, 100.0, 100.0))
        };
        let result_local = universal_kriging(&local_points, &variogram, params_local).unwrap();
        let local_value = result_local.estimate.get(5, 5).unwrap();

        let params_utm = UniversalKrigingParams {
            drift_order: DriftOrder::Quadratic,
            ..make_params(10, 10, (ox_utm, oy_utm, ox_utm + 100.0, oy_utm + 100.0))
        };
        let result_utm = universal_kriging(&utm_points, &variogram, params_utm).unwrap();
        let utm_value = result_utm.estimate.get(5, 5).unwrap();

        assert!(!local_value.is_nan() && !utm_value.is_nan());
        let rel_diff = (local_value - utm_value).abs() / local_value.abs().max(1.0);
        assert!(
            rel_diff < 1e-6,
            "prediction should be translation-invariant: local={local_value:.6}, \
             utm={utm_value:.6}, rel_diff={rel_diff:.3e}"
        );
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
        assert!((dv[0] - 1.0).abs() < 1e-10); // 1
        assert!((dv[1] - 2.0).abs() < 1e-10); // x
        assert!((dv[2] - 3.0).abs() < 1e-10); // y
        assert!((dv[3] - 4.0).abs() < 1e-10); // x²
        assert!((dv[4] - 6.0).abs() < 1e-10); // xy
        assert!((dv[5] - 9.0).abs() < 1e-10); // y²
    }
}
