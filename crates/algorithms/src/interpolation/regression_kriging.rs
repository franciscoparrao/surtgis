//! Regression Kriging (RK) interpolation
//!
//! Hybrid method that decomposes the spatial field into:
//! ```text
//! Z(x) = m(x) + ε(x)
//! ```
//! where m(x) is a deterministic trend estimated by OLS regression
//! and ε(x) is a spatially correlated residual interpolated by Ordinary Kriging.
//!
//! Consistently outperforms IDW by 15–30% RMSE in mountainous terrain.
//!
//! Reference:
//! Hengl, T. et al. (2007). About regression-kriging. Computers & Geosciences.
//! Zhu, Q. & Lin, H. (2010). Comparing ordinary kriging and regression
//! kriging for soil properties. Pedosphere.
//! Florinsky, I.V. (2025). Digital Terrain Analysis, §3.3.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::kriging::{ordinary_kriging, OrdinaryKrigingParams};
use super::variogram::{
    empirical_variogram, fit_best_variogram, FittedVariogram, VariogramParams,
};
use super::SamplePoint;

/// Parameters for Regression Kriging interpolation
#[derive(Debug, Clone)]
pub struct RegressionKrigingParams {
    /// Output raster rows
    pub rows: usize,
    /// Output raster columns
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
    /// Maximum number of nearest points for OK on residuals (default 16)
    pub max_points: usize,
    /// Maximum search radius for OK on residuals
    pub max_radius: Option<f64>,
    /// Whether to produce a kriging variance raster
    pub compute_variance: bool,
    /// Variogram parameters for residual fitting
    pub variogram_params: VariogramParams,
}

impl Default for RegressionKrigingParams {
    fn default() -> Self {
        Self {
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
            max_points: 16,
            max_radius: None,
            compute_variance: false,
            variogram_params: VariogramParams::default(),
        }
    }
}

/// Result of Regression Kriging interpolation
pub struct RegressionKrigingResult {
    /// Final interpolated values (trend + kriged residuals)
    pub estimate: Raster<f64>,
    /// Kriging variance of the residual component. `None` if not requested.
    pub variance: Option<Raster<f64>>,
    /// OLS regression coefficients [intercept, β_x, β_y]
    pub coefficients: Vec<f64>,
}

/// Perform Regression Kriging interpolation.
///
/// Steps:
/// 1. Fit OLS linear trend: m(x,y) = β₀ + β₁·x + β₂·y
/// 2. Compute residuals: ε(xᵢ) = z(xᵢ) - m(xᵢ)
/// 3. Fit variogram on residuals
/// 4. Interpolate residuals using Ordinary Kriging
/// 5. Final: Z̃(x) = m(x) + ε̃(x)
///
/// # Arguments
/// * `points` — Sample points with (x, y, value)
/// * `params` — Output grid specification and parameters
///
/// # Returns
/// [`RegressionKrigingResult`] with interpolated raster, optional variance,
/// and the OLS coefficients.
///
/// # Errors
/// - If fewer than 4 points are provided (need 3 for OLS + 1)
/// - If the OLS or kriging system is singular
pub fn regression_kriging(
    points: &[SamplePoint],
    params: RegressionKrigingParams,
) -> Result<RegressionKrigingResult> {
    let n = points.len();
    if n < 4 {
        return Err(Error::Algorithm(
            "Regression Kriging requires at least 4 sample points".into(),
        ));
    }

    // Step 1: OLS linear regression  m(x,y) = β₀ + β₁·x + β₂·y
    // Normal equations: (XᵀX)β = Xᵀz
    let coefficients = ols_fit(points)?;

    // Step 2: Compute residuals
    let residuals: Vec<SamplePoint> = points
        .iter()
        .map(|pt| {
            let trend = coefficients[0] + coefficients[1] * pt.x + coefficients[2] * pt.y;
            SamplePoint::new(pt.x, pt.y, pt.value - trend)
        })
        .collect();

    // Step 3: Fit variogram on residuals
    let emp = empirical_variogram(&residuals, params.variogram_params.clone())?;
    let residual_variogram = fit_best_variogram(&emp)?;

    // Step 4: OK on residuals
    let ok_params = OrdinaryKrigingParams {
        rows: params.rows,
        cols: params.cols,
        transform: params.transform,
        max_points: params.max_points,
        max_radius: params.max_radius,
        compute_variance: params.compute_variance,
    };
    let kriged_residuals = ordinary_kriging(&residuals, &residual_variogram, ok_params)?;

    // Step 5: Combine trend + kriged residuals
    let rows = params.rows;
    let cols = params.cols;
    let transform = params.transform;

    let final_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let (x, y) = transform.pixel_to_geo(col, row);
                let trend = coefficients[0] + coefficients[1] * x + coefficients[2] * y;
                let residual = kriged_residuals.estimate.get(row, col).unwrap_or(f64::NAN);
                if !residual.is_nan() {
                    *row_data_col = trend + residual;
                }
            }
            row_data
        })
        .collect();

    let mut estimate = Raster::new(rows, cols);
    estimate.set_transform(transform);
    estimate.set_nodata(Some(f64::NAN));
    *estimate.data_mut() = Array2::from_shape_vec((rows, cols), final_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(RegressionKrigingResult {
        estimate,
        variance: kriged_residuals.variance,
        coefficients,
    })
}

/// Perform Regression Kriging with a pre-fitted variogram for the residuals.
///
/// Use this when you want to control the variogram fitting externally.
pub fn regression_kriging_with_variogram(
    points: &[SamplePoint],
    residual_variogram: &FittedVariogram,
    params: RegressionKrigingParams,
) -> Result<RegressionKrigingResult> {
    let n = points.len();
    if n < 4 {
        return Err(Error::Algorithm(
            "Regression Kriging requires at least 4 sample points".into(),
        ));
    }

    let coefficients = ols_fit(points)?;

    let residuals: Vec<SamplePoint> = points
        .iter()
        .map(|pt| {
            let trend = coefficients[0] + coefficients[1] * pt.x + coefficients[2] * pt.y;
            SamplePoint::new(pt.x, pt.y, pt.value - trend)
        })
        .collect();

    let ok_params = OrdinaryKrigingParams {
        rows: params.rows,
        cols: params.cols,
        transform: params.transform,
        max_points: params.max_points,
        max_radius: params.max_radius,
        compute_variance: params.compute_variance,
    };
    let kriged_residuals = ordinary_kriging(&residuals, residual_variogram, ok_params)?;

    let rows = params.rows;
    let cols = params.cols;
    let transform = params.transform;

    let final_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let (x, y) = transform.pixel_to_geo(col, row);
                let trend = coefficients[0] + coefficients[1] * x + coefficients[2] * y;
                let residual = kriged_residuals.estimate.get(row, col).unwrap_or(f64::NAN);
                if !residual.is_nan() {
                    *row_data_col = trend + residual;
                }
            }
            row_data
        })
        .collect();

    let mut estimate = Raster::new(rows, cols);
    estimate.set_transform(transform);
    estimate.set_nodata(Some(f64::NAN));
    *estimate.data_mut() = Array2::from_shape_vec((rows, cols), final_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(RegressionKrigingResult {
        estimate,
        variance: kriged_residuals.variance,
        coefficients,
    })
}

/// Fit OLS linear regression: z = β₀ + β₁·x + β₂·y
/// Solves normal equations (XᵀX)β = Xᵀz using 3×3 system.
fn ols_fit(points: &[SamplePoint]) -> Result<Vec<f64>> {
    let n = points.len() as f64;

    // Accumulate sums for normal equations
    let mut sx = 0.0_f64;
    let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut syy = 0.0_f64;
    let mut sz = 0.0_f64;
    let mut sxz = 0.0_f64;
    let mut syz = 0.0_f64;

    for pt in points {
        sx += pt.x;
        sy += pt.y;
        sxx += pt.x * pt.x;
        sxy += pt.x * pt.y;
        syy += pt.y * pt.y;
        sz += pt.value;
        sxz += pt.x * pt.value;
        syz += pt.y * pt.value;
    }

    // Normal equations: (XᵀX)β = Xᵀz
    // [n    sx   sy ] [β₀]   [sz ]
    // [sx   sxx  sxy] [β₁] = [sxz]
    // [sy   sxy  syy] [β₂]   [syz]
    let mut mat = [n, sx, sy,
        sx, sxx, sxy,
        sy, sxy, syy];
    let mut rhs = [sz, sxz, syz];

    // Solve 3×3 system
    for col in 0..3 {
        let mut max_val = mat[col * 3 + col].abs();
        let mut max_row = col;
        for row in (col + 1)..3 {
            let val = mat[row * 3 + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(Error::Algorithm(
                "OLS: singular system (collinear points?)".into(),
            ));
        }

        if max_row != col {
            for j in 0..3 {
                mat.swap(col * 3 + j, max_row * 3 + j);
            }
            rhs.swap(col, max_row);
        }

        let pivot = mat[col * 3 + col];
        for row in (col + 1)..3 {
            let factor = mat[row * 3 + col] / pivot;
            mat[row * 3 + col] = 0.0;
            for j in (col + 1)..3 {
                mat[row * 3 + j] -= factor * mat[col * 3 + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    let mut beta = vec![0.0_f64; 3];
    for col in (0..3).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..3 {
            sum -= mat[col * 3 + j] * beta[j];
        }
        beta[col] = sum / mat[col * 3 + col];
    }

    Ok(beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_trended_points(n: usize, seed: u64) -> Vec<SamplePoint> {
        let mut points = Vec::with_capacity(n);
        let mut rng = seed;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let y = (rng >> 33) as f64 / (1u64 << 31) as f64 * 100.0;
            // Strong trend + spatially correlated residual
            let value = 3.0 * x + 2.0 * y + 50.0
                + 10.0 * ((x / 20.0).sin() + (y / 20.0).sin());
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let noise = (rng >> 33) as f64 / (1u64 << 31) as f64 * 4.0 - 2.0;
            points.push(SamplePoint::new(x, y, value + noise));
        }
        points
    }

    fn make_params(rows: usize, cols: usize, extent: (f64, f64, f64, f64)) -> RegressionKrigingParams {
        let (x_min, y_min, x_max, y_max) = extent;
        let x_res = (x_max - x_min) / cols as f64;
        let y_res = -(y_max - y_min) / rows as f64;
        RegressionKrigingParams {
            rows,
            cols,
            transform: GeoTransform::new(x_min, y_max, x_res, y_res),
            ..Default::default()
        }
    }

    #[test]
    fn test_ols_fit_linear() {
        // Perfect linear: z = 2x + 3y + 1
        let points = vec![
            SamplePoint::new(0.0, 0.0, 1.0),
            SamplePoint::new(10.0, 0.0, 21.0),
            SamplePoint::new(0.0, 10.0, 31.0),
            SamplePoint::new(10.0, 10.0, 51.0),
            SamplePoint::new(5.0, 5.0, 26.0),
        ];
        let beta = ols_fit(&points).unwrap();
        assert!(
            (beta[0] - 1.0).abs() < 0.01,
            "intercept: expected 1.0, got {:.4}",
            beta[0]
        );
        assert!(
            (beta[1] - 2.0).abs() < 0.01,
            "β_x: expected 2.0, got {:.4}",
            beta[1]
        );
        assert!(
            (beta[2] - 3.0).abs() < 0.01,
            "β_y: expected 3.0, got {:.4}",
            beta[2]
        );
    }

    #[test]
    fn test_rk_basic() {
        let points = generate_trended_points(80, 42);
        let params = make_params(10, 10, (0.0, 0.0, 100.0, 100.0));

        let result = regression_kriging(&points, params).unwrap();

        // Should have reasonable trend coefficients (~3 for x, ~2 for y)
        assert!(
            result.coefficients[1] > 1.0 && result.coefficients[1] < 5.0,
            "β_x should be ~3.0, got {:.2}",
            result.coefficients[1]
        );

        // No NaN in interior
        let mut nan_count = 0;
        for row in 1..9 {
            for col in 1..9 {
                if result.estimate.get(row, col).unwrap().is_nan() {
                    nan_count += 1;
                }
            }
        }
        assert!(nan_count == 0, "Interior should have no NaN, got {}", nan_count);
    }

    #[test]
    fn test_rk_too_few_points() {
        let points = vec![
            SamplePoint::new(0.0, 0.0, 1.0),
            SamplePoint::new(1.0, 0.0, 2.0),
            SamplePoint::new(0.0, 1.0, 3.0),
        ];
        let params = make_params(5, 5, (0.0, 0.0, 10.0, 10.0));
        assert!(regression_kriging(&points, params).is_err());
    }

    #[test]
    fn test_rk_with_variance() {
        let points = generate_trended_points(60, 99);
        let params = RegressionKrigingParams {
            compute_variance: true,
            ..make_params(8, 8, (0.0, 0.0, 100.0, 100.0))
        };

        let result = regression_kriging(&points, params).unwrap();
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
    fn test_rk_with_prefit_variogram() {
        let points = generate_trended_points(50, 77);
        let variogram = super::super::variogram::FittedVariogram {
            model: super::super::variogram::VariogramModel::Exponential,
            nugget: 1.0,
            sill: 30.0,
            range: 25.0,
            partial_sill: 29.0,
            rss: 0.0,
        };
        let params = make_params(8, 8, (0.0, 0.0, 100.0, 100.0));

        let result = regression_kriging_with_variogram(&points, &variogram, params).unwrap();
        let center = result.estimate.get(4, 4).unwrap();
        assert!(!center.is_nan(), "Center should not be NaN");
    }
}
