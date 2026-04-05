//! Trend analysis for raster time series.
//!
//! - **Linear regression**: per-pixel OLS slope, intercept, R², p-value
//! - **Mann-Kendall**: non-parametric trend significance test
//! - **Sen's slope**: robust non-parametric slope estimator

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Result of per-pixel linear trend analysis.
pub struct LinearTrendResult {
    /// Slope (change per time step)
    pub slope: Raster<f64>,
    /// Intercept (value at t=0)
    pub intercept: Raster<f64>,
    /// R² (coefficient of determination, 0-1)
    pub r_squared: Raster<f64>,
    /// p-value (two-sided t-test for slope ≠ 0)
    pub p_value: Raster<f64>,
}

/// Result of Mann-Kendall trend test.
pub struct MannKendallResult {
    /// Kendall's tau (-1 to 1)
    pub tau: Raster<f64>,
    /// p-value (two-sided)
    pub p_value: Raster<f64>,
    /// Trend significance at alpha=0.05: 1=increasing, -1=decreasing, 0=no trend
    pub trend: Raster<f64>,
    /// Sen's slope (median of pairwise slopes)
    pub sens_slope: Raster<f64>,
}

fn validate_series(rasters: &[&Raster<f64>]) -> Result<(usize, usize)> {
    if rasters.len() < 3 {
        return Err(Error::Other("trend analysis requires at least 3 time steps".into()));
    }
    let (rows, cols) = rasters[0].shape();
    for r in rasters.iter().skip(1) {
        if r.shape() != (rows, cols) {
            return Err(Error::SizeMismatch {
                er: rows, ec: cols, ar: r.rows(), ac: r.cols(),
            });
        }
    }
    Ok((rows, cols))
}

/// Per-pixel ordinary least squares linear regression.
///
/// Fits y = slope * t + intercept for each pixel, where t = 0, 1, ..., n-1.
/// Optionally, provide custom time values (e.g., fractional years).
///
/// # Arguments
/// * `rasters` - Time-ordered raster stack (at least 3)
/// * `times` - Optional custom time values. If None, uses 0..n-1.
pub fn linear_trend(rasters: &[&Raster<f64>], times: Option<&[f64]>) -> Result<LinearTrendResult> {
    let (rows, cols) = validate_series(rasters)?;
    let n = rasters.len();

    let t_vals: Vec<f64> = match times {
        Some(t) => {
            if t.len() != n {
                return Err(Error::Other(format!(
                    "times length {} != raster count {}", t.len(), n
                )));
            }
            t.to_vec()
        }
        None => (0..n).map(|i| i as f64).collect(),
    };

    let total = rows * cols;
    let mut slope_flat = vec![f64::NAN; total];
    let mut intercept_flat = vec![f64::NAN; total];
    let mut r2_flat = vec![f64::NAN; total];
    let mut pval_flat = vec![f64::NAN; total];

    slope_flat.par_chunks_mut(cols)
        .zip(intercept_flat.par_chunks_mut(cols))
        .zip(r2_flat.par_chunks_mut(cols))
        .zip(pval_flat.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(row, (((slope_row, intercept_row), r2_row), pval_row))| {
            let mut tv = Vec::with_capacity(n);
            let mut yv = Vec::with_capacity(n);

            for col in 0..cols {
                tv.clear();
                yv.clear();
                for (i, r) in rasters.iter().enumerate() {
                    let v = unsafe { r.get_unchecked(row, col) };
                    if v.is_finite() {
                        tv.push(t_vals[i]);
                        yv.push(v);
                    }
                }

                if yv.len() < 3 {
                    continue;
                }

                let k = yv.len() as f64;
                let t_mean = tv.iter().sum::<f64>() / k;
                let y_mean = yv.iter().sum::<f64>() / k;

                let mut ss_tt = 0.0;
                let mut ss_ty = 0.0;
                let mut ss_yy = 0.0;
                for i in 0..yv.len() {
                    let dt = tv[i] - t_mean;
                    let dy = yv[i] - y_mean;
                    ss_tt += dt * dt;
                    ss_ty += dt * dy;
                    ss_yy += dy * dy;
                }

                if ss_tt < 1e-30 {
                    continue;
                }

                let b = ss_ty / ss_tt;
                let a = y_mean - b * t_mean;
                let r2 = if ss_yy > 1e-30 { (ss_ty * ss_ty) / (ss_tt * ss_yy) } else { 0.0 };

                slope_row[col] = b;
                intercept_row[col] = a;
                r2_row[col] = r2;

                // p-value from t-statistic
                let df = k - 2.0;
                if df >= 1.0 {
                    let ss_res = ss_yy - ss_ty * ss_ty / ss_tt;
                    let mse = ss_res.max(0.0) / df;
                    let se_b = (mse / ss_tt).sqrt();
                    if se_b > 1e-30 {
                        let t_stat = b / se_b;
                        pval_row[col] = t_test_pvalue(t_stat.abs(), df as u32);
                    } else if b.abs() > 1e-30 {
                        // Perfect fit: zero residuals → p-value effectively 0
                        pval_row[col] = 0.0;
                    }
                }
            }
        });

    let make_raster = |flat: Vec<f64>| {
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(rasters[0].transform().clone());
        r.set_nodata(Some(f64::NAN));
        if let Some(crs) = rasters[0].crs() {
            r.set_crs(Some(crs.clone()));
        }
        r
    };

    Ok(LinearTrendResult {
        slope: make_raster(slope_flat),
        intercept: make_raster(intercept_flat),
        r_squared: make_raster(r2_flat),
        p_value: make_raster(pval_flat),
    })
}

/// Mann-Kendall non-parametric trend test with Sen's slope.
///
/// Tests whether there is a monotonic trend in the time series.
/// More robust than linear regression for non-normal data.
///
/// # Arguments
/// * `rasters` - Time-ordered raster stack (at least 3)
pub fn mann_kendall(rasters: &[&Raster<f64>]) -> Result<MannKendallResult> {
    let (rows, cols) = validate_series(rasters)?;
    let n = rasters.len();

    let total = rows * cols;
    let mut tau_flat = vec![f64::NAN; total];
    let mut pval_flat = vec![f64::NAN; total];
    let mut trend_flat = vec![f64::NAN; total];
    let mut sens_flat = vec![f64::NAN; total];

    tau_flat.par_chunks_mut(cols)
        .zip(pval_flat.par_chunks_mut(cols))
        .zip(trend_flat.par_chunks_mut(cols))
        .zip(sens_flat.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(row, (((tau_row, pval_row), trend_row), sens_row))| {
            let mut vals = Vec::with_capacity(n);

            for col in 0..cols {
                vals.clear();
                for r in rasters {
                    let v = unsafe { r.get_unchecked(row, col) };
                    vals.push(v);
                }

                // Need at least 3 non-NaN values for a meaningful test
                let valid_count = vals.iter().filter(|v| v.is_finite()).count();
                if valid_count < 3 {
                    continue;
                }

                // Mann-Kendall S statistic
                let mut s: i64 = 0;
                let mut slopes = Vec::new();
                let k = vals.len();
                for i in 0..k {
                    if !vals[i].is_finite() { continue; }
                    for j in (i + 1)..k {
                        if !vals[j].is_finite() { continue; }
                        let diff = vals[j] - vals[i];
                        if diff > 0.0 { s += 1; }
                        else if diff < 0.0 { s -= 1; }
                        slopes.push(diff / (j - i) as f64);
                    }
                }

                // Sen's slope = median of pairwise slopes
                if !slopes.is_empty() {
                    slopes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = slopes.len() / 2;
                    sens_row[col] = if slopes.len() % 2 == 0 {
                        (slopes[mid - 1] + slopes[mid]) / 2.0
                    } else {
                        slopes[mid]
                    };
                }

                // Kendall's tau
                let n_pairs = valid_count * (valid_count - 1) / 2;
                if n_pairs > 0 {
                    tau_row[col] = s as f64 / n_pairs as f64;
                }

                // Variance of S (normal approximation for p-value)
                let vn = valid_count as f64;
                let var_s = vn * (vn - 1.0) * (2.0 * vn + 5.0) / 18.0;
                if var_s > 0.0 {
                    let z = if s > 0 {
                        (s as f64 - 1.0) / var_s.sqrt()
                    } else if s < 0 {
                        (s as f64 + 1.0) / var_s.sqrt()
                    } else {
                        0.0
                    };
                    let p = 2.0 * normal_cdf(-z.abs());
                    pval_row[col] = p;

                    // Significance at alpha=0.05
                    trend_row[col] = if p < 0.05 {
                        if s > 0 { 1.0 } else { -1.0 }
                    } else {
                        0.0
                    };
                }
            }
        });

    let make_raster = |flat: Vec<f64>| {
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(rasters[0].transform().clone());
        r.set_nodata(Some(f64::NAN));
        if let Some(crs) = rasters[0].crs() {
            r.set_crs(Some(crs.clone()));
        }
        r
    };

    Ok(MannKendallResult {
        tau: make_raster(tau_flat),
        p_value: make_raster(pval_flat),
        trend: make_raster(trend_flat),
        sens_slope: make_raster(sens_flat),
    })
}

/// Per-pixel Sen's slope estimator (standalone, without full Mann-Kendall test).
///
/// Returns the median of all pairwise slopes: (y_j - y_i) / (j - i) for j > i.
pub fn sens_slope(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let (rows, cols) = validate_series(rasters)?;
    let n = rasters.len();
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    out.as_slice_mut().unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            let mut slopes = Vec::new();
            for col in 0..cols {
                slopes.clear();
                for i in 0..n {
                    let vi = unsafe { rasters[i].get_unchecked(row, col) };
                    if !vi.is_finite() { continue; }
                    for j in (i + 1)..n {
                        let vj = unsafe { rasters[j].get_unchecked(row, col) };
                        if !vj.is_finite() { continue; }
                        slopes.push((vj - vi) / (j - i) as f64);
                    }
                }
                if !slopes.is_empty() {
                    slopes.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = slopes.len() / 2;
                    out_row[col] = if slopes.len() % 2 == 0 {
                        (slopes[mid - 1] + slopes[mid]) / 2.0
                    } else {
                        slopes[mid]
                    };
                }
            }
        });

    let mut result = Raster::from_array(out);
    result.set_transform(rasters[0].transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

// ─── Statistical helpers ───────────────────────────────────────────────────

/// Approximate p-value for two-sided t-test using the Beta incomplete function approximation.
fn t_test_pvalue(t_abs: f64, df: u32) -> f64 {
    // Use normal approximation for large df
    if df > 100 {
        return 2.0 * normal_cdf(-t_abs);
    }
    // For smaller df, use a simple approximation:
    // p ≈ 2 * (1 - Φ(t * √(df/(df+t²)) * correction))
    let x = df as f64 / (df as f64 + t_abs * t_abs);
    let a = df as f64 / 2.0;
    let b = 0.5;
    let p = regularized_incomplete_beta(x, a, b);
    p.min(1.0).max(0.0)
}

/// Standard normal CDF approximation (Abramowitz & Stegun 26.2.17).
fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() { return f64::NAN; }
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/√(2π)
    let p = d * (-x * x / 2.0).exp()
        * (t * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937
                    + t * (-1.821255978
                        + t * 1.330274429)))));
    if x >= 0.0 { 1.0 - p } else { p }
}

/// Regularized incomplete beta function I_x(a, b) via continued fraction.
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }

    // Use continued fraction (Lentz's method)
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let front = (a * x.ln() + b * (1.0 - x).ln() - ln_beta).exp() / a;

    // Continued fraction for I_x(a, b)
    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 { d = 1e-30; }
    d = 1.0 / d;
    let mut f = d;

    for m in 1..=200u32 {
        let mf = m as f64;

        // Even step
        let num_even = mf * (b - mf) * x / ((a + 2.0 * mf - 1.0) * (a + 2.0 * mf));
        d = 1.0 + num_even * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_even / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        f *= c * d;

        // Odd step
        let num_odd = -(a + mf) * (a + b + mf) * x / ((a + 2.0 * mf) * (a + 2.0 * mf + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < 1e-30 { d = 1e-30; }
        c = 1.0 + num_odd / c;
        if c.abs() < 1e-30 { c = 1e-30; }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;

        if (delta - 1.0).abs() < 1e-10 {
            break;
        }
    }

    front * f
}

/// Lanczos approximation of ln(Γ(x)).
fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let y = x;
    let mut tmp = x + 5.5;
    tmp -= (x + 0.5) * tmp.ln();
    let mut ser = 1.000000000190015;
    for (j, c) in coeffs.iter().enumerate() {
        ser += c / (y + 1.0 + j as f64);
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_raster(data: Vec<f64>) -> Raster<f64> {
        let arr = Array2::from_shape_vec((1, 1), data).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_linear_trend_perfect() {
        // y = 2*t + 10 → slope=2, intercept=10, R²=1
        let rasters: Vec<Raster<f64>> = (0..5).map(|t| make_raster(vec![10.0 + 2.0 * t as f64])).collect();
        let refs: Vec<&Raster<f64>> = rasters.iter().collect();

        let result = linear_trend(&refs, None).unwrap();

        assert!((result.slope.data()[[0, 0]] - 2.0).abs() < 1e-8, "slope");
        assert!((result.intercept.data()[[0, 0]] - 10.0).abs() < 1e-8, "intercept");
        assert!((result.r_squared.data()[[0, 0]] - 1.0).abs() < 1e-8, "R²");
        assert!(result.p_value.data()[[0, 0]] < 0.001, "p-value should be very small");
    }

    #[test]
    fn test_linear_trend_no_trend() {
        // Constant values → slope=0
        let rasters: Vec<Raster<f64>> = (0..5).map(|_| make_raster(vec![42.0])).collect();
        let refs: Vec<&Raster<f64>> = rasters.iter().collect();

        let result = linear_trend(&refs, None).unwrap();
        assert!(result.slope.data()[[0, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_mann_kendall_increasing() {
        let rasters: Vec<Raster<f64>> = (0..10).map(|t| make_raster(vec![t as f64 * 3.0 + 5.0])).collect();
        let refs: Vec<&Raster<f64>> = rasters.iter().collect();

        let result = mann_kendall(&refs).unwrap();

        assert!((result.tau.data()[[0, 0]] - 1.0).abs() < 1e-10, "perfect increasing → tau=1");
        assert!(result.p_value.data()[[0, 0]] < 0.05, "significant trend");
        assert!((result.trend.data()[[0, 0]] - 1.0).abs() < 1e-10, "increasing");
        assert!((result.sens_slope.data()[[0, 0]] - 3.0).abs() < 1e-10, "Sen's slope=3");
    }

    #[test]
    fn test_mann_kendall_decreasing() {
        let rasters: Vec<Raster<f64>> = (0..10).map(|t| make_raster(vec![100.0 - t as f64 * 5.0])).collect();
        let refs: Vec<&Raster<f64>> = rasters.iter().collect();

        let result = mann_kendall(&refs).unwrap();

        assert!((result.tau.data()[[0, 0]] - (-1.0)).abs() < 1e-10, "perfect decreasing → tau=-1");
        assert!((result.trend.data()[[0, 0]] - (-1.0)).abs() < 1e-10, "decreasing");
    }

    #[test]
    fn test_sens_slope_with_nan() {
        let r1 = make_raster(vec![10.0]);
        let r2 = make_raster(vec![f64::NAN]);
        let r3 = make_raster(vec![20.0]);
        let r4 = make_raster(vec![30.0]);

        let result = sens_slope(&[&r1, &r2, &r3, &r4]).unwrap();
        // Pairwise slopes: (20-10)/2=5, (30-10)/3=6.67, (30-20)/1=10
        // Median of [5, 6.67, 10] = 6.67
        let s = result.data()[[0, 0]];
        assert!(s.is_finite());
        assert!((s - 20.0 / 3.0).abs() < 1e-8);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }
}
