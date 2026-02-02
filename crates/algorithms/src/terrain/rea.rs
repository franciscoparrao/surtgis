//! Representative Elementary Area (REA) Analysis
//!
//! Diagnostic tool to determine the adequate DEM resolution for morphometric
//! analysis. Computes a morphometric variable at multiple grid spacings and
//! measures how it changes relative to the finest resolution.
//!
//! The REA is the coarsest resolution at which the morphometric variable
//! remains stable (correlation plateau or RMSE stabilization).
//!
//! Reference:
//! Florinsky, I.V. (2025). Digital terrain analysis in soil science and geology,
//! Chapter 10: "Adequate resolution of DEMs".
//! Wood, J. & Fisher, P. (1993). Assessing interpolation accuracy in elevation models.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Morphometric variable to evaluate across scales
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReaVariable {
    /// Slope magnitude (default)
    Slope,
    /// Plan curvature (kh)
    PlanCurvature,
    /// Profile curvature (kv)
    ProfileCurvature,
}

impl Default for ReaVariable {
    fn default() -> Self {
        ReaVariable::Slope
    }
}

/// Parameters for REA analysis
#[derive(Debug, Clone)]
pub struct ReaParams {
    /// Scale factors to evaluate (multiples of native resolution).
    /// E.g. [1, 2, 4, 8, 16, 32] means 1×, 2×, 4×, ... coarser.
    /// Factor 1 is always included as the reference.
    pub scale_factors: Vec<usize>,
    /// Morphometric variable to compute at each scale
    pub variable: ReaVariable,
    /// Correlation threshold for REA identification (default: 0.95)
    pub correlation_threshold: f64,
}

impl Default for ReaParams {
    fn default() -> Self {
        Self {
            scale_factors: vec![1, 2, 4, 8, 16, 32],
            variable: ReaVariable::Slope,
            correlation_threshold: 0.95,
        }
    }
}

/// Result of REA analysis at one scale
#[derive(Debug, Clone)]
pub struct ReaScaleResult {
    /// Scale factor (1 = native resolution)
    pub scale_factor: usize,
    /// Effective cell size at this scale (native_cellsize × factor)
    pub cell_size: f64,
    /// Pearson correlation with native-resolution variable
    pub correlation: f64,
    /// RMSE relative to native-resolution variable
    pub rmse: f64,
    /// Mean value of the variable at this scale
    pub mean: f64,
    /// Standard deviation of the variable at this scale
    pub std_dev: f64,
}

/// Result of REA analysis
#[derive(Debug)]
pub struct ReaResult {
    /// Per-scale statistics
    pub scales: Vec<ReaScaleResult>,
    /// Identified REA cell size (coarsest where correlation >= threshold).
    /// `None` if correlation drops below threshold for all scales.
    pub rea_cell_size: Option<f64>,
    /// Index of the REA scale in `scales` vector
    pub rea_index: Option<usize>,
}

/// Aggregate (downsample) a DEM by a factor using mean pooling.
fn aggregate_dem(dem: &Raster<f64>, factor: usize) -> Raster<f64> {
    let (rows, cols) = dem.shape();
    let new_rows = rows / factor;
    let new_cols = cols / factor;
    let data = dem.data();

    let mut agg_data = Array2::from_elem((new_rows, new_cols), f64::NAN);

    for r in 0..new_rows {
        for c in 0..new_cols {
            let mut sum = 0.0;
            let mut count = 0;

            for dr in 0..factor {
                for dc in 0..factor {
                    let sr = r * factor + dr;
                    let sc = c * factor + dc;
                    if sr < rows && sc < cols {
                        let z = data[[sr, sc]];
                        if !z.is_nan() {
                            sum += z;
                            count += 1;
                        }
                    }
                }
            }

            if count > 0 {
                agg_data[[r, c]] = sum / count as f64;
            }
        }
    }

    let mut out = Raster::new(new_rows, new_cols);
    let tf = dem.transform();
    let new_cs = tf.cell_size().abs() * factor as f64;
    out.set_transform(surtgis_core::raster::GeoTransform::new(
        tf.origin_x, tf.origin_y, new_cs, -new_cs,
    ));
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() = agg_data;
    out
}

/// Compute slope from a DEM using Horn's method (returns slope in radians).
fn compute_slope(dem: &Raster<f64>) -> Array2<f64> {
    let (rows, cols) = dem.shape();
    let cs = dem.transform().cell_size().abs();
    let data = dem.data();
    let mut slope = Array2::from_elem((rows, cols), f64::NAN);

    for r in 1..rows.saturating_sub(1) {
        for c in 1..cols.saturating_sub(1) {
            let z1 = data[[r - 1, c - 1]];
            let z2 = data[[r - 1, c]];
            let z3 = data[[r - 1, c + 1]];
            let z4 = data[[r, c - 1]];
            let z6 = data[[r, c + 1]];
            let z7 = data[[r + 1, c - 1]];
            let z8 = data[[r + 1, c]];
            let z9 = data[[r + 1, c + 1]];

            if z1.is_nan() || z2.is_nan() || z3.is_nan()
                || z4.is_nan() || z6.is_nan()
                || z7.is_nan() || z8.is_nan() || z9.is_nan()
            {
                continue;
            }

            let dz_dx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * cs);
            let dz_dy = ((z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)) / (8.0 * cs);

            slope[[r, c]] = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
        }
    }

    slope
}

/// Compute curvature from a DEM using Evans-Young method.
fn compute_curvature(dem: &Raster<f64>, curvature_type: ReaVariable) -> Array2<f64> {
    let (rows, cols) = dem.shape();
    let cs = dem.transform().cell_size().abs();
    let data = dem.data();
    let mut curv = Array2::from_elem((rows, cols), f64::NAN);

    for r in 1..rows.saturating_sub(1) {
        for c in 1..cols.saturating_sub(1) {
            let z1 = data[[r - 1, c - 1]];
            let z2 = data[[r - 1, c]];
            let z3 = data[[r - 1, c + 1]];
            let z4 = data[[r, c - 1]];
            let z5 = data[[r, c]];
            let z6 = data[[r, c + 1]];
            let z7 = data[[r + 1, c - 1]];
            let z8 = data[[r + 1, c]];
            let z9 = data[[r + 1, c + 1]];

            if z1.is_nan() || z2.is_nan() || z3.is_nan()
                || z4.is_nan() || z5.is_nan() || z6.is_nan()
                || z7.is_nan() || z8.is_nan() || z9.is_nan()
            {
                continue;
            }

            // Evans-Young partial derivatives
            let p = ((z3 + z6 + z9) - (z1 + z4 + z7)) / (6.0 * cs);
            let q = ((z1 + z2 + z3) - (z7 + z8 + z9)) / (6.0 * cs);
            let r_d = ((z1 + z3 + z4 + z6 + z7 + z9) - 2.0 * (z2 + z5 + z8)) / (3.0 * cs * cs);
            let s_d = (z3 + z7 - z1 - z9) / (4.0 * cs * cs);
            let t_d = ((z1 + z2 + z3 + z7 + z8 + z9) - 2.0 * (z4 + z5 + z6)) / (3.0 * cs * cs);

            let pq = p * p + q * q;

            let val = match curvature_type {
                ReaVariable::PlanCurvature => {
                    if pq < 1e-15 {
                        0.0
                    } else {
                        -(t_d * p * p - 2.0 * s_d * p * q + r_d * q * q)
                            / (pq * (1.0 + pq).sqrt())
                    }
                }
                ReaVariable::ProfileCurvature => {
                    if pq < 1e-15 {
                        0.0
                    } else {
                        -(r_d * p * p + 2.0 * s_d * p * q + t_d * q * q)
                            / (pq * (1.0 + pq).powf(1.5))
                    }
                }
                _ => unreachable!(),
            };

            curv[[r, c]] = val;
        }
    }

    curv
}

/// Expand a coarse-resolution array back to fine resolution using nearest-neighbor.
fn expand_to_fine(coarse: &Array2<f64>, factor: usize, fine_rows: usize, fine_cols: usize) -> Array2<f64> {
    let mut fine = Array2::from_elem((fine_rows, fine_cols), f64::NAN);
    let (cr, cc) = coarse.dim();

    for r in 0..fine_rows {
        for c in 0..fine_cols {
            let sr = r / factor;
            let sc = c / factor;
            if sr < cr && sc < cc {
                fine[[r, c]] = coarse[[sr, sc]];
            }
        }
    }

    fine
}

/// Compute Pearson correlation between two arrays (ignoring NaN cells)
fn pearson_correlation(a: &Array2<f64>, b: &Array2<f64>) -> (f64, f64) {
    let (rows, cols) = a.dim();
    let mut sum_a = 0.0;
    let mut sum_b = 0.0;
    let mut sum_ab = 0.0;
    let mut sum_a2 = 0.0;
    let mut sum_b2 = 0.0;
    let mut sum_diff2 = 0.0;
    let mut n = 0_usize;

    for r in 0..rows {
        for c in 0..cols {
            let va = a[[r, c]];
            let vb = b[[r, c]];
            if !va.is_nan() && !vb.is_nan() {
                sum_a += va;
                sum_b += vb;
                sum_ab += va * vb;
                sum_a2 += va * va;
                sum_b2 += vb * vb;
                sum_diff2 += (va - vb) * (va - vb);
                n += 1;
            }
        }
    }

    if n < 2 {
        return (f64::NAN, f64::NAN);
    }

    let nf = n as f64;
    let mean_a = sum_a / nf;
    let mean_b = sum_b / nf;
    let var_a = sum_a2 / nf - mean_a * mean_a;
    let var_b = sum_b2 / nf - mean_b * mean_b;
    let cov = sum_ab / nf - mean_a * mean_b;

    let denom = (var_a * var_b).sqrt();
    let corr = if denom > 1e-15 { cov / denom } else { f64::NAN };
    let rmse = (sum_diff2 / nf).sqrt();

    (corr, rmse)
}

/// Perform REA analysis on a DEM.
///
/// For each scale factor, aggregates the DEM, computes the specified
/// morphometric variable, and correlates it with the native-resolution
/// version to quantify information loss.
///
/// # Arguments
/// * `dem` — Input DEM at native resolution
/// * `params` — REA parameters
///
/// # Returns
/// `ReaResult` with per-scale statistics and identified REA
pub fn rea_analysis(dem: &Raster<f64>, params: ReaParams) -> Result<ReaResult> {
    let (rows, cols) = dem.shape();
    if rows < 3 || cols < 3 {
        return Err(Error::Algorithm("DEM must be at least 3×3 for REA".into()));
    }

    if params.scale_factors.is_empty() {
        return Err(Error::Algorithm("scale_factors must not be empty".into()));
    }

    // Compute reference variable at native resolution
    let ref_var = match params.variable {
        ReaVariable::Slope => compute_slope(dem),
        ReaVariable::PlanCurvature | ReaVariable::ProfileCurvature => {
            compute_curvature(dem, params.variable)
        }
    };

    // Statistics for reference
    let (ref_mean, ref_std) = array_stats(&ref_var);

    let mut scales = Vec::new();
    let cell_size = dem.transform().cell_size().abs();

    for &factor in &params.scale_factors {
        if factor == 0 {
            continue;
        }

        if factor == 1 {
            // Native resolution: perfect correlation
            scales.push(ReaScaleResult {
                scale_factor: 1,
                cell_size,
                correlation: 1.0,
                rmse: 0.0,
                mean: ref_mean,
                std_dev: ref_std,
            });
            continue;
        }

        // Check if DEM is large enough for this factor
        if rows / factor < 3 || cols / factor < 3 {
            continue; // Skip scales that make the DEM too small
        }

        // Aggregate DEM
        let agg_dem = aggregate_dem(dem, factor);

        // Compute variable at coarse resolution
        let coarse_var = match params.variable {
            ReaVariable::Slope => compute_slope(&agg_dem),
            ReaVariable::PlanCurvature | ReaVariable::ProfileCurvature => {
                compute_curvature(&agg_dem, params.variable)
            }
        };

        // Expand back to fine resolution for comparison
        let expanded = expand_to_fine(&coarse_var, factor, rows, cols);

        // Compute correlation and RMSE
        let (corr, rmse) = pearson_correlation(&ref_var, &expanded);
        let (mean, std_dev) = array_stats(&coarse_var);

        scales.push(ReaScaleResult {
            scale_factor: factor,
            cell_size: cell_size * factor as f64,
            correlation: corr,
            rmse,
            mean,
            std_dev,
        });
    }

    // Identify REA: coarsest scale with correlation >= threshold
    let threshold = params.correlation_threshold;
    let mut rea_index = None;
    for (i, s) in scales.iter().enumerate().rev() {
        if !s.correlation.is_nan() && s.correlation >= threshold {
            rea_index = Some(i);
            break;
        }
    }

    let rea_cell_size = rea_index.map(|i| scales[i].cell_size);

    Ok(ReaResult {
        scales,
        rea_cell_size,
        rea_index,
    })
}

/// Compute mean and standard deviation of an array (ignoring NaN)
fn array_stats(arr: &Array2<f64>) -> (f64, f64) {
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut n = 0_usize;

    for &v in arr.iter() {
        if !v.is_nan() {
            sum += v;
            sum2 += v * v;
            n += 1;
        }
    }

    if n < 2 {
        return (f64::NAN, f64::NAN);
    }

    let mean = sum / n as f64;
    let var = (sum2 / n as f64 - mean * mean).max(0.0);
    (mean, var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::GeoTransform;

    fn tilted_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut data = Array2::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                data[[r, c]] = r as f64 * 10.0 + c as f64 * 5.0;
            }
        }
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        *dem.data_mut() = data;
        dem
    }

    fn complex_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut data = Array2::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                let x = c as f64 / cols as f64 * std::f64::consts::PI * 4.0;
                let y = r as f64 / rows as f64 * std::f64::consts::PI * 4.0;
                data[[r, c]] = x.sin() * y.cos() * 100.0 + r as f64 * 2.0;
            }
        }
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        *dem.data_mut() = data;
        dem
    }

    #[test]
    fn test_rea_basic() {
        let dem = tilted_dem(64, 64);
        let result = rea_analysis(&dem, ReaParams::default()).unwrap();

        assert!(!result.scales.is_empty());
        // First scale (factor 1) should have correlation 1.0
        assert_eq!(result.scales[0].scale_factor, 1);
        assert!((result.scales[0].correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rea_correlation_decreases() {
        let dem = complex_dem(128, 128);
        let result = rea_analysis(&dem, ReaParams {
            scale_factors: vec![1, 2, 4, 8, 16],
            ..Default::default()
        }).unwrap();

        // Correlation should generally decrease with coarser resolution
        let corrs: Vec<f64> = result.scales.iter()
            .filter(|s| !s.correlation.is_nan())
            .map(|s| s.correlation)
            .collect();

        assert!(corrs.len() >= 3, "Should have at least 3 valid scales");
        assert!(corrs[0] > corrs[corrs.len() - 1],
            "Correlation should decrease: first={:.3}, last={:.3}",
            corrs[0], corrs[corrs.len() - 1]);
    }

    #[test]
    fn test_rea_identifies_rea_cell_size() {
        let dem = complex_dem(128, 128);
        let result = rea_analysis(&dem, ReaParams {
            scale_factors: vec![1, 2, 4, 8, 16, 32],
            correlation_threshold: 0.5,
            ..Default::default()
        }).unwrap();

        // With a low threshold, some scale should qualify
        assert!(result.rea_cell_size.is_some(),
            "Should find REA with threshold 0.5");
    }

    #[test]
    fn test_rea_curvature() {
        let dem = complex_dem(64, 64);
        let result = rea_analysis(&dem, ReaParams {
            scale_factors: vec![1, 2, 4, 8],
            variable: ReaVariable::PlanCurvature,
            ..Default::default()
        }).unwrap();

        assert!(!result.scales.is_empty());
        // Plan curvature should also degrade with resolution
        let first_corr = result.scales.iter()
            .find(|s| s.scale_factor > 1 && !s.correlation.is_nan())
            .map(|s| s.correlation)
            .unwrap_or(1.0);
        assert!(first_corr < 1.0, "Curvature should degrade at coarser scales");
    }

    #[test]
    fn test_rea_too_small_dem() {
        let dem = tilted_dem(2, 2);
        assert!(rea_analysis(&dem, ReaParams::default()).is_err());
    }

    #[test]
    fn test_aggregate_dem() {
        let dem = tilted_dem(8, 8);
        let agg = aggregate_dem(&dem, 2);
        assert_eq!(agg.rows(), 4);
        assert_eq!(agg.cols(), 4);

        // Check that aggregation is mean of 2×2 block
        let orig = dem.data();
        let expected = (orig[[0, 0]] + orig[[0, 1]] + orig[[1, 0]] + orig[[1, 1]]) / 4.0;
        let actual = agg.data()[[0, 0]];
        assert!(
            (actual - expected).abs() < 1e-10,
            "Aggregated value should be mean: expected={}, got={}",
            expected, actual
        );
    }
}
