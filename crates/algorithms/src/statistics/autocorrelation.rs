//! Spatial autocorrelation for raster data
//!
//! - **Global Moran's I**: Overall spatial clustering measure
//! - **Local Getis-Ord Gi***: Hotspot/coldspot detection

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Result of Global Moran's I computation
#[derive(Debug, Clone)]
pub struct MoransIResult {
    /// Moran's I statistic (-1 to +1)
    pub i: f64,
    /// Expected I under randomness
    pub expected: f64,
    /// Z-score
    pub z_score: f64,
    /// P-value (two-tailed)
    pub p_value: f64,
}

/// Result of Local Getis-Ord Gi*
#[derive(Debug, Clone)]
pub struct GetisOrdResult {
    /// Raster of Gi* z-scores
    pub z_scores: Raster<f64>,
    /// Raster of p-values
    pub p_values: Raster<f64>,
}

/// Compute Global Moran's I for a raster
///
/// Uses a queen's case (8-neighbor) spatial weight matrix.
///
/// # Arguments
/// * `raster` - Input raster
///
/// # Returns
/// MoransIResult with I statistic, z-score, and p-value
pub fn global_morans_i(raster: &Raster<f64>) -> Result<MoransIResult> {
    let (rows, cols) = raster.shape();

    // Collect valid values and compute mean
    let mut values: Vec<(usize, usize, f64)> = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { raster.get_unchecked(row, col) };
            if !v.is_nan() {
                values.push((row, col, v));
            }
        }
    }

    let n = values.len() as f64;
    if n < 3.0 {
        return Err(Error::Algorithm("Need at least 3 valid cells".into()));
    }

    let mean = values.iter().map(|(_, _, v)| v).sum::<f64>() / n;

    // Deviations from mean
    let deviations: Vec<f64> = values.iter().map(|(_, _, v)| v - mean).collect();
    let sum_sq = deviations.iter().map(|d| d * d).sum::<f64>();

    if sum_sq.abs() < f64::EPSILON {
        return Ok(MoransIResult {
            i: 0.0,
            expected: -1.0 / (n - 1.0),
            z_score: 0.0,
            p_value: 1.0,
        });
    }

    // Build lookup for fast neighbor access
    let mut grid: Array2<Option<usize>> = Array2::from_elem((rows, cols), None);
    for (idx, &(row, col, _)) in values.iter().enumerate() {
        grid[(row, col)] = Some(idx);
    }

    // Compute numerator: sum of w_ij * (x_i - mean)(x_j - mean)
    let mut numerator = 0.0;
    let mut w_sum = 0.0;

    for &(row, col, _) in &values {
        let i_idx = grid[(row, col)].unwrap();
        let dev_i = deviations[i_idx];

        for dr in -1_isize..=1 {
            for dc in -1_isize..=1 {
                if dr == 0 && dc == 0 {
                    continue;
                }
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                    if let Some(j_idx) = grid[(nr as usize, nc as usize)] {
                        numerator += dev_i * deviations[j_idx];
                        w_sum += 1.0;
                    }
                }
            }
        }
    }

    let morans_i = (n / w_sum) * (numerator / sum_sq);
    let expected_i = -1.0 / (n - 1.0);

    // Variance under randomization assumption
    let s1 = 2.0 * w_sum; // Each w_ij=1, so s1 = 2*W (sum of (w_ij + w_ji)^2)
    let s2_part: f64 = values.iter().map(|&(row, col, _)| {
        let mut neighbors = 0.0_f64;
        for dr in -1_isize..=1 {
            for dc in -1_isize..=1 {
                if dr == 0 && dc == 0 { continue; }
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                    if grid[(nr as usize, nc as usize)].is_some() {
                        neighbors += 1.0;
                    }
                }
            }
        }
        let total = neighbors * 2.0; // in-degree + out-degree
        total * total
    }).sum::<f64>();

    let s0 = w_sum;
    let nn = n;
    let nn1 = nn - 1.0;

    // Simplified variance
    let var_i = (nn * ((nn * nn - 3.0 * nn + 3.0) * s1 - nn * s2_part + 3.0 * s0 * s0)
        - (nn * nn - nn) * s1
        + 2.0 * nn * s2_part
        - 6.0 * s0 * s0)
        / ((nn - 1.0) * (nn - 2.0) * (nn - 3.0) * s0 * s0);

    let var_i_safe = if var_i > 0.0 { var_i } else { 1.0 / (nn1 * nn1) };
    let z_score = (morans_i - expected_i) / var_i_safe.sqrt();

    // Two-tailed p-value approximation using normal distribution
    let p_value = 2.0 * normal_cdf(-z_score.abs());

    Ok(MoransIResult {
        i: morans_i,
        expected: expected_i,
        z_score,
        p_value,
    })
}

/// Compute Local Getis-Ord Gi* statistic for each cell
///
/// Identifies statistically significant spatial clusters of high values
/// (hotspots) and low values (coldspots) using a neighborhood radius.
///
/// # Arguments
/// * `raster` - Input raster
/// * `radius` - Neighborhood radius in cells
///
/// # Returns
/// GetisOrdResult with z-score and p-value rasters
pub fn local_getis_ord(raster: &Raster<f64>, radius: usize) -> Result<GetisOrdResult> {
    if radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }

    let (rows, cols) = raster.shape();
    let r = radius as isize;

    // Global statistics
    let mut global_sum = 0.0;
    let mut global_sum_sq = 0.0;
    let mut n = 0.0;

    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { raster.get_unchecked(row, col) };
            if !v.is_nan() {
                global_sum += v;
                global_sum_sq += v * v;
                n += 1.0;
            }
        }
    }

    if n < 2.0 {
        return Err(Error::Algorithm("Need at least 2 valid cells".into()));
    }

    let global_mean = global_sum / n;
    let s = ((global_sum_sq / n) - global_mean * global_mean).sqrt();

    if s.abs() < f64::EPSILON {
        // All values identical
        let mut z_scores = raster.with_same_meta::<f64>(rows, cols);
        z_scores.set_nodata(Some(f64::NAN));
        *z_scores.data_mut() = Array2::from_elem((rows, cols), 0.0);
        let mut p_values = raster.with_same_meta::<f64>(rows, cols);
        p_values.set_nodata(Some(f64::NAN));
        *p_values.data_mut() = Array2::from_elem((rows, cols), 1.0);
        return Ok(GetisOrdResult { z_scores, p_values });
    }

    // Compute Gi* for each cell in parallel
    let z_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_z = vec![f64::NAN; cols];

            for col in 0..cols {
                let center = unsafe { raster.get_unchecked(row, col) };
                if center.is_nan() {
                    continue;
                }

                // Sum of neighbors (including self for Gi*)
                let mut local_sum = 0.0;
                let mut w_count = 0.0;

                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = row as isize + dr;
                        let nc = col as isize + dc;
                        if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                            let v = unsafe { raster.get_unchecked(nr as usize, nc as usize) };
                            if !v.is_nan() {
                                local_sum += v;
                                w_count += 1.0;
                            }
                        }
                    }
                }

                // Gi* = (local_sum - mean * w) / (s * sqrt((n*w - w²) / (n-1)))
                let numerator = local_sum - global_mean * w_count;
                let denom = s * ((n * w_count - w_count * w_count) / (n - 1.0)).sqrt();

                if denom.abs() > f64::EPSILON {
                    row_z[col] = numerator / denom;
                } else {
                    row_z[col] = 0.0;
                }
            }

            row_z
        })
        .collect();

    let mut z_scores = raster.with_same_meta::<f64>(rows, cols);
    z_scores.set_nodata(Some(f64::NAN));
    *z_scores.data_mut() = Array2::from_shape_vec((rows, cols), z_data.clone())
        .map_err(|e| Error::Other(e.to_string()))?;

    // P-values from z-scores
    let p_data: Vec<f64> = z_data.iter()
        .map(|&z| {
            if z.is_nan() {
                f64::NAN
            } else {
                2.0 * normal_cdf(-z.abs())
            }
        })
        .collect();

    let mut p_values = raster.with_same_meta::<f64>(rows, cols);
    p_values.set_nodata(Some(f64::NAN));
    *p_values.data_mut() = Array2::from_shape_vec((rows, cols), p_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(GetisOrdResult { z_scores, p_values })
}

/// Approximate CDF of standard normal distribution
/// Uses Abramowitz & Stegun approximation (error < 7.5e-8)
fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 { return 0.0; }
    if x > 8.0 { return 1.0; }

    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-x * x / 2.0).exp()
        * (t * (0.3193815
            + t * (-0.3565638
                + t * (1.781478
                    + t * (-1.821256
                        + t * 1.330274)))));

    if x > 0.0 { 1.0 - p } else { p }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_morans_i_uniform() {
        let mut r = Raster::filled(10, 10, 5.0_f64);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let result = global_morans_i(&r).unwrap();
        assert!((result.i).abs() < 1e-10, "Uniform should have I≈0");
    }

    #[test]
    fn test_morans_i_clustered() {
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        // Left half = 0, right half = 100 → strong spatial clustering
        for row in 0..10 {
            for col in 0..10 {
                r.set(row, col, if col < 5 { 0.0 } else { 100.0 }).unwrap();
            }
        }
        let result = global_morans_i(&r).unwrap();
        assert!(result.i > 0.5, "Clustered data should have high positive I, got {}", result.i);
    }

    #[test]
    fn test_getis_ord_hotspot() {
        let mut r = Raster::filled(20, 20, 1.0_f64);
        r.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        // Create a hotspot cluster at center
        for row in 8..12 {
            for col in 8..12 {
                r.set(row, col, 100.0).unwrap();
            }
        }

        let result = local_getis_ord(&r, 2).unwrap();
        let center_z = result.z_scores.get(10, 10).unwrap();
        let edge_z = result.z_scores.get(0, 0).unwrap();

        assert!(center_z > edge_z, "Center hotspot should have higher z-score");
        assert!(center_z > 1.96, "Center should be significant at p<0.05, z={}", center_z);
    }

    #[test]
    fn test_getis_ord_radius_zero() {
        let r = Raster::filled(5, 5, 1.0_f64);
        let result = local_getis_ord(&r, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.002);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.002);
    }
}
