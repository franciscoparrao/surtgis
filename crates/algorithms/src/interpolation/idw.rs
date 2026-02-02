//! Inverse Distance Weighting (IDW) interpolation
//!
//! Estimates values at unknown locations as a weighted average of nearby
//! sample points, where weights are inversely proportional to distance
//! raised to a power parameter.
//!
//! Reference:
//! Shepard, D. (1968). A two-dimensional interpolation function for
//! irregularly-spaced data. ACM National Conference.

use crate::maybe_rayon::*;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::SamplePoint;

/// Adaptive power parameters for density-dependent IDW.
///
/// Adjusts the power exponent per cell based on local point density.
/// In sparse areas, power increases (sharper falloff); in dense areas,
/// power decreases (smoother interpolation).
///
/// Reference: Chen (2015). Adaptive IDW with variable power parameter.
#[derive(Debug, Clone)]
pub struct AdaptivePower {
    /// Number of nearest points for density estimation (default: 5)
    pub k: usize,
    /// Reference distance: when d_mean == d_ref, power is unchanged.
    /// `None` → automatically computed as mean spacing of all points.
    pub d_ref: Option<f64>,
    /// Sensitivity exponent (default: 0.5). Higher = more adaptation.
    pub alpha: f64,
    /// Maximum allowed power (default: 6.0)
    pub max_power: f64,
    /// Minimum allowed power (default: 0.5)
    pub min_power: f64,
}

impl Default for AdaptivePower {
    fn default() -> Self {
        Self {
            k: 5,
            d_ref: None,
            alpha: 0.5,
            max_power: 6.0,
            min_power: 0.5,
        }
    }
}

/// Anisotropic search configuration for directional IDW.
///
/// Uses an elliptical search window instead of circular, giving more
/// weight to points along a preferred direction (e.g., along a valley).
#[derive(Debug, Clone)]
pub struct Anisotropy {
    /// Major axis direction in radians (0=East, π/2=North)
    pub angle: f64,
    /// Ratio of minor/major axis (0..1]. 1.0 = isotropic (no effect).
    pub ratio: f64,
}

/// Parameters for IDW interpolation
#[derive(Debug, Clone)]
pub struct IdwParams {
    /// Power parameter (default: 2.0).
    /// Higher values give more weight to nearby points.
    pub power: f64,
    /// Maximum search radius. Points beyond this distance are ignored.
    /// `None` means all points are used (global IDW).
    pub max_radius: Option<f64>,
    /// Maximum number of nearest points to use.
    /// `None` means use all points within radius.
    pub max_points: Option<usize>,
    /// Minimum distance threshold. If a sample point is closer than this
    /// to the target cell, its value is used directly (avoids singularity).
    pub snap_distance: f64,
    /// Output raster dimensions (rows, cols)
    pub rows: usize,
    /// Output raster columns
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
    /// Adaptive power: vary exponent based on local point density.
    /// When `Some`, power is adjusted per cell. When `None`, fixed power is used.
    pub adaptive: Option<AdaptivePower>,
    /// Anisotropic search: use elliptical distance instead of Euclidean.
    /// When `Some`, distances are stretched along the minor axis direction.
    pub anisotropy: Option<Anisotropy>,
}

impl Default for IdwParams {
    fn default() -> Self {
        Self {
            power: 2.0,
            max_radius: None,
            max_points: None,
            snap_distance: 1e-10,
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
            adaptive: None,
            anisotropy: None,
        }
    }
}

/// Compute anisotropic distance squared between two points.
/// Stretches coordinates perpendicular to the major axis direction.
fn aniso_dist_sq(dx: f64, dy: f64, cos_a: f64, sin_a: f64, ratio_inv: f64) -> f64 {
    // Rotate into aligned frame
    let u = dx * cos_a + dy * sin_a;   // along major axis
    let v = -dx * sin_a + dy * cos_a;  // along minor axis
    // Stretch minor axis → makes distant points along minor seem farther
    u * u + (v * ratio_inv) * (v * ratio_inv)
}

/// Perform IDW interpolation from scattered points to a raster grid.
///
/// # Algorithm
///
/// For each output cell at position (x, y):
///
/// ```text
/// z(x,y) = Σ(wi * zi) / Σ(wi)
/// where wi = 1 / d(x,y, xi,yi)^p
/// ```
///
/// # Arguments
/// * `points` - Scattered sample points with values
/// * `params` - IDW parameters (power, radius, output grid)
///
/// # Returns
/// Raster with interpolated values. Cells with no points within radius are NaN.
pub fn idw(points: &[SamplePoint], params: IdwParams) -> Result<Raster<f64>> {
    if points.is_empty() {
        return Err(Error::Algorithm("No sample points provided".into()));
    }

    let rows = params.rows;
    let cols = params.cols;
    let power = params.power;
    let snap = params.snap_distance;
    let max_radius_sq = params.max_radius.map(|r| r * r);

    let use_max_points = params.max_points.is_some();
    let max_points = params.max_points.unwrap_or(points.len());

    // Precompute anisotropy parameters
    let (use_aniso, cos_a, sin_a, ratio_inv) = match &params.anisotropy {
        Some(a) if a.ratio < 1.0 && a.ratio > 0.0 => {
            (true, a.angle.cos(), a.angle.sin(), 1.0 / a.ratio)
        }
        _ => (false, 1.0, 0.0, 1.0),
    };

    // Precompute adaptive power reference distance
    let d_ref = match &params.adaptive {
        Some(ap) => {
            ap.d_ref.unwrap_or_else(|| {
                // Auto: mean nearest-neighbor distance among sample points
                let n = points.len();
                if n < 2 { return 1.0; }
                let sum: f64 = points.iter().map(|p| {
                    points.iter()
                        .filter(|q| (q.x - p.x).abs() > 1e-15 || (q.y - p.y).abs() > 1e-15)
                        .map(|q| p.dist_sq(q.x, q.y))
                        .fold(f64::MAX, f64::min)
                        .sqrt()
                }).sum();
                (sum / n as f64).max(1e-10)
            })
        }
        None => 1.0,
    };
    let adaptive = params.adaptive.clone();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let (cx, cy) = params.transform.pixel_to_geo(col, row);

                // Collect distances and values
                let mut candidates: Vec<(f64, f64)> = Vec::new();
                let mut snapped = None;

                for pt in points {
                    let dx = cx - pt.x;
                    let dy = cy - pt.y;

                    let dsq = if use_aniso {
                        aniso_dist_sq(dx, dy, cos_a, sin_a, ratio_inv)
                    } else {
                        dx * dx + dy * dy
                    };

                    // Check snap distance
                    if dsq < snap * snap {
                        snapped = Some(pt.value);
                        break;
                    }

                    // Check max radius
                    if let Some(max_sq) = max_radius_sq
                        && dsq > max_sq {
                            continue;
                        }

                    candidates.push((dsq, pt.value));
                }

                if let Some(val) = snapped {
                    *row_data_col = val;
                    continue;
                }

                if candidates.is_empty() {
                    continue; // NaN
                }

                // Sort by distance if limiting points or adaptive
                let need_sort = (use_max_points && candidates.len() > max_points)
                    || adaptive.is_some();
                if need_sort {
                    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }
                if use_max_points && candidates.len() > max_points {
                    candidates.truncate(max_points);
                }

                // Determine effective power (adaptive or fixed)
                let eff_power = if let Some(ref ap) = adaptive {
                    let k = ap.k.min(candidates.len());
                    let d_mean: f64 = candidates[..k].iter()
                        .map(|(dsq, _)| dsq.sqrt())
                        .sum::<f64>() / k as f64;
                    let p_local = power * (d_mean / d_ref).powf(ap.alpha);
                    p_local.clamp(ap.min_power, ap.max_power)
                } else {
                    power
                };

                // Weighted average
                let mut sum_w = 0.0;
                let mut sum_wz = 0.0;

                for &(dsq, val) in &candidates {
                    let d = dsq.sqrt();
                    let w = 1.0 / d.powf(eff_power);
                    sum_w += w;
                    sum_wz += w * val;
                }

                if sum_w > 0.0 {
                    *row_data_col = sum_wz / sum_w;
                }
            }

            row_data
        })
        .collect();

    let mut output = Raster::from_vec(data, rows, cols)?;
    output.set_transform(params.transform);
    output.set_nodata(Some(f64::NAN));

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_points() -> Vec<SamplePoint> {
        vec![
            SamplePoint::new(0.5, 9.5, 10.0),  // top-left
            SamplePoint::new(9.5, 9.5, 20.0),  // top-right
            SamplePoint::new(0.5, 0.5, 30.0),  // bottom-left
            SamplePoint::new(9.5, 0.5, 40.0),  // bottom-right
        ]
    }

    fn default_params() -> IdwParams {
        IdwParams {
            rows: 10,
            cols: 10,
            transform: GeoTransform::new(0.0, 10.0, 1.0, -1.0),
            ..Default::default()
        }
    }

    #[test]
    fn test_idw_basic() {
        let points = sample_points();
        let result = idw(&points, default_params()).unwrap();

        // No NaN values expected (global IDW)
        for row in 0..10 {
            for col in 0..10 {
                let val = result.get(row, col).unwrap();
                assert!(!val.is_nan(), "NaN at ({}, {})", row, col);
            }
        }
    }

    #[test]
    fn test_idw_at_sample_points() {
        let points = sample_points();
        let result = idw(&points, default_params()).unwrap();

        // Near top-left point (0.5, 9.5) → value ~10.0
        let val = result.get(0, 0).unwrap();
        assert!(
            (val - 10.0).abs() < 3.0,
            "Near top-left should be ~10.0, got {}",
            val
        );

        // Near bottom-right point (9.5, 0.5) → value ~40.0
        let val = result.get(9, 9).unwrap();
        assert!(
            (val - 40.0).abs() < 3.0,
            "Near bottom-right should be ~40.0, got {}",
            val
        );
    }

    #[test]
    fn test_idw_center_is_average() {
        // With 4 equidistant corners, center should be ~average
        let points = sample_points();
        let result = idw(&points, default_params()).unwrap();

        let center = result.get(5, 5).unwrap();
        let avg = (10.0 + 20.0 + 30.0 + 40.0) / 4.0; // 25.0

        assert!(
            (center - avg).abs() < 5.0,
            "Center should be ~{}, got {}",
            avg,
            center
        );
    }

    #[test]
    fn test_idw_with_radius() {
        let points = sample_points();
        let params = IdwParams {
            max_radius: Some(2.0), // Very small radius
            ..default_params()
        };

        let result = idw(&points, params).unwrap();

        // Center should be NaN (no points within 2.0 units)
        let center = result.get(5, 5).unwrap();
        assert!(center.is_nan(), "Center should be NaN with small radius");
    }

    #[test]
    fn test_idw_with_max_points() {
        let points = sample_points();
        let params = IdwParams {
            max_points: Some(2),
            ..default_params()
        };

        let result = idw(&points, params).unwrap();

        // Should still produce valid values
        let val = result.get(0, 0).unwrap();
        assert!(!val.is_nan());
    }

    #[test]
    fn test_idw_power_effect() {
        let points = sample_points();

        let low_power = idw(&points, IdwParams {
            power: 1.0,
            ..default_params()
        }).unwrap();

        let high_power = idw(&points, IdwParams {
            power: 4.0,
            ..default_params()
        }).unwrap();

        // With higher power, values near sample points should be closer
        // to the sample value (sharper falloff)
        let near_pt_low = low_power.get(0, 0).unwrap();
        let near_pt_high = high_power.get(0, 0).unwrap();

        // High power should be closer to 10.0 (the nearby sample value)
        assert!(
            (near_pt_high - 10.0).abs() <= (near_pt_low - 10.0).abs() + 0.1,
            "Higher power should weight nearby points more: low={}, high={}",
            near_pt_low,
            near_pt_high
        );
    }

    #[test]
    fn test_idw_empty_points() {
        let result = idw(&[], default_params());
        assert!(result.is_err());
    }

    #[test]
    fn test_idw_adaptive_power() {
        // Adaptive power should produce different results than fixed
        let points = sample_points();

        let fixed = idw(&points, IdwParams {
            power: 2.0,
            ..default_params()
        }).unwrap();

        let adaptive = idw(&points, IdwParams {
            power: 2.0,
            adaptive: Some(AdaptivePower::default()),
            ..default_params()
        }).unwrap();

        // Values should differ (adaptive adjusts power per cell)
        let f_center = fixed.get(5, 5).unwrap();
        let a_center = adaptive.get(5, 5).unwrap();

        // Both should be valid
        assert!(!f_center.is_nan() && !a_center.is_nan());
        // They may differ (adaptive adjusts power based on distance)
        // Near corners they should still be close to sample values
        let a_corner = adaptive.get(0, 0).unwrap();
        assert!(
            (a_corner - 10.0).abs() < 5.0,
            "Adaptive IDW near sample should be close: got {}",
            a_corner
        );
    }

    #[test]
    fn test_idw_anisotropic() {
        // Create points along x-axis: anisotropy along x should weight them more
        let points = vec![
            SamplePoint::new(2.0, 5.0, 100.0),
            SamplePoint::new(8.0, 5.0, 100.0),
            SamplePoint::new(5.0, 2.0, 0.0),
            SamplePoint::new(5.0, 8.0, 0.0),
        ];

        // Isotropic: center should be ~50 (equidistant from all 4)
        let iso = idw(&points, IdwParams {
            ..default_params()
        }).unwrap();

        // Anisotropic along x-axis: east-west points (100.0) should dominate
        let aniso = idw(&points, IdwParams {
            anisotropy: Some(Anisotropy { angle: 0.0, ratio: 0.3 }),
            ..default_params()
        }).unwrap();

        let iso_center = iso.get(5, 5).unwrap();
        let aniso_center = aniso.get(5, 5).unwrap();

        // Anisotropic should pull toward the x-axis points (100.0)
        assert!(
            aniso_center > iso_center,
            "Anisotropic along x should weight E-W points more: iso={:.1}, aniso={:.1}",
            iso_center, aniso_center
        );
    }

    #[test]
    fn test_idw_anisotropy_ratio_one_is_isotropic() {
        let points = sample_points();

        let iso = idw(&points, default_params()).unwrap();
        let aniso_1 = idw(&points, IdwParams {
            anisotropy: Some(Anisotropy { angle: 0.5, ratio: 1.0 }),
            ..default_params()
        }).unwrap();

        // ratio=1.0 should be identical to isotropic
        let v1 = iso.get(3, 7).unwrap();
        let v2 = aniso_1.get(3, 7).unwrap();
        assert!(
            (v1 - v2).abs() < 1e-10,
            "Ratio 1.0 should be isotropic: {} vs {}",
            v1, v2
        );
    }

    #[test]
    fn test_idw_single_point() {
        let points = vec![SamplePoint::new(5.0, 5.0, 42.0)];
        let result = idw(&points, default_params()).unwrap();

        // All cells should have value 42.0 (single point dominates everywhere)
        for row in 0..10 {
            for col in 0..10 {
                let val = result.get(row, col).unwrap();
                assert!(
                    (val - 42.0).abs() < 1e-6,
                    "Single point IDW should be 42.0 everywhere, got {} at ({}, {})",
                    val, row, col
                );
            }
        }
    }
}
