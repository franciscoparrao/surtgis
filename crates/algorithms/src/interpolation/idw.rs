//! Inverse Distance Weighting (IDW) interpolation
//!
//! Estimates values at unknown locations as a weighted average of nearby
//! sample points, where weights are inversely proportional to distance
//! raised to a power parameter.
//!
//! Reference:
//! Shepard, D. (1968). A two-dimensional interpolation function for
//! irregularly-spaced data. ACM National Conference.

use rayon::prelude::*;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::SamplePoint;

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
        }
    }
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

    // Pre-sort by distance if max_points is set
    let use_max_points = params.max_points.is_some();
    let max_points = params.max_points.unwrap_or(points.len());

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let (cx, cy) = params.transform.pixel_to_geo(col, row);

                // Collect distances and values
                let mut candidates: Vec<(f64, f64)> = Vec::new();
                let mut snapped = None;

                for pt in points {
                    let dsq = pt.dist_sq(cx, cy);

                    // Check snap distance
                    if dsq < snap * snap {
                        snapped = Some(pt.value);
                        break;
                    }

                    // Check max radius
                    if let Some(max_sq) = max_radius_sq {
                        if dsq > max_sq {
                            continue;
                        }
                    }

                    candidates.push((dsq, pt.value));
                }

                if let Some(val) = snapped {
                    row_data[col] = val;
                    continue;
                }

                if candidates.is_empty() {
                    continue; // NaN
                }

                // Sort by distance if limiting points
                if use_max_points && candidates.len() > max_points {
                    candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    candidates.truncate(max_points);
                }

                // Weighted average
                let mut sum_w = 0.0;
                let mut sum_wz = 0.0;

                for &(dsq, val) in &candidates {
                    let d = dsq.sqrt();
                    let w = 1.0 / d.powf(power);
                    sum_w += w;
                    sum_wz += w * val;
                }

                if sum_w > 0.0 {
                    row_data[col] = sum_wz / sum_w;
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
