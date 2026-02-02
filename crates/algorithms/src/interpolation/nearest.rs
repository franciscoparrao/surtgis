//! Nearest Neighbor interpolation
//!
//! Assigns each grid cell the value of the closest sample point.
//! Fast and simple, produces a Voronoi-like tessellation.

use crate::maybe_rayon::*;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::SamplePoint;

/// Parameters for Nearest Neighbor interpolation
#[derive(Debug, Clone)]
pub struct NearestNeighborParams {
    /// Maximum search radius. Cells beyond this distance from all
    /// sample points are set to NaN. `None` for unlimited.
    pub max_radius: Option<f64>,
    /// Output raster rows
    pub rows: usize,
    /// Output raster columns
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
}

impl Default for NearestNeighborParams {
    fn default() -> Self {
        Self {
            max_radius: None,
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
        }
    }
}

/// Perform Nearest Neighbor interpolation from scattered points to a raster grid.
///
/// Each cell receives the value of the closest sample point (Voronoi assignment).
///
/// # Arguments
/// * `points` - Scattered sample points with values
/// * `params` - Output grid parameters
///
/// # Returns
/// Raster with interpolated values
pub fn nearest_neighbor(points: &[SamplePoint], params: NearestNeighborParams) -> Result<Raster<f64>> {
    if points.is_empty() {
        return Err(Error::Algorithm("No sample points provided".into()));
    }

    let rows = params.rows;
    let cols = params.cols;
    let max_radius_sq = params.max_radius.map(|r| r * r);

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let (cx, cy) = params.transform.pixel_to_geo(col, row);

                let mut min_dist_sq = f64::MAX;
                let mut nearest_val = f64::NAN;

                for pt in points {
                    let dsq = pt.dist_sq(cx, cy);
                    if dsq < min_dist_sq {
                        min_dist_sq = dsq;
                        nearest_val = pt.value;
                    }
                }

                // Check max radius
                if let Some(max_sq) = max_radius_sq {
                    if min_dist_sq > max_sq {
                        continue;
                    }
                }

                row_data[col] = nearest_val;
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
            SamplePoint::new(2.0, 8.0, 10.0),  // top-left quadrant
            SamplePoint::new(8.0, 8.0, 20.0),  // top-right quadrant
            SamplePoint::new(2.0, 2.0, 30.0),  // bottom-left quadrant
            SamplePoint::new(8.0, 2.0, 40.0),  // bottom-right quadrant
        ]
    }

    fn default_params() -> NearestNeighborParams {
        NearestNeighborParams {
            rows: 10,
            cols: 10,
            transform: GeoTransform::new(0.0, 10.0, 1.0, -1.0),
            ..Default::default()
        }
    }

    #[test]
    fn test_nearest_basic() {
        let points = sample_points();
        let result = nearest_neighbor(&points, default_params()).unwrap();

        // All cells should have one of the 4 values
        for row in 0..10 {
            for col in 0..10 {
                let val = result.get(row, col).unwrap();
                assert!(
                    val == 10.0 || val == 20.0 || val == 30.0 || val == 40.0,
                    "Unexpected value {} at ({}, {})",
                    val, row, col
                );
            }
        }
    }

    #[test]
    fn test_nearest_voronoi() {
        let points = sample_points();
        let result = nearest_neighbor(&points, default_params()).unwrap();

        // Top-left corner should be closest to point at (2, 8) → value 10
        let tl = result.get(0, 0).unwrap();
        assert_eq!(tl, 10.0, "Top-left should be 10.0, got {}", tl);

        // Bottom-right corner should be closest to point at (8, 2) → value 40
        let br = result.get(9, 9).unwrap();
        assert_eq!(br, 40.0, "Bottom-right should be 40.0, got {}", br);
    }

    #[test]
    fn test_nearest_with_radius() {
        let points = sample_points();
        let params = NearestNeighborParams {
            max_radius: Some(1.0),
            ..default_params()
        };

        let result = nearest_neighbor(&points, params).unwrap();

        // Center cell is far from all points → should be NaN
        let center = result.get(5, 5).unwrap();
        assert!(center.is_nan(), "Center should be NaN, got {}", center);
    }

    #[test]
    fn test_nearest_single_point() {
        let points = vec![SamplePoint::new(5.0, 5.0, 99.0)];
        let result = nearest_neighbor(&points, default_params()).unwrap();

        // All cells should be 99.0
        for row in 0..10 {
            for col in 0..10 {
                assert_eq!(result.get(row, col).unwrap(), 99.0);
            }
        }
    }

    #[test]
    fn test_nearest_empty() {
        let result = nearest_neighbor(&[], default_params());
        assert!(result.is_err());
    }
}
