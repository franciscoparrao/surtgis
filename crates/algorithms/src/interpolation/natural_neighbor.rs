//! Natural Neighbor (Sibson) Interpolation
//!
//! Locally adaptive interpolation that preserves data values exactly and
//! produces C1-continuous surfaces (except at data points where it is C0).
//!
//! Uses the discrete Sibson method: for each query point, measures the
//! "area stolen" from each natural neighbor's Voronoi cell on a local
//! sub-grid, then weights accordingly.
//!
//! Reference:
//! Sibson, R. (1981). "A brief description of natural neighbour interpolation."
//! In Interpreting Multivariate Data, pp. 21–36.

use crate::maybe_rayon::*;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::SamplePoint;

/// Parameters for Natural Neighbor interpolation
#[derive(Debug, Clone)]
pub struct NaturalNeighborParams {
    /// Maximum number of candidate points to consider (default: 20).
    /// Only the k nearest points participate in weight computation.
    pub max_neighbors: usize,
    /// Sub-grid resolution for discrete area estimation (default: 11).
    /// Higher values give more accurate weights but slower computation.
    /// Must be odd (will be rounded up if even).
    pub sub_resolution: usize,
    /// Output raster dimensions (rows)
    pub rows: usize,
    /// Output raster dimensions (cols)
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
}

impl Default for NaturalNeighborParams {
    fn default() -> Self {
        Self {
            max_neighbors: 20,
            sub_resolution: 11,
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
        }
    }
}

/// Perform Natural Neighbor (Sibson) interpolation.
///
/// # Algorithm (Discrete Sibson)
///
/// For each output cell at position (x, y):
/// 1. Find the k nearest sample points
/// 2. Create a local sub-grid centered at (x, y)
/// 3. Assign each sub-cell to its nearest sample point (original Voronoi)
/// 4. Add the query point and re-assign sub-cells (modified Voronoi)
/// 5. Count sub-cells "stolen" from each neighbor → Sibson weights
/// 6. Interpolated value = weighted sum of neighbor values
///
/// # Properties
/// - Exact interpolation: passes through data points
/// - C1 continuous (except at data points: C0)
/// - Locally adaptive: only uses natural neighbors
///
/// # Arguments
/// * `points` — Scattered sample points with values
/// * `params` — Natural Neighbor parameters
///
/// # Returns
/// Raster with interpolated values
pub fn natural_neighbor(points: &[SamplePoint], params: NaturalNeighborParams) -> Result<Raster<f64>> {
    if points.is_empty() {
        return Err(Error::Algorithm("No sample points provided".into()));
    }
    if points.len() < 3 {
        return Err(Error::Algorithm("Natural Neighbor requires at least 3 points".into()));
    }

    let rows = params.rows;
    let cols = params.cols;
    let k = params.max_neighbors.min(points.len());
    // Ensure odd sub-resolution
    let sub_res = if params.sub_resolution % 2 == 0 {
        params.sub_resolution + 1
    } else {
        params.sub_resolution
    };

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let (cx, cy) = params.transform.pixel_to_geo(col, row);

                // Find k nearest points
                let mut dists: Vec<(usize, f64)> = points.iter()
                    .enumerate()
                    .map(|(i, p)| (i, p.dist_sq(cx, cy)))
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                dists.truncate(k);

                // Check for exact hit (point coincides with cell center)
                if dists[0].1 < 1e-20 {
                    *row_data_col = points[dists[0].0].value;
                    continue;
                }

                // Determine sub-grid extent based on distance to farthest neighbor
                let max_dist = dists.last().unwrap().1.sqrt();
                let half_extent = max_dist * 0.6; // Sub-grid covers inner 60% of neighbor range
                let step = 2.0 * half_extent / (sub_res - 1) as f64;

                if step < 1e-15 {
                    // All points at same location, use nearest
                    *row_data_col = points[dists[0].0].value;
                    continue;
                }

                // Compute Sibson weights via area stealing on sub-grid
                let neighbor_indices: Vec<usize> = dists.iter().map(|(i, _)| *i).collect();
                let neighbor_count = neighbor_indices.len();

                // Count sub-cells assigned to each neighbor (original Voronoi)
                let mut orig_count = vec![0_usize; neighbor_count];
                // Count sub-cells stolen by query point
                let mut stolen_count = vec![0_usize; neighbor_count];
                let mut total_stolen = 0_usize;

                for sr in 0..sub_res {
                    for sc in 0..sub_res {
                        let sx = cx - half_extent + sc as f64 * step;
                        let sy = cy - half_extent + sr as f64 * step;

                        // Find nearest among candidate neighbors
                        let mut best_idx = 0;
                        let mut best_dist = f64::MAX;
                        for (ni, &pi) in neighbor_indices.iter().enumerate() {
                            let d = points[pi].dist_sq(sx, sy);
                            if d < best_dist {
                                best_dist = d;
                                best_idx = ni;
                            }
                        }

                        orig_count[best_idx] += 1;

                        // Check if query point is closer
                        let dq = (sx - cx) * (sx - cx) + (sy - cy) * (sy - cy);
                        if dq < best_dist {
                            // This sub-cell is stolen by query point
                            stolen_count[best_idx] += 1;
                            total_stolen += 1;
                        }
                    }
                }

                if total_stolen == 0 {
                    // Fallback: use nearest neighbor
                    *row_data_col = points[neighbor_indices[0]].value;
                    continue;
                }

                // Sibson weight = stolen_from_i / total_stolen
                let mut sum_wv = 0.0;
                let mut sum_w = 0.0;
                for (ni, &pi) in neighbor_indices.iter().enumerate() {
                    if stolen_count[ni] > 0 {
                        let w = stolen_count[ni] as f64 / total_stolen as f64;
                        sum_wv += w * points[pi].value;
                        sum_w += w;
                    }
                }

                if sum_w > 0.0 {
                    *row_data_col = sum_wv / sum_w;
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
            SamplePoint::new(1.0, 9.0, 10.0),
            SamplePoint::new(9.0, 9.0, 20.0),
            SamplePoint::new(1.0, 1.0, 30.0),
            SamplePoint::new(9.0, 1.0, 40.0),
            SamplePoint::new(5.0, 5.0, 25.0),
        ]
    }

    fn default_params() -> NaturalNeighborParams {
        NaturalNeighborParams {
            rows: 10,
            cols: 10,
            transform: GeoTransform::new(0.0, 10.0, 1.0, -1.0),
            sub_resolution: 11,
            ..Default::default()
        }
    }

    #[test]
    fn test_nn_basic() {
        let points = sample_points();
        let result = natural_neighbor(&points, default_params()).unwrap();

        // Should produce valid values everywhere
        for row in 0..10 {
            for col in 0..10 {
                let val = result.get(row, col).unwrap();
                assert!(!val.is_nan(), "NaN at ({}, {})", row, col);
            }
        }
    }

    #[test]
    fn test_nn_exact_at_sample() {
        // Natural Neighbor should pass through data points exactly
        let points = vec![
            SamplePoint::new(0.5, 9.5, 10.0),
            SamplePoint::new(9.5, 9.5, 20.0),
            SamplePoint::new(0.5, 0.5, 30.0),
            SamplePoint::new(9.5, 0.5, 40.0),
            SamplePoint::new(5.5, 5.5, 25.0),
        ];

        let result = natural_neighbor(&points, NaturalNeighborParams {
            rows: 10,
            cols: 10,
            transform: GeoTransform::new(0.0, 10.0, 1.0, -1.0),
            ..Default::default()
        }).unwrap();

        // Cell (0,0) center is (0.5, 9.5) → should be exactly 10.0
        let v = result.get(0, 0).unwrap();
        assert!(
            (v - 10.0).abs() < 1e-6,
            "At data point should be exact: expected 10.0, got {:.6}",
            v
        );
    }

    #[test]
    fn test_nn_range() {
        let points = sample_points();
        let result = natural_neighbor(&points, default_params()).unwrap();

        // Values should be within range of input values [10, 40]
        for row in 0..10 {
            for col in 0..10 {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    assert!(
                        val >= 9.0 && val <= 41.0,
                        "Value outside expected range: got {:.1} at ({}, {})",
                        val, row, col
                    );
                }
            }
        }
    }

    #[test]
    fn test_nn_smooth() {
        // Adjacent cells should have similar values (smoothness)
        let points = sample_points();
        let result = natural_neighbor(&points, default_params()).unwrap();

        let v1 = result.get(5, 5).unwrap();
        let v2 = result.get(5, 6).unwrap();
        assert!(
            (v1 - v2).abs() < 10.0,
            "Adjacent cells should be similar: {:.1} vs {:.1}",
            v1, v2
        );
    }

    #[test]
    fn test_nn_needs_3_points() {
        let points = vec![
            SamplePoint::new(0.0, 0.0, 1.0),
            SamplePoint::new(1.0, 1.0, 2.0),
        ];
        assert!(natural_neighbor(&points, default_params()).is_err());
    }

    #[test]
    fn test_nn_empty_points() {
        assert!(natural_neighbor(&[], default_params()).is_err());
    }

    #[test]
    fn test_nn_vs_idw() {
        // Natural Neighbor should generally be smoother than IDW
        let points = sample_points();

        let nn = natural_neighbor(&points, default_params()).unwrap();

        use super::super::idw::{idw, IdwParams};
        let idw_result = idw(&points, IdwParams {
            rows: 10,
            cols: 10,
            transform: GeoTransform::new(0.0, 10.0, 1.0, -1.0),
            ..Default::default()
        }).unwrap();

        // Both should produce valid results
        let nn_center = nn.get(5, 5).unwrap();
        let idw_center = idw_result.get(5, 5).unwrap();
        assert!(!nn_center.is_nan() && !idw_center.is_nan());
    }
}
