//! Spatial interpolation algorithms
//!
//! Interpolate scattered point data onto regular raster grids:
//! - IDW: Inverse Distance Weighting
//! - Nearest Neighbor: value from closest point
//! - TIN: Triangulated Irregular Network (linear barycentric)
//! - TPS: Thin Plate Spline
//! - Variogram: empirical variogram computation and model fitting
//! - Ordinary Kriging: BLUE geostatistical interpolation
//! - Universal Kriging: kriging with polynomial drift
//! - Regression Kriging: OLS trend + OK on residuals

mod idw;
pub mod kdtree;
pub mod kriging;
mod natural_neighbor;
mod nearest;
mod regression_kriging;
mod tin;
mod tps;
mod universal_kriging;
pub mod variogram;

pub use idw::{AdaptivePower, Anisotropy, IdwParams, idw};
pub use kdtree::{KdTree, NearestResult};
pub use kriging::{KrigingResult, OrdinaryKrigingParams, ordinary_kriging};
pub use natural_neighbor::{NaturalNeighborParams, natural_neighbor};
pub use nearest::{NearestNeighborParams, nearest_neighbor};
pub use regression_kriging::{
    RegressionKrigingParams, RegressionKrigingResult, regression_kriging,
    regression_kriging_with_variogram,
};
pub use tin::{TinParams, tin_interpolation};
pub use tps::{TpsParams, tps_interpolation};
pub use universal_kriging::{
    DriftOrder, UniversalKrigingParams, UniversalKrigingResult, universal_kriging,
};
pub use variogram::{
    EmpiricalVariogram, FittedVariogram, VariogramModel, VariogramParams, empirical_variogram,
    fit_best_variogram, fit_variogram,
};

/// A sample point with x, y coordinates and a value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplePoint {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Sample value at `(x, y)`.
    pub value: f64,
}

impl SamplePoint {
    /// Construct a sample point from its coordinates and value.
    pub fn new(x: f64, y: f64, value: f64) -> Self {
        Self { x, y, value }
    }

    /// Squared Euclidean distance to another point
    #[inline]
    pub fn dist_sq(&self, other_x: f64, other_y: f64) -> f64 {
        let dx = self.x - other_x;
        let dy = self.y - other_y;
        dx * dx + dy * dy
    }

    /// Euclidean distance to another point
    #[inline]
    pub fn dist(&self, other_x: f64, other_y: f64) -> f64 {
        self.dist_sq(other_x, other_y).sqrt()
    }
}

/// Deduplicate sample points whose coordinates coincide within `tol`.
///
/// Kriging systems (both ordinary and universal) become singular when two
/// or more samples share (nearly) the same location: the corresponding
/// rows/columns of the covariance matrix are then linearly dependent
/// (γ(0) = 0 for the diagonal, and every off-diagonal entry references the
/// same inter-point distances). Real-world sample sets — borehole logs,
/// station networks, repeated field visits — routinely contain such exact
/// or near-exact duplicate coordinates.
///
/// Chilès & Delfiner (2012), *Geostatistics: Modeling Spatial Uncertainty*,
/// §3.6.1, recommend collapsing duplicate observations to their mean value
/// before kriging rather than discarding them or leaving the system
/// singular (which previously caused a silent fallback to IDW — see A-8).
///
/// `tol` is a coordinate tolerance in CRS units (e.g. `1e-6` for
/// projected/UTM coordinates). Points within `tol` of an already-visited
/// point are merged into it; the merged point's coordinates and value are
/// the arithmetic mean of the cluster.
pub(crate) fn dedupe_points(points: &[SamplePoint], tol: f64) -> Vec<SamplePoint> {
    let tol_sq = tol * tol;
    let n = points.len();
    let mut visited = vec![false; n];
    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;
        let mut sum_x = points[i].x;
        let mut sum_y = points[i].y;
        let mut sum_z = points[i].value;
        let mut count = 1usize;

        for j in (i + 1)..n {
            if visited[j] {
                continue;
            }
            let dx = points[j].x - points[i].x;
            let dy = points[j].y - points[i].y;
            if dx * dx + dy * dy <= tol_sq {
                visited[j] = true;
                sum_x += points[j].x;
                sum_y += points[j].y;
                sum_z += points[j].value;
                count += 1;
            }
        }

        let inv = 1.0 / count as f64;
        out.push(SamplePoint::new(sum_x * inv, sum_y * inv, sum_z * inv));
    }

    out
}

#[cfg(test)]
mod dedupe_tests {
    use super::*;

    #[test]
    fn dedupe_merges_exact_duplicates_by_averaging() {
        let pts = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(0.0, 0.0, 20.0),
            SamplePoint::new(5.0, 5.0, 100.0),
        ];
        let deduped = dedupe_points(&pts, 1e-6);
        assert_eq!(deduped.len(), 2);
        let merged = deduped.iter().find(|p| p.x == 0.0 && p.y == 0.0).unwrap();
        assert!((merged.value - 15.0).abs() < 1e-10);
    }

    #[test]
    fn dedupe_merges_near_duplicates_within_tolerance() {
        let pts = vec![
            SamplePoint::new(100.0, 200.0, 1.0),
            SamplePoint::new(100.0 + 1e-8, 200.0, 3.0),
        ];
        let deduped = dedupe_points(&pts, 1e-6);
        assert_eq!(deduped.len(), 1);
        assert!((deduped[0].value - 2.0).abs() < 1e-10);
    }

    #[test]
    fn dedupe_leaves_distinct_points_untouched() {
        let pts = vec![
            SamplePoint::new(0.0, 0.0, 1.0),
            SamplePoint::new(1.0, 1.0, 2.0),
            SamplePoint::new(2.0, 2.0, 3.0),
        ];
        let deduped = dedupe_points(&pts, 1e-6);
        assert_eq!(deduped.len(), 3);
    }
}
