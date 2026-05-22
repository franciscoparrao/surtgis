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
    pub x: f64,
    pub y: f64,
    pub value: f64,
}

impl SamplePoint {
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
