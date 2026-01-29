//! Spatial interpolation algorithms
//!
//! Interpolate scattered point data onto regular raster grids:
//! - IDW: Inverse Distance Weighting
//! - Nearest Neighbor: value from closest point
//! - TIN: Triangulated Irregular Network (linear barycentric)

mod idw;
mod nearest;
mod tin;

pub use idw::{idw, IdwParams};
pub use nearest::{nearest_neighbor, NearestNeighborParams};
pub use tin::{tin_interpolation, TinParams};

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
