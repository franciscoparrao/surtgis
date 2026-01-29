//! Terrain analysis algorithms
//!
//! Algorithms for analyzing Digital Elevation Models (DEMs):
//! - Slope: rate of change of elevation
//! - Aspect: direction of steepest descent
//! - Hillshade: shaded relief visualization
//! - Curvature: surface curvature (TODO)
//! - TPI: Topographic Position Index (TODO)
//! - TRI: Terrain Ruggedness Index (TODO)

mod aspect;
mod hillshade;
mod slope;

pub use aspect::{aspect, Aspect, AspectOutput};
pub use hillshade::{hillshade, Hillshade, HillshadeParams};
pub use slope::{slope, Slope, SlopeParams, SlopeUnits};
