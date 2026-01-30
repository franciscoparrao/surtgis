//! Terrain analysis algorithms
//!
//! Algorithms for analyzing Digital Elevation Models (DEMs):
//! - Slope: rate of change of elevation
//! - Aspect: direction of steepest descent
//! - Hillshade: shaded relief visualization
//! - Curvature: profile, plan, and general surface curvature
//! - TPI: Topographic Position Index
//! - TRI: Terrain Ruggedness Index
//! - Landform: multi-scale landform classification

mod aspect;
mod curvature;
mod hillshade;
mod landform;
pub(crate) mod slope;
mod tpi;
mod tri;

pub use aspect::{aspect, Aspect, AspectOutput};
pub use curvature::{curvature, Curvature, CurvatureParams, CurvatureType};
pub use hillshade::{hillshade, Hillshade, HillshadeParams};
pub use landform::{landform_classification, Landform, LandformParams};
pub use slope::{slope, Slope, SlopeParams, SlopeUnits};
pub use tpi::{tpi, Tpi, TpiParams};
pub use tri::{tri, Tri, TriParams};
