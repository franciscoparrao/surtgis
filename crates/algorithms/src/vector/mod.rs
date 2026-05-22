//! Vector analysis algorithms
//!
//! Geometric operations on vector features:
//! - Buffer: expand or shrink geometries
//! - Simplify: reduce vertex count (Douglas-Peucker, Visvalingam)
//! - Convex hull: minimum enclosing convex polygon
//! - Centroid: geometric center
//! - Bounding box: axis-aligned envelope
//! - Dissolve: merge features by attribute
//! - Clip: intersect geometries with a boundary
//! - Area / Length: geometric measurements

mod buffer;
mod clip;
mod measurements;
pub mod overlay;
mod simplify;
mod spatial;

pub use buffer::{BufferParams, buffer_geometry, buffer_points};
pub use clip::{ClipRect, clip_by_rect};
pub use measurements::{area, length, perimeter};
pub use overlay::{
    difference, dissolve as dissolve_overlay, intersection, symmetric_difference, union,
};
pub use simplify::{SimplifyParams, simplify_dp, simplify_vw};
pub use spatial::{BoundingBox, bounding_box, centroid, convex_hull, dissolve};
