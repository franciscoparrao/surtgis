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
mod simplify;
mod spatial;

pub use buffer::{buffer_geometry, buffer_points, BufferParams};
pub use clip::{clip_by_rect, ClipRect};
pub use measurements::{area, length, perimeter};
pub use simplify::{simplify_dp, simplify_vw, SimplifyParams};
pub use spatial::{bounding_box, centroid, convex_hull, dissolve, BoundingBox};
