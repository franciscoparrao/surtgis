//! Raster data structures and operations

#[cfg(feature = "complex")]
pub mod complex;
mod element;
mod geotransform;
mod grid;
mod neighborhood;
mod validate;

#[cfg(feature = "complex")]
pub use complex::{complex_from_parts, complex_to_parts, magnitude, phase};
pub use element::{RasterCell, RasterElement};
pub use geotransform::GeoTransform;
pub use grid::Raster;
pub use neighborhood::{Neighborhood, NeighborhoodIterator};
pub use validate::{check_aligned, check_same_crs, check_same_shape};
