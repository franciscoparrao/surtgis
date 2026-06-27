//! Raster data structures and operations

#[cfg(feature = "complex")]
pub mod complex;
mod element;
mod geotransform;
mod grid;
mod neighborhood;

#[cfg(feature = "complex")]
pub use complex::{complex_from_parts, complex_to_parts, magnitude, phase};
pub use element::{RasterCell, RasterElement};
pub use geotransform::GeoTransform;
pub use grid::Raster;
pub use neighborhood::{Neighborhood, NeighborhoodIterator};
