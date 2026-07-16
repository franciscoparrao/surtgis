//! Raster data structures and operations

mod any_raster;
#[cfg(feature = "complex")]
pub mod complex;
mod element;
mod geotransform;
mod grid;
mod neighborhood;
mod validate;

pub use any_raster::{AnyRaster, DataType};
// Re-export at the module path too (the macro itself is hoisted to the
// crate root by `#[macro_export]`), so both `surtgis_core::dispatch_any!`
// and `surtgis_core::raster::dispatch_any!` work.
pub use crate::dispatch_any;
#[cfg(feature = "complex")]
pub use complex::{complex_from_amp_phase, complex_from_parts, complex_to_parts, magnitude, phase};
pub use element::{RasterCell, RasterElement};
pub use geotransform::GeoTransform;
pub use grid::Raster;
pub use neighborhood::{Neighborhood, NeighborhoodIterator};
pub use validate::{check_aligned, check_same_crs, check_same_shape};
