//! # SurtGis Algorithms
//!
//! Geospatial analysis algorithms for SurtGis.
//!
//! ## Available Algorithm Categories
//!
//! - **terrain**: Slope, aspect, hillshade, curvatures, TPI, TRI
//! - **hydrology**: Flow direction, accumulation, watersheds (TODO)
//! - **imagery**: Spectral indices, classification (TODO)
//! - **vector**: Buffer, overlay, simplify (TODO)
//! - **interpolation**: IDW, kriging, splines (TODO)

pub mod terrain;
// pub mod hydrology;  // TODO
// pub mod imagery;    // TODO
// pub mod vector;     // TODO
// pub mod interpolation; // TODO

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::terrain::{
        aspect, hillshade, slope, Aspect, Hillshade, Slope, SlopeUnits,
    };
    pub use surtgis_core::prelude::*;
}
