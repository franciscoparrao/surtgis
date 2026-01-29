//! # SurtGis Algorithms
//!
//! Geospatial analysis algorithms for SurtGis.
//!
//! ## Available Algorithm Categories
//!
//! - **terrain**: Slope, aspect, hillshade, curvatures, TPI, TRI
//! - **hydrology**: Fill sinks, flow direction, flow accumulation, watershed delineation
//! - **imagery**: Spectral indices, band math, reclassification
//! - **vector**: Buffer, overlay, simplify (TODO)
//! - **interpolation**: IDW, kriging, splines (TODO)

pub mod terrain;
pub mod hydrology;
pub mod imagery;
// pub mod vector;     // TODO
// pub mod interpolation; // TODO

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::terrain::{
        aspect, hillshade, slope, Aspect, Hillshade, Slope, SlopeUnits,
    };
    pub use crate::hydrology::{
        fill_sinks, flow_direction, flow_accumulation, watershed,
        FillSinks, FlowDirection, FlowAccumulation, Watershed,
    };
    pub use crate::imagery::{
        ndvi, ndwi, mndwi, nbr, savi, evi, bsi,
        normalized_difference, band_math, band_math_binary, reclassify,
        BandMathOp, SpectralIndex,
    };
    pub use surtgis_core::prelude::*;
}
