//! # SurtGis Algorithms
//!
//! Geospatial analysis algorithms for SurtGis.
//!
//! ## Available Algorithm Categories
//!
//! - **terrain**: Slope, aspect, hillshade
//! - **hydrology**: Fill sinks, flow direction, flow accumulation, watershed delineation
//! - **imagery**: Spectral indices, band math, reclassification
//! - **interpolation**: IDW, nearest neighbor, TIN
//! - **vector**: Buffer, overlay, simplify (TODO)

pub mod terrain;
pub mod hydrology;
pub mod imagery;
pub mod interpolation;
// pub mod vector; // TODO

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
    pub use crate::interpolation::{
        idw, nearest_neighbor, tin_interpolation,
        IdwParams, NearestNeighborParams, TinParams, SamplePoint,
    };
    pub use surtgis_core::prelude::*;
}
