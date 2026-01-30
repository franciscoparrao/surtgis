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
//! - **vector**: Buffer, simplify, spatial operations, clipping, measurements

pub mod terrain;
pub mod hydrology;
pub mod imagery;
pub mod interpolation;
pub mod vector;

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
    pub use crate::vector::{
        buffer_points, buffer_geometry, BufferParams,
        simplify_dp, simplify_vw, SimplifyParams,
        bounding_box, centroid, convex_hull, dissolve, BoundingBox,
        clip_by_rect, ClipRect,
        area, length, perimeter,
    };
    pub use surtgis_core::prelude::*;
}
