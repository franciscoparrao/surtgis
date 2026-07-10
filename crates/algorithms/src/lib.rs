//! # SurtGis Algorithms
//!
//! Geospatial analysis algorithms for SurtGis.
//!
//! ## Available Algorithm Categories
//!
//! - **terrain**: Slope, aspect, hillshade, curvature, TPI, TRI, landform, TWI, SPI, STI,
//!   geomorphons, viewshed, SVF, openness, convergence, multiscale curvatures,
//!   feature-preserving smoothing, wind exposure, solar radiation, MRVBF/MRRTF,
//!   northness/eastness, DEV, multidirectional hillshade, shape index, curvedness,
//!   log transform, accumulation zones
//! - **hydrology**: Fill sinks, Priority-Flood, flow direction, flow accumulation, watershed, HAND, stream network
//! - **imagery**: Spectral indices (NDVI, NDWI, NDRE, GNDVI, NGRDI, RECI, etc.), band math, reclassification
//! - **interpolation**: IDW, nearest neighbor, TIN
//! - **vector**: Buffer, simplify, spatial operations, clipping, measurements
//! - **morphology**: Erosion, dilation, opening, closing, gradient, top-hat, black-hat
//! - **statistics**: Focal statistics, zonal statistics, spatial autocorrelation
//!
//! ## `ndarray` / `geo` / `geo-types` re-exports
//!
//! Vector algorithms return `geo`/`geo-types` geometries and raster
//! algorithms are built on `ndarray::Array2` — both are re-exported here
//! (`surtgis_algorithms::{ndarray, geo, geo_types}`) so downstream code
//! depends on the exact versions this crate's public API uses, matching
//! [`surtgis_core`]'s policy of the same name. A major-version bump of
//! any of the three is a major (breaking) bump of `surtgis-algorithms`.

// Part of the engine ecosystem's stable surface: every public item must be documented.
#![deny(missing_docs)]

mod maybe_rayon;

/// Re-exported so downstream code can depend on the exact `geo` version
/// this crate's vector algorithms use. See the crate-level docs.
pub use geo;
/// Re-exported so downstream code can depend on the exact `geo-types`
/// version this crate's vector algorithms use. See the crate-level docs.
pub use geo_types;
/// Re-exported so downstream code can depend on the exact `ndarray`
/// version this crate's public API uses. See the crate-level docs.
pub use ndarray;

pub mod classification;
pub mod fluvial;
pub mod hydrology;
pub mod imagery;
pub mod interpolation;
pub mod landscape;
pub mod morphology;
pub mod pansharpening;
pub mod sampling;
pub mod segmentation;
pub mod statistics;
pub mod temporal;
pub mod terrain;
pub mod texture;
pub mod vector;

/// Curated prelude of the most commonly used algorithms.
///
/// This is a convenience subset, not the full API surface — every module
/// (`terrain`, `hydrology`, `imagery`, `interpolation`, `morphology`,
/// `statistics`, `temporal`, `vector`, plus `classification`, `fluvial`,
/// `landscape`, `pansharmening`, `sampling`, `segmentation`, `texture`)
/// exposes its complete set of algorithms via `crate::<module>::*`
/// regardless of whether a given symbol is re-exported here.
pub mod prelude {
    pub use crate::hydrology::{
        HandParams, PriorityFloodParams, WatershedParams, fill_sinks, flow_accumulation,
        flow_direction, hand, priority_flood, priority_flood_flat, watershed,
    };
    pub use crate::imagery::{BandMathOp, SpectralIndex, band_math, evi, mndwi, ndvi, ndwi, savi};
    pub use crate::interpolation::{
        IdwParams, KrigingResult, NearestNeighborParams, OrdinaryKrigingParams, fit_variogram, idw,
        nearest_neighbor, ordinary_kriging,
    };
    pub use crate::morphology::{
        ClosingParams, DilateParams, ErodeParams, OpeningParams, StructuringElement, closing,
        dilate, erode, opening,
    };
    pub use crate::statistics::{
        FocalParams, FocalStatistic, ZonalResult, ZonalStatistic, focal_statistics,
        zonal_statistics, zonal_statistics_raster,
    };
    pub use crate::temporal::{
        LinearTrendResult, MannKendallResult, TemporalStats, linear_trend, mann_kendall,
        temporal_stats,
    };
    pub use crate::terrain::{
        AspectOutput, CurvatureParams, CurvatureType, HillshadeParams, LandformParams, SlopeParams,
        SlopeUnits, TpiParams, TriParams, aspect, curvature, hillshade, landform_classification,
        slope, tpi, tri,
    };
    pub use crate::vector::{
        BoundingBox, area, bounding_box, buffer_geometry, centroid, clip_by_rect, perimeter,
    };
    pub use surtgis_core::prelude::*;
}
