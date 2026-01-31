//! Terrain analysis algorithms
//!
//! Algorithms for analyzing Digital Elevation Models (DEMs):
//! - Slope, Aspect, Hillshade, Curvature, TPI, TRI, Landform
//! - TWI: Topographic Wetness Index
//! - SPI/STI: Stream Power / Sediment Transport Index
//! - Geomorphons: pattern-based landform classification
//! - Viewshed: line-of-sight visibility analysis
//! - Sky View Factor, Openness: sky visibility metrics
//! - Convergence Index: flow convergence/divergence
//! - Multiscale Curvatures: Florinsky's robust method
//! - Feature-Preserving Smoothing: noise removal preserving edges
//! - Wind Exposure: topographic wind shelter/exposure
//! - Solar Radiation: beam + diffuse with terrain effects
//! - MRVBF/MRRTF: multi-resolution valley/ridge flatness

mod aspect;
mod convergence;
mod curvature;
mod geomorphons;
mod hillshade;
mod landform;
mod mrvbf;
mod multiscale_curvatures;
mod openness;
pub(crate) mod slope;
mod smoothing;
mod solar_radiation;
mod sky_view_factor;
mod spi_sti;
mod tpi;
mod tri;
mod twi;
mod viewshed;
mod wind_exposure;

pub use aspect::{aspect, Aspect, AspectOutput};
pub use convergence::{convergence_index, ConvergenceParams};
pub use curvature::{curvature, Curvature, CurvatureParams, CurvatureType};
pub use geomorphons::{geomorphons, GeomorphonParams};
pub use hillshade::{hillshade, Hillshade, HillshadeParams};
pub use landform::{landform_classification, Landform, LandformParams};
pub use mrvbf::{mrvbf, MrvbfParams};
pub use multiscale_curvatures::{multiscale_curvatures, MultiscaleCurvatureParams, MultiscaleCurvatureType};
pub use openness::{positive_openness, negative_openness, OpennessParams};
pub use slope::{slope, Slope, SlopeParams, SlopeUnits};
pub use smoothing::{feature_preserving_smoothing, SmoothingParams};
pub use solar_radiation::{solar_radiation, SolarParams, SolarRadiationResult};
pub use sky_view_factor::{sky_view_factor, SvfParams};
pub use spi_sti::{spi, sti, StiParams};
pub use tpi::{tpi, Tpi, TpiParams};
pub use tri::{tri, Tri, TriParams};
pub use twi::twi;
pub use viewshed::{viewshed, viewshed_multiple, ViewshedParams};
pub use wind_exposure::{wind_exposure, WindExposureParams};
