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
//! - Northness/Eastness: aspect decomposition
//! - DEV: Deviation from Mean Elevation
//! - Multidirectional Hillshade: multi-azimuth shaded relief
//! - Shape Index & Curvedness: differential geometry measures
//! - Log Transform: curvature visualization
//! - Accumulation Zones: flow convergence classification

mod accumulation_zones;
mod aspect;
mod chebyshev_spectral;
mod convergence;
mod curvature;
pub mod derivatives;
mod curvature_advanced;
mod dev;
mod gaussian_scale_space;
mod geomorphons;
mod hillshade;
mod horizon_angles;
mod landform;
mod lineament;
mod log_transform;
mod mrvbf;
mod multidirectional_hillshade;
mod multiscale_curvatures;
mod northness_eastness;
mod openness;
mod rea;
mod shape_index;
pub(crate) mod slope;
mod smoothing;
mod solar_radiation;
mod sky_view_factor;
mod spi_sti;
mod tpi;
mod tri;
mod twi;
mod viewshed;
mod msv;
mod pderl_viewshed;
mod spheroidal_grid;
mod ssa_2d;
mod uncertainty;
mod vrm;
mod wind_exposure;

pub use accumulation_zones::{
    accumulation_zones, ZONE_ACCUMULATION, ZONE_DISPERSION,
    ZONE_TRANSITIONAL_ACC, ZONE_TRANSITIONAL_DISP,
};
pub use aspect::{aspect, Aspect, AspectOutput};
pub use convergence::{convergence_index, ConvergenceParams};
pub use curvature::{curvature, Curvature, CurvatureParams, CurvatureType, DerivativeMethod, CurvatureFormula};
pub use curvature_advanced::{advanced_curvatures, all_curvatures, AdvancedCurvatureType, AllCurvatures};
pub use dev::{dev, Dev, DevParams};
pub use geomorphons::{geomorphons, GeomorphonParams};
pub use hillshade::{hillshade, Hillshade, HillshadeParams};
pub use landform::{landform_classification, Landform, LandformParams};
pub use log_transform::log_transform;
pub use mrvbf::{mrvbf, MrvbfParams};
pub use multidirectional_hillshade::{multidirectional_hillshade, MultiHillshadeParams};
pub use multiscale_curvatures::{multiscale_curvatures, MultiscaleCurvatureParams, MultiscaleCurvatureType};
pub use northness_eastness::{northness, eastness, northness_eastness};
pub use openness::{positive_openness, negative_openness, OpennessParams};
pub use shape_index::{shape_index, curvedness};
pub use slope::{slope, Slope, SlopeParams, SlopeUnits};
pub use smoothing::{
    feature_preserving_smoothing, SmoothingParams,
    gaussian_smoothing, GaussianSmoothingParams,
    iterative_mean_smoothing, IterativeMeanParams,
    fft_low_pass, FftLowPassParams,
};
pub use solar_radiation::{solar_radiation, solar_radiation_shadowed, solar_radiation_annual, SolarParams, SolarRadiationResult, MonthlySolarResult, DiffuseModel};
pub use sky_view_factor::{sky_view_factor, SvfParams};
pub use spi_sti::{spi, sti, StiParams};
pub use tpi::{tpi, Tpi, TpiParams};
pub use tri::{tri, Tri, TriParams};
pub use twi::twi;
pub use viewshed::{viewshed, viewshed_xdraw, viewshed_multiple, viewshed_probabilistic, observer_optimization, ViewshedParams, ProbabilisticViewshedParams, ObserverOptimizationParams, ObserverOptimizationResult};
pub use horizon_angles::{horizon_angles, horizon_angle_map, HorizonParams, HorizonAngles, horizon_angles_fast, horizon_angle_map_fast, FastHorizonParams};
pub use vrm::{vrm, VrmParams};
pub use wind_exposure::{wind_exposure, WindExposureParams};
pub use derivatives::{Derivatives, evans_young, zevenbergen_thorne, horn, extract_window};
pub use gaussian_scale_space::{gaussian_scale_space, scale_space_derivatives, GssParams, GssResult, ScaleLevel};
pub use chebyshev_spectral::{chebyshev_derivatives, ChebyshevParams, ChebyshevDerivatives};
pub use rea::{rea_analysis, ReaParams, ReaResult, ReaScaleResult, ReaVariable};
pub use lineament::{lineament_detection, LineamentParams, LineamentResult, LineamentType};
pub use msv::{msv, MsvParams, MsvCombination, MsvResult};
pub use pderl_viewshed::{viewshed_pderl, PderlViewshedParams};
pub use solar_radiation::{solar_vector, surface_normal, cos_incidence_vectorial, Vec3};
pub use spheroidal_grid::{cell_dimensions, geographic_cell_sizes, slope_geographic, vincenty_distance, SpheroidalParams, CellDimensions};
pub use ssa_2d::{ssa_2d, Ssa2dParams};
pub use uncertainty::{uncertainty, UncertaintyParams, UncertaintyResult};
