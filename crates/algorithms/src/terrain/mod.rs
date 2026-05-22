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
mod circular_variance_aspect;
mod contour;
mod convergence;
mod cost_distance;
mod curvature;
mod curvature_advanced;
pub mod derivatives;
mod dev;
mod diff_from_mean_elev;
mod directional_relief;
mod downslope_index;
mod edge_density;
mod elev_above_pit;
mod elev_relative_to_min_max;
mod gaussian_scale_space;
mod geomorphons;
mod hillshade;
mod horizon_angles;
mod hypsometric_hillshade;
mod landform;
mod lineament;
mod log_transform;
mod ls_factor;
mod max_branch_length;
mod mrvbf;
mod msv;
mod multidirectional_hillshade;
mod multiscale_curvatures;
mod neighbours;
mod normal_vector_deviation;
mod northness_eastness;
mod openness;
mod pderl_viewshed;
mod pennock;
mod percent_elev_range;
mod rea;
mod relative_aspect;
mod relative_slope_position;
mod shape_index;
mod sky_view_factor;
pub(crate) mod slope;
mod smoothing;
mod solar_radiation;
mod spherical_std_dev;
mod spheroidal_grid;
mod spi_sti;
mod ssa_2d;
mod surface_area_ratio;
mod tpi;
mod tri;
mod twi;
mod uncertainty;
mod valley_depth;
mod viewshed;
mod vrm;
mod wind_exposure;

pub use accumulation_zones::{
    ZONE_ACCUMULATION, ZONE_DISPERSION, ZONE_TRANSITIONAL_ACC, ZONE_TRANSITIONAL_DISP,
    accumulation_zones,
};
pub use aspect::{Aspect, AspectOutput, AspectStreaming, aspect};
pub use chebyshev_spectral::{ChebyshevDerivatives, ChebyshevParams, chebyshev_derivatives};
pub use circular_variance_aspect::{
    CircularVarianceParams, CircularVarianceStreaming, circular_variance_aspect,
};
pub use contour::{ContourParams, contour_lines};
pub use convergence::{ConvergenceParams, ConvergenceStreaming, convergence_index};
pub use cost_distance::{CostDistanceParams, cost_distance};
pub use curvature::{
    Curvature, CurvatureFormula, CurvatureParams, CurvatureStreaming, CurvatureType,
    DerivativeMethod, curvature,
};
pub use curvature_advanced::{
    AdvancedCurvatureType, AllCurvatures, advanced_curvatures, all_curvatures,
};
pub use derivatives::{Derivatives, evans_young, extract_window, horn, zevenbergen_thorne};
pub use dev::{Dev, DevParams, DevStreaming, dev};
pub use diff_from_mean_elev::{DiffFromMeanParams, DiffFromMeanStreaming, diff_from_mean_elev};
pub use directional_relief::{DirectionalReliefParams, directional_relief};
pub use downslope_index::{DownslopeIndexParams, downslope_index};
pub use edge_density::{EdgeDensityParams, edge_density};
pub use elev_above_pit::elev_above_pit;
pub use elev_relative_to_min_max::elev_relative_to_min_max;
pub use gaussian_scale_space::{
    GssParams, GssResult, ScaleLevel, gaussian_scale_space, scale_space_derivatives,
};
pub use geomorphons::{GeomorphonParams, geomorphons};
pub use hillshade::{Hillshade, HillshadeParams, HillshadeStreaming, hillshade};
pub use horizon_angles::{
    FastHorizonParams, HorizonAngles, HorizonParams, horizon_angle_map, horizon_angle_map_fast,
    horizon_angles, horizon_angles_fast,
};
pub use hypsometric_hillshade::hypsometric_hillshade;
pub use landform::{Landform, LandformParams, landform_classification};
pub use lineament::{LineamentParams, LineamentResult, LineamentType, lineament_detection};
pub use log_transform::log_transform;
pub use ls_factor::{LsFactorParams, ls_factor};
pub use max_branch_length::max_branch_length;
pub use mrvbf::{MrvbfParams, mrvbf};
pub use msv::{MsvCombination, MsvParams, MsvResult, msv};
pub use multidirectional_hillshade::{
    MultiHillshadeParams, MultiHillshadeStreaming, multidirectional_hillshade,
};
pub use multiscale_curvatures::{
    MultiscaleCurvatureParams, MultiscaleCurvatureType, multiscale_curvatures,
};
pub use neighbours::{NeighbourStatsResult, neighbour_stats};
pub use normal_vector_deviation::{
    NormalDeviationParams, NormalDeviationStreaming, normal_vector_deviation,
};
pub use northness_eastness::{
    EastnessStreaming, NorthnessStreaming, eastness, northness, northness_eastness,
};
pub use openness::{OpennessParams, negative_openness, positive_openness};
pub use pderl_viewshed::{PderlViewshedParams, viewshed_pderl};
pub use pennock::{PennockParams, pennock};
pub use percent_elev_range::{
    PercentElevRangeParams, PercentElevRangeStreaming, percent_elev_range,
};
pub use rea::{ReaParams, ReaResult, ReaScaleResult, ReaVariable, rea_analysis};
pub use relative_aspect::{RelativeAspectParams, relative_aspect};
pub use relative_slope_position::relative_slope_position;
pub use shape_index::{curvedness, shape_index};
pub use sky_view_factor::{SvfParams, sky_view_factor};
pub use slope::{Slope, SlopeParams, SlopeStreaming, SlopeUnits, slope};
pub use smoothing::{
    FftLowPassParams, GaussianSmoothingParams, IterativeMeanParams, SmoothingParams,
    feature_preserving_smoothing, fft_low_pass, gaussian_smoothing, iterative_mean_smoothing,
};
pub use solar_radiation::{
    DiffuseModel, MonthlySolarResult, SolarParams, SolarRadiationResult, solar_radiation,
    solar_radiation_annual, solar_radiation_annual_only, solar_radiation_shadowed,
};
pub use solar_radiation::{Vec3, cos_incidence_vectorial, solar_vector, surface_normal};
pub use spherical_std_dev::{SphericalStdDevParams, SphericalStdDevStreaming, spherical_std_dev};
pub use spheroidal_grid::{
    CellDimensions, SpheroidalParams, cell_dimensions, geographic_cell_sizes, slope_geographic,
    vincenty_distance,
};
pub use spi_sti::{StiParams, spi, sti};
pub use ssa_2d::{Ssa2dParams, ssa_2d};
pub use surface_area_ratio::{SarParams, surface_area_ratio};
pub use tpi::{Tpi, TpiParams, TpiStreaming, tpi};
pub use tri::{Tri, TriParams, TriStreaming, tri};
pub use twi::twi;
pub use uncertainty::{UncertaintyParams, UncertaintyResult, uncertainty};
pub use valley_depth::valley_depth;
pub use viewshed::{
    ObserverOptimizationParams, ObserverOptimizationResult, ProbabilisticViewshedParams,
    ViewshedParams, observer_optimization, viewshed, viewshed_multiple, viewshed_probabilistic,
    viewshed_xdraw,
};
pub use vrm::{VrmParams, VrmStreaming, vrm};
pub use wind_exposure::{WindExposureParams, wind_exposure};
