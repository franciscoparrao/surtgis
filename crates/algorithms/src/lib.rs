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

mod maybe_rayon;

pub mod terrain;
pub mod hydrology;
pub mod imagery;
pub mod interpolation;
pub mod landscape;
pub mod morphology;
pub mod statistics;
pub mod vector;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::terrain::{
        aspect, curvature, hillshade, landform_classification, slope, tpi, tri,
        Aspect, Curvature, CurvatureType, CurvatureFormula, DerivativeMethod,
        Hillshade, Landform, Slope, SlopeUnits,
        Tpi, TpiParams, Tri, TriParams, LandformParams,
        // New terrain algorithms
        twi, spi, sti, StiParams,
        geomorphons, GeomorphonParams,
        viewshed, viewshed_xdraw, viewshed_multiple, ViewshedParams,
        sky_view_factor, SvfParams,
        positive_openness, negative_openness, OpennessParams,
        convergence_index, ConvergenceParams,
        multiscale_curvatures, MultiscaleCurvatureParams, MultiscaleCurvatureType,
        feature_preserving_smoothing, SmoothingParams,
        gaussian_smoothing, GaussianSmoothingParams,
        iterative_mean_smoothing, IterativeMeanParams,
        wind_exposure, WindExposureParams,
        solar_radiation, solar_radiation_shadowed, SolarParams, SolarRadiationResult, DiffuseModel,
        mrvbf, MrvbfParams,
        // Quick-win algorithms
        northness, eastness, northness_eastness,
        dev, Dev, DevParams,
        multidirectional_hillshade, MultiHillshadeParams,
        shape_index, curvedness,
        log_transform,
        accumulation_zones,
        ZONE_ACCUMULATION, ZONE_DISPERSION,
        ZONE_TRANSITIONAL_ACC, ZONE_TRANSITIONAL_DISP,
        // Advanced curvature system (Florinsky 12)
        advanced_curvatures, all_curvatures,
        AdvancedCurvatureType, AllCurvatures,
        // Horizon angles (shared infrastructure + HORAYZON fast)
        horizon_angles, horizon_angle_map, HorizonParams, HorizonAngles,
        horizon_angles_fast, horizon_angle_map_fast, FastHorizonParams,
        // Shared derivatives module
        Derivatives, evans_young, zevenbergen_thorne, horn, extract_window,
        // Gaussian scale-space
        gaussian_scale_space, scale_space_derivatives, GssParams, GssResult, ScaleLevel,
        // Chebyshev spectral method
        chebyshev_derivatives, ChebyshevParams, ChebyshevDerivatives,
        // VRM
        vrm, VrmParams,
        // Probabilistic viewshed
        viewshed_probabilistic, ProbabilisticViewshedParams,
        // REA analysis
        rea_analysis, ReaParams, ReaResult, ReaScaleResult, ReaVariable,
        // Solar annual
        solar_radiation_annual, MonthlySolarResult,
        // FFT smoothing
        fft_low_pass, FftLowPassParams,
        // P3: MSV, PDERL, spheroidal, 2D-SSA, Perez, Corripio
        msv, MsvParams, MsvCombination, MsvResult,
        viewshed_pderl, PderlViewshedParams,
        solar_vector, surface_normal, cos_incidence_vectorial, Vec3,
        cell_dimensions, geographic_cell_sizes, slope_geographic, vincenty_distance,
        SpheroidalParams, CellDimensions,
        ssa_2d, Ssa2dParams,
        uncertainty, UncertaintyParams, UncertaintyResult,
    };
    pub use crate::hydrology::{
        fill_sinks, flow_direction, flow_accumulation, watershed, hand,
        priority_flood, priority_flood_flat, stream_network,
        flow_accumulation_mfd, flow_direction_dinf, flow_dinf,
        breach_depressions,
        FillSinks, FlowDirection, FlowAccumulation, Watershed, HandParams,
        PriorityFlood, PriorityFloodParams, StreamNetworkParams, MfdParams,
        DinfResult, BreachParams,
        // P3: Adaptive MFD, TFGA, nested depressions, parallel watershed
        flow_accumulation_mfd_adaptive, AdaptiveMfdParams,
        flow_accumulation_tfga, TfgaParams,
        nested_depressions, NestedDepressionParams, NestedDepressionResult, Depression,
        watershed_parallel, ParallelWatershedParams,
    };
    pub use crate::imagery::{
        ndvi, ndwi, mndwi, nbr, savi, evi, bsi,
        ndre, gndvi, ngrdi, reci,
        normalized_difference, band_math, band_math_binary, reclassify,
        index_builder,
        BandMathOp, SpectralIndex,
    };
    pub use crate::interpolation::{
        idw, nearest_neighbor, tin_interpolation, tps_interpolation,
        empirical_variogram, fit_variogram, fit_best_variogram,
        ordinary_kriging, universal_kriging,
        regression_kriging, regression_kriging_with_variogram,
        natural_neighbor, NaturalNeighborParams,
        AdaptivePower, Anisotropy,
        IdwParams, NearestNeighborParams, TinParams, TpsParams, SamplePoint,
        EmpiricalVariogram, FittedVariogram, VariogramModel, VariogramParams,
        OrdinaryKrigingParams, KrigingResult,
        UniversalKrigingParams, UniversalKrigingResult, DriftOrder,
        RegressionKrigingParams, RegressionKrigingResult,
        KdTree, NearestResult,
    };
    pub use crate::vector::{
        buffer_points, buffer_geometry, BufferParams,
        simplify_dp, simplify_vw, SimplifyParams,
        bounding_box, centroid, convex_hull, dissolve, BoundingBox,
        clip_by_rect, ClipRect,
        area, length, perimeter,
    };
    pub use crate::morphology::{
        erode, dilate, opening, closing, gradient, top_hat, black_hat,
        Erode, Dilate, Opening, Closing, Gradient, TopHat, BlackHat,
        ErodeParams, DilateParams, OpeningParams, ClosingParams, GradientParams,
        TopHatParams, BlackHatParams, StructuringElement,
    };
    pub use crate::statistics::{
        focal_statistics, FocalStatistic, FocalParams,
        zonal_statistics, ZonalResult, ZonalStatistic,
        global_morans_i, local_getis_ord, MoransIResult, GetisOrdResult,
    };
    pub use surtgis_core::prelude::*;
}
