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

pub mod classification;
pub mod fluvial;
pub mod hydrology;
pub mod imagery;
pub mod interpolation;
pub mod landscape;
pub mod morphology;
pub mod segmentation;
pub mod statistics;
pub mod temporal;
pub mod terrain;
pub mod texture;
pub mod vector;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::hydrology::{
        AdaptiveMfdParams,
        BreachParams,
        Depression,
        DinfResult,
        FillSinks,
        FlowAccumulation,
        FlowDirection,
        HandParams,
        MfdParams,
        NestedDepressionParams,
        NestedDepressionResult,
        ParallelWatershedParams,
        PriorityFlood,
        PriorityFloodParams,
        StreamNetworkParams,
        TfgaParams,
        Watershed,
        breach_depressions,
        fill_sinks,
        flow_accumulation,
        flow_accumulation_mfd,
        // P3: Adaptive MFD, TFGA, nested depressions, parallel watershed
        flow_accumulation_mfd_adaptive,
        flow_accumulation_tfga,
        flow_dinf,
        flow_direction,
        flow_direction_dinf,
        hand,
        nested_depressions,
        priority_flood,
        priority_flood_flat,
        stream_network,
        watershed,
        watershed_parallel,
    };
    pub use crate::imagery::{
        BandMathOp, SpectralIndex, band_math, band_math_binary, bsi, evi, gndvi, index_builder,
        mndwi, nbr, ndre, ndvi, ndwi, ngrdi, normalized_difference, reci, reclassify, savi,
    };
    pub use crate::interpolation::{
        AdaptivePower, Anisotropy, DriftOrder, EmpiricalVariogram, FittedVariogram, IdwParams,
        KdTree, KrigingResult, NaturalNeighborParams, NearestNeighborParams, NearestResult,
        OrdinaryKrigingParams, RegressionKrigingParams, RegressionKrigingResult, SamplePoint,
        TinParams, TpsParams, UniversalKrigingParams, UniversalKrigingResult, VariogramModel,
        VariogramParams, empirical_variogram, fit_best_variogram, fit_variogram, idw,
        natural_neighbor, nearest_neighbor, ordinary_kriging, regression_kriging,
        regression_kriging_with_variogram, tin_interpolation, tps_interpolation, universal_kriging,
    };
    pub use crate::morphology::{
        BlackHat, BlackHatParams, Closing, ClosingParams, Dilate, DilateParams, Erode, ErodeParams,
        Gradient, GradientParams, Opening, OpeningParams, StructuringElement, TopHat, TopHatParams,
        black_hat, closing, dilate, erode, gradient, opening, top_hat,
    };
    pub use crate::statistics::{
        FocalParams, FocalStatistic, GetisOrdResult, MoransIResult, ZonalResult, ZonalStatistic,
        focal_statistics, global_morans_i, local_getis_ord, zonal_statistics,
    };
    pub use crate::temporal::{
        AnomalyMethod, LinearTrendResult, MannKendallResult, PhenologyParams, PhenologyResult,
        TemporalStats, linear_trend, mann_kendall, sens_slope, temporal_anomaly, temporal_count,
        temporal_max, temporal_mean, temporal_min, temporal_percentile, temporal_stats,
        temporal_std, vegetation_phenology,
    };
    pub use crate::terrain::{
        AdvancedCurvatureType,
        AllCurvatures,
        Aspect,
        CellDimensions,
        ChebyshevDerivatives,
        ChebyshevParams,
        ConvergenceParams,
        Curvature,
        CurvatureFormula,
        CurvatureType,
        DerivativeMethod,
        // Shared derivatives module
        Derivatives,
        Dev,
        DevParams,
        DiffuseModel,
        FastHorizonParams,
        FftLowPassParams,
        GaussianSmoothingParams,
        GeomorphonParams,
        GssParams,
        GssResult,
        Hillshade,
        HorizonAngles,
        HorizonParams,
        IterativeMeanParams,
        Landform,
        LandformParams,
        MonthlySolarResult,
        MrvbfParams,
        MsvCombination,
        MsvParams,
        MsvResult,
        MultiHillshadeParams,
        MultiscaleCurvatureParams,
        MultiscaleCurvatureType,
        OpennessParams,
        PderlViewshedParams,
        ProbabilisticViewshedParams,
        ReaParams,
        ReaResult,
        ReaScaleResult,
        ReaVariable,
        ScaleLevel,
        Slope,
        SlopeUnits,
        SmoothingParams,
        SolarParams,
        SolarRadiationResult,
        SpheroidalParams,
        Ssa2dParams,
        StiParams,
        SvfParams,
        Tpi,
        TpiParams,
        Tri,
        TriParams,
        UncertaintyParams,
        UncertaintyResult,
        Vec3,
        ViewshedParams,
        VrmParams,
        WindExposureParams,
        ZONE_ACCUMULATION,
        ZONE_DISPERSION,
        ZONE_TRANSITIONAL_ACC,
        ZONE_TRANSITIONAL_DISP,
        accumulation_zones,
        // Advanced curvature system (Florinsky 12)
        advanced_curvatures,
        all_curvatures,
        aspect,
        cell_dimensions,
        // Chebyshev spectral method
        chebyshev_derivatives,
        convergence_index,
        cos_incidence_vectorial,
        curvature,
        curvedness,
        dev,
        eastness,
        evans_young,
        extract_window,
        feature_preserving_smoothing,
        // FFT smoothing
        fft_low_pass,
        // Gaussian scale-space
        gaussian_scale_space,
        gaussian_smoothing,
        geographic_cell_sizes,
        geomorphons,
        hillshade,
        horizon_angle_map,
        horizon_angle_map_fast,
        // Horizon angles (shared infrastructure + HORAYZON fast)
        horizon_angles,
        horizon_angles_fast,
        horn,
        iterative_mean_smoothing,
        landform_classification,
        log_transform,
        mrvbf,
        // P3: MSV, PDERL, spheroidal, 2D-SSA, Perez, Corripio
        msv,
        multidirectional_hillshade,
        multiscale_curvatures,
        negative_openness,
        // Quick-win algorithms
        northness,
        northness_eastness,
        positive_openness,
        // REA analysis
        rea_analysis,
        scale_space_derivatives,
        shape_index,
        sky_view_factor,
        slope,
        slope_geographic,
        solar_radiation,
        // Solar annual
        solar_radiation_annual,
        solar_radiation_shadowed,
        solar_vector,
        spi,
        ssa_2d,
        sti,
        surface_normal,
        tpi,
        tri,
        // New terrain algorithms
        twi,
        uncertainty,
        viewshed,
        viewshed_multiple,
        viewshed_pderl,
        // Probabilistic viewshed
        viewshed_probabilistic,
        viewshed_xdraw,
        vincenty_distance,
        // VRM
        vrm,
        wind_exposure,
        zevenbergen_thorne,
    };
    pub use crate::vector::{
        BoundingBox, BufferParams, ClipRect, SimplifyParams, area, bounding_box, buffer_geometry,
        buffer_points, centroid, clip_by_rect, convex_hull, dissolve, length, perimeter,
        simplify_dp, simplify_vw,
    };
    pub use surtgis_core::prelude::*;
}
