//! CLI command and subcommand definitions.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

// ─── CLI structure ──────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "surtgis")]
#[command(author, version, about = "High-performance geospatial analysis", long_about = None)]
#[command(allow_negative_numbers = true)]
pub struct Cli {
    /// Verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Compress output GeoTIFFs (deflate)
    #[arg(long, global = true)]
    pub compress: bool,

    /// Force streaming mode for large rasters (auto-detected if >500MB)
    #[arg(long, global = true)]
    pub streaming: bool,

    /// Maximum memory to use (e.g., 4G, 1024MB, 500MiB).
    /// If raster would exceed this when decompressed, force streaming.
    #[arg(long, global = true)]
    pub max_memory: Option<String>,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Show information about a raster file
    Info {
        /// Input raster file
        input: PathBuf,
    },
    /// Terrain analysis algorithms
    Terrain {
        #[command(subcommand)]
        algorithm: TerrainCommands,
    },
    /// Hydrology algorithms
    Hydrology {
        #[command(subcommand)]
        algorithm: HydrologyCommands,
    },
    /// Fluvial-tectonic morphometry (chi, ksn, knickpoints, divide migration)
    Fluvial {
        #[command(subcommand)]
        algorithm: FluvialCommands,
    },
    /// Imagery / spectral index algorithms
    Imagery {
        #[command(subcommand)]
        algorithm: ImageryCommands,
    },
    /// Mathematical morphology algorithms
    Morphology {
        #[command(subcommand)]
        algorithm: MorphologyCommands,
    },
    /// Landscape ecology metrics (global patch/class/landscape level)
    Landscape {
        #[command(subcommand)]
        algorithm: LandscapeCommands,
    },
    /// Extract raster values at point locations to CSV
    Extract {
        /// Directory with features.json and feature rasters (from `pipeline features`)
        #[arg(long)]
        features_dir: PathBuf,
        /// Vector file with labeled points (.geojson, .shp, .gpkg)
        #[arg(long)]
        points: PathBuf,
        /// Property name containing the target label/value
        #[arg(long)]
        target: String,
        /// Output CSV file
        output: PathBuf,
    },
    /// Extract raster patches centered on points or sampled from polygons for CNN training
    ExtractPatches {
        /// Directory with features.json and feature rasters (auto-discovers .tif if no json)
        #[arg(long)]
        features_dir: PathBuf,
        /// Vector file with POINTS (.geojson, .shp, .gpkg). Mutually exclusive with --polygons.
        #[arg(long, conflicts_with = "polygons")]
        points: Option<PathBuf>,
        /// Vector file with POLYGONS for grid-sampling. Mutually exclusive with --points.
        #[arg(long, conflicts_with = "points")]
        polygons: Option<PathBuf>,
        /// Property name on the vector feature that carries the label
        #[arg(long)]
        label_col: String,
        /// Patch side length in pixels (square). Default 256.
        #[arg(long, default_value = "256")]
        size: usize,
        /// Grid stride in pixels when sampling polygons. Default = size (no overlap).
        #[arg(long)]
        stride: Option<usize>,
        /// Skip patches where fraction of NaN pixels exceeds this threshold. Default 0.1.
        #[arg(long, default_value = "0.1")]
        skip_nan_threshold: f64,
        /// Optional random subsample cap (uses deterministic seed)
        #[arg(long)]
        max_patches: Option<usize>,
        /// Random seed used for subsampling. Default 42.
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Geospatial Foundation Model profile to target. When set, applies
        /// per-band z-score normalization using the model's published stats
        /// and validates the band count. Use --size to override tile size.
        /// Supported: prithvi-v2, clay-v1.5.
        #[arg(long)]
        profile: Option<String>,
        /// Output format for the patch tensor: `npy` (single .npy file, default)
        /// or `zarr` (chunked Zarr v2 directory, one chunk per chip). Labels
        /// and manifest are always emitted as .npy / .csv regardless.
        #[arg(long, default_value = "npy")]
        output_format: String,
        /// Emit STAC ML-AOI Collection + Items describing the chips as
        /// labelled training data. Writes `<output>/stac/collection.json`
        /// and `<output>/stac/items/chip_NNNNNN.json`. When a --profile is
        /// also set, the Collection embeds the MLM extension declaring
        /// the target foundation model.
        #[arg(long)]
        emit_stac: bool,
        /// EPSG code of the input vector file (--points or --polygons).
        /// Defaults to 4326 — the GeoJSON spec mandates WGS84 lon/lat,
        /// and most shapefiles in the wild also use it. When the raster's
        /// CRS differs, vector coordinates are reprojected on the fly via
        /// proj4rs. Set explicitly when your vector is in projected
        /// coords already (e.g. 32718 for UTM 18S) to skip reprojection.
        #[arg(long, default_value = "4326")]
        points_crs: u32,
        /// Output directory (creates patches.npy or patches.zarr/, plus
        /// labels.npy, manifest.csv, meta.json, and optional stac/)
        output: PathBuf,
    },
    /// Rayshader-style shaded relief composite to PNG
    Relief {
        /// Input DEM file
        input: PathBuf,
        /// Output PNG file
        output: PathBuf,
        /// Color scheme for the base layer
        #[arg(long, default_value = "terrain")]
        colormap: String,
        /// Sun azimuth in degrees (0 = North, clockwise)
        #[arg(long, default_value = "315")]
        sun_azimuth: f64,
        /// Sun altitude in degrees above the horizon
        #[arg(long, default_value = "45")]
        sun_altitude: f64,
        /// Enable ray-traced cast shadows
        #[arg(long)]
        shadows: bool,
        /// Number of sun samples for the soft-shadow penumbra (>=2 enables soft shadows). Default 1 = hard shadows.
        #[arg(long, default_value = "1")]
        soft: usize,
        /// Enable ambient occlusion via sky-view factor
        #[arg(long)]
        ambient: bool,
        /// Detect water bodies (large flat regions) and tint them
        #[arg(long)]
        water: bool,
        /// Vertical exaggeration for the normal-based sphere shade
        #[arg(long, default_value = "1.0")]
        z_factor: f64,
        /// Search radius (cells) for the ambient SVF computation
        #[arg(long, default_value = "30")]
        radius: usize,
    },
    /// 3D rayshader-style relief render to PNG (headless, no window)
    #[cfg(feature = "relief-3d")]
    #[command(name = "relief-3d")]
    Relief3d {
        /// Input DEM file
        input: PathBuf,
        /// Output PNG file
        output: PathBuf,
        /// Color scheme for the base layer
        #[arg(long, default_value = "terrain")]
        colormap: String,
        /// Output width in pixels
        #[arg(long, default_value = "1920")]
        width: u32,
        /// Output height in pixels
        #[arg(long, default_value = "1080")]
        height: u32,
        /// Sun azimuth in degrees (0 = North, clockwise)
        #[arg(long, default_value = "315")]
        sun_azimuth: f64,
        /// Sun altitude in degrees above the horizon
        #[arg(long, default_value = "45")]
        sun_altitude: f64,
        /// Enable ray-traced cast shadows in the 2D texture
        #[arg(long)]
        shadows: bool,
        /// Number of sun samples for soft-shadow penumbra (>=2)
        #[arg(long, default_value = "11")]
        soft: usize,
        /// Enable ambient occlusion in the 2D texture
        #[arg(long)]
        ambient: bool,
        /// Vertical exaggeration for the displaced mesh
        #[arg(long, default_value = "0.45")]
        vertical_exaggeration: f32,
        /// Camera azimuth (deg) around the DEM centre
        #[arg(long, default_value = "45")]
        camera_azimuth: f32,
        /// Camera polar angle (deg from straight-up)
        #[arg(long, default_value = "60")]
        camera_polar: f32,
        /// Camera distance (scene units; DEM longer side = 2)
        #[arg(long, default_value = "3.2")]
        camera_distance: f32,
        /// Atmospheric haze density (P3 M1). 0 = off, 0.3-0.6 typical.
        #[arg(long, default_value = "0.0")]
        haze: f32,
        /// P4-M4: render via the quadtree LOD pipeline. Bounds GPU
        /// memory to ~192 MB regardless of DEM size by streaming
        /// per-chunk per-LOD data to a fixed pool. Use on big DEMs
        /// (≥ 3 K side) where the default path would blow the
        /// adapter's single-buffer cap.
        #[arg(long, default_value = "false")]
        lod: bool,
    },
    /// Clip a raster by polygon or bounding box
    Clip {
        /// Input raster file
        input: PathBuf,
        /// Vector file with polygon(s) (.geojson, .shp, .gpkg)
        #[arg(long)]
        polygon: Option<PathBuf>,
        /// Bounding box: xmin,ymin,xmax,ymax (same CRS as input)
        #[arg(long)]
        bbox: Option<String>,
        /// Output file
        output: PathBuf,
    },
    /// Reproject a raster from one CRS to another (native, no GDAL dependency)
    #[cfg(feature = "projections")]
    Reproject {
        /// Input raster file (GeoTIFF)
        input: PathBuf,
        /// Output raster file (GeoTIFF)
        output: PathBuf,
        /// Target CRS (e.g. EPSG:32719 or just 32719)
        #[arg(long)]
        to: String,
        /// Source CRS override (e.g. EPSG:4326); defaults to value embedded in input
        #[arg(long)]
        from: Option<String>,
        /// Resampling method: nearest | bilinear (default)
        #[arg(long, default_value = "bilinear")]
        method: String,
        /// Output pixel size in target CRS units; defaults to a sensible
        /// auto-inferred value preserving approximate resolution
        #[arg(long)]
        pixel_size: Option<f64>,
    },
    /// Rasterize a vector file to a raster grid (.geojson, .shp, .gpkg)
    Rasterize {
        /// Input vector file (.geojson, .shp, .gpkg)
        input: PathBuf,
        /// Output raster file
        output: PathBuf,
        /// Reference raster for grid dimensions and transform
        #[arg(long)]
        reference: PathBuf,
        /// Property to use as raster value (default: sequential 1..N)
        #[arg(long)]
        attribute: Option<String>,
    },
    /// Resample a raster to match the grid of a reference raster
    Resample {
        /// Input raster to resample
        input: PathBuf,
        /// Output resampled raster
        output: PathBuf,
        /// Reference raster defining the target grid
        #[arg(long)]
        reference: PathBuf,
        /// Interpolation method: nearest or bilinear
        #[arg(short, long, default_value = "bilinear")]
        method: String,
    },
    /// Mosaic multiple rasters into one covering the union extent
    Mosaic {
        /// Input raster files (at least 2)
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,
        /// Output GeoTIFF file
        output: PathBuf,
    },
    /// Read and process Cloud Optimized GeoTIFFs (COGs) via HTTP
    #[cfg(feature = "cloud")]
    Cog {
        #[command(subcommand)]
        action: CogCommands,
    },
    /// Search and fetch data from STAC catalogs (Planetary Computer, Earth Search)
    #[cfg(feature = "cloud")]
    Stac {
        #[command(subcommand)]
        action: StacCommands,
    },
    /// Pipeline: integrated workflows for specific use cases (cloud feature only)
    #[cfg(feature = "cloud")]
    Pipeline {
        #[command(subcommand)]
        action: PipelineCommands,
    },
    /// Vector geoprocessing: intersection, union, difference, dissolve, buffer
    Vector {
        #[command(subcommand)]
        action: VectorCommands,
    },
    /// Geostatistical interpolation: variogram, kriging, universal kriging, regression kriging
    Interpolation {
        #[command(subcommand)]
        action: InterpolationCommands,
    },
    /// Temporal analysis: trend, anomaly, phenology, statistics
    Temporal {
        #[command(subcommand)]
        action: TemporalCommands,
    },
    /// Classification: unsupervised/supervised raster classification (k-means, PCA, etc.)
    Classification {
        #[command(subcommand)]
        algorithm: ClassificationCommands,
    },
    /// Texture analysis: edge detection and GLCM texture measures
    Texture {
        #[command(subcommand)]
        algorithm: TextureCommands,
    },
    /// Image segmentation: SLIC superpixels and Felzenszwalb-Huttenlocher
    Segmentation {
        #[command(subcommand)]
        algorithm: SegmentationCommands,
    },
    /// Statistics: focal, zonal, and spatial autocorrelation
    Statistics {
        #[command(subcommand)]
        algorithm: StatisticsCommands,
    },
    /// Machine learning: train, predict, and benchmark classifiers on geospatial features
    #[cfg(feature = "ml")]
    Ml {
        #[command(subcommand)]
        action: MlCommands,
    },
    /// Generate shell completion scripts (write to a file in your shell's
    /// completion directory, e.g. `surtgis completions bash > /etc/bash_completion.d/surtgis`)
    Completions {
        /// Target shell
        shell: clap_complete::Shell,
    },
}

// ─── Terrain subcommands ────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum TerrainCommands {
    /// Calculate slope from DEM
    Slope {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Output units: degrees, percent, radians
        #[arg(short, long, default_value = "degrees")]
        units: String,
        /// Vertical exaggeration: z' = z_factor * z (GDAL convention)
        #[arg(short, long, default_value = "1.0")]
        z_factor: f64,
    },
    /// Calculate aspect from DEM
    Aspect {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Output format: degrees, radians, compass
        #[arg(short, long, default_value = "degrees")]
        format: String,
    },
    /// Calculate hillshade from DEM
    Hillshade {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Sun azimuth in degrees (0=North, clockwise)
        #[arg(short, long, default_value = "315")]
        azimuth: f64,
        /// Sun altitude in degrees above horizon
        #[arg(short = 'l', long, default_value = "45")]
        altitude: f64,
        /// Vertical exaggeration: z' = z_factor * z (GDAL convention)
        #[arg(short, long, default_value = "1.0")]
        z_factor: f64,
    },
    /// Calculate surface curvature from DEM
    Curvature {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Curvature type: general, profile, plan
        #[arg(short = 't', long, default_value = "general")]
        curvature_type: String,
        /// Vertical exaggeration: z' = z_factor * z (GDAL convention)
        #[arg(short, long, default_value = "1.0")]
        z_factor: f64,
    },
    /// Calculate Topographic Position Index
    Tpi {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Calculate Terrain Ruggedness Index
    Tri {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Landform classification (multi-scale TPI + slope)
    Landform {
        /// Input DEM file
        input: PathBuf,
        /// Output file (class codes 1-11)
        output: PathBuf,
        /// Small-scale TPI radius
        #[arg(long, default_value = "3")]
        small_radius: usize,
        /// Large-scale TPI radius
        #[arg(long, default_value = "10")]
        large_radius: usize,
        /// Standardized TPI threshold (z-score)
        #[arg(long, default_value = "1.0")]
        threshold: f64,
        /// Slope threshold (degrees) for gentle/steep
        #[arg(long, default_value = "6.0")]
        slope_threshold: f64,
    },
    /// Geomorphon landform classification (Jasiewicz & Stepinski 2013)
    Geomorphons {
        input: PathBuf,
        output: PathBuf,
        /// Lookup radius in cells
        #[arg(short, long, default_value = "10")]
        radius: usize,
        /// Flatness threshold in degrees
        #[arg(short, long, default_value = "1.0")]
        flatness: f64,
    },
    /// Northness: cos(aspect), north-facing = 1, south-facing = -1
    Northness { input: PathBuf, output: PathBuf },
    /// Eastness: sin(aspect), east-facing = 1, west-facing = -1
    Eastness { input: PathBuf, output: PathBuf },
    /// Positive topographic openness (sky visibility above)
    OpennessPositive {
        input: PathBuf,
        output: PathBuf,
        /// Search radius in cells
        #[arg(short, long, default_value = "10")]
        radius: usize,
        /// Number of azimuth directions
        #[arg(short, long, default_value = "8")]
        directions: usize,
    },
    /// Negative topographic openness (enclosure below)
    OpennessNegative {
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value = "10")]
        radius: usize,
        #[arg(short, long, default_value = "8")]
        directions: usize,
    },
    /// Sky View Factor (0=enclosed, 1=flat horizon)
    Svf {
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value = "10")]
        radius: usize,
        #[arg(short, long, default_value = "16")]
        directions: usize,
    },
    /// MRVBF/MRRTF: Multi-Resolution Valley/Ridge Bottom Flatness
    Mrvbf {
        input: PathBuf,
        /// Output MRVBF file
        output: PathBuf,
        /// Optional MRRTF output file
        #[arg(long)]
        mrrtf_output: Option<PathBuf>,
    },
    /// Deviation from Mean Elevation
    Dev {
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value = "10")]
        radius: usize,
    },
    /// Vector Ruggedness Measure
    Vrm {
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Florinsky advanced curvature (14 types)
    AdvancedCurvature {
        input: PathBuf,
        output: PathBuf,
        /// Curvature type: mean_h|mean, gaussian_k|gaussian, kmin|minimal, kmax|maximal, kh|horizontal, kv|vertical, khe|horizontal_excess, kve|vertical_excess, ka|accumulation, kr|ring, rotor, laplacian, unsphericity, difference
        #[arg(short = 't', long, default_value = "mean_h")]
        curvature_type: String,
    },
    /// Viewshed: binary line-of-sight visibility from an observer point
    Viewshed {
        input: PathBuf,
        output: PathBuf,
        /// Observer row (pixel coordinate)
        #[arg(long)]
        observer_row: usize,
        /// Observer column (pixel coordinate)
        #[arg(long)]
        observer_col: usize,
        /// Observer height above ground (meters)
        #[arg(long, default_value = "1.8")]
        observer_height: f64,
        /// Target height above ground (meters)
        #[arg(long, default_value = "0.0")]
        target_height: f64,
        /// Maximum visibility radius in cells (0 = unlimited)
        #[arg(long, default_value = "0")]
        max_radius: usize,
    },
    /// Convergence Index (-100=convergent, +100=divergent)
    Convergence {
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value = "3")]
        radius: usize,
    },
    /// Multi-directional hillshade (6 azimuths combined)
    MultiHillshade { input: PathBuf, output: PathBuf },
    /// LS-Factor for RUSLE soil erosion model
    LsFactor {
        /// Flow accumulation raster
        #[arg(long)]
        flow_acc: PathBuf,
        /// Slope raster (radians)
        #[arg(long)]
        slope: PathBuf,
        /// Output file
        output: PathBuf,
        /// Cell size in meters
        #[arg(long, default_value = "1.0")]
        cell_size: f64,
    },
    /// Valley depth: vertical distance to ridge surface
    ValleyDepth { input: PathBuf, output: PathBuf },
    /// Relative Slope Position (0=valley, 1=ridge)
    RelativeSlopePosition {
        /// HAND raster
        #[arg(long)]
        hand: PathBuf,
        /// Valley depth raster
        #[arg(long)]
        valley_depth: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// Surface Area Ratio (3D/2D area roughness)
    SurfaceAreaRatio {
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Compute all standard terrain factors in one pass
    All {
        /// Input DEM file
        input: PathBuf,
        /// Output directory for all terrain products
        #[arg(short, long)]
        outdir: PathBuf,
    },
    /// Solar radiation (clear-sky insolation for a given day/hour)
    SolarRadiation {
        input: PathBuf,
        output: PathBuf,
        /// Day of year (1-365)
        #[arg(long)]
        day: u32,
        /// Hour (solar time, 0-24)
        #[arg(long)]
        hour: f64,
        /// Latitude in degrees (negative for southern hemisphere)
        #[arg(long, allow_hyphen_values = true)]
        latitude: f64,
    },
    /// Annual solar radiation (integrated over full year)
    SolarRadiationAnnual {
        input: PathBuf,
        output: PathBuf,
        /// Latitude in degrees (negative for southern hemisphere)
        #[arg(long, allow_hyphen_values = true)]
        latitude: f64,
    },
    /// Contour lines as raster
    ContourLines {
        input: PathBuf,
        output: PathBuf,
        /// Contour interval
        #[arg(long, default_value = "100")]
        interval: f64,
        /// Base contour value
        #[arg(long, default_value = "0")]
        base: f64,
    },
    /// Cost distance from source points
    CostDistance {
        /// Cost surface raster
        input: PathBuf,
        /// Source points raster (non-zero = source)
        #[arg(long)]
        sources: PathBuf,
        /// Output accumulated cost raster
        output: PathBuf,
    },
    /// Shape index (concavity/convexity, -1 to +1)
    ShapeIndex { input: PathBuf, output: PathBuf },
    /// Curvedness (magnitude of curvature)
    Curvedness { input: PathBuf, output: PathBuf },
    /// Gaussian smoothing
    GaussianSmoothing {
        input: PathBuf,
        output: PathBuf,
        /// Sigma (standard deviation in cells)
        #[arg(long, default_value = "1.0")]
        sigma: f64,
        /// Kernel radius in cells (default: ceil(3*sigma))
        #[arg(long)]
        radius: Option<usize>,
    },
    /// Feature-preserving smoothing (edge-aware)
    FeaturePreservingSmoothing {
        input: PathBuf,
        output: PathBuf,
        /// Smoothing strength
        #[arg(long, default_value = "1.0")]
        strength: f64,
        /// Number of iterations
        #[arg(long, default_value = "3")]
        iterations: usize,
    },
    /// Wind exposure index
    WindExposure {
        input: PathBuf,
        output: PathBuf,
        /// Wind direction in degrees from north (0=N, 90=E)
        #[arg(long, default_value = "270")]
        direction: f64,
        /// Search radius in cells
        #[arg(long, default_value = "100")]
        radius: usize,
    },
    /// Horizon angles for a given azimuth
    HorizonAngle {
        input: PathBuf,
        output: PathBuf,
        /// Azimuth in degrees from north
        #[arg(long, default_value = "180")]
        azimuth: f64,
        /// Search radius in cells
        #[arg(long, default_value = "100")]
        radius: usize,
    },
    /// Accumulation zones (contributing area classification)
    AccumulationZones { input: PathBuf, output: PathBuf },
    /// Stream Power Index (SPI = A × tan(slope))
    Spi {
        /// Flow accumulation raster
        #[arg(long)]
        flow_acc: PathBuf,
        /// Slope raster (radians)
        #[arg(long)]
        slope: PathBuf,
        output: PathBuf,
    },
    /// Sediment Transport Index (STI)
    Sti {
        /// Flow accumulation raster
        #[arg(long)]
        flow_acc: PathBuf,
        /// Slope raster (radians)
        #[arg(long)]
        slope: PathBuf,
        output: PathBuf,
    },
    /// Topographic Wetness Index (TWI = ln(A / tan(slope)))
    Twi {
        /// Flow accumulation raster
        #[arg(long)]
        flow_acc: PathBuf,
        /// Slope raster (radians)
        #[arg(long)]
        slope: PathBuf,
        output: PathBuf,
    },
    /// Log transform (ln(x+1))
    LogTransform { input: PathBuf, output: PathBuf },
    /// DEM uncertainty analysis (Monte Carlo)
    Uncertainty {
        input: PathBuf,
        /// Output directory (mean, std, etc.)
        #[arg(short, long)]
        outdir: PathBuf,
        /// Error standard deviation (meters)
        #[arg(long, default_value = "1.0")]
        error_std: f64,
        /// Number of simulations
        #[arg(long, default_value = "100")]
        n_simulations: usize,
    },
    /// PDERL viewshed (reference plane algorithm)
    ViewshedPderl {
        input: PathBuf,
        output: PathBuf,
        /// Observer row
        #[arg(long)]
        row: usize,
        /// Observer column
        #[arg(long)]
        col: usize,
        /// Observer height above ground
        #[arg(long, default_value = "1.7")]
        height: f64,
    },
    /// XDraw viewshed (approximate, fast)
    ViewshedXdraw {
        input: PathBuf,
        output: PathBuf,
        /// Observer row
        #[arg(long)]
        row: usize,
        /// Observer column
        #[arg(long)]
        col: usize,
        /// Observer height above ground
        #[arg(long, default_value = "1.7")]
        height: f64,
    },
    /// Multiple-observer cumulative viewshed
    ViewshedMultiple {
        input: PathBuf,
        output: PathBuf,
        /// Observer locations as row,col pairs (e.g. "10,20;30,40;50,60")
        #[arg(long)]
        observers: String,
        /// Observer height above ground
        #[arg(long, default_value = "1.7")]
        height: f64,
    },
    /// Hypsometrically tinted hillshade (hillshade × normalized elevation)
    HypsometricHillshade {
        input: PathBuf,
        output: PathBuf,
        /// Sun azimuth in degrees
        #[arg(short, long, default_value = "315")]
        azimuth: f64,
        /// Sun altitude in degrees
        #[arg(short = 'l', long, default_value = "45")]
        altitude: f64,
        /// Vertical exaggeration: z' = z_factor * z (GDAL convention)
        #[arg(short, long, default_value = "1.0")]
        z_factor: f64,
    },
    /// Elevation relative to global min/max (normalized 0–1)
    ElevRelative { input: PathBuf, output: PathBuf },
    /// Difference from mean elevation (non-normalized, in DEM units)
    DiffFromMean {
        input: PathBuf,
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "10")]
        radius: usize,
    },
    /// Percent elevation range (local position 0–100%)
    PercentElevRange {
        input: PathBuf,
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "10")]
        radius: usize,
    },
    /// Elevation above pit / depth in sink
    ElevAbovePit { input: PathBuf, output: PathBuf },
    /// Circular variance of aspect (0=uniform, 1=dispersed)
    CircularVarianceAspect {
        input: PathBuf,
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "3")]
        radius: usize,
    },
    /// Neighbour elevation statistics (3×3 window, 5 outputs)
    Neighbours {
        input: PathBuf,
        /// Output path: directory (all 5 stats) or file (with --stat)
        #[arg(short, long)]
        output: PathBuf,
        /// Single statistic: max_downslope_change, min_downslope_change, max_upslope_change, num_downslope, num_upslope
        #[arg(long)]
        stat: Option<String>,
    },
    /// Pennock landform classification (7 classes)
    Pennock {
        input: PathBuf,
        output: PathBuf,
        /// Slope threshold (degrees) for "level" class
        #[arg(long, default_value = "3.0")]
        slope_threshold: f64,
        /// Profile curvature threshold for linear vs convex/concave
        #[arg(long, default_value = "0.1")]
        curv_threshold: f64,
    },
    /// Edge density (proportion of edge pixels in focal window)
    EdgeDensity {
        input: PathBuf,
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "3")]
        radius: usize,
        /// Sobel magnitude threshold for edge detection
        #[arg(short, long, default_value = "0.5")]
        threshold: f64,
    },
    /// Relative aspect (local vs regional aspect difference, 0–180°)
    RelativeAspect {
        input: PathBuf,
        output: PathBuf,
        /// Gaussian sigma for regional smoothing
        #[arg(long, default_value = "50")]
        sigma: f64,
    },
    /// Average normal vector angular deviation (degrees)
    NormalDeviation {
        input: PathBuf,
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "3")]
        radius: usize,
    },
    /// Spherical standard deviation of surface normals
    SphericalStdDev {
        input: PathBuf,
        output: PathBuf,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "3")]
        radius: usize,
    },
    /// Directional relief (elevation range along azimuth)
    DirectionalRelief {
        input: PathBuf,
        output: PathBuf,
        /// Search radius in cells
        #[arg(short, long, default_value = "10")]
        radius: usize,
        /// Azimuth in degrees (0 = multidirectional average)
        #[arg(short, long, default_value = "0")]
        azimuth: f64,
    },
    /// Downslope index (distance to reach elevation drop, Hjerdt 2004)
    DownslopeIndex {
        input: PathBuf,
        output: PathBuf,
        /// Elevation drop threshold in DEM units
        #[arg(short, long, default_value = "2.0")]
        drop: f64,
    },
    /// Maximum upstream branch length (longest D8 flow path)
    MaxBranchLength { input: PathBuf, output: PathBuf },
    /// Excess topography above a threshold hillslope angle (Blöthe et al. 2015)
    ExcessTopography {
        /// Input DEM file
        input: PathBuf,
        /// Output excess topography raster (z − slope-limited surface)
        output: PathBuf,
        /// Threshold hillslope angle in degrees (0 < θ < 90)
        #[arg(short, long, default_value = "30.0")]
        threshold: f64,
        /// Maximum fast-sweeping rounds
        #[arg(long, default_value = "200")]
        max_iterations: usize,
    },
}

// ─── Hydrology subcommands ──────────────────────────────────────────────

#[derive(Subcommand)]
pub enum HydrologyCommands {
    /// Fill sinks / depressions in DEM (Planchon-Darboux 2001)
    FillSinks {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Minimum slope to enforce
        #[arg(long, default_value = "0.01")]
        min_slope: f64,
    },
    /// D8 flow direction from DEM
    FlowDirection {
        /// Input DEM file
        input: PathBuf,
        /// Output file (D8 codes: 1,2,4,8,16,32,64,128)
        output: PathBuf,
    },
    /// Flow accumulation from flow direction raster
    FlowAccumulation {
        /// Input flow direction raster (D8 codes)
        input: PathBuf,
        /// Output file (upstream cell count)
        output: PathBuf,
    },
    /// Watershed delineation from flow direction
    Watershed {
        /// Input flow direction raster (D8 codes)
        input: PathBuf,
        /// Output file (basin IDs)
        output: PathBuf,
        /// Pour points as "row,col;row,col;..."
        #[arg(long)]
        pour_points: String,
    },
    /// Priority-Flood depression filling (Barnes 2014, optimal O(n log n))
    PriorityFlood {
        input: PathBuf,
        output: PathBuf,
        /// Minimum slope epsilon
        #[arg(long, default_value = "0.0001")]
        epsilon: f64,
    },
    /// Breach depressions (carve channels through barriers)
    Breach {
        input: PathBuf,
        output: PathBuf,
        /// Maximum breach depth (meters)
        #[arg(long, default_value = "100.0")]
        max_depth: f64,
        /// Maximum breach length (cells)
        #[arg(long, default_value = "1000")]
        max_length: usize,
        /// Fill remaining unfilled depressions
        #[arg(long)]
        fill_remaining: bool,
    },
    /// D-infinity flow direction (Tarboton 1997, continuous angles)
    FlowDirectionDinf { input: PathBuf, output: PathBuf },
    /// D-infinity flow accumulation from a D-inf angle raster
    FlowAccumulationDinf {
        /// Input D-inf angle raster (output of flow-direction-dinf)
        input: PathBuf,
        output: PathBuf,
    },
    /// Multiple Flow Direction accumulation (Quinn et al. 1991)
    FlowAccumulationMfd {
        input: PathBuf,
        output: PathBuf,
        /// Flow partition exponent
        #[arg(long, default_value = "1.1")]
        exponent: f64,
    },
    /// Topographic Wetness Index (from DEM, full pipeline)
    Twi {
        /// Input DEM file
        input: PathBuf,
        output: PathBuf,
        /// Flow routing method for the accumulation step
        #[arg(long, default_value = "d8", value_parser = ["d8", "dinf", "mfd"])]
        method: String,
    },
    /// Height Above Nearest Drainage (from DEM, full pipeline)
    Hand {
        /// Input DEM file
        input: PathBuf,
        output: PathBuf,
        /// Stream extraction threshold (contributing cells)
        #[arg(long, default_value = "1000")]
        threshold: f64,
    },
    /// Stream network extraction.
    ///
    /// By default the input is treated as a DEM and the handler runs the
    /// full pipeline (priority_flood → flow_direction → flow_accumulation
    /// → threshold). When `--from-facc` is passed, the input is treated
    /// as a pre-computed flow_accumulation raster and only the threshold
    /// step runs — this is the path you want when composing with an
    /// externally computed flow direction / accumulation, e.g. for the
    /// fluvial module's chi/ksn workflow.
    StreamNetwork {
        /// Input raster. DEM by default; flow_accumulation when `--from-facc`.
        input: PathBuf,
        output: PathBuf,
        /// Contributing area threshold (cell counts).
        #[arg(long, default_value = "1000")]
        threshold: f64,
        /// Treat `input` as a pre-computed flow_accumulation raster and
        /// skip the DEM → fdir → facc recomputation. Use when you already
        /// have flow_dir/acc and want the resulting `stream-network` to
        /// be topologically consistent with them.
        #[arg(long)]
        from_facc: bool,
    },
    /// Drainage density: stream length per unit area
    DrainageDensity {
        /// Stream network raster (binary: 1=stream)
        input: PathBuf,
        output: PathBuf,
        #[arg(short, long, default_value = "10")]
        radius: usize,
        #[arg(long, default_value = "1.0")]
        cell_size: f64,
    },
    /// Hypsometric integral per watershed
    HypsometricIntegral {
        /// DEM file
        #[arg(long)]
        dem: PathBuf,
        /// Watershed raster (i32 IDs)
        #[arg(long)]
        watersheds: PathBuf,
    },
    /// Sediment Connectivity Index (Borselli 2008)
    SedimentConnectivity {
        /// Slope raster (radians)
        #[arg(long)]
        slope: PathBuf,
        /// Flow accumulation raster
        #[arg(long)]
        flow_acc: PathBuf,
        /// D8 flow direction raster
        #[arg(long)]
        flow_dir: PathBuf,
        /// Output file
        output: PathBuf,
        /// Stream threshold (flow accumulation cells)
        #[arg(long, default_value = "1000")]
        threshold: f64,
    },
    /// Basin morphometric parameters per watershed
    BasinMorphometry {
        /// Watershed raster (i32 IDs)
        input: PathBuf,
        #[arg(long, default_value = "1.0")]
        cell_size: f64,
    },
    /// Melton ruggedness ratio per watershed (debris-flow / lahar screening)
    Melton {
        /// Watershed raster (i32 IDs)
        input: PathBuf,
        /// DEM raster aligned with the watershed raster
        #[arg(long)]
        dem: PathBuf,
        #[arg(long, default_value = "1.0")]
        cell_size: f64,
        /// Optional output raster mapping each basin to its Melton ratio
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Energy-cone lahar / mass-flow inundation (Malin & Sheridan 1982)
    EnergyCone {
        /// Input DEM file
        input: PathBuf,
        /// Output energy-height-above-ground raster (>0 = reached)
        output: PathBuf,
        /// Source cell(s) as "row,col" (multiple separated by ';')
        #[arg(long)]
        source: String,
        /// Energy-cone angle φ in degrees (H/L = tan φ); smaller = more mobile
        #[arg(long, default_value = "10.0")]
        cone_angle: f64,
        /// Collapse height added to the source elevation to set the apex
        #[arg(long, default_value = "0.0")]
        collapse_height: f64,
    },
    /// LAHARZ lahar / debris-flow inundation (Iverson, Schilling & Vallance 1998)
    Laharz {
        /// Input DEM file
        input: PathBuf,
        /// D8 flow-direction raster (u8, from `hydrology flow-direction`)
        #[arg(long)]
        flow_dir: PathBuf,
        /// Output inundation-depth raster (>0 = inundated)
        output: PathBuf,
        /// Source cell(s) as "row,col" (multiple separated by ';'). Seed
        /// proximal CHANNEL cells, not the summit — a summit cell's D8 descent
        /// often runs down the wrong drainage. Sources route independently and
        /// the footprint is their union.
        #[arg(long)]
        source: String,
        /// Flow volume in m³ (applied to each source)
        #[arg(long)]
        volume: f64,
        /// Flow type preset: lahar | debris-flow | rock-avalanche
        #[arg(long, default_value = "lahar")]
        flow_type: String,
        /// Override the lateral-spread aspect ratio (width:depth). Omit to use
        /// the preset default; 0 = canonical fill-to-area-A (long thin ribbons).
        #[arg(long)]
        spread_aspect: Option<f64>,
    },
    /// Compute full hydrology pipeline from DEM
    All {
        /// Input DEM file
        input: PathBuf,
        /// Output directory
        #[arg(short, long)]
        outdir: PathBuf,
        /// Stream threshold
        #[arg(long, default_value = "1000")]
        threshold: f64,
    },
}

// ─── Imagery subcommands ────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum ImageryCommands {
    /// NDVI: Normalized Difference Vegetation Index
    Ndvi {
        /// NIR band file
        #[arg(long)]
        nir: PathBuf,
        /// Red band file
        #[arg(long)]
        red: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// NDWI: Normalized Difference Water Index
    Ndwi {
        /// Green band file
        #[arg(long)]
        green: PathBuf,
        /// NIR band file
        #[arg(long)]
        nir: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// MNDWI: Modified Normalized Difference Water Index
    Mndwi {
        /// Green band file
        #[arg(long)]
        green: PathBuf,
        /// SWIR band file
        #[arg(long)]
        swir: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// NBR: Normalized Burn Ratio
    Nbr {
        /// NIR band file
        #[arg(long)]
        nir: PathBuf,
        /// SWIR band file
        #[arg(long)]
        swir: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// SAVI: Soil-Adjusted Vegetation Index
    Savi {
        /// NIR band file
        #[arg(long)]
        nir: PathBuf,
        /// Red band file
        #[arg(long)]
        red: PathBuf,
        /// Output file
        output: PathBuf,
        /// Soil brightness correction factor (0..1)
        #[arg(short, long, default_value = "0.5")]
        l_factor: f64,
    },
    /// EVI: Enhanced Vegetation Index
    Evi {
        /// NIR band file
        #[arg(long)]
        nir: PathBuf,
        /// Red band file
        #[arg(long)]
        red: PathBuf,
        /// Blue band file
        #[arg(long)]
        blue: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// BSI: Bare Soil Index
    Bsi {
        /// SWIR band file
        #[arg(long)]
        swir: PathBuf,
        /// Red band file
        #[arg(long)]
        red: PathBuf,
        /// NIR band file
        #[arg(long)]
        nir: PathBuf,
        /// Blue band file
        #[arg(long)]
        blue: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// Band math: arithmetic between two raster bands
    BandMath {
        /// First input raster
        #[arg(short)]
        a: PathBuf,
        /// Second input raster
        #[arg(short)]
        b: PathBuf,
        /// Output file
        output: PathBuf,
        /// Operation: add, subtract, multiply, divide, power, min, max
        #[arg(long)]
        op: String,
    },
    /// Compute custom spectral index from arithmetic expression
    ///
    /// Common formulas:
    ///   NDVI:  "(NIR - Red) / (NIR + Red)"
    ///   EXG:   "2 * Green - Red - Blue"
    ///   VARI:  "(Green - Red) / (Green + Red - Blue)"
    ///   Clay:  "SWIR1 / SWIR2"
    Calc {
        /// Arithmetic expression using band names
        #[arg(short, long)]
        expression: String,
        /// Band assignments as NAME=path (repeatable)
        #[arg(short, long, value_name = "NAME=FILE")]
        band: Vec<String>,
        /// Output file
        output: PathBuf,
    },
    /// EVI2: Two-band Enhanced Vegetation Index
    Evi2 {
        #[arg(long)]
        nir: PathBuf,
        #[arg(long)]
        red: PathBuf,
        output: PathBuf,
    },
    /// GNDVI: Green Normalized Difference Vegetation Index
    Gndvi {
        #[arg(long)]
        nir: PathBuf,
        #[arg(long)]
        green: PathBuf,
        output: PathBuf,
    },
    /// NGRDI: Normalized Green-Red Difference Index
    Ngrdi {
        #[arg(long)]
        green: PathBuf,
        #[arg(long)]
        red: PathBuf,
        output: PathBuf,
    },
    /// RECI: Red Edge Chlorophyll Index
    Reci {
        #[arg(long)]
        nir: PathBuf,
        #[arg(long)]
        red_edge: PathBuf,
        output: PathBuf,
    },
    /// NDRE: Normalized Difference Red Edge Index
    Ndre {
        #[arg(long)]
        nir: PathBuf,
        #[arg(long)]
        red_edge: PathBuf,
        output: PathBuf,
    },
    /// NDSI: Normalized Difference Snow Index
    Ndsi {
        #[arg(long)]
        green: PathBuf,
        #[arg(long)]
        swir: PathBuf,
        output: PathBuf,
    },
    /// NDMI: Normalized Difference Moisture Index
    Ndmi {
        #[arg(long)]
        nir: PathBuf,
        #[arg(long)]
        swir: PathBuf,
        output: PathBuf,
    },
    /// NDBI: Normalized Difference Built-up Index
    Ndbi {
        #[arg(long)]
        swir: PathBuf,
        #[arg(long)]
        nir: PathBuf,
        output: PathBuf,
    },
    /// MSAVI: Modified Soil-Adjusted Vegetation Index
    Msavi {
        #[arg(long)]
        nir: PathBuf,
        #[arg(long)]
        red: PathBuf,
        output: PathBuf,
    },
    /// Reclassify raster values into discrete classes
    Reclassify {
        input: PathBuf,
        output: PathBuf,
        /// Class definition as "min,max,value" (repeatable)
        #[arg(long, value_name = "MIN,MAX,VALUE")]
        class: Vec<String>,
        /// Default value for unclassified cells
        #[arg(long, default_value = "NaN")]
        default: String,
    },
    /// Per-pixel median composite across multiple rasters
    MedianComposite {
        /// Input raster files (at least 2, repeatable)
        #[arg(short, long)]
        input: Vec<PathBuf>,
        /// Output file
        output: PathBuf,
    },
    /// dNBR: differenced Normalized Burn Ratio (pre/post fire)
    Dnbr {
        #[arg(long)]
        pre_nir: PathBuf,
        #[arg(long)]
        pre_swir: PathBuf,
        #[arg(long)]
        post_nir: PathBuf,
        #[arg(long)]
        post_swir: PathBuf,
        output: PathBuf,
        /// Also output burn severity classification
        #[arg(long)]
        severity_output: Option<PathBuf>,
    },
    /// Cloud mask using Sentinel-2 SCL band
    CloudMask {
        /// Input raster to mask
        input: PathBuf,
        /// SCL classification raster
        #[arg(long)]
        scl: PathBuf,
        /// Output file
        output: PathBuf,
        /// SCL classes to keep (comma-separated, default: 4,5,6,11)
        #[arg(long, default_value = "4,5,6,11")]
        keep: String,
    },
    /// Radiometric calibration: DN → TOA reflectance, surface reflectance, DOS1
    Calibrate {
        #[command(subcommand)]
        action: CalibrateCommands,
    },
    /// Pansharpening: fuse high-res pan with low-res multispectral
    Pansharpen {
        #[command(subcommand)]
        action: PansharpenCommands,
    },
    /// Stack 1/3/4 single-band rasters into a single multi-band GeoTIFF
    Stack {
        /// Output multi-band GeoTIFF
        output: PathBuf,
        /// One or more input single-band rasters (repeatable). Order
        /// determines band order in the output — pass R, G, B for an
        /// RGB stack.
        #[arg(long = "band", required = true)]
        bands: Vec<PathBuf>,
    },
    /// Bitemporal change detection: MAD / IR-MAD (Nielsen 1998, 2007)
    ChangeDetection {
        #[command(subcommand)]
        action: ChangeDetectionCommands,
    },
    /// Inter-tile colour balancing: histogram or moment matching
    ColorBalance {
        #[command(subcommand)]
        action: ColorBalanceCommands,
    },
    /// Distance-weighted feather-blend mosaic of aligned rasters
    MosaicFeather {
        /// Output blended raster
        output: PathBuf,
        /// Input rasters (repeatable)
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,
    },
    /// SAR: convert linear-power backscatter (σ⁰) to decibels (10·log₁₀)
    SarDb {
        /// Input linear-power backscatter raster
        input: PathBuf,
        /// Output dB raster
        output: PathBuf,
    },
    /// SAR: dual-pol water index (VV − VH)/(VV + VH)
    SarWaterIndex {
        /// Co-pol band (e.g. VV), linear power
        #[arg(long)]
        co_pol: PathBuf,
        /// Cross-pol band (e.g. VH), linear power
        #[arg(long)]
        cross_pol: PathBuf,
        /// Output index raster
        output: PathBuf,
    },
    /// SAR: threshold backscatter/index into a binary water mask
    SarWaterMask {
        /// Input backscatter (e.g. VV in dB) or water index
        input: PathBuf,
        /// Output u8 water mask (1=water, 0=land, 255=nodata)
        output: PathBuf,
        /// Decision threshold
        #[arg(long)]
        threshold: f64,
        /// Water is ABOVE the threshold (use for a water index). Default:
        /// water is below the threshold (use for backscatter).
        #[arg(long)]
        water_above: bool,
    },
    /// SAR: Lee adaptive speckle filter (Lee 1980), linear-power input
    SarLee {
        /// Input backscatter (linear power)
        input: PathBuf,
        /// Output filtered raster
        output: PathBuf,
        /// Odd window side length
        #[arg(long, default_value = "7")]
        window_size: usize,
        /// Equivalent number of looks (ENL); 1.0 for single-look
        #[arg(long, default_value = "1.0")]
        looks: f64,
        /// Use the edge-aligned refined Lee (1981) instead of the classic Lee
        #[arg(long)]
        refined: bool,
    },
}

// ─── Colour balance subcommands ────────────────────────────────────────

#[derive(Subcommand)]
pub enum ColorBalanceCommands {
    /// Histogram (empirical CDF) matching — full distribution alignment
    Histogram {
        /// Source raster to be remapped
        source: PathBuf,
        /// Reference raster whose distribution shape is the target
        reference: PathBuf,
        /// Output remapped raster
        output: PathBuf,
    },
    /// Linear moment (mean, stddev) matching
    Moments {
        /// Source raster to be remapped
        source: PathBuf,
        /// Reference raster whose (μ, σ) are the target
        reference: PathBuf,
        /// Output remapped raster
        output: PathBuf,
    },
}

// ─── Change detection (MAD / IR-MAD) subcommands ───────────────────────

#[derive(Subcommand)]
pub enum ChangeDetectionCommands {
    /// One-shot MAD: canonical correlation analysis between two timestamps
    Mad {
        /// Output directory (one .tif per MAD variate)
        #[arg(long)]
        output_dir: PathBuf,
        /// Band rasters at time 1 (repeat for multi-band)
        #[arg(long = "t1", required = true)]
        t1: Vec<PathBuf>,
        /// Band rasters at time 2 (must match t1 in count and shape)
        #[arg(long = "t2", required = true)]
        t2: Vec<PathBuf>,
        /// Output filename prefix
        #[arg(long, default_value = "mad")]
        prefix: String,
    },
    /// IR-MAD: iteratively reweighted MAD with chi-square weights
    IrMad {
        /// Output directory (one .tif per MAD variate + weights)
        #[arg(long)]
        output_dir: PathBuf,
        /// Band rasters at time 1 (repeat for multi-band)
        #[arg(long = "t1", required = true)]
        t1: Vec<PathBuf>,
        /// Band rasters at time 2 (must match t1 in count and shape)
        #[arg(long = "t2", required = true)]
        t2: Vec<PathBuf>,
        /// Maximum iterations
        #[arg(long, default_value = "25")]
        max_iter: usize,
        /// Convergence tolerance on max canonical-correlation change
        #[arg(long, default_value = "0.001")]
        tol: f64,
        /// Diagonal regularisation factor (fraction of per-band variance)
        #[arg(long, default_value = "0.0")]
        regularisation: f64,
        /// Output filename prefix
        #[arg(long, default_value = "irmad")]
        prefix: String,
    },
}

// ─── Pansharpen subcommands ────────────────────────────────────────────

#[derive(Subcommand)]
pub enum PansharpenCommands {
    /// Brovey transform (Gillespie 1987) — simplest, fastest
    Brovey {
        /// Panchromatic raster (high resolution)
        #[arg(long)]
        pan: PathBuf,
        /// One or more upsampled MS band rasters (repeatable)
        #[arg(long = "band", required = true)]
        bands: Vec<PathBuf>,
        /// Output directory — one .tif per band is written here
        #[arg(long)]
        output_dir: PathBuf,
        /// Output filename prefix (default: pansharp)
        #[arg(long, default_value = "pansharp")]
        prefix: String,
    },
    /// PCA pansharpening (Chavez 1991) — works for any band count
    Pca {
        /// Panchromatic raster
        #[arg(long)]
        pan: PathBuf,
        /// One or more upsampled MS band rasters
        #[arg(long = "band", required = true)]
        bands: Vec<PathBuf>,
        /// Output directory
        #[arg(long)]
        output_dir: PathBuf,
        /// Output filename prefix
        #[arg(long, default_value = "pansharp")]
        prefix: String,
    },
    /// Gram-Schmidt pansharpening (Laben & Brower 2000; patent expired 2018)
    GramSchmidt {
        /// Panchromatic raster
        #[arg(long)]
        pan: PathBuf,
        /// One or more upsampled MS band rasters
        #[arg(long = "band", required = true)]
        bands: Vec<PathBuf>,
        /// Output directory
        #[arg(long)]
        output_dir: PathBuf,
        /// Output filename prefix
        #[arg(long, default_value = "pansharp")]
        prefix: String,
    },
}

// ─── Calibrate subcommands ─────────────────────────────────────────────

#[derive(Subcommand)]
pub enum CalibrateCommands {
    /// Landsat 8/9 Collection 2 Level-1: DN → TOA reflectance
    LandsatToa {
        /// Input DN raster
        input: PathBuf,
        /// Output reflectance raster
        output: PathBuf,
        /// REFLECTANCE_MULT_BAND_x from MTL (e.g. 2e-5)
        #[arg(short = 'M', long = "mult")]
        multiplicative: f64,
        /// REFLECTANCE_ADD_BAND_x from MTL (e.g. -0.1)
        #[arg(short = 'A', long = "add")]
        additive: f64,
        /// SUN_ELEVATION from MTL, in degrees
        #[arg(long)]
        sun_elevation: f64,
    },
    /// Landsat Collection 2 Level-2: DN → surface reflectance (fixed 2.75e-5/-0.2)
    LandsatSrC2 {
        /// Input DN raster
        input: PathBuf,
        /// Output reflectance raster
        output: PathBuf,
    },
    /// Sentinel-2: DN → reflectance (works for L1C TOA and L2A BOA)
    S2 {
        /// Input DN raster
        input: PathBuf,
        /// Output reflectance raster
        output: PathBuf,
        /// QUANTIFICATION_VALUE / BOA_QUANTIFICATION_VALUE (default 10000)
        #[arg(long, default_value = "10000.0")]
        quantification: f64,
        /// BOA_ADD_OFFSET (PSD baseline 04.00+: -1000; older products: 0)
        #[arg(long, default_value = "0.0")]
        offset: f64,
    },
    /// DOS1 dark-object subtraction (per band)
    Dos1 {
        /// Input raster
        input: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Quantile of finite cells used as dark-object estimate (default 0.001)
        #[arg(long, default_value = "0.001")]
        quantile: f64,
    },
}

// ─── Morphology subcommands ─────────────────────────────────────────────

#[derive(Subcommand)]
pub enum MorphologyCommands {
    /// Erosion (minimum filter)
    Erode {
        /// Input raster
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Structuring element shape: square, cross, disk
        #[arg(long, default_value = "square")]
        shape: String,
        /// Structuring element radius in cells
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Dilation (maximum filter)
    Dilate {
        /// Input raster
        input: PathBuf,
        /// Output file
        output: PathBuf,
        #[arg(long, default_value = "square")]
        shape: String,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Opening (erosion then dilation) — removes small bright features
    Opening {
        /// Input raster
        input: PathBuf,
        /// Output file
        output: PathBuf,
        #[arg(long, default_value = "square")]
        shape: String,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Closing (dilation then erosion) — removes small dark features
    Closing {
        /// Input raster
        input: PathBuf,
        /// Output file
        output: PathBuf,
        #[arg(long, default_value = "square")]
        shape: String,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Morphological gradient (dilation - erosion) — edge detection
    Gradient {
        /// Input raster
        input: PathBuf,
        /// Output file
        output: PathBuf,
        #[arg(long, default_value = "square")]
        shape: String,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Top-hat transform (original - opening) — bright feature extraction
    TopHat {
        /// Input raster
        input: PathBuf,
        /// Output file
        output: PathBuf,
        #[arg(long, default_value = "square")]
        shape: String,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Black-hat transform (closing - original) — dark feature extraction
    BlackHat {
        /// Input raster
        input: PathBuf,
        /// Output file
        output: PathBuf,
        #[arg(long, default_value = "square")]
        shape: String,
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
}

// ─── Landscape subcommands ─────────────────────────────────────────────

#[derive(Subcommand)]
pub enum LandscapeCommands {
    /// Label connected patches in a classification raster
    LabelPatches {
        /// Input classification raster (integer class values)
        input: PathBuf,
        /// Output labeled patch raster (i32 IDs)
        output: PathBuf,
        /// Connectivity: 4 (cardinal) or 8 (cardinal+diagonal)
        #[arg(short, long, default_value = "4")]
        connectivity: u8,
    },
    /// Compute per-patch metrics (PARA, FRAC, area, perimeter) as CSV
    PatchMetrics {
        /// Input classification raster
        input: PathBuf,
        /// Output CSV file with per-patch metrics
        #[arg(short, long)]
        output: PathBuf,
        /// Connectivity: 4 or 8
        #[arg(short, long, default_value = "4")]
        connectivity: u8,
    },
    /// Compute per-class metrics (AI, COHESION, proportion)
    ClassMetrics {
        /// Input classification raster
        input: PathBuf,
        /// Connectivity: 4 or 8
        #[arg(short, long, default_value = "4")]
        connectivity: u8,
    },
    /// Compute global landscape metrics (SHDI, SIDI)
    LandscapeMetrics {
        /// Input classification raster
        input: PathBuf,
    },
    /// Full landscape analysis: label + patch metrics + class metrics + landscape metrics
    Analyze {
        /// Input classification raster
        input: PathBuf,
        /// Output labeled raster (optional)
        #[arg(long)]
        output_labels: Option<PathBuf>,
        /// Output CSV with per-patch metrics (optional)
        #[arg(long)]
        output_csv: Option<PathBuf>,
        /// Connectivity: 4 or 8
        #[arg(short, long, default_value = "4")]
        connectivity: u8,
    },
}

// ─── COG subcommands ──────────────────────────────────────────────────

#[cfg(feature = "cloud")]
#[derive(Subcommand)]
pub enum CogCommands {
    /// Show metadata of a remote COG
    Info {
        /// URL of the COG file
        url: String,
    },
    /// Read a bounding box from a remote COG and save to local file
    Fetch {
        /// URL of the COG file
        url: String,
        /// Output GeoTIFF file
        output: PathBuf,
        /// Bounding box: min_x,min_y,max_x,max_y
        #[arg(long)]
        bbox: String,
        /// Overview level (0 = full resolution)
        #[arg(long)]
        overview: Option<usize>,
    },
    /// Calculate slope from a remote COG DEM
    Slope {
        /// URL of the COG DEM
        url: String,
        /// Output file
        output: PathBuf,
        /// Bounding box: min_x,min_y,max_x,max_y
        #[arg(long)]
        bbox: String,
        /// Output units: degrees, percent, radians
        #[arg(short, long, default_value = "degrees")]
        units: String,
        /// Vertical exaggeration: z' = z_factor * z (GDAL convention)
        #[arg(short, long, default_value = "1.0")]
        z_factor: f64,
    },
    /// Calculate aspect from a remote COG DEM
    Aspect {
        /// URL of the COG DEM
        url: String,
        /// Output file
        output: PathBuf,
        /// Bounding box: min_x,min_y,max_x,max_y
        #[arg(long)]
        bbox: String,
        /// Output format: degrees, radians, compass
        #[arg(short, long, default_value = "degrees")]
        format: String,
    },
    /// Calculate hillshade from a remote COG DEM
    Hillshade {
        /// URL of the COG DEM
        url: String,
        /// Output file
        output: PathBuf,
        /// Bounding box: min_x,min_y,max_x,max_y
        #[arg(long)]
        bbox: String,
        /// Sun azimuth in degrees (0=North, clockwise)
        #[arg(short, long, default_value = "315")]
        azimuth: f64,
        /// Sun altitude in degrees above horizon
        #[arg(short = 'l', long, default_value = "45")]
        altitude: f64,
        /// Vertical exaggeration: z' = z_factor * z (GDAL convention)
        #[arg(short, long, default_value = "1.0")]
        z_factor: f64,
    },
    /// Calculate TPI from a remote COG DEM
    Tpi {
        /// URL of the COG DEM
        url: String,
        /// Output file
        output: PathBuf,
        /// Bounding box: min_x,min_y,max_x,max_y
        #[arg(long)]
        bbox: String,
        /// Neighborhood radius in cells
        #[arg(short, long, default_value = "1")]
        radius: usize,
    },
    /// Fill sinks from a remote COG DEM
    FillSinks {
        /// URL of the COG DEM
        url: String,
        /// Output file
        output: PathBuf,
        /// Bounding box: min_x,min_y,max_x,max_y
        #[arg(long)]
        bbox: String,
        /// Minimum slope to enforce
        #[arg(long, default_value = "0.01")]
        min_slope: f64,
    },
}

// ─── STAC subcommands ────────────────────────────────────────────────

#[cfg(feature = "cloud")]
#[derive(Subcommand)]
pub enum StacCommands {
    /// Search a STAC catalog and list matching items
    Search {
        /// Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "es")]
        catalog: String,
        /// Bounding box: west,south,east,north
        #[arg(long)]
        bbox: Option<String>,
        /// Datetime or range (e.g. "2024-06-01/2024-06-30")
        #[arg(long)]
        datetime: Option<String>,
        /// Collections (comma-separated, e.g. "sentinel-2-l2a")
        #[arg(long, visible_alias = "collection")]
        collections: Option<String>,
        /// Maximum items to return
        #[arg(long, default_value = "10")]
        limit: u32,
    },
    /// Search a STAC catalog, fetch the first COG asset, and save to a GeoTIFF
    Fetch {
        /// Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "es")]
        catalog: String,
        /// Bounding box: west,south,east,north
        #[arg(long)]
        bbox: String,
        /// Collection (e.g. "sentinel-2-l2a")
        #[arg(long, visible_alias = "collections")]
        collection: String,
        /// Asset key to fetch (e.g. "red", "nir", "B04"). Auto-detects COG if omitted.
        #[arg(long)]
        asset: Option<String>,
        /// Datetime or range
        #[arg(long)]
        datetime: Option<String>,
        /// Variable name for Zarr stores (e.g. "precipitation_amount_1hour_Accumulation")
        #[arg(long)]
        variable: Option<String>,
        /// Time step for Zarr: "first", "last", or ISO datetime (e.g. "2020-06-15")
        #[arg(long, default_value = "first")]
        time_step: String,
        /// Output GeoTIFF file
        output: PathBuf,
    },
    /// Search STAC catalog, fetch ALL matching COG assets, mosaic, and save
    FetchMosaic {
        /// Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "es")]
        catalog: String,
        /// Bounding box: west,south,east,north
        #[arg(long)]
        bbox: String,
        /// Collection (e.g. "cop-dem-glo-30", "sentinel-2-l2a")
        #[arg(long, visible_alias = "collections")]
        collection: String,
        /// Asset key to fetch (e.g. "data", "red", "B04"). Auto-detects COG if omitted.
        #[arg(long)]
        asset: Option<String>,
        /// Datetime or range
        #[arg(long)]
        datetime: Option<String>,
        /// Maximum items to fetch and mosaic (caps RAM/time on wide queries)
        #[arg(long, default_value = "20", visible_alias = "max-scenes")]
        max_items: u32,
        /// Output GeoTIFF file
        output: PathBuf,
    },
    /// End-to-end satellite composite: search -> mosaic per date -> cloud-mask -> median composite
    Composite {
        /// Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "es")]
        catalog: String,
        /// Bounding box: west,south,east,north
        #[arg(long)]
        bbox: String,
        /// Collection (e.g. "sentinel-2-l2a")
        #[arg(long)]
        collection: String,
        /// Data asset to composite (e.g. "red", "nir", "B04")
        #[arg(long)]
        asset: String,
        /// Datetime range (e.g. "2024-01-01/2024-12-31")
        #[arg(long)]
        datetime: String,
        /// Maximum number of temporal scenes to composite
        #[arg(long, default_value = "12")]
        max_scenes: usize,
        /// SCL asset key for cloud masking
        #[arg(long, default_value = "scl")]
        scl_asset: String,
        /// SCL classes to keep (comma-separated, default: vegetation,soil,water,snow)
        #[arg(long, default_value = "4,5,6,11")]
        scl_keep: String,
        /// Align output to this raster's grid (resamples to match origin, cell size, dims)
        #[arg(long)]
        align_to: Option<PathBuf>,
        /// Multi-band output naming: "prefix" → {stem}_{band}.tif (default), "asset" → {band}.tif
        #[arg(long, default_value = "prefix")]
        naming: String,
        /// Cache downloaded COG tiles locally (~/.cache/surtgis/cog/) for fast re-runs
        #[arg(long)]
        cache: bool,
        /// Rows per processing strip (larger = fewer HTTP requests but more RAM). Default: 512
        #[arg(long, default_value = "512")]
        strip_rows: usize,
        /// Bands to download + process together per scene (RAM↔HTTP dial). Higher = fewer HTTP
        /// requests (less rate-limit pressure) but more RAM per strip. 1 = minimum RAM (default).
        /// For 38 GB hosts, 3-5 is comfortable with ES; for PC, up to n_bands is fine.
        #[arg(long, default_value = "1")]
        band_chunk_size: usize,
        /// Abort the composite if more than this many tiles fail after retries
        /// (0 = never abort; just print a summary of how many tiles/scenes were
        /// affected). Multi-band composites (--asset with commas) only.
        #[arg(long, default_value = "0")]
        max_tile_failures: usize,
        /// Output GeoTIFF file
        output: PathBuf,
    },
    /// Download a time series: one cloud-free composite per interval (monthly, biweekly, etc.)
    TimeSeries {
        /// Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "es")]
        catalog: String,
        /// Bounding box: west,south,east,north
        #[arg(long)]
        bbox: String,
        /// Collection (e.g. "sentinel-2-l2a")
        #[arg(long)]
        collection: String,
        /// Data asset to download (e.g. "B04", "nir", "red")
        #[arg(long)]
        asset: String,
        /// Full datetime range (e.g. "2020-01-01/2024-12-31")
        #[arg(long)]
        datetime: String,
        /// Temporal interval: "monthly", "biweekly", "weekly", or custom days (e.g. "30")
        #[arg(long, default_value = "monthly")]
        interval: String,
        /// SCL asset key for cloud masking (use "none" to skip)
        #[arg(long, default_value = "scl")]
        scl_asset: String,
        /// Maximum scenes per interval
        #[arg(long, default_value = "8")]
        max_scenes: usize,
        /// Align output to this raster's grid (e.g., a DEM)
        #[arg(long)]
        align_to: Option<PathBuf>,
        /// Output directory (one GeoTIFF per interval)
        output: PathBuf,
    },
    /// Download climate data (Zarr) for a region and time range
    ///
    /// Searches a STAC catalog for climate datasets (ERA5, TerraClimate),
    /// aggregates over time intervals, and writes one GeoTIFF per interval.
    DownloadClimate {
        /// Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "pc")]
        catalog: String,
        /// Bounding box: west,south,east,north
        #[arg(long)]
        bbox: String,
        /// Collection (e.g. "era5-pds")
        #[arg(long)]
        collection: String,
        /// Variable name (e.g. "precipitation_amount_1hour_Accumulation")
        #[arg(long)]
        variable: String,
        /// Datetime range (e.g. "2020-01-01/2020-12-31")
        #[arg(long)]
        datetime: String,
        /// Temporal aggregation: none, daily-sum, daily-mean, monthly-mean, monthly-sum, yearly-mean, yearly-sum
        #[arg(long, default_value = "monthly-mean")]
        aggregate: String,
        /// Output directory (one GeoTIFF per interval)
        output: PathBuf,
    },
    /// List all available STAC catalogs (curated + indexed)
    ListCatalogs {
        /// Search for catalogs by keyword (e.g., "sentinel-2", "dem", "thermal")
        #[arg(long)]
        search: Option<String>,
    },
    /// List collections available in a STAC catalog
    ListCollections {
        /// Catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "pc")]
        catalog: String,
    },
}

// ─── Pipeline workflows ─────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum PipelineCommands {
    /// Compute susceptibility factors from DEM + S2 imagery
    Susceptibility {
        /// DEM source: STAC collection ID (cop-dem-glo-30, cop-dem-glo-90, nasadem, 3dep-seamless) or local file path
        #[arg(long)]
        dem: String,

        /// S2 source: "sentinel-2-l2a" or "earth-search" or "skip"
        #[arg(long)]
        s2: String,

        /// Bounding box: "west,south,east,north"
        #[arg(long)]
        bbox: String,

        /// Date range: "YYYY-MM-DD/YYYY-MM-DD"
        #[arg(long)]
        datetime: String,

        /// Output directory (will be created)
        #[arg(long)]
        outdir: PathBuf,

        /// Max scenes for S2 (default: 12)
        #[arg(long, default_value = "12")]
        max_scenes: usize,

        /// Cloud mask classes to keep for S2 (default: "4,5,6,11")
        #[arg(long, default_value = "4,5,6,11")]
        scl_keep: String,
    },

    /// Generate geomorphometric feature stack from DEM
    #[command(about = "Generate geomorphometric feature stack from DEM")]
    Features {
        /// Input DEM file
        input: PathBuf,
        /// Output directory
        #[arg(short, long)]
        outdir: PathBuf,
        /// Skip hydrology features (faster)
        #[arg(long)]
        skip_hydrology: bool,
        /// Include extra features (valley depth, surface area ratio, landform, wind exposure, accumulation zones)
        #[arg(long)]
        extras: bool,
    },
    /// End-to-end temporal analysis: STAC download → spectral index → trend/stats/phenology
    #[cfg(feature = "cloud")]
    Temporal {
        /// STAC catalog: "pc" (Planetary Computer), "es" (Earth Search), or full URL
        #[arg(long, default_value = "es")]
        catalog: String,
        /// Bounding box: west,south,east,north
        #[arg(long)]
        bbox: String,
        /// Collection (e.g. "sentinel-2-l2a", "landsat-c2-l2")
        #[arg(long)]
        collection: String,
        /// Date range: "YYYY-MM-DD/YYYY-MM-DD"
        #[arg(long)]
        datetime: String,
        /// Temporal interval: "monthly", "biweekly", "weekly", "quarterly", "yearly", or custom days
        #[arg(long, default_value = "monthly")]
        interval: String,
        /// Spectral index to compute: ndvi, ndwi, mndwi, nbr, savi, evi, evi2, bsi, ndbi, ndmi, ndsi, gndvi, ngrdi, ndre, msavi
        #[arg(long)]
        index: String,
        /// Analysis type (comma-separated): stats, trend, phenology
        #[arg(long)]
        analysis: String,
        /// Trend method (when analysis includes "trend"): linear, mann-kendall
        #[arg(long, default_value = "linear")]
        method: String,
        /// Phenology threshold for SOS/EOS (0-1)
        #[arg(long, default_value = "0.5")]
        threshold: f64,
        /// Phenology smoothing window (odd number)
        #[arg(long, default_value = "5")]
        smooth: usize,
        /// Statistics to compute (when analysis includes "stats")
        #[arg(long, default_value = "mean,std,min,max,count")]
        stats: String,
        /// Maximum scenes per interval window
        #[arg(long, default_value = "8")]
        max_scenes: usize,
        /// Output directory
        #[arg(long)]
        outdir: PathBuf,
        /// Keep per-interval index rasters as intermediate outputs
        #[arg(long)]
        keep_intermediates: bool,
        /// Align output to this reference raster's grid (e.g., a DEM)
        #[arg(long)]
        align_to: Option<PathBuf>,
    },
}

// ─── Temporal analysis subcommands ──────────────────────────────────────

#[derive(Subcommand)]
pub enum TemporalCommands {
    /// Per-pixel temporal statistics (mean, std, min, max, count, percentile)
    Stats {
        /// Input rasters (time-ordered GeoTIFFs)
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,
        /// Output directory for statistic rasters
        #[arg(short, long)]
        outdir: PathBuf,
        /// Which statistics to compute (comma-separated): mean,std,min,max,count,p10,p25,p50,p75,p90
        #[arg(long, default_value = "mean,std,min,max,count")]
        stats: String,
    },
    /// Pixel-wise linear trend analysis (OLS regression)
    Trend {
        /// Input rasters (time-ordered GeoTIFFs)
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,
        /// Output directory for trend rasters (slope, intercept, r2, pvalue)
        #[arg(short, long)]
        outdir: PathBuf,
        /// Method: "linear" (OLS) or "mann-kendall" (non-parametric)
        #[arg(long, default_value = "linear")]
        method: String,
        /// Time values (comma-separated, e.g. fractional years). If omitted, uses 0,1,2,...
        #[arg(long)]
        times: Option<String>,
    },
    /// Change detection between two dates
    Change {
        /// Before raster (time T1)
        #[arg(long)]
        before: PathBuf,
        /// After raster (time T2)
        #[arg(long)]
        after: PathBuf,
        /// Output difference raster
        output: PathBuf,
        /// Threshold for significant decrease
        #[arg(long, default_value = "-1.0")]
        decrease_threshold: f64,
        /// Threshold for significant increase
        #[arg(long, default_value = "1.0")]
        increase_threshold: f64,
    },
    /// Anomaly detection vs reference period
    Anomaly {
        /// Reference period rasters (baseline, at least 2)
        #[arg(long, required = true)]
        reference: Vec<PathBuf>,
        /// Target rasters to evaluate
        #[arg(long, required = true)]
        target: Vec<PathBuf>,
        /// Output directory
        #[arg(short, long)]
        outdir: PathBuf,
        /// Method: "zscore", "difference", or "percent"
        #[arg(long, default_value = "zscore")]
        method: String,
    },
    /// Vegetation phenology metrics from NDVI/EVI time series
    Phenology {
        /// Input rasters (time-ordered NDVI/EVI GeoTIFFs, at least 6)
        #[arg(short, long, required = true)]
        input: Vec<PathBuf>,
        /// Output directory for phenology rasters (sos, eos, peak, amplitude, etc.)
        #[arg(short, long)]
        outdir: PathBuf,
        /// Day-of-year for each input (comma-separated). If omitted, uses indices.
        #[arg(long)]
        doys: Option<String>,
        /// Threshold for SOS/EOS as fraction of amplitude (0-1)
        #[arg(long, default_value = "0.5")]
        threshold: f64,
        /// Smoothing window size (odd number, 0=none)
        #[arg(long, default_value = "5")]
        smooth: usize,
    },
}

// ─── Vector subcommands ────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum VectorCommands {
    /// Geometric intersection of two vector layers (A ∩ B)
    Intersection {
        /// Input layer A (GeoJSON/Shapefile/GeoPackage)
        input_a: PathBuf,
        /// Input layer B (overlay)
        input_b: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// Geometric union of two vector layers (A ∪ B)
    Union {
        input_a: PathBuf,
        input_b: PathBuf,
        output: PathBuf,
    },
    /// Geometric difference: features of A not covered by B (A - B)
    Difference {
        input_a: PathBuf,
        input_b: PathBuf,
        output: PathBuf,
    },
    /// Symmetric difference: areas in A or B but not both (A ⊕ B)
    SymDifference {
        input_a: PathBuf,
        input_b: PathBuf,
        output: PathBuf,
    },
    /// Dissolve all features into a single geometry
    Dissolve {
        /// Input layer
        input: PathBuf,
        /// Output file
        output: PathBuf,
    },
    /// Buffer features by a distance
    Buffer {
        /// Input layer
        input: PathBuf,
        /// Buffer distance (in CRS units)
        #[arg(long)]
        distance: f64,
        /// Number of segments per quarter circle (default: 8)
        #[arg(long, default_value = "8")]
        segments: usize,
        /// Output file
        output: PathBuf,
    },
}

// ─── Interpolation subcommands ─────────────────────────────────────────

#[derive(Subcommand)]
pub enum InterpolationCommands {
    /// Compute empirical variogram and fit a theoretical model
    Variogram {
        /// Input points (GeoJSON/Shapefile with a numeric attribute)
        points: PathBuf,
        /// Attribute name containing the values to analyze
        #[arg(long)]
        attribute: String,
        /// Number of lag bins (default: 15)
        #[arg(long, default_value = "15")]
        bins: usize,
        /// Maximum lag distance (auto-detect if omitted)
        #[arg(long)]
        max_lag: Option<f64>,
        /// Output JSON file with variogram data and fitted model
        output: PathBuf,
    },
    /// Ordinary Kriging interpolation from points to raster
    Kriging {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with values
        #[arg(long)]
        attribute: String,
        /// Reference raster for output grid (resolution, extent, CRS)
        #[arg(long)]
        reference: PathBuf,
        /// Variogram model: spherical, exponential, gaussian, matern
        #[arg(long, default_value = "spherical")]
        model: String,
        /// Variogram range (auto-fit if omitted)
        #[arg(long)]
        range: Option<f64>,
        /// Variogram sill (auto-fit if omitted)
        #[arg(long)]
        sill: Option<f64>,
        /// Variogram nugget (default: 0)
        #[arg(long, default_value = "0")]
        nugget: f64,
        /// Maximum neighbors for kriging (default: 16)
        #[arg(long, default_value = "16")]
        max_neighbors: usize,
        /// Output raster
        output: PathBuf,
    },
    /// Universal Kriging with polynomial drift
    UniversalKriging {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with values
        #[arg(long)]
        attribute: String,
        /// Reference raster for output grid
        #[arg(long)]
        reference: PathBuf,
        /// Drift order: linear or quadratic
        #[arg(long, default_value = "linear")]
        drift: String,
        /// Variogram model: spherical, exponential, gaussian
        #[arg(long, default_value = "spherical")]
        model: String,
        /// Maximum neighbors (default: 16)
        #[arg(long, default_value = "16")]
        max_neighbors: usize,
        /// Output raster
        output: PathBuf,
    },
    /// Regression-Kriging: ML prediction + kriging on residuals
    RegressionKriging {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with target values
        #[arg(long)]
        attribute: String,
        /// Directory with covariable rasters (features)
        #[arg(long)]
        features: PathBuf,
        /// Reference raster for output grid
        #[arg(long)]
        reference: PathBuf,
        /// Variogram model for residuals
        #[arg(long, default_value = "spherical")]
        model: String,
        /// Output raster
        output: PathBuf,
    },
    /// Inverse Distance Weighting interpolation
    Idw {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with values
        #[arg(long)]
        attribute: String,
        /// Reference raster for output grid
        #[arg(long)]
        reference: PathBuf,
        /// Power parameter (default: 2.0, higher = more local)
        #[arg(long, default_value = "2.0")]
        power: f64,
        /// Maximum search radius (default: unlimited)
        #[arg(long)]
        max_radius: Option<f64>,
        /// Maximum neighbors (default: all)
        #[arg(long)]
        max_points: Option<usize>,
        /// Output raster
        output: PathBuf,
    },
    /// Nearest Neighbor interpolation (Voronoi/Thiessen)
    NearestNeighbor {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with values
        #[arg(long)]
        attribute: String,
        /// Reference raster for output grid
        #[arg(long)]
        reference: PathBuf,
        /// Maximum search radius (default: unlimited)
        #[arg(long)]
        max_radius: Option<f64>,
        /// Output raster
        output: PathBuf,
    },
    /// Natural Neighbor interpolation (Sibson)
    NaturalNeighbor {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with values
        #[arg(long)]
        attribute: String,
        /// Reference raster for output grid
        #[arg(long)]
        reference: PathBuf,
        /// Output raster
        output: PathBuf,
    },
    /// Thin Plate Spline interpolation
    Tps {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with values
        #[arg(long)]
        attribute: String,
        /// Reference raster for output grid
        #[arg(long)]
        reference: PathBuf,
        /// Smoothing parameter (0 = exact, >0 = smoothing spline)
        #[arg(long, default_value = "0.0")]
        smoothing: f64,
        /// Output raster
        output: PathBuf,
    },
    /// TIN (Triangulated Irregular Network) interpolation
    Tin {
        /// Input points (GeoJSON/Shapefile)
        points: PathBuf,
        /// Attribute name with values
        #[arg(long)]
        attribute: String,
        /// Reference raster for output grid
        #[arg(long)]
        reference: PathBuf,
        /// Output raster
        output: PathBuf,
    },
}

// ─── ML subcommands ────────────────────────────────────────────────────

#[cfg(feature = "ml")]
#[derive(Subcommand)]
pub enum MlCommands {
    /// Extract training samples from feature rasters at point locations
    ExtractSamples {
        /// Directory with features.json and feature rasters (from `pipeline features`)
        #[arg(long)]
        features_dir: PathBuf,
        /// Vector file with labeled points (.geojson, .shp, .gpkg)
        #[arg(long)]
        points: PathBuf,
        /// Property name containing the target label/class
        #[arg(long)]
        target: String,
        /// Output CSV file with extracted samples
        output: PathBuf,
    },
    /// Train a classifier or regressor on a CSV dataset
    Train {
        /// Input CSV file (from extract-samples or external)
        input: PathBuf,
        /// Target column name
        #[arg(long)]
        target: String,
        /// Model type: rf (random-forest), default: rf
        #[arg(long, default_value = "rf")]
        model: String,
        /// Number of estimators (trees)
        #[arg(long, default_value = "100")]
        n_estimators: usize,
        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        folds: usize,
        /// Task type: classification or regression
        #[arg(long, default_value = "classification")]
        task: String,
        /// Output model file (.json)
        output: PathBuf,
    },
    /// Predict on rasters using a trained model
    Predict {
        /// Trained model file (.json)
        #[arg(long)]
        model: PathBuf,
        /// Directory with features.json and feature rasters
        #[arg(long)]
        features_dir: PathBuf,
        /// Output prediction raster (.tif)
        output: PathBuf,
    },
    /// Benchmark multiple learners on a dataset
    Benchmark {
        /// Input CSV file
        input: PathBuf,
        /// Target column name
        #[arg(long)]
        target: String,
        /// Number of cross-validation folds
        #[arg(long, default_value = "5")]
        folds: usize,
        /// Task type: classification or regression
        #[arg(long, default_value = "classification")]
        task: String,
    },
}

// ─── Classification subcommands ────────────────────────────────────────

#[derive(Subcommand)]
pub enum ClassificationCommands {
    /// K-means unsupervised clustering
    Kmeans {
        /// Input raster
        input: PathBuf,
        /// Output classified raster
        output: PathBuf,
        /// Number of clusters
        #[arg(short, long, default_value = "5")]
        classes: usize,
        /// Maximum iterations
        #[arg(long, default_value = "100")]
        max_iter: usize,
        /// Convergence threshold
        #[arg(long, default_value = "0.001")]
        convergence: f64,
        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// ISODATA adaptive clustering (auto-splits/merges clusters)
    Isodata {
        /// Input raster
        input: PathBuf,
        /// Output classified raster
        output: PathBuf,
        /// Initial number of clusters
        #[arg(short, long, default_value = "5")]
        classes: usize,
        /// Minimum clusters
        #[arg(long, default_value = "2")]
        min_classes: usize,
        /// Maximum clusters
        #[arg(long, default_value = "10")]
        max_classes: usize,
        /// Maximum iterations
        #[arg(long, default_value = "50")]
        max_iter: usize,
    },
    /// Principal Component Analysis (multi-band input)
    Pca {
        /// Input raster bands (comma-separated paths)
        #[arg(long)]
        bands: String,
        /// Output directory for PC rasters
        output: PathBuf,
        /// Number of components (default: all)
        #[arg(short, long)]
        components: Option<usize>,
    },
    /// Minimum distance supervised classification
    MinDistance {
        /// Input raster
        input: PathBuf,
        /// Output classified raster
        output: PathBuf,
        /// Training raster (class labels)
        #[arg(long)]
        training: PathBuf,
    },
    /// Maximum likelihood supervised classification
    MaxLikelihood {
        /// Input raster
        input: PathBuf,
        /// Output classified raster
        output: PathBuf,
        /// Training raster (class labels)
        #[arg(long)]
        training: PathBuf,
    },
}

// ─── Texture subcommands ───────────────────────────────────────────────

#[derive(Subcommand)]
pub enum TextureCommands {
    /// GLCM texture (Haralick): energy, contrast, homogeneity, correlation, entropy
    Glcm {
        /// Input raster
        input: PathBuf,
        /// Output texture raster
        output: PathBuf,
        /// Texture measure: energy, contrast, homogeneity, correlation, entropy, dissimilarity
        #[arg(short, long, default_value = "contrast")]
        texture: String,
        /// Window radius
        #[arg(short, long, default_value = "3")]
        radius: usize,
        /// Quantization levels
        #[arg(long, default_value = "32")]
        levels: usize,
    },
    /// GLCM all 6 textures in a single pass (one output file per feature)
    GlcmAll {
        /// Input raster
        input: PathBuf,
        /// Output directory (one .tif per feature is written here)
        output_dir: PathBuf,
        /// Window radius
        #[arg(short, long, default_value = "3")]
        radius: usize,
        /// Quantization levels
        #[arg(long, default_value = "32")]
        levels: usize,
    },
    /// Local Binary Patterns (Ojala 2002)
    Lbp {
        /// Input raster
        input: PathBuf,
        /// Output LBP raster
        output: PathBuf,
        /// Variant: standard (0..255 raw code) or riu2 (rotation-invariant uniform)
        #[arg(long, default_value = "standard")]
        variant: String,
    },
    /// Sobel edge detection (gradient magnitude)
    Sobel {
        /// Input raster
        input: PathBuf,
        /// Output edge raster
        output: PathBuf,
    },
    /// Laplacian edge detection (second derivative)
    Laplacian {
        /// Input raster
        input: PathBuf,
        /// Output edge raster
        output: PathBuf,
    },
}

// ─── Segmentation subcommands ──────────────────────────────────────────

#[derive(Subcommand)]
pub enum SegmentationCommands {
    /// SLIC superpixels (Achanta 2012)
    Slic {
        /// Output label raster (i32 GeoTIFF, 1..=N labels, 0 = nodata)
        output: PathBuf,
        /// One or more input band rasters (repeatable)
        #[arg(long = "band", required = true)]
        bands: Vec<PathBuf>,
        /// Target number of superpixels
        #[arg(short = 'n', long, default_value = "100")]
        n_segments: usize,
        /// Compactness (higher = more geometric superpixels)
        #[arg(short = 'm', long, default_value = "10.0")]
        compactness: f64,
        /// Maximum k-means iterations
        #[arg(long, default_value = "10")]
        max_iter: usize,
        /// Disable connectivity enforcement post-pass
        #[arg(long, default_value = "false")]
        no_connectivity: bool,
    },
    /// Felzenszwalb-Huttenlocher graph segmentation (2004)
    Felzenszwalb {
        /// Output label raster (i32 GeoTIFF, 1..=N labels, 0 = nodata)
        output: PathBuf,
        /// One or more input band rasters (repeatable)
        #[arg(long = "band", required = true)]
        bands: Vec<PathBuf>,
        /// Scale parameter k (higher = larger segments)
        #[arg(short = 'k', long, default_value = "1.0")]
        scale: f64,
        /// Minimum component size (smaller components are merged)
        #[arg(long, default_value = "20")]
        min_size: usize,
    },
}

// ─── Statistics subcommands ────────────────────────────────────────────

#[derive(Subcommand)]
pub enum StatisticsCommands {
    /// Focal (moving window) statistics
    Focal {
        /// Input raster
        input: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Statistic: mean, std, min, max, range, sum, count, median
        #[arg(short, long, default_value = "mean")]
        stat: String,
        /// Window radius (size = 2r+1)
        #[arg(short, long, default_value = "3")]
        radius: usize,
        /// Use circular window instead of square
        #[arg(long)]
        circular: bool,
    },
    /// Zonal statistics (JSON output by zone)
    Zonal {
        /// Input values raster
        input: PathBuf,
        /// Zone raster (integer classes)
        #[arg(long)]
        zones: PathBuf,
        /// Output JSON file
        output: PathBuf,
    },
    /// Zonal statistics as raster (each cell = zone statistic)
    ZonalRaster {
        /// Input values raster
        input: PathBuf,
        /// Zone raster (integer classes)
        #[arg(long)]
        zones: PathBuf,
        /// Output raster
        output: PathBuf,
        /// Statistic: mean, std, min, max, range, sum, count, median
        #[arg(short, long, default_value = "mean")]
        stat: String,
    },
    /// Global Moran's I spatial autocorrelation (prints result)
    MoransI {
        /// Input raster
        input: PathBuf,
    },
    /// Local Getis-Ord Gi* hotspot analysis
    GetisOrd {
        /// Input raster
        input: PathBuf,
        /// Output z-scores raster
        output: PathBuf,
        /// Neighborhood radius
        #[arg(short, long, default_value = "3")]
        radius: usize,
    },
}

// ─── Fluvial-tectonic morphometry ────────────────────────────────────────
//
// Subcommands under `surtgis fluvial`. v1 of the spec ships chi here;
// ksn, knickpoints, concavity, divide-migration follow in later sprints.

/// Fluvial-tectonic morphometry subcommands.
#[derive(Debug, Subcommand)]
pub enum FluvialCommands {
    /// χ (chi) integral transform (Perron & Royden 2013).
    ///
    /// Computes the path integral of (A₀/A(x))^θref along each stream
    /// cell starting from the network's outlets. χ is a base-level
    /// reference distance that linearises elevation profiles in steady
    /// state; together with ksn it is one of the two most common metrics
    /// in geomorphology-from-topography.
    Chi {
        /// Binary stream network raster (1 = stream, 0 = non-stream),
        /// e.g. the output of `surtgis hydrology stream-network`.
        stream: PathBuf,
        /// D8 flow direction raster from `surtgis hydrology flow-direction`.
        flow_dir: PathBuf,
        /// Flow accumulation raster (cell counts) from
        /// `surtgis hydrology flow-accumulation`.
        flow_acc: PathBuf,
        /// Output χ raster (Float32 GeoTIFF, NaN = non-stream).
        output: PathBuf,
        /// Reference concavity exponent. Default 0.45 (bedrock channels
        /// in steady state).
        #[arg(long, default_value = "0.45")]
        theta_ref: f64,
        /// Reference drainage area in m². Default 1·10⁶ m² (1 km²).
        #[arg(long, default_value = "1e6")]
        a_0_m2: f64,
        /// Cell size override in metres. Default: read from the raster
        /// transform. Required when the input does not declare a
        /// projected CRS; not used to override an existing transform.
        #[arg(long)]
        cell_size_m: Option<f64>,
    },
    /// Normalised channel steepness index `ksn` (Wobus et al. 2006).
    ///
    /// Channel-following slope × A^θref, smoothed over a moving window
    /// along the network. `ksn` is the workhorse proxy for U/K (uplift
    /// rate divided by erodibility) used in tectonic geomorphology.
    Ksn {
        /// Binary stream network raster (1 = stream, 0 = non-stream).
        stream: PathBuf,
        /// D8 flow direction raster.
        flow_dir: PathBuf,
        /// Flow accumulation raster (cell counts).
        flow_acc: PathBuf,
        /// DEM (elevation in metres).
        dem: PathBuf,
        /// Output `ksn` raster (Float32 GeoTIFF, NaN = non-stream / outlet).
        output: PathBuf,
        /// Reference concavity exponent. Default 0.45.
        #[arg(long, default_value = "0.45")]
        theta_ref: f64,
        /// Smoothing window length in metres. Default 500 (Wobus 2006 standard).
        #[arg(long, default_value = "500")]
        segment_length_m: f64,
        /// Minimum drainage area in m² for a cell to contribute. Default 1e6 (1 km²).
        #[arg(long, default_value = "1e6")]
        min_drainage_area_m2: f64,
        /// Cell size override in metres. Defaults to the raster transform's pixel size.
        #[arg(long)]
        cell_size_m: Option<f64>,
        /// Optional path to write per-segment vector output as a GeoJSON
        /// LineString FeatureCollection (one feature per stream segment
        /// between confluences/outlet, attributes: `ksn_mean`, `n_cells`).
        #[arg(long)]
        segments: Option<PathBuf>,
        /// Preserve the source raster's CRS in the GeoJSON output and
        /// declare it via a legacy `crs` member (GeoJSON 2008 compat).
        /// Default: reproject coordinates to WGS84 (RFC 7946 compliant)
        /// — recommended for web-mapping pipelines (MapLibre, deck.gl)
        /// and modern geopandas / QGIS workflows.
        #[arg(long)]
        keep_crs: bool,
    },
    /// Knickpoint detection (Neely et al. 2017, TVD denoising + curvature).
    ///
    /// Detects sharp slope breaks along each river long profile.
    /// Knickpoints are classified as `concave` (slope decreases
    /// downstream → likely lithologic contrast) or `convex` (slope
    /// increases downstream → likely transient tectonic pulse).
    Knickpoints {
        /// Binary stream network raster (1 = stream, 0 = non-stream).
        stream: PathBuf,
        /// D8 flow direction raster.
        flow_dir: PathBuf,
        /// Flow accumulation raster (cell counts).
        flow_acc: PathBuf,
        /// DEM (elevation in metres).
        dem: PathBuf,
        /// Output GeoJSON Point FeatureCollection of knickpoints.
        output: PathBuf,
        /// Reference concavity exponent. Default 0.45.
        #[arg(long, default_value = "0.45")]
        theta_ref: f64,
        /// TVD regularization (larger = stronger smoothing). Default 0.5.
        #[arg(long, default_value = "0.5")]
        tvd_lambda: f64,
        /// Threshold on |d²z/dχ²| (units: 1/m). Default 1.0.
        #[arg(long, default_value = "1.0")]
        curvature_threshold: f64,
        /// Minimum elevation magnitude across the knickpoint window
        /// (metres). Default 10 m.
        #[arg(long, default_value = "10")]
        min_magnitude_m: f64,
        /// Number of cells to exclude at each segment end (per spec
        /// pitfall §8.9: confluences and outlets induce spurious
        /// curvature). Default 5.
        #[arg(long, default_value = "5")]
        confluence_buffer_cells: usize,
        /// Cell size override in metres. Default: read from raster.
        #[arg(long)]
        cell_size_m: Option<f64>,
        /// Optional path to write a categorical raster (0=no knickpoint,
        /// 1=concave, 2=convex) alongside the GeoJSON points.
        #[arg(long)]
        raster: Option<PathBuf>,
        /// Preserve the source raster's CRS in the GeoJSON output and
        /// declare it via a legacy `crs` member. Default: reproject to
        /// WGS84 (RFC 7946 compliant).
        #[arg(long)]
        keep_crs: bool,
    },
    /// Concavity index θ per basin (Perron & Royden 2013).
    ///
    /// Grid-search θ ∈ [low, high] minimising the residual RMSE of
    /// elevation~χ regression. Bootstrap (default n=200) for a 95% CI.
    /// Output is a CSV table with one row per qualifying basin.
    Concavity {
        /// Binary stream network raster.
        stream: PathBuf,
        /// D8 flow direction raster.
        flow_dir: PathBuf,
        /// Flow accumulation raster (cell counts).
        flow_acc: PathBuf,
        /// DEM (elevation in metres).
        dem: PathBuf,
        /// Basin id raster (e.g. from `surtgis hydrology watershed`),
        /// i32 with 0 = no basin.
        basins: PathBuf,
        /// Output CSV path. Columns:
        /// basin_id, theta_opt, theta_ci_low, theta_ci_high, n_cells, rmse.
        output: PathBuf,
        /// θ search range as comma-separated (low,high). Default "0.1,0.9".
        #[arg(long, default_value = "0.1,0.9")]
        theta_range: String,
        /// θ grid step. Default 0.05.
        #[arg(long, default_value = "0.05")]
        theta_step: f64,
        /// Bootstrap iterations for the 95% CI. Default 200; set to 0 to
        /// skip bootstrap (CI returned as [theta_opt, theta_opt]).
        #[arg(long, default_value = "200")]
        bootstrap_n: usize,
        /// Minimum basin stream-cell count to attempt an estimate.
        /// Default 30.
        #[arg(long, default_value = "30")]
        min_basin_cells: usize,
        /// Seed for the deterministic bootstrap resampler. Default 42.
        #[arg(long, default_value = "42")]
        seed: u64,
        /// Cell size override in metres.
        #[arg(long)]
        cell_size_m: Option<f64>,
    },
    /// Divide-migration metrics (Willett 2014, Whipple 2017).
    ///
    /// For each pair of adjacent watershed basins, computes the median
    /// asymmetry across the divide: Gilbert elevation/relief difference
    /// and (optionally) χ across-divide difference. Output is a GeoJSON
    /// LineString FeatureCollection — one feature per adjacent-basin pair.
    DivideMigration {
        /// Basin id raster (i32, 0 = no basin).
        basins: PathBuf,
        /// DEM (elevation in metres).
        dem: PathBuf,
        /// Flow accumulation raster (currently informational; reserved
        /// for future area-weighted statistics).
        flow_acc: PathBuf,
        /// Output GeoJSON path.
        output: PathBuf,
        /// Optional pre-computed χ raster from `surtgis fluvial chi`.
        /// When supplied, median_chi_diff is reported; otherwise NaN.
        #[arg(long)]
        chi: Option<PathBuf>,
        /// Minimum cumulative divide polyline length (m) to report.
        /// Default 500 m — short divides are rarely robust.
        #[arg(long, default_value = "500")]
        min_divide_length_m: f64,
        /// Cell size override in metres.
        #[arg(long)]
        cell_size_m: Option<f64>,
        /// Preserve the source raster's CRS in the GeoJSON output and
        /// declare it via a legacy `crs` member. Default: reproject to
        /// WGS84 (RFC 7946 compliant).
        #[arg(long)]
        keep_crs: bool,
    },
}
