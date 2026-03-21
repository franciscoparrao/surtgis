//! SurtGis CLI - High-performance geospatial analysis

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use surtgis_algorithms::hydrology::{
    breach_depressions, fill_sinks, flow_accumulation, flow_accumulation_mfd, flow_direction,
    flow_direction_dinf, hand, priority_flood, stream_network, watershed, BreachParams,
    FillSinksParams, HandParams, MfdParams, PriorityFloodParams, StreamNetworkParams,
    WatershedParams,
};
use surtgis_algorithms::imagery::{
    band_math_binary, bsi, cloud_mask_scl, evi, evi2, gndvi, index_builder, median_composite,
    mndwi, msavi, nbr, ndbi, ndmi, ndre, ndsi, ndvi, ndwi, ngrdi, reci, reclassify, savi,
    BandMathOp, EviParams, ReclassEntry, ReclassifyParams, SaviParams,
};
use std::collections::{BTreeMap, HashMap};
use surtgis_algorithms::landscape::{
    class_metrics, label_patches, landscape_metrics, patch_metrics, patches_to_csv,
    Connectivity,
};
use surtgis_algorithms::morphology::{
    black_hat, closing, dilate, erode, gradient, opening, top_hat, StructuringElement,
};
use surtgis_algorithms::terrain::{
    advanced_curvatures, aspect, convergence_index, curvature, dev, eastness, geomorphons,
    hillshade, landform_classification, mrvbf, multidirectional_hillshade, negative_openness,
    northness, positive_openness, sky_view_factor, slope, tpi, tri, twi, viewshed, vrm,
    AdvancedCurvatureType, AspectOutput, AspectStreaming, ConvergenceParams, CurvatureParams,
    CurvatureStreaming, CurvatureType, DevParams, EastnessStreaming, GeomorphonParams,
    HillshadeParams, HillshadeStreaming, LandformParams, MultiHillshadeParams, MrvbfParams,
    NorthnessStreaming, OpennessParams, SlopeParams, SlopeStreaming, SlopeUnits, SvfParams,
    TpiParams, TpiStreaming, TriParams, TriStreaming, ViewshedParams, VrmParams,
};
use surtgis_core::StripProcessor;
use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};

#[cfg(feature = "cloud")]
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking, read_cog};
#[cfg(feature = "cloud")]
use surtgis_cloud::{BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacItem, StacSearchParams};

// ─── CLI structure ──────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "surtgis")]
#[command(author, version, about = "High-performance geospatial analysis", long_about = None)]
struct Cli {
    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Compress output GeoTIFFs (deflate)
    #[arg(long, global = true)]
    compress: bool,

    /// Force streaming mode for large rasters (auto-detected if >500MB)
    #[arg(long, global = true)]
    streaming: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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
}

// ─── Terrain subcommands ────────────────────────────────────────────────

#[derive(Subcommand)]
enum TerrainCommands {
    /// Calculate slope from DEM
    Slope {
        /// Input DEM file
        input: PathBuf,
        /// Output file
        output: PathBuf,
        /// Output units: degrees, percent, radians
        #[arg(short, long, default_value = "degrees")]
        units: String,
        /// Z-factor for unit conversion
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
        /// Z-factor for vertical exaggeration
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
        /// Z-factor for unit conversion
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
    Northness {
        input: PathBuf,
        output: PathBuf,
    },
    /// Eastness: sin(aspect), east-facing = 1, west-facing = -1
    Eastness {
        input: PathBuf,
        output: PathBuf,
    },
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
        /// Curvature type: mean_h, gaussian_k, kmin, kmax, kh, kv, khe, kve, ka, kr, rotor, laplacian, unsphericity, difference
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
    MultiHillshade {
        input: PathBuf,
        output: PathBuf,
    },
    /// Compute all standard terrain factors in one pass
    All {
        /// Input DEM file
        input: PathBuf,
        /// Output directory for all terrain products
        #[arg(short, long)]
        outdir: PathBuf,
    },
}

// ─── Hydrology subcommands ──────────────────────────────────────────────

#[derive(Subcommand)]
enum HydrologyCommands {
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
    FlowDirectionDinf {
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
    /// Stream network extraction (from DEM, full pipeline)
    StreamNetwork {
        /// Input DEM file
        input: PathBuf,
        output: PathBuf,
        /// Contributing area threshold
        #[arg(long, default_value = "1000")]
        threshold: f64,
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
enum ImageryCommands {
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
}

// ─── Morphology subcommands ─────────────────────────────────────────────

#[derive(Subcommand)]
enum MorphologyCommands {
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
enum LandscapeCommands {
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
enum CogCommands {
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
        /// Z-factor for unit conversion
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
        /// Z-factor for vertical exaggeration
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
enum StacCommands {
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
        #[arg(long)]
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
        #[arg(long)]
        collection: String,
        /// Asset key to fetch (e.g. "red", "nir", "B04"). Auto-detects COG if omitted.
        #[arg(long)]
        asset: Option<String>,
        /// Datetime or range
        #[arg(long)]
        datetime: Option<String>,
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
        #[arg(long)]
        collection: String,
        /// Asset key to fetch (e.g. "data", "red", "B04"). Auto-detects COG if omitted.
        #[arg(long)]
        asset: Option<String>,
        /// Datetime or range
        #[arg(long)]
        datetime: Option<String>,
        /// Maximum items to fetch and mosaic
        #[arg(long, default_value = "20")]
        max_items: u32,
        /// Output GeoTIFF file
        output: PathBuf,
    },
    /// End-to-end satellite composite: search → mosaic per date → cloud-mask → median composite
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
        /// Output GeoTIFF file
        output: PathBuf,
    },
}

// ─── Helpers ────────────────────────────────────────────────────────────

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
}

fn spinner(msg: &str) -> ProgressBar {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb
}

fn read_dem(path: &PathBuf) -> Result<surtgis_core::Raster<f64>> {
    let pb = spinner("Reading raster...");
    let raster: surtgis_core::Raster<f64> =
        read_geotiff(path, None).context("Failed to read raster")?;
    pb.finish_and_clear();
    info!("Input: {} x {}", raster.cols(), raster.rows());
    Ok(raster)
}

fn read_u8(path: &PathBuf) -> Result<surtgis_core::Raster<u8>> {
    let pb = spinner("Reading raster...");
    let raster: surtgis_core::Raster<u8> =
        read_geotiff(path, None).context("Failed to read raster")?;
    pb.finish_and_clear();
    Ok(raster)
}

fn write_opts(compress: bool) -> GeoTiffOptions {
    GeoTiffOptions {
        compression: if compress {
            "deflate".to_string()
        } else {
            "NONE".to_string()
        },
    }
}

fn write_result(raster: &surtgis_core::Raster<f64>, path: &PathBuf, compress: bool) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress)))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

fn write_result_u8(
    raster: &surtgis_core::Raster<u8>,
    path: &PathBuf,
    compress: bool,
) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress)))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

fn write_result_i32(
    raster: &surtgis_core::Raster<i32>,
    path: &PathBuf,
    compress: bool,
) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(write_opts(compress)))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

fn done(name: &str, path: &std::path::Path, elapsed: std::time::Duration) {
    println!("{} saved to: {}", name, path.display());
    println!("  Processing time: {:.2?}", elapsed);
}

fn parse_se(shape: &str, radius: usize) -> Result<StructuringElement> {
    let se = match shape.to_lowercase().as_str() {
        "square" | "sq" => StructuringElement::Square(radius),
        "cross" | "cr" => StructuringElement::Cross(radius),
        "disk" | "circle" => StructuringElement::Disk(radius),
        _ => anyhow::bail!("Unknown shape: {}. Use square, cross, or disk.", shape),
    };
    se.validate()
        .map_err(|e| anyhow::anyhow!("Invalid structuring element: {}", e))?;
    Ok(se)
}

fn parse_connectivity(c: u8) -> Result<Connectivity> {
    match c {
        4 => Ok(Connectivity::Four),
        8 => Ok(Connectivity::Eight),
        _ => anyhow::bail!("Connectivity must be 4 or 8, got: {}", c),
    }
}

fn parse_band_math_op(s: &str) -> Result<BandMathOp> {
    match s.to_lowercase().as_str() {
        "add" | "+" => Ok(BandMathOp::Add),
        "subtract" | "sub" | "-" => Ok(BandMathOp::Subtract),
        "multiply" | "mul" | "*" => Ok(BandMathOp::Multiply),
        "divide" | "div" | "/" => Ok(BandMathOp::Divide),
        "power" | "pow" | "^" => Ok(BandMathOp::Power),
        "min" => Ok(BandMathOp::Min),
        "max" => Ok(BandMathOp::Max),
        _ => anyhow::bail!(
            "Unknown operation: {}. Use add, subtract, multiply, divide, power, min, max.",
            s
        ),
    }
}

fn parse_band_assignments(bands: &[String]) -> Result<Vec<(String, PathBuf)>> {
    bands
        .iter()
        .map(|s| {
            let parts: Vec<&str> = s.splitn(2, '=').collect();
            if parts.len() != 2 {
                anyhow::bail!("Band must be NAME=path, got: {}", s);
            }
            Ok((parts[0].to_string(), PathBuf::from(parts[1])))
        })
        .collect()
}

fn parse_reclass_entry(s: &str) -> Result<ReclassEntry> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 3 {
        anyhow::bail!("Class must be 'min,max,value', got: {}", s);
    }
    let min: f64 = parts[0].trim().parse().context("Invalid min")?;
    let max: f64 = parts[1].trim().parse().context("Invalid max")?;
    let value: f64 = parts[2].trim().parse().context("Invalid value")?;
    Ok(ReclassEntry { min, max, value })
}

fn parse_scl_classes(s: &str) -> Result<Vec<u8>> {
    s.split(',')
        .map(|c| {
            c.trim()
                .parse::<u8>()
                .with_context(|| format!("Invalid SCL class: {}", c))
        })
        .collect()
}

fn parse_pour_points(s: &str) -> Result<Vec<(usize, usize)>> {
    s.split(';')
        .map(|pair| {
            let parts: Vec<&str> = pair.trim().split(',').collect();
            if parts.len() != 2 {
                anyhow::bail!("Pour point must be 'row,col', got: {}", pair);
            }
            let row: usize = parts[0].trim().parse().context("Invalid row")?;
            let col: usize = parts[1].trim().parse().context("Invalid col")?;
            Ok((row, col))
        })
        .collect()
}

fn parse_advanced_curvature_type(s: &str) -> Result<AdvancedCurvatureType> {
    match s.to_lowercase().as_str() {
        "mean_h" | "mean" | "h" => Ok(AdvancedCurvatureType::MeanH),
        "gaussian_k" | "gaussian" | "k" => Ok(AdvancedCurvatureType::GaussianK),
        "kmin" | "minimal" => Ok(AdvancedCurvatureType::MinimalKmin),
        "kmax" | "maximal" => Ok(AdvancedCurvatureType::MaximalKmax),
        "kh" | "horizontal" => Ok(AdvancedCurvatureType::HorizontalKh),
        "kv" | "vertical" => Ok(AdvancedCurvatureType::VerticalKv),
        "khe" | "horizontal_excess" => Ok(AdvancedCurvatureType::HorizontalExcessKhe),
        "kve" | "vertical_excess" => Ok(AdvancedCurvatureType::VerticalExcessKve),
        "ka" | "accumulation" => Ok(AdvancedCurvatureType::AccumulationKa),
        "kr" | "ring" => Ok(AdvancedCurvatureType::RingKr),
        "rotor" => Ok(AdvancedCurvatureType::Rotor),
        "laplacian" => Ok(AdvancedCurvatureType::Laplacian),
        "unsphericity" | "m" => Ok(AdvancedCurvatureType::UnsphericitytM),
        "difference" | "e" => Ok(AdvancedCurvatureType::DifferenceE),
        _ => anyhow::bail!(
            "Unknown curvature type: {}. Use mean_h, gaussian_k, kmin, kmax, kh, kv, khe, kve, ka, kr, rotor, laplacian, unsphericity, difference.",
            s
        ),
    }
}

// ─── COG helpers ────────────────────────────────────────────────────────

#[cfg(feature = "cloud")]
fn parse_bbox(s: &str) -> Result<BBox> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        anyhow::bail!(
            "Bbox must be min_x,min_y,max_x,max_y (got {} parts)",
            parts.len()
        );
    }
    let min_x: f64 = parts[0].trim().parse().context("Invalid min_x")?;
    let min_y: f64 = parts[1].trim().parse().context("Invalid min_y")?;
    let max_x: f64 = parts[2].trim().parse().context("Invalid max_x")?;
    let max_y: f64 = parts[3].trim().parse().context("Invalid max_y")?;
    Ok(BBox::new(min_x, min_y, max_x, max_y))
}

#[cfg(feature = "cloud")]
fn read_cog_dem(url: &str, bbox: &BBox) -> Result<surtgis_core::Raster<f64>> {
    let pb = spinner("Fetching COG tiles...");
    let opts = CogReaderOptions::default();
    let raster: surtgis_core::Raster<f64> =
        read_cog(url, bbox, opts).context("Failed to read remote COG")?;
    pb.finish_and_clear();
    let (rows, cols) = raster.shape();
    info!(
        "Remote raster: {} x {} ({} cells)",
        cols,
        rows,
        raster.len()
    );
    Ok(raster)
}

/// Fetch a single asset from a STAC item as a raster.
#[cfg(feature = "cloud")]
/// Sentinel-2 band name aliases: common name → catalog-specific keys.
/// Tries the exact key first, then aliases.
#[cfg(feature = "cloud")]
fn resolve_asset_key<'a>(item: &'a StacItem, key: &'a str) -> Option<(&'a str, &'a surtgis_cloud::stac_models::StacAsset)> {
    // Try exact key first
    if let Some(asset) = item.asset(key) {
        return Some((key, asset));
    }

    // Alias table: common name ↔ Sentinel-2 band codes
    let aliases: &[(&str, &[&str])] = &[
        ("red",     &["B04", "b04", "Red"]),
        ("green",   &["B03", "b03", "Green"]),
        ("blue",    &["B02", "b02", "Blue"]),
        ("nir",     &["B08", "b08", "nir08", "Nir"]),
        ("nir08",   &["B08", "b08", "nir"]),
        ("nir09",   &["B09", "b09"]),
        ("rededge1",&["B05", "b05"]),
        ("rededge2",&["B06", "b06"]),
        ("rededge3",&["B07", "b07"]),
        ("swir16",  &["B11", "b11", "swir1", "SWIR1"]),
        ("swir22",  &["B12", "b12", "swir2", "SWIR2"]),
        ("scl",     &["SCL"]),
        ("coastal", &["B01", "b01"]),
        ("wvp",     &["B09", "b09"]),
        // Reverse: band code → common name
        ("B02",  &["blue", "Blue"]),
        ("B03",  &["green", "Green"]),
        ("B04",  &["red", "Red"]),
        ("B08",  &["nir", "nir08"]),
        ("B05",  &["rededge1"]),
        ("B06",  &["rededge2"]),
        ("B07",  &["rededge3"]),
        ("B11",  &["swir16", "swir1"]),
        ("B12",  &["swir22", "swir2"]),
        ("SCL",  &["scl"]),
    ];

    let key_lower = key.to_lowercase();
    for &(name, alt_keys) in aliases {
        if name.to_lowercase() == key_lower {
            for &alt in alt_keys {
                if let Some(asset) = item.asset(alt) {
                    return Some((alt, asset));
                }
            }
        }
    }
    None
}

/// Fetch a single asset from a STAC item as a raster.
#[cfg(feature = "cloud")]
fn fetch_stac_asset(
    item: &StacItem,
    asset_key: &str,
    bbox: &BBox,
    client: &StacClientBlocking,
) -> Result<surtgis_core::Raster<f64>> {
    let (resolved_key, stac_asset) = resolve_asset_key(item, asset_key)
        .ok_or_else(|| {
            let available: Vec<&str> = item.assets.keys().map(|k| k.as_str()).collect();
            anyhow::anyhow!(
                "Item {} missing asset '{}'. Available: {}",
                item.id, asset_key, available.join(", ")
            )
        })?;

    if resolved_key != asset_key {
        info!("Resolved asset '{}' → '{}'", asset_key, resolved_key);
    }

    let stac_asset = stac_asset.clone();

    let href = client
        .sign_asset_href(&stac_asset.href, item.collection.as_deref().unwrap_or(""))
        .context("Failed to sign asset URL")?;

    let opts = CogReaderOptions::default();
    let mut reader =
        CogReaderBlocking::open(&href, opts).context("Failed to open remote COG")?;

    // Auto-reproject bbox if COG is in a projected CRS
    let read_bb = {
        use surtgis_cloud::reproject;
        let epsg = item
            .epsg()
            .or_else(|| reader.metadata().crs.as_ref().and_then(|c| c.epsg()));
        if let Some(epsg) = epsg {
            if !reproject::is_wgs84(epsg) {
                reproject::reproject_bbox_to_cog(bbox, epsg)
            } else {
                *bbox
            }
        } else {
            *bbox
        }
    };

    let raster: surtgis_core::Raster<f64> = reader
        .read_bbox(&read_bb, None)
        .context("Failed to read bounding box from COG")?;
    Ok(raster)
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    let compress = cli.compress;
    let streaming = cli.streaming;
    let verbose = cli.verbose;
    setup_logging(verbose);

    match cli.command {
        // ── Info ─────────────────────────────────────────────────────
        Commands::Info { input } => {
            let raster = read_dem(&input)?;
            let (rows, cols) = raster.shape();
            let bounds = raster.bounds();
            let stats = raster.statistics();

            println!("File: {}", input.display());
            println!(
                "Dimensions: {} x {} ({} cells)",
                cols,
                rows,
                raster.len()
            );
            println!("Cell size: {}", raster.cell_size());
            println!(
                "Bounds: ({:.6}, {:.6}) - ({:.6}, {:.6})",
                bounds.0, bounds.1, bounds.2, bounds.3
            );
            if let Some(crs) = raster.crs() {
                println!("CRS: {}", crs);
            }
            if let Some(nodata) = raster.nodata() {
                println!("NoData: {}", nodata);
            }
            println!("\nStatistics:");
            if let Some(min) = stats.min {
                println!("  Min: {:.4}", min);
            }
            if let Some(max) = stats.max {
                println!("  Max: {:.4}", max);
            }
            if let Some(mean) = stats.mean {
                println!("  Mean: {:.4}", mean);
            }
            println!(
                "  Valid cells: {} ({:.1}%)",
                stats.valid_count,
                100.0 * stats.valid_count as f64 / raster.len() as f64
            );
        }

        // ── Terrain ──────────────────────────────────────────────────
        Commands::Terrain { algorithm } => match algorithm {
            TerrainCommands::Slope {
                input,
                output,
                units,
                z_factor,
            } => {
                let units = match units.to_lowercase().as_str() {
                    "degrees" | "deg" | "d" => SlopeUnits::Degrees,
                    "percent" | "pct" | "%" => SlopeUnits::Percent,
                    "radians" | "rad" | "r" => SlopeUnits::Radians,
                    _ => {
                        eprintln!("Unknown units: {}. Using degrees.", units);
                        SlopeUnits::Degrees
                    }
                };

                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = SlopeStreaming { units, z_factor };
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to calculate slope (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("Slope (streaming): {} x {}", cols, rows);
                    done("Slope", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result = slope(&dem, SlopeParams { units, z_factor })
                        .context("Failed to calculate slope")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("Slope", &output, elapsed);
                }
            }

            TerrainCommands::Aspect {
                input,
                output,
                format,
            } => {
                let fmt = match format.to_lowercase().as_str() {
                    "degrees" | "deg" | "d" => AspectOutput::Degrees,
                    "radians" | "rad" | "r" => AspectOutput::Radians,
                    "compass" | "c" => AspectOutput::Compass,
                    _ => {
                        eprintln!("Unknown format: {}. Using degrees.", format);
                        AspectOutput::Degrees
                    }
                };

                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = AspectStreaming { output_format: fmt };
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to calculate aspect (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("Aspect (streaming): {} x {}", cols, rows);
                    done("Aspect", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result = aspect(&dem, fmt).context("Failed to calculate aspect")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("Aspect", &output, elapsed);
                }
            }

            TerrainCommands::Hillshade {
                input,
                output,
                azimuth,
                altitude,
                z_factor,
            } => {
                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = HillshadeStreaming { azimuth, altitude, z_factor };
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to calculate hillshade (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("Hillshade (streaming): {} x {}", cols, rows);
                    done("Hillshade", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result = hillshade(
                        &dem,
                        HillshadeParams {
                            azimuth,
                            altitude,
                            z_factor,
                            normalized: false,
                        },
                    )
                    .context("Failed to calculate hillshade")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("Hillshade", &output, elapsed);
                }
            }

            TerrainCommands::Curvature {
                input,
                output,
                curvature_type,
                z_factor,
            } => {
                let ct = match curvature_type.to_lowercase().as_str() {
                    "general" | "mean" | "g" => CurvatureType::General,
                    "profile" | "prof" | "p" => CurvatureType::Profile,
                    "plan" | "tangential" | "t" => CurvatureType::Plan,
                    _ => {
                        eprintln!("Unknown type: {}. Using general.", curvature_type);
                        CurvatureType::General
                    }
                };
                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = CurvatureStreaming {
                        curvature_type: ct,
                        z_factor,
                        ..Default::default()
                    };
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to calculate curvature (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("Curvature (streaming): {} x {}", cols, rows);
                    done("Curvature", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result = curvature(
                        &dem,
                        CurvatureParams {
                            curvature_type: ct,
                            z_factor,
                            ..Default::default()
                        },
                    )
                    .context("Failed to calculate curvature")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("Curvature", &output, elapsed);
                }
            }

            TerrainCommands::Tpi {
                input,
                output,
                radius,
            } => {
                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = TpiStreaming { radius };
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to calculate TPI (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("TPI (streaming): {} x {}", cols, rows);
                    done("TPI", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result =
                        tpi(&dem, TpiParams { radius }).context("Failed to calculate TPI")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("TPI", &output, elapsed);
                }
            }

            TerrainCommands::Tri {
                input,
                output,
                radius,
            } => {
                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = TriStreaming { radius };
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to calculate TRI (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("TRI (streaming): {} x {}", cols, rows);
                    done("TRI", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result =
                        tri(&dem, TriParams { radius }).context("Failed to calculate TRI")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("TRI", &output, elapsed);
                }
            }

            TerrainCommands::Landform {
                input,
                output,
                small_radius,
                large_radius,
                threshold,
                slope_threshold,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = landform_classification(
                    &dem,
                    LandformParams {
                        small_radius,
                        large_radius,
                        tpi_threshold: threshold,
                        slope_threshold,
                    },
                )
                .context("Failed to classify landforms")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Landform classification", &output, elapsed);
            }

            TerrainCommands::Geomorphons {
                input,
                output,
                radius,
                flatness,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = geomorphons(
                    &dem,
                    GeomorphonParams {
                        radius,
                        flatness_threshold: flatness,
                    },
                )
                .context("Failed to compute geomorphons")?;
                let elapsed = start.elapsed();
                write_result_u8(&result, &output, compress)?;
                done("Geomorphons", &output, elapsed);
            }

            TerrainCommands::Northness { input, output } => {
                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = NorthnessStreaming;
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to compute northness (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("Northness (streaming): {} x {}", cols, rows);
                    done("Northness", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result = northness(&dem).context("Failed to compute northness")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("Northness", &output, elapsed);
                }
            }

            TerrainCommands::Eastness { input, output } => {
                // Auto-detect streaming for large files
                let use_streaming = streaming || std::fs::metadata(&input)
                    .map(|m| m.len() > 500_000_000).unwrap_or(false);

                if use_streaming {
                    let algo = EastnessStreaming;
                    let processor = StripProcessor::new(256);
                    let start = Instant::now();
                    let (rows, cols) = processor.process(&input, &output, &algo, compress)
                        .context("Failed to compute eastness (streaming)")?;
                    let elapsed = start.elapsed();
                    println!("Eastness (streaming): {} x {}", cols, rows);
                    done("Eastness", &output, elapsed);
                } else {
                    let dem = read_dem(&input)?;
                    let start = Instant::now();
                    let result = eastness(&dem).context("Failed to compute eastness")?;
                    let elapsed = start.elapsed();
                    write_result(&result, &output, compress)?;
                    done("Eastness", &output, elapsed);
                }
            }

            TerrainCommands::OpennessPositive {
                input,
                output,
                radius,
                directions,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    positive_openness(&dem, OpennessParams { radius, directions })
                        .context("Failed to compute positive openness")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Positive openness", &output, elapsed);
            }

            TerrainCommands::OpennessNegative {
                input,
                output,
                radius,
                directions,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    negative_openness(&dem, OpennessParams { radius, directions })
                        .context("Failed to compute negative openness")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Negative openness", &output, elapsed);
            }

            TerrainCommands::Svf {
                input,
                output,
                radius,
                directions,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = sky_view_factor(&dem, SvfParams { radius, directions })
                    .context("Failed to compute SVF")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Sky View Factor", &output, elapsed);
            }

            TerrainCommands::Mrvbf {
                input,
                output,
                mrrtf_output,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let (mrvbf_r, mrrtf_r) =
                    mrvbf(&dem, MrvbfParams::default()).context("Failed to compute MRVBF")?;
                let elapsed = start.elapsed();
                write_result(&mrvbf_r, &output, compress)?;
                done("MRVBF", &output, elapsed);
                if let Some(mrrtf_path) = mrrtf_output {
                    write_result(&mrrtf_r, &mrrtf_path, compress)?;
                    println!("MRRTF saved to: {}", mrrtf_path.display());
                }
            }

            TerrainCommands::Dev {
                input,
                output,
                radius,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    dev(&dem, DevParams { radius }).context("Failed to compute DEV")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("DEV", &output, elapsed);
            }

            TerrainCommands::Vrm {
                input,
                output,
                radius,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    vrm(&dem, VrmParams { radius }).context("Failed to compute VRM")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("VRM", &output, elapsed);
            }

            TerrainCommands::AdvancedCurvature {
                input,
                output,
                curvature_type,
            } => {
                let ct = parse_advanced_curvature_type(&curvature_type)?;
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = advanced_curvatures(&dem, ct)
                    .context("Failed to compute advanced curvature")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Advanced curvature", &output, elapsed);
            }

            TerrainCommands::Viewshed {
                input,
                output,
                observer_row,
                observer_col,
                observer_height,
                target_height,
                max_radius,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = viewshed(
                    &dem,
                    ViewshedParams {
                        observer_row,
                        observer_col,
                        observer_height,
                        target_height,
                        max_radius,
                    },
                )
                .context("Failed to compute viewshed")?;
                let elapsed = start.elapsed();
                write_result_u8(&result, &output, compress)?;
                done("Viewshed", &output, elapsed);
            }

            TerrainCommands::Convergence {
                input,
                output,
                radius,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = convergence_index(&dem, ConvergenceParams { radius })
                    .context("Failed to compute convergence index")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Convergence index", &output, elapsed);
            }

            TerrainCommands::MultiHillshade { input, output } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    multidirectional_hillshade(&dem, MultiHillshadeParams::default())
                        .context("Failed to compute multi-directional hillshade")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Multi-directional hillshade", &output, elapsed);
            }

            TerrainCommands::All { input, outdir } => {
                std::fs::create_dir_all(&outdir)
                    .context("Failed to create output directory")?;
                let dem = read_dem(&input)?;
                let start = Instant::now();

                println!("Computing all terrain factors...");

                let s = slope(
                    &dem,
                    SlopeParams {
                        units: SlopeUnits::Degrees,
                        z_factor: 1.0,
                    },
                )
                .context("slope")?;
                write_result(&s, &outdir.join("slope.tif"), compress)?;
                println!("  slope.tif");

                let a = aspect(&dem, AspectOutput::Degrees).context("aspect")?;
                write_result(&a, &outdir.join("aspect.tif"), compress)?;
                println!("  aspect.tif");

                let h = hillshade(
                    &dem,
                    HillshadeParams {
                        azimuth: 315.0,
                        altitude: 45.0,
                        z_factor: 1.0,
                        normalized: false,
                    },
                )
                .context("hillshade")?;
                write_result(&h, &outdir.join("hillshade.tif"), compress)?;
                println!("  hillshade.tif");

                let n = northness(&dem).context("northness")?;
                write_result(&n, &outdir.join("northness.tif"), compress)?;
                println!("  northness.tif");

                let e = eastness(&dem).context("eastness")?;
                write_result(&e, &outdir.join("eastness.tif"), compress)?;
                println!("  eastness.tif");

                let c = curvature(
                    &dem,
                    CurvatureParams {
                        curvature_type: CurvatureType::General,
                        z_factor: 1.0,
                        ..Default::default()
                    },
                )
                .context("curvature")?;
                write_result(&c, &outdir.join("curvature.tif"), compress)?;
                println!("  curvature.tif");

                let t = tpi(&dem, TpiParams { radius: 10 }).context("tpi")?;
                write_result(&t, &outdir.join("tpi.tif"), compress)?;
                println!("  tpi.tif");

                let tr = tri(&dem, TriParams { radius: 1 }).context("tri")?;
                write_result(&tr, &outdir.join("tri.tif"), compress)?;
                println!("  tri.tif");

                let g = geomorphons(
                    &dem,
                    GeomorphonParams {
                        radius: 10,
                        flatness_threshold: 1.0,
                    },
                )
                .context("geomorphons")?;
                write_result_u8(&g, &outdir.join("geomorphons.tif"), compress)?;
                println!("  geomorphons.tif");

                let d = dev(&dem, DevParams { radius: 10 }).context("dev")?;
                write_result(&d, &outdir.join("dev.tif"), compress)?;
                println!("  dev.tif");

                let v = vrm(&dem, VrmParams { radius: 1 }).context("vrm")?;
                write_result(&v, &outdir.join("vrm.tif"), compress)?;
                println!("  vrm.tif");

                let ci =
                    convergence_index(&dem, ConvergenceParams { radius: 3 }).context("convergence")?;
                write_result(&ci, &outdir.join("convergence.tif"), compress)?;
                println!("  convergence.tif");

                let op = positive_openness(&dem, OpennessParams { radius: 10, directions: 8 })
                    .context("openness_positive")?;
                write_result(&op, &outdir.join("openness_positive.tif"), compress)?;
                println!("  openness_positive.tif");

                let on = negative_openness(&dem, OpennessParams { radius: 10, directions: 8 })
                    .context("openness_negative")?;
                write_result(&on, &outdir.join("openness_negative.tif"), compress)?;
                println!("  openness_negative.tif");

                let svf_r = sky_view_factor(&dem, SvfParams { radius: 10, directions: 16 })
                    .context("svf")?;
                write_result(&svf_r, &outdir.join("svf.tif"), compress)?;
                println!("  svf.tif");

                let (mrvbf_r, mrrtf_r) = mrvbf(&dem, MrvbfParams::default()).context("mrvbf")?;
                write_result(&mrvbf_r, &outdir.join("mrvbf.tif"), compress)?;
                write_result(&mrrtf_r, &outdir.join("mrrtf.tif"), compress)?;
                println!("  mrvbf.tif, mrrtf.tif");

                let elapsed = start.elapsed();
                println!(
                    "\nAll terrain factors saved to: {}",
                    outdir.display()
                );
                println!("  17 products, processing time: {:.2?}", elapsed);
            }
        },

        // ── Hydrology ────────────────────────────────────────────────
        Commands::Hydrology { algorithm } => match algorithm {
            HydrologyCommands::FillSinks {
                input,
                output,
                min_slope,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = fill_sinks(&dem, FillSinksParams { min_slope })
                    .context("Failed to fill sinks")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Fill sinks", &output, elapsed);
            }

            HydrologyCommands::FlowDirection { input, output } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    flow_direction(&dem).context("Failed to calculate flow direction")?;
                let elapsed = start.elapsed();
                write_result_u8(&result, &output, compress)?;
                done("Flow direction", &output, elapsed);
            }

            HydrologyCommands::FlowAccumulation { input, output } => {
                let flow_dir = read_u8(&input)?;
                let start = Instant::now();
                let result = flow_accumulation(&flow_dir)
                    .context("Failed to calculate flow accumulation")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Flow accumulation", &output, elapsed);
            }

            HydrologyCommands::Watershed {
                input,
                output,
                pour_points,
            } => {
                let points = parse_pour_points(&pour_points)?;
                if points.is_empty() {
                    anyhow::bail!("At least one pour point is required");
                }
                let flow_dir = read_u8(&input)?;
                let start = Instant::now();
                let result = watershed(
                    &flow_dir,
                    WatershedParams {
                        pour_points: points,
                    },
                )
                .context("Failed to delineate watersheds")?;
                let elapsed = start.elapsed();
                write_result_i32(&result, &output, compress)?;
                done("Watershed", &output, elapsed);
            }

            HydrologyCommands::PriorityFlood {
                input,
                output,
                epsilon,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = priority_flood(&dem, PriorityFloodParams { epsilon })
                    .context("Failed to run priority flood")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Priority flood", &output, elapsed);
            }

            HydrologyCommands::Breach {
                input,
                output,
                max_depth,
                max_length,
                fill_remaining,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = breach_depressions(
                    &dem,
                    BreachParams {
                        max_depth,
                        max_length,
                        fill_remaining,
                    },
                )
                .context("Failed to breach depressions")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Breach depressions", &output, elapsed);
            }

            HydrologyCommands::FlowDirectionDinf { input, output } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = flow_direction_dinf(&dem)
                    .context("Failed to compute D-infinity flow direction")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Flow direction (D-inf)", &output, elapsed);
            }

            HydrologyCommands::FlowAccumulationMfd {
                input,
                output,
                exponent,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = flow_accumulation_mfd(&dem, MfdParams { exponent })
                    .context("Failed to compute MFD accumulation")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Flow accumulation (MFD)", &output, elapsed);
            }

            HydrologyCommands::Twi { input, output } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                // Internal pipeline: fill -> flow_dir -> flow_acc -> slope -> twi
                let filled =
                    priority_flood(&dem, PriorityFloodParams { epsilon: 0.0001 })
                        .context("Failed to fill depressions")?;
                let fdir =
                    flow_direction(&filled).context("Failed to compute flow direction")?;
                let facc = flow_accumulation(&fdir)
                    .context("Failed to compute flow accumulation")?;
                let slope_rad = slope(
                    &filled,
                    SlopeParams {
                        units: SlopeUnits::Radians,
                        z_factor: 1.0,
                    },
                )
                .context("Failed to compute slope")?;
                let result = twi(&facc, &slope_rad).context("Failed to compute TWI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("TWI", &output, elapsed);
            }

            HydrologyCommands::Hand {
                input,
                output,
                threshold,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let filled =
                    priority_flood(&dem, PriorityFloodParams { epsilon: 0.0001 })
                        .context("Failed to fill depressions")?;
                let fdir =
                    flow_direction(&filled).context("Failed to compute flow direction")?;
                let facc = flow_accumulation(&fdir)
                    .context("Failed to compute flow accumulation")?;
                let result = hand(
                    &dem,
                    &fdir,
                    &facc,
                    HandParams {
                        stream_threshold: threshold,
                    },
                )
                .context("Failed to compute HAND")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("HAND", &output, elapsed);
            }

            HydrologyCommands::StreamNetwork {
                input,
                output,
                threshold,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let filled =
                    priority_flood(&dem, PriorityFloodParams { epsilon: 0.0001 })
                        .context("Failed to fill depressions")?;
                let fdir =
                    flow_direction(&filled).context("Failed to compute flow direction")?;
                let facc = flow_accumulation(&fdir)
                    .context("Failed to compute flow accumulation")?;
                let result = stream_network(&facc, StreamNetworkParams { threshold })
                    .context("Failed to extract stream network")?;
                let elapsed = start.elapsed();
                write_result_u8(&result, &output, compress)?;
                done("Stream network", &output, elapsed);
            }

            HydrologyCommands::All {
                input,
                outdir,
                threshold,
            } => {
                std::fs::create_dir_all(&outdir)
                    .context("Failed to create output directory")?;
                let dem = read_dem(&input)?;
                let start = Instant::now();

                println!("Computing full hydrology pipeline...");

                let filled =
                    priority_flood(&dem, PriorityFloodParams { epsilon: 0.0001 })
                        .context("fill")?;
                write_result(&filled, &outdir.join("filled.tif"), compress)?;
                println!("  filled.tif");

                let fdir = flow_direction(&filled).context("flow direction")?;
                write_result_u8(&fdir, &outdir.join("flow_direction_d8.tif"), compress)?;
                println!("  flow_direction_d8.tif");

                let fdir_dinf =
                    flow_direction_dinf(&filled).context("flow direction dinf")?;
                write_result(
                    &fdir_dinf,
                    &outdir.join("flow_direction_dinf.tif"),
                    compress,
                )?;
                println!("  flow_direction_dinf.tif");

                let facc = flow_accumulation(&fdir).context("flow accumulation")?;
                write_result(&facc, &outdir.join("flow_accumulation.tif"), compress)?;
                println!("  flow_accumulation.tif");

                let facc_mfd = flow_accumulation_mfd(&filled, MfdParams { exponent: 1.1 })
                    .context("mfd accumulation")?;
                write_result(
                    &facc_mfd,
                    &outdir.join("flow_accumulation_mfd.tif"),
                    compress,
                )?;
                println!("  flow_accumulation_mfd.tif");

                let slope_rad = slope(
                    &filled,
                    SlopeParams {
                        units: SlopeUnits::Radians,
                        z_factor: 1.0,
                    },
                )
                .context("slope")?;
                let twi_r = twi(&facc, &slope_rad).context("twi")?;
                write_result(&twi_r, &outdir.join("twi.tif"), compress)?;
                println!("  twi.tif");

                let streams = stream_network(&facc, StreamNetworkParams { threshold })
                    .context("streams")?;
                write_result_u8(&streams, &outdir.join("stream_network.tif"), compress)?;
                println!("  stream_network.tif");

                let hand_r = hand(
                    &dem,
                    &fdir,
                    &facc,
                    HandParams {
                        stream_threshold: threshold,
                    },
                )
                .context("hand")?;
                write_result(&hand_r, &outdir.join("hand.tif"), compress)?;
                println!("  hand.tif");

                let elapsed = start.elapsed();
                println!(
                    "\nFull hydrology pipeline saved to: {}",
                    outdir.display()
                );
                println!("  8 products, processing time: {:.2?}", elapsed);
            }
        },

        // ── Imagery ──────────────────────────────────────────────────
        Commands::Imagery { algorithm } => match algorithm {
            ImageryCommands::Ndvi { nir, red, output } => {
                let nir_r = read_dem(&nir)?;
                let red_r = read_dem(&red)?;
                let start = Instant::now();
                let result = ndvi(&nir_r, &red_r).context("Failed to calculate NDVI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NDVI", &output, elapsed);
            }

            ImageryCommands::Ndwi { green, nir, output } => {
                let green_r = read_dem(&green)?;
                let nir_r = read_dem(&nir)?;
                let start = Instant::now();
                let result = ndwi(&green_r, &nir_r).context("Failed to calculate NDWI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NDWI", &output, elapsed);
            }

            ImageryCommands::Mndwi {
                green,
                swir,
                output,
            } => {
                let green_r = read_dem(&green)?;
                let swir_r = read_dem(&swir)?;
                let start = Instant::now();
                let result =
                    mndwi(&green_r, &swir_r).context("Failed to calculate MNDWI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("MNDWI", &output, elapsed);
            }

            ImageryCommands::Nbr { nir, swir, output } => {
                let nir_r = read_dem(&nir)?;
                let swir_r = read_dem(&swir)?;
                let start = Instant::now();
                let result = nbr(&nir_r, &swir_r).context("Failed to calculate NBR")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NBR", &output, elapsed);
            }

            ImageryCommands::Savi {
                nir,
                red,
                output,
                l_factor,
            } => {
                let nir_r = read_dem(&nir)?;
                let red_r = read_dem(&red)?;
                let start = Instant::now();
                let result = savi(&nir_r, &red_r, SaviParams { l_factor })
                    .context("Failed to calculate SAVI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("SAVI", &output, elapsed);
            }

            ImageryCommands::Evi {
                nir,
                red,
                blue,
                output,
            } => {
                let nir_r = read_dem(&nir)?;
                let red_r = read_dem(&red)?;
                let blue_r = read_dem(&blue)?;
                let start = Instant::now();
                let result = evi(&nir_r, &red_r, &blue_r, EviParams::default())
                    .context("Failed to calculate EVI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("EVI", &output, elapsed);
            }

            ImageryCommands::Bsi {
                swir,
                red,
                nir,
                blue,
                output,
            } => {
                let swir_r = read_dem(&swir)?;
                let red_r = read_dem(&red)?;
                let nir_r = read_dem(&nir)?;
                let blue_r = read_dem(&blue)?;
                let start = Instant::now();
                let result = bsi(&swir_r, &red_r, &nir_r, &blue_r)
                    .context("Failed to calculate BSI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("BSI", &output, elapsed);
            }

            ImageryCommands::BandMath { a, b, output, op } => {
                let op = parse_band_math_op(&op)?;
                let a_r = read_dem(&a)?;
                let b_r = read_dem(&b)?;
                let start = Instant::now();
                let result = band_math_binary(&a_r, &b_r, op)
                    .context("Failed to perform band math")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Band math", &output, elapsed);
            }

            ImageryCommands::Calc {
                expression,
                band,
                output,
            } => {
                let assignments = parse_band_assignments(&band)?;
                let mut rasters: Vec<(String, surtgis_core::Raster<f64>)> = Vec::new();
                for (name, path) in &assignments {
                    let r = read_dem(path)?;
                    rasters.push((name.clone(), r));
                }
                let band_refs: HashMap<&str, &surtgis_core::Raster<f64>> = rasters
                    .iter()
                    .map(|(name, r)| (name.as_str(), r))
                    .collect();
                let start = Instant::now();
                let result = index_builder(&expression, &band_refs)
                    .context("Failed to evaluate expression")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Calc", &output, elapsed);
            }

            ImageryCommands::Evi2 { nir, red, output } => {
                let nir_r = read_dem(&nir)?;
                let red_r = read_dem(&red)?;
                let start = Instant::now();
                let result = evi2(&nir_r, &red_r).context("Failed to calculate EVI2")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("EVI2", &output, elapsed);
            }

            ImageryCommands::Gndvi { nir, green, output } => {
                let nir_r = read_dem(&nir)?;
                let green_r = read_dem(&green)?;
                let start = Instant::now();
                let result = gndvi(&nir_r, &green_r).context("Failed to calculate GNDVI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("GNDVI", &output, elapsed);
            }

            ImageryCommands::Ngrdi { green, red, output } => {
                let green_r = read_dem(&green)?;
                let red_r = read_dem(&red)?;
                let start = Instant::now();
                let result = ngrdi(&green_r, &red_r).context("Failed to calculate NGRDI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NGRDI", &output, elapsed);
            }

            ImageryCommands::Reci { nir, red_edge, output } => {
                let nir_r = read_dem(&nir)?;
                let re_r = read_dem(&red_edge)?;
                let start = Instant::now();
                let result = reci(&nir_r, &re_r).context("Failed to calculate RECI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("RECI", &output, elapsed);
            }

            ImageryCommands::Ndre { nir, red_edge, output } => {
                let nir_r = read_dem(&nir)?;
                let re_r = read_dem(&red_edge)?;
                let start = Instant::now();
                let result = ndre(&nir_r, &re_r).context("Failed to calculate NDRE")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NDRE", &output, elapsed);
            }

            ImageryCommands::Ndsi { green, swir, output } => {
                let green_r = read_dem(&green)?;
                let swir_r = read_dem(&swir)?;
                let start = Instant::now();
                let result = ndsi(&green_r, &swir_r).context("Failed to calculate NDSI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NDSI", &output, elapsed);
            }

            ImageryCommands::Ndmi { nir, swir, output } => {
                let nir_r = read_dem(&nir)?;
                let swir_r = read_dem(&swir)?;
                let start = Instant::now();
                let result = ndmi(&nir_r, &swir_r).context("Failed to calculate NDMI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NDMI", &output, elapsed);
            }

            ImageryCommands::Ndbi { swir, nir, output } => {
                let swir_r = read_dem(&swir)?;
                let nir_r = read_dem(&nir)?;
                let start = Instant::now();
                let result = ndbi(&swir_r, &nir_r).context("Failed to calculate NDBI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("NDBI", &output, elapsed);
            }

            ImageryCommands::Msavi { nir, red, output } => {
                let nir_r = read_dem(&nir)?;
                let red_r = read_dem(&red)?;
                let start = Instant::now();
                let result = msavi(&nir_r, &red_r).context("Failed to calculate MSAVI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("MSAVI", &output, elapsed);
            }

            ImageryCommands::Reclassify {
                input,
                output,
                class,
                default,
            } => {
                let classes: Vec<ReclassEntry> = class
                    .iter()
                    .map(|s| parse_reclass_entry(s))
                    .collect::<Result<Vec<_>>>()?;
                let default_value: f64 = if default.to_lowercase() == "nan" {
                    f64::NAN
                } else {
                    default.parse().context("Invalid default value")?
                };
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result = reclassify(
                    &raster,
                    ReclassifyParams {
                        classes,
                        default_value,
                    },
                )
                .context("Failed to reclassify")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Reclassify", &output, elapsed);
            }

            ImageryCommands::MedianComposite { input, output } => {
                if input.len() < 2 {
                    anyhow::bail!("median-composite requires at least 2 input rasters");
                }
                let rasters: Vec<surtgis_core::Raster<f64>> = input
                    .iter()
                    .map(|p| {
                        read_dem(p)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
                let start = Instant::now();
                let result =
                    median_composite(&refs).context("Failed to compute median composite")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                println!("  {} input rasters composited", input.len());
                done("Median composite", &output, elapsed);
            }

            ImageryCommands::CloudMask {
                input,
                scl,
                output,
                keep,
            } => {
                let data = read_dem(&input)?;
                let scl_r = read_dem(&scl)?;
                let classes = parse_scl_classes(&keep)?;
                let start = Instant::now();
                let result = cloud_mask_scl(&data, &scl_r, &classes)
                    .context("Failed to apply cloud mask")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Cloud mask", &output, elapsed);
            }
        },

        // ── Morphology ───────────────────────────────────────────────
        Commands::Morphology { algorithm } => match algorithm {
            MorphologyCommands::Erode {
                input,
                output,
                shape,
                radius,
            } => {
                let se = parse_se(&shape, radius)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result = erode(&raster, &se).context("Failed to erode")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Erode", &output, elapsed);
            }

            MorphologyCommands::Dilate {
                input,
                output,
                shape,
                radius,
            } => {
                let se = parse_se(&shape, radius)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result = dilate(&raster, &se).context("Failed to dilate")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Dilate", &output, elapsed);
            }

            MorphologyCommands::Opening {
                input,
                output,
                shape,
                radius,
            } => {
                let se = parse_se(&shape, radius)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result = opening(&raster, &se).context("Failed to open")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Opening", &output, elapsed);
            }

            MorphologyCommands::Closing {
                input,
                output,
                shape,
                radius,
            } => {
                let se = parse_se(&shape, radius)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result = closing(&raster, &se).context("Failed to close")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Closing", &output, elapsed);
            }

            MorphologyCommands::Gradient {
                input,
                output,
                shape,
                radius,
            } => {
                let se = parse_se(&shape, radius)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    gradient(&raster, &se).context("Failed to compute gradient")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Gradient", &output, elapsed);
            }

            MorphologyCommands::TopHat {
                input,
                output,
                shape,
                radius,
            } => {
                let se = parse_se(&shape, radius)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    top_hat(&raster, &se).context("Failed to compute top-hat")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Top-hat", &output, elapsed);
            }

            MorphologyCommands::BlackHat {
                input,
                output,
                shape,
                radius,
            } => {
                let se = parse_se(&shape, radius)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    black_hat(&raster, &se).context("Failed to compute black-hat")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Black-hat", &output, elapsed);
            }
        },

        // ── Landscape ───────────────────────────────────────────────
        Commands::Landscape { algorithm } => match algorithm {
            LandscapeCommands::LabelPatches {
                input,
                output,
                connectivity,
            } => {
                let conn = parse_connectivity(connectivity)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let (labels, num_patches) = label_patches(&raster, conn)
                    .context("Failed to label patches")?;
                let elapsed = start.elapsed();
                write_result_i32(&labels, &output, compress)?;
                println!("{} patches found", num_patches);
                done("Label patches", &output, elapsed);
            }

            LandscapeCommands::PatchMetrics {
                input,
                output,
                connectivity,
            } => {
                let conn = parse_connectivity(connectivity)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let (labels, num_patches) = label_patches(&raster, conn)
                    .context("Failed to label patches")?;
                let patches = patch_metrics(&raster, &labels, num_patches)
                    .context("Failed to compute patch metrics")?;
                let elapsed = start.elapsed();
                let csv = patches_to_csv(&patches);
                std::fs::write(&output, &csv).context("Failed to write CSV")?;
                println!("{} patches, {} classes", patches.len(),
                    patches.iter().map(|p| p.class).collect::<std::collections::HashSet<_>>().len());
                done("Patch metrics", &output, elapsed);
            }

            LandscapeCommands::ClassMetrics {
                input,
                connectivity,
            } => {
                let conn = parse_connectivity(connectivity)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let (labels, num_patches) = label_patches(&raster, conn)
                    .context("Failed to label patches")?;
                let patches = patch_metrics(&raster, &labels, num_patches)
                    .context("Failed to compute patch metrics")?;
                let cm = class_metrics(&raster, &patches)
                    .context("Failed to compute class metrics")?;
                let elapsed = start.elapsed();

                println!("{:<10} {:>10} {:>10} {:>8} {:>12} {:>8} {:>10}",
                    "Class", "Area(m²)", "Proportion", "Patches", "MeanArea", "AI", "Cohesion");
                println!("{}", "-".repeat(78));
                for c in &cm {
                    println!("{:<10} {:>10.1} {:>10.4} {:>8} {:>12.1} {:>8.1} {:>10.1}",
                        c.class, c.area_m2, c.proportion, c.num_patches,
                        c.mean_patch_area_m2, c.ai, c.cohesion);
                }
                println!("\n  Processing time: {:.2?}", elapsed);
            }

            LandscapeCommands::LandscapeMetrics { input } => {
                let raster = read_dem(&input)?;
                let start = Instant::now();
                let lm = landscape_metrics(&raster)
                    .context("Failed to compute landscape metrics")?;
                let elapsed = start.elapsed();

                println!("Landscape Metrics:");
                println!("  SHDI (Shannon):  {:.4}", lm.shdi);
                println!("  SIDI (Simpson):  {:.4}", lm.sidi);
                println!("  Classes:         {}", lm.num_classes);
                println!("  Total cells:     {}", lm.total_cells);
                println!("  Total area:      {:.1} m²", lm.total_area_m2);
                println!("  Processing time: {:.2?}", elapsed);
            }

            LandscapeCommands::Analyze {
                input,
                output_labels,
                output_csv,
                connectivity,
            } => {
                let conn = parse_connectivity(connectivity)?;
                let raster = read_dem(&input)?;
                let start = Instant::now();

                // 1. Label patches
                let (labels, num_patches) = label_patches(&raster, conn)
                    .context("Failed to label patches")?;
                println!("Patches: {}", num_patches);

                // 2. Patch metrics
                let patches = patch_metrics(&raster, &labels, num_patches)
                    .context("Failed to compute patch metrics")?;

                // 3. Class metrics
                let cm = class_metrics(&raster, &patches)
                    .context("Failed to compute class metrics")?;

                println!("\n{:<10} {:>10} {:>10} {:>8} {:>8} {:>10}",
                    "Class", "Proportion", "Patches", "AI", "Cohesion", "MeanArea");
                println!("{}", "-".repeat(66));
                for c in &cm {
                    println!("{:<10} {:>10.4} {:>8} {:>8.1} {:>10.1} {:>10.1}",
                        c.class, c.proportion, c.num_patches, c.ai, c.cohesion,
                        c.mean_patch_area_m2);
                }

                // 4. Landscape metrics
                let lm = landscape_metrics(&raster)
                    .context("Failed to compute landscape metrics")?;
                println!("\nLandscape: SHDI={:.4}  SIDI={:.4}  classes={}",
                    lm.shdi, lm.sidi, lm.num_classes);

                // 5. Write outputs
                if let Some(ref lp) = output_labels {
                    write_result_i32(&labels, lp, compress)?;
                    println!("Labels saved to: {}", lp.display());
                }
                if let Some(ref cp) = output_csv {
                    let csv = patches_to_csv(&patches);
                    std::fs::write(cp, &csv).context("Failed to write CSV")?;
                    println!("Patch CSV saved to: {}", cp.display());
                }

                let elapsed = start.elapsed();
                println!("\n  Processing time: {:.2?}", elapsed);
            }
        },

        // ── Mosaic ───────────────────────────────────────────────────
        Commands::Mosaic { input, output } => {
            if input.len() < 2 {
                anyhow::bail!("mosaic requires at least 2 input rasters");
            }
            let rasters: Vec<surtgis_core::Raster<f64>> = input
                .iter()
                .map(|p| read_dem(p))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
            let start = Instant::now();
            let result = surtgis_core::mosaic(&refs, None)
                .context("Failed to mosaic rasters")?;
            let elapsed = start.elapsed();
            let (rows, cols) = result.shape();
            write_result(&result, &output, compress)?;
            println!(
                "Mosaic: {} tiles → {} x {} ({} cells)",
                input.len(),
                cols,
                rows,
                result.len()
            );
            done("Mosaic", &output, elapsed);
        }

        // ── COG (remote processing) ───────────────────────────────────
        #[cfg(feature = "cloud")]
        Commands::Cog { action } => match action {
            CogCommands::Info { url } => {
                let pb = spinner("Opening remote COG...");
                let opts = CogReaderOptions::default();
                let reader = CogReaderBlocking::open(&url, opts)
                    .context("Failed to open remote COG")?;
                pb.finish_and_clear();

                let meta = reader.metadata();
                let ovr = reader.overviews();

                println!("URL: {}", meta.url);
                println!(
                    "Dimensions: {} x {} ({} cells)",
                    meta.width,
                    meta.height,
                    meta.width as u64 * meta.height as u64
                );
                println!("Tile size: {} x {}", meta.tile_width, meta.tile_height);
                println!(
                    "Bits/sample: {}, Sample format: {}",
                    meta.bits_per_sample, meta.sample_format
                );
                println!(
                    "Compression: {}",
                    match meta.compression {
                        1 => "None",
                        5 => "LZW",
                        8 | 32946 => "DEFLATE",
                        _ => "Other",
                    }
                );
                let gt = &meta.geo_transform;
                println!("Origin: ({:.6}, {:.6})", gt.origin_x, gt.origin_y);
                println!(
                    "Pixel size: ({:.6}, {:.6})",
                    gt.pixel_width, gt.pixel_height
                );
                if let Some(crs) = &meta.crs {
                    println!("CRS: {}", crs);
                }
                if let Some(nd) = meta.nodata {
                    println!("NoData: {}", nd);
                }
                if !ovr.is_empty() {
                    println!("Overviews ({}):", ovr.len());
                    for o in &ovr {
                        println!("  [{}] {} x {}", o.index, o.width, o.height);
                    }
                }
            }

            CogCommands::Fetch {
                url,
                output,
                bbox,
                overview,
            } => {
                let bbox = parse_bbox(&bbox)?;
                let pb = spinner("Fetching COG tiles...");
                let opts = CogReaderOptions::default();
                let mut reader = CogReaderBlocking::open(&url, opts)
                    .context("Failed to open remote COG")?;
                let start = Instant::now();
                let raster: surtgis_core::Raster<f64> = reader
                    .read_bbox(&bbox, overview)
                    .context("Failed to read bounding box")?;
                pb.finish_and_clear();
                let elapsed = start.elapsed();
                let (rows, cols) = raster.shape();
                println!("Fetched: {} x {} ({} cells)", cols, rows, raster.len());
                write_result(&raster, &output, compress)?;
                done("COG fetch", &output, elapsed);
            }

            CogCommands::Slope {
                url,
                output,
                bbox,
                units,
                z_factor,
            } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let units = match units.to_lowercase().as_str() {
                    "degrees" | "deg" | "d" => SlopeUnits::Degrees,
                    "percent" | "pct" | "%" => SlopeUnits::Percent,
                    "radians" | "rad" | "r" => SlopeUnits::Radians,
                    _ => {
                        eprintln!("Unknown units: {}. Using degrees.", units);
                        SlopeUnits::Degrees
                    }
                };
                let start = Instant::now();
                let result = slope(&dem, SlopeParams { units, z_factor })
                    .context("Failed to calculate slope")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("COG slope", &output, elapsed);
            }

            CogCommands::Aspect {
                url,
                output,
                bbox,
                format,
            } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let fmt = match format.to_lowercase().as_str() {
                    "degrees" | "deg" | "d" => AspectOutput::Degrees,
                    "radians" | "rad" | "r" => AspectOutput::Radians,
                    "compass" | "c" => AspectOutput::Compass,
                    _ => {
                        eprintln!("Unknown format: {}. Using degrees.", format);
                        AspectOutput::Degrees
                    }
                };
                let start = Instant::now();
                let result =
                    aspect(&dem, fmt).context("Failed to calculate aspect")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("COG aspect", &output, elapsed);
            }

            CogCommands::Hillshade {
                url,
                output,
                bbox,
                azimuth,
                altitude,
                z_factor,
            } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let start = Instant::now();
                let result = hillshade(
                    &dem,
                    HillshadeParams {
                        azimuth,
                        altitude,
                        z_factor,
                        normalized: false,
                    },
                )
                .context("Failed to calculate hillshade")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("COG hillshade", &output, elapsed);
            }

            CogCommands::Tpi {
                url,
                output,
                bbox,
                radius,
            } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let start = Instant::now();
                let result = tpi(&dem, TpiParams { radius })
                    .context("Failed to calculate TPI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("COG TPI", &output, elapsed);
            }

            CogCommands::FillSinks {
                url,
                output,
                bbox,
                min_slope,
            } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let start = Instant::now();
                let result = fill_sinks(&dem, FillSinksParams { min_slope })
                    .context("Failed to fill sinks")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("COG fill-sinks", &output, elapsed);
            }
        },

        // ── STAC catalog search & fetch ─────────────────────────────────
        #[cfg(feature = "cloud")]
        Commands::Stac { action } => match action {
            StacCommands::Search {
                catalog,
                bbox,
                datetime,
                collections,
                limit,
            } => {
                let cat = StacCatalog::from_str_or_url(&catalog);
                let pb = spinner("Searching STAC catalog...");
                let client =
                    StacClientBlocking::new(cat, StacClientOptions::default())
                        .context("Failed to create STAC client")?;

                let mut params = StacSearchParams::new().limit(limit);
                if let Some(ref b) = bbox {
                    let bb = parse_bbox(b)?;
                    params = params.bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y);
                }
                if let Some(ref dt) = datetime {
                    params = params.datetime(dt);
                }
                if let Some(ref cols) = collections {
                    let c: Vec<&str> = cols.split(',').map(|s| s.trim()).collect();
                    params = params.collections(&c);
                }

                let results =
                    client.search(&params).context("STAC search failed")?;
                pb.finish_and_clear();

                println!(
                    "Found {} items (matched: {})",
                    results.len(),
                    results
                        .number_matched
                        .map_or("?".to_string(), |n| n.to_string())
                );
                println!();

                for item in &results.features {
                    let dt = item
                        .properties
                        .datetime
                        .as_deref()
                        .unwrap_or("-");
                    let cc = item
                        .properties
                        .eo_cloud_cover
                        .map(|c| format!("{:.1}%", c))
                        .unwrap_or_else(|| "-".to_string());
                    let col = item.collection.as_deref().unwrap_or("-");
                    let asset_keys: Vec<&str> =
                        item.assets.keys().map(|k| k.as_str()).collect();

                    println!("  {} [{}]", item.id, col);
                    println!("    datetime: {}  cloud: {}", dt, cc);
                    println!("    assets: {}", asset_keys.join(", "));
                }

                if results.has_next() {
                    println!(
                        "\n  (more results available — increase --limit to fetch more)"
                    );
                }
            }

            StacCommands::Fetch {
                catalog,
                bbox,
                collection,
                asset,
                datetime,
                output,
            } => {
                let cat = StacCatalog::from_str_or_url(&catalog);
                let bb = parse_bbox(&bbox)?;

                let mut params = StacSearchParams::new()
                    .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                    .collections(&[collection.as_str()])
                    .limit(1);
                if let Some(ref dt) = datetime {
                    params = params.datetime(dt);
                }

                let pb = spinner("Searching STAC catalog...");
                let client =
                    StacClientBlocking::new(cat, StacClientOptions::default())
                        .context("Failed to create STAC client")?;
                let results =
                    client.search(&params).context("STAC search failed")?;

                let item = results.features.first().ok_or_else(|| {
                    anyhow::anyhow!("No items found matching the search criteria")
                })?;
                pb.finish_and_clear();

                println!(
                    "Item: {} [{}]",
                    item.id,
                    item.collection.as_deref().unwrap_or("-")
                );

                // Determine asset key
                let asset_key = if let Some(ref k) = asset {
                    k.clone()
                } else {
                    let (k, _) = item.first_cog_asset().ok_or_else(|| {
                        anyhow::anyhow!(
                            "No COG asset found. Specify --asset explicitly. Available: {}",
                            item.assets
                                .keys()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?;
                    println!("Auto-detected asset: {}", k);
                    k.clone()
                };

                let stac_asset = item.asset(&asset_key).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Asset '{}' not found. Available: {}",
                        asset_key,
                        item.assets
                            .keys()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;

                // Sign the href if needed
                let href = client
                    .sign_asset_href(
                        &stac_asset.href,
                        item.collection.as_deref().unwrap_or(""),
                    )
                    .context("Failed to sign asset URL")?;

                // Read via CogReader
                let pb = spinner("Fetching COG tiles...");
                let start = Instant::now();
                let opts = CogReaderOptions::default();
                let mut reader = CogReaderBlocking::open(&href, opts)
                    .context("Failed to open remote COG")?;

                // Auto-reproject bbox if COG is in a projected CRS (e.g. UTM)
                let read_bb = {
                    use surtgis_cloud::reproject;
                    // Prefer proj:epsg from STAC item (no extra HTTP request)
                    let epsg = item.epsg().or_else(|| {
                        reader
                            .metadata()
                            .crs
                            .as_ref()
                            .and_then(|c| c.epsg())
                    });
                    if let Some(epsg) = epsg {
                        if !reproject::is_wgs84(epsg) {
                            let reprojected =
                                reproject::reproject_bbox_to_cog(&bb, epsg);
                            println!("Reprojected bbox to EPSG:{}", epsg);
                            reprojected
                        } else {
                            bb
                        }
                    } else {
                        bb
                    }
                };

                let raster: surtgis_core::Raster<f64> = reader
                    .read_bbox(&read_bb, None)
                    .context("Failed to read bounding box from COG")?;
                pb.finish_and_clear();
                let elapsed = start.elapsed();

                let (rows, cols) = raster.shape();
                println!(
                    "Fetched: {} x {} ({} cells)",
                    cols,
                    rows,
                    raster.len()
                );
                write_result(&raster, &output, compress)?;
                done("STAC fetch", &output, elapsed);
            }

            StacCommands::FetchMosaic {
                catalog,
                bbox,
                collection,
                asset,
                datetime,
                max_items,
                output,
            } => {
                let cat = StacCatalog::from_str_or_url(&catalog);
                let bb = parse_bbox(&bbox)?;

                let mut params = StacSearchParams::new()
                    .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                    .collections(&[collection.as_str()])
                    .limit(max_items);
                if let Some(ref dt) = datetime {
                    params = params.datetime(dt);
                }

                let pb = spinner("Searching STAC catalog...");
                let client_opts = StacClientOptions {
                    max_items: max_items as usize,
                    ..StacClientOptions::default()
                };
                let client = StacClientBlocking::new(cat, client_opts)
                    .context("Failed to create STAC client")?;
                let items = client.search_all(&params).context("STAC search failed")?;
                pb.finish_and_clear();

                if items.is_empty() {
                    anyhow::bail!("No items found matching the search criteria");
                }
                println!("Found {} items, fetching and mosaicking...", items.len());

                // Determine asset key from first item
                let asset_key = if let Some(ref k) = asset {
                    k.clone()
                } else {
                    let (k, _) = items[0].first_cog_asset().ok_or_else(|| {
                        anyhow::anyhow!(
                            "No COG asset found. Specify --asset explicitly. Available: {}",
                            items[0]
                                .assets
                                .keys()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    })?;
                    println!("Auto-detected asset: {}", k);
                    k.clone()
                };

                let start = Instant::now();
                let mut rasters: Vec<surtgis_core::Raster<f64>> = Vec::new();

                for (i, item) in items.iter().enumerate() {
                    let pb = spinner(&format!(
                        "Fetching tile {} of {} [{}]...",
                        i + 1,
                        items.len(),
                        item.id
                    ));

                    let stac_asset = match item.asset(&asset_key) {
                        Some(a) => a,
                        None => {
                            pb.finish_and_clear();
                            eprintln!(
                                "  Warning: item {} missing asset '{}', skipping",
                                item.id, asset_key
                            );
                            continue;
                        }
                    };

                    let href = client
                        .sign_asset_href(
                            &stac_asset.href,
                            item.collection.as_deref().unwrap_or(""),
                        )
                        .context("Failed to sign asset URL")?;

                    let opts = CogReaderOptions::default();
                    let mut reader = match CogReaderBlocking::open(&href, opts) {
                        Ok(r) => r,
                        Err(e) => {
                            pb.finish_and_clear();
                            eprintln!(
                                "  Warning: failed to open COG for {}: {}, skipping",
                                item.id, e
                            );
                            continue;
                        }
                    };

                    // Auto-reproject bbox if COG is in a projected CRS
                    let read_bb = {
                        use surtgis_cloud::reproject;
                        let epsg = item.epsg().or_else(|| {
                            reader
                                .metadata()
                                .crs
                                .as_ref()
                                .and_then(|c| c.epsg())
                        });
                        if let Some(epsg) = epsg {
                            if !reproject::is_wgs84(epsg) {
                                reproject::reproject_bbox_to_cog(&bb, epsg)
                            } else {
                                bb
                            }
                        } else {
                            bb
                        }
                    };

                    match reader.read_bbox::<f64>(&read_bb, None) {
                        Ok(raster) => {
                            pb.finish_and_clear();
                            let (rows, cols) = raster.shape();
                            println!(
                                "  [{}/{}] {} — {} x {}",
                                i + 1,
                                items.len(),
                                item.id,
                                cols,
                                rows
                            );
                            rasters.push(raster);
                        }
                        Err(e) => {
                            pb.finish_and_clear();
                            eprintln!(
                                "  Warning: failed to read tile {}: {}, skipping",
                                item.id, e
                            );
                        }
                    }
                }

                if rasters.is_empty() {
                    anyhow::bail!("No tiles were successfully fetched");
                }

                let pb = spinner("Mosaicking tiles...");
                let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
                let result = surtgis_core::mosaic(&refs, None)
                    .context("Failed to mosaic tiles")?;
                pb.finish_and_clear();

                let elapsed = start.elapsed();
                let (rows, cols) = result.shape();
                println!(
                    "Mosaic: {} tiles → {} x {} ({} cells)",
                    rasters.len(),
                    cols,
                    rows,
                    result.len()
                );
                write_result(&result, &output, compress)?;
                done("STAC fetch-mosaic", &output, elapsed);
            }

            StacCommands::Composite {
                catalog,
                bbox,
                collection,
                asset,
                datetime,
                max_scenes,
                scl_asset,
                scl_keep,
                output,
            } => {
                let cat = StacCatalog::from_str_or_url(&catalog);
                let bb = parse_bbox(&bbox)?;
                let keep_classes = parse_scl_classes(&scl_keep)?;

                // Search with high limit to find enough items across dates
                let search_limit = (max_scenes * 4) as u32; // ~4 tiles per date
                let params = StacSearchParams::new()
                    .bbox(bb.min_x, bb.min_y, bb.max_x, bb.max_y)
                    .collections(&[collection.as_str()])
                    .datetime(&datetime)
                    .limit(search_limit);

                let pb = spinner("Searching STAC catalog...");
                let client_opts = StacClientOptions {
                    max_items: (max_scenes * 4) as usize,
                    ..StacClientOptions::default()
                };
                let client = StacClientBlocking::new(cat, client_opts)
                    .context("Failed to create STAC client")?;
                let items = client.search_all(&params).context("STAC search failed")?;
                pb.finish_and_clear();

                if items.is_empty() {
                    anyhow::bail!("No items found matching the search criteria");
                }

                // Group items by acquisition date (YYYY-MM-DD)
                let mut by_date: BTreeMap<String, Vec<&StacItem>> = BTreeMap::new();
                for item in &items {
                    let date = item
                        .properties
                        .datetime
                        .as_deref()
                        .unwrap_or("")
                        .get(..10)
                        .unwrap_or("unknown")
                        .to_string();
                    by_date.entry(date).or_default().push(item);
                }

                // Take up to max_scenes dates
                let dates: Vec<String> =
                    by_date.keys().take(max_scenes).cloned().collect();

                println!(
                    "Found {} items across {} dates (using {} dates)",
                    items.len(),
                    by_date.len(),
                    dates.len()
                );

                let start = Instant::now();
                let mut clean_scenes: Vec<surtgis_core::Raster<f64>> = Vec::new();

                for (di, date) in dates.iter().enumerate() {
                    let group = &by_date[date];
                    println!(
                        "Processing date {}/{}: {} ({} tiles)...",
                        di + 1,
                        dates.len(),
                        date,
                        group.len()
                    );

                    // Fetch data + SCL tiles for this date
                    let mut data_tiles: Vec<surtgis_core::Raster<f64>> = Vec::new();
                    let mut scl_tiles: Vec<surtgis_core::Raster<f64>> = Vec::new();

                    for item in group {
                        // Fetch data asset
                        match fetch_stac_asset(item, &asset, &bb, &client) {
                            Ok(r) => data_tiles.push(r),
                            Err(e) => {
                                eprintln!(
                                    "  Warning: {} asset '{}': {}, skipping",
                                    item.id, asset, e
                                );
                                continue;
                            }
                        }

                        // Fetch SCL asset
                        match fetch_stac_asset(item, &scl_asset, &bb, &client) {
                            Ok(r) => scl_tiles.push(r),
                            Err(e) => {
                                // Remove the data tile we just added since SCL failed
                                data_tiles.pop();
                                eprintln!(
                                    "  Warning: {} asset '{}': {}, skipping",
                                    item.id, scl_asset, e
                                );
                            }
                        }
                    }

                    if data_tiles.is_empty() {
                        eprintln!("  No tiles for date {}, skipping", date);
                        continue;
                    }

                    // Mosaic spatial tiles for this date
                    let data_mosaic = if data_tiles.len() == 1 {
                        data_tiles.into_iter().next().unwrap()
                    } else {
                        let refs: Vec<&surtgis_core::Raster<f64>> =
                            data_tiles.iter().collect();
                        surtgis_core::mosaic(&refs, None)
                            .context("Failed to mosaic data tiles")?
                    };

                    let scl_mosaic = if scl_tiles.len() == 1 {
                        scl_tiles.into_iter().next().unwrap()
                    } else {
                        let refs: Vec<&surtgis_core::Raster<f64>> =
                            scl_tiles.iter().collect();
                        surtgis_core::mosaic(&refs, None)
                            .context("Failed to mosaic SCL tiles")?
                    };

                    // Cloud mask
                    let clean = cloud_mask_scl(&data_mosaic, &scl_mosaic, &keep_classes)
                        .context("Failed to apply cloud mask")?;

                    // Count clear pixels
                    let total = clean.len();
                    let clear = clean
                        .data()
                        .iter()
                        .filter(|v| v.is_finite())
                        .count();
                    let (rows, cols) = clean.shape();
                    println!(
                        "  {} x {}, {:.0}% clear",
                        cols,
                        rows,
                        100.0 * clear as f64 / total as f64
                    );

                    clean_scenes.push(clean);
                }

                if clean_scenes.len() < 2 {
                    if clean_scenes.len() == 1 {
                        println!("Only 1 scene available, writing directly (no composite)");
                        write_result(&clean_scenes[0], &output, compress)?;
                    } else {
                        anyhow::bail!("No clean scenes available for compositing");
                    }
                } else {
                    // Median composite
                    let pb = spinner(&format!(
                        "Computing median composite of {} scenes...",
                        clean_scenes.len()
                    ));
                    let refs: Vec<&surtgis_core::Raster<f64>> =
                        clean_scenes.iter().collect();
                    let result = median_composite(&refs)
                        .context("Failed to compute median composite")?;
                    pb.finish_and_clear();

                    let (rows, cols) = result.shape();
                    println!(
                        "Composite: {} scenes → {} x {} ({} cells)",
                        clean_scenes.len(),
                        cols,
                        rows,
                        result.len()
                    );
                    write_result(&result, &output, compress)?;
                }

                let elapsed = start.elapsed();
                done("STAC composite", &output, elapsed);
            }
        },
    }

    Ok(())
}
