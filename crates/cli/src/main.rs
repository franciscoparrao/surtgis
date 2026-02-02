//! SurtGis CLI - High-performance geospatial analysis

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use surtgis_algorithms::hydrology::{
    fill_sinks, flow_accumulation, flow_direction, watershed, FillSinksParams, WatershedParams,
};
use surtgis_algorithms::imagery::{
    band_math_binary, bsi, evi, mndwi, nbr, ndvi, ndwi, savi, BandMathOp, EviParams, SaviParams,
};
use surtgis_algorithms::morphology::{
    black_hat, closing, dilate, erode, gradient, opening, top_hat, StructuringElement,
};
use surtgis_algorithms::terrain::{
    aspect, curvature, hillshade, landform_classification, slope, tpi, tri, AspectOutput,
    CurvatureParams, CurvatureType, HillshadeParams, LandformParams, SlopeParams, SlopeUnits,
    TpiParams, TriParams,
};
use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};

#[cfg(feature = "cloud")]
use surtgis_cloud::blocking::{CogReaderBlocking, StacClientBlocking, read_cog};
#[cfg(feature = "cloud")]
use surtgis_cloud::{BBox, CogReaderOptions, StacCatalog, StacClientOptions, StacSearchParams};

// ─── CLI structure ──────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "surtgis")]
#[command(author, version, about = "High-performance geospatial analysis", long_about = None)]
struct Cli {
    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

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

fn write_result(raster: &surtgis_core::Raster<f64>, path: &PathBuf) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(GeoTiffOptions::default()))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

fn write_result_u8(raster: &surtgis_core::Raster<u8>, path: &PathBuf) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(GeoTiffOptions::default()))
        .context("Failed to write output")?;
    pb.finish_and_clear();
    Ok(())
}

fn write_result_i32(raster: &surtgis_core::Raster<i32>, path: &PathBuf) -> Result<()> {
    let pb = spinner("Writing output...");
    write_geotiff(raster, path, Some(GeoTiffOptions::default()))
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

// ─── COG helpers ────────────────────────────────────────────────────────

#[cfg(feature = "cloud")]
fn parse_bbox(s: &str) -> Result<BBox> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 4 {
        anyhow::bail!("Bbox must be min_x,min_y,max_x,max_y (got {} parts)", parts.len());
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
    info!("Remote raster: {} x {} ({} cells)", cols, rows, raster.len());
    Ok(raster)
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    match cli.command {
        // ── Info ─────────────────────────────────────────────────────
        Commands::Info { input } => {
            let raster = read_dem(&input)?;
            let (rows, cols) = raster.shape();
            let bounds = raster.bounds();
            let stats = raster.statistics();

            println!("File: {}", input.display());
            println!("Dimensions: {} x {} ({} cells)", cols, rows, raster.len());
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
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = slope(&dem, SlopeParams { units, z_factor })
                    .context("Failed to calculate slope")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
                done("Slope", &output, elapsed);
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
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = aspect(&dem, fmt).context("Failed to calculate aspect")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
                done("Aspect", &output, elapsed);
            }

            TerrainCommands::Hillshade {
                input,
                output,
                azimuth,
                altitude,
                z_factor,
            } => {
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
                write_result(&result, &output)?;
                done("Hillshade", &output, elapsed);
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
                write_result(&result, &output)?;
                done("Curvature", &output, elapsed);
            }

            TerrainCommands::Tpi {
                input,
                output,
                radius,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    tpi(&dem, TpiParams { radius }).context("Failed to calculate TPI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
                done("TPI", &output, elapsed);
            }

            TerrainCommands::Tri {
                input,
                output,
                radius,
            } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    tri(&dem, TriParams { radius }).context("Failed to calculate TRI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
                done("TRI", &output, elapsed);
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
                write_result(&result, &output)?;
                done("Landform classification", &output, elapsed);
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
                write_result(&result, &output)?;
                done("Fill sinks", &output, elapsed);
            }

            HydrologyCommands::FlowDirection { input, output } => {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    flow_direction(&dem).context("Failed to calculate flow direction")?;
                let elapsed = start.elapsed();
                write_result_u8(&result, &output)?;
                done("Flow direction", &output, elapsed);
            }

            HydrologyCommands::FlowAccumulation { input, output } => {
                let flow_dir = read_u8(&input)?;
                let start = Instant::now();
                let result = flow_accumulation(&flow_dir)
                    .context("Failed to calculate flow accumulation")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
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
                write_result_i32(&result, &output)?;
                done("Watershed", &output, elapsed);
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
                write_result(&result, &output)?;
                done("NDVI", &output, elapsed);
            }

            ImageryCommands::Ndwi { green, nir, output } => {
                let green_r = read_dem(&green)?;
                let nir_r = read_dem(&nir)?;
                let start = Instant::now();
                let result = ndwi(&green_r, &nir_r).context("Failed to calculate NDWI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
                done("MNDWI", &output, elapsed);
            }

            ImageryCommands::Nbr { nir, swir, output } => {
                let nir_r = read_dem(&nir)?;
                let swir_r = read_dem(&swir)?;
                let start = Instant::now();
                let result = nbr(&nir_r, &swir_r).context("Failed to calculate NBR")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
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
                let result =
                    savi(&nir_r, &red_r, SaviParams { l_factor })
                        .context("Failed to calculate SAVI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
                done("Band math", &output, elapsed);
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
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
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
                let result = gradient(&raster, &se).context("Failed to compute gradient")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
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
                let result = top_hat(&raster, &se).context("Failed to compute top-hat")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
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
                write_result(&result, &output)?;
                done("Black-hat", &output, elapsed);
            }
        },

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
                println!("Dimensions: {} x {} ({} cells)", meta.width, meta.height,
                    meta.width as u64 * meta.height as u64);
                println!("Tile size: {} x {}", meta.tile_width, meta.tile_height);
                println!("Bits/sample: {}, Sample format: {}", meta.bits_per_sample, meta.sample_format);
                println!("Compression: {}", match meta.compression {
                    1 => "None",
                    5 => "LZW",
                    8 | 32946 => "DEFLATE",
                    _ => "Other",
                });
                let gt = &meta.geo_transform;
                println!("Origin: ({:.6}, {:.6})", gt.origin_x, gt.origin_y);
                println!("Pixel size: ({:.6}, {:.6})", gt.pixel_width, gt.pixel_height);
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

            CogCommands::Fetch { url, output, bbox, overview } => {
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
                write_result(&raster, &output)?;
                done("COG fetch", &output, elapsed);
            }

            CogCommands::Slope { url, output, bbox, units, z_factor } => {
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
                write_result(&result, &output)?;
                done("COG slope", &output, elapsed);
            }

            CogCommands::Aspect { url, output, bbox, format } => {
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
                let result = aspect(&dem, fmt).context("Failed to calculate aspect")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
                done("COG aspect", &output, elapsed);
            }

            CogCommands::Hillshade { url, output, bbox, azimuth, altitude, z_factor } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let start = Instant::now();
                let result = hillshade(
                    &dem,
                    HillshadeParams { azimuth, altitude, z_factor, normalized: false },
                ).context("Failed to calculate hillshade")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
                done("COG hillshade", &output, elapsed);
            }

            CogCommands::Tpi { url, output, bbox, radius } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let start = Instant::now();
                let result = tpi(&dem, TpiParams { radius })
                    .context("Failed to calculate TPI")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
                done("COG TPI", &output, elapsed);
            }

            CogCommands::FillSinks { url, output, bbox, min_slope } => {
                let bbox = parse_bbox(&bbox)?;
                let dem = read_cog_dem(&url, &bbox)?;
                let start = Instant::now();
                let result = fill_sinks(&dem, FillSinksParams { min_slope })
                    .context("Failed to fill sinks")?;
                let elapsed = start.elapsed();
                write_result(&result, &output)?;
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
                let client = StacClientBlocking::new(cat, StacClientOptions::default())
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

                let results = client.search(&params).context("STAC search failed")?;
                pb.finish_and_clear();

                println!(
                    "Found {} items (matched: {})",
                    results.len(),
                    results.number_matched.map_or("?".to_string(), |n| n.to_string())
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
                    let asset_keys: Vec<&str> = item.assets.keys().map(|k| k.as_str()).collect();

                    println!("  {} [{}]", item.id, col);
                    println!("    datetime: {}  cloud: {}", dt, cc);
                    println!("    assets: {}", asset_keys.join(", "));
                }

                if results.has_next() {
                    println!("\n  (more results available — increase --limit to fetch more)");
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
                let client = StacClientBlocking::new(cat, StacClientOptions::default())
                    .context("Failed to create STAC client")?;
                let results = client.search(&params).context("STAC search failed")?;

                let item = results.features.first().ok_or_else(|| {
                    anyhow::anyhow!("No items found matching the search criteria")
                })?;
                pb.finish_and_clear();

                println!("Item: {} [{}]", item.id, item.collection.as_deref().unwrap_or("-"));

                // Determine asset key
                let asset_key = if let Some(ref k) = asset {
                    k.clone()
                } else {
                    let (k, _) = item.first_cog_asset().ok_or_else(|| {
                        anyhow::anyhow!("No COG asset found. Specify --asset explicitly. Available: {}",
                            item.assets.keys().cloned().collect::<Vec<_>>().join(", "))
                    })?;
                    println!("Auto-detected asset: {}", k);
                    k.clone()
                };

                let stac_asset = item.asset(&asset_key).ok_or_else(|| {
                    anyhow::anyhow!("Asset '{}' not found. Available: {}",
                        asset_key,
                        item.assets.keys().cloned().collect::<Vec<_>>().join(", "))
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
                        reader.metadata().crs.as_ref().and_then(|c| c.epsg())
                    });
                    if let Some(epsg) = epsg {
                        if !reproject::is_wgs84(epsg) {
                            let reprojected = reproject::reproject_bbox_to_cog(&bb, epsg);
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
                println!("Fetched: {} x {} ({} cells)", cols, rows, raster.len());
                write_result(&raster, &output)?;
                done("STAC fetch", &output, elapsed);
            }
        },
    }

    Ok(())
}
