//! SurtGis CLI - High-performance geospatial analysis

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

use surtgis_algorithms::terrain::{
    aspect, hillshade, slope, AspectOutput, HillshadeParams, SlopeParams, SlopeUnits,
};
use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};

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
    /// Terrain analysis algorithms
    Terrain {
        #[command(subcommand)]
        algorithm: TerrainCommands,
    },
    /// Show information about a raster file
    Info {
        /// Input raster file
        input: PathBuf,
    },
}

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
}

fn setup_logging(verbose: bool) {
    let level = if verbose { Level::DEBUG } else { Level::INFO };
    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");
}

fn create_progress_bar(msg: &str) -> ProgressBar {
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    setup_logging(cli.verbose);

    match cli.command {
        Commands::Info { input } => {
            let pb = create_progress_bar("Reading raster...");
            let raster: surtgis_core::Raster<f64> = read_geotiff(&input, None)
                .context("Failed to read raster")?;
            pb.finish_and_clear();

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
            println!("  Valid cells: {} ({:.1}%)",
                stats.valid_count,
                100.0 * stats.valid_count as f64 / raster.len() as f64
            );
        }

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

                let pb = create_progress_bar("Reading DEM...");
                let dem: surtgis_core::Raster<f64> = read_geotiff(&input, None)
                    .context("Failed to read DEM")?;
                pb.finish_and_clear();

                info!("Input: {} x {}", dem.cols(), dem.rows());

                let pb = create_progress_bar("Calculating slope...");
                let start = Instant::now();
                let result = slope(&dem, SlopeParams { units, z_factor })
                    .context("Failed to calculate slope")?;
                let elapsed = start.elapsed();
                pb.finish_and_clear();

                info!("Slope calculation: {:.2?}", elapsed);

                let pb = create_progress_bar("Writing output...");
                write_geotiff(&result, &output, Some(GeoTiffOptions::default()))
                    .context("Failed to write output")?;
                pb.finish_and_clear();

                println!("✓ Slope saved to: {}", output.display());
                println!("  Processing time: {:.2?}", elapsed);
            }

            TerrainCommands::Aspect {
                input,
                output,
                format,
            } => {
                let format = match format.to_lowercase().as_str() {
                    "degrees" | "deg" | "d" => AspectOutput::Degrees,
                    "radians" | "rad" | "r" => AspectOutput::Radians,
                    "compass" | "c" => AspectOutput::Compass,
                    _ => {
                        eprintln!("Unknown format: {}. Using degrees.", format);
                        AspectOutput::Degrees
                    }
                };

                let pb = create_progress_bar("Reading DEM...");
                let dem: surtgis_core::Raster<f64> = read_geotiff(&input, None)
                    .context("Failed to read DEM")?;
                pb.finish_and_clear();

                let pb = create_progress_bar("Calculating aspect...");
                let start = Instant::now();
                let result = aspect(&dem, format).context("Failed to calculate aspect")?;
                let elapsed = start.elapsed();
                pb.finish_and_clear();

                let pb = create_progress_bar("Writing output...");
                write_geotiff(&result, &output, Some(GeoTiffOptions::default()))
                    .context("Failed to write output")?;
                pb.finish_and_clear();

                println!("✓ Aspect saved to: {}", output.display());
                println!("  Processing time: {:.2?}", elapsed);
            }

            TerrainCommands::Hillshade {
                input,
                output,
                azimuth,
                altitude,
                z_factor,
            } => {
                let pb = create_progress_bar("Reading DEM...");
                let dem: surtgis_core::Raster<f64> = read_geotiff(&input, None)
                    .context("Failed to read DEM")?;
                pb.finish_and_clear();

                let pb = create_progress_bar("Calculating hillshade...");
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
                pb.finish_and_clear();

                let pb = create_progress_bar("Writing output...");
                write_geotiff(&result, &output, Some(GeoTiffOptions::default()))
                    .context("Failed to write output")?;
                pb.finish_and_clear();

                println!("✓ Hillshade saved to: {}", output.display());
                println!("  Processing time: {:.2?}", elapsed);
            }
        },
    }

    Ok(())
}
