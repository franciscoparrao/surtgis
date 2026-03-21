//! Handler for COG (Cloud Optimized GeoTIFF) subcommands.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::hydrology::{fill_sinks, FillSinksParams};
use surtgis_algorithms::terrain::{
    aspect, hillshade, slope, tpi, AspectOutput, HillshadeParams, SlopeParams, SlopeUnits,
    TpiParams,
};
use surtgis_cloud::blocking::CogReaderBlocking;
use surtgis_cloud::CogReaderOptions;

use crate::commands::CogCommands;
use crate::helpers::{done, parse_bbox, spinner, write_result};
use crate::streaming::read_cog_dem;

pub fn handle(action: CogCommands, compress: bool) -> Result<()> {
    match action {
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
    }

    Ok(())
}
