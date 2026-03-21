//! Handler for hydrology subcommands.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::hydrology::{
    basin_morphometry, breach_depressions, drainage_density, fill_sinks, flow_accumulation,
    flow_accumulation_mfd, flow_direction, flow_direction_dinf, hand, hypsometric_integral,
    priority_flood, sediment_connectivity, stream_network, watershed,
    BreachParams, DrainageDensityParams,
    FillSinksParams, HandParams, MfdParams, PriorityFloodParams, SedimentConnectivityParams,
    StreamNetworkParams, WatershedParams,
};
use surtgis_algorithms::terrain::{slope, twi, SlopeParams, SlopeUnits};

use crate::commands::HydrologyCommands;
use crate::helpers::{
    done, parse_pour_points, read_dem, read_u8, write_result, write_result_i32, write_result_u8,
};

pub fn handle(algorithm: HydrologyCommands, compress: bool) -> Result<()> {
    match algorithm {
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

        HydrologyCommands::DrainageDensity { input, output, radius, cell_size } => {
            let streams = read_dem(&input)?;
            let start = Instant::now();
            let result = drainage_density(&streams, DrainageDensityParams { radius, cell_size })
                .context("Failed to compute drainage density")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Drainage density", &output, elapsed);
        }

        HydrologyCommands::HypsometricIntegral { dem, watersheds } => {
            let dem_r = read_dem(&dem)?;
            let ws_r: surtgis_core::Raster<i32> = surtgis_core::io::read_geotiff(&watersheds, None)
                .context("Failed to read watershed raster")?;
            let start = Instant::now();
            let hi = hypsometric_integral(&dem_r, &ws_r)
                .context("Failed to compute hypsometric integral")?;
            let elapsed = start.elapsed();
            println!("{:<12} {:>10}", "Watershed", "HI");
            println!("{}", "-".repeat(24));
            let mut sorted: Vec<_> = hi.iter().collect();
            sorted.sort_by_key(|&(&k, _)| k);
            for &(&id, &val) in &sorted {
                println!("{:<12} {:>10.4}", id, val);
            }
            println!("\n  Processing time: {:.2?}", elapsed);
        }

        HydrologyCommands::SedimentConnectivity { slope, flow_acc, flow_dir, output, threshold } => {
            let slp = read_dem(&slope)?;
            let facc = read_dem(&flow_acc)?;
            let fdir: surtgis_core::Raster<u8> = surtgis_core::io::read_geotiff(&flow_dir, None)
                .context("Failed to read flow direction")?;
            let start = Instant::now();
            let result = sediment_connectivity(
                &slp, &facc, &fdir,
                SedimentConnectivityParams { stream_threshold: threshold },
                None,
            ).context("Failed to compute sediment connectivity")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Sediment connectivity", &output, elapsed);
        }

        HydrologyCommands::BasinMorphometry { input, cell_size } => {
            let ws_r: surtgis_core::Raster<i32> = surtgis_core::io::read_geotiff(&input, None)
                .context("Failed to read watershed raster")?;
            let start = Instant::now();
            let metrics = basin_morphometry(&ws_r, cell_size)
                .context("Failed to compute basin morphometry")?;
            let elapsed = start.elapsed();
            println!("{:<8} {:>10} {:>10} {:>10} {:>10} {:>10}",
                "Basin", "Area(m2)", "Perim(m)", "Circular", "Elongat", "Compact");
            println!("{}", "-".repeat(68));
            for m in &metrics {
                println!("{:<8} {:>10.1} {:>10.1} {:>10.4} {:>10.4} {:>10.4}",
                    m.watershed_id, m.area_m2, m.perimeter_m,
                    m.circularity, m.elongation, m.compactness);
            }
            println!("\n  Processing time: {:.2?}", elapsed);
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
    }

    Ok(())
}
