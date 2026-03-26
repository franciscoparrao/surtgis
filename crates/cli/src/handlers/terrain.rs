//! Handler for terrain analysis subcommands.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::terrain::{
    advanced_curvatures, aspect, convergence_index, curvature, dev, eastness, geomorphons,
    hillshade, landform_classification, ls_factor, mrvbf, multidirectional_hillshade,
    negative_openness, northness, positive_openness, relative_slope_position, sky_view_factor,
    slope, surface_area_ratio, tpi, tri, valley_depth, viewshed, vrm,
    AspectOutput, AspectStreaming, ConvergenceParams, ConvergenceStreaming,
    CurvatureParams, CurvatureStreaming, CurvatureType, DevParams, DevStreaming,
    EastnessStreaming, GeomorphonParams, HillshadeParams, HillshadeStreaming, LandformParams,
    LsFactorParams, MultiHillshadeParams, MultiHillshadeStreaming, MrvbfParams,
    NorthnessStreaming, OpennessParams, SarParams, SlopeParams, SlopeStreaming, SlopeUnits,
    SvfParams, TpiParams, TpiStreaming, TriParams, TriStreaming, ViewshedParams, VrmParams,
    VrmStreaming,
};
use surtgis_core::StripProcessor;

use crate::commands::TerrainCommands;
use crate::helpers::{
    done, parse_advanced_curvature_type, read_dem, write_result, write_result_u8,
};
use crate::memory;

pub fn handle(algorithm: TerrainCommands, compress: bool, streaming: bool, mem_limit_bytes: Option<u64>) -> Result<()> {
    match algorithm {
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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = streaming || std::fs::metadata(&input)
                .map(|m| m.len() > 500_000_000).unwrap_or(false);

            if use_streaming {
                let algo = DevStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute DEV (streaming)")?;
                let elapsed = start.elapsed();
                println!("DEV (streaming): {} x {}", cols, rows);
                done("DEV", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    dev(&dem, DevParams { radius }).context("Failed to compute DEV")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("DEV", &output, elapsed);
            }
        }

        TerrainCommands::Vrm {
            input,
            output,
            radius,
        } => {
            let use_streaming = streaming || std::fs::metadata(&input)
                .map(|m| m.len() > 500_000_000).unwrap_or(false);

            if use_streaming {
                let algo = VrmStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute VRM (streaming)")?;
                let elapsed = start.elapsed();
                println!("VRM (streaming): {} x {}", cols, rows);
                done("VRM", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    vrm(&dem, VrmParams { radius }).context("Failed to compute VRM")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("VRM", &output, elapsed);
            }
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
            let use_streaming = streaming || std::fs::metadata(&input)
                .map(|m| m.len() > 500_000_000).unwrap_or(false);

            if use_streaming {
                let algo = ConvergenceStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute convergence index (streaming)")?;
                let elapsed = start.elapsed();
                println!("Convergence (streaming): {} x {}", cols, rows);
                done("Convergence index", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = convergence_index(&dem, ConvergenceParams { radius })
                    .context("Failed to compute convergence index")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Convergence index", &output, elapsed);
            }
        }

        TerrainCommands::MultiHillshade { input, output } => {
            let use_streaming = streaming || std::fs::metadata(&input)
                .map(|m| m.len() > 500_000_000).unwrap_or(false);

            if use_streaming {
                let algo = MultiHillshadeStreaming::default();
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute multi-directional hillshade (streaming)")?;
                let elapsed = start.elapsed();
                println!("Multi-directional hillshade (streaming): {} x {}", cols, rows);
                done("Multi-directional hillshade", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result =
                    multidirectional_hillshade(&dem, MultiHillshadeParams::default())
                        .context("Failed to compute multi-directional hillshade")?;
                let elapsed = start.elapsed();
                write_result(&result, &output, compress)?;
                done("Multi-directional hillshade", &output, elapsed);
            }
        }

        TerrainCommands::LsFactor { flow_acc, slope: slope_path, output, cell_size } => {
            let facc = read_dem(&flow_acc)?;
            let slp = read_dem(&slope_path)?;
            let start = Instant::now();
            let result = ls_factor(&facc, &slp, LsFactorParams { cell_size, ..Default::default() })
                .context("Failed to compute LS-Factor")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("LS-Factor", &output, elapsed);
        }

        TerrainCommands::ValleyDepth { input, output } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = valley_depth(&dem).context("Failed to compute valley depth")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Valley depth", &output, elapsed);
        }

        TerrainCommands::RelativeSlopePosition { hand, valley_depth: vd_path, output } => {
            let hand_r = read_dem(&hand)?;
            let vd_r = read_dem(&vd_path)?;
            let start = Instant::now();
            let result = relative_slope_position(&hand_r, &vd_r)
                .context("Failed to compute relative slope position")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Relative slope position", &output, elapsed);
        }

        TerrainCommands::SurfaceAreaRatio { input, output, radius } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = surface_area_ratio(&dem, SarParams { radius })
                .context("Failed to compute surface area ratio")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Surface area ratio", &output, elapsed);
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
    }

    Ok(())
}

/// Compute all terrain factors from DEM (public for pipeline use)
pub fn handle_terrain_all(input: &std::path::Path, outdir: &std::path::Path, compress: bool) -> Result<()> {
    std::fs::create_dir_all(outdir)
        .context("Failed to create output directory")?;
    let dem = read_dem(&input.to_path_buf())?;
    let start = std::time::Instant::now();

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

    Ok(())
}
