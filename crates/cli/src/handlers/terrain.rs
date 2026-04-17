//! Handler for terrain analysis subcommands.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::terrain::{
    advanced_curvatures, aspect, circular_variance_aspect, convergence_index, curvature, dev,
    diff_from_mean_elev, directional_relief, downslope_index, eastness, edge_density,
    elev_above_pit, elev_relative_to_min_max, geomorphons, hillshade, hypsometric_hillshade,
    landform_classification, ls_factor, max_branch_length, mrvbf, multidirectional_hillshade,
    negative_openness, neighbour_stats, normal_vector_deviation, northness, pennock,
    percent_elev_range, positive_openness, relative_aspect, relative_slope_position,
    sky_view_factor, slope, spherical_std_dev, surface_area_ratio, tpi, tri, valley_depth,
    viewshed, vrm,
    AspectOutput, AspectStreaming, CircularVarianceParams, CircularVarianceStreaming,
    ConvergenceParams, ConvergenceStreaming, CurvatureParams, CurvatureStreaming, CurvatureType,
    DevParams, DevStreaming, DiffFromMeanParams, DiffFromMeanStreaming, DirectionalReliefParams,
    DownslopeIndexParams, EdgeDensityParams, EastnessStreaming, GeomorphonParams, HillshadeParams,
    HillshadeStreaming, LandformParams, LsFactorParams, MultiHillshadeParams,
    MultiHillshadeStreaming, MrvbfParams, NormalDeviationParams, NormalDeviationStreaming,
    NorthnessStreaming, OpennessParams, PennockParams, PercentElevRangeParams,
    PercentElevRangeStreaming, RelativeAspectParams, SarParams, SlopeParams, SlopeStreaming,
    SlopeUnits, SphericalStdDevParams, SphericalStdDevStreaming, SvfParams, TpiParams,
    TpiStreaming, TriParams, TriStreaming, ViewshedParams, VrmParams, VrmStreaming,
};
use surtgis_core::StripProcessor;

use crate::commands::TerrainCommands;
use crate::helpers::{
    done, parse_advanced_curvature_type, read_dem, spinner, write_result, write_result_u8,
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

            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

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

        TerrainCommands::SolarRadiation { input, output, day, hour: _hour, latitude } => {
            use surtgis_algorithms::terrain::{solar_radiation, SolarParams};
            // solar_radiation needs slope+aspect as input (radians)
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Computing slope+aspect for solar...");
            let slope_r = slope(&dem, SlopeParams { units: SlopeUnits::Radians, z_factor: 1.0 })
                .context("Slope failed")?;
            let aspect_r = aspect(&dem, AspectOutput::Radians)
                .context("Aspect failed")?;
            pb.finish_and_clear();
            let pb = spinner(&format!("Solar radiation (day={})...", day));
            let result = solar_radiation(&slope_r, &aspect_r, SolarParams {
                day, latitude, transmittance: 0.7, ..SolarParams::default()
            }).context("Solar radiation failed")?;
            pb.finish_and_clear();
            write_result(&result.total, &output, compress)?;
            done("Solar radiation", &output, start.elapsed());
        }

        TerrainCommands::SolarRadiationAnnual { input, output, latitude } => {
            use surtgis_algorithms::terrain::{solar_radiation_annual, SolarParams};
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Computing slope+aspect...");
            let slope_r = slope(&dem, SlopeParams { units: SlopeUnits::Radians, z_factor: 1.0 })
                .context("Slope failed")?;
            let aspect_r = aspect(&dem, AspectOutput::Radians)
                .context("Aspect failed")?;
            pb.finish_and_clear();
            let pb = spinner("Annual solar radiation...");
            let result = solar_radiation_annual(&slope_r, &aspect_r, SolarParams {
                day: 1, latitude, transmittance: 0.7, ..SolarParams::default()
            }).context("Annual solar radiation failed")?;
            pb.finish_and_clear();
            write_result(&result.annual.total, &output, compress)?;
            done("Annual solar radiation", &output, start.elapsed());
        }

        TerrainCommands::ContourLines { input, output, interval, base } => {
            use surtgis_algorithms::terrain::{contour_lines, ContourParams};
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = contour_lines(&dem, ContourParams { interval, base })
                .context("Contour lines failed")?;
            write_result(&result, &output, compress)?;
            done("Contour lines", &output, start.elapsed());
        }

        TerrainCommands::CostDistance { input, sources, output } => {
            use surtgis_algorithms::terrain::{cost_distance, CostDistanceParams};
            let cost = read_dem(&input)?;
            let pb = spinner("Reading sources...");
            let src: surtgis_core::Raster<f64> = surtgis_core::io::read_geotiff(&sources, None)
                .context("Failed to read source raster")?;
            pb.finish_and_clear();
            // Extract source positions from non-zero cells
            let mut src_positions = Vec::new();
            let (sr, sc) = src.shape();
            for r in 0..sr {
                for c in 0..sc {
                    if let Some(&v) = src.data().get([r, c]) {
                        if v != 0.0 && v.is_finite() { src_positions.push((r, c)); }
                    }
                }
            }
            println!("  {} source cells", src_positions.len());
            let start = Instant::now();
            let pb = spinner("Cost distance...");
            let result = cost_distance(&cost, CostDistanceParams { sources: src_positions })
                .context("Cost distance failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Cost distance", &output, start.elapsed());
        }

        TerrainCommands::ShapeIndex { input, output } => {
            use surtgis_algorithms::terrain::shape_index;
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = shape_index(&dem).context("Shape index failed")?;
            write_result(&result, &output, compress)?;
            done("Shape index", &output, start.elapsed());
        }

        TerrainCommands::Curvedness { input, output } => {
            use surtgis_algorithms::terrain::curvedness;
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = curvedness(&dem).context("Curvedness failed")?;
            write_result(&result, &output, compress)?;
            done("Curvedness", &output, start.elapsed());
        }

        TerrainCommands::GaussianSmoothing { input, output, sigma, radius } => {
            use surtgis_algorithms::terrain::{gaussian_smoothing, GaussianSmoothingParams};
            let dem = read_dem(&input)?;
            let r = radius.unwrap_or((3.0 * sigma).ceil() as usize);
            let start = Instant::now();
            let pb = spinner(&format!("Gaussian smoothing (sigma={}, radius={})...", sigma, r));
            let result = gaussian_smoothing(&dem, GaussianSmoothingParams { radius: r, sigma })
                .context("Smoothing failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Gaussian smoothing", &output, start.elapsed());
        }

        TerrainCommands::FeaturePreservingSmoothing { input, output, strength: _strength, iterations } => {
            use surtgis_algorithms::terrain::{feature_preserving_smoothing, SmoothingParams};
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Feature-preserving smoothing...");
            let result = feature_preserving_smoothing(&dem, SmoothingParams {
                radius: 2, iterations, threshold: 15.0,
            }).context("Smoothing failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Feature-preserving smoothing", &output, start.elapsed());
        }

        TerrainCommands::WindExposure { input, output, direction, radius } => {
            use surtgis_algorithms::terrain::{wind_exposure, WindExposureParams};
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner(&format!("Wind exposure (dir={}°, radius={})...", direction, radius));
            let result = wind_exposure(&dem, WindExposureParams {
                radius, directions: 8,
                wind_direction: Some(direction.to_radians()),
                wind_window: 45.0,
            }).context("Wind exposure failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Wind exposure", &output, start.elapsed());
        }

        TerrainCommands::HorizonAngle { input, output, azimuth, radius } => {
            use surtgis_algorithms::terrain::horizon_angle_map;
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner(&format!("Horizon angle (az={}°, radius={})...", azimuth, radius));
            let result = horizon_angle_map(&dem, azimuth.to_radians(), radius)
                .context("Horizon angle failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Horizon angle", &output, start.elapsed());
        }

        TerrainCommands::AccumulationZones { input, output } => {
            use surtgis_algorithms::terrain::accumulation_zones;
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Accumulation zones...");
            let result = accumulation_zones(&dem)
                .context("Accumulation zones failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Accumulation zones", &output, start.elapsed());
        }

        TerrainCommands::Spi { flow_acc, slope: slope_path, output } => {
            use surtgis_algorithms::terrain::spi;
            let fa = read_dem(&flow_acc)?;
            let sl = read_dem(&slope_path)?;
            let start = Instant::now();
            let result = spi(&fa, &sl).context("SPI failed")?;
            write_result(&result, &output, compress)?;
            done("SPI", &output, start.elapsed());
        }

        TerrainCommands::Sti { flow_acc, slope: slope_path, output } => {
            use surtgis_algorithms::terrain::{sti, StiParams};
            let fa = read_dem(&flow_acc)?;
            let sl = read_dem(&slope_path)?;
            let start = Instant::now();
            let result = sti(&fa, &sl, StiParams { m: 0.4, n: 1.3 })
                .context("STI failed")?;
            write_result(&result, &output, compress)?;
            done("STI", &output, start.elapsed());
        }

        TerrainCommands::Twi { flow_acc, slope: slope_path, output } => {
            use surtgis_algorithms::terrain::twi;
            let fa = read_dem(&flow_acc)?;
            let sl = read_dem(&slope_path)?;
            let start = Instant::now();
            let result = twi(&fa, &sl).context("TWI failed")?;
            write_result(&result, &output, compress)?;
            done("TWI", &output, start.elapsed());
        }

        TerrainCommands::LogTransform { input, output } => {
            use surtgis_algorithms::terrain::log_transform;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = log_transform(&raster).context("Log transform failed")?;
            write_result(&result, &output, compress)?;
            done("Log transform", &output, start.elapsed());
        }

        TerrainCommands::Uncertainty { input, outdir, error_std, n_simulations: _n } => {
            use surtgis_algorithms::terrain::{uncertainty, UncertaintyParams};
            std::fs::create_dir_all(&outdir).context("Failed to create output directory")?;
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Uncertainty analysis...");
            let result = uncertainty(&dem, UncertaintyParams { dem_rmse: error_std })
                .context("Uncertainty failed")?;
            pb.finish_and_clear();
            write_result(&result.slope_rmse, &outdir.join("slope_rmse.tif"), compress)?;
            write_result(&result.aspect_rmse, &outdir.join("aspect_rmse.tif"), compress)?;
            done("Uncertainty", &outdir, start.elapsed());
        }

        TerrainCommands::ViewshedPderl { input, output, row, col, height } => {
            use surtgis_algorithms::terrain::{viewshed_pderl, PderlViewshedParams};
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("PDERL viewshed...");
            let result = viewshed_pderl(&dem, PderlViewshedParams {
                observer_row: row, observer_col: col,
                observer_height: height, target_height: 0.0,
                max_radius: 0,
            }).context("Viewshed failed")?;
            pb.finish_and_clear();
            write_result_u8(&result, &output, compress)?;
            done("PDERL Viewshed", &output, start.elapsed());
        }

        TerrainCommands::ViewshedXdraw { input, output, row, col, height } => {
            use surtgis_algorithms::terrain::viewshed_xdraw;
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("XDraw viewshed...");
            let result = viewshed_xdraw(&dem, ViewshedParams {
                observer_row: row, observer_col: col,
                observer_height: height, target_height: 0.0,
                max_radius: 0,
            }).context("Viewshed failed")?;
            pb.finish_and_clear();
            write_result_u8(&result, &output, compress)?;
            done("XDraw Viewshed", &output, start.elapsed());
        }

        TerrainCommands::ViewshedMultiple { input, output, observers, height } => {
            use surtgis_algorithms::terrain::viewshed_multiple;
            let dem = read_dem(&input)?;
            let obs: Vec<(usize, usize)> = observers.split(';')
                .filter_map(|s| {
                    let parts: Vec<&str> = s.trim().split(',').collect();
                    if parts.len() == 2 {
                        Some((parts[0].trim().parse().ok()?, parts[1].trim().parse().ok()?))
                    } else { None }
                })
                .collect();
            println!("  {} observer locations", obs.len());
            let start = Instant::now();
            let pb = spinner("Multiple viewshed...");
            let result = viewshed_multiple(&dem, &obs, height)
                .context("Multiple viewshed failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Multiple Viewshed", &output, start.elapsed());
        }

        TerrainCommands::HypsometricHillshade { input, output, azimuth, altitude, z_factor } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = hypsometric_hillshade(
                &dem,
                HillshadeParams { azimuth, altitude, z_factor, normalized: true },
            ).context("Failed to compute hypsometric hillshade")?;
            write_result(&result, &output, compress)?;
            done("Hypsometric Hillshade", &output, start.elapsed());
        }

        TerrainCommands::ElevRelative { input, output } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = elev_relative_to_min_max(&dem)
                .context("Failed to compute elevation relative to min/max")?;
            write_result(&result, &output, compress)?;
            done("Elev Relative to Min/Max", &output, start.elapsed());
        }

        TerrainCommands::DiffFromMean { input, output, radius } => {
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

            if use_streaming {
                let algo = DiffFromMeanStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute diff-from-mean (streaming)")?;
                let elapsed = start.elapsed();
                println!("Diff from Mean (streaming): {} x {}", cols, rows);
                done("Diff from Mean Elev", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = diff_from_mean_elev(&dem, DiffFromMeanParams { radius })
                    .context("Failed to compute diff from mean elevation")?;
                write_result(&result, &output, compress)?;
                done("Diff from Mean Elev", &output, start.elapsed());
            }
        }

        TerrainCommands::PercentElevRange { input, output, radius } => {
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

            if use_streaming {
                let algo = PercentElevRangeStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute percent-elev-range (streaming)")?;
                let elapsed = start.elapsed();
                println!("Percent Elev Range (streaming): {} x {}", cols, rows);
                done("Percent Elev Range", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = percent_elev_range(&dem, PercentElevRangeParams { radius })
                    .context("Failed to compute percent elevation range")?;
                write_result(&result, &output, compress)?;
                done("Percent Elev Range", &output, start.elapsed());
            }
        }

        TerrainCommands::ElevAbovePit { input, output } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = elev_above_pit(&dem)
                .context("Failed to compute elevation above pit")?;
            write_result(&result, &output, compress)?;
            done("Elev Above Pit", &output, start.elapsed());
        }

        TerrainCommands::CircularVarianceAspect { input, output, radius } => {
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

            if use_streaming {
                let algo = CircularVarianceStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute circular variance of aspect (streaming)")?;
                let elapsed = start.elapsed();
                println!("Circular Variance Aspect (streaming): {} x {}", cols, rows);
                done("Circular Variance Aspect", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = circular_variance_aspect(&dem, CircularVarianceParams { radius })
                    .context("Failed to compute circular variance of aspect")?;
                write_result(&result, &output, compress)?;
                done("Circular Variance Aspect", &output, start.elapsed());
            }
        }

        TerrainCommands::Neighbours { input, output, stat } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = neighbour_stats(&dem)
                .context("Failed to compute neighbour statistics")?;

            if let Some(stat_name) = stat {
                // Single statistic → write to output file
                let raster = match stat_name.as_str() {
                    "max_downslope_change" => &result.max_downslope_change,
                    "min_downslope_change" => &result.min_downslope_change,
                    "max_upslope_change" => &result.max_upslope_change,
                    "num_downslope" => &result.num_downslope,
                    "num_upslope" => &result.num_upslope,
                    other => anyhow::bail!(
                        "Unknown stat '{}'. Valid: max_downslope_change, min_downslope_change, max_upslope_change, num_downslope, num_upslope",
                        other
                    ),
                };
                write_result(raster, &output, compress)?;
                done(&format!("Neighbours ({})", stat_name), &output, start.elapsed());
            } else {
                // All 5 → write to output directory
                std::fs::create_dir_all(&output)
                    .context("Failed to create output directory")?;
                write_result(&result.max_downslope_change, &output.join("max_downslope_change.tif"), compress)?;
                write_result(&result.min_downslope_change, &output.join("min_downslope_change.tif"), compress)?;
                write_result(&result.max_upslope_change, &output.join("max_upslope_change.tif"), compress)?;
                write_result(&result.num_downslope, &output.join("num_downslope.tif"), compress)?;
                write_result(&result.num_upslope, &output.join("num_upslope.tif"), compress)?;
                println!("  5 neighbour statistics saved to {}", output.display());
                done("Neighbour Stats", &output, start.elapsed());
            }
        }

        TerrainCommands::Pennock { input, output, slope_threshold, curv_threshold } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = pennock(&dem, PennockParams {
                slope_threshold,
                profile_curv_threshold: curv_threshold,
                plan_curv_threshold: 0.0,
            }).context("Failed to compute Pennock classification")?;
            write_result(&result, &output, compress)?;
            done("Pennock Landform", &output, start.elapsed());
        }

        TerrainCommands::EdgeDensity { input, output, radius, threshold } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = edge_density(&dem, EdgeDensityParams { radius, threshold })
                .context("Failed to compute edge density")?;
            write_result(&result, &output, compress)?;
            done("Edge Density", &output, start.elapsed());
        }

        TerrainCommands::RelativeAspect { input, output, sigma } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = relative_aspect(&dem, RelativeAspectParams { sigma })
                .context("Failed to compute relative aspect")?;
            write_result(&result, &output, compress)?;
            done("Relative Aspect", &output, start.elapsed());
        }

        TerrainCommands::NormalDeviation { input, output, radius } => {
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

            if use_streaming {
                let algo = NormalDeviationStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute normal deviation (streaming)")?;
                let elapsed = start.elapsed();
                println!("Normal Deviation (streaming): {} x {}", cols, rows);
                done("Normal Deviation", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = normal_vector_deviation(&dem, NormalDeviationParams { radius })
                    .context("Failed to compute normal vector deviation")?;
                write_result(&result, &output, compress)?;
                done("Normal Deviation", &output, start.elapsed());
            }
        }

        TerrainCommands::SphericalStdDev { input, output, radius } => {
            let use_streaming = memory::should_stream(&input, mem_limit_bytes, streaming)?;

            if use_streaming {
                let algo = SphericalStdDevStreaming { radius };
                let processor = StripProcessor::new(256);
                let start = Instant::now();
                let (rows, cols) = processor.process(&input, &output, &algo, compress)
                    .context("Failed to compute spherical std dev (streaming)")?;
                let elapsed = start.elapsed();
                println!("Spherical Std Dev (streaming): {} x {}", cols, rows);
                done("Spherical Std Dev", &output, elapsed);
            } else {
                let dem = read_dem(&input)?;
                let start = Instant::now();
                let result = spherical_std_dev(&dem, SphericalStdDevParams { radius })
                    .context("Failed to compute spherical std dev of normals")?;
                write_result(&result, &output, compress)?;
                done("Spherical Std Dev", &output, start.elapsed());
            }
        }

        TerrainCommands::DirectionalRelief { input, output, radius, azimuth } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = directional_relief(&dem, DirectionalReliefParams { radius, azimuth })
                .context("Failed to compute directional relief")?;
            write_result(&result, &output, compress)?;
            done("Directional Relief", &output, start.elapsed());
        }

        TerrainCommands::DownslopeIndex { input, output, drop: drop_val } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let result = downslope_index(&dem, DownslopeIndexParams { drop: drop_val })
                .context("Failed to compute downslope index")?;
            write_result(&result, &output, compress)?;
            done("Downslope Index", &output, start.elapsed());
        }

        TerrainCommands::MaxBranchLength { input, output } => {
            let dem = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Computing max branch length (fill + flow dir + topo sort)...");
            let result = max_branch_length(&dem)
                .context("Failed to compute max branch length")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Max Branch Length", &output, start.elapsed());
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
