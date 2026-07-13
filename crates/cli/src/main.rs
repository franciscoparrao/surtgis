//! SurtGis CLI - High-performance geospatial analysis

mod commands;
mod handlers;
mod helpers;
mod memory;
// streaming + stac_introspect import from surtgis_cloud unconditionally
// and are consumed only by the cog/stac handlers (both cloud-gated).
// Gate them too so --no-default-features builds cleanly.
#[cfg(feature = "cloud")]
mod composite_sink;
#[cfg(feature = "cloud")]
mod stac_introspect;
#[cfg(feature = "cloud")]
mod streaming;

// mimalloc: global allocator (feature `mimalloc`, on by default). glibc
// malloc tends to hold freed memory in fragmented heap arenas indefinitely
// under mixed-size alloc/free workloads, which matches the v0.6.24 bug
// signature (linear RSS growth during long stac composite runs after an
// initial stable phase). mimalloc returns memory to the OS much more
// aggressively. It compiles C, so the feature can be disabled
// (`--no-default-features`) for pure-Rust / musl-static / no-cc builds,
// falling back to the system allocator.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use std::process::ExitCode;

use commands::{Cli, Commands};

/// Exit code for "input file not found" (`std::io::ErrorKind::NotFound`
/// anywhere in the error's source chain), distinct from the generic
/// failure code so scripts can tell a missing/mistyped path apart from a
/// computation error without parsing stderr text.
const EXIT_NOT_FOUND: u8 = 3;
/// Generic failure: everything that isn't a clap usage error (exit 2,
/// handled by clap itself before `main` runs) or a not-found error.
const EXIT_FAILURE: u8 = 1;

fn exit_code_for(err: &anyhow::Error) -> u8 {
    for cause in err.chain() {
        if let Some(io_err) = cause.downcast_ref::<std::io::Error>()
            && io_err.kind() == std::io::ErrorKind::NotFound
        {
            return EXIT_NOT_FOUND;
        }
    }
    EXIT_FAILURE
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e:?}");
            ExitCode::from(exit_code_for(&e))
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let compress = cli.compress;
    let streaming = cli.streaming;
    let verbose = cli.verbose;

    // Parse max_memory to bytes if provided
    let mem_limit_bytes = cli
        .max_memory
        .as_ref()
        .map(|s| memory::parse_memory_size(s))
        .transpose()?;

    helpers::setup_logging(verbose);

    match cli.command {
        Commands::Completions { shell } => {
            clap_complete::generate(
                shell,
                &mut commands::Cli::command(),
                "surtgis",
                &mut std::io::stdout(),
            );
        }
        Commands::Info { input } => handlers::info::handle(input)?,
        Commands::Terrain { algorithm } => {
            handlers::terrain::handle(algorithm, compress, streaming, mem_limit_bytes)?
        }
        Commands::Hydrology { algorithm } => {
            handlers::hydrology::handle(algorithm, compress, mem_limit_bytes)?
        }
        Commands::Fluvial { algorithm } => handlers::fluvial::handle(algorithm, compress)?,
        Commands::Imagery { algorithm } => handlers::imagery::handle(algorithm, compress)?,
        Commands::Morphology { algorithm } => handlers::morphology::handle(algorithm, compress)?,
        Commands::Landscape { algorithm } => handlers::landscape::handle(algorithm, compress)?,
        Commands::Extract {
            features_dir,
            points,
            target,
            output,
        } => handlers::extract::handle(&features_dir, &points, &target, &output)?,
        Commands::ExtractPatches {
            features_dir,
            points,
            polygons,
            label_col,
            size,
            stride,
            skip_nan_threshold,
            max_patches,
            seed,
            profile,
            output_format,
            emit_stac,
            points_crs,
            output,
        } => handlers::extract_patches::handle(
            &features_dir,
            points.as_deref(),
            polygons.as_deref(),
            &label_col,
            size,
            stride,
            skip_nan_threshold,
            max_patches,
            seed,
            profile.as_deref(),
            &output_format,
            emit_stac,
            points_crs,
            &output,
        )?,
        Commands::Relief {
            input,
            output,
            colormap,
            sun_azimuth,
            sun_altitude,
            shadows,
            soft,
            ambient,
            water,
            z_factor,
            radius,
        } => handlers::relief::handle(
            &input,
            &output,
            &colormap,
            sun_azimuth,
            sun_altitude,
            shadows,
            soft,
            ambient,
            water,
            z_factor,
            radius,
        )?,
        #[cfg(feature = "relief-3d")]
        Commands::Relief3d {
            input,
            output,
            colormap,
            width,
            height,
            sun_azimuth,
            sun_altitude,
            shadows,
            soft,
            ambient,
            vertical_exaggeration,
            camera_azimuth,
            camera_polar,
            camera_distance,
            haze,
            lod,
        } => handlers::relief_3d::handle(
            &input,
            &output,
            &colormap,
            width,
            height,
            sun_azimuth,
            sun_altitude,
            shadows,
            soft,
            ambient,
            vertical_exaggeration,
            camera_azimuth,
            camera_polar,
            camera_distance,
            haze,
            lod,
        )?,
        Commands::Clip {
            input,
            polygon,
            bbox,
            output,
        } => handlers::clip::handle_clip(input, polygon, bbox, output, compress, mem_limit_bytes)?,
        #[cfg(feature = "projections")]
        Commands::Reproject {
            input,
            output,
            to,
            from,
            method,
            pixel_size,
        } => handlers::reproject::handle(input, output, to, from, method, pixel_size, compress)?,
        Commands::Rasterize {
            input,
            output,
            reference,
            attribute,
        } => handlers::clip::handle_rasterize(input, output, reference, attribute, compress)?,
        Commands::Resample {
            input,
            output,
            reference,
            method,
        } => handlers::clip::handle_resample(input, output, reference, method, compress)?,
        Commands::Mosaic { input, output } => handlers::mosaic::handle(input, output, compress)?,
        #[cfg(feature = "cloud")]
        Commands::Cog { action } => handlers::cog::handle(action, compress)?,
        #[cfg(feature = "cloud")]
        Commands::Stac { action } => handlers::stac::handle(action, compress)?,
        #[cfg(feature = "cloud")]
        Commands::Pipeline { action } => {
            handlers::pipeline::handle(action, compress, mem_limit_bytes)?
        }
        Commands::Vector { action } => handlers::vector::handle(action)?,
        Commands::Interpolation { action } => handlers::interpolation::handle(action, compress)?,
        Commands::Temporal { action } => handlers::temporal::handle(action, compress)?,
        Commands::Classification { algorithm } => {
            handlers::classification::handle(algorithm, compress)?
        }
        Commands::Texture { algorithm } => handlers::texture::handle(algorithm, compress)?,
        Commands::Segmentation { algorithm } => {
            handlers::segmentation::handle(algorithm, compress)?
        }
        Commands::Statistics { algorithm } => handlers::statistics::handle(algorithm, compress)?,
        #[cfg(feature = "ml")]
        Commands::Ml { action } => handlers::ml::handle(action, compress)?,
        #[cfg(feature = "onnx")]
        Commands::Infer {
            model,
            features,
            out,
            tile_size,
            halo,
            argmax,
        } => handlers::infer::run(&model, &features, &out, tile_size, halo, argmax, compress)?,
    }

    Ok(())
}
