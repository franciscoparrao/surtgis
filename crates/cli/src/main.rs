//! SurtGis CLI - High-performance geospatial analysis

mod commands;
mod handlers;
mod helpers;
mod memory;
// streaming + stac_introspect import from surtgis_cloud unconditionally
// and are consumed only by the cog/stac handlers (both cloud-gated).
// Gate them too so --no-default-features builds cleanly.
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
use clap::Parser;

use commands::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();
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
    }

    Ok(())
}
