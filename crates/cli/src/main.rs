//! SurtGis CLI - High-performance geospatial analysis

mod commands;
mod handlers;
mod helpers;
mod memory;
mod stac_introspect;
mod streaming;

use anyhow::Result;
use clap::Parser;

use commands::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let compress = cli.compress;
    let streaming = cli.streaming;
    let verbose = cli.verbose;

    // Parse max_memory to bytes if provided
    let mem_limit_bytes = cli.max_memory
        .as_ref()
        .map(|s| memory::parse_memory_size(s))
        .transpose()?;

    helpers::setup_logging(verbose);

    match cli.command {
        Commands::Info { input } => handlers::info::handle(input)?,
        Commands::Terrain { algorithm } => handlers::terrain::handle(algorithm, compress, streaming, mem_limit_bytes)?,
        Commands::Hydrology { algorithm } => handlers::hydrology::handle(algorithm, compress, mem_limit_bytes)?,
        Commands::Imagery { algorithm } => handlers::imagery::handle(algorithm, compress)?,
        Commands::Morphology { algorithm } => handlers::morphology::handle(algorithm, compress)?,
        Commands::Landscape { algorithm } => handlers::landscape::handle(algorithm, compress)?,
        Commands::Clip { input, polygon, output } => handlers::clip::handle_clip(input, polygon, output, compress, mem_limit_bytes)?,
        Commands::Rasterize { input, output, reference, attribute } => handlers::clip::handle_rasterize(input, output, reference, attribute, compress)?,
        Commands::Resample { input, output, reference, method } => handlers::clip::handle_resample(input, output, reference, method, compress)?,
        Commands::Mosaic { input, output } => handlers::mosaic::handle(input, output, compress)?,
        #[cfg(feature = "cloud")]
        Commands::Cog { action } => handlers::cog::handle(action, compress)?,
        #[cfg(feature = "cloud")]
        Commands::Stac { action } => handlers::stac::handle(action, compress)?,
        Commands::Pipeline { action } => handlers::pipeline::handle(action, compress, mem_limit_bytes)?,
        Commands::Vector { action } => handlers::vector::handle(action)?,
        Commands::Interpolation { action } => handlers::interpolation::handle(action, compress)?,
        Commands::Temporal { action } => handlers::temporal::handle(action, compress)?,
        Commands::Classification { algorithm } => handlers::classification::handle(algorithm, compress)?,
        Commands::Texture { algorithm } => handlers::texture::handle(algorithm, compress)?,
        Commands::Statistics { algorithm } => handlers::statistics::handle(algorithm, compress)?,
        #[cfg(feature = "ml")]
        Commands::Ml { action } => handlers::ml::handle(action, compress)?,
    }

    Ok(())
}
