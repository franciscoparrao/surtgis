//! SurtGis CLI - High-performance geospatial analysis

mod commands;
mod handlers;
mod helpers;
mod streaming;

use anyhow::Result;
use clap::Parser;

use commands::{Cli, Commands};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let compress = cli.compress;
    let streaming = cli.streaming;
    let verbose = cli.verbose;
    helpers::setup_logging(verbose);

    match cli.command {
        Commands::Info { input } => handlers::info::handle(input)?,
        Commands::Terrain { algorithm } => handlers::terrain::handle(algorithm, compress, streaming)?,
        Commands::Hydrology { algorithm } => handlers::hydrology::handle(algorithm, compress)?,
        Commands::Imagery { algorithm } => handlers::imagery::handle(algorithm, compress)?,
        Commands::Morphology { algorithm } => handlers::morphology::handle(algorithm, compress)?,
        Commands::Landscape { algorithm } => handlers::landscape::handle(algorithm, compress)?,
        Commands::Clip { input, polygon, output } => handlers::clip::handle_clip(input, polygon, output, compress)?,
        Commands::Rasterize { input, output, reference, attribute } => handlers::clip::handle_rasterize(input, output, reference, attribute, compress)?,
        Commands::Mosaic { input, output } => handlers::mosaic::handle(input, output, compress)?,
        #[cfg(feature = "cloud")]
        Commands::Cog { action } => handlers::cog::handle(action, compress)?,
        #[cfg(feature = "cloud")]
        Commands::Stac { action } => handlers::stac::handle(action, compress)?,
    }

    Ok(())
}
