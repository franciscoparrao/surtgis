//! Handlers for `surtgis ecw` — the native ECW v2 reader.

use crate::commands::EcwCommands;
use crate::helpers::write_opts;
use anyhow::{Context, Result, bail};
use std::time::Instant;
use surtgis_core::io::write_geotiff;
use surtgis_ecw::{EcwReader, RegionParams};

/// Dispatch an `surtgis ecw` subcommand.
pub fn handle(action: EcwCommands, compress: bool) -> Result<()> {
    match action {
        EcwCommands::Info { input } => {
            let reader = EcwReader::open(&input)
                .with_context(|| format!("Failed to open {}", input.display()))?;
            let h = reader.header();
            println!("ECW version {} ({:?})", h.version, h.compress_format);
            println!(
                "Size:        {} x {} cells, {} bands",
                h.x_size, h.y_size, h.nr_bands
            );
            println!(
                "Cell size:   {} x {} ({:?})",
                h.cell_increment_x, h.cell_increment_y, h.cell_units
            );
            println!("Origin:      ({}, {})", h.origin_x, h.origin_y);
            match h.epsg() {
                Some(code) => {
                    println!("CRS:         {} / {} -> EPSG:{code}", h.datum, h.projection)
                }
                None => println!(
                    "CRS:         {} / {} (no EPSG mapping)",
                    h.datum, h.projection
                ),
            }
            println!(
                "Compression: target rate {}:1, blocks {}x{}, {} blocks total",
                h.compression_rate, h.x_block_size, h.y_block_size, h.total_blocks
            );
            println!("Pyramid:     {} levels", h.num_levels);
            for (i, level) in h.levels.iter().enumerate() {
                println!(
                    "  level {i}: {} x {} ({} x {} blocks, binsize {:?})",
                    level.x_size,
                    level.y_size,
                    level.nr_x_blocks,
                    level.nr_y_blocks,
                    level.bin_sizes
                );
            }
        }
        EcwCommands::Convert {
            input,
            output,
            reduction,
            window,
        } => {
            let mut reader = EcwReader::open(&input)
                .with_context(|| format!("Failed to open {}", input.display()))?;
            let (x_size, y_size) = (reader.header().x_size, reader.header().y_size);

            let params = match window {
                Some(w) => {
                    let [x, y, width, height]: [u32; 4] = w
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("--window takes exactly 4 values"))?;
                    if width == 0 || height == 0 {
                        bail!("--window width/height must be positive");
                    }
                    RegionParams {
                        start_x: x,
                        start_y: y,
                        end_x: x + width - 1,
                        end_y: y + height - 1,
                        number_x: (width >> reduction).max(1),
                        number_y: (height >> reduction).max(1),
                    }
                }
                None => RegionParams {
                    start_x: 0,
                    start_y: 0,
                    end_x: x_size - 1,
                    end_y: y_size - 1,
                    number_x: (x_size >> reduction).max(1),
                    number_y: (y_size >> reduction).max(1),
                },
            };

            let start = Instant::now();
            let bands = reader
                .read_region(params)
                .context("Failed to decode ECW region")?;
            let elapsed = start.elapsed();
            let (rows, cols) = bands[0].shape();
            println!(
                "Decoded {} bands of {}x{} (1:{}) in {:.2?}",
                bands.len(),
                cols,
                rows,
                1u32 << reduction,
                elapsed
            );

            let options = write_opts(compress);
            let stem = output.display().to_string();
            let stem = stem.strip_suffix(".tif").unwrap_or(&stem).to_string();
            for (i, band) in bands.iter().enumerate() {
                let path = format!("{}_b{}.tif", stem, i + 1);
                write_geotiff(band, &path, Some(options.clone()))
                    .with_context(|| format!("Failed to write {path}"))?;
                println!("  wrote {path}");
            }
        }
    }
    Ok(())
}
