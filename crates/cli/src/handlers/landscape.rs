//! Handler for landscape ecology subcommands.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::time::Instant;

use surtgis_algorithms::landscape::{
    class_metrics, label_patches, landscape_metrics, patch_metrics, patches_to_csv,
};

use crate::commands::LandscapeCommands;
use crate::helpers::{done, parse_connectivity, read_dem, write_result_i32};

pub fn handle(algorithm: LandscapeCommands, compress: bool) -> Result<()> {
    match algorithm {
        LandscapeCommands::LabelPatches {
            input,
            output,
            connectivity,
        } => {
            let conn = parse_connectivity(connectivity)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let (labels, num_patches) = label_patches(&raster, conn)
                .context("Failed to label patches")?;
            let elapsed = start.elapsed();
            write_result_i32(&labels, &output, compress)?;
            println!("{} patches found", num_patches);
            done("Label patches", &output, elapsed);
        }

        LandscapeCommands::PatchMetrics {
            input,
            output,
            connectivity,
        } => {
            let conn = parse_connectivity(connectivity)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let (labels, num_patches) = label_patches(&raster, conn)
                .context("Failed to label patches")?;
            let patches = patch_metrics(&raster, &labels, num_patches)
                .context("Failed to compute patch metrics")?;
            let elapsed = start.elapsed();
            let csv = patches_to_csv(&patches);
            std::fs::write(&output, &csv).context("Failed to write CSV")?;
            println!("{} patches, {} classes", patches.len(),
                patches.iter().map(|p| p.class).collect::<HashSet<_>>().len());
            done("Patch metrics", &output, elapsed);
        }

        LandscapeCommands::ClassMetrics {
            input,
            connectivity,
        } => {
            let conn = parse_connectivity(connectivity)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let (labels, num_patches) = label_patches(&raster, conn)
                .context("Failed to label patches")?;
            let patches = patch_metrics(&raster, &labels, num_patches)
                .context("Failed to compute patch metrics")?;
            let cm = class_metrics(&raster, &patches)
                .context("Failed to compute class metrics")?;
            let elapsed = start.elapsed();

            println!("{:<10} {:>10} {:>10} {:>8} {:>12} {:>8} {:>10}",
                "Class", "Area(m\u{b2})", "Proportion", "Patches", "MeanArea", "AI", "Cohesion");
            println!("{}", "-".repeat(78));
            for c in &cm {
                println!("{:<10} {:>10.1} {:>10.4} {:>8} {:>12.1} {:>8.1} {:>10.1}",
                    c.class, c.area_m2, c.proportion, c.num_patches,
                    c.mean_patch_area_m2, c.ai, c.cohesion);
            }
            println!("\n  Processing time: {:.2?}", elapsed);
        }

        LandscapeCommands::LandscapeMetrics { input } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let lm = landscape_metrics(&raster)
                .context("Failed to compute landscape metrics")?;
            let elapsed = start.elapsed();

            println!("Landscape Metrics:");
            println!("  SHDI (Shannon):  {:.4}", lm.shdi);
            println!("  SIDI (Simpson):  {:.4}", lm.sidi);
            println!("  Classes:         {}", lm.num_classes);
            println!("  Total cells:     {}", lm.total_cells);
            println!("  Total area:      {:.1} m\u{b2}", lm.total_area_m2);
            println!("  Processing time: {:.2?}", elapsed);
        }

        LandscapeCommands::Analyze {
            input,
            output_labels,
            output_csv,
            connectivity,
        } => {
            let conn = parse_connectivity(connectivity)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();

            // 1. Label patches
            let (labels, num_patches) = label_patches(&raster, conn)
                .context("Failed to label patches")?;
            println!("Patches: {}", num_patches);

            // 2. Patch metrics
            let patches = patch_metrics(&raster, &labels, num_patches)
                .context("Failed to compute patch metrics")?;

            // 3. Class metrics
            let cm = class_metrics(&raster, &patches)
                .context("Failed to compute class metrics")?;

            println!("\n{:<10} {:>10} {:>10} {:>8} {:>8} {:>10}",
                "Class", "Proportion", "Patches", "AI", "Cohesion", "MeanArea");
            println!("{}", "-".repeat(66));
            for c in &cm {
                println!("{:<10} {:>10.4} {:>8} {:>8.1} {:>10.1} {:>10.1}",
                    c.class, c.proportion, c.num_patches, c.ai, c.cohesion,
                    c.mean_patch_area_m2);
            }

            // 4. Landscape metrics
            let lm = landscape_metrics(&raster)
                .context("Failed to compute landscape metrics")?;
            println!("\nLandscape: SHDI={:.4}  SIDI={:.4}  classes={}",
                lm.shdi, lm.sidi, lm.num_classes);

            // 5. Write outputs
            if let Some(ref lp) = output_labels {
                write_result_i32(&labels, lp, compress)?;
                println!("Labels saved to: {}", lp.display());
            }
            if let Some(ref cp) = output_csv {
                let csv = patches_to_csv(&patches);
                std::fs::write(cp, &csv).context("Failed to write CSV")?;
                println!("Patch CSV saved to: {}", cp.display());
            }

            let elapsed = start.elapsed();
            println!("\n  Processing time: {:.2?}", elapsed);
        }
    }

    Ok(())
}
