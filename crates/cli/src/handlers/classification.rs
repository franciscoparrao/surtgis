//! Handlers for classification algorithms.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::classification::{
    isodata, kmeans_raster, maximum_likelihood, minimum_distance, pca,
    signatures_from_training, IsodataParams, KmeansParams, PcaParams,
};
use surtgis_core::io::read_geotiff;

use crate::commands::ClassificationCommands;
use crate::helpers::{done, read_dem, spinner, write_result};

pub fn handle(algorithm: ClassificationCommands, compress: bool) -> Result<()> {
    match algorithm {
        ClassificationCommands::Kmeans {
            input, output, classes, max_iter, convergence, seed,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("K-means clustering...");
            let result = kmeans_raster(
                &raster,
                KmeansParams {
                    k: classes,
                    max_iterations: max_iter,
                    convergence,
                    seed,
                },
            )
            .context("K-means failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("K-means", &output, start.elapsed());
        }

        ClassificationCommands::Isodata {
            input, output, classes, min_classes, max_classes, max_iter,
        } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("ISODATA clustering...");
            let result = isodata(
                &raster,
                IsodataParams {
                    initial_k: classes,
                    min_k: min_classes,
                    max_k: max_classes,
                    max_iterations: max_iter,
                    min_samples: 10,
                    max_std_dev: 10.0,
                    min_merge_distance: 5.0,
                    convergence: 0.001,
                },
            )
            .context("ISODATA failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("ISODATA", &output, start.elapsed());
        }

        ClassificationCommands::Pca {
            bands, output, components,
        } => {
            let paths: Vec<&str> = bands.split(',').map(|s| s.trim()).collect();
            let pb = spinner(&format!("Reading {} bands...", paths.len()));
            let rasters: Vec<surtgis_core::Raster<f64>> = paths
                .iter()
                .map(|p| read_geotiff(p, None).with_context(|| format!("Failed to read {}", p)))
                .collect::<Result<Vec<_>>>()?;
            pb.finish_and_clear();

            let refs: Vec<&surtgis_core::Raster<f64>> = rasters.iter().collect();
            let start = Instant::now();
            let pb = spinner("PCA...");
            let result = pca(&refs, PcaParams { n_components: components })
                .context("PCA failed")?;
            pb.finish_and_clear();

            // Write each component
            let stem = output.file_stem().unwrap_or_default().to_string_lossy();
            for (i, comp) in result.components.iter().enumerate() {
                let path = output.with_file_name(format!("{}_PC{}.tif", stem, i + 1));
                write_result(comp, &path, compress)?;
                println!(
                    "  PC{}: variance={:.4} ({:.1}%) → {}",
                    i + 1,
                    result.eigenvalues[i],
                    result.variance_explained[i] * 100.0,
                    path.display()
                );
            }
            done("PCA", &output, start.elapsed());
        }

        ClassificationCommands::MinDistance {
            input, output, training,
        } => {
            let raster = read_dem(&input)?;
            let training_raster = read_dem(&training)?;
            let start = Instant::now();
            let pb = spinner("Computing signatures...");
            let sigs = signatures_from_training(&training_raster, &raster)
                .context("Failed to extract signatures")?;
            pb.finish_and_clear();
            println!("  {} class signatures extracted", sigs.len());

            let pb = spinner("Minimum distance classification...");
            let result = minimum_distance(&raster, &sigs)
                .context("Minimum distance failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Minimum Distance", &output, start.elapsed());
        }

        ClassificationCommands::MaxLikelihood {
            input, output, training,
        } => {
            let raster = read_dem(&input)?;
            let training_raster = read_dem(&training)?;
            let start = Instant::now();
            let pb = spinner("Computing signatures...");
            let sigs = signatures_from_training(&training_raster, &raster)
                .context("Failed to extract signatures")?;
            pb.finish_and_clear();
            println!("  {} class signatures extracted", sigs.len());

            let pb = spinner("Maximum likelihood classification...");
            let result = maximum_likelihood(&raster, &sigs)
                .context("Maximum likelihood failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Maximum Likelihood", &output, start.elapsed());
        }
    }
    Ok(())
}
