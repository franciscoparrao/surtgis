//! Handlers for texture analysis algorithms.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::texture::{haralick_glcm, laplacian, sobel_edge, GlcmParams, GlcmTexture};

use crate::commands::TextureCommands;
use crate::helpers::{done, read_dem, spinner, write_result};

pub fn handle(algorithm: TextureCommands, compress: bool) -> Result<()> {
    match algorithm {
        TextureCommands::Glcm {
            input, output, texture, radius, levels,
        } => {
            let tex = match texture.to_lowercase().as_str() {
                "energy" => GlcmTexture::Energy,
                "contrast" => GlcmTexture::Contrast,
                "homogeneity" => GlcmTexture::Homogeneity,
                "correlation" => GlcmTexture::Correlation,
                "entropy" => GlcmTexture::Entropy,
                "dissimilarity" => GlcmTexture::Dissimilarity,
                _ => {
                    eprintln!("Unknown texture '{}', using Contrast", texture);
                    GlcmTexture::Contrast
                }
            };
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner(&format!("GLCM {} (radius={})...", texture, radius));
            let result = haralick_glcm(
                &raster,
                GlcmParams {
                    radius,
                    n_levels: levels,
                    distance: 1,
                    texture: tex,
                },
            )
            .context("GLCM failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done(&format!("GLCM {}", texture), &output, start.elapsed());
        }

        TextureCommands::Sobel { input, output } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Sobel edge detection...");
            let result = sobel_edge(&raster).context("Sobel failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Sobel", &output, start.elapsed());
        }

        TextureCommands::Laplacian { input, output } => {
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner("Laplacian edge detection...");
            let result = laplacian(&raster).context("Laplacian failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done("Laplacian", &output, start.elapsed());
        }
    }
    Ok(())
}
