//! Handlers for texture analysis algorithms.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::texture::{
    GlcmParams, GlcmTexture, LbpParams, LbpVariant, haralick_glcm, haralick_glcm_multi, laplacian,
    lbp, sobel_edge,
};

use crate::commands::TextureCommands;
use crate::helpers::{done, read_dem, spinner, write_result};

pub fn handle(algorithm: TextureCommands, compress: bool) -> Result<()> {
    match algorithm {
        TextureCommands::Glcm {
            input,
            output,
            texture,
            radius,
            levels,
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

        TextureCommands::GlcmAll {
            input,
            output_dir,
            radius,
            levels,
        } => {
            std::fs::create_dir_all(&output_dir)
                .with_context(|| format!("failed to create {}", output_dir.display()))?;
            let raster = read_dem(&input)?;
            let textures = [
                ("energy", GlcmTexture::Energy),
                ("contrast", GlcmTexture::Contrast),
                ("homogeneity", GlcmTexture::Homogeneity),
                ("correlation", GlcmTexture::Correlation),
                ("entropy", GlcmTexture::Entropy),
                ("dissimilarity", GlcmTexture::Dissimilarity),
            ];
            let kinds: Vec<GlcmTexture> = textures.iter().map(|(_, k)| *k).collect();
            let start = Instant::now();
            let pb = spinner(&format!("GLCM all 6 textures (radius={})...", radius));
            let params = GlcmParams {
                radius,
                n_levels: levels,
                distance: 1,
                texture: GlcmTexture::Contrast, // ignored by multi
            };
            let results =
                haralick_glcm_multi(&raster, &params, &kinds).context("GLCM multi failed")?;
            pb.finish_and_clear();
            for ((name, _), out) in textures.iter().zip(results.iter()) {
                let path = output_dir.join(format!("glcm_{}.tif", name));
                write_result(out, &path, compress)?;
            }
            done(
                "GLCM all 6 textures",
                &output_dir.join("glcm_*.tif"),
                start.elapsed(),
            );
        }

        TextureCommands::Lbp {
            input,
            output,
            variant,
        } => {
            let v = match variant.to_lowercase().as_str() {
                "standard" => LbpVariant::Standard,
                "riu2" | "uniform" | "rotation-invariant" => LbpVariant::RotationInvariantUniform,
                _ => {
                    eprintln!("Unknown LBP variant '{}', using standard", variant);
                    LbpVariant::Standard
                }
            };
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let pb = spinner(&format!("LBP ({})...", variant));
            let result = lbp(&raster, LbpParams { variant: v }).context("LBP failed")?;
            pb.finish_and_clear();
            write_result(&result, &output, compress)?;
            done(&format!("LBP {}", variant), &output, start.elapsed());
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
