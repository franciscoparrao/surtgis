//! Handler for mathematical morphology subcommands.

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::morphology::{
    black_hat, closing, dilate, erode, gradient, opening, top_hat,
};

use crate::commands::MorphologyCommands;
use crate::helpers::{done, parse_se, read_dem, write_result};

pub fn handle(algorithm: MorphologyCommands, compress: bool) -> Result<()> {
    match algorithm {
        MorphologyCommands::Erode {
            input,
            output,
            shape,
            radius,
        } => {
            let se = parse_se(&shape, radius)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = erode(&raster, &se).context("Failed to erode")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Erode", &output, elapsed);
        }

        MorphologyCommands::Dilate {
            input,
            output,
            shape,
            radius,
        } => {
            let se = parse_se(&shape, radius)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = dilate(&raster, &se).context("Failed to dilate")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Dilate", &output, elapsed);
        }

        MorphologyCommands::Opening {
            input,
            output,
            shape,
            radius,
        } => {
            let se = parse_se(&shape, radius)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = opening(&raster, &se).context("Failed to open")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Opening", &output, elapsed);
        }

        MorphologyCommands::Closing {
            input,
            output,
            shape,
            radius,
        } => {
            let se = parse_se(&shape, radius)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result = closing(&raster, &se).context("Failed to close")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Closing", &output, elapsed);
        }

        MorphologyCommands::Gradient {
            input,
            output,
            shape,
            radius,
        } => {
            let se = parse_se(&shape, radius)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result =
                gradient(&raster, &se).context("Failed to compute gradient")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Gradient", &output, elapsed);
        }

        MorphologyCommands::TopHat {
            input,
            output,
            shape,
            radius,
        } => {
            let se = parse_se(&shape, radius)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result =
                top_hat(&raster, &se).context("Failed to compute top-hat")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Top-hat", &output, elapsed);
        }

        MorphologyCommands::BlackHat {
            input,
            output,
            shape,
            radius,
        } => {
            let se = parse_se(&shape, radius)?;
            let raster = read_dem(&input)?;
            let start = Instant::now();
            let result =
                black_hat(&raster, &se).context("Failed to compute black-hat")?;
            let elapsed = start.elapsed();
            write_result(&result, &output, compress)?;
            done("Black-hat", &output, elapsed);
        }
    }

    Ok(())
}
