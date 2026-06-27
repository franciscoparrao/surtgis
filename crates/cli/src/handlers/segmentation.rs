//! Handlers for image segmentation (SLIC, Felzenszwalb-Huttenlocher).

use anyhow::{Context, Result};
use std::time::Instant;

use surtgis_algorithms::segmentation::{FelzenszwalbParams, SlicParams, felzenszwalb, slic};

use crate::commands::SegmentationCommands;
use crate::helpers::{done, read_dem, spinner, write_result_i32};

pub fn handle(algorithm: SegmentationCommands, compress: bool) -> Result<()> {
    match algorithm {
        SegmentationCommands::Slic {
            output,
            bands,
            n_segments,
            compactness,
            max_iter,
            no_connectivity,
        } => {
            let rasters = bands
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading band {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<_> = rasters.iter().collect();
            let start = Instant::now();
            let pb = spinner(&format!(
                "SLIC (n_segments={}, m={}, bands={})...",
                n_segments,
                compactness,
                bands.len()
            ));
            let result = slic(
                &refs,
                SlicParams {
                    n_segments,
                    compactness,
                    max_iter,
                    enforce_connectivity: !no_connectivity,
                },
            )
            .context("SLIC failed")?;
            pb.finish_and_clear();
            write_result_i32(&result, &output, compress)?;
            done("SLIC", &output, start.elapsed());
        }
        SegmentationCommands::Felzenszwalb {
            output,
            bands,
            scale,
            min_size,
        } => {
            let rasters = bands
                .iter()
                .map(|p| read_dem(p).with_context(|| format!("reading band {}", p.display())))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<_> = rasters.iter().collect();
            let start = Instant::now();
            let pb = spinner(&format!(
                "Felzenszwalb (k={}, min_size={}, bands={})...",
                scale,
                min_size,
                bands.len()
            ));
            let result = felzenszwalb(&refs, FelzenszwalbParams { scale, min_size })
                .context("Felzenszwalb failed")?;
            pb.finish_and_clear();
            write_result_i32(&result, &output, compress)?;
            done("Felzenszwalb", &output, start.elapsed());
        }
    }
    Ok(())
}
