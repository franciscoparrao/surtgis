//! Handler for pipeline workflows.

use anyhow::{Context, Result};
use std::path::Path;
use std::path::PathBuf;

use crate::commands::PipelineCommands;
use crate::helpers;

pub fn handle(
    command: PipelineCommands,
    compress: bool,
    mem_limit_bytes: Option<u64>,
) -> Result<()> {
    match command {
        PipelineCommands::Susceptibility {
            dem,
            s2,
            bbox,
            datetime,
            outdir,
            max_scenes,
            scl_keep,
        } => handle_susceptibility(
            &dem, &s2, &bbox, &datetime, &outdir, max_scenes, &scl_keep, compress,
            mem_limit_bytes,
        ),
    }
}

pub fn handle_susceptibility(
    dem_source: &str,
    s2_source: &str,
    _bbox_str: &str,
    _datetime_str: &str,
    outdir: &Path,
    _max_scenes: usize,
    _scl_keep_str: &str,
    compress: bool,
    mem_limit_bytes: Option<u64>,
) -> Result<()> {
    // Create output directory structure
    std::fs::create_dir_all(outdir).context("Failed to create output directory")?;
    std::fs::create_dir_all(&outdir.join("terrain"))
        .context("Failed to create terrain directory")?;
    std::fs::create_dir_all(&outdir.join("hydrology"))
        .context("Failed to create hydrology directory")?;

    // ========== STEP 1: DEM ==========
    let pb = helpers::spinner("Pipeline: DEM");

    let dem_path = if dem_source.contains('/') || dem_source.contains('\\') {
        // Local file
        PathBuf::from(dem_source)
    } else {
        // STAC source - for MVP, skip STAC download
        pb.finish_and_clear();
        return Err(anyhow::anyhow!(
            "MVP: provide local DEM path (STAC download not implemented)"
        ));
    };

    // Verify DEM exists
    if !dem_path.exists() {
        pb.finish_and_clear();
        return Err(anyhow::anyhow!("DEM file not found: {}", dem_path.display()));
    }

    // Copy DEM to output
    let output_dem = outdir.join("dem.tif");
    std::fs::copy(&dem_path, &output_dem)
        .context("Failed to copy DEM to output directory")?;
    pb.finish_and_clear();

    // ========== STEP 2: Terrain All ==========
    let pb = helpers::spinner("Pipeline: Terrain (17 products)");

    handle_terrain_all(&dem_path, &outdir.join("terrain"), compress, mem_limit_bytes)?;

    pb.finish_and_clear();

    // ========== STEP 3: Hydrology All ==========
    let pb = helpers::spinner("Pipeline: Hydrology (8 products)");

    handle_hydrology_all(&dem_path, &outdir.join("hydrology"), compress, mem_limit_bytes)?;

    pb.finish_and_clear();

    // ========== STEP 4: S2 (optional) ==========
    if s2_source != "skip" {
        let pb = helpers::spinner("Pipeline: Sentinel-2 (experimental)");
        // TODO: Implement S2 download via STAC
        // For MVP: skip S2 processing
        pb.finish_and_clear();
    }

    // ========== Output Summary ==========
    println!("\n✅ Pipeline complete: {}", outdir.display());
    println!("\nOutputs:");
    println!("  📁 Terrain factors (17):");
    println!("     {}/terrain/*.tif", outdir.display());
    println!("  📁 Hydrology factors (8):");
    println!("     {}/hydrology/*.tif", outdir.display());
    println!("  📁 Original DEM:");
    println!("     {}/dem.tif", outdir.display());

    Ok(())
}

/// Compute all terrain factors from DEM
pub fn handle_terrain_all(
    input: &Path,
    outdir: &Path,
    compress: bool,
    _mem_limit_bytes: Option<u64>,
) -> Result<()> {
    // Delegate to terrain handler
    super::terrain::handle_terrain_all(input, outdir, compress)
}

/// Compute all hydrology factors from DEM
pub fn handle_hydrology_all(
    input: &Path,
    outdir: &Path,
    compress: bool,
    mem_limit_bytes: Option<u64>,
) -> Result<()> {
    // Delegate to hydrology handler
    super::hydrology::handle_hydrology_all(input, outdir, compress, mem_limit_bytes)
}
