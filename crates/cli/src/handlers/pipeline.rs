//! Handler for pipeline workflows.

use anyhow::{Context, Result};
use std::path::Path;
use std::path::PathBuf;

use surtgis_algorithms::imagery::{
    bsi, evi, EviParams, mndwi, nbr, ndbi, ndmi, ndsi, ndvi, ndwi, savi, SaviParams,
};
use surtgis_core::Raster;

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
    std::fs::create_dir_all(&outdir.join("imagery"))
        .context("Failed to create imagery directory")?;

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

    // ========== STEP 4: S2 Imagery (optional) ==========
    if s2_source != "skip" {
        let pb = helpers::spinner("Pipeline: Sentinel-2 (6 bands)");

        // Download 6 S2 bands aligned to DEM grid
        let bands = download_s2_bands(
            s2_source,
            &_bbox_str,
            &_datetime_str,
            _max_scenes,
            &_scl_keep_str,
            &output_dem,
        );

        match bands {
            Ok(s2_bands) => {
                pb.finish_and_clear();

                // ========== STEP 5: Imagery Indices ==========
                let pb = helpers::spinner("Pipeline: Imagery indices (10)");

                compute_imagery_indices(&s2_bands, &outdir.join("imagery"), compress)?;

                pb.finish_and_clear();
            }
            Err(e) => {
                pb.finish_and_clear();
                eprintln!("⚠️  Skipping S2 imagery: {}", e);
                // Continue without S2 - not a fatal error for MVP
            }
        }
    }

    // ========== Output Summary ==========
    println!("\n✅ Pipeline complete: {}", outdir.display());
    println!("\nOutputs:");
    println!("  📁 Terrain factors (17):");
    println!("     {}/terrain/*.tif", outdir.display());
    println!("  📁 Hydrology factors (8):");
    println!("     {}/hydrology/*.tif", outdir.display());
    if s2_source != "skip" && outdir.join("imagery/ndvi.tif").exists() {
        println!("  📁 Imagery indices (10):");
        println!("     {}/imagery/*.tif", outdir.display());
    }
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

/// Container for Sentinel-2 bands required for spectral indices
struct S2Bands {
    blue: Raster<f64>,    // B02
    green: Raster<f64>,   // B03
    red: Raster<f64>,     // B04
    nir: Raster<f64>,     // B08
    swir1: Raster<f64>,   // B11
    swir2: Raster<f64>,   // B12
}

/// Fetch a single S2 band from STAC catalog
///
/// In production, this would:
/// 1. Call stac::handle_stac_composite() with specific asset key
/// 2. Return the aligned, cloud-masked raster
///
/// For MVP: returns error to trigger graceful fallback
fn fetch_s2_band(
    _collection: &str,     // "sentinel-2-l2a"
    _band: &str,           // "B04", "B08", etc.
    _bbox: &str,
    _datetime: &str,
    _max_scenes: usize,
    _align_to: &Path,
) -> Result<Raster<f64>> {
    anyhow::bail!(
        "STAC integration not implemented in MVP. \
         S2 imagery download requires surtgis-cloud STAC client setup."
    )
}

/// Download S2 bands from STAC catalog
/// Returns 6 bands (B02, B03, B04, B08, B11, B12) aligned to DEM grid
fn download_s2_bands(
    s2_source: &str,
    bbox_str: &str,
    datetime_str: &str,
    max_scenes: usize,
    scl_keep_str: &str,
    align_to: &Path,
) -> Result<S2Bands> {
    // Map S2 source to collection name
    let s2_lower = s2_source.to_lowercase();
    let collection = match s2_lower.as_str() {
        "sentinel-2" | "sentinel-2-l2a" => "sentinel-2-l2a",
        _ => s2_source,
    };

    // Fetch 6 critical bands (B02, B03, B04, B08, B11, B12)
    let blue = fetch_s2_band(collection, "B02", bbox_str, datetime_str, max_scenes, align_to)
        .context("Failed to fetch Blue band (B02)")?;
    let green = fetch_s2_band(collection, "B03", bbox_str, datetime_str, max_scenes, align_to)
        .context("Failed to fetch Green band (B03)")?;
    let red = fetch_s2_band(collection, "B04", bbox_str, datetime_str, max_scenes, align_to)
        .context("Failed to fetch Red band (B04)")?;
    let nir = fetch_s2_band(collection, "B08", bbox_str, datetime_str, max_scenes, align_to)
        .context("Failed to fetch NIR band (B08)")?;
    let swir1 = fetch_s2_band(collection, "B11", bbox_str, datetime_str, max_scenes, align_to)
        .context("Failed to fetch SWIR1 band (B11)")?;
    let swir2 = fetch_s2_band(collection, "B12", bbox_str, datetime_str, max_scenes, align_to)
        .context("Failed to fetch SWIR2 band (B12)")?;

    // Validate all bands have same dimensions
    if !(blue.rows() == green.rows() && green.rows() == red.rows() && red.rows() == nir.rows()
        && nir.rows() == swir1.rows() && swir1.rows() == swir2.rows())
    {
        anyhow::bail!(
            "S2 bands have mismatched dimensions (rows). \
             This shouldn't happen if aligned to same grid."
        );
    }

    let _ = scl_keep_str; // Cloud masking already applied in STAC composite

    Ok(S2Bands {
        blue,
        green,
        red,
        nir,
        swir1,
        swir2,
    })
}

/// Compute 10 spectral indices from S2 bands
/// Saves results to outdir with standard filenames
fn compute_imagery_indices(bands: &S2Bands, outdir: &Path, compress: bool) -> Result<()> {
    // Create imagery output directory
    std::fs::create_dir_all(outdir)
        .context("Failed to create imagery output directory")?;

    // NDVI = (NIR - Red) / (NIR + Red)
    let ndvi = ndvi(&bands.nir, &bands.red)
        .context("Failed to compute NDVI")?;
    helpers::write_result(&ndvi, &outdir.join("ndvi.tif"), compress)
        .context("Failed to write NDVI")?;

    // NDWI = (Green - NIR) / (Green + NIR)
    let ndwi = ndwi(&bands.green, &bands.nir)
        .context("Failed to compute NDWI")?;
    helpers::write_result(&ndwi, &outdir.join("ndwi.tif"), compress)
        .context("Failed to write NDWI")?;

    // MNDWI = (Green - SWIR1) / (Green + SWIR1)
    let mndwi = mndwi(&bands.green, &bands.swir1)
        .context("Failed to compute MNDWI")?;
    helpers::write_result(&mndwi, &outdir.join("mndwi.tif"), compress)
        .context("Failed to write MNDWI")?;

    // NBR = (NIR - SWIR2) / (NIR + SWIR2)
    let nbr = nbr(&bands.nir, &bands.swir2)
        .context("Failed to compute NBR")?;
    helpers::write_result(&nbr, &outdir.join("nbr.tif"), compress)
        .context("Failed to write NBR")?;

    // SAVI = (1 + L) * (NIR - Red) / (NIR + Red + L), with L=0.5
    let savi = savi(&bands.nir, &bands.red, SaviParams { l_factor: 0.5 })
        .context("Failed to compute SAVI")?;
    helpers::write_result(&savi, &outdir.join("savi.tif"), compress)
        .context("Failed to write SAVI")?;

    // EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    let evi = evi(&bands.nir, &bands.red, &bands.blue, EviParams::default())
        .context("Failed to compute EVI")?;
    helpers::write_result(&evi, &outdir.join("evi.tif"), compress)
        .context("Failed to write EVI")?;

    // BSI = ((SWIR2 + Red) - (NIR + Blue)) / ((SWIR2 + Red) + (NIR + Blue))
    let bsi = bsi(&bands.swir2, &bands.red, &bands.nir, &bands.blue)
        .context("Failed to compute BSI")?;
    helpers::write_result(&bsi, &outdir.join("bsi.tif"), compress)
        .context("Failed to write BSI")?;

    // NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
    let ndbi = ndbi(&bands.swir1, &bands.nir)
        .context("Failed to compute NDBI")?;
    helpers::write_result(&ndbi, &outdir.join("ndbi.tif"), compress)
        .context("Failed to write NDBI")?;

    // NDMI = (NIR - SWIR1) / (NIR + SWIR1)
    let ndmi = ndmi(&bands.nir, &bands.swir1)
        .context("Failed to compute NDMI")?;
    helpers::write_result(&ndmi, &outdir.join("ndmi.tif"), compress)
        .context("Failed to write NDMI")?;

    // NDSI = (Green - SWIR1) / (Green + SWIR1)
    let ndsi = ndsi(&bands.green, &bands.swir1)
        .context("Failed to compute NDSI")?;
    helpers::write_result(&ndsi, &outdir.join("ndsi.tif"), compress)
        .context("Failed to write NDSI")?;

    println!("  10 spectral indices computed");
    Ok(())
}
