//! Handler for pipeline workflows.

use anyhow::{Context, Result};
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::imagery::{
    bsi, evi, EviParams, mndwi, nbr, ndbi, ndmi, ndsi, ndvi, ndwi, savi, SaviParams,
};
use surtgis_algorithms::terrain::{
    accumulation_zones, landform_classification, surface_area_ratio, valley_depth, wind_exposure,
    LandformParams, SarParams, WindExposureParams,
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
        PipelineCommands::Features {
            input,
            outdir,
            skip_hydrology,
            extras,
            compress: compress_opt,
        } => {
            let effective_compress = compress_opt.unwrap_or(compress);
            handle_features_generate(
                &input,
                &outdir,
                skip_hydrology,
                extras,
                effective_compress,
                mem_limit_bytes,
            )
        }
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

    println!("🌍 SurtGIS Susceptibility Pipeline");
    println!("═══════════════════════════════════════════");
    println!("  Output: {}", outdir.display());
    println!("  DEM: {}", dem_source);
    if s2_source != "skip" {
        println!("  S2: {} ({})", s2_source, _datetime_str);
    }
    println!();

    // ========== STEP 1: DEM ==========
    println!("📍 STEP 1: Loading DEM...");
    let pb = helpers::spinner("Reading and validating DEM");

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

    // Get DEM info
    let dem = surtgis_core::io::read_geotiff::<f64, _>(&dem_path, None)
        .context("Failed to read DEM for info")?;
    let (rows, cols) = dem.shape();
    let size_mb = (rows * cols * 8) as f64 / 1_000_000.0;

    pb.finish_and_clear();
    println!("  ✅ DEM loaded: {}×{} pixels ({:.1}MB)", cols, rows, size_mb);

    // ========== STEP 2: Terrain All ==========
    println!("\n📍 STEP 2: Terrain factors (17 products)...");
    let pb = helpers::spinner("Computing slope, aspect, curvature, etc.");

    handle_terrain_all(&dem_path, &outdir.join("terrain"), compress, mem_limit_bytes)?;

    pb.finish_and_clear();
    println!("  ✅ Terrain complete");

    // ========== STEP 3: Hydrology All ==========
    println!("\n📍 STEP 3: Hydrology factors (8 products)...");
    let pb = helpers::spinner("Computing flow direction, flow accumulation, streams, etc.");

    handle_hydrology_all(&dem_path, &outdir.join("hydrology"), compress, mem_limit_bytes)?;

    pb.finish_and_clear();
    println!("  ✅ Hydrology complete");

    // ========== STEP 4: S2 Imagery (optional) ==========
    if s2_source != "skip" {
        println!("\n📍 STEP 4: Sentinel-2 imagery (6 bands from STAC)...");
        eprintln!("  Downloading from Planetary Computer:");
        eprintln!("    Collection: {}", s2_source);
        eprintln!("    BBox: {}", _bbox_str);
        eprintln!("    Dates: {}", _datetime_str);
        eprintln!("    Max scenes: {}", _max_scenes);

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
                println!("  ✅ S2 bands downloaded and aligned");

                // ========== STEP 5: Imagery Indices ==========
                println!("\n📍 STEP 5: Computing spectral indices (10 indices)...");
                let pb = helpers::spinner("NDVI, NDWI, MNDWI, NBR, SAVI, EVI, BSI, NDBI, NDMI, NDSI");

                compute_imagery_indices(&s2_bands, &outdir.join("imagery"), compress)?;

                pb.finish_and_clear();
                println!("  ✅ Spectral indices complete");
            }
            Err(e) => {
                eprintln!("\n⚠️  S2 imagery skipped: {}", e);
                eprintln!("  (Continuing with terrain + hydrology only)");
                // Continue without S2 - not a fatal error for MVP
            }
        }
    }

    // ========== Output Summary ==========
    println!("\n═══════════════════════════════════════════");
    println!("✅ PIPELINE COMPLETE");
    println!("═══════════════════════════════════════════");
    println!("\n📂 Output directory: {}\n", outdir.display());

    // Count and list files
    let terrain_count = std::fs::read_dir(outdir.join("terrain"))
        .map(|entries| entries.filter(|e| e.is_ok() && e.as_ref().unwrap().path().extension().map(|ext| ext == "tif").unwrap_or(false)).count())
        .unwrap_or(0);
    let hydrology_count = std::fs::read_dir(outdir.join("hydrology"))
        .map(|entries| entries.filter(|e| e.is_ok() && e.as_ref().unwrap().path().extension().map(|ext| ext == "tif").unwrap_or(false)).count())
        .unwrap_or(0);
    let imagery_count = if s2_source != "skip" && outdir.join("imagery").exists() {
        std::fs::read_dir(outdir.join("imagery"))
            .map(|entries| entries.filter(|e| e.is_ok() && e.as_ref().unwrap().path().extension().map(|ext| ext == "tif").unwrap_or(false)).count())
            .unwrap_or(0)
    } else {
        0
    };

    println!("📊 Generated products:");
    println!("   🏔️ Terrain factors:   {} files", terrain_count);
    println!("     {}/terrain/*.tif", outdir.display());
    println!("   💧 Hydrology factors: {} files", hydrology_count);
    println!("     {}/hydrology/*.tif", outdir.display());
    if imagery_count > 0 {
        println!("   🛰️ Spectral indices:  {} files", imagery_count);
        println!("     {}/imagery/*.tif", outdir.display());
    }
    println!("   📍 Input DEM:");
    println!("     {}/dem.tif", outdir.display());

    let total_files = terrain_count + hydrology_count + imagery_count + 1;
    println!("\n📈 Total: {} files", total_files);
    println!("\n✨ Ready for susceptibility analysis!");

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

/// Fetch a single S2 band from STAC catalog (Planetary Computer)
///
/// Delegates to stac::fetch_s2_band_from_stac() which:
/// - Searches for Sentinel-2 L2A scenes in bbox + datetime
/// - Downloads multiple scenes and composites (cloud-free)
/// - Aligns to DEM grid
fn fetch_s2_band(
    collection: &str,     // "sentinel-2-l2a"
    band: &str,           // "B04", "B08", etc.
    bbox: &str,           // "west,south,east,north"
    datetime: &str,       // "YYYY-MM-DD/YYYY-MM-DD"
    max_scenes: usize,
    align_to: &Path,
) -> Result<Raster<f64>> {
    // Load reference DEM for alignment
    let dem_ref = surtgis_core::io::read_geotiff::<f64, _>(align_to, None)
        .context("Failed to read DEM for grid reference")?;

    // Call the generic STAC handler (collection-agnostic with auto cloud masking)
    super::stac::fetch_stac_band(
        "pc",  // Planetary Computer catalog
        bbox,
        collection,
        &format!("B{:02}", &band[1..]),  // Band identifier (e.g., B04)
        datetime,
        max_scenes,
        Some(&dem_ref),  // Align to DEM grid
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

// ─── Feature stack pipeline ─────────────────────────────────────────────

/// Generate a geomorphometric feature stack from a DEM.
///
/// Produces terrain features (slope, aspect, curvature, etc.), optional
/// hydrology features (flow direction, TWI, HAND, etc.), and optional
/// extra features (valley depth, surface area ratio, wind exposure, etc.).
/// Writes a `features.json` metadata file listing all generated bands.
pub fn handle_features_generate(
    input: &Path,
    outdir: &Path,
    skip_hydrology: bool,
    extras: bool,
    compress: bool,
    mem_limit_bytes: Option<u64>,
) -> Result<()> {
    let total_start = Instant::now();

    // Create output directory structure
    std::fs::create_dir_all(outdir).context("Failed to create output directory")?;
    let terrain_dir = outdir.join("terrain");
    let hydro_dir = outdir.join("hydrology");

    println!("SurtGIS Feature Stack Generator");
    println!("=========================================");
    println!("  Input:  {}", input.display());
    println!("  Output: {}", outdir.display());
    println!(
        "  Mode:   terrain{}{}",
        if !skip_hydrology { " + hydrology" } else { "" },
        if extras { " + extras" } else { "" }
    );
    println!();

    // ========== STEP 1: Terrain All ==========
    println!("STEP 1: Computing terrain features (17 products)...");
    let step_start = Instant::now();

    super::terrain::handle_terrain_all(input, &terrain_dir, compress)
        .context("Failed to compute terrain features")?;

    println!(
        "  Terrain complete ({:.1}s)",
        step_start.elapsed().as_secs_f64()
    );

    // Track all features for metadata
    let mut features: Vec<FeatureEntry> = vec![
        FeatureEntry::new(1, "slope", "terrain/slope.tif", "degrees"),
        FeatureEntry::new(2, "aspect", "terrain/aspect.tif", "degrees"),
        FeatureEntry::new(3, "hillshade", "terrain/hillshade.tif", "unitless"),
        FeatureEntry::new(4, "northness", "terrain/northness.tif", "unitless"),
        FeatureEntry::new(5, "eastness", "terrain/eastness.tif", "unitless"),
        FeatureEntry::new(6, "curvature", "terrain/curvature.tif", "1/m"),
        FeatureEntry::new(7, "tpi", "terrain/tpi.tif", "meters"),
        FeatureEntry::new(8, "tri", "terrain/tri.tif", "meters"),
        FeatureEntry::new(9, "geomorphons", "terrain/geomorphons.tif", "class"),
        FeatureEntry::new(10, "dev", "terrain/dev.tif", "meters"),
        FeatureEntry::new(11, "vrm", "terrain/vrm.tif", "unitless"),
        FeatureEntry::new(12, "convergence", "terrain/convergence.tif", "unitless"),
        FeatureEntry::new(
            13,
            "openness_positive",
            "terrain/openness_positive.tif",
            "degrees",
        ),
        FeatureEntry::new(
            14,
            "openness_negative",
            "terrain/openness_negative.tif",
            "degrees",
        ),
        FeatureEntry::new(15, "svf", "terrain/svf.tif", "unitless"),
        FeatureEntry::new(16, "mrvbf", "terrain/mrvbf.tif", "unitless"),
        FeatureEntry::new(17, "mrrtf", "terrain/mrrtf.tif", "unitless"),
    ];

    // ========== STEP 2: Extras (optional) ==========
    if extras {
        println!("\nSTEP 2: Computing extra terrain features...");
        let step_start = Instant::now();

        let dem = helpers::read_dem(&input.to_path_buf())
            .context("Failed to read DEM for extras")?;

        let vd = valley_depth(&dem).context("Failed to compute valley depth")?;
        helpers::write_result(&vd, &terrain_dir.join("valley_depth.tif"), compress)
            .context("Failed to write valley depth")?;
        println!("  valley_depth.tif");

        let sar = surface_area_ratio(&dem, SarParams::default())
            .context("Failed to compute surface area ratio")?;
        helpers::write_result(&sar, &terrain_dir.join("surface_area_ratio.tif"), compress)
            .context("Failed to write surface area ratio")?;
        println!("  surface_area_ratio.tif");

        let lf = landform_classification(&dem, LandformParams::default())
            .context("Failed to compute landform classification")?;
        helpers::write_result(&lf, &terrain_dir.join("landform.tif"), compress)
            .context("Failed to write landform classification")?;
        println!("  landform.tif");

        let we = wind_exposure(&dem, WindExposureParams::default())
            .context("Failed to compute wind exposure")?;
        helpers::write_result(&we, &terrain_dir.join("wind_exposure.tif"), compress)
            .context("Failed to write wind exposure")?;
        println!("  wind_exposure.tif");

        let az = accumulation_zones(&dem)
            .context("Failed to compute accumulation zones")?;
        helpers::write_result(&az, &terrain_dir.join("accumulation_zones.tif"), compress)
            .context("Failed to write accumulation zones")?;
        println!("  accumulation_zones.tif");

        let next_band = features.len() + 1;
        features.push(FeatureEntry::new(
            next_band,
            "valley_depth",
            "terrain/valley_depth.tif",
            "meters",
        ));
        features.push(FeatureEntry::new(
            next_band + 1,
            "surface_area_ratio",
            "terrain/surface_area_ratio.tif",
            "ratio",
        ));
        features.push(FeatureEntry::new(
            next_band + 2,
            "landform",
            "terrain/landform.tif",
            "class",
        ));
        features.push(FeatureEntry::new(
            next_band + 3,
            "wind_exposure",
            "terrain/wind_exposure.tif",
            "unitless",
        ));
        features.push(FeatureEntry::new(
            next_band + 4,
            "accumulation_zones",
            "terrain/accumulation_zones.tif",
            "class",
        ));

        println!(
            "  Extras complete ({:.1}s)",
            step_start.elapsed().as_secs_f64()
        );
    }

    // ========== STEP 3: Hydrology (optional) ==========
    if !skip_hydrology {
        let step_label = if extras { "STEP 3" } else { "STEP 2" };
        println!("\n{}: Computing hydrology features (8 products)...", step_label);
        let step_start = Instant::now();

        super::hydrology::handle_hydrology_all(input, &hydro_dir, compress, mem_limit_bytes)
            .context("Failed to compute hydrology features")?;

        let next_band = features.len() + 1;
        features.push(FeatureEntry::new(
            next_band,
            "filled",
            "hydrology/filled.tif",
            "meters",
        ));
        features.push(FeatureEntry::new(
            next_band + 1,
            "flow_direction_d8",
            "hydrology/flow_direction_d8.tif",
            "D8 code",
        ));
        features.push(FeatureEntry::new(
            next_band + 2,
            "flow_direction_dinf",
            "hydrology/flow_direction_dinf.tif",
            "radians",
        ));
        features.push(FeatureEntry::new(
            next_band + 3,
            "flow_accumulation",
            "hydrology/flow_accumulation.tif",
            "cells",
        ));
        features.push(FeatureEntry::new(
            next_band + 4,
            "flow_accumulation_mfd",
            "hydrology/flow_accumulation_mfd.tif",
            "cells",
        ));
        features.push(FeatureEntry::new(
            next_band + 5,
            "twi",
            "hydrology/twi.tif",
            "unitless",
        ));
        features.push(FeatureEntry::new(
            next_band + 6,
            "stream_network",
            "hydrology/stream_network.tif",
            "binary",
        ));
        features.push(FeatureEntry::new(
            next_band + 7,
            "hand",
            "hydrology/hand.tif",
            "meters",
        ));

        println!(
            "  Hydrology complete ({:.1}s)",
            step_start.elapsed().as_secs_f64()
        );
    }

    // ========== Write features.json ==========
    write_features_metadata(outdir, &features)?;

    // ========== Summary ==========
    let total_elapsed = total_start.elapsed();
    let terrain_count = count_tif_files(&terrain_dir);
    let hydro_count = if !skip_hydrology {
        count_tif_files(&hydro_dir)
    } else {
        0
    };

    println!();
    println!("=========================================");
    println!("FEATURE STACK COMPLETE");
    println!("=========================================");
    println!();
    println!("Generated products:");
    println!("  Terrain features:  {} files", terrain_count);
    if !skip_hydrology {
        println!("  Hydrology features: {} files", hydro_count);
    }
    println!("  Total features:    {}", features.len());
    println!("  Metadata:          {}/features.json", outdir.display());
    println!();
    println!("Total time: {:.1}s", total_elapsed.as_secs_f64());

    Ok(())
}

/// A single feature entry for the metadata JSON.
#[derive(serde::Serialize)]
struct FeatureEntry {
    band: usize,
    name: String,
    file: String,
    unit: String,
}

impl FeatureEntry {
    fn new(band: usize, name: &str, file: &str, unit: &str) -> Self {
        Self {
            band,
            name: name.to_string(),
            file: file.to_string(),
            unit: unit.to_string(),
        }
    }
}

/// Write `features.json` metadata listing all generated feature bands.
fn write_features_metadata(outdir: &Path, features: &[FeatureEntry]) -> Result<()> {
    let metadata = serde_json::json!({
        "version": "0.4.0",
        "features": features,
        "total_features": features.len(),
    });

    let json_path = outdir.join("features.json");
    let json_str =
        serde_json::to_string_pretty(&metadata).context("Failed to serialize features.json")?;
    std::fs::write(&json_path, json_str).context("Failed to write features.json")?;
    println!("\nMetadata written to: {}", json_path.display());

    Ok(())
}

/// Count .tif files in a directory.
fn count_tif_files(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter(|e| {
                    e.is_ok()
                        && e.as_ref()
                            .unwrap()
                            .path()
                            .extension()
                            .map(|ext| ext == "tif")
                            .unwrap_or(false)
                })
                .count()
        })
        .unwrap_or(0)
}
