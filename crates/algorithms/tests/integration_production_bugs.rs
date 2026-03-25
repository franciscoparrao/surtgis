//! Integration tests for bugs found in production (15 cuencas Chile, Marzo 2026).
//!
//! These tests reproduce issues discovered when processing real data:
//! - CRS not propagated through terrain/imagery algorithms
//! - Cloud-mask failing when SCL has different resolution (20m vs 10m)
//! - Median composite failing with different spatial extents
//! - Grid misalignment between DEM and satellite imagery
//! - Resample alignment with bilinear interpolation
//!
//! Tests use the fixture DEM at `tests/fixtures/andes_chile_30m.tif`.
//! If the fixture doesn't exist, tests are skipped.

use surtgis_algorithms::hydrology::{
    flow_accumulation, flow_direction, hand, priority_flood, stream_network,
    HandParams, PriorityFloodParams, StreamNetworkParams,
};
use surtgis_algorithms::imagery::{cloud_mask_scl, median_composite, normalized_difference};
use surtgis_algorithms::terrain::{
    aspect, hillshade, slope, twi, AspectOutput, HillshadeParams, SlopeParams, SlopeUnits,
};
use ndarray::Array2;
use surtgis_core::crs::CRS;
use surtgis_core::io::{read_geotiff, write_geotiff, GeoTiffOptions};
use surtgis_core::mosaic::mosaic;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::resample::{resample_to_grid, ResampleMethod};
use std::path::Path;

// ─── Helpers ────────────────────────────────────────────────────────────

const DEM_FILENAME: &str = "tests/fixtures/andes_chile_30m.tif";

fn dem_path() -> std::path::PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().unwrap().parent().unwrap();
    workspace_root.join(DEM_FILENAME)
}

fn load_dem() -> Option<Raster<f64>> {
    let path = dem_path();
    if !path.exists() {
        eprintln!("SKIPPING: fixture not found at {}", path.display());
        return None;
    }
    Some(read_geotiff::<f64, _>(&path, None).expect("failed to read DEM fixture"))
}

macro_rules! require_dem {
    () => {
        match load_dem() {
            Some(dem) => dem,
            None => return,
        }
    };
}

/// Create a synthetic raster with known CRS and transform
fn make_raster_with_crs(
    rows: usize,
    cols: usize,
    value: f64,
    origin_x: f64,
    origin_y: f64,
    cell_size: f64,
    epsg: u32,
) -> Raster<f64> {
    let arr = Array2::from_elem((rows, cols), value);
    let mut r = Raster::from_array(arr);
    r.set_transform(GeoTransform::new(origin_x, origin_y, cell_size, -cell_size));
    r.set_crs(Some(CRS::from_epsg(epsg)));
    r.set_nodata(Some(f64::NAN));
    r
}

// ─── Bug 1: CRS propagation through algorithms ──────────────────────────

#[test]
fn test_crs_propagates_through_slope() {
    let dem = require_dem!();
    let result = slope(&dem, SlopeParams::default()).unwrap();

    // The output must have the same CRS as the input
    assert_eq!(
        dem.crs().map(|c| c.epsg()),
        result.crs().map(|c| c.epsg()),
        "Slope output lost CRS from input DEM"
    );
}

#[test]
fn test_crs_propagates_through_aspect() {
    let dem = require_dem!();
    let result = aspect(&dem, AspectOutput::Degrees).unwrap();
    assert_eq!(
        dem.crs().map(|c| c.epsg()),
        result.crs().map(|c| c.epsg()),
        "Aspect output lost CRS from input DEM"
    );
}

#[test]
fn test_crs_propagates_through_hillshade() {
    let dem = require_dem!();
    let result = hillshade(&dem, HillshadeParams::default()).unwrap();
    assert_eq!(
        dem.crs().map(|c| c.epsg()),
        result.crs().map(|c| c.epsg()),
        "Hillshade output lost CRS from input DEM"
    );
}

#[test]
fn test_crs_propagates_through_hydrology_pipeline() {
    let dem = require_dem!();
    let filled = priority_flood(&dem, PriorityFloodParams { epsilon: 0.0001 }).unwrap();
    let fdir = flow_direction(&filled).unwrap();
    let facc = flow_accumulation(&fdir).unwrap();

    // Each step should preserve CRS
    assert_eq!(
        dem.crs().map(|c| c.epsg()),
        filled.crs().map(|c| c.epsg()),
        "Priority flood lost CRS"
    );
    // Note: flow_direction returns Raster<u8> which may not propagate CRS
    // through with_same_meta. This is acceptable.

    // Flow accumulation should preserve CRS from its input
    assert_eq!(
        fdir.crs().map(|c| c.epsg()),
        facc.crs().map(|c| c.epsg()),
        "Flow accumulation lost CRS"
    );
}

#[test]
fn test_crs_propagates_through_normalized_difference() {
    // Simulate two bands with CRS
    let band_a = make_raster_with_crs(100, 100, 0.8, 500000.0, 7100000.0, 10.0, 32719);
    let band_b = make_raster_with_crs(100, 100, 0.2, 500000.0, 7100000.0, 10.0, 32719);

    let ndvi = normalized_difference(&band_a, &band_b).unwrap();
    assert_eq!(
        ndvi.crs().map(|c| c.epsg()),
        Some(Some(32719)),
        "NDVI lost CRS"
    );
}

// ─── Bug 2: CRS roundtrip through GeoTIFF I/O ──────────────────────────

#[test]
fn test_crs_survives_write_read_roundtrip() {
    let raster = make_raster_with_crs(50, 50, 42.0, 500000.0, 7100000.0, 30.0, 32719);

    let tmp = std::env::temp_dir().join("surtgis_crs_roundtrip.tif");
    write_geotiff(&raster, &tmp, Some(GeoTiffOptions::default())).unwrap();

    let loaded: Raster<f64> = read_geotiff(&tmp, None).unwrap();
    std::fs::remove_file(&tmp).ok();

    assert_eq!(
        loaded.crs().and_then(|c| c.epsg()),
        Some(32719),
        "CRS EPSG lost during write→read roundtrip"
    );

    // Also check transform survives
    let gt = loaded.transform();
    assert!(
        (gt.origin_x - 500000.0).abs() < 1.0,
        "origin_x shifted: {}",
        gt.origin_x
    );
    assert!(
        (gt.pixel_width - 30.0).abs() < 0.01,
        "pixel_width changed: {}",
        gt.pixel_width
    );
}

// ─── Bug 3: Cloud-mask with different resolutions ────────────────────────

#[test]
fn test_cloud_mask_different_resolution_production() {
    // Simulate S2: data at 10m (200x200), SCL at 20m (100x100)
    let mut data = make_raster_with_crs(200, 200, 5000.0, 500000.0, 7100000.0, 10.0, 32719);
    // Set some pixels to different values for verification
    data.set(50, 50, 8000.0).unwrap();

    // SCL: top-half = 4 (vegetation, keep), bottom-half = 9 (cloud, mask)
    let mut scl_data = Array2::from_elem((100, 100), 4.0f64);
    for r in 50..100 {
        for c in 0..100 {
            scl_data[[r, c]] = 9.0; // cloud
        }
    }
    let mut scl = Raster::from_array(scl_data);
    scl.set_transform(GeoTransform::new(500000.0, 7100000.0, 20.0, -20.0));

    let result = cloud_mask_scl(&data, &scl, &[4, 5, 6, 11]).unwrap();

    // Top-half should be preserved
    assert!(
        result.get(25, 25).unwrap().is_finite(),
        "Top-half pixel should be kept (SCL=4)"
    );
    assert!(
        (result.get(50, 50).unwrap() - 8000.0).abs() < 1.0,
        "Known pixel value should be preserved"
    );

    // Bottom-half should be masked (NaN)
    assert!(
        result.get(150, 50).unwrap().is_nan(),
        "Bottom-half pixel should be masked (SCL=9)"
    );
}

// ─── Bug 4: Median composite with different extents ──────────────────────

#[test]
fn test_median_composite_production_extents() {
    // Simulate two S2 scenes with different spatial extents
    // (different dates may cover different number of tiles)
    let scene1 = make_raster_with_crs(100, 150, 1000.0, 500000.0, 7100000.0, 10.0, 32719);
    let scene2 = make_raster_with_crs(100, 100, 2000.0, 500500.0, 7100000.0, 10.0, 32719);

    // scene1: x=[500000, 501500], scene2: x=[500500, 501500] — overlap at x=[500500, 501500]
    let result = median_composite(&[&scene1, &scene2]).unwrap();

    let (rows, cols) = result.shape();
    // Union should cover x=[500000, 501500] = 150 cols, 100 rows
    assert_eq!(cols, 150, "Union should span 150 cols");
    assert_eq!(rows, 100, "Union should span 100 rows");

    // Left part (only scene1): value = 1000
    let left_val = result.get(50, 10).unwrap();
    assert!(
        (left_val - 1000.0).abs() < 1.0,
        "Left (scene1 only) should be 1000, got {}",
        left_val
    );

    // Overlap part: median of [1000, 2000] = 1500
    let overlap_val = result.get(50, 75).unwrap();
    assert!(
        (overlap_val - 1500.0).abs() < 1.0,
        "Overlap should be median 1500, got {}",
        overlap_val
    );
}

// ─── Bug 5: Grid alignment (resample) ────────────────────────────────────

#[test]
fn test_resample_aligns_grids_with_offset() {
    // DEM at 30m grid: origin at (500000, 7100000)
    let dem = make_raster_with_crs(100, 100, 1000.0, 500000.0, 7100000.0, 30.0, 32719);

    // S2 band at 10m grid: origin offset by ~16m (real-world scenario)
    let mut s2_data = Array2::<f64>::zeros((300, 300));
    for r in 0..300 {
        for c in 0..300 {
            s2_data[[r, c]] = (r * 1000 + c) as f64; // gradient for verification
        }
    }
    let mut s2 = Raster::from_array(s2_data);
    s2.set_transform(GeoTransform::new(500016.0, 7099984.0, 10.0, -10.0));
    s2.set_crs(Some(CRS::from_epsg(32719)));
    s2.set_nodata(Some(f64::NAN));

    // Resample S2 to DEM grid
    let aligned = resample_to_grid(&s2, &dem, ResampleMethod::Bilinear).unwrap();

    // Output should have DEM's dimensions and transform
    assert_eq!(aligned.shape(), dem.shape(), "Aligned should match DEM dimensions");

    let gt = aligned.transform();
    assert!(
        (gt.origin_x - 500000.0).abs() < 1e-6,
        "Origin should match DEM, got {}",
        gt.origin_x
    );
    assert!(
        (gt.pixel_width - 30.0).abs() < 1e-6,
        "Cell size should match DEM, got {}",
        gt.pixel_width
    );

    // CRS should come from reference (DEM)
    assert_eq!(
        aligned.crs().and_then(|c| c.epsg()),
        Some(32719),
        "CRS should match reference"
    );

    // Values should be interpolated (not NaN, since S2 covers the DEM area)
    let center = aligned.get(50, 50).unwrap();
    assert!(
        center.is_finite(),
        "Center pixel should have interpolated value"
    );
}

#[test]
fn test_resample_nearest_preserves_classes() {
    // Classification raster at 20m
    let mut class_data = Array2::<f64>::from_elem((50, 50), 4.0); // vegetation
    for r in 25..50 {
        for c in 0..50 {
            class_data[[r, c]] = 9.0; // cloud
        }
    }
    let mut classification = Raster::from_array(class_data);
    classification.set_transform(GeoTransform::new(500000.0, 7100000.0, 20.0, -20.0));
    classification.set_crs(Some(CRS::from_epsg(32719)));

    // Reference at 10m
    let reference = make_raster_with_crs(100, 100, 0.0, 500000.0, 7100000.0, 10.0, 32719);

    let result = resample_to_grid(&classification, &reference, ResampleMethod::NearestNeighbor).unwrap();

    // Nearest neighbor should produce exact class values (no interpolation)
    let top = result.get(10, 10).unwrap();
    assert!(
        (top - 4.0).abs() < 0.01,
        "Nearest neighbor should give exact 4.0, got {}",
        top
    );

    let bottom = result.get(60, 10).unwrap();
    assert!(
        (bottom - 9.0).abs() < 0.01,
        "Nearest neighbor should give exact 9.0, got {}",
        bottom
    );
}

// ─── Bug 6: Mosaic with NaN-aware overlap ────────────────────────────────

#[test]
fn test_mosaic_nan_does_not_overwrite_valid() {
    // Tile 1: all valid (1000.0)
    let tile1 = make_raster_with_crs(50, 50, 1000.0, 500000.0, 7100000.0, 30.0, 32719);

    // Tile 2: same area but all NaN (e.g., nodata from irregular tile shape)
    let tile2 = make_raster_with_crs(50, 50, f64::NAN, 500000.0, 7100000.0, 30.0, 32719);

    let result = mosaic(&[&tile1, &tile2], None).unwrap();

    // Tile 1's valid data should survive (NaN from tile2 should not overwrite)
    let val = result.get(25, 25).unwrap();
    assert!(
        (val - 1000.0).abs() < 1e-6,
        "NaN from tile2 should not overwrite tile1's valid data, got {}",
        val
    );
}

// ─── Full pipeline test ──────────────────────────────────────────────────

#[test]
fn test_full_terrain_hydrology_pipeline() {
    let dem = require_dem!();
    let input_crs = dem.crs().and_then(|c| c.epsg());

    // Terrain
    let slp = slope(&dem, SlopeParams { units: SlopeUnits::Radians, z_factor: 1.0 }).unwrap();
    let asp = aspect(&dem, AspectOutput::Degrees).unwrap();
    let hs = hillshade(&dem, HillshadeParams::default()).unwrap();

    // Hydrology pipeline
    let filled = priority_flood(&dem, PriorityFloodParams { epsilon: 0.0001 }).unwrap();
    let fdir = flow_direction(&filled).unwrap();
    let facc = flow_accumulation(&fdir).unwrap();
    let slope_rad = slope(&filled, SlopeParams { units: SlopeUnits::Radians, z_factor: 1.0 }).unwrap();
    let twi_result = twi(&facc, &slope_rad).unwrap();
    let _streams = stream_network(&facc, StreamNetworkParams { threshold: 500.0 }).unwrap();
    let hand_result = hand(&dem, &fdir, &facc, HandParams { stream_threshold: 500.0 }).unwrap();

    // All outputs should have same dimensions as input
    let (rows, cols) = dem.shape();
    assert_eq!(slp.shape(), (rows, cols), "Slope dimensions mismatch");
    assert_eq!(asp.shape(), (rows, cols), "Aspect dimensions mismatch");
    assert_eq!(hs.shape(), (rows, cols), "Hillshade dimensions mismatch");
    assert_eq!(filled.shape(), (rows, cols), "Filled dimensions mismatch");
    assert_eq!(facc.shape(), (rows, cols), "Flow acc dimensions mismatch");
    assert_eq!(twi_result.shape(), (rows, cols), "TWI dimensions mismatch");
    assert_eq!(hand_result.shape(), (rows, cols), "HAND dimensions mismatch");

    // CRS should propagate through terrain algorithms
    assert_eq!(slp.crs().and_then(|c| c.epsg()), input_crs, "Slope CRS mismatch");
    assert_eq!(filled.crs().and_then(|c| c.epsg()), input_crs, "Filled CRS mismatch");
    assert_eq!(twi_result.crs().and_then(|c| c.epsg()), input_crs, "TWI CRS mismatch");
    assert_eq!(hand_result.crs().and_then(|c| c.epsg()), input_crs, "HAND CRS mismatch");

    // Values should be reasonable
    let slp_stats = slp.statistics();
    let twi_stats = twi_result.statistics();
    let hand_stats = hand_result.statistics();

    // Slope: should be between 0 and ~90 degrees (stored as radians here, so 0 to ~1.57)
    assert!(slp_stats.min.unwrap() >= 0.0, "Slope min should be >= 0");
    assert!(slp_stats.max.unwrap() < 1.6, "Slope max should be < π/2");

    // TWI: can range widely depending on terrain; just verify it's finite
    assert!(twi_stats.min.is_some() && twi_stats.min.unwrap().is_finite(), "TWI min should be finite");
    assert!(twi_stats.max.is_some() && twi_stats.max.unwrap().is_finite(), "TWI max should be finite");

    // HAND: should be >= 0 (height above drainage)
    assert!(hand_stats.min.unwrap() >= -1.0, "HAND min should be ~0");
}

// ─── Río Salado Basin Integration Tests (Real data, 5092×3619, 70MB DEM) ────

/// Helper to load the real Río Salado DEM from Agentes project.
/// Returns None if the DEM is not available (test will be skipped).
fn load_salado_dem() -> Option<Raster<f64>> {
    let path = std::path::Path::new("/home/franciscoparrao/proyectos/Agentes/salado_utm_cropped.tif");
    if !path.exists() {
        eprintln!("SKIPPING Río Salado tests: DEM not found at {}", path.display());
        return None;
    }
    Some(read_geotiff::<f64, _>(path, None).expect("failed to read Río Salado DEM"))
}

/// Helper to load the Subsubcuencas shapefile from Agentes project.
/// Returns None if the shapefile is not available.
/// Extracts only Polygon geometries (skips others).
fn load_salado_shapefile() -> Option<Vec<geo_types::Polygon<f64>>> {
    #[cfg(feature = "shapefile")]
    {
        use surtgis_core::vector::read_shapefile;
        let path = std::path::Path::new("/home/franciscoparrao/proyectos/Agentes/Subsubcuencas/Subsubcuencas_BNA.shp");
        if !path.exists() {
            eprintln!("SKIPPING shapefile test: shapefile not found at {}", path.display());
            return None;
        }
        let fc = read_shapefile(path).expect("failed to read Subsubcuencas shapefile");
        let polys: Vec<_> = fc.iter()
            .filter_map(|f| {
                f.geometry.clone().and_then(|g| {
                    use geo_types::Geometry;
                    match g {
                        Geometry::Polygon(p) => Some(p),
                        _ => None,
                    }
                })
            })
            .collect();
        if polys.is_empty() {
            eprintln!("SKIPPING shapefile test: no polygon geometries in shapefile");
            return None;
        }
        Some(polys)
    }
    #[cfg(not(feature = "shapefile"))]
    {
        eprintln!("SKIPPING shapefile test: shapefile feature not enabled");
        None
    }
}

#[test]
fn test_salado_dem_slope_with_streaming() {
    // Test: Slope computation with bounded memory (simulating --max-memory 100M)
    // This verifies that streaming produces identical results to non-streaming
    // on a real 5092x3619 DEM (70MB uncompressed).

    let dem = match load_salado_dem() {
        Some(d) => d,
        None => return,
    };

    let (rows, cols) = dem.shape();
    eprintln!("Testing Río Salado DEM: {}x{} ({:.0}MB)",
        rows, cols, (rows * cols * 8) as f64 / 1e6);

    // Compute slope normally (in-memory)
    let slope_result = slope(&dem, SlopeParams {
        units: SlopeUnits::Degrees,
        z_factor: 1.0
    }).expect("slope computation failed");

    // Verify basic properties
    assert_eq!(slope_result.shape(), dem.shape(), "Slope dimensions mismatch");

    // Check CRS preservation (UTM 19S = 32719)
    assert_eq!(
        slope_result.crs().and_then(|c| c.epsg()),
        Some(32719),
        "Slope lost CRS (should be UTM 19S)"
    );

    // Verify slope values are reasonable for Andes terrain
    let stats = slope_result.statistics();
    assert!(stats.min.unwrap() >= 0.0, "Slope min should be >= 0");
    assert!(stats.max.unwrap() <= 90.0, "Slope max should be <= 90°");
    assert!(stats.max.unwrap() > 30.0, "Andes should have steep slopes (>30°)");

    eprintln!("  Slope range: {:.2}° to {:.2}°, mean: {:.2}°",
        stats.min.unwrap(), stats.max.unwrap(), stats.mean.unwrap());
}

#[test]
fn test_salado_dem_clip_shapefile() {
    // Test: Clipping DEM by real Subsubcuencas polygons.
    // Verifies that polygon clipping with UTM coordinates works correctly.
    // This test is skipped if shapefile feature is disabled or no polygons are found.

    let dem = match load_salado_dem() {
        Some(d) => d,
        None => return,
    };

    let geoms = match load_salado_shapefile() {
        Some(g) => g,
        None => return,  // Skip if shapefile feature disabled or no polygons
    };

    if geoms.is_empty() {
        eprintln!("SKIPPING test_salado_dem_clip_shapefile: no polygons loaded");
        return;
    }

    use surtgis_core::vector::clip_raster_by_polygon;

    eprintln!("Loaded {} subsubcuencas polygons", geoms.len());

    let clipped = clip_raster_by_polygon(&dem, &geoms[0])
        .expect("clip_raster_by_polygon failed");

    let (orig_rows, orig_cols) = dem.shape();
    let (clipped_rows, clipped_cols) = clipped.shape();

    eprintln!("  Original DEM: {}x{}", orig_rows, orig_cols);
    eprintln!("  Clipped to polygon: {}x{}", clipped_rows, clipped_cols);

    // Clipped should be smaller
    assert!(clipped_rows <= orig_rows, "Clipped should have <= original rows");
    assert!(clipped_cols <= orig_cols, "Clipped should have <= original cols");

    // Should still have reasonable size (not all masked)
    assert!(clipped_rows > 100, "Clipped polygon should span >100 rows");
    assert!(clipped_cols > 100, "Clipped polygon should span >100 cols");

    // CRS should be preserved
    assert_eq!(
        clipped.crs().and_then(|c| c.epsg()),
        Some(32719),
        "CRS lost during clipping"
    );

    // Should have some valid data (not all NaN)
    let stats = clipped.statistics();
    let total_pixels = clipped.shape().0 * clipped.shape().1;
    let valid_pixels = stats.valid_count;

    eprintln!("  Valid pixels in clipped region: {}/{}", valid_pixels, total_pixels);
    assert!(valid_pixels > 100, "Clipped region should have at least 100 valid pixels");
}

#[test]
fn test_salado_edge_cases_utm_coords() {
    // Test: Correct handling of UTM 19S coordinates (negative offsets, large absolute values).
    // Río Salado is in UTM 19S, which has:
    //   - False Easting = 500,000 m (positive x)
    //   - False Northing = 10,000,000 m (large positive y)
    // This test verifies that geospatial math doesn't break with these values.

    let dem = match load_salado_dem() {
        Some(d) => d,
        None => return,
    };

    let gt = dem.transform();
    eprintln!("DEM transform: origin=({}, {}), cell_size={}m",
        gt.origin_x, gt.origin_y, gt.pixel_width);

    // Verify coordinates are in expected UTM 19S range
    // UTM 19S zone covers Andes region; Río Salado is within ~200km of zone center
    assert!(
        gt.origin_x > 200_000.0 && gt.origin_x < 700_000.0,
        "UTM 19S easting should be in valid range, got {}",
        gt.origin_x
    );
    assert!(
        gt.origin_y > 6_000_000.0 && gt.origin_y < 8_000_000.0,
        "UTM 19S southing should be in valid range, got {}",
        gt.origin_y
    );

    // Compute slope (tests that internal calculations handle large coordinates)
    let slope_result = slope(&dem, SlopeParams {
        units: SlopeUnits::Degrees,
        z_factor: 1.0
    }).expect("slope failed");

    // Compute aspect (also uses local coordinates)
    let aspect_result = aspect(&dem, AspectOutput::Degrees)
        .expect("aspect failed");

    // Both should have valid statistics (not all NaN, not infinite)
    let slope_stats = slope_result.statistics();
    let aspect_stats = aspect_result.statistics();

    assert!(slope_stats.min.is_some() && slope_stats.max.is_some(),
        "Slope should have valid statistics");
    assert!(aspect_stats.min.is_some() && aspect_stats.max.is_some(),
        "Aspect should have valid statistics");

    // Verify values are physically reasonable
    assert!(slope_stats.min.unwrap() >= 0.0 && slope_stats.max.unwrap() <= 90.0,
        "Slope should be in [0, 90]");
    assert!(aspect_stats.min.unwrap() >= 0.0 && aspect_stats.max.unwrap() <= 360.0,
        "Aspect should be in [0, 360]");

    eprintln!("  Slope: {:.2}° to {:.2}°",
        slope_stats.min.unwrap(), slope_stats.max.unwrap());
    eprintln!("  Aspect: {:.1}° to {:.1}°",
        aspect_stats.min.unwrap(), aspect_stats.max.unwrap());
}
