//! Integration tests for GribReader against real GRIB2 data.

#![cfg(feature = "grib")]

use std::path::Path;
use surtgis_cloud::grib_reader::GribReader;
use surtgis_cloud::tile_index::BBox;

#[test]
#[ignore]
fn test_grib_gfs_metadata() {
    let path = Path::new("/tmp/gfs_test.grib2");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let reader = GribReader::open(path).expect("Failed to open GFS GRIB2");
    let meta = reader.metadata();

    println!("Messages: {}", meta.message_count);
    for msg in meta.messages.iter().take(10) {
        println!("  [{}] {}.{}: {}", msg.index, msg.message_index.0, msg.message_index.1, msg.description);
    }

    assert!(meta.message_count > 0);
}

#[test]
#[ignore]
fn test_grib_gfs_read_message() {
    let path = Path::new("/tmp/gfs_test.grib2");
    if !path.exists() { return; }

    let reader = GribReader::open(path).expect("Failed to open");

    // Read first message
    let raster = reader.read_message(0).expect("Failed to read message 0");
    let (rows, cols) = raster.shape();
    let t = raster.transform();
    println!("Message 0: {}x{} cells", rows, cols);
    println!("GeoTransform: origin=({:.4}, {:.4}), pixel=({:.6}, {:.6})",
        t.origin_x, t.origin_y, t.pixel_width, t.pixel_height);
    assert!(rows > 0 && cols > 0);

    let data = raster.data();
    let valid = data.iter().filter(|&&v| !v.is_nan()).count();
    println!("Valid: {} / {}", valid, raster.len());
}

#[test]
#[ignore]
fn test_grib_gfs_read_temperature() {
    let path = Path::new("/tmp/gfs_test.grib2");
    if !path.exists() { return; }

    let reader = GribReader::open(path).expect("Failed to open");

    // Try to find temperature
    let raster = reader.read_by_parameter("temperature")
        .expect("Failed to find temperature");

    let (rows, cols) = raster.shape();
    let t = raster.transform();
    println!("Temperature: {}x{}", rows, cols);
    println!("GeoTransform: origin=({}, {}), pixel=({}, {})",
        t.origin_x, t.origin_y, t.pixel_width, t.pixel_height);
    let (bx0, by0, bx1, by1) = t.bounds(cols, rows);
    println!("Bounds: ({}, {}) to ({}, {})", bx0, by0, bx1, by1);

    // Use a bbox in 0-360 space directly to verify
    let bbox = BBox {
        min_x: 289.0, min_y: -33.5,  // 289 = 360-71
        max_x: 290.0, max_y: -33.0,
    };
    let cropped = reader.read_bbox_by_parameter("temperature", &bbox)
        .expect("Failed to crop temperature (0-360 coords)");
    let (cr, cc) = cropped.shape();
    println!("Cropped: {}x{}", cr, cc);
    assert!(cr > 0 && cc > 0);
}
