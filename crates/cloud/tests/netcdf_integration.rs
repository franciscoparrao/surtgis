//! Integration tests for NetCdfReader against real CMIP6 data.
//!
//! Requires /tmp/cmip6_test.nc to exist (download manually or via test setup).

#![cfg(feature = "netcdf")]

use std::path::Path;

use surtgis_cloud::netcdf_reader::NetCdfReader;
use surtgis_cloud::tile_index::BBox;

#[cfg(feature = "zarr")]
use surtgis_cloud::{TimeReduction, TimeSelector};

#[cfg(not(feature = "zarr"))]
use surtgis_cloud::netcdf_reader::{TimeReduction, TimeSelector};

/// Test: open a real CMIP6 precipitation file and read metadata.
#[test]
#[ignore]
fn test_netcdf_cmip6_metadata() {
    let path = Path::new("/tmp/cmip6_test.nc");
    if !path.exists() {
        eprintln!("Skipping: {} not found", path.display());
        return;
    }

    let reader = NetCdfReader::open(path, "pr").expect("Failed to open CMIP6 NetCDF");
    let meta = reader.metadata();

    println!("Variable: {}", meta.variable);
    println!("Shape: {:?}", meta.shape);
    println!("Dims: {:?}", meta.dimension_names);
    println!("Time range: {:?}", meta.time_range);
    println!("Nodata: {:?}", meta.nodata);
    println!("Available: {:?}", meta.available_variables);

    assert_eq!(meta.variable, "pr");
    assert_eq!(meta.dimension_names.len(), 3);
    assert!(meta.shape[0] > 0); // time
    assert!(meta.shape[1] > 0); // lat
    assert!(meta.shape[2] > 0); // lon
    assert!(meta.time_range.is_some());
}

/// Test: read a bbox from CMIP6, first time step.
#[test]
#[ignore]
fn test_netcdf_cmip6_read_bbox() {
    let path = Path::new("/tmp/cmip6_test.nc");
    if !path.exists() {
        return;
    }

    let reader = NetCdfReader::open(path, "pr").expect("Failed to open");

    let bbox = BBox {
        min_x: -71.0,
        min_y: -33.5,
        max_x: -70.5,
        max_y: -33.0,
    };

    let time = TimeReduction::Single(TimeSelector::First);
    let raster = reader
        .read_bbox(&bbox, &time)
        .expect("Failed to read CMIP6 bbox");

    let (rows, cols) = raster.shape();
    println!("CMIP6 raster: {}x{} ({} cells)", rows, cols, raster.len());
    assert!(rows > 0 && cols > 0);

    let data = raster.data();
    let valid = data.iter().filter(|&&v| !v.is_nan()).count();
    println!("Valid cells: {} / {}", valid, raster.len());
    assert!(valid > 0, "All values are NaN");

    // Print sample values (precipitation flux in kg m-2 s-1)
    println!("Sample values (kg m-2 s-1):");
    for r in 0..rows.min(3) {
        for c in 0..cols.min(3) {
            print!("{:.6e}  ", data[[r, c]]);
        }
        println!();
    }
}

/// Test: time aggregation (monthly mean) on CMIP6 data.
#[test]
#[ignore]
fn test_netcdf_cmip6_monthly_mean() {
    let path = Path::new("/tmp/cmip6_test.nc");
    if !path.exists() {
        return;
    }

    let reader = NetCdfReader::open(path, "pr").expect("Failed to open");

    let bbox = BBox {
        min_x: -71.0,
        min_y: -33.5,
        max_x: -70.5,
        max_y: -33.0,
    };

    // Aggregate first 30 days
    let meta = reader.metadata();
    if let Some((t0, _)) = meta.time_range {
        let start = t0;
        let end = t0 + chrono::TimeDelta::days(29);

        #[cfg(feature = "zarr")]
        let time = TimeReduction::Aggregate {
            start,
            end,
            method: surtgis_cloud::AggMethod::Mean,
        };

        #[cfg(not(feature = "zarr"))]
        let time = TimeReduction::Aggregate {
            start,
            end,
            method: surtgis_cloud::netcdf_reader::AggMethod::Mean,
        };

        let raster = reader
            .read_bbox(&bbox, &time)
            .expect("Failed to aggregate CMIP6");

        let (rows, cols) = raster.shape();
        println!("CMIP6 monthly mean: {}x{}", rows, cols);
        assert!(rows > 0 && cols > 0);
    }
}
