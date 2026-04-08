//! Integration tests for ZarrReader against real Planetary Computer data.
//!
//! These tests hit the network and are slow. Run with:
//! ```
//! cargo test -p surtgis-cloud --features "native,zarr" -- --ignored zarr
//! ```

#![cfg(all(feature = "zarr", feature = "native"))]

use surtgis_cloud::blocking::{StacClientBlocking, ZarrReaderBlocking};
use surtgis_cloud::{
    BBox, StacCatalog, StacClientOptions, StacSearchParams,
    TimeReduction, TimeSelector, ZarrReaderOptions,
};

/// Helper: search PC for a collection, find a specific zarr asset, sign it.
/// Returns (https_store_url, sas_token, variable_name).
fn get_signed_zarr_url(collection: &str, target_asset: Option<&str>) -> (String, Option<String>, String) {
    let client = StacClientBlocking::new(
        StacCatalog::PlanetaryComputer,
        StacClientOptions::default(),
    )
    .expect("Failed to create STAC client");

    let params = StacSearchParams::new()
        .collections(&[collection])
        .limit(1);

    let results = client.search(&params).expect("STAC search failed");
    let item = results.features.first().expect("No items found");

    println!("Item: {} [{}]", item.id, item.collection.as_deref().unwrap_or("-"));
    println!("Assets: {}", item.assets.keys().cloned().collect::<Vec<_>>().join(", "));

    // Find the asset
    let (asset_key, stac_asset) = if let Some(key) = target_asset {
        (key.to_string(), item.asset(key).unwrap_or_else(|| panic!("Asset '{}' not found", key)))
    } else {
        let zarr_keys = ["zarr-https", "zarr-abfs", "zarr"];
        zarr_keys
            .iter()
            .find_map(|k| item.asset(k).map(|a| (k.to_string(), a)))
            .or_else(|| item.first_zarr_asset().map(|(k, a)| (k.clone(), a)))
            .expect("No Zarr asset found")
    };

    println!("Asset '{}': {}", asset_key, stac_asset.href);

    // Get collection auth info (token + storage account + container)
    let auth = client
        .get_collection_zarr_auth(collection)
        .expect("Failed to get collection auth");

    let (sas, store_url) = if let Some((token, account, _container)) = auth {
        // Convert abfs:// to https:// with the correct storage account
        let url = surtgis_cloud::zarr_auth::abfs_to_https_with_account(
            &stac_asset.href,
            Some(&account),
        );
        (Some(token), url)
    } else {
        (None, stac_asset.href.clone())
    };

    let variable = asset_key.clone();

    println!("Store URL: {}", store_url);
    println!("Has SAS: {}", sas.is_some());

    (store_url, sas, variable)
}

/// ERA5-PDS: open store, read metadata.
#[test]
#[ignore]
fn test_zarr_era5_metadata() {
    // ERA5 on PC: each asset is a separate Zarr store for one variable
    let (store_url, sas_token, variable) =
        get_signed_zarr_url("era5-pds", Some("precipitation_amount_1hour_Accumulation"));

    let opts = ZarrReaderOptions { sas_token };
    let reader = ZarrReaderBlocking::open(&store_url, &variable, opts)
        .expect("Failed to open ERA5 Zarr store");

    let meta = reader.metadata();
    println!("Variable: {}", meta.variable);
    println!("Shape: {:?}", meta.shape);
    println!("Dims: {:?}", meta.dimension_names);
    println!("Time range: {:?}", meta.time_range);
    println!("Available variables: {:?}", meta.available_variables);

    assert_eq!(meta.dimension_names.len(), 3);
    assert!(meta.shape[0] > 0);
    assert!(meta.crs.is_some());
}

/// ERA5-PDS: read a small bbox, single time step.
#[test]
#[ignore]
fn test_zarr_era5_read_bbox() {
    let (store_url, sas_token, variable) =
        get_signed_zarr_url("era5-pds", Some("precipitation_amount_1hour_Accumulation"));
    let opts = ZarrReaderOptions { sas_token };

    let reader = ZarrReaderBlocking::open(&store_url, &variable, opts)
        .expect("Failed to open ERA5");

    let meta = reader.metadata();
    println!("Variable: {}, shape: {:?}", meta.variable, meta.shape);

    // Small bbox in central Chile
    let bbox = BBox {
        min_x: -71.0,
        min_y: -33.5,
        max_x: -70.5,
        max_y: -33.0,
    };

    let time = TimeReduction::Single(TimeSelector::First);
    let raster = reader
        .read_bbox(&bbox, &time)
        .expect("Failed to read ERA5 bbox");

    let (rows, cols) = raster.shape();
    println!("ERA5 raster: {}x{} ({} cells)", rows, cols, raster.len());
    assert!(rows > 0);
    assert!(cols > 0);

    // Check we got actual data
    let data = raster.data();
    let valid_count = data.iter().filter(|&&v| !v.is_nan()).count();
    println!("Valid cells: {} / {}", valid_count, raster.len());
    assert!(valid_count > 0, "All values are NaN");
}

/// ERA5: read a different variable (temperature).
#[test]
#[ignore]
fn test_zarr_era5_temperature() {
    let (store_url, sas_token, variable) =
        get_signed_zarr_url("era5-pds", Some("air_temperature_at_2_metres_1hour_Maximum"));
    let opts = ZarrReaderOptions { sas_token };

    let reader = ZarrReaderBlocking::open(&store_url, &variable, opts)
        .expect("Failed to open ERA5 temperature");

    let meta = reader.metadata();
    println!("TerraClimate shape: {:?}", meta.shape);
    println!("TerraClimate dims: {:?}", meta.dimension_names);
    println!("TerraClimate variables: {:?}", meta.available_variables);

    let bbox = BBox {
        min_x: -71.0,
        min_y: -33.5,
        max_x: -70.5,
        max_y: -33.0,
    };

    let time = TimeReduction::Single(TimeSelector::First);
    let raster = reader
        .read_bbox(&bbox, &time)
        .expect("Failed to read TerraClimate bbox");

    let (rows, cols) = raster.shape();
    println!("TerraClimate raster: {}x{}", rows, cols);
    assert!(rows > 0 && cols > 0);

    let data = raster.data();
    let valid_count = data.iter().filter(|&&v| !v.is_nan()).count();
    println!("Valid cells: {} / {}", valid_count, raster.len());
    assert!(valid_count > 0, "All values are NaN");
}

/// ERA5: time aggregation (mean of multiple steps).
#[test]
#[ignore]
fn test_zarr_era5_time_aggregation() {
    use chrono::NaiveDate;
    use surtgis_cloud::AggMethod;

    let (store_url, sas_token, variable) =
        get_signed_zarr_url("era5-pds", Some("precipitation_amount_1hour_Accumulation"));
    let opts = ZarrReaderOptions { sas_token };

    let reader = ZarrReaderBlocking::open(&store_url, &variable, opts)
        .expect("Failed to open ERA5");

    let bbox = BBox {
        min_x: -71.0,
        min_y: -33.5,
        max_x: -70.5,
        max_y: -33.0,
    };

    // Aggregate first 24 hours
    let start = NaiveDate::from_ymd_opt(2020, 1, 1)
        .unwrap()
        .and_hms_opt(0, 0, 0)
        .unwrap()
        .and_utc();
    let end = NaiveDate::from_ymd_opt(2020, 1, 1)
        .unwrap()
        .and_hms_opt(23, 0, 0)
        .unwrap()
        .and_utc();

    let time = TimeReduction::Aggregate {
        start,
        end,
        method: AggMethod::Mean,
    };

    let raster = reader
        .read_bbox(&bbox, &time)
        .expect("Failed to aggregate ERA5 time");

    let (rows, cols) = raster.shape();
    println!("ERA5 daily mean: {}x{}", rows, cols);
    assert!(rows > 0 && cols > 0);
}
