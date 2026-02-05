//! Integration tests for the STAC client.
//!
//! Tests marked `#[ignore]` require network access to real STAC catalogs.
//! Run with: `cargo test -p surtgis-cloud --features native -- --ignored stac`

use surtgis_cloud::stac_client::{StacCatalog, StacClient, StacClientOptions};
use surtgis_cloud::stac_models::StacSearchParams;

/// Search Earth Search for Sentinel-2 data over Madrid.
#[tokio::test]
#[ignore]
async fn stac_earth_search_sentinel2() {
    let client = StacClient::new(StacCatalog::EarthSearch, StacClientOptions::default())
        .expect("failed to create client");

    let params = StacSearchParams::new()
        .bbox(-3.75, 40.38, -3.65, 40.45)
        .datetime("2024-06-01/2024-06-30")
        .collections(&["sentinel-2-l2a"])
        .limit(5);

    let results = client.search(&params).await.expect("search failed");

    println!("Found {} items", results.len());
    assert!(!results.is_empty(), "should find at least one item");

    for item in &results.features {
        println!(
            "  {} dt={} cc={:?}",
            item.id,
            item.properties.datetime.as_deref().unwrap_or("-"),
            item.properties.eo_cloud_cover
        );
        assert!(!item.assets.is_empty(), "item should have assets");
        assert!(item.collection.is_some(), "item should have collection");
    }
}

/// Search Planetary Computer for Sentinel-2 data.
#[tokio::test]
#[ignore]
async fn stac_planetary_computer_sentinel2() {
    let client =
        StacClient::new(StacCatalog::PlanetaryComputer, StacClientOptions::default())
            .expect("failed to create client");

    let params = StacSearchParams::new()
        .bbox(-3.75, 40.38, -3.65, 40.45)
        .datetime("2024-06-01/2024-06-30")
        .collections(&["sentinel-2-l2a"])
        .limit(3);

    let results = client.search(&params).await.expect("search failed");

    println!("Found {} items (matched: {:?})", results.len(), results.number_matched);
    assert!(!results.is_empty(), "should find at least one item");

    // Verify assets exist
    let item = &results.features[0];
    assert!(
        item.asset("B04").is_some() || item.asset("red").is_some(),
        "should have a red band asset"
    );
}

/// Test Planetary Computer URL signing via /sign endpoint.
#[tokio::test]
#[ignore]
async fn stac_pc_url_signing() {
    let client =
        StacClient::new(StacCatalog::PlanetaryComputer, StacClientOptions::default())
            .expect("failed to create client");

    let href = "https://elevationeuwest.blob.core.windows.net/copernicus-dem/COP30_hh/Copernicus_DSM_COG_10_N40_00_W004_00_DEM.tif";

    let signed = client
        .sign_asset_href(href, "cop-dem-glo-30")
        .await
        .expect("signing failed");

    println!("Signed URL: {}", &signed[..signed.len().min(150)]);
    assert!(signed.contains("sig=") || signed.contains("se="), "should contain SAS token params");
    assert!(signed.starts_with(href), "should start with original href");
}

/// Test paginated search.
#[tokio::test]
#[ignore]
async fn stac_paginated_search() {
    let mut opts = StacClientOptions::default();
    opts.max_items = 15;

    let client = StacClient::new(StacCatalog::EarthSearch, opts)
        .expect("failed to create client");

    let params = StacSearchParams::new()
        .bbox(-3.75, 40.38, -3.65, 40.45)
        .datetime("2024-01-01/2024-12-31")
        .collections(&["sentinel-2-l2a"])
        .limit(5); // 5 per page

    let items = client.search_all(&params).await.expect("search_all failed");

    println!("Fetched {} items across pages", items.len());
    assert!(items.len() > 5, "should have fetched more than one page");
    assert!(items.len() <= 15, "should respect max_items");
}
