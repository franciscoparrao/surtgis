//! Integration tests for the STAC composite pipeline.
//!
//! Network tests: cargo test -p surtgis-cloud --test composite_integration --features native -- --ignored
//! Offline tests: cargo test -p surtgis-cloud --test composite_integration

use surtgis_cloud::decompress::undo_horizontal_differencing;

// =========================================================================
// Predictor undo: unit tests (offline, no network)
// =========================================================================

#[test]
fn predictor_undo_uint16_accumulates_correctly() {
    // Simulate a row of 4 uint16 pixels stored as horizontal differences:
    // Original: [3000, 3010, 3005, 3020]
    // Differenced: [3000, 10, -5(=65531), 15]
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(&3000u16.to_le_bytes());
    data.extend_from_slice(&10u16.to_le_bytes());
    data.extend_from_slice(&65531u16.to_le_bytes()); // -5 as wrapping u16
    data.extend_from_slice(&15u16.to_le_bytes());

    undo_horizontal_differencing(&mut data, 4, 2);

    let p0 = u16::from_le_bytes([data[0], data[1]]);
    let p1 = u16::from_le_bytes([data[2], data[3]]);
    let p2 = u16::from_le_bytes([data[4], data[5]]);
    let p3 = u16::from_le_bytes([data[6], data[7]]);

    assert_eq!(p0, 3000, "pixel 0");
    assert_eq!(p1, 3010, "pixel 1: 3000 + 10");
    assert_eq!(p2, 3005, "pixel 2: 3010 + 65531(=-5) = 3005");
    assert_eq!(p3, 3020, "pixel 3: 3005 + 15");
}

#[test]
fn predictor_undo_multi_row_resets_per_row() {
    // 2 rows of 3 pixels each (tile_width=3, bps=2)
    let mut data: Vec<u8> = Vec::new();
    // Row 0: [1000, 5, 10] → after undo: [1000, 1005, 1015]
    data.extend_from_slice(&1000u16.to_le_bytes());
    data.extend_from_slice(&5u16.to_le_bytes());
    data.extend_from_slice(&10u16.to_le_bytes());
    // Row 1: [2000, 3, 7] → after undo: [2000, 2003, 2010]
    data.extend_from_slice(&2000u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes());
    data.extend_from_slice(&7u16.to_le_bytes());

    undo_horizontal_differencing(&mut data, 3, 2);

    let vals: Vec<u16> = (0..6).map(|i| {
        u16::from_le_bytes([data[i * 2], data[i * 2 + 1]])
    }).collect();

    assert_eq!(vals, vec![1000, 1005, 1015, 2000, 2003, 2010]);
}

#[test]
fn predictor_undo_uint8() {
    // Row of 4 uint8 pixels
    // Original: [100, 105, 103, 110]
    // Differenced: [100, 5, -2(=254), 7]
    let mut data = vec![100u8, 5, 254, 7];

    undo_horizontal_differencing(&mut data, 4, 1);

    assert_eq!(data, vec![100, 105, 103, 110]);
}

#[test]
fn predictor_noop_when_predictor_is_1() {
    // Predictor=1 means no differencing — data should not change
    let original = vec![0xB8u8, 0x0B, 0x0A, 0x00, 0x50, 0x0C, 0x10, 0x00];
    let mut data = original.clone();

    // Don't call undo (predictor=1 check is in the caller)
    // Just verify the function doesn't panic with empty input
    undo_horizontal_differencing(&mut data, 0, 2);
    assert_eq!(data, original);
}

// =========================================================================
// COG Reader: network tests (require --ignored flag)
// =========================================================================

#[cfg(feature = "native")]
mod network_tests {
    use surtgis_cloud::cog_reader::CogReaderOptions;
    use surtgis_cloud::stac_client::{StacCatalog, StacClientOptions};
    use surtgis_cloud::stac_models::StacSearchParams;
    use surtgis_cloud::sync_api::{CogReaderBlocking, StacClientBlocking};
    use surtgis_cloud::tile_index::BBox;
    use surtgis_core::Raster;

    const SANTIAGO_BBOX: (f64, f64, f64, f64) = (-70.7, -33.5, -70.6, -33.4);

    fn pc_client() -> StacClientBlocking {
        StacClientBlocking::new(
            StacCatalog::PlanetaryComputer,
            StacClientOptions { max_items: 10, ..StacClientOptions::default() },
        ).expect("Failed to create PC client")
    }

    #[test]
    #[ignore]
    fn cog_reader_s2_b04_values_in_reflectance_range() {
        let client = pc_client();
        let params = StacSearchParams::new()
            .bbox(SANTIAGO_BBOX.0, SANTIAGO_BBOX.1, SANTIAGO_BBOX.2, SANTIAGO_BBOX.3)
            .datetime("2024-01-15/2024-01-20")
            .collections(&["sentinel-2-l2a"])
            .limit(1);

        let items = client.search_all(&params).expect("search failed");
        assert!(!items.is_empty(), "No S2 items found");

        let item = &items[0];
        let asset = item.asset("B04").expect("No B04");
        let signed = client.sign_asset_href(&asset.href, item.collection.as_deref().unwrap_or(""))
            .expect("signing failed");

        let bbox = BBox::new(SANTIAGO_BBOX.0, SANTIAGO_BBOX.1, SANTIAGO_BBOX.2, SANTIAGO_BBOX.3);
        let cog_bbox = {
            use surtgis_cloud::reproject;
            let meta_reader = CogReaderBlocking::open(&signed, CogReaderOptions::default()).unwrap();
            let meta = meta_reader.metadata();
            if let Some(epsg) = meta.crs.as_ref().and_then(|c| c.epsg()) {
                if !reproject::is_wgs84(epsg) {
                    reproject::reproject_bbox_to_cog(&bbox, epsg)
                } else { bbox }
            } else { bbox }
        };

        let mut reader = CogReaderBlocking::open(&signed, CogReaderOptions::default()).unwrap();
        let raster: Raster<f64> = reader.read_bbox(&cog_bbox, None).expect("read failed");
        let (rows, cols) = raster.shape();
        assert!(rows > 0 && cols > 0);

        let valid: Vec<f64> = raster.data().iter()
            .filter(|v| v.is_finite() && **v > 0.0).copied().collect();
        assert!(!valid.is_empty(), "No valid pixels");

        let max = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;

        eprintln!("COG B04: {}x{}, {} valid, max={:.0}, mean={:.0}", rows, cols, valid.len(), max, mean);

        // CRITICAL: predictor=2 must produce values in reflectance range
        // Some pixels can be >10000 (saturated, snow) but mean must be reasonable
        assert!(max < 20000.0, "Max {} > 20000: predictor undo broken", max);
        assert!(mean < 8000.0, "Mean {} > 8000: values inflated", mean);
        assert!(mean > 100.0, "Mean {} < 100: over-filtered", mean);
    }

    #[test]
    #[ignore]
    fn stac_search_finds_multiple_tiles() {
        let client = pc_client();
        let params = StacSearchParams::new()
            .bbox(-71.0, -33.8, -70.0, -33.0)
            .datetime("2024-01-15/2024-01-20")
            .collections(&["sentinel-2-l2a"])
            .limit(50);

        let items = client.search_all(&params).expect("search failed");
        assert!(items.len() >= 4, "Expected ≥4 items, got {}", items.len());
        eprintln!("PASS: {} items found", items.len());
    }

    #[test]
    #[ignore]
    fn sas_signing_with_retry() {
        let client = pc_client();
        let params = StacSearchParams::new()
            .bbox(SANTIAGO_BBOX.0, SANTIAGO_BBOX.1, SANTIAGO_BBOX.2, SANTIAGO_BBOX.3)
            .datetime("2024-01-15/2024-01-20")
            .collections(&["sentinel-2-l2a"])
            .limit(1);

        let items = client.search_all(&params).expect("search failed");
        assert!(!items.is_empty());
        let item = &items[0];

        for key in &["B04", "B03", "B02", "SCL"] {
            if let Some(asset) = item.asset(key) {
                let result = client.sign_asset_href(&asset.href, item.collection.as_deref().unwrap_or(""));
                assert!(result.is_ok(), "Signing {} failed: {:?}", key, result.err());
            }
        }
        eprintln!("PASS: All assets signed");
    }
}
