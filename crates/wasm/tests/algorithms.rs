//! Exercises the `#[wasm_bindgen]` surface end-to-end through its actual
//! JS-facing signature (raw GeoTIFF `&[u8]` in, `Vec<u8>` out), not just
//! the underlying Rust algorithm. `wasm_bindgen_test` compiles these as
//! plain `#[test]` on non-wasm targets (`cargo test -p surtgis-wasm`) and
//! as browser/Node tests under `wasm-pack test --node` — same source,
//! both harnesses.

use surtgis_core::{GeoTransform, Raster};
use surtgis_wasm::*;
use wasm_bindgen_test::*;

/// Small synthetic DEM (east-west ramp, no nodata) encoded as in-memory
/// GeoTIFF bytes — the input shape every terrain/hydrology binding here
/// expects. Generated fresh per test, no external fixture needed.
fn synth_dem_bytes(rows: usize, cols: usize) -> Vec<u8> {
    let mut dem: Raster<f64> = Raster::new(rows, cols);
    dem.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));
    for r in 0..rows {
        for c in 0..cols {
            dem.set(r, c, (r as f64) + (c as f64) * 2.0).unwrap();
        }
    }
    surtgis_core::io::write_geotiff_to_buffer(&dem, None).unwrap()
}

/// Uniform single-band raster (stand-in for a spectral band), for the
/// imagery bindings which take two co-registered bands.
fn synth_band_bytes(rows: usize, cols: usize, value: f64) -> Vec<u8> {
    let mut band: Raster<f64> = Raster::new(rows, cols);
    band.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));
    for r in 0..rows {
        for c in 0..cols {
            band.set(r, c, value + (r * cols + c) as f64 * 0.01)
                .unwrap();
        }
    }
    surtgis_core::io::write_geotiff_to_buffer(&band, None).unwrap()
}

#[wasm_bindgen_test(unsupported = test)]
fn slope_roundtrips_through_geotiff_bytes() {
    let dem = synth_dem_bytes(15, 15);
    let out = slope(&dem, "degrees").expect("slope should succeed on a valid DEM");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

#[wasm_bindgen_test(unsupported = test)]
fn aspect_degrees_roundtrips() {
    let dem = synth_dem_bytes(15, 15);
    let out = aspect_degrees(&dem).expect("aspect should succeed");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

#[wasm_bindgen_test(unsupported = test)]
fn hillshade_compute_roundtrips() {
    let dem = synth_dem_bytes(15, 15);
    let out =
        hillshade_compute(&dem, 315.0, 45.0).expect("hillshade should succeed with valid angles");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

#[wasm_bindgen_test(unsupported = test)]
fn curvature_compute_roundtrips() {
    let dem = synth_dem_bytes(15, 15);
    let out = curvature_compute(&dem, "general").expect("curvature should succeed");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

#[wasm_bindgen_test(unsupported = test)]
fn tpi_compute_roundtrips() {
    let dem = synth_dem_bytes(15, 15);
    let out = tpi_compute(&dem, 2).expect("tpi should succeed");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

#[wasm_bindgen_test(unsupported = test)]
fn fill_depressions_then_flow_direction_chains() {
    let dem = synth_dem_bytes(15, 15);
    let filled = fill_depressions(&dem).expect("fill_depressions should succeed");
    let fdir =
        flow_direction_d8(&filled).expect("flow_direction_d8 should succeed on a filled DEM");
    let result: Raster<u8> = surtgis_core::io::read_geotiff_from_buffer(&fdir, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

#[wasm_bindgen_test(unsupported = test)]
fn ndvi_from_two_bands_roundtrips() {
    let nir = synth_band_bytes(10, 10, 0.6);
    let red = synth_band_bytes(10, 10, 0.2);
    let out = ndvi(&nir, &red).expect("ndvi should succeed on two co-registered bands");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (10, 10));
    // NDVI = (nir - red) / (nir + red); with nir > red throughout, every
    // cell should be strictly positive.
    for r in 0..10 {
        for c in 0..10 {
            let v = result.get(r, c).unwrap();
            assert!(v > 0.0, "NDVI at ({r},{c}) should be positive, got {v}");
        }
    }
}

#[wasm_bindgen_test(unsupported = test)]
fn morph_erode_roundtrips() {
    let dem = synth_dem_bytes(15, 15);
    let out = morph_erode(&dem, 1).expect("morph_erode should succeed");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

#[wasm_bindgen_test(unsupported = test)]
fn focal_mean_roundtrips() {
    let dem = synth_dem_bytes(15, 15);
    let out = focal_mean(&dem, 2).expect("focal_mean should succeed");
    let result: Raster<f64> = surtgis_core::io::read_geotiff_from_buffer(&out, None).unwrap();
    assert_eq!(result.shape(), (15, 15));
}

/// Every binding funnels read errors through the same `dem_op!`/manual
/// `map_err(...JsValue::from_str)` path — garbage bytes must surface as a
/// `JsValue` error, not panic across the WASM boundary (a panic there
/// aborts the whole module in a real browser, not just this call).
///
/// wasm32-only (no `unsupported = test`): `JsValue::from_str` is an FFI
/// shim into the JS engine and is `unimplemented!()` on native targets,
/// so this specific test can only run under `wasm-pack test`, not plain
/// `cargo test`.
#[wasm_bindgen_test]
fn slope_on_invalid_bytes_returns_js_error_not_panic() {
    let garbage = vec![0u8, 1, 2, 3, 4];
    let result = slope(&garbage, "degrees");
    assert!(
        result.is_err(),
        "invalid GeoTIFF bytes should return Err, not panic"
    );
}
