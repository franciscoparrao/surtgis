//! Regression: a planar (PlanarConfiguration=2, INTERLEAVE=BAND) multi-band
//! TIFF must be rejected with a clear error, never panic.
//!
//! The `tiff` crate's `read_image()` returns only the first plane for a
//! planar TIFF, while `SamplesPerPixel` stays > 1. The dtype-preserving
//! `read_geotiff_any_from_buffer` then indexed `buf[i*spp + b]` out of
//! bounds and panicked (audit R3, agent D reproduced `surtgis info` crashing
//! at native.rs on a perfectly valid GDAL-written planar file). Both the
//! any-path and the fixed-`T` path now return an actionable error instead.
//!
//! Fixture: 8×8, 3 bands, int16, written by GDAL/rasterio with
//! `interleave='band'`.

use surtgis_core::io::{read_geotiff_any_from_buffer, read_geotiff_from_buffer};

const PLANAR: &[u8] = include_bytes!("fixtures/planar_interleave_band.tif");

#[test]
fn planar_tiff_any_path_errors_instead_of_panicking() {
    let result = read_geotiff_any_from_buffer(PLANAR, None);
    let err = result.expect_err("planar multi-band TIFF must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("planar") || msg.contains("INTERLEAVE"),
        "error should name the planar limitation, got: {msg}"
    );
}

#[test]
fn planar_tiff_fixed_type_path_errors_cleanly() {
    // The fixed-`T` path previously returned a misleading "Invalid raster
    // dimensions"; it must now surface the same planar error.
    let result = read_geotiff_from_buffer::<f64>(PLANAR, None);
    let err = result.expect_err("planar multi-band TIFF must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("planar") || msg.contains("INTERLEAVE"),
        "error should name the planar limitation, got: {msg}"
    );
}
