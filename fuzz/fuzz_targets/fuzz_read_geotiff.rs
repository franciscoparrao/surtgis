#![no_main]
//! Fuzz target for the native GeoTIFF reader.
//!
//! Feeds arbitrary bytes into `surtgis_core::io::read_geotiff_from_buffer::<f64>`,
//! the entry point used to decode GeoTIFF buffers (including ones sourced from
//! untrusted third-party STAC catalogs). We only care that no input can cause a
//! panic, arithmetic overflow, or unbounded allocation — the decoded result is
//! discarded.
//!
//! Note: the underlying `tiff` decoder is configured with `Limits::unlimited()`
//! in the core crate, so a TIFF declaring enormous dimensions can legitimately
//! request a very large allocation. libFuzzer's `-rss_limit_mb` guards against
//! that; a genuine OOM here is a signal worth triaging (see the S1 report).

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // band = None → read the first/only band. Result intentionally ignored.
    let _ = surtgis_core::io::read_geotiff_from_buffer::<f64>(data, None);
});
