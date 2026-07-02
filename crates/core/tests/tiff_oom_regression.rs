//! Regression: a hostile TIFF must not trigger an unbounded allocation.
//!
//! The 338-byte fixture declares huge image dimensions / tag counts; with
//! `Limits::unlimited()` the decoder tried to allocate ~32 GB in one shot
//! (OOM/DoS, found by fuzzing). The reader now bounds decode buffers
//! relative to the source size, so this must return quickly without OOM —
//! Ok or Err, but never a multi-GB allocation.

use surtgis_core::io::read_geotiff_from_buffer;

#[test]
fn hostile_tiff_does_not_oom() {
    let data = include_bytes!("fixtures/oom_ifd_count.tif");
    // The point is that this returns (either variant) instead of aborting
    // the process with an allocation failure.
    let _ = read_geotiff_from_buffer::<f64>(data, None);
}
