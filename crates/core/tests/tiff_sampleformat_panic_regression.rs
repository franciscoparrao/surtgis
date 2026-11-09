//! Regression: a malformed TIFF must not panic the decoder.
//!
//! The 335-byte fixture declares `SampleFormat` (tag 339) with count 0.
//! `tiff` 0.10.3 indexed the resulting empty sample-format vector
//! (`image.rs:153`, `sample_format[0]`) and panicked with "index out of
//! bounds: the len is 0 but the index is 0" — a denial of service on the
//! GeoTIFF path, which decodes untrusted bytes (COGs from arbitrary
//! third-party STAC catalog hrefs). Found by the nightly fuzzer.
//!
//! Two layers now prevent it: the `tiff` 0.11 bump turns the bad tag into
//! a decode error, and `decode_raw` wraps the decoder in `catch_unwind` so
//! any *other* decoder panic on hostile input also becomes an error rather
//! than unwinding out of the library. Either way: this must return `Err`,
//! never panic.

use surtgis_core::io::read_geotiff_from_buffer;

#[test]
fn malformed_sampleformat_returns_err_not_panic() {
    let data = include_bytes!("fixtures/malformed_sampleformat_count0.tif");
    let result = read_geotiff_from_buffer::<f64>(data, None);
    assert!(
        result.is_err(),
        "malformed SampleFormat=count0 TIFF should decode to Err, got {:?}",
        result.map(|r| r.shape())
    );
}
