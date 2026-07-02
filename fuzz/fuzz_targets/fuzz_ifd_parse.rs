#![no_main]
//! Fuzz target for the hand-rolled IFD/COG header parser in `surtgis-cloud`.
//!
//! The COG reader consumes bytes fetched over HTTP Range requests from arbitrary
//! hrefs in third-party STAC catalogs, so `parse_header` / `parse_ifd` see fully
//! attacker-controlled input. We exercise:
//!   1. `parse_header` on the raw buffer.
//!   2. `parse_ifd` on the bytes at the declared first-IFD offset (clamped),
//!      using the byte order the header reports.
//!   3. `parse_ifd` on the whole buffer under BOTH byte orders (belt-and-braces,
//!      independent of a valid header) to stress the entry-count arithmetic.
//!
//! All functions return `Result`; we only assert the absence of panics/overflow.

use libfuzzer_sys::fuzz_target;
use surtgis_cloud::ifd::{parse_header, parse_ifd, TiffByteOrder};

fuzz_target!(|data: &[u8]| {
    // 1 + 2: valid-header-driven path.
    if let Ok(header) = parse_header(data) {
        let off = header.first_ifd_offset as usize;
        if off <= data.len() {
            let _ = parse_ifd(header.byte_order, &data[off..]);
        }
    }

    // 3: parse the buffer directly as an IFD under both byte orders. This
    // reaches the `entry_count * 12` sizing arithmetic without needing the
    // 8-byte header to validate first.
    let _ = parse_ifd(TiffByteOrder::LittleEndian, data);
    let _ = parse_ifd(TiffByteOrder::BigEndian, data);
});
