#![no_main]
//! Fuzz target for tile decompression and the TIFF predictors in `surtgis-cloud`.
//!
//! Three code paths, all fed attacker-controlled tile bytes:
//!   1. `decompress_tile` with a compression code derived from the input. The
//!      real DEFLATE/LZW decoders (weezl, flate2) run on arbitrary bytes.
//!   2. `undo_horizontal_differencing` (Predictor=2).
//!   3. `undo_floating_point_predictor_multi` (Predictor=3) — the family that
//!      caused the v0.6.21 mis-decode; the target of the overflow/OOB hunt.
//!
//! The predictor functions index into the buffer using `tile_width *
//! bytes_per_sample * samples_per_pixel` arithmetic. To make that arithmetic the
//! thing under test (rather than a trivially-huge allocation), the small header
//! we carve from the input keeps dims modest; `overflow-checks = true` in the
//! fuzz profile turns any silent `usize`/`u32` overflow into an abort.

use libfuzzer_sys::fuzz_target;
use surtgis_cloud::decompress::{
    decompress_tile, undo_floating_point_predictor_multi, undo_horizontal_differencing,
};

// Compression codes the codec dispatch understands, plus a couple of unknowns.
const CODES: [u16; 6] = [
    1,     // NONE
    5,     // LZW
    8,     // DEFLATE
    32946, // ADOBE_DEFLATE
    7,     // unknown → UnsupportedCompression
    0,     // unknown → UnsupportedCompression
];

fuzz_target!(|data: &[u8]| {
    // Need a few header bytes to derive parameters; rest is the tile payload.
    if data.len() < 4 {
        return;
    }
    let (hdr, payload) = data.split_at(4);

    // --- 1. decompress_tile -------------------------------------------------
    let code = CODES[(hdr[0] as usize) % CODES.len()];
    // expected_raw_size only sizes a Vec::with_capacity hint; cap it hard so a
    // crafted value can't request a giant allocation on its own.
    let expected = (hdr[1] as usize) * 64;
    let _ = decompress_tile(payload, code, expected);

    // --- 2 + 3. predictors --------------------------------------------------
    // bytes_per_sample ∈ {1,2,4,8}; tile_width small (1..=64); spp ∈ {1,3,4}.
    let bps = [1usize, 2, 4, 8][(hdr[2] & 0b11) as usize];
    let tile_width = ((hdr[3] as usize) % 64) + 1;
    let spp = [1usize, 3, 4][(hdr[2] >> 2) as usize % 3];

    // Horizontal differencing works in-place on a mutable copy of the payload.
    let mut buf2 = payload.to_vec();
    undo_horizontal_differencing(&mut buf2, tile_width, bps);

    // Floating-point predictor (multi-sample). Only meaningful for bps > 1, but
    // the function guards internally; call it regardless to fuzz those guards.
    let mut buf3 = payload.to_vec();
    undo_floating_point_predictor_multi(&mut buf3, tile_width, bps, spp);
});
