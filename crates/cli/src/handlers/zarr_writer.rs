//! Minimal Zarr v2 writer for `extract-patches` output.
//!
//! Why hand-rolled: the `zarrs` crate is a heavyweight dep and ties us
//! to a feature gate; for our single use case (write a fixed-shape f32
//! tensor as chunked Zarr v2, one chunk per training sample) the spec
//! footprint is small enough that a hand-written writer is simpler than
//! integrating a general-purpose library.
//!
//! Output layout for shape `[N, C, T, H, W]` with chunk `[1, C, T, H, W]`:
//!
//! ```
//! patches.zarr/
//!   .zarray         # array metadata (shape, dtype, chunks, fill)
//!   .zattrs         # user attributes (band names, profile, timestamps)
//!   0.0.0.0.0       # chunk for chip 0
//!   1.0.0.0.0       # chunk for chip 1
//!   ...
//! ```
//!
//! Each chunk file is the raw little-endian f32 bytes of one chip.
//! No compression in v1 — users who want it can re-roll with the
//! `zarr` Python package, which can read this output directly.
//!
//! Spec reference: https://zarr.readthedocs.io/en/stable/spec/v2.html

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Initialise a Zarr v2 array directory: writes `.zarray` and `.zattrs`.
/// Caller writes chunks separately via `write_chunk`.
///
/// `shape` and `chunks` must have the same rank.
pub fn init_zarr_v2_array(
    dir: &Path,
    shape: &[usize],
    chunks: &[usize],
    dtype: &str,
    fill_value: serde_json::Value,
    attrs: &serde_json::Value,
) -> Result<()> {
    assert_eq!(shape.len(), chunks.len(),
        "shape and chunks must have the same rank");

    fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create {}", dir.display()))?;

    let zarray = serde_json::json!({
        "zarr_format": 2,
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": serde_json::Value::Null,
        "fill_value": fill_value,
        "filters": serde_json::Value::Null,
        "order": "C",
    });
    fs::write(dir.join(".zarray"), serde_json::to_string_pretty(&zarray)?)
        .with_context(|| format!("Failed to write {}/.zarray", dir.display()))?;

    fs::write(dir.join(".zattrs"), serde_json::to_string_pretty(attrs)?)
        .with_context(|| format!("Failed to write {}/.zattrs", dir.display()))?;

    Ok(())
}

/// Write one chunk to a Zarr v2 array.
///
/// `chunk_coord` is the chunk index along each axis (e.g. `[5, 0, 0, 0, 0]`
/// for chip 5 when chunk shape is `[1, C, T, H, W]`). The filename is the
/// dot-joined coordinate ("5.0.0.0.0"), per the Zarr v2 spec.
///
/// `bytes` is the raw little-endian payload of the chunk in C-order.
pub fn write_chunk(dir: &Path, chunk_coord: &[usize], bytes: &[u8]) -> Result<()> {
    let name: Vec<String> = chunk_coord.iter().map(|c| c.to_string()).collect();
    let path = dir.join(name.join("."));
    let mut f = File::create(&path)
        .with_context(|| format!("Failed to create chunk {}", path.display()))?;
    f.write_all(bytes)
        .with_context(|| format!("Failed to write chunk {}", path.display()))?;
    Ok(())
}
