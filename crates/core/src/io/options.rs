//! Shared GeoTIFF write options for the native and GDAL I/O backends.
//!
//! Previously `native.rs` and `gdal_io.rs` each defined their own
//! `GeoTiffOptions` struct (different fields, different `Default`), and
//! `io::GeoTiffOptions` resolved to whichever one matched the active
//! `gdal` feature. That meant the same name silently named a different
//! type depending on compile-time features — confusing for callers and a
//! trap for conditional-compilation bugs. This module defines a single
//! struct used by both backends; each backend reads only the fields that
//! apply to it and ignores the rest.

/// Options for writing GeoTIFF files.
///
/// Shared by the native (`tiff`-crate based) and GDAL-based write paths.
/// Not every field applies to every backend:
///
/// - `compression` — honoured by both. The native backend only
///   distinguishes `"NONE"` (case-insensitive) from anything else, and
///   always uses DEFLATE for the latter (the `tiff` crate's own
///   compression support). The GDAL backend forwards the string verbatim
///   as the `COMPRESS` creation option, so it also accepts values the
///   native backend can't act on (e.g. `"LZW"`, `"ZSTD"`) — in that case
///   the native backend falls back to its DEFLATE-or-nothing behaviour.
/// - `tile_size`, `cog`, `bigtiff` — GDAL-only. The native backend always
///   writes strip-organised, non-BigTIFF output and silently ignores
///   these fields.
#[derive(Debug, Clone)]
pub struct GeoTiffOptions {
    /// Compression type, e.g. `"DEFLATE"`, `"LZW"`, `"ZSTD"`, `"NONE"`.
    pub compression: String,
    /// Tile size for tiled TIFFs (0 for strips). GDAL backend only.
    pub tile_size: usize,
    /// Create a Cloud-Optimized GeoTIFF. GDAL backend only (reserved;
    /// not yet consumed by `write_geotiff`).
    pub cog: bool,
    /// BigTIFF for files > 4GB. GDAL backend only.
    pub bigtiff: bool,
}

impl Default for GeoTiffOptions {
    fn default() -> Self {
        Self {
            // Compress by default: smaller files with no real downside,
            // and both backends support DEFLATE.
            compression: "DEFLATE".to_string(),
            tile_size: 256,
            cog: false,
            bigtiff: false,
        }
    }
}
