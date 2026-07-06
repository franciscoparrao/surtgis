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
/// - `compression` — honoured by both. The native backend recognises
///   `"NONE"`/`"DEFLATE"` (case-insensitive) and returns an explicit
///   `Error::Other` for anything else (e.g. `"LZW"`, `"ZSTD"` — valid for
///   the GDAL backend, but the native `tiff`-crate writer has no encoder
///   for them). It used to silently fall back to DEFLATE for any
///   unrecognised value; that silent fallback was removed because it
///   masked the fact that the requested codec was never applied. The
///   GDAL backend forwards the string verbatim as the `COMPRESS`
///   creation option.
/// - `tile_size`, `cog` — GDAL-only; the native backend always writes
///   strip-organised output and ignores these fields.
/// - `bigtiff` — honoured by both. The native backend uses
///   `TiffEncoder::new_big` when set, in `write_geotiff` (single-band)
///   and `write_geotiff_stack` (native-dtype N-band). The older
///   `write_geotiff_multiband` (1/3/4-band Float32 RGB/RGBA) predates
///   this and still ignores it.
#[derive(Debug, Clone)]
pub struct GeoTiffOptions {
    /// Compression type, e.g. `"DEFLATE"`, `"LZW"`, `"ZSTD"`, `"NONE"`.
    pub compression: String,
    /// Tile size for tiled TIFFs (0 for strips). GDAL backend only.
    pub tile_size: usize,
    /// Create a Cloud-Optimized GeoTIFF. GDAL backend only (reserved;
    /// not yet consumed by `write_geotiff`).
    pub cog: bool,
    /// BigTIFF for files > 4GB. Honoured by the GDAL backend and by the
    /// native backend's `write_geotiff` / `write_geotiff_stack` (not by
    /// the older `write_geotiff_multiband`; see the struct docs above).
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
