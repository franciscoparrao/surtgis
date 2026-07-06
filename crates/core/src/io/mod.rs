//! I/O operations for reading and writing geospatial data

#[cfg(feature = "gdal")]
mod gdal_io;
mod native;
mod options;
pub mod strip_reader;
pub mod strip_writer;

// Single GeoTiffOptions struct shared by both backends (see options.rs) —
// no longer feature-dependent, so the type is stable regardless of which
// backend is compiled in.
pub use options::GeoTiffOptions;

#[cfg(feature = "gdal")]
pub use gdal_io::{read_geotiff, write_geotiff};

#[cfg(not(feature = "gdal"))]
pub use native::{read_geotiff, write_geotiff};

// Multi-band writer — only via the native backend for now
// (1, 3, or 4 bands).
pub use native::write_geotiff_multiband;

// N-band ("stack") writer — native backend, arbitrary band count (not
// limited to 1/3/4), native sample dtype, BlackIsZero photometric
// interpretation (never RGB/RGBA), optional GDAL_METADATA band names.
pub use native::{NativeGraySample, write_geotiff_stack};

// Multi-band reading — de-interleaves every band into its own raster
// (native backend). The single-band `read_geotiff(path, Some(band))` also
// selects a band from a multi-band file.
pub use native::{read_geotiff_bands, read_geotiff_bands_from_buffer};

// Buffer-based I/O (always available, no filesystem dependency)
pub use native::{read_geotiff_from_buffer, write_geotiff_to_buffer};

// Streaming I/O
pub use strip_reader::StripReader;
pub use strip_writer::{StripWriterConfig, write_geotiff_streaming};
