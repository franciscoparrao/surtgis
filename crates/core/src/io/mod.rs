//! I/O operations for reading and writing geospatial data

#[cfg(feature = "gdal")]
mod gdal_io;
mod native;
pub mod strip_reader;
pub mod strip_writer;

#[cfg(feature = "gdal")]
pub use gdal_io::{GeoTiffOptions, read_geotiff, write_geotiff};

#[cfg(not(feature = "gdal"))]
pub use native::{GeoTiffOptions, read_geotiff, write_geotiff};

// Multi-band writer — only via the native backend for now
// (1, 3, or 4 bands).
pub use native::write_geotiff_multiband;

// Native GeoTiffOptions under an unambiguous name. When the `gdal` feature is
// on, `io::GeoTiffOptions` resolves to the GDAL variant, but the native-only
// `write_geotiff_multiband` needs the native options type — which is otherwise
// unnameable from outside the crate (the `native` module is private).
pub use native::GeoTiffOptions as NativeGeoTiffOptions;

// Multi-band reading — de-interleaves every band into its own raster
// (native backend). The single-band `read_geotiff(path, Some(band))` also
// selects a band from a multi-band file.
pub use native::{read_geotiff_bands, read_geotiff_bands_from_buffer};

// Buffer-based I/O (always available, no filesystem dependency)
pub use native::{read_geotiff_from_buffer, write_geotiff_to_buffer};

// Streaming I/O
pub use strip_reader::StripReader;
pub use strip_writer::{StripWriterConfig, write_geotiff_streaming};
