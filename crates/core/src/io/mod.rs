//! I/O operations for reading and writing geospatial data

#[cfg(feature = "gdal")]
mod gdal_io;
mod native;
pub mod strip_reader;
pub mod strip_writer;

#[cfg(feature = "gdal")]
pub use gdal_io::{read_geotiff, write_geotiff, GeoTiffOptions};

#[cfg(not(feature = "gdal"))]
pub use native::{read_geotiff, write_geotiff, GeoTiffOptions};

// Buffer-based I/O (always available, no filesystem dependency)
pub use native::{read_geotiff_from_buffer, write_geotiff_to_buffer};

// Streaming I/O
pub use strip_reader::StripReader;
pub use strip_writer::{write_geotiff_streaming, StripWriterConfig};
