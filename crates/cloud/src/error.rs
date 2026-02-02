//! Error types for the cloud COG reader.

use thiserror::Error;

/// Errors produced by the cloud COG reader.
#[derive(Error, Debug)]
pub enum CloudError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("server does not support Range requests for {url}")]
    RangeNotSupported { url: String },

    #[error("invalid TIFF: {reason}")]
    InvalidTiff { reason: String },

    #[error("unsupported compression: {0}")]
    UnsupportedCompression(u16),

    #[error("unsupported data type: bits_per_sample={bps}, sample_format={sf}")]
    UnsupportedDataType { bps: u16, sf: u16 },

    #[error("unsupported planar configuration: {0} (only chunky=1 supported)")]
    UnsupportedPlanarConfig(u16),

    #[error("authentication error: {0}")]
    Auth(String),

    #[error("network error: {0}")]
    Network(String),

    #[error("tile {tile_idx} out of range (max {max})")]
    TileOutOfRange { tile_idx: usize, max: usize },

    #[error("no IFD entries found in TIFF")]
    NoIfd,

    #[error("bbox does not intersect raster extent")]
    BBoxOutside,

    #[error("decompression failed: {0}")]
    Decompress(String),

    #[error("core error: {0}")]
    Core(#[from] surtgis_core::Error),
}

/// Result alias for cloud operations.
pub type Result<T> = std::result::Result<T, CloudError>;
