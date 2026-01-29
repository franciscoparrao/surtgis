//! Error types for SurtGis

use thiserror::Error;

/// Main error type for SurtGis operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid raster dimensions: {width}x{height}")]
    InvalidDimensions { width: usize, height: usize },

    #[error("Index out of bounds: ({row}, {col}) in raster of size ({rows}, {cols})")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    #[error("Raster size mismatch: expected ({er}, {ec}), got ({ar}, {ac})")]
    SizeMismatch { er: usize, ec: usize, ar: usize, ac: usize },

    #[error("CRS mismatch: {0} vs {1}")]
    CrsMismatch(String, String),

    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    #[error("No data value not set")]
    NoDataNotSet,

    #[error("GDAL error: {0}")]
    #[cfg(feature = "gdal")]
    Gdal(String),

    #[error("Invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        name: &'static str,
        value: String,
        reason: String,
    },

    #[error("Algorithm error: {0}")]
    Algorithm(String),

    #[error("{0}")]
    Other(String),
}

#[cfg(feature = "gdal")]
impl From<gdal::errors::GdalError> for Error {
    fn from(e: gdal::errors::GdalError) -> Self {
        Error::Gdal(e.to_string())
    }
}

/// Result type alias for SurtGis operations
pub type Result<T> = std::result::Result<T, Error>;
