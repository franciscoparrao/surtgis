//! Error types for SurtGis

use thiserror::Error;

/// Main error type for SurtGis operations.
#[derive(Error, Debug)]
pub enum Error {
    /// An underlying filesystem / I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A raster was requested with zero or otherwise invalid dimensions.
    #[error("Invalid raster dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Requested width in cells.
        width: usize,
        /// Requested height in cells.
        height: usize,
    },

    /// A `(row, col)` access fell outside the raster bounds.
    #[error("Index out of bounds: ({row}, {col}) in raster of size ({rows}, {cols})")]
    IndexOutOfBounds {
        /// Requested row.
        row: usize,
        /// Requested column.
        col: usize,
        /// Raster height (number of rows).
        rows: usize,
        /// Raster width (number of columns).
        cols: usize,
    },

    /// Two rasters that had to be the same shape disagreed.
    #[error("Raster size mismatch: expected ({er}, {ec}), got ({ar}, {ac})")]
    SizeMismatch {
        /// Expected number of rows.
        er: usize,
        /// Expected number of columns.
        ec: usize,
        /// Actual number of rows.
        ar: usize,
        /// Actual number of columns.
        ac: usize,
    },

    /// Two rasters that had to share a grid shape `(rows, cols)` disagreed.
    ///
    /// Structured successor of [`Error::SizeMismatch`]: carries the shapes
    /// as `(rows, cols)` tuples plus a free-form `context` describing which
    /// inputs disagreed (e.g. `"input raster 2 vs raster 0"`), so callers
    /// can both match programmatically and render an actionable message.
    #[error("Shape mismatch ({context}): expected {expected:?} (rows, cols), got {got:?}")]
    ShapeMismatch {
        /// Shape of the reference raster as `(rows, cols)`.
        expected: (usize, usize),
        /// Shape of the offending raster as `(rows, cols)`.
        got: (usize, usize),
        /// Which inputs disagreed (operation and/or input positions).
        context: String,
    },

    /// Multi-raster inputs are not on the same georeferenced grid.
    ///
    /// Returned when rasters share a shape but their geotransforms
    /// (origin, pixel size, rotation) or EPSG codes differ, so a
    /// cell-by-cell operation across them would silently combine
    /// pixels from different ground locations.
    #[error("Misaligned rasters: {reason}")]
    Misaligned {
        /// Human-readable description of the first alignment violation found.
        reason: String,
    },

    /// Two inputs declared incompatible coordinate reference systems.
    #[error("CRS mismatch: {0} vs {1}")]
    CrsMismatch(String, String),

    /// A pixel data type the operation does not support.
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),

    /// An operation that needs a nodata value was called on a raster without one.
    #[error("No data value not set")]
    NoDataNotSet,

    /// An error surfaced by the optional GDAL backend.
    #[error("GDAL error: {0}")]
    #[cfg(feature = "gdal")]
    Gdal(String),

    /// A caller-supplied parameter was out of range or otherwise invalid.
    #[error("Invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// The offending value (formatted).
        value: String,
        /// Why it was rejected.
        reason: String,
    },

    /// A numerical / algorithmic failure inside a computation.
    #[error("Algorithm error: {0}")]
    Algorithm(String),

    /// Any other error, carrying a human-readable message.
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
