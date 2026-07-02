//! Error types for the cloud COG reader.

use thiserror::Error;

/// Errors produced by the cloud COG reader.
#[derive(Error, Debug)]
pub enum CloudError {
    /// An underlying HTTP request failed (transport, status, or body error).
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// The server does not accept HTTP Range requests, which are required for
    /// partial COG reads.
    #[error("server does not support Range requests for {url}")]
    RangeNotSupported {
        /// URL of the resource that rejected the Range request.
        url: String,
    },

    /// The TIFF/COG structure could not be parsed.
    #[error("invalid TIFF: {reason}")]
    InvalidTiff {
        /// Human-readable explanation of what was malformed.
        reason: String,
    },

    /// The tile uses a TIFF compression scheme this reader cannot decode.
    #[error("unsupported compression: {0}")]
    UnsupportedCompression(u16),

    /// The combination of bits-per-sample and sample-format is not supported.
    #[error("unsupported data type: bits_per_sample={bps}, sample_format={sf}")]
    UnsupportedDataType {
        /// TIFF `BitsPerSample` value.
        bps: u16,
        /// TIFF `SampleFormat` value.
        sf: u16,
    },

    /// The TIFF uses a planar configuration other than chunky (interleaved).
    #[error("unsupported planar configuration: {0} (only chunky=1 supported)")]
    UnsupportedPlanarConfig(u16),

    /// Authentication (e.g. AWS signing or SAS token) failed.
    #[error("authentication error: {0}")]
    Auth(String),

    /// The server answered with a non-success HTTP status (after the client
    /// exhausted its retries for retryable statuses such as 429/5xx).
    ///
    /// Carries the [`reqwest::StatusCode`] structurally so callers can match
    /// on it (e.g. rate limiting vs. server error) instead of parsing the
    /// error message.
    #[error("HTTP {status} fetching {url}")]
    HttpStatus {
        /// The HTTP status code returned by the server.
        status: reqwest::StatusCode,
        /// URL of the request that failed.
        url: String,
    },

    /// A network-level failure not represented by a specific HTTP error.
    #[error("network error: {0}")]
    Network(String),

    /// A requested tile index is outside the IFD's tile array.
    #[error("tile {tile_idx} out of range (max {max})")]
    TileOutOfRange {
        /// Requested tile index.
        tile_idx: usize,
        /// Number of tiles available in the IFD.
        max: usize,
    },

    /// The TIFF contained no Image File Directory entries.
    #[error("no IFD entries found in TIFF")]
    NoIfd,

    /// The requested bounding box does not intersect the raster's extent.
    #[error("bbox does not intersect raster extent")]
    BBoxOutside,

    /// A compressed tile could not be decompressed.
    #[error("decompression failed: {0}")]
    Decompress(String),

    /// An error originating in `surtgis-core`.
    #[error("core error: {0}")]
    Core(#[from] surtgis_core::Error),

    /// An error from the Zarr reader.
    #[cfg(feature = "zarr")]
    #[error("zarr error: {0}")]
    Zarr(String),

    /// An error from the NetCDF reader.
    #[cfg(feature = "netcdf")]
    #[error("netcdf error: {0}")]
    NetCdf(String),

    /// An error from the GRIB reader.
    #[cfg(feature = "grib")]
    #[error("grib error: {0}")]
    Grib(String),

    /// The requested variable was not present in the Zarr store.
    #[cfg(feature = "zarr")]
    #[error("zarr variable '{variable}' not found")]
    ZarrVariableNotFound {
        /// Name of the variable that was not found.
        variable: String,
    },

    /// A CF-conventions inconsistency was detected in the Zarr metadata.
    #[cfg(feature = "zarr")]
    #[error("zarr CF conventions error: {0}")]
    ZarrCfError(String),

    /// The requested time step is outside the Zarr time axis.
    #[cfg(feature = "zarr")]
    #[error("zarr time out of range: {requested} (available: {available})")]
    ZarrTimeOutOfRange {
        /// The requested time value.
        requested: String,
        /// The range of times actually available.
        available: String,
    },
}

impl CloudError {
    /// Return the HTTP status code carried by this error, if any.
    ///
    /// Covers both the structured [`CloudError::HttpStatus`] variant and
    /// status errors surfaced through the underlying [`reqwest::Error`].
    pub fn status(&self) -> Option<reqwest::StatusCode> {
        match self {
            CloudError::HttpStatus { status, .. } => Some(*status),
            CloudError::Http(e) => e.status(),
            _ => None,
        }
    }
}

/// Result alias for cloud operations.
pub type Result<T> = std::result::Result<T, CloudError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_status_variant_exposes_structured_status() {
        let err = CloudError::HttpStatus {
            status: reqwest::StatusCode::TOO_MANY_REQUESTS,
            url: "https://example.com/cog.tif".into(),
        };
        assert_eq!(err.status(), Some(reqwest::StatusCode::TOO_MANY_REQUESTS));
        // Display keeps the "HTTP <code> <reason> fetching <url>" shape that
        // downstream substring-based classifiers rely on (e.g. " 429").
        let msg = err.to_string();
        assert!(msg.contains(" 429"), "message was: {msg}");
        assert!(msg.contains("fetching https://example.com/cog.tif"));
    }

    #[test]
    fn non_http_errors_have_no_status() {
        let err = CloudError::Network("connection reset".into());
        assert_eq!(err.status(), None);
        let err = CloudError::NoIfd;
        assert_eq!(err.status(), None);
    }
}
