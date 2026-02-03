//! Application messages for inter-thread communication.

use std::path::PathBuf;
use std::time::Duration;

use surtgis_core::raster::Raster;

use super::DatasetId;

/// Messages sent from background threads to the main UI loop.
pub enum AppMessage {
    /// A raster was loaded from disk.
    RasterLoaded {
        path: PathBuf,
        raster: Raster<f64>,
    },
    /// An algorithm completed successfully.
    AlgoComplete {
        name: String,
        result: Raster<f64>,
        elapsed: Duration,
    },
    /// An algorithm completed with u8 output (flow direction, geomorphons, etc.).
    AlgoCompleteU8 {
        name: String,
        result: Raster<u8>,
        elapsed: Duration,
    },
    /// An algorithm completed with i32 output (watershed IDs, etc.).
    AlgoCompleteI32 {
        name: String,
        result: Raster<i32>,
        elapsed: Duration,
    },
    /// An algorithm or IO operation failed.
    Error {
        context: String,
        message: String,
    },
    /// A log message for the console.
    Log(LogEntry),
    /// A raster was saved to disk.
    RasterSaved {
        dataset_id: DatasetId,
        path: PathBuf,
    },

    /// STAC search completed.
    StacSearchComplete {
        items: Vec<crate::panels::stac_browser::StacSearchResult>,
        total: Option<u64>,
    },
    /// A STAC asset was downloaded and loaded as a raster.
    StacAssetLoaded {
        item_id: String,
        asset_key: String,
        raster: Raster<f64>,
    },
    /// A STAC operation failed.
    StacError {
        message: String,
    },
}

/// Log level for console messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warning,
    Error,
    Success,
}

/// A log entry for the console panel.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: std::time::SystemTime,
}

impl LogEntry {
    pub fn info(msg: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Info,
            message: msg.into(),
            timestamp: std::time::SystemTime::now(),
        }
    }

    pub fn warning(msg: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Warning,
            message: msg.into(),
            timestamp: std::time::SystemTime::now(),
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Error,
            message: msg.into(),
            timestamp: std::time::SystemTime::now(),
        }
    }

    pub fn success(msg: impl Into<String>) -> Self {
        Self {
            level: LogLevel::Success,
            message: msg.into(),
            timestamp: std::time::SystemTime::now(),
        }
    }
}
