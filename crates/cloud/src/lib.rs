//! # SurtGis Cloud
//!
//! Cloud Optimized GeoTIFF (COG) reader with HTTP Range request support.
//!
//! This crate provides efficient remote reading of COG files by fetching only
//! the tiles needed for a given bounding box, with LRU caching and concurrent
//! HTTP requests.
//!
//! ## Features
//!
//! - `deflate` (default): DEFLATE decompression via `flate2`
//! - `lzw` (default): LZW decompression via `weezl`
//! - `native`: Sync API via tokio `block_on` (not available on WASM)
//! - `wasm`: WASM bindings via `wasm-bindgen`

pub mod auth;
pub mod cache;
pub mod cloud_reader;
pub mod cog_reader;
pub mod decompress;
pub mod error;
pub mod geotiff_keys;
pub mod http;
pub mod ifd;
pub mod reproject;
pub mod stac_client;
pub mod stac_models;
pub mod stac_reader;
pub mod tile_index;

pub mod sync_api;
pub mod wasm_api;

#[cfg(feature = "zarr")]
pub mod zarr_cf;
#[cfg(feature = "zarr")]
pub mod zarr_auth;
#[cfg(feature = "zarr")]
pub mod zarr_reader;

pub use cloud_reader::{CloudRasterReader, RasterMeta};
pub use cog_reader::{CogMetadata, CogReader, CogReaderOptions, OverviewInfo};
pub use error::{CloudError, Result};
pub use stac_client::{StacCatalog, StacClient, StacClientOptions};
pub use stac_models::{AssetFormat, StacItem, StacItemCollection, StacSearchParams};
pub use tile_index::BBox;

#[cfg(feature = "zarr")]
pub use zarr_reader::{
    AggMethod, TimeReduction, TimeSelector, ZarrMetadata, ZarrReader, ZarrReaderOptions,
};

/// Blocking API re-exported as `blocking` module (native only).
#[cfg(feature = "native")]
pub mod blocking {
    pub use crate::sync_api::*;
}
