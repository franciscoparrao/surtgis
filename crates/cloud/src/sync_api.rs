//! Blocking (synchronous) API for native platforms.
//!
//! Wraps the async [`CogReader`] with a Tokio runtime so callers don't need
//! to manage their own async runtime.

#[cfg(feature = "native")]
mod inner {
    use surtgis_core::raster::{Raster, RasterElement};

    use crate::cog_reader::{CogMetadata, CogReader, CogReaderOptions, OverviewInfo};
    use crate::error::Result;
    use crate::tile_index::BBox;

    /// Blocking wrapper around [`CogReader`].
    ///
    /// Uses an internal single-threaded Tokio runtime. Not available on WASM.
    pub struct CogReaderBlocking {
        rt: tokio::runtime::Runtime,
        inner: CogReader,
    }

    impl CogReaderBlocking {
        /// Open a remote COG (blocking).
        pub fn open(url: &str, options: CogReaderOptions) -> Result<Self> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| crate::error::CloudError::Network(e.to_string()))?;

            let inner = rt.block_on(CogReader::open(url, options))?;

            Ok(Self { rt, inner })
        }

        /// Read a bounding box (blocking).
        pub fn read_bbox<T: RasterElement>(
            &mut self,
            bbox: &BBox,
            overview: Option<usize>,
        ) -> Result<Raster<T>> {
            self.rt.block_on(self.inner.read_bbox(bbox, overview))
        }

        /// Read the full raster extent (blocking).
        pub fn read_full<T: RasterElement>(
            &mut self,
            overview: Option<usize>,
        ) -> Result<Raster<T>> {
            self.rt.block_on(self.inner.read_full(overview))
        }

        /// Return COG metadata.
        pub fn metadata(&self) -> CogMetadata {
            self.inner.metadata()
        }

        /// Return overview information.
        pub fn overviews(&self) -> Vec<OverviewInfo> {
            self.inner.overviews()
        }
    }

    /// One-shot convenience function: open a COG, read a bbox, return the raster.
    pub fn read_cog<T: RasterElement>(
        url: &str,
        bbox: &BBox,
        options: CogReaderOptions,
    ) -> Result<Raster<T>> {
        let mut reader = CogReaderBlocking::open(url, options)?;
        reader.read_bbox(bbox, None)
    }

    /// One-shot: open a COG and read the full extent.
    pub fn read_cog_full<T: RasterElement>(
        url: &str,
        options: CogReaderOptions,
    ) -> Result<Raster<T>> {
        let mut reader = CogReaderBlocking::open(url, options)?;
        reader.read_full(None)
    }

    // ── STAC blocking wrappers ───────────────────────────────────────

    use crate::stac_client::{StacCatalog, StacClient, StacClientOptions};
    use crate::stac_models::{StacItem, StacItemCollection, StacSearchParams};

    /// Blocking wrapper around [`StacClient`].
    pub struct StacClientBlocking {
        rt: tokio::runtime::Runtime,
        inner: StacClient,
    }

    impl StacClientBlocking {
        /// Create a new blocking STAC client.
        pub fn new(catalog: StacCatalog, options: StacClientOptions) -> Result<Self> {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| crate::error::CloudError::Network(e.to_string()))?;

            let inner = StacClient::new(catalog, options)?;
            Ok(Self { rt, inner })
        }

        /// Execute a single search request (blocking).
        pub fn search(&self, params: &StacSearchParams) -> Result<StacItemCollection> {
            self.rt.block_on(self.inner.search(params))
        }

        /// Search with automatic pagination (blocking).
        pub fn search_all(&self, params: &StacSearchParams) -> Result<Vec<StacItem>> {
            self.rt.block_on(self.inner.search_all(params))
        }

        /// Sign an asset href for Planetary Computer (blocking).
        pub fn sign_asset_href(&self, href: &str, collection: &str) -> Result<String> {
            self.rt.block_on(self.inner.sign_asset_href(href, collection))
        }
    }

    /// One-shot: search a STAC catalog and read the first matching COG asset (blocking).
    pub fn stac_search_and_read<T: RasterElement>(
        catalog: StacCatalog,
        params: &StacSearchParams,
        asset_key: Option<&str>,
        bbox: &BBox,
    ) -> Result<Raster<T>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| crate::error::CloudError::Network(e.to_string()))?;

        rt.block_on(crate::stac_reader::search_and_read(catalog, params, asset_key, bbox))
    }
}

#[cfg(feature = "native")]
pub use inner::*;
