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
    /// Uses a multi-threaded Tokio runtime (2 worker threads) for efficient
    /// parallel HTTP range request I/O. Not available on WASM.
    pub struct CogReaderBlocking {
        rt: tokio::runtime::Runtime,
        inner: CogReader,
    }

    impl CogReaderBlocking {
        /// Open a remote COG (blocking).
        pub fn open(url: &str, options: CogReaderOptions) -> Result<Self> {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
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

        /// Get collection Zarr auth info (token, account, container) for PC (blocking).
        #[cfg(feature = "zarr")]
        pub fn get_collection_zarr_auth(
            &self,
            collection: &str,
        ) -> Result<Option<(String, String, String)>> {
            self.rt.block_on(self.inner.get_collection_zarr_auth(collection))
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

    // ── CloudRasterReader trait impls ────────────────────────────────

    impl crate::cloud_reader::CloudRasterReader for CogReaderBlocking {
        fn read_bbox_f64(&mut self, bbox: &BBox) -> Result<Raster<f64>> {
            self.read_bbox(bbox, None)
        }

        fn raster_meta(&self) -> crate::cloud_reader::RasterMeta {
            let m = self.metadata();
            crate::cloud_reader::RasterMeta {
                geo_transform: m.geo_transform,
                crs: m.crs,
                nodata: m.nodata,
                width: m.width as usize,
                height: m.height as usize,
            }
        }
    }
}

#[cfg(feature = "native")]
pub use inner::*;

// ── Zarr blocking wrappers ──────────────────────────────────────────

#[cfg(all(feature = "zarr", feature = "native"))]
mod zarr_inner {
    use surtgis_core::raster::Raster;

    use crate::error::Result;
    use crate::tile_index::BBox;
    use crate::zarr_reader::{TimeReduction, ZarrMetadata, ZarrReader, ZarrReaderOptions};

    /// Blocking wrapper around [`ZarrReader`].
    pub struct ZarrReaderBlocking {
        rt: tokio::runtime::Runtime,
        inner: ZarrReader,
    }

    impl ZarrReaderBlocking {
        /// Open a Zarr store and select a variable (blocking).
        pub fn open(
            store_url: &str,
            variable: &str,
            options: ZarrReaderOptions,
        ) -> Result<Self> {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .map_err(|e| crate::error::CloudError::Network(e.to_string()))?;

            let inner = rt.block_on(ZarrReader::open(store_url, variable, options))?;
            Ok(Self { rt, inner })
        }

        /// Read a geographic bounding box at a specific time (blocking).
        pub fn read_bbox(
            &self,
            bbox: &BBox,
            time: &TimeReduction,
        ) -> Result<Raster<f64>> {
            self.rt.block_on(self.inner.read_bbox(bbox, time))
        }

        /// Read the full spatial extent at a specific time (blocking).
        pub fn read_full(&self, time: &TimeReduction) -> Result<Raster<f64>> {
            self.rt.block_on(self.inner.read_full(time))
        }

        /// Return metadata.
        pub fn metadata(&self) -> &ZarrMetadata {
            self.inner.metadata()
        }
    }

    /// One-shot: list variables in a Zarr store (blocking).
    pub fn zarr_list_variables(
        store_url: &str,
        options: ZarrReaderOptions,
    ) -> Result<Vec<String>> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| crate::error::CloudError::Network(e.to_string()))?;
        rt.block_on(ZarrReader::list_variables(store_url, options))
    }

    // ── CloudRasterReader impl for ZarrReaderBlocking ───────────────

    /// Wraps a [`ZarrReaderBlocking`] with a fixed [`TimeReduction`] for use
    /// through the [`CloudRasterReader`](crate::cloud_reader::CloudRasterReader) trait.
    pub struct ZarrReaderWithTime {
        pub reader: ZarrReaderBlocking,
        pub time: TimeReduction,
    }

    impl crate::cloud_reader::CloudRasterReader for ZarrReaderWithTime {
        fn read_bbox_f64(&mut self, bbox: &crate::tile_index::BBox) -> crate::error::Result<surtgis_core::raster::Raster<f64>> {
            self.reader.read_bbox(bbox, &self.time)
        }

        fn raster_meta(&self) -> crate::cloud_reader::RasterMeta {
            let m = self.reader.metadata();
            crate::cloud_reader::RasterMeta {
                geo_transform: m.geo_transform.clone(),
                crs: m.crs.clone(),
                nodata: m.nodata,
                width: *m.shape.last().unwrap_or(&0) as usize,
                height: m.shape.get(m.shape.len().wrapping_sub(2)).copied().unwrap_or(0) as usize,
            }
        }
    }
}

#[cfg(all(feature = "zarr", feature = "native"))]
pub use zarr_inner::*;
