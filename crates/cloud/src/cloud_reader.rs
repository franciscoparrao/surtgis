//! Common trait for cloud raster readers (COG, Zarr, etc.).
//!
//! Provides a unified interface for reading 2D rasters from cloud sources,
//! regardless of the underlying format.

use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::CRS;

use crate::error::Result;
use crate::tile_index::BBox;

/// Unified metadata for any cloud raster source.
#[derive(Debug, Clone)]
pub struct RasterMeta {
    pub geo_transform: GeoTransform,
    pub crs: Option<CRS>,
    pub nodata: Option<f64>,
    pub width: usize,
    pub height: usize,
}

/// Common interface for reading 2D rasters from cloud sources.
///
/// Both [`CogReaderBlocking`](crate::blocking::CogReaderBlocking) and
/// [`ZarrReaderBlocking`](crate::blocking::ZarrReaderBlocking) implement
/// this trait, enabling format-agnostic raster access.
///
/// `open()` is intentionally excluded — COG and Zarr have fundamentally
/// different initialization requirements (URL vs URL+variable+time).
/// Uses `&mut self` because COG readers mutate an internal LRU cache.
pub trait CloudRasterReader {
    /// Read a geographic bounding box, returning a 2D f64 raster.
    fn read_bbox_f64(&mut self, bbox: &BBox) -> Result<Raster<f64>>;

    /// Return unified metadata about the raster source.
    fn raster_meta(&self) -> RasterMeta;
}
