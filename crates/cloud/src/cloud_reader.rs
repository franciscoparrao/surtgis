//! Common trait for cloud raster readers (COG, Zarr, etc.).
//!
//! Provides a unified interface for reading 2D rasters from cloud sources,
//! regardless of the underlying format.

use surtgis_core::CRS;
use surtgis_core::raster::{GeoTransform, Raster};

use crate::error::Result;
use crate::tile_index::BBox;

/// Unified metadata for any cloud raster source.
#[derive(Debug, Clone)]
pub struct RasterMeta {
    /// Affine transform mapping pixel coordinates to world coordinates.
    pub geo_transform: GeoTransform,
    /// Coordinate reference system, if known.
    pub crs: Option<CRS>,
    /// Nodata value, if declared.
    pub nodata: Option<f64>,
    /// Raster width in pixels.
    pub width: usize,
    /// Raster height in pixels.
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
