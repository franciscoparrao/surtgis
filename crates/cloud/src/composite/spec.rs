//! Declarative configuration of a composite run.
//!
//! [`CompositeSpec`] is a plain data description of *what* to composite —
//! catalog, collection, bbox, bands, time range and the tuning knobs — with
//! no I/O handles or trait objects, so it can be logged, compared and (in a
//! later R8 step) serialized as a checkpoint/resume manifest. The runtime
//! collaborators (asset resolver, mask applier, output sink, progress) are
//! passed separately to [`CompositeEngine::run`](super::CompositeEngine::run).

use surtgis_core::{CRS, GeoTransform, Raster};

use crate::BBox;

/// The output raster grid a composite is written onto.
///
/// Either supplied by the caller (to align the composite to an existing
/// reference raster, see [`OutputGrid::from_reference`]) or derived by the
/// engine from the first COG it probes.
#[derive(Debug, Clone)]
pub struct OutputGrid {
    /// Number of columns.
    pub cols: usize,
    /// Number of rows.
    pub rows: usize,
    /// Affine transform (origin at the top-left corner, `pixel_height < 0`).
    pub transform: GeoTransform,
    /// Coordinate reference system, if known.
    pub crs: Option<CRS>,
    /// Grid bounding box in the grid CRS (its `max_y` anchors row 0).
    pub bbox: BBox,
}

impl OutputGrid {
    /// Build a grid matching an existing reference raster (the `--align-to`
    /// case): same shape, transform and CRS, so the composite lands
    /// pixel-for-pixel on top of it.
    pub fn from_reference(reference: &Raster<f64>) -> Self {
        let gt = reference.transform();
        let (rows, cols) = reference.shape();
        let min_x = gt.origin_x;
        let max_y = gt.origin_y;
        let max_x = min_x + cols as f64 * gt.pixel_width;
        let min_y = max_y + rows as f64 * gt.pixel_height;
        Self {
            cols,
            rows,
            transform: *gt,
            crs: reference.crs().cloned(),
            bbox: BBox::new(min_x, min_y, max_x, max_y),
        }
    }

    /// Target output pixel size (used for COG overview selection).
    pub fn pixel_size(&self) -> f64 {
        self.transform.pixel_width.abs()
    }

    /// EPSG code of the grid CRS, if known.
    pub fn epsg(&self) -> Option<u32> {
        self.crs.as_ref().and_then(|c| c.epsg())
    }
}

/// Declarative description of a multiband composite run.
#[derive(Debug, Clone)]
pub struct CompositeSpec {
    /// Catalog identifier: `"pc"`, `"es"`, or a full STAC API URL.
    pub catalog: String,
    /// Collection id (e.g. `"sentinel-2-l2a"`).
    pub collection: String,
    /// Area of interest, in WGS84 (EPSG:4326) degrees.
    pub bbox_wgs84: BBox,
    /// Band asset keys to composite, in output order (e.g. `["red", "nir"]`).
    pub band_keys: Vec<String>,
    /// Cloud-mask asset key shared by all bands (e.g. `"scl"`), or `None`
    /// for collections that need no masking (SAR, derived products).
    pub mask_key: Option<String>,
    /// STAC datetime query (single instant or `start/end` range).
    pub datetime: String,
    /// Maximum number of scene dates to composite.
    pub max_scenes: usize,
    /// Output grid to align to. `None` derives the grid from the first COG.
    pub align_grid: Option<OutputGrid>,
    /// Requested strip height in rows (the memory model may cap it).
    pub strip_rows: usize,
    /// Bands downloaded together per chunk (RAM ↔ HTTP dial; 1 = min RAM).
    pub band_chunk_size: usize,
    /// RAM budget in GB for the strip-sizing model.
    pub budget_gb: f64,
    /// Abort after this many real tile failures (0 = never abort).
    pub max_tile_failures: usize,
    /// Cache decoded COG tiles on disk between strips.
    pub use_cache: bool,
}

impl CompositeSpec {
    /// Number of output bands.
    pub fn n_bands(&self) -> usize {
        self.band_keys.len()
    }
}
