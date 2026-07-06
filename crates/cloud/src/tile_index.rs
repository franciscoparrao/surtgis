//! Tile grid math for COG files.
//!
//! Maps geographic bounding boxes to tile indices and computes the pixel
//! regions within tiles that contribute to the output raster.

use surtgis_core::raster::GeoTransform;

/// A geographic bounding box.
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    /// Minimum x (western/left) coordinate.
    pub min_x: f64,
    /// Minimum y (southern/bottom) coordinate.
    pub min_y: f64,
    /// Maximum x (eastern/right) coordinate.
    pub max_x: f64,
    /// Maximum y (northern/top) coordinate.
    pub max_y: f64,
}

impl BBox {
    /// Construct a bounding box from its four corner coordinates.
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// Check if two bboxes intersect.
    pub fn intersects(&self, other: &BBox) -> bool {
        self.min_x < other.max_x
            && self.max_x > other.min_x
            && self.min_y < other.max_y
            && self.max_y > other.min_y
    }
}

/// Descriptor for a tile that needs to be fetched.
#[derive(Debug, Clone)]
pub struct TileRequest {
    /// Linear tile index in the TIFF tile array.
    pub tile_idx: usize,
    /// Tile column in the tile grid.
    pub tile_col: usize,
    /// Tile row in the tile grid.
    pub tile_row: usize,
}

/// Result of mapping a bbox to the tile grid.
#[derive(Debug, Clone)]
pub struct TileMapping {
    /// Tiles that need to be fetched.
    pub tiles: Vec<TileRequest>,
    /// Number of tile columns in the tile grid.
    pub tiles_across: usize,
    /// Number of tile rows in the tile grid.
    pub tiles_down: usize,
    /// Pixel range in the full image: (min_col, min_row, max_col_exclusive, max_row_exclusive).
    pub pixel_window: (usize, usize, usize, usize),
    /// Output raster dimensions (rows, cols).
    pub output_shape: (usize, usize),
}

/// Compute which tiles are needed for a given bounding box.
///
/// Returns the tile indices and the pixel window within the full image.
pub fn tiles_for_bbox(
    bbox: &BBox,
    geo_transform: &GeoTransform,
    image_width: u32,
    image_height: u32,
    tile_width: u32,
    tile_height: u32,
) -> Option<TileMapping> {
    let iw = image_width as usize;
    let ih = image_height as usize;
    let tw = tile_width as usize;
    let th = tile_height as usize;

    // Convert bbox corners to pixel coordinates.
    // For north-up images, min_y maps to max_row and max_y to min_row.
    let (col_a, row_a) = geo_transform.geo_to_pixel(bbox.min_x, bbox.max_y);
    let (col_b, row_b) = geo_transform.geo_to_pixel(bbox.max_x, bbox.min_y);

    let min_col = col_a.min(col_b).floor() as isize;
    let max_col = col_a.max(col_b).ceil() as isize;
    let min_row = row_a.min(row_b).floor() as isize;
    let max_row = row_a.max(row_b).ceil() as isize;

    // Clamp to image bounds.
    let min_col = (min_col.max(0) as usize).min(iw);
    let max_col = (max_col.max(0) as usize).min(iw);
    let min_row = (min_row.max(0) as usize).min(ih);
    let max_row = (max_row.max(0) as usize).min(ih);

    if min_col >= max_col || min_row >= max_row {
        return None;
    }

    let tiles_across = iw.div_ceil(tw);
    let tiles_down = ih.div_ceil(th);

    // Tile range
    let tile_col_min = min_col / tw;
    let tile_col_max = max_col.div_ceil(tw); // exclusive
    let tile_row_min = min_row / th;
    let tile_row_max = max_row.div_ceil(th);

    let tile_col_max = tile_col_max.min(tiles_across);
    let tile_row_max = tile_row_max.min(tiles_down);

    let mut tiles = Vec::new();
    for tr in tile_row_min..tile_row_max {
        for tc in tile_col_min..tile_col_max {
            let tile_idx = tr * tiles_across + tc;
            tiles.push(TileRequest {
                tile_idx,
                tile_col: tc,
                tile_row: tr,
            });
        }
    }

    let output_rows = max_row - min_row;
    let output_cols = max_col - min_col;

    Some(TileMapping {
        tiles,
        tiles_across,
        tiles_down,
        pixel_window: (min_col, min_row, max_col, max_row),
        output_shape: (output_rows, output_cols),
    })
}

/// Select the best overview level for a given bounding box and desired output size.
///
/// Returns the index into the IFD list (0 = full resolution).
/// `ifd_dimensions` is a slice of (width, height) for each IFD.
pub fn select_overview(
    bbox: &BBox,
    geo_transform: &GeoTransform,
    ifd_dimensions: &[(u32, u32)],
    target_pixels: Option<usize>,
) -> usize {
    let target = match target_pixels {
        Some(t) => t,
        None => return 0, // Full resolution if no target
    };

    // Calculate how many pixels the bbox covers at full resolution
    let (col_a, row_a) = geo_transform.geo_to_pixel(bbox.min_x, bbox.max_y);
    let (col_b, row_b) = geo_transform.geo_to_pixel(bbox.max_x, bbox.min_y);
    let full_cols = (col_b - col_a).abs();
    let full_rows = (row_b - row_a).abs();
    let full_pixels = (full_cols * full_rows) as usize;

    if full_pixels <= target {
        return 0;
    }

    // Find the coarsest overview that still has enough pixels.
    // IFDs are typically: [full, overview1, overview2, ...] with decreasing resolution.
    let ratio = (full_pixels as f64 / target as f64).sqrt();

    select_overview_by_ratio(ifd_dimensions, ratio)
}

/// Select the coarsest overview whose downsample factor (`scale = full_width /
/// overview_width`) does not exceed `ratio`.
///
/// `ifd_dimensions[0]` must be the full-resolution dimensions; the rest are
/// overviews in any order (typically decreasing resolution). Returns `0`
/// (full resolution) when no overview qualifies — i.e. `ratio` is smaller
/// than even the finest overview's scale, so downsampling that far would
/// lose more detail than the caller wants.
///
/// Note this deliberately does NOT stop at the *first* overview whose scale
/// is `<= ratio` (that would pick the finest, most wasteful, qualifying
/// overview) — it scans all of them and keeps the one with the *largest*
/// qualifying scale, i.e. the coarsest overview that still meets the target.
pub fn select_overview_by_ratio(ifd_dimensions: &[(u32, u32)], ratio: f64) -> usize {
    if ifd_dimensions.is_empty() {
        return 0;
    }
    let (fw, _fh) = ifd_dimensions[0];
    let mut best_idx = 0usize;
    let mut best_scale = 1.0f64; // full resolution's own "scale"
    for (i, &(w, _h)) in ifd_dimensions.iter().enumerate().skip(1) {
        if w == 0 {
            continue;
        }
        let scale = fw as f64 / w as f64;
        if scale <= ratio && scale > best_scale {
            best_scale = scale;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiles_for_bbox_simple() {
        let gt = GeoTransform::new(0.0, 100.0, 1.0, -1.0);
        let bbox = BBox::new(10.0, 10.0, 30.0, 30.0);

        let mapping = tiles_for_bbox(&bbox, &gt, 100, 100, 32, 32).unwrap();

        // Pixels: col 10..30, row 70..90
        assert_eq!(mapping.pixel_window, (10, 70, 30, 90));
        assert_eq!(mapping.output_shape, (20, 20));
        assert!(!mapping.tiles.is_empty());
    }

    #[test]
    fn test_tiles_for_bbox_outside() {
        let gt = GeoTransform::new(0.0, 100.0, 1.0, -1.0);
        let bbox = BBox::new(200.0, 200.0, 300.0, 300.0);

        let mapping = tiles_for_bbox(&bbox, &gt, 100, 100, 32, 32);
        assert!(mapping.is_none());
    }

    #[test]
    fn test_bbox_intersects() {
        let a = BBox::new(0.0, 0.0, 10.0, 10.0);
        let b = BBox::new(5.0, 5.0, 15.0, 15.0);
        let c = BBox::new(20.0, 20.0, 30.0, 30.0);

        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_select_overview() {
        let gt = GeoTransform::new(0.0, 1000.0, 1.0, -1.0);
        let bbox = BBox::new(0.0, 0.0, 1000.0, 1000.0);
        let dims = vec![(1000, 1000), (500, 500), (250, 250)];

        // Full resolution needed
        assert_eq!(select_overview(&bbox, &gt, &dims, None), 0);

        // Target 250k pixels → ~500x500 → overview index 1
        let idx = select_overview(&bbox, &gt, &dims, Some(250_000));
        assert!(idx >= 1);
    }

    /// Regression test for the "returns finest qualifying overview instead of
    /// coarsest" bug: with scales [1, 2, 4, 8, 16], a ratio of 8 must select
    /// the scale=8 overview (index 3), not stop early at scale=2 (index 1).
    #[test]
    fn test_select_overview_by_ratio_picks_coarsest_not_finest() {
        // full_width = 1600 → widths for scale 1,2,4,8,16
        let dims = vec![(1600, 1600), (800, 800), (400, 400), (200, 200), (100, 100)];

        // ratio=8 → exact match at scale=8 → index 3.
        assert_eq!(select_overview_by_ratio(&dims, 8.0), 3);

        // ratio=5 → largest scale <= 5 is scale=4 → index 2 (NOT scale=2/index 1).
        assert_eq!(select_overview_by_ratio(&dims, 5.0), 2);

        // ratio=1 → no overview qualifies (finest overview has scale=2 > 1) → full res.
        assert_eq!(select_overview_by_ratio(&dims, 1.0), 0);

        // ratio=16 → coarsest overview exactly matches → index 4.
        assert_eq!(select_overview_by_ratio(&dims, 16.0), 4);

        // ratio beyond the coarsest overview still returns the coarsest, not "no match".
        assert_eq!(select_overview_by_ratio(&dims, 100.0), 4);

        // Empty dimensions → full resolution (no panic).
        assert_eq!(select_overview_by_ratio(&[], 8.0), 0);
    }
}
