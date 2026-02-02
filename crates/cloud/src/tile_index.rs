//! Tile grid math for COG files.
//!
//! Maps geographic bounding boxes to tile indices and computes the pixel
//! regions within tiles that contribute to the output raster.

use surtgis_core::raster::GeoTransform;

/// A geographic bounding box.
#[derive(Debug, Clone, Copy)]
pub struct BBox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl BBox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self { min_x, min_y, max_x, max_y }
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

    let tiles_across = (iw + tw - 1) / tw;
    let tiles_down = (ih + th - 1) / th;

    // Tile range
    let tile_col_min = min_col / tw;
    let tile_col_max = (max_col + tw - 1) / tw; // exclusive
    let tile_row_min = min_row / th;
    let tile_row_max = (max_row + th - 1) / th;

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

    for (i, &(w, _h)) in ifd_dimensions.iter().enumerate().skip(1) {
        let (fw, _fh) = ifd_dimensions[0];
        let scale = fw as f64 / w as f64;
        if scale <= ratio {
            return i;
        }
    }

    // Most coarse overview
    ifd_dimensions.len().saturating_sub(1)
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
}
