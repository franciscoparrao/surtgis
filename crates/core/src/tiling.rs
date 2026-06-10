//! 2-D tile iteration with overlap for windowed processing.
//!
//! [`TileGrid`] partitions a raster grid into fixed-size tiles and
//! yields, for each tile, both its **core** extent (the cells the tile
//! owns — cores never overlap and together cover the grid exactly) and
//! its **read** extent (core expanded by `overlap` cells on every
//! side, clamped to the grid). Algorithms read the expanded window,
//! compute, and write back only the core — the standard halo pattern
//! for kernels, embeddings, and seam-free tiled inference.
//!
//! This is the 2-D counterpart of [`StripProcessor`](crate::streaming::StripProcessor),
//! which handles the 1-D (row strip) case with file-backed I/O.
//!
//! ```
//! use surtgis_core::tiling::TileGrid;
//!
//! let grid = TileGrid::new(1000, 1500, 256, 16);
//! for tile in grid {
//!     // read tile.read_* window, compute, write tile.core_* cells
//!     assert!(tile.read_rows <= 256 + 2 * 16);
//! }
//! ```

/// One tile of a [`TileGrid`]: a core extent plus a read extent that
/// includes the clamped overlap halo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    /// Tile index in row-major tile order.
    pub index: usize,
    /// First row of the core extent.
    pub core_row: usize,
    /// First column of the core extent.
    pub core_col: usize,
    /// Core extent height (≤ tile_size; smaller on the last band).
    pub core_rows: usize,
    /// Core extent width (≤ tile_size; smaller on the last band).
    pub core_cols: usize,
    /// First row of the read (halo-expanded) extent.
    pub read_row: usize,
    /// First column of the read (halo-expanded) extent.
    pub read_col: usize,
    /// Read extent height.
    pub read_rows: usize,
    /// Read extent width.
    pub read_cols: usize,
}

impl Tile {
    /// Offset of the core's first row inside the read window.
    pub fn core_offset_row(&self) -> usize {
        self.core_row - self.read_row
    }

    /// Offset of the core's first column inside the read window.
    pub fn core_offset_col(&self) -> usize {
        self.core_col - self.read_col
    }
}

/// Iterator over the tiles covering a `rows × cols` grid.
///
/// Cores tile the grid exactly (no gaps, no overlap); read extents
/// add `overlap` cells of halo on each side, clamped to the grid
/// bounds. Construction is cheap; the iterator allocates nothing.
#[derive(Debug, Clone)]
pub struct TileGrid {
    rows: usize,
    cols: usize,
    tile_size: usize,
    overlap: usize,
    tiles_down: usize,
    tiles_across: usize,
    next: usize,
}

impl TileGrid {
    /// Create a tile grid. `tile_size` is the core side length; it is
    /// clamped to at least 1. Empty grids produce no tiles.
    pub fn new(rows: usize, cols: usize, tile_size: usize, overlap: usize) -> Self {
        let tile_size = tile_size.max(1);
        let tiles_down = rows.div_ceil(tile_size);
        let tiles_across = cols.div_ceil(tile_size);
        Self {
            rows,
            cols,
            tile_size,
            overlap,
            tiles_down,
            tiles_across,
            next: 0,
        }
    }

    /// Total number of tiles.
    pub fn len(&self) -> usize {
        self.tiles_down * self.tiles_across
    }

    /// Whether the grid produces no tiles.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The tile at a given row-major tile index, if in range.
    pub fn get(&self, index: usize) -> Option<Tile> {
        if index >= self.len() {
            return None;
        }
        let trow = index / self.tiles_across;
        let tcol = index % self.tiles_across;

        let core_row = trow * self.tile_size;
        let core_col = tcol * self.tile_size;
        let core_rows = self.tile_size.min(self.rows - core_row);
        let core_cols = self.tile_size.min(self.cols - core_col);

        let read_row = core_row.saturating_sub(self.overlap);
        let read_col = core_col.saturating_sub(self.overlap);
        let read_end_row = (core_row + core_rows + self.overlap).min(self.rows);
        let read_end_col = (core_col + core_cols + self.overlap).min(self.cols);

        Some(Tile {
            index,
            core_row,
            core_col,
            core_rows,
            core_cols,
            read_row,
            read_col,
            read_rows: read_end_row - read_row,
            read_cols: read_end_col - read_col,
        })
    }
}

impl Iterator for TileGrid {
    type Item = Tile;

    fn next(&mut self) -> Option<Tile> {
        let tile = self.get(self.next)?;
        self.next += 1;
        Some(tile)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len().saturating_sub(self.next);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TileGrid {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cores_cover_grid_exactly() {
        let (rows, cols) = (1000, 1500);
        let mut covered = vec![false; rows * cols];
        for tile in TileGrid::new(rows, cols, 256, 16) {
            for r in tile.core_row..tile.core_row + tile.core_rows {
                for c in tile.core_col..tile.core_col + tile.core_cols {
                    assert!(!covered[r * cols + c], "core overlap at ({}, {})", r, c);
                    covered[r * cols + c] = true;
                }
            }
        }
        assert!(covered.iter().all(|&v| v), "gap in core coverage");
    }

    #[test]
    fn read_extent_clamps_at_borders() {
        let grid = TileGrid::new(100, 100, 50, 10);
        let tiles: Vec<Tile> = grid.collect();
        assert_eq!(tiles.len(), 4);

        // Top-left tile: no halo above/left, halo below/right
        let tl = tiles[0];
        assert_eq!((tl.read_row, tl.read_col), (0, 0));
        assert_eq!((tl.read_rows, tl.read_cols), (60, 60));
        assert_eq!((tl.core_offset_row(), tl.core_offset_col()), (0, 0));

        // Bottom-right tile: halo above/left only
        let br = tiles[3];
        assert_eq!((br.read_row, br.read_col), (40, 40));
        assert_eq!((br.read_rows, br.read_cols), (60, 60));
        assert_eq!((br.core_offset_row(), br.core_offset_col()), (10, 10));
    }

    #[test]
    fn interior_tile_has_full_halo() {
        let grid = TileGrid::new(300, 300, 100, 8);
        let center = grid.get(4).unwrap(); // tile (1,1) of 3x3
        assert_eq!((center.core_row, center.core_col), (100, 100));
        assert_eq!((center.read_row, center.read_col), (92, 92));
        assert_eq!((center.read_rows, center.read_cols), (116, 116));
    }

    #[test]
    fn ragged_last_tiles() {
        let grid = TileGrid::new(70, 130, 64, 4);
        assert_eq!(grid.len(), 6); // 2 down x 3 across
        let last = grid.get(5).unwrap();
        assert_eq!((last.core_rows, last.core_cols), (6, 2));
        assert_eq!(
            last.read_row + last.read_rows,
            70,
            "read extent must clamp to grid"
        );
        assert_eq!(last.read_col + last.read_cols, 130);
    }

    #[test]
    fn zero_overlap_read_equals_core() {
        for tile in TileGrid::new(128, 128, 32, 0) {
            assert_eq!(tile.read_row, tile.core_row);
            assert_eq!(tile.read_rows, tile.core_rows);
            assert_eq!(tile.read_cols, tile.core_cols);
        }
    }

    #[test]
    fn empty_grid_yields_nothing() {
        assert_eq!(TileGrid::new(0, 100, 64, 4).count(), 0);
        assert!(TileGrid::new(0, 0, 64, 0).is_empty());
    }

    #[test]
    fn exact_size_iterator() {
        let mut grid = TileGrid::new(512, 512, 128, 16);
        assert_eq!(grid.len(), 16);
        grid.next();
        assert_eq!(grid.size_hint(), (15, Some(15)));
    }
}
