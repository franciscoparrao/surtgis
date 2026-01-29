//! Tiled processing for large rasters

use rayon::prelude::*;
use surtgis_core::raster::{Raster, RasterElement};

/// A tile representing a subset of a raster
#[derive(Debug, Clone)]
pub struct Tile {
    /// Row offset in the source raster
    pub row_offset: usize,
    /// Column offset in the source raster
    pub col_offset: usize,
    /// Number of rows in this tile
    pub rows: usize,
    /// Number of columns in this tile
    pub cols: usize,
    /// Overlap with neighboring tiles (for kernel operations)
    pub overlap: usize,
}

impl Tile {
    /// Create a new tile
    pub fn new(row_offset: usize, col_offset: usize, rows: usize, cols: usize, overlap: usize) -> Self {
        Self {
            row_offset,
            col_offset,
            rows,
            cols,
            overlap,
        }
    }

    /// Get the valid (non-overlapping) bounds within this tile
    pub fn valid_bounds(&self, src_rows: usize, src_cols: usize) -> (usize, usize, usize, usize) {
        let start_row = self.overlap;
        let start_col = self.overlap;

        let end_row = if self.row_offset + self.rows >= src_rows {
            self.rows
        } else {
            self.rows - self.overlap
        };

        let end_col = if self.col_offset + self.cols >= src_cols {
            self.cols
        } else {
            self.cols - self.overlap
        };

        (start_row, start_col, end_row, end_col)
    }

    /// Convert tile-local coordinates to source raster coordinates
    pub fn to_source_coords(&self, local_row: usize, local_col: usize) -> (usize, usize) {
        (
            self.row_offset + local_row,
            self.col_offset + local_col,
        )
    }
}

/// Iterator over tiles covering a raster
pub struct TileIterator {
    total_rows: usize,
    total_cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    overlap: usize,
    current_row: usize,
    current_col: usize,
}

impl TileIterator {
    /// Create a new tile iterator
    pub fn new(
        total_rows: usize,
        total_cols: usize,
        tile_size: usize,
        overlap: usize,
    ) -> Self {
        Self {
            total_rows,
            total_cols,
            tile_rows: tile_size,
            tile_cols: tile_size,
            overlap,
            current_row: 0,
            current_col: 0,
        }
    }

    /// Get the effective step size (tile size minus overlap)
    fn step_size(&self) -> (usize, usize) {
        (
            self.tile_rows.saturating_sub(self.overlap * 2).max(1),
            self.tile_cols.saturating_sub(self.overlap * 2).max(1),
        )
    }
}

impl Iterator for TileIterator {
    type Item = Tile;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            return None;
        }

        let (step_rows, step_cols) = self.step_size();

        // Calculate tile bounds with overlap
        let row_start = self.current_row.saturating_sub(self.overlap);
        let col_start = self.current_col.saturating_sub(self.overlap);

        let row_end = (self.current_row + self.tile_rows + self.overlap).min(self.total_rows);
        let col_end = (self.current_col + self.tile_cols + self.overlap).min(self.total_cols);

        let tile = Tile::new(
            row_start,
            col_start,
            row_end - row_start,
            col_end - col_start,
            self.overlap,
        );

        // Move to next tile
        self.current_col += step_cols;
        if self.current_col >= self.total_cols {
            self.current_col = 0;
            self.current_row += step_rows;
        }

        Some(tile)
    }
}

/// Processor for tiled raster operations
pub struct TiledProcessor {
    tile_size: usize,
    overlap: usize,
}

impl TiledProcessor {
    /// Create a new tiled processor
    pub fn new(tile_size: usize, overlap: usize) -> Self {
        Self { tile_size, overlap }
    }

    /// Default processor with 512x512 tiles and 1 pixel overlap
    pub fn default_for_terrain() -> Self {
        Self::new(512, 1)
    }

    /// Process a raster with a function applied to each cell
    ///
    /// The function receives (row, col, value, neighbors) where neighbors
    /// is a closure to access neighboring cells safely.
    pub fn process<T, U, F>(
        &self,
        input: &Raster<T>,
        output: &mut Raster<U>,
        f: F,
    ) where
        T: RasterElement,
        U: RasterElement,
        F: Fn(usize, usize, T, &dyn Fn(isize, isize) -> Option<T>) -> U + Sync + Send,
    {
        let (rows, cols) = input.shape();
        let tiles: Vec<_> = TileIterator::new(rows, cols, self.tile_size, self.overlap).collect();

        // Process tiles in parallel
        let results: Vec<(Tile, Vec<(usize, usize, U)>)> = tiles
            .into_par_iter()
            .map(|tile| {
                let mut tile_results = Vec::new();

                // Process each cell in the tile
                for local_row in 0..tile.rows {
                    for local_col in 0..tile.cols {
                        let (src_row, src_col) = tile.to_source_coords(local_row, local_col);

                        // Skip if outside valid output region (overlap areas)
                        if src_row == 0 && local_row > 0 {
                            continue;
                        }

                        let value = unsafe { input.get_unchecked(src_row, src_col) };

                        // Neighbor access closure
                        let neighbor = |dr: isize, dc: isize| -> Option<T> {
                            let nr = src_row as isize + dr;
                            let nc = src_col as isize + dc;

                            if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                                None
                            } else {
                                Some(unsafe { input.get_unchecked(nr as usize, nc as usize) })
                            }
                        };

                        let result = f(src_row, src_col, value, &neighbor);
                        tile_results.push((src_row, src_col, result));
                    }
                }

                (tile, tile_results)
            })
            .collect();

        // Merge results into output
        for (_, tile_results) in results {
            for (row, col, value) in tile_results {
                unsafe {
                    output.set_unchecked(row, col, value);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_iterator() {
        let tiles: Vec<_> = TileIterator::new(100, 100, 32, 1).collect();
        assert!(!tiles.is_empty());

        // First tile should start at (0, 0)
        assert_eq!(tiles[0].row_offset, 0);
        assert_eq!(tiles[0].col_offset, 0);
    }

    #[test]
    fn test_tile_coverage() {
        let rows = 100;
        let cols = 100;
        let mut covered = vec![vec![false; cols]; rows];

        for tile in TileIterator::new(rows, cols, 32, 0) {
            for r in tile.row_offset..tile.row_offset + tile.rows {
                for c in tile.col_offset..tile.col_offset + tile.cols {
                    if r < rows && c < cols {
                        covered[r][c] = true;
                    }
                }
            }
        }

        // All cells should be covered
        for r in 0..rows {
            for c in 0..cols {
                assert!(covered[r][c], "Cell ({}, {}) not covered", r, c);
            }
        }
    }
}
