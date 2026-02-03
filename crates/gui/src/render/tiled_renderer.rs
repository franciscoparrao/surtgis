//! Tiled renderer with LRU cache for large rasters (> 4096x4096).
//!
//! Splits the raster into 512×512 tiles and only keeps the most recently used
//! ones in GPU memory, allowing smooth viewing of rasters far larger than VRAM.

use std::collections::HashMap;

use egui::{ColorImage, Context, TextureHandle, TextureOptions};

/// Tile size in pixels.
const TILE_SIZE: usize = 512;

/// Maximum number of tiles kept in cache.
const MAX_TILES: usize = 256;

/// Threshold: tiled renderer activates when rows*cols exceeds this.
pub const TILED_THRESHOLD: usize = 4096 * 4096;

/// Unique key for a tile within a generation of the raster data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TileKey {
    generation: u64,
    col: usize,
    row: usize,
}

/// Entry in the LRU list.
struct TileEntry {
    texture: TextureHandle,
    /// Last frame this tile was accessed.
    last_used: u64,
}

/// Tiled renderer with an LRU eviction cache.
pub struct TiledRenderer {
    cache: HashMap<TileKey, TileEntry>,
    /// Monotonically increasing generation counter; bumped when the raster data changes.
    generation: u64,
    /// Frame counter for LRU.
    frame: u64,
    /// Raster dimensions of the current generation.
    raster_rows: usize,
    raster_cols: usize,
}

impl Default for TiledRenderer {
    fn default() -> Self {
        Self {
            cache: HashMap::new(),
            generation: 0,
            frame: 0,
            raster_rows: 0,
            raster_cols: 0,
        }
    }
}

impl TiledRenderer {
    /// Call when the raster data changes (new dataset, different colormap, etc.).
    pub fn invalidate(&mut self, rows: usize, cols: usize) {
        self.generation += 1;
        self.raster_rows = rows;
        self.raster_cols = cols;
        // Old-generation tiles will be evicted on demand.
    }

    /// Should the tiled renderer be used for this raster size?
    pub fn should_use(rows: usize, cols: usize) -> bool {
        rows * cols > TILED_THRESHOLD
    }

    /// Number of tile columns.
    pub fn tile_cols(&self) -> usize {
        (self.raster_cols + TILE_SIZE - 1) / TILE_SIZE
    }

    /// Number of tile rows.
    pub fn tile_rows(&self) -> usize {
        (self.raster_rows + TILE_SIZE - 1) / TILE_SIZE
    }

    /// Begin a new frame (for LRU tracking).
    pub fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Determine which tile indices are visible within `viewport` given the
    /// current `scale` and `offset` of the map canvas.
    ///
    /// Returns `(col, row)` pairs of visible tiles.
    pub fn visible_tiles(
        &self,
        viewport_width: f32,
        viewport_height: f32,
        scale: f32,
        offset_x: f32,
        offset_y: f32,
    ) -> Vec<(usize, usize)> {
        if scale <= 0.0 {
            return Vec::new();
        }

        let tile_screen = TILE_SIZE as f32 * scale;

        // Top-left of raster in screen space.
        let x0 = offset_x;
        let y0 = offset_y;

        let col_start = ((-x0) / tile_screen).floor().max(0.0) as usize;
        let row_start = ((-y0) / tile_screen).floor().max(0.0) as usize;
        let col_end = ((viewport_width - x0) / tile_screen).ceil().max(0.0) as usize;
        let row_end = ((viewport_height - y0) / tile_screen).ceil().max(0.0) as usize;

        let tc = self.tile_cols();
        let tr = self.tile_rows();

        let mut out = Vec::new();
        for r in row_start..row_end.min(tr) {
            for c in col_start..col_end.min(tc) {
                out.push((c, r));
            }
        }
        out
    }

    /// Get or create the texture for the tile at `(col, row)`.
    ///
    /// `render_fn` is called to produce RGBA bytes for the tile when it's not
    /// in cache. It receives `(pixel_x, pixel_y, tile_w, tile_h)` — the origin
    /// within the full raster and the actual tile dimensions (may be smaller at edges).
    pub fn get_or_render(
        &mut self,
        col: usize,
        row: usize,
        ctx: &Context,
        render_fn: &dyn Fn(usize, usize, usize, usize) -> Vec<u8>,
    ) -> &TextureHandle {
        let key = TileKey {
            generation: self.generation,
            col,
            row,
        };

        // Touch or insert.
        if !self.cache.contains_key(&key) {
            self.evict_if_needed();

            let px = col * TILE_SIZE;
            let py = row * TILE_SIZE;
            let tw = TILE_SIZE.min(self.raster_cols.saturating_sub(px));
            let th = TILE_SIZE.min(self.raster_rows.saturating_sub(py));

            let rgba = render_fn(px, py, tw, th);
            let image = ColorImage::from_rgba_unmultiplied([tw, th], &rgba);
            let tex = ctx.load_texture(
                format!("tile_{}_{}", col, row),
                image,
                TextureOptions::NEAREST,
            );

            self.cache.insert(
                key,
                TileEntry {
                    texture: tex,
                    last_used: self.frame,
                },
            );
        } else {
            self.cache.get_mut(&key).unwrap().last_used = self.frame;
        }

        &self.cache[&key].texture
    }

    /// Evict the least-recently used tiles until we're under the limit.
    fn evict_if_needed(&mut self) {
        while self.cache.len() >= MAX_TILES {
            // Find the key with the oldest last_used.
            let oldest = self
                .cache
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| *k);
            if let Some(k) = oldest {
                self.cache.remove(&k);
            } else {
                break;
            }
        }
    }
}
