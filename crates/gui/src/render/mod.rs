pub mod map_tiles;
pub mod tiled_renderer;

/// Map visualization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapMode {
    /// Direct raster rendering (original map_canvas path).
    Simple,
    /// Slippy map basemap with raster overlay (walkers).
    Basemap,
}

impl Default for MapMode {
    fn default() -> Self {
        Self::Simple
    }
}
