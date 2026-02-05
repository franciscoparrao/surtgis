pub mod map_tiles;
pub mod tiled_renderer;

/// Map visualization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MapMode {
    /// Direct raster rendering (original map_canvas path).
    #[default]
    Simple,
    /// Slippy map basemap with raster overlay (walkers).
    Basemap,
}
