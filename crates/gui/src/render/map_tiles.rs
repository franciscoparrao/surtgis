//! Basemap rendering using walkers (OpenStreetMap slippy tiles) with raster overlay.

use egui::{Color32, Rect, TextureHandle, Ui};
use walkers::sources::OpenStreetMap;
use walkers::{HttpTiles, Map, MapMemory, Plugin, Position, Projector, lon_lat};

/// Persistent basemap state (survives between frames).
pub struct BasemapState {
    pub tiles: HttpTiles,
    pub memory: MapMemory,
    /// Map center position (lon, lat).
    pub center: Position,
}

impl BasemapState {
    /// Create a new basemap state centred at the given WGS-84 lon/lat.
    pub fn new(ctx: &egui::Context, lon: f64, lat: f64) -> Self {
        Self {
            tiles: HttpTiles::new(OpenStreetMap, ctx.clone()),
            memory: MapMemory::default(),
            center: lon_lat(lon, lat),
        }
    }

    /// Re-centre the map on the given WGS-84 lon/lat.
    pub fn set_center(&mut self, lon: f64, lat: f64) {
        self.center = lon_lat(lon, lat);
    }
}

/// Plugin that draws a raster texture as an overlay on top of the basemap tiles.
struct RasterOverlay<'a> {
    texture: &'a TextureHandle,
    /// Raster geographic bounds in WGS-84: (west, south, east, north).
    bounds: (f64, f64, f64, f64),
    opacity: f32,
}

impl<'a> Plugin for RasterOverlay<'a> {
    fn run(
        self: Box<Self>,
        ui: &mut Ui,
        _response: &egui::Response,
        projector: &Projector,
        _memory: &MapMemory,
    ) {
        let (west, south, east, north) = self.bounds;

        // Project the four corners to screen coordinates.
        let nw = projector.project(lon_lat(west, north));
        let se = projector.project(lon_lat(east, south));

        let screen_rect = Rect::from_min_max(
            egui::pos2(nw.x, nw.y),
            egui::pos2(se.x, se.y),
        );

        // Draw the raster texture with the given opacity.
        let tint = Color32::from_white_alpha((self.opacity * 255.0) as u8);
        let painter = ui.painter();
        painter.image(
            self.texture.id(),
            screen_rect,
            Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            tint,
        );
    }
}

/// Render the basemap with an optional raster overlay.
///
/// * `texture` — pre-built RGBA texture of the active raster (same as `map_canvas` path).
/// * `bounds` — geographic extent of the raster in WGS-84 (west, south, east, north).
/// * `opacity` — overlay opacity \[0.0, 1.0\].
pub fn show_basemap(
    ui: &mut Ui,
    state: &mut BasemapState,
    texture: Option<&TextureHandle>,
    bounds: Option<(f64, f64, f64, f64)>,
    opacity: f32,
) {
    let mut map = Map::new(Some(&mut state.tiles), &mut state.memory, state.center);

    if let (Some(tex), Some(b)) = (texture, bounds) {
        map = map.with_plugin(RasterOverlay {
            texture: tex,
            bounds: b,
            opacity,
        });
    }

    ui.add(map);
}
