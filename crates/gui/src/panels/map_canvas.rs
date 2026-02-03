//! Map canvas panel: displays the active raster with colormap, zoom, pan, and scale bar.

use egui::{Color32, Pos2, Rect, Sense, TextureHandle, Vec2};

use surtgis_colormap::{auto_params, raster_to_rgba};

use crate::state::workspace::{Dataset, DatasetRaster};

/// State for the map canvas (zoom/pan).
pub struct MapCanvasState {
    /// Zoom level (1.0 = fit to panel).
    pub zoom: f32,
    /// Pan offset in pixels.
    pub offset: Vec2,
    /// Is the user currently panning?
    dragging: bool,
    /// Last known cursor position in raster pixel coordinates.
    pub cursor_pixel: Option<(usize, usize)>,
    /// Last known cursor position in geographic coordinates.
    pub cursor_geo: Option<(f64, f64)>,
    /// Pixel value at cursor.
    pub cursor_value: Option<String>,
}

impl Default for MapCanvasState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            offset: Vec2::ZERO,
            dragging: false,
            cursor_pixel: None,
            cursor_geo: None,
            cursor_value: None,
        }
    }
}

/// Render the map canvas into the given UI area.
pub fn show_map_canvas(
    ui: &mut egui::Ui,
    dataset: Option<&mut Dataset>,
    texture: &mut Option<TextureHandle>,
    state: &mut MapCanvasState,
    ctx: &egui::Context,
) {
    let available = ui.available_size();

    let Some(dataset) = dataset else {
        ui.centered_and_justified(|ui| {
            ui.label("No dataset loaded. Use File > Open to load a GeoTIFF.");
        });
        return;
    };

    let (rows, cols) = (dataset.raster.rows(), dataset.raster.cols());
    if rows == 0 || cols == 0 {
        ui.label("Empty raster");
        return;
    }

    // Generate or reuse RGBA cache + texture
    ensure_texture(dataset, texture, ctx);

    let Some(tex) = texture.as_ref() else {
        ui.label("Rendering...");
        return;
    };

    // Compute display size respecting aspect ratio
    let aspect = cols as f32 / rows as f32;
    let fit_w = available.x;
    let fit_h = available.y - 24.0; // reserve space for status bar
    let base_scale = if fit_w / fit_h > aspect {
        fit_h / rows as f32
    } else {
        fit_w / cols as f32
    };
    let scale = base_scale * state.zoom;
    let img_size = Vec2::new(cols as f32 * scale, rows as f32 * scale);

    // Center the image
    let center = ui.available_rect_before_wrap().center();
    let img_origin = Pos2::new(
        center.x - img_size.x / 2.0 + state.offset.x,
        center.y - img_size.y / 2.0 + state.offset.y,
    );
    let img_rect = Rect::from_min_size(img_origin, img_size);

    // Interaction area
    let (response, painter) = ui.allocate_painter(
        Vec2::new(available.x, fit_h),
        Sense::click_and_drag(),
    );

    // Background
    painter.rect_filled(response.rect, 0.0, Color32::from_gray(30));

    // Draw the raster texture
    painter.image(
        tex.id(),
        img_rect,
        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
        Color32::WHITE,
    );

    // Handle zoom (mouse wheel) — zoom towards cursor
    let scroll = ui.input(|i| i.raw_scroll_delta.y);
    if scroll != 0.0 && response.hovered() {
        let factor = if scroll > 0.0 { 1.1 } else { 1.0 / 1.1 };
        let new_zoom = (state.zoom * factor).clamp(0.1, 50.0);
        // Zoom towards cursor position
        if let Some(cursor) = response.hover_pos() {
            let ratio = new_zoom / state.zoom;
            state.offset = cursor.to_vec2() - ratio * (cursor.to_vec2() - state.offset - center.to_vec2())
                - center.to_vec2();
            // Simplified: just adjust offset proportionally
            state.offset.x *= ratio;
            state.offset.y *= ratio;
        }
        state.zoom = new_zoom;
    }

    // Handle pan (drag)
    if response.dragged() {
        state.offset += response.drag_delta();
        state.dragging = true;
    } else {
        state.dragging = false;
    }

    // Cursor position tracking
    state.cursor_pixel = None;
    state.cursor_geo = None;
    state.cursor_value = None;

    if let Some(hover_pos) = response.hover_pos() {
        let rel = hover_pos - img_origin;
        let px = (rel.x / scale) as isize;
        let py = (rel.y / scale) as isize;

        if px >= 0 && py >= 0 && (px as usize) < cols && (py as usize) < rows {
            let col = px as usize;
            let row = py as usize;
            state.cursor_pixel = Some((col, row));

            let (geo_x, geo_y) = dataset.raster.pixel_to_geo(col, row);
            state.cursor_geo = Some((geo_x, geo_y));

            // Read pixel value
            state.cursor_value = Some(read_pixel_value(&dataset.raster, row, col));
        }
    }

    // ── Scale bar ─────────────────────────────────────────────────────
    draw_scale_bar(&painter, &response.rect, &dataset.raster, scale);

    // Status bar at bottom
    let status_rect = Rect::from_min_size(
        Pos2::new(response.rect.left(), response.rect.bottom() - 20.0),
        Vec2::new(response.rect.width(), 20.0),
    );
    painter.rect_filled(status_rect, 0.0, Color32::from_gray(40));

    let status_text = if let (Some((gx, gy)), Some(val)) =
        (state.cursor_geo, &state.cursor_value)
    {
        format!(
            "X: {:.6}  Y: {:.6}  |  Value: {}  |  Zoom: {:.0}%  |  {}x{}",
            gx,
            gy,
            val,
            state.zoom * 100.0,
            cols,
            rows,
        )
    } else {
        format!(
            "{}x{} pixels  |  Zoom: {:.0}%  |  Colormap: {}",
            cols,
            rows,
            state.zoom * 100.0,
            dataset.colormap.name(),
        )
    };

    painter.text(
        status_rect.center(),
        egui::Align2::CENTER_CENTER,
        status_text,
        egui::FontId::monospace(11.0),
        Color32::LIGHT_GRAY,
    );
}

/// Draw a scale bar in the bottom-left corner of the map canvas.
fn draw_scale_bar(
    painter: &egui::Painter,
    canvas_rect: &Rect,
    raster: &DatasetRaster,
    pixel_scale: f32,
) {
    // Get the cell size (geographic units per pixel)
    let cell_size = match raster {
        DatasetRaster::F64(r) => r.cell_size(),
        DatasetRaster::U8(r) => r.cell_size(),
        DatasetRaster::I32(r) => r.cell_size(),
    };
    if cell_size <= 0.0 || pixel_scale <= 0.0 {
        return;
    }

    // Geo units per screen pixel
    let geo_per_px = cell_size / pixel_scale as f64;

    // Target bar width ~100-200 screen pixels
    let target_screen_px = 150.0_f64;
    let raw_geo = geo_per_px * target_screen_px;

    // Round to a nice number
    let magnitude = 10.0_f64.powf(raw_geo.log10().floor());
    let nice = if raw_geo / magnitude < 2.0 {
        magnitude
    } else if raw_geo / magnitude < 5.0 {
        2.0 * magnitude
    } else {
        5.0 * magnitude
    };

    let bar_screen_px = (nice / geo_per_px) as f32;
    if bar_screen_px < 20.0 || bar_screen_px > 400.0 {
        return;
    }

    // Format the label
    let label = if nice >= 1000.0 {
        format!("{:.0} km", nice / 1000.0)
    } else if nice >= 1.0 {
        format!("{:.0} m", nice)
    } else if nice >= 0.01 {
        format!("{:.2} m", nice)
    } else {
        // Probably degrees
        if nice >= 1.0 / 3600.0 {
            format!("{:.1}\"", nice * 3600.0)
        } else {
            format!("{:.6}\u{00b0}", nice)
        }
    };

    let bar_y = canvas_rect.bottom() - 40.0;
    let bar_x = canvas_rect.left() + 16.0;
    let bar_height = 6.0_f32;

    // Background
    let bg_rect = Rect::from_min_size(
        Pos2::new(bar_x - 4.0, bar_y - 16.0),
        Vec2::new(bar_screen_px + 8.0, bar_height + 24.0),
    );
    painter.rect_filled(bg_rect, 3.0, Color32::from_black_alpha(140));

    // Bar
    let bar_rect = Rect::from_min_size(
        Pos2::new(bar_x, bar_y),
        Vec2::new(bar_screen_px, bar_height),
    );
    painter.rect_filled(bar_rect, 1.0, Color32::WHITE);

    // End ticks
    let tick_h = 10.0_f32;
    painter.line_segment(
        [
            Pos2::new(bar_x, bar_y - tick_h / 2.0),
            Pos2::new(bar_x, bar_y + bar_height + tick_h / 2.0),
        ],
        egui::Stroke::new(1.5, Color32::WHITE),
    );
    painter.line_segment(
        [
            Pos2::new(bar_x + bar_screen_px, bar_y - tick_h / 2.0),
            Pos2::new(bar_x + bar_screen_px, bar_y + bar_height + tick_h / 2.0),
        ],
        egui::Stroke::new(1.5, Color32::WHITE),
    );

    // Label
    painter.text(
        Pos2::new(bar_x + bar_screen_px / 2.0, bar_y - 10.0),
        egui::Align2::CENTER_BOTTOM,
        label,
        egui::FontId::proportional(11.0),
        Color32::WHITE,
    );
}

/// Ensure the RGBA cache and texture are up to date.
fn ensure_texture(
    dataset: &mut Dataset,
    texture: &mut Option<TextureHandle>,
    ctx: &egui::Context,
) {
    if dataset.rgba_cache.is_none() {
        let rgba = match &dataset.raster {
            DatasetRaster::F64(r) => {
                let params = auto_params(r, dataset.colormap);
                raster_to_rgba(r, &params)
            }
            DatasetRaster::U8(r) => {
                let params = auto_params(r, dataset.colormap);
                raster_to_rgba(r, &params)
            }
            DatasetRaster::I32(r) => {
                let params = auto_params(r, dataset.colormap);
                raster_to_rgba(r, &params)
            }
        };
        dataset.rgba_cache = Some(rgba);
    }

    if texture.is_none() {
        if let Some(rgba) = &dataset.rgba_cache {
            let (rows, cols) = (dataset.raster.rows(), dataset.raster.cols());
            let image = egui::ColorImage::from_rgba_unmultiplied([cols, rows], rgba);
            *texture = Some(ctx.load_texture(
                &dataset.name,
                image,
                egui::TextureOptions::NEAREST,
            ));
        }
    }
}

fn read_pixel_value(raster: &DatasetRaster, row: usize, col: usize) -> String {
    match raster {
        DatasetRaster::F64(r) => match r.get(row, col) {
            Ok(v) => {
                if v.is_nan() {
                    "NoData".to_string()
                } else {
                    format!("{:.4}", v)
                }
            }
            Err(_) => "—".to_string(),
        },
        DatasetRaster::U8(r) => match r.get(row, col) {
            Ok(v) => format!("{}", v),
            Err(_) => "—".to_string(),
        },
        DatasetRaster::I32(r) => match r.get(row, col) {
            Ok(v) => format!("{}", v),
            Err(_) => "—".to_string(),
        },
    }
}

/// Invalidate the texture cache (call when colormap or data changes).
pub fn invalidate_texture(dataset: &mut Dataset, texture: &mut Option<TextureHandle>) {
    dataset.rgba_cache = None;
    *texture = None;
}
