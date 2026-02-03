//! 3D wireframe view of the active DEM using isometric projection
//! rendered via egui `Painter`.

use egui::{Color32, Pos2, Sense, Stroke, Ui};

use surtgis_colormap::evaluate;

use crate::state::workspace::{Dataset, DatasetRaster};

/// Persistent state for the 3D view.
pub struct View3dState {
    /// Camera azimuth in degrees.
    pub azimuth: f32,
    /// Camera elevation angle in degrees (0 = side, 90 = top).
    pub elevation: f32,
    /// Vertical exaggeration factor.
    pub z_exaggeration: f32,
    /// Subsample step (every N-th pixel).
    pub grid_step: usize,
    /// Is the user dragging to rotate?
    dragging: bool,
}

impl Default for View3dState {
    fn default() -> Self {
        Self {
            azimuth: 225.0,
            elevation: 35.0,
            z_exaggeration: 2.0,
            grid_step: 4,
            dragging: false,
        }
    }
}

/// Render the 3D wireframe view.
pub fn show_view_3d(ui: &mut Ui, dataset: Option<&Dataset>, state: &mut View3dState) {
    // Controls
    ui.horizontal(|ui| {
        ui.label("Azimuth:");
        ui.add(egui::Slider::new(&mut state.azimuth, 0.0..=360.0).suffix("°"));
    });
    ui.horizontal(|ui| {
        ui.label("Elevation:");
        ui.add(egui::Slider::new(&mut state.elevation, 5.0..=85.0).suffix("°"));
    });
    ui.horizontal(|ui| {
        ui.label("Z Exag:");
        ui.add(egui::Slider::new(&mut state.z_exaggeration, 0.1..=20.0));
    });
    ui.horizontal(|ui| {
        ui.label("Step:");
        ui.add(egui::Slider::new(&mut state.grid_step, 1..=32));
    });

    ui.separator();

    let Some(dataset) = dataset else {
        ui.centered_and_justified(|ui| {
            ui.label("No dataset loaded.");
        });
        return;
    };

    let raster_f64 = match &dataset.raster {
        DatasetRaster::F64(r) => r,
        _ => {
            ui.label("3D view requires an f64 raster (DEM).");
            return;
        }
    };

    let rows = raster_f64.rows();
    let cols = raster_f64.cols();
    if rows < 2 || cols < 2 {
        ui.label("Raster too small for 3D view.");
        return;
    }

    let available = ui.available_size();
    let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());

    // Background
    painter.rect_filled(response.rect, 0.0, Color32::from_gray(20));

    // Handle drag rotation
    if response.dragged() {
        let delta = response.drag_delta();
        state.azimuth = (state.azimuth + delta.x * 0.5) % 360.0;
        if state.azimuth < 0.0 {
            state.azimuth += 360.0;
        }
        state.elevation = (state.elevation - delta.y * 0.3).clamp(5.0, 85.0);
        state.dragging = true;
    } else {
        state.dragging = false;
    }

    // Compute elevation stats for normalisation.
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let step = state.grid_step.max(1);
    for r in (0..rows).step_by(step) {
        for c in (0..cols).step_by(step) {
            if let Ok(v) = raster_f64.get(r, c) {
                if v.is_finite() {
                    z_min = z_min.min(v);
                    z_max = z_max.max(v);
                }
            }
        }
    }

    if !z_min.is_finite() || !z_max.is_finite() || (z_max - z_min).abs() < 1e-12 {
        ui.painter().text(
            response.rect.center(),
            egui::Align2::CENTER_CENTER,
            "Flat or no-data raster",
            egui::FontId::proportional(14.0),
            Color32::GRAY,
        );
        return;
    }

    let z_range = z_max - z_min;

    // Projection parameters.
    let az_rad = (state.azimuth as f64).to_radians();
    let el_rad = (state.elevation as f64).to_radians();
    let cos_az = az_rad.cos();
    let sin_az = az_rad.sin();
    let cos_el = el_rad.cos();
    let sin_el = el_rad.sin();
    let z_exag = state.z_exaggeration as f64;

    // Grid dimensions in steps.
    let n_rows = (rows + step - 1) / step;
    let n_cols = (cols + step - 1) / step;

    // Project 3D → 2D.  x_raster = col, y_raster = row, z = elevation.
    // Normalise so the grid spans ~[-1, 1] in x/y.
    let scale_xy = 1.0 / (n_cols.max(n_rows) as f64);
    let offset_x = -(n_cols as f64) * 0.5 * scale_xy;
    let offset_y = -(n_rows as f64) * 0.5 * scale_xy;

    // Project one grid point to screen.
    let project = |gc: usize, gr: usize, z_norm: f64| -> Pos2 {
        let x3 = offset_x + gc as f64 * scale_xy;
        let y3 = offset_y + gr as f64 * scale_xy;
        let z3 = z_norm * z_exag * 0.5; // keep in ~ [-exag/2, exag/2]

        // Rotate around Z axis (azimuth), then tilt (elevation).
        let xr = x3 * cos_az - y3 * sin_az;
        let yr = x3 * sin_az + y3 * cos_az;
        let zr = z3;

        let x_screen = xr;
        let y_screen = -yr * sin_el - zr * cos_el;

        // Map to canvas.
        let cx = response.rect.center().x as f64;
        let cy = response.rect.center().y as f64;
        let canvas_scale = (available.x.min(available.y) * 0.42) as f64;

        Pos2::new(
            (cx + x_screen * canvas_scale) as f32,
            (cy + y_screen * canvas_scale) as f32,
        )
    };

    let colormap = dataset.colormap;

    // Draw grid lines (row-wise then column-wise).
    let get_z_norm = |r: usize, c: usize| -> Option<f64> {
        raster_f64.get(r, c).ok().and_then(|v| {
            if v.is_finite() {
                Some((v - z_min) / z_range)
            } else {
                None
            }
        })
    };

    let color_for = |t: f64| -> Color32 {
        let rgb = evaluate(colormap, t);
        Color32::from_rgb(rgb.r, rgb.g, rgb.b)
    };

    // Row-wise lines (along columns for each row).
    for gr in 0..n_rows {
        let rr = (gr * step).min(rows - 1);
        let mut prev: Option<(Pos2, f64)> = None;
        for gc in 0..n_cols {
            let cc = (gc * step).min(cols - 1);
            if let Some(t) = get_z_norm(rr, cc) {
                let p = project(gc, gr, t * 2.0 - 1.0);
                if let Some((pp, pt)) = prev {
                    let avg_t = (pt + t) * 0.5;
                    painter.line_segment([pp, p], Stroke::new(1.0, color_for(avg_t)));
                }
                prev = Some((p, t));
            } else {
                prev = None;
            }
        }
    }

    // Column-wise lines (along rows for each column).
    for gc in 0..n_cols {
        let cc = (gc * step).min(cols - 1);
        let mut prev: Option<(Pos2, f64)> = None;
        for gr in 0..n_rows {
            let rr = (gr * step).min(rows - 1);
            if let Some(t) = get_z_norm(rr, cc) {
                let p = project(gc, gr, t * 2.0 - 1.0);
                if let Some((pp, pt)) = prev {
                    let avg_t = (pt + t) * 0.5;
                    painter.line_segment([pp, p], Stroke::new(1.0, color_for(avg_t)));
                }
                prev = Some((p, t));
            } else {
                prev = None;
            }
        }
    }

    // Info label
    painter.text(
        Pos2::new(response.rect.left() + 8.0, response.rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "{}x{} step={} ({} lines)  Az={:.0}° El={:.0}° Zx{:.1}",
            cols,
            rows,
            step,
            n_rows * n_cols * 2, // rough line count
            state.azimuth,
            state.elevation,
            state.z_exaggeration,
        ),
        egui::FontId::monospace(10.0),
        Color32::LIGHT_GRAY,
    );
}
