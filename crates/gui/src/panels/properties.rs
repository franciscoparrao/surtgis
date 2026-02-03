//! Properties panel: 5 tabs for the active dataset.
//!
//! Tabs: Description | Settings | Legend | Statistics | Metadata

use egui::Ui;

use surtgis_colormap::ColorScheme;

use crate::state::workspace::{Dataset, DatasetRaster};

/// Actions from the properties panel.
pub enum PropertiesAction {
    ChangeColormap(ColorScheme),
    None,
}

/// Show the properties panel for the active dataset.
pub fn show_properties(ui: &mut Ui, dataset: Option<&Dataset>) -> PropertiesAction {
    let mut action = PropertiesAction::None;

    let Some(ds) = dataset else {
        ui.centered_and_justified(|ui| {
            ui.label("No dataset selected.");
        });
        return action;
    };

    ui.heading(&ds.name);
    ui.separator();

    // Tab bar
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // ── Description ────────────────────────────────────
            egui::CollapsingHeader::new("Description")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label(format!("Name: {}", ds.name));
                    ui.label(format!("Size: {} x {} pixels", ds.raster.cols(), ds.raster.rows()));
                    if let Some(ref path) = ds.source_path {
                        ui.label(format!("File: {}", path.display()));
                    }
                    if let Some(ref prov) = ds.provenance {
                        ui.label(format!("Algorithm: {}", prov));
                    }
                    let type_name = match &ds.raster {
                        DatasetRaster::F64(_) => "Float64",
                        DatasetRaster::U8(_) => "UInt8",
                        DatasetRaster::I32(_) => "Int32",
                    };
                    ui.label(format!("Type: {}", type_name));
                });

            // ── Settings ───────────────────────────────────────
            egui::CollapsingHeader::new("Settings")
                .default_open(true)
                .show(ui, |ui| {
                    ui.label("Colormap:");
                    let current = ds.colormap;
                    egui::ComboBox::from_id_salt("prop_colormap")
                        .selected_text(current.name())
                        .show_ui(ui, |ui| {
                            for &scheme in ColorScheme::ALL {
                                if ui.selectable_label(scheme == current, scheme.name()).clicked() {
                                    action = PropertiesAction::ChangeColormap(scheme);
                                }
                            }
                        });

                    ui.label(format!("Opacity: {:.0}%", ds.opacity * 100.0));
                    ui.label(format!("Visible: {}", if ds.visible { "Yes" } else { "No" }));
                });

            // ── Legend ──────────────────────────────────────────
            egui::CollapsingHeader::new("Legend")
                .default_open(false)
                .show(ui, |ui| {
                    // Draw a simple color bar
                    let (rect, _) = ui.allocate_exact_size(
                        egui::vec2(ui.available_width().min(200.0), 20.0),
                        egui::Sense::hover(),
                    );
                    let painter = ui.painter_at(rect);
                    let steps = 64;
                    let step_w = rect.width() / steps as f32;
                    for i in 0..steps {
                        let t = i as f64 / (steps - 1) as f64;
                        let c = surtgis_colormap::evaluate(ds.colormap, t);
                        let x = rect.left() + i as f32 * step_w;
                        painter.rect_filled(
                            egui::Rect::from_min_size(
                                egui::pos2(x, rect.top()),
                                egui::vec2(step_w + 1.0, rect.height()),
                            ),
                            0.0,
                            egui::Color32::from_rgb(c.r, c.g, c.b),
                        );
                    }
                    ui.label(format!("Scheme: {}", ds.colormap.name()));
                });

            // ── Statistics ─────────────────────────────────────
            egui::CollapsingHeader::new("Statistics")
                .default_open(false)
                .show(ui, |ui| {
                    match &ds.raster {
                        DatasetRaster::F64(r) => {
                            let stats = r.statistics();
                            if let Some(min) = stats.min {
                                ui.label(format!("Min: {:.6}", min));
                            }
                            if let Some(max) = stats.max {
                                ui.label(format!("Max: {:.6}", max));
                            }
                            if let Some(mean) = stats.mean {
                                ui.label(format!("Mean: {:.6}", mean));
                            }
                            ui.label(format!("Valid cells: {}", stats.valid_count));
                            ui.label(format!("NoData cells: {}", stats.nodata_count));
                        }
                        DatasetRaster::U8(r) => {
                            let stats = r.statistics();
                            ui.label(format!("Min: {:?}", stats.min));
                            ui.label(format!("Max: {:?}", stats.max));
                            ui.label(format!("Valid cells: {}", stats.valid_count));
                        }
                        DatasetRaster::I32(r) => {
                            let stats = r.statistics();
                            ui.label(format!("Min: {:?}", stats.min));
                            ui.label(format!("Max: {:?}", stats.max));
                            ui.label(format!("Valid cells: {}", stats.valid_count));
                        }
                    }
                });

            // ── Metadata ───────────────────────────────────────
            egui::CollapsingHeader::new("Metadata")
                .default_open(false)
                .show(ui, |ui| {
                    let (bounds, cell_size, crs_info) = match &ds.raster {
                        DatasetRaster::F64(r) => (r.bounds(), r.cell_size(), r.crs().map(|c| format!("{:?}", c))),
                        DatasetRaster::U8(r) => (r.bounds(), r.cell_size(), r.crs().map(|c| format!("{:?}", c))),
                        DatasetRaster::I32(r) => (r.bounds(), r.cell_size(), r.crs().map(|c| format!("{:?}", c))),
                    };
                    ui.label(format!("Cell size: {:.6}", cell_size));
                    ui.label(format!("Bounds: ({:.4}, {:.4}) - ({:.4}, {:.4})",
                        bounds.0, bounds.1, bounds.2, bounds.3));
                    if let Some(crs) = crs_info {
                        ui.label(format!("CRS: {}", crs));
                    } else {
                        ui.label("CRS: (none)");
                    }
                });
        });

    action
}

