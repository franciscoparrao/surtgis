//! Menu bar: File, View, Tools, Help.

use egui::Ui;

use surtgis_colormap::ColorScheme;

use crate::registry::{AlgoCategory, AlgorithmEntry, algorithms_by_category};
use crate::render::MapMode;

/// Actions triggered by menu items.
pub enum MenuAction {
    Open,
    Save,
    Exit,
    ChangeColormap(ColorScheme),
    ZoomToFit,
    ChangeMapMode(MapMode),
    About,
    /// Launch an algorithm by its ID.
    RunAlgorithm(String),
    None,
}

/// Show the main menu bar. Returns the action triggered (if any).
pub fn show_menu_bar(
    ui: &mut Ui,
    current_colormap: ColorScheme,
    current_map_mode: MapMode,
    registry: &[AlgorithmEntry],
) -> MenuAction {
    let mut action = MenuAction::None;

    egui::MenuBar::new().ui(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui.button("Open GeoTIFF...").clicked() {
                action = MenuAction::Open;
                ui.close();
            }
            if ui.button("Save As...").clicked() {
                action = MenuAction::Save;
                ui.close();
            }
            ui.separator();
            if ui.button("Exit").clicked() {
                action = MenuAction::Exit;
                ui.close();
            }
        });

        ui.menu_button("View", |ui| {
            if ui.button("Zoom to Fit").clicked() {
                action = MenuAction::ZoomToFit;
                ui.close();
            }
            ui.separator();
            ui.menu_button("Map Mode", |ui| {
                if ui.selectable_label(current_map_mode == MapMode::Simple, "Simple").clicked() {
                    action = MenuAction::ChangeMapMode(MapMode::Simple);
                    ui.close();
                }
                if ui.selectable_label(current_map_mode == MapMode::Basemap, "Basemap (OSM)").clicked() {
                    action = MenuAction::ChangeMapMode(MapMode::Basemap);
                    ui.close();
                }
            });
            ui.separator();
            ui.menu_button("Colormap", |ui| {
                for &scheme in ColorScheme::ALL {
                    let is_current = scheme == current_colormap;
                    if ui.selectable_label(is_current, scheme.name()).clicked() {
                        action = MenuAction::ChangeColormap(scheme);
                        ui.close();
                    }
                }
            });
        });

        // Tools menu auto-generated from registry categories
        ui.menu_button("Tools", |ui| {
            for &category in AlgoCategory::ALL {
                let algos = algorithms_by_category(registry, category);
                if algos.is_empty() {
                    continue;
                }
                ui.menu_button(category.name(), |ui| {
                    for algo in algos {
                        if ui.button(algo.name).clicked() {
                            action = MenuAction::RunAlgorithm(algo.id.to_string());
                            ui.close();
                        }
                    }
                });
            }
        });

        ui.menu_button("Help", |ui| {
            if ui.button("About SurtGis").clicked() {
                action = MenuAction::About;
                ui.close();
            }
        });
    });

    action
}
