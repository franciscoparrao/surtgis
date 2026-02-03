//! Menu bar: File, View, Tools, Help.

use egui::Ui;

use surtgis_colormap::ColorScheme;

/// Actions triggered by menu items.
pub enum MenuAction {
    Open,
    Save,
    Exit,
    ChangeColormap(ColorScheme),
    ZoomToFit,
    About,
    None,
}

/// Show the main menu bar. Returns the action triggered (if any).
pub fn show_menu_bar(ui: &mut Ui, current_colormap: ColorScheme) -> MenuAction {
    let mut action = MenuAction::None;

    egui::menu::bar(ui, |ui| {
        ui.menu_button("File", |ui| {
            if ui.button("Open GeoTIFF...").clicked() {
                action = MenuAction::Open;
                ui.close_menu();
            }
            if ui.button("Save As...").clicked() {
                action = MenuAction::Save;
                ui.close_menu();
            }
            ui.separator();
            if ui.button("Exit").clicked() {
                action = MenuAction::Exit;
                ui.close_menu();
            }
        });

        ui.menu_button("View", |ui| {
            if ui.button("Zoom to Fit").clicked() {
                action = MenuAction::ZoomToFit;
                ui.close_menu();
            }
            ui.separator();
            ui.menu_button("Colormap", |ui| {
                for &scheme in ColorScheme::ALL {
                    let is_current = scheme == current_colormap;
                    if ui.selectable_label(is_current, scheme.name()).clicked() {
                        action = MenuAction::ChangeColormap(scheme);
                        ui.close_menu();
                    }
                }
            });
        });

        ui.menu_button("Help", |ui| {
            if ui.button("About SurtGis").clicked() {
                action = MenuAction::About;
                ui.close_menu();
            }
        });
    });

    action
}
