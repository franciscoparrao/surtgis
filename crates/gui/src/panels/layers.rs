//! Layer stack panel: visibility, opacity, draw order.

use egui::Ui;

use crate::state::workspace::Workspace;
use crate::state::DatasetId;

/// Actions returned from the layers panel.
pub enum LayerAction {
    /// Toggle visibility of a layer.
    ToggleVisibility(DatasetId),
    /// Change opacity of a layer.
    ChangeOpacity(DatasetId, f32),
    /// Select a layer as active.
    Select(DatasetId),
    /// Remove a dataset.
    Remove(DatasetId),
    /// No action.
    None,
}

/// Show the layer stack panel.
pub fn show_layers(ui: &mut Ui, workspace: &Workspace) -> LayerAction {
    let mut action = LayerAction::None;

    ui.heading("Layers");
    ui.separator();

    if workspace.len() == 0 {
        ui.label("No layers.");
        return action;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let active = workspace.active_dataset;

            // Display layers in reverse order (top layer first)
            let order: Vec<DatasetId> = workspace.layer_order().iter().copied().rev().collect();

            for id in order {
                let Some(ds) = workspace.get(id) else { continue };

                let is_active = active == Some(id);

                ui.horizontal(|ui| {
                    // Visibility checkbox
                    let mut vis = ds.visible;
                    if ui.checkbox(&mut vis, "").changed() {
                        action = LayerAction::ToggleVisibility(id);
                    }

                    // Name (selectable)
                    let response = ui.selectable_label(is_active, &ds.name);
                    if response.clicked() {
                        action = LayerAction::Select(id);
                    }

                    // Opacity slider (compact)
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let mut opacity = ds.opacity;
                        let slider = egui::Slider::new(&mut opacity, 0.0..=1.0)
                            .show_value(false)
                            .custom_formatter(|v, _| format!("{:.0}%", v * 100.0));
                        if ui.add_sized([60.0, 16.0], slider).changed() {
                            action = LayerAction::ChangeOpacity(id, opacity);
                        }
                    });
                });
            }
        });

    // Remove button for active dataset
    if let Some(active_id) = workspace.active_dataset {
        ui.separator();
        if ui.button("Remove Selected").clicked() {
            action = LayerAction::Remove(active_id);
        }
    }

    action
}
