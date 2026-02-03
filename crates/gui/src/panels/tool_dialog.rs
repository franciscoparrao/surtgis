//! Tool dialog: auto-generated UI from algorithm ParamDef.

use std::collections::HashMap;

use egui::Ui;

use crate::registry::{AlgorithmEntry, ParamKind, ParamValue};
use crate::state::DatasetId;

/// State for the tool dialog.
pub struct ToolDialogState {
    /// The algorithm being configured (None = no dialog open).
    pub algo_id: Option<String>,
    /// Current parameter values.
    pub values: HashMap<String, ParamValue>,
}

impl Default for ToolDialogState {
    fn default() -> Self {
        Self {
            algo_id: None,
            values: HashMap::new(),
        }
    }
}

impl ToolDialogState {
    /// Open the dialog for an algorithm, initializing params to defaults.
    pub fn open(&mut self, algo: &AlgorithmEntry) {
        self.algo_id = Some(algo.id.to_string());
        self.values.clear();

        for param in &algo.params {
            let value = match &param.kind {
                ParamKind::Float { default, .. } => ParamValue::Float(*default),
                ParamKind::Int { default, .. } => ParamValue::Int(*default),
                ParamKind::Bool { default } => ParamValue::Bool(*default),
                ParamKind::Choice { default, .. } => ParamValue::Choice(*default),
                ParamKind::InputRaster => ParamValue::InputRaster(None),
            };
            self.values.insert(param.name.to_string(), value);
        }
    }

    pub fn close(&mut self) {
        self.algo_id = None;
        self.values.clear();
    }
}

/// Result of showing the tool dialog.
pub enum ToolDialogAction {
    /// User clicked Run.
    Run,
    /// User clicked Cancel.
    Cancel,
    /// Dialog is still open, no action.
    None,
}

/// Show the tool dialog for the currently selected algorithm.
/// Returns the action taken.
pub fn show_tool_dialog(
    ui: &mut Ui,
    state: &mut ToolDialogState,
    registry: &[AlgorithmEntry],
    dataset_names: &[(DatasetId, String)],
) -> ToolDialogAction {
    let Some(algo_id) = &state.algo_id else {
        return ToolDialogAction::None;
    };

    let Some(algo) = registry.iter().find(|a| a.id == algo_id.as_str()) else {
        state.close();
        return ToolDialogAction::None;
    };

    let mut action = ToolDialogAction::None;

    ui.heading(algo.name);
    ui.label(algo.description);
    ui.separator();

    // Render each parameter
    for param in &algo.params {
        let Some(value) = state.values.get_mut(param.name) else {
            continue;
        };

        ui.horizontal(|ui| {
            ui.label(param.label);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                match (&param.kind, value) {
                    (
                        ParamKind::Float {
                            min, max, speed, ..
                        },
                        ParamValue::Float(v),
                    ) => {
                        ui.add(
                            egui::DragValue::new(v)
                                .range(*min..=*max)
                                .speed(*speed),
                        );
                    }
                    (ParamKind::Int { min, max, .. }, ParamValue::Int(v)) => {
                        ui.add(egui::DragValue::new(v).range(*min..=*max));
                    }
                    (ParamKind::Bool { .. }, ParamValue::Bool(v)) => {
                        ui.checkbox(v, "");
                    }
                    (ParamKind::Choice { options, .. }, ParamValue::Choice(idx)) => {
                        let current = options.get(*idx).unwrap_or(&"?");
                        egui::ComboBox::from_id_salt(param.name)
                            .selected_text(*current)
                            .show_ui(ui, |ui| {
                                for (i, opt) in options.iter().enumerate() {
                                    ui.selectable_value(idx, i, *opt);
                                }
                            });
                    }
                    (ParamKind::InputRaster, ParamValue::InputRaster(selected_id)) => {
                        let current_name = selected_id
                            .and_then(|id| {
                                dataset_names.iter().find(|(did, _)| *did == id)
                            })
                            .map(|(_, name)| name.as_str())
                            .unwrap_or("(select)");

                        egui::ComboBox::from_id_salt(param.name)
                            .selected_text(current_name)
                            .show_ui(ui, |ui| {
                                for (did, name) in dataset_names {
                                    if ui
                                        .selectable_label(
                                            *selected_id == Some(*did),
                                            name,
                                        )
                                        .clicked()
                                    {
                                        *selected_id = Some(*did);
                                    }
                                }
                            });
                    }
                    _ => {}
                }
            });
        });
    }

    ui.separator();
    ui.horizontal(|ui| {
        if ui.button("Run").clicked() {
            action = ToolDialogAction::Run;
        }
        if ui.button("Cancel").clicked() {
            action = ToolDialogAction::Cancel;
        }
    });

    action
}
