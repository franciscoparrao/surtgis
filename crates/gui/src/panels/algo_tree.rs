//! Algorithm tree panel: browse algorithms by category.

use egui::Ui;

use crate::registry::{AlgoCategory, AlgorithmEntry};

/// Show the algorithm tree. Returns the selected algorithm ID if one was clicked.
pub fn show_algo_tree(
    ui: &mut Ui,
    registry: &[AlgorithmEntry],
    selected: &mut Option<String>,
) -> Option<String> {
    let mut clicked = None;

    ui.heading("Algorithms");
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for &category in AlgoCategory::ALL {
                let algos: Vec<&AlgorithmEntry> = registry
                    .iter()
                    .filter(|a| a.category == category)
                    .collect();

                if algos.is_empty() {
                    continue;
                }

                egui::CollapsingHeader::new(category.name())
                    .default_open(true)
                    .show(ui, |ui| {
                        for algo in &algos {
                            let is_selected = selected
                                .as_ref()
                                .map(|s| s == algo.id)
                                .unwrap_or(false);

                            let response = ui.selectable_label(is_selected, algo.name);

                            if response.clicked() {
                                *selected = Some(algo.id.to_string());
                                clicked = Some(algo.id.to_string());
                            }

                            response.on_hover_text(algo.description);
                        }
                    });
            }
        });

    clicked
}
