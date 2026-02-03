//! Data manager panel: tree of datasets grouped by grid system.

use egui::Ui;

use crate::state::workspace::{DatasetRaster, Workspace};
use crate::state::DatasetId;

/// Show the data manager panel. Returns the dataset ID if one was clicked.
pub fn show_data_manager(ui: &mut Ui, workspace: &Workspace) -> Option<DatasetId> {
    let mut clicked = None;

    ui.heading("Data");
    ui.separator();

    if workspace.len() == 0 {
        ui.label("No datasets loaded.");
        return None;
    }

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let active = workspace.active_dataset;

            for ds in workspace.datasets_ordered() {
                let is_active = active == Some(ds.id);
                let icon = match &ds.raster {
                    DatasetRaster::F64(_) => "\u{1f4ca}",
                    DatasetRaster::U8(_) => "\u{1f3f7}",
                    DatasetRaster::I32(_) => "#",
                };

                let dims = format!("{}x{}", ds.raster.cols(), ds.raster.rows());
                let label = format!("{} {} ({})", icon, ds.name, dims);

                let response = ui.selectable_label(is_active, &label);

                if response.clicked() {
                    clicked = Some(ds.id);
                }

                // Tooltip with details
                response.on_hover_ui(|ui| {
                    ui.label(format!("Name: {}", ds.name));
                    ui.label(format!("Size: {}x{}", ds.raster.cols(), ds.raster.rows()));
                    ui.label(format!("Colormap: {}", ds.colormap.name()));
                    if let Some(ref prov) = ds.provenance {
                        ui.label(format!("Source: {}", prov));
                    }
                    if let Some(ref path) = ds.source_path {
                        ui.label(format!("Path: {}", path.display()));
                    }
                });
            }
        });

    clicked
}
