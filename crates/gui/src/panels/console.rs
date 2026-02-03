//! Console panel: log messages with colored levels.

use egui::{Color32, RichText, ScrollArea, Ui};

use crate::state::{LogEntry, LogLevel};

/// Show the console panel with log messages.
pub fn show_console(ui: &mut Ui, logs: &[LogEntry]) {
    ui.horizontal(|ui| {
        ui.heading("Console");
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(format!("{} messages", logs.len()));
        });
    });
    ui.separator();

    ScrollArea::vertical()
        .auto_shrink([false, false])
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for entry in logs {
                let (prefix, color) = match entry.level {
                    LogLevel::Info => ("[INFO]", Color32::from_rgb(150, 180, 220)),
                    LogLevel::Warning => ("[WARN]", Color32::from_rgb(230, 180, 50)),
                    LogLevel::Error => ("[ERROR]", Color32::from_rgb(220, 60, 60)),
                    LogLevel::Success => ("[OK]", Color32::from_rgb(60, 200, 80)),
                };

                let time = entry
                    .timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                let secs = time.as_secs() % 86400;
                let h = secs / 3600;
                let m = (secs % 3600) / 60;
                let s = secs % 60;

                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("{:02}:{:02}:{:02}", h, m, s))
                            .color(Color32::GRAY)
                            .monospace()
                            .size(11.0),
                    );
                    ui.label(
                        RichText::new(prefix)
                            .color(color)
                            .monospace()
                            .size(11.0),
                    );
                    ui.label(
                        RichText::new(&entry.message)
                            .monospace()
                            .size(11.0),
                    );
                });
            }
        });
}
