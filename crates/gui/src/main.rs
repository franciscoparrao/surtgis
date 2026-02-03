//! SurtGis Desktop GUI
//!
//! SAGA-like desktop application for geospatial analysis.

mod app;
mod dock;
mod executor;
mod io;
mod menu;
mod panels;
mod registry;
mod state;

use app::SurtGisApp;

fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SurtGis — Geospatial Analysis")
            .with_inner_size([1400.0, 900.0])
            .with_min_inner_size([800.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "SurtGis",
        native_options,
        Box::new(|cc| Ok(Box::new(SurtGisApp::new(cc)))),
    )
}
