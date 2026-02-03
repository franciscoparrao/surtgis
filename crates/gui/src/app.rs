//! Main application: SurtGisApp implements eframe::App.

use std::collections::HashMap;

use crossbeam_channel::{Receiver, Sender};
use egui::TextureHandle;
use egui_dock::{DockArea, DockState, Style, TabViewer};

use surtgis_colormap::ColorScheme;
use surtgis_core::raster::Raster;

use crate::dock::{create_dock_state, PanelId};
use crate::executor::dispatch_algorithm;
use crate::io;
use crate::menu::{show_menu_bar, MenuAction};
use crate::panels::algo_tree::show_algo_tree;
use crate::panels::console::show_console;
use crate::panels::map_canvas::{invalidate_texture, show_map_canvas, MapCanvasState};
use crate::panels::tool_dialog::{show_tool_dialog, ToolDialogAction, ToolDialogState};
use crate::registry::{build_registry, AlgorithmEntry, ParamValue};
use crate::state::workspace::{Dataset, DatasetRaster, Workspace};
use crate::state::{AppMessage, DatasetId, LogEntry};

/// The main application state.
pub struct SurtGisApp {
    /// Dock state for panel layout.
    dock_state: DockState<PanelId>,

    /// Message channels for background thread communication.
    tx: Sender<AppMessage>,
    rx: Receiver<AppMessage>,

    /// The workspace (loaded datasets).
    workspace: Workspace,

    /// Algorithm registry.
    registry: Vec<AlgorithmEntry>,

    /// Console log entries.
    logs: Vec<LogEntry>,

    /// Currently selected algorithm in the tree.
    selected_algo: Option<String>,

    /// Tool dialog state.
    tool_dialog: ToolDialogState,

    /// Map canvas state (zoom/pan).
    map_state: MapCanvasState,

    /// Active texture handle for the map canvas.
    map_texture: Option<TextureHandle>,

    /// Whether an algorithm is currently running.
    running: bool,

    /// Show about dialog.
    show_about: bool,
}

impl SurtGisApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Configure dark theme with custom visuals
        let mut visuals = egui::Visuals::dark();
        visuals.window_shadow = egui::epaint::Shadow::NONE;
        cc.egui_ctx.set_visuals(visuals);

        let (tx, rx) = crossbeam_channel::unbounded();

        let mut app = Self {
            dock_state: create_dock_state(),
            tx,
            rx,
            workspace: Workspace::new(),
            registry: build_registry(),
            logs: Vec::new(),
            selected_algo: None,
            tool_dialog: ToolDialogState::default(),
            map_state: MapCanvasState::default(),
            map_texture: None,
            running: false,
            show_about: false,
        };

        app.logs
            .push(LogEntry::info("SurtGis Desktop GUI started"));
        app.logs.push(LogEntry::info(format!(
            "{} algorithms available",
            app.registry.len()
        )));

        app
    }

    /// Process pending messages from background threads.
    fn process_messages(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                AppMessage::RasterLoaded { path, raster } => {
                    let name = path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string();

                    let dataset = Dataset {
                        id: DatasetId(0), // Will be set by workspace.add_dataset
                        name,
                        source_path: Some(path),
                        raster: DatasetRaster::F64(raster),
                        colormap: ColorScheme::Terrain,
                        visible: true,
                        rgba_cache: None,
                        provenance: None,
                    };

                    self.workspace.add_dataset(dataset);
                    // Invalidate texture so the new dataset gets rendered
                    self.map_texture = None;
                }

                AppMessage::AlgoComplete {
                    name,
                    result,
                    elapsed: _,
                } => {
                    self.running = false;
                    let colormap = suggest_colormap(&name);
                    let dataset = Dataset {
                        id: DatasetId(0),
                        name,
                        source_path: None,
                        raster: DatasetRaster::F64(result),
                        colormap,
                        visible: true,
                        rgba_cache: None,
                        provenance: Some(
                            self.tool_dialog
                                .algo_id
                                .clone()
                                .unwrap_or_default(),
                        ),
                    };
                    self.workspace.add_dataset(dataset);
                    self.map_texture = None;
                }

                AppMessage::AlgoCompleteU8 {
                    name,
                    result,
                    elapsed: _,
                } => {
                    self.running = false;
                    let colormap = suggest_colormap(&name);
                    let dataset = Dataset {
                        id: DatasetId(0),
                        name,
                        source_path: None,
                        raster: DatasetRaster::U8(result),
                        colormap,
                        visible: true,
                        rgba_cache: None,
                        provenance: Some(
                            self.tool_dialog
                                .algo_id
                                .clone()
                                .unwrap_or_default(),
                        ),
                    };
                    self.workspace.add_dataset(dataset);
                    self.map_texture = None;
                }

                AppMessage::AlgoCompleteI32 {
                    name,
                    result,
                    elapsed: _,
                } => {
                    self.running = false;
                    let dataset = Dataset {
                        id: DatasetId(0),
                        name,
                        source_path: None,
                        raster: DatasetRaster::I32(result),
                        colormap: ColorScheme::Terrain,
                        visible: true,
                        rgba_cache: None,
                        provenance: Some(
                            self.tool_dialog
                                .algo_id
                                .clone()
                                .unwrap_or_default(),
                        ),
                    };
                    self.workspace.add_dataset(dataset);
                    self.map_texture = None;
                }

                AppMessage::Error { context, message } => {
                    self.running = false;
                    self.logs
                        .push(LogEntry::error(format!("{}: {}", context, message)));
                }

                AppMessage::Log(entry) => {
                    self.logs.push(entry);
                }

                AppMessage::RasterSaved { path, .. } => {
                    self.logs
                        .push(LogEntry::success(format!("Saved to {}", path.display())));
                }
            }
        }
    }

    /// Execute the algorithm configured in the tool dialog.
    fn run_algorithm(&mut self) {
        let Some(algo_id) = &self.tool_dialog.algo_id else {
            return;
        };

        // Get the active dataset as primary input
        let Some(active) = self.workspace.active() else {
            self.logs
                .push(LogEntry::error("No active dataset to process"));
            return;
        };

        // Clone the primary input raster (must be f64 for most algorithms)
        let input = match &active.raster {
            DatasetRaster::F64(r) => r.clone(),
            DatasetRaster::U8(r) => {
                // Convert u8 to f64 for processing
                let mut result = Raster::<f64>::new(r.rows(), r.cols());
                result.set_transform(r.transform().clone());
                result.set_crs(r.crs().cloned());
                for row in 0..r.rows() {
                    for col in 0..r.cols() {
                        if let Ok(v) = r.get(row, col) {
                            let _ = result.set(row, col, v as f64);
                        }
                    }
                }
                result
            }
            DatasetRaster::I32(r) => {
                let mut result = Raster::<f64>::new(r.rows(), r.cols());
                result.set_transform(r.transform().clone());
                result.set_crs(r.crs().cloned());
                for row in 0..r.rows() {
                    for col in 0..r.cols() {
                        if let Ok(v) = r.get(row, col) {
                            let _ = result.set(row, col, v as f64);
                        }
                    }
                }
                result
            }
        };

        // Collect extra inputs for multi-input algorithms
        let mut extra_inputs: HashMap<String, Raster<f64>> = HashMap::new();
        for (param_name, param_value) in &self.tool_dialog.values {
            if let ParamValue::InputRaster(Some(dataset_id)) = param_value {
                if let Some(ds) = self.workspace.get(*dataset_id) {
                    if let DatasetRaster::F64(r) = &ds.raster {
                        extra_inputs.insert(param_name.clone(), r.clone());
                    }
                }
            }
        }

        self.running = true;
        dispatch_algorithm(
            algo_id,
            input,
            &self.tool_dialog.values,
            extra_inputs,
            self.tx.clone(),
        );
    }
}

impl eframe::App for SurtGisApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Process pending messages
        self.process_messages();

        // Request repaint while algorithm is running (to show log updates)
        if self.running {
            ctx.request_repaint();
        }

        // Menu bar
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            let colormap = self
                .workspace
                .active()
                .map(|d| d.colormap)
                .unwrap_or(ColorScheme::Terrain);

            match show_menu_bar(ui, colormap) {
                MenuAction::Open => {
                    io::open_geotiff(self.tx.clone());
                }
                MenuAction::Save => {
                    if let Some(active) = self.workspace.active() {
                        if let DatasetRaster::F64(r) = &active.raster {
                            io::save_geotiff(r.clone(), active.id, self.tx.clone());
                        } else {
                            self.logs.push(LogEntry::warning(
                                "Save currently supports only f64 rasters",
                            ));
                        }
                    } else {
                        self.logs.push(LogEntry::warning("No dataset to save"));
                    }
                }
                MenuAction::Exit => {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                MenuAction::ChangeColormap(scheme) => {
                    if let Some(active) = self.workspace.active_mut() {
                        active.colormap = scheme;
                        invalidate_texture(active, &mut self.map_texture);
                    }
                }
                MenuAction::ZoomToFit => {
                    self.map_state.zoom = 1.0;
                    self.map_state.offset = egui::Vec2::ZERO;
                }
                MenuAction::About => {
                    self.show_about = true;
                }
                MenuAction::None => {}
            }
        });

        // Dataset selector bar
        if self.workspace.len() > 0 {
            egui::TopBottomPanel::top("dataset_bar").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Dataset:");
                    let names = self.workspace.dataset_names();
                    let active_id = self.workspace.active_dataset;
                    let current_name = active_id
                        .and_then(|id| names.iter().find(|(did, _)| *did == id))
                        .map(|(_, n)| n.as_str())
                        .unwrap_or("(none)");

                    egui::ComboBox::from_id_salt("active_dataset")
                        .selected_text(current_name)
                        .show_ui(ui, |ui| {
                            for (id, name) in &names {
                                if ui
                                    .selectable_label(active_id == Some(*id), name)
                                    .clicked()
                                {
                                    self.workspace.active_dataset = Some(*id);
                                    self.map_texture = None;
                                }
                            }
                        });

                    ui.separator();
                    ui.label(format!("{} datasets loaded", self.workspace.len()));

                    if self.running {
                        ui.separator();
                        ui.spinner();
                        ui.label("Processing...");
                    }
                });
            });
        }

        // About dialog
        if self.show_about {
            egui::Window::new("About SurtGis")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.heading("SurtGis Desktop");
                    ui.label("High-performance geospatial analysis");
                    ui.label(format!("Version {}", env!("CARGO_PKG_VERSION")));
                    ui.separator();
                    ui.label(format!("{} algorithms available", self.registry.len()));
                    ui.separator();
                    if ui.button("Close").clicked() {
                        self.show_about = false;
                    }
                });
        }

        // Main dock area
        let registry = self.registry.clone();
        let dataset_names = self.workspace.dataset_names();

        let mut tab_viewer = SurtGisTabViewer {
            workspace: &mut self.workspace,
            logs: &self.logs,
            registry: &registry,
            selected_algo: &mut self.selected_algo,
            tool_dialog: &mut self.tool_dialog,
            map_state: &mut self.map_state,
            map_texture: &mut self.map_texture,
            dataset_names: &dataset_names,
            algo_clicked: None,
            run_requested: false,
            ctx,
        };

        DockArea::new(&mut self.dock_state)
            .style(Style::from_egui(ctx.style().as_ref()))
            .show(ctx, &mut tab_viewer);

        // Extract results before dropping the borrow
        let algo_clicked = tab_viewer.algo_clicked.take();
        let run_requested = tab_viewer.run_requested;
        drop(tab_viewer);

        // Handle tool dialog actions
        if let Some(algo_id) = algo_clicked {
            if let Some(algo) = registry.iter().find(|a| a.id == algo_id.as_str()) {
                self.tool_dialog.open(algo);
            }
        }

        if run_requested {
            self.run_algorithm();
            self.tool_dialog.close();
        }
    }
}

/// TabViewer implementation for egui_dock.
struct SurtGisTabViewer<'a> {
    workspace: &'a mut Workspace,
    logs: &'a [LogEntry],
    registry: &'a [AlgorithmEntry],
    selected_algo: &'a mut Option<String>,
    tool_dialog: &'a mut ToolDialogState,
    map_state: &'a mut MapCanvasState,
    map_texture: &'a mut Option<TextureHandle>,
    dataset_names: &'a [(DatasetId, String)],
    algo_clicked: Option<String>,
    run_requested: bool,
    ctx: &'a egui::Context,
}

impl<'a> TabViewer for SurtGisTabViewer<'a> {
    type Tab = PanelId;

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        tab.to_string().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        match tab {
            PanelId::MapCanvas => {
                let active = self
                    .workspace
                    .active_dataset
                    .and_then(|id| self.workspace.get_mut(id));
                show_map_canvas(ui, active, self.map_texture, self.map_state, self.ctx);
            }

            PanelId::AlgoTree => {
                // Show tool dialog if open, otherwise show the tree
                if self.tool_dialog.algo_id.is_some() {
                    match show_tool_dialog(
                        ui,
                        self.tool_dialog,
                        self.registry,
                        self.dataset_names,
                    ) {
                        ToolDialogAction::Run => {
                            self.run_requested = true;
                        }
                        ToolDialogAction::Cancel => {
                            self.tool_dialog.close();
                        }
                        ToolDialogAction::None => {}
                    }
                } else {
                    if let Some(algo_id) =
                        show_algo_tree(ui, self.registry, self.selected_algo)
                    {
                        self.algo_clicked = Some(algo_id);
                    }
                }
            }

            PanelId::Console => {
                show_console(ui, self.logs);
            }
        }
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false // Panels cannot be closed in MVP
    }
}

/// Suggest a colormap based on the algorithm name.
fn suggest_colormap(algo_name: &str) -> ColorScheme {
    let lower = algo_name.to_lowercase();
    if lower.contains("ndvi") {
        ColorScheme::Ndvi
    } else if lower.contains("ndwi") || lower.contains("water") || lower.contains("twi") {
        ColorScheme::Water
    } else if lower.contains("geomorphon") {
        ColorScheme::Geomorphons
    } else if lower.contains("curvature") || lower.contains("tpi") || lower.contains("dev") {
        ColorScheme::BlueWhiteRed
    } else if lower.contains("accumulation") {
        ColorScheme::Accumulation
    } else if lower.contains("hillshade") {
        ColorScheme::Grayscale
    } else if lower.contains("aspect") {
        ColorScheme::Divergent
    } else {
        ColorScheme::Terrain
    }
}
