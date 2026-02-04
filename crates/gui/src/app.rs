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
use crate::panels::data_manager::show_data_manager;
use crate::panels::layers::{show_layers, LayerAction};
use crate::panels::map_canvas::{invalidate_texture, show_map_canvas, MapCanvasState};
use crate::panels::properties::{show_properties, PropertiesAction};
use crate::panels::stac_browser::{
    StacBrowserAction, StacBrowserState, StacSearchState, show_stac_browser,
};
use crate::panels::tool_dialog::{show_tool_dialog, ToolDialogAction, ToolDialogState};
use crate::panels::view_3d::{View3dState, show_view_3d};
use crate::registry::{build_registry, AlgorithmEntry, ParamValue};
use crate::render::map_tiles::{BasemapState, show_basemap};
use crate::render::tiled_renderer::TiledRenderer;
use crate::render::MapMode;
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

    /// Map display mode (Simple / Basemap).
    map_mode: MapMode,

    /// Basemap state (lazy-initialised on first use).
    basemap: Option<BasemapState>,

    /// Tiled renderer for large rasters.
    tiled_renderer: TiledRenderer,

    /// STAC browser panel state.
    stac_browser: StacBrowserState,

    /// 3D wireframe view state.
    view_3d: View3dState,
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
            map_mode: MapMode::Simple,
            basemap: None,
            tiled_renderer: TiledRenderer::default(),
            stac_browser: StacBrowserState::default(),
            view_3d: View3dState::default(),
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
                        opacity: 1.0,
                        rgba_cache: None,
                        provenance: None,
                    };

                    self.workspace.add_dataset(dataset);
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
                        opacity: 1.0,
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
                        opacity: 1.0,
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
                        opacity: 1.0,
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

                AppMessage::StacSearchComplete { items, total } => {
                    self.stac_browser.results = items;
                    self.stac_browser.total_matched = total;
                    self.stac_browser.search_state = StacSearchState::Results;
                    self.stac_browser.selected_item = None;
                    self.logs.push(LogEntry::info(format!(
                        "STAC search: {} results",
                        self.stac_browser.results.len()
                    )));
                }

                AppMessage::StacAssetLoaded {
                    item_id,
                    asset_key,
                    raster,
                } => {
                    let name = format!("{} ({})", item_id, asset_key);
                    let dataset = Dataset {
                        id: DatasetId(0),
                        name: name.clone(),
                        source_path: None,
                        raster: DatasetRaster::F64(raster),
                        colormap: ColorScheme::Terrain,
                        visible: true,
                        opacity: 1.0,
                        rgba_cache: None,
                        provenance: Some("STAC download".to_string()),
                    };
                    self.workspace.add_dataset(dataset);
                    self.map_texture = None;
                    self.logs
                        .push(LogEntry::success(format!("Downloaded: {}", name)));
                }

                AppMessage::StacError { message } => {
                    self.stac_browser.search_state =
                        StacSearchState::Error(message.clone());
                    self.logs.push(LogEntry::error(format!("STAC: {}", message)));
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

    /// Handle STAC browser actions.
    fn handle_stac_action(&mut self, action: StacBrowserAction) {
        match action {
            StacBrowserAction::Search => {
                self.launch_stac_search();
            }
            StacBrowserAction::Download { item_idx, asset_key } => {
                self.launch_stac_download(item_idx, asset_key);
            }
            StacBrowserAction::UseMapExtent => {
                if let Some(active) = self.workspace.active() {
                    let (min_x, min_y, max_x, max_y) = active.raster.bounds();
                    self.stac_browser.bbox_west = format!("{:.6}", min_x);
                    self.stac_browser.bbox_south = format!("{:.6}", min_y);
                    self.stac_browser.bbox_east = format!("{:.6}", max_x);
                    self.stac_browser.bbox_north = format!("{:.6}", max_y);
                } else {
                    self.logs.push(LogEntry::warning(
                        "No dataset loaded to get map extent from",
                    ));
                }
            }
            StacBrowserAction::None => {}
        }
    }

    /// Launch a STAC search in a background thread.
    #[cfg(feature = "cloud")]
    fn launch_stac_search(&mut self) {
        use crate::panels::stac_browser::{StacCatalogChoice, StacSearchResult};

        let west: f64 = self.stac_browser.bbox_west.parse().unwrap_or(0.0);
        let south: f64 = self.stac_browser.bbox_south.parse().unwrap_or(0.0);
        let east: f64 = self.stac_browser.bbox_east.parse().unwrap_or(0.0);
        let north: f64 = self.stac_browser.bbox_north.parse().unwrap_or(0.0);

        let datetime = if !self.stac_browser.date_start.is_empty()
            && !self.stac_browser.date_end.is_empty()
        {
            Some(format!(
                "{}/{}",
                self.stac_browser.date_start, self.stac_browser.date_end
            ))
        } else {
            None
        };

        let collection = if self.stac_browser.collection_filter.is_empty() {
            None
        } else {
            Some(self.stac_browser.collection_filter.clone())
        };

        let max_cloud = self.stac_browser.max_cloud;

        let catalog_str = match &self.stac_browser.catalog {
            StacCatalogChoice::PlanetaryComputer => "pc".to_string(),
            StacCatalogChoice::EarthSearch => "es".to_string(),
            StacCatalogChoice::Custom(url) => url.clone(),
        };

        self.stac_browser.search_state = StacSearchState::Searching;
        let tx = self.tx.clone();

        std::thread::spawn(move || {
            let catalog = surtgis_cloud::StacCatalog::from_str_or_url(&catalog_str);
            let options = surtgis_cloud::StacClientOptions::default();
            let client = match surtgis_cloud::blocking::StacClientBlocking::new(catalog, options) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(AppMessage::StacError {
                        message: format!("Failed to create STAC client: {}", e),
                    });
                    return;
                }
            };

            let mut params = surtgis_cloud::StacSearchParams::new()
                .bbox(west, south, east, north);

            if let Some(dt) = &datetime {
                params = params.datetime(dt);
            }
            if let Some(col) = &collection {
                params = params.collections(&[col.as_str()]);
            }

            match client.search_all(&params) {
                Ok(items) => {
                    let results: Vec<StacSearchResult> = items
                        .iter()
                        .filter(|item| {
                            item.properties
                                .eo_cloud_cover
                                .map(|c| c <= max_cloud)
                                .unwrap_or(true)
                        })
                        .map(|item| {
                            let cog = item.first_cog_asset();
                            StacSearchResult {
                                id: item.id.clone(),
                                datetime: item
                                    .properties
                                    .datetime
                                    .clone()
                                    .unwrap_or_else(|| "—".into()),
                                cloud_cover: item.properties.eo_cloud_cover,
                                platform: item.properties.platform.clone(),
                                gsd: item.properties.gsd,
                                collection: item
                                    .collection
                                    .clone()
                                    .unwrap_or_else(|| "—".into()),
                                asset_keys: item.assets.keys().cloned().collect(),
                                cog_href: cog.map(|(_, a)| a.href.clone()),
                                cog_key: cog.map(|(k, _)| k.clone()),
                            }
                        })
                        .collect();

                    let total = None; // search_all doesn't return total count
                    let _ = tx.send(AppMessage::StacSearchComplete {
                        items: results,
                        total,
                    });
                }
                Err(e) => {
                    let _ = tx.send(AppMessage::StacError {
                        message: format!("{}", e),
                    });
                }
            }
        });
    }

    #[cfg(not(feature = "cloud"))]
    fn launch_stac_search(&mut self) {
        self.logs.push(LogEntry::error(
            "STAC search requires the 'cloud' feature",
        ));
    }

    /// Download a STAC asset in a background thread.
    #[cfg(feature = "cloud")]
    fn launch_stac_download(&mut self, item_idx: usize, asset_key: String) {
        use crate::panels::stac_browser::StacCatalogChoice;

        let Some(result) = self.stac_browser.results.get(item_idx) else {
            return;
        };

        // Find the href for the requested asset key
        let href = if asset_key == result.cog_key.as_deref().unwrap_or("") {
            result.cog_href.clone()
        } else {
            None
        };

        let Some(href) = href else {
            self.logs.push(LogEntry::error(format!(
                "No download URL for asset '{}'",
                asset_key
            )));
            return;
        };

        let item_id = result.id.clone();
        let collection = result.collection.clone();
        let catalog_str = match &self.stac_browser.catalog {
            StacCatalogChoice::PlanetaryComputer => "pc".to_string(),
            StacCatalogChoice::EarthSearch => "es".to_string(),
            StacCatalogChoice::Custom(url) => url.clone(),
        };

        let tx = self.tx.clone();
        self.logs.push(LogEntry::info(format!(
            "Downloading {} / {} ...",
            item_id, asset_key
        )));

        std::thread::spawn(move || {
            // Sign the URL if needed (Planetary Computer)
            let catalog = surtgis_cloud::StacCatalog::from_str_or_url(&catalog_str);
            let signed_href = if catalog.needs_signing() {
                let options = surtgis_cloud::StacClientOptions::default();
                match surtgis_cloud::blocking::StacClientBlocking::new(catalog, options) {
                    Ok(client) => match client.sign_asset_href(&href, &collection) {
                        Ok(h) => h,
                        Err(e) => {
                            let _ = tx.send(AppMessage::StacError {
                                message: format!("Signing failed: {}", e),
                            });
                            return;
                        }
                    },
                    Err(e) => {
                        let _ = tx.send(AppMessage::StacError {
                            message: format!("Client error: {}", e),
                        });
                        return;
                    }
                }
            } else {
                href
            };

            // Read COG
            let options = surtgis_cloud::CogReaderOptions::default();
            match surtgis_cloud::blocking::CogReaderBlocking::open(&signed_href, options) {
                Ok(mut reader) => match reader.read_full::<f64>(None) {
                    Ok(raster) => {
                        let _ = tx.send(AppMessage::StacAssetLoaded {
                            item_id,
                            asset_key,
                            raster,
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(AppMessage::StacError {
                            message: format!("COG read failed: {}", e),
                        });
                    }
                },
                Err(e) => {
                    let _ = tx.send(AppMessage::StacError {
                        message: format!("COG open failed: {}", e),
                    });
                }
            }
        });
    }

    #[cfg(not(feature = "cloud"))]
    fn launch_stac_download(&mut self, _item_idx: usize, _asset_key: String) {
        self.logs.push(LogEntry::error(
            "STAC download requires the 'cloud' feature",
        ));
    }
}

impl eframe::App for SurtGisApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Process pending messages
        self.process_messages();

        // Request repaint while algorithm is running or STAC search in progress
        if self.running || self.stac_browser.search_state == StacSearchState::Searching {
            ctx.request_repaint();
        }

        // Menu bar
        let registry_ref = self.registry.clone();
        let current_map_mode = self.map_mode;
        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            let colormap = self
                .workspace
                .active()
                .map(|d| d.colormap)
                .unwrap_or(ColorScheme::Terrain);

            match show_menu_bar(ui, colormap, current_map_mode, &registry_ref) {
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
                MenuAction::ChangeMapMode(mode) => {
                    self.map_mode = mode;
                    if mode == MapMode::Basemap && self.basemap.is_none() {
                        // Lazy-init basemap centred on the active dataset (or default)
                        let (lon, lat) = self
                            .workspace
                            .active()
                            .map(|ds| {
                                let (min_x, min_y, max_x, max_y) = ds.raster.bounds();
                                ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
                            })
                            .unwrap_or((-3.7, 40.4)); // default: Madrid
                        self.basemap = Some(BasemapState::new(ctx, lon, lat));
                    }
                }
                MenuAction::About => {
                    self.show_about = true;
                }
                MenuAction::RunAlgorithm(algo_id) => {
                    if let Some(algo) = self.registry.iter().find(|a| a.id == algo_id.as_str()) {
                        self.tool_dialog.open(algo);
                    }
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
            data_manager_select: None,
            layer_action: LayerAction::None,
            properties_action: PropertiesAction::None,
            stac_browser: &mut self.stac_browser,
            stac_action: StacBrowserAction::None,
            view_3d: &mut self.view_3d,
            map_mode: self.map_mode,
            basemap: &mut self.basemap,
            tiled_renderer: &mut self.tiled_renderer,
            ctx,
        };

        DockArea::new(&mut self.dock_state)
            .style(Style::from_egui(ctx.style().as_ref()))
            .show(ctx, &mut tab_viewer);

        // Extract results before dropping the borrow
        let algo_clicked = tab_viewer.algo_clicked.take();
        let run_requested = tab_viewer.run_requested;
        let dm_select = tab_viewer.data_manager_select;
        let layer_action = std::mem::replace(&mut tab_viewer.layer_action, LayerAction::None);
        let props_action =
            std::mem::replace(&mut tab_viewer.properties_action, PropertiesAction::None);
        let stac_action =
            std::mem::replace(&mut tab_viewer.stac_action, StacBrowserAction::None);
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

        // Handle data manager selection
        if let Some(id) = dm_select {
            self.workspace.active_dataset = Some(id);
            self.map_texture = None;
        }

        // Handle layer actions
        match layer_action {
            LayerAction::ToggleVisibility(id) => {
                if let Some(ds) = self.workspace.get_mut(id) {
                    ds.visible = !ds.visible;
                    ds.rgba_cache = None;
                }
                self.map_texture = None;
            }
            LayerAction::ChangeOpacity(id, opacity) => {
                if let Some(ds) = self.workspace.get_mut(id) {
                    ds.opacity = opacity;
                }
            }
            LayerAction::Select(id) => {
                self.workspace.active_dataset = Some(id);
                self.map_texture = None;
            }
            LayerAction::Remove(id) => {
                let name = self
                    .workspace
                    .get(id)
                    .map(|d| d.name.clone())
                    .unwrap_or_default();
                self.workspace.remove(id);
                self.map_texture = None;
                self.logs
                    .push(LogEntry::info(format!("Removed layer: {}", name)));
            }
            LayerAction::None => {}
        }

        // Handle properties actions
        match props_action {
            PropertiesAction::ChangeColormap(scheme) => {
                if let Some(active) = self.workspace.active_mut() {
                    active.colormap = scheme;
                    invalidate_texture(active, &mut self.map_texture);
                }
            }
            PropertiesAction::None => {}
        }

        // Handle STAC browser actions
        self.handle_stac_action(stac_action);
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
    /// Action from data manager (select a dataset).
    data_manager_select: Option<DatasetId>,
    /// Action from layers panel.
    layer_action: LayerAction,
    /// Action from properties panel.
    properties_action: PropertiesAction,
    /// STAC browser state and action.
    stac_browser: &'a mut StacBrowserState,
    stac_action: StacBrowserAction,
    /// 3D view state.
    view_3d: &'a mut View3dState,
    /// Current map mode.
    map_mode: MapMode,
    /// Basemap state (for Basemap mode).
    basemap: &'a mut Option<BasemapState>,
    /// Tiled renderer (for large rasters).
    tiled_renderer: &'a mut TiledRenderer,
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
                match self.map_mode {
                    MapMode::Simple => {
                        let active = self
                            .workspace
                            .active_dataset
                            .and_then(|id| self.workspace.get_mut(id));
                        show_map_canvas(ui, active, self.map_texture, self.map_state, self.ctx);
                    }
                    MapMode::Basemap => {
                        if let Some(basemap) = self.basemap {
                            let active = self.workspace.active();
                            let (tex, bounds, opacity) = match active {
                                Some(ds) => (
                                    self.map_texture.as_ref(),
                                    Some(ds.raster.bounds()),
                                    ds.opacity,
                                ),
                                None => (None, None, 1.0),
                            };
                            show_basemap(ui, basemap, tex, bounds, opacity);
                        } else {
                            ui.label("Basemap not initialised.");
                        }
                    }
                }
            }

            PanelId::View3D => {
                let active_ds = self.workspace.active();
                show_view_3d(ui, active_ds, self.view_3d);
            }

            PanelId::AlgoTree => {
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

            PanelId::StacBrowser => {
                self.stac_action = show_stac_browser(ui, self.stac_browser);
            }

            PanelId::Console => {
                show_console(ui, self.logs);
            }

            PanelId::DataManager => {
                if let Some(id) = show_data_manager(ui, self.workspace) {
                    self.data_manager_select = Some(id);
                }
            }

            PanelId::Layers => {
                self.layer_action = show_layers(ui, self.workspace);
            }

            PanelId::Properties => {
                let active_ds = self.workspace.active();
                self.properties_action = show_properties(ui, active_ds);
            }
        }
    }

    fn closeable(&mut self, _tab: &mut Self::Tab) -> bool {
        false // Panels cannot be closed
    }
}

/// Suggest a colormap based on the algorithm name.
fn suggest_colormap(algo_name: &str) -> ColorScheme {
    let lower = algo_name.to_lowercase();
    if lower.contains("ndvi") || lower.contains("gndvi") || lower.contains("ndre")
        || lower.contains("savi") || lower.contains("evi") || lower.contains("reci")
        || lower.contains("ngrdi") || lower.contains("msavi")
    {
        ColorScheme::Ndvi
    } else if lower.contains("ndwi") || lower.contains("water") || lower.contains("twi")
        || lower.contains("mrvbf") || lower.contains("depression") || lower.contains("ndmi")
    {
        ColorScheme::Water
    } else if lower.contains("geomorphon") || lower.contains("landform") {
        ColorScheme::Geomorphons
    } else if lower.contains("curvature") || lower.contains("tpi") || lower.contains("dev")
        || lower.contains("moran") || lower.contains("getis") || lower.contains("convergence")
        || lower.contains("wind")
    {
        ColorScheme::BlueWhiteRed
    } else if lower.contains("accumulation") || lower.contains("spi") || lower.contains("sti")
        || lower.contains("cost_distance")
    {
        ColorScheme::Accumulation
    } else if lower.contains("hillshade") || lower.contains("viewshed") {
        ColorScheme::Grayscale
    } else if lower.contains("aspect") {
        ColorScheme::Divergent
    } else if lower.contains("solar") || lower.contains("radiation") {
        ColorScheme::Terrain
    } else if lower.contains("ndsi") || lower.contains("snow") {
        ColorScheme::Water
    } else if lower.contains("ndbi") || lower.contains("built") {
        ColorScheme::Accumulation
    } else if lower.contains("shannon") || lower.contains("simpson") || lower.contains("diversity")
        || lower.contains("patch")
    {
        ColorScheme::Ndvi
    } else if lower.contains("contour") {
        ColorScheme::Grayscale
    } else if lower.contains("kmeans") || lower.contains("isodata")
        || lower.contains("minimum distance") || lower.contains("maximum likelihood")
        || lower.contains("strahler") || lower.contains("isobasin")
    {
        ColorScheme::Geomorphons
    } else if lower.contains("glcm") || lower.contains("texture") || lower.contains("entropy") {
        ColorScheme::Terrain
    } else if lower.contains("sobel") || lower.contains("laplacian") || lower.contains("edge") {
        ColorScheme::Grayscale
    } else if lower.contains("flood") {
        ColorScheme::Water
    } else if lower.contains("flow path") || lower.contains("path length") {
        ColorScheme::Accumulation
    } else if lower.contains("pca") || lower.contains("pc1") {
        ColorScheme::BlueWhiteRed
    } else if lower.contains("raster diff") || lower.contains("change") || lower.contains("cva") {
        ColorScheme::BlueWhiteRed
    } else {
        ColorScheme::Terrain
    }
}
