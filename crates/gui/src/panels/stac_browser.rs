//! STAC browser panel: search spatio-temporal asset catalogs and download COGs.
//!
//! Requires the `cloud` feature. Without it, the panel shows a placeholder message.

use egui::Ui;

/// Which catalog to search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StacCatalogChoice {
    PlanetaryComputer,
    EarthSearch,
    Custom(String),
}

impl StacCatalogChoice {
    pub fn label(&self) -> &str {
        match self {
            Self::PlanetaryComputer => "Planetary Computer",
            Self::EarthSearch => "Earth Search",
            Self::Custom(_) => "Custom",
        }
    }
}

impl Default for StacCatalogChoice {
    fn default() -> Self {
        Self::PlanetaryComputer
    }
}

/// Current state of a search operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StacSearchState {
    Idle,
    Searching,
    Results,
    Error(String),
}

impl Default for StacSearchState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Simplified STAC search result for UI display.
#[derive(Debug, Clone)]
pub struct StacSearchResult {
    pub id: String,
    pub datetime: String,
    pub cloud_cover: Option<f64>,
    pub platform: Option<String>,
    pub gsd: Option<f64>,
    pub collection: String,
    /// Available asset keys (e.g. "B04", "visual", "SCL").
    pub asset_keys: Vec<String>,
    /// First COG asset href (for download).
    pub cog_href: Option<String>,
    /// First COG asset key.
    pub cog_key: Option<String>,
}

/// Actions the STAC browser can emit for the main app to handle.
#[derive(Debug)]
pub enum StacBrowserAction {
    Search,
    Download {
        item_idx: usize,
        asset_key: String,
    },
    UseMapExtent,
    None,
}

/// Full state of the STAC browser panel.
pub struct StacBrowserState {
    pub catalog: StacCatalogChoice,
    pub custom_url: String,

    // Search parameters
    pub bbox_west: String,
    pub bbox_south: String,
    pub bbox_east: String,
    pub bbox_north: String,
    pub date_start: String,
    pub date_end: String,
    pub collection_filter: String,
    pub max_cloud: f64,

    // Results
    pub search_state: StacSearchState,
    pub results: Vec<StacSearchResult>,
    pub total_matched: Option<u64>,
    pub selected_item: Option<usize>,

    /// Catalog selector combo index (0=PC, 1=ES, 2=Custom).
    catalog_idx: usize,
}

impl Default for StacBrowserState {
    fn default() -> Self {
        Self {
            catalog: StacCatalogChoice::PlanetaryComputer,
            custom_url: String::new(),
            bbox_west: String::new(),
            bbox_south: String::new(),
            bbox_east: String::new(),
            bbox_north: String::new(),
            date_start: String::new(),
            date_end: String::new(),
            collection_filter: "sentinel-2-l2a".to_string(),
            max_cloud: 30.0,
            search_state: StacSearchState::Idle,
            results: Vec::new(),
            total_matched: None,
            selected_item: None,
            catalog_idx: 0,
        }
    }
}

/// Show the STAC browser panel. Returns an action for the main app to process.
pub fn show_stac_browser(ui: &mut Ui, state: &mut StacBrowserState) -> StacBrowserAction {
    #[cfg(not(feature = "cloud"))]
    {
        let _ = state;
        ui.centered_and_justified(|ui| {
            ui.label("STAC browser requires the 'cloud' feature.\nRebuild with: cargo build --features cloud");
        });
        return StacBrowserAction::None;
    }

    #[cfg(feature = "cloud")]
    show_stac_browser_inner(ui, state)
}

#[cfg(feature = "cloud")]
fn show_stac_browser_inner(ui: &mut Ui, state: &mut StacBrowserState) -> StacBrowserAction {
    let mut action = StacBrowserAction::None;

    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.heading("STAC Catalog Search");
        ui.separator();

        // ── Catalog selector ────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Catalog:");
            egui::ComboBox::from_id_salt("stac_catalog")
                .selected_text(match state.catalog_idx {
                    0 => "Planetary Computer",
                    1 => "Earth Search",
                    _ => "Custom",
                })
                .show_ui(ui, |ui| {
                    if ui.selectable_label(state.catalog_idx == 0, "Planetary Computer").clicked() {
                        state.catalog_idx = 0;
                        state.catalog = StacCatalogChoice::PlanetaryComputer;
                    }
                    if ui.selectable_label(state.catalog_idx == 1, "Earth Search").clicked() {
                        state.catalog_idx = 1;
                        state.catalog = StacCatalogChoice::EarthSearch;
                    }
                    if ui.selectable_label(state.catalog_idx == 2, "Custom URL").clicked() {
                        state.catalog_idx = 2;
                        state.catalog = StacCatalogChoice::Custom(state.custom_url.clone());
                    }
                });
        });

        if state.catalog_idx == 2 {
            ui.horizontal(|ui| {
                ui.label("URL:");
                if ui.text_edit_singleline(&mut state.custom_url).changed() {
                    state.catalog = StacCatalogChoice::Custom(state.custom_url.clone());
                }
            });
        }

        ui.separator();

        // ── Bounding box ────────────────────────────────────────────
        ui.label("Bounding Box (WGS-84):");
        egui::Grid::new("stac_bbox").num_columns(4).show(ui, |ui| {
            ui.label("West:");
            ui.text_edit_singleline(&mut state.bbox_west);
            ui.label("South:");
            ui.text_edit_singleline(&mut state.bbox_south);
            ui.end_row();

            ui.label("East:");
            ui.text_edit_singleline(&mut state.bbox_east);
            ui.label("North:");
            ui.text_edit_singleline(&mut state.bbox_north);
            ui.end_row();
        });

        if ui.button("Use Map Extent").clicked() {
            action = StacBrowserAction::UseMapExtent;
        }

        ui.separator();

        // ── Temporal filter ─────────────────────────────────────────
        ui.label("Date Range:");
        ui.horizontal(|ui| {
            ui.label("From:");
            ui.text_edit_singleline(&mut state.date_start);
            ui.label("To:");
            ui.text_edit_singleline(&mut state.date_end);
        });

        ui.separator();

        // ── Collection + cloud cover ────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Collection:");
            ui.text_edit_singleline(&mut state.collection_filter);
        });

        ui.horizontal(|ui| {
            ui.label("Max cloud cover:");
            ui.add(egui::Slider::new(&mut state.max_cloud, 0.0..=100.0).suffix("%"));
        });

        ui.separator();

        // ── Search button ───────────────────────────────────────────
        let is_searching = state.search_state == StacSearchState::Searching;
        ui.horizontal(|ui| {
            if ui.add_enabled(!is_searching, egui::Button::new("Search")).clicked() {
                action = StacBrowserAction::Search;
            }
            if is_searching {
                ui.spinner();
                ui.label("Searching...");
            }
        });

        if let StacSearchState::Error(ref msg) = state.search_state {
            ui.colored_label(egui::Color32::RED, format!("Error: {}", msg));
        }

        // ── Results table ───────────────────────────────────────────
        if !state.results.is_empty() {
            ui.separator();
            if let Some(total) = state.total_matched {
                ui.label(format!(
                    "Showing {} of {} results",
                    state.results.len(),
                    total
                ));
            } else {
                ui.label(format!("{} results", state.results.len()));
            }

            egui::ScrollArea::both()
                .max_height(300.0)
                .show(ui, |ui| {
                    egui::Grid::new("stac_results")
                        .striped(true)
                        .num_columns(6)
                        .min_col_width(60.0)
                        .show(ui, |ui| {
                            // Header
                            ui.strong("Date");
                            ui.strong("Cloud%");
                            ui.strong("Platform");
                            ui.strong("GSD");
                            ui.strong("Collection");
                            ui.strong("Action");
                            ui.end_row();

                            for (idx, item) in state.results.iter().enumerate() {
                                let is_selected = state.selected_item == Some(idx);

                                if ui
                                    .selectable_label(is_selected, &item.datetime)
                                    .clicked()
                                {
                                    state.selected_item = Some(idx);
                                }

                                ui.label(
                                    item.cloud_cover
                                        .map(|c| format!("{:.1}", c))
                                        .unwrap_or_else(|| "—".into()),
                                );
                                ui.label(
                                    item.platform
                                        .as_deref()
                                        .unwrap_or("—"),
                                );
                                ui.label(
                                    item.gsd
                                        .map(|g| format!("{:.1}m", g))
                                        .unwrap_or_else(|| "—".into()),
                                );
                                ui.label(&item.collection);

                                if item.cog_href.is_some() {
                                    if ui.small_button("Download").clicked() {
                                        action = StacBrowserAction::Download {
                                            item_idx: idx,
                                            asset_key: item
                                                .cog_key
                                                .clone()
                                                .unwrap_or_default(),
                                        };
                                    }
                                } else {
                                    ui.label("—");
                                }

                                ui.end_row();
                            }
                        });
                });

            // ── Asset detail for selected item ──────────────────────
            if let Some(sel) = state.selected_item {
                if let Some(item) = state.results.get(sel) {
                    ui.separator();
                    ui.label(format!("Assets for: {}", item.id));
                    for key in &item.asset_keys {
                        ui.horizontal(|ui| {
                            ui.monospace(key);
                            if ui.small_button("Download").clicked() {
                                action = StacBrowserAction::Download {
                                    item_idx: sel,
                                    asset_key: key.clone(),
                                };
                            }
                        });
                    }
                }
            }
        }
    });

    action
}
