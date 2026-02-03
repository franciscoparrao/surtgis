//! Dock layout: SAGA-style panel arrangement using egui_dock.
//!
//! Layout: MapCanvas (center, ~72%) | Right panel (AlgoTree/ToolDialog, ~28%)
//!         ─────────────────────────┼───────────────────────────────────────
//!         Console (bottom, ~25% of total height)

use egui_dock::{DockState, NodeIndex};

/// Panel identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelId {
    MapCanvas,
    AlgoTree,
    Console,
    DataManager,
    Layers,
    Properties,
}

impl std::fmt::Display for PanelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PanelId::MapCanvas => write!(f, "Map"),
            PanelId::AlgoTree => write!(f, "Algorithms"),
            PanelId::Console => write!(f, "Console"),
            PanelId::DataManager => write!(f, "Data"),
            PanelId::Layers => write!(f, "Layers"),
            PanelId::Properties => write!(f, "Properties"),
        }
    }
}

/// Create the initial dock layout (SAGA-style).
///
/// ```text
/// ┌──────────────────────────┬─────────────────┐
/// │                          │  AlgoTree        │
/// │       MapCanvas          │  Properties      │
/// │                          │  (tabbed)        │
/// │                          ├─────────────────┤
/// │                          │  DataManager     │
/// │                          │  Layers          │
/// │                          │  (tabbed)        │
/// ├──────────────────────────┴─────────────────┤
/// │               Console                       │
/// └─────────────────────────────────────────────┘
/// ```
pub fn create_dock_state() -> DockState<PanelId> {
    // Start with the map canvas as the main surface
    let mut dock_state = DockState::new(vec![PanelId::MapCanvas]);

    // Split: main area (top) and console (bottom) — 75% / 25%
    let [top, _bottom] = dock_state.main_surface_mut().split_below(
        NodeIndex::root(),
        0.75,
        vec![PanelId::Console],
    );

    // Split top area: map canvas (left 72%) and right sidebar (28%)
    let [_map, right] = dock_state.main_surface_mut().split_right(
        top,
        0.72,
        vec![PanelId::AlgoTree, PanelId::Properties],
    );

    // Split right sidebar: top (AlgoTree+Properties) and bottom (DataManager+Layers)
    let [_right_top, _right_bottom] = dock_state.main_surface_mut().split_below(
        right,
        0.5,
        vec![PanelId::DataManager, PanelId::Layers],
    );

    dock_state
}
