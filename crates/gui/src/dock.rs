//! Dock layout: SAGA-style panel arrangement using egui_dock.
//!
//! Layout:
//! ```text
//! ┌──────────────────────────┬─────────────────┐
//! │  MapCanvas               │  AlgoTree        │
//! │  View3D                  │  Properties      │
//! │  (tabbed)                │  StacBrowser     │
//! │                          │  (tabbed)        │
//! │                          ├─────────────────┤
//! │                          │  DataManager     │
//! │                          │  Layers          │
//! │                          │  (tabbed)        │
//! ├──────────────────────────┴─────────────────┤
//! │               Console                       │
//! └─────────────────────────────────────────────┘
//! ```

use egui_dock::{DockState, NodeIndex};

/// Panel identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PanelId {
    MapCanvas,
    View3D,
    AlgoTree,
    StacBrowser,
    Console,
    DataManager,
    Layers,
    Properties,
}

impl std::fmt::Display for PanelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PanelId::MapCanvas => write!(f, "Map"),
            PanelId::View3D => write!(f, "3D View"),
            PanelId::AlgoTree => write!(f, "Algorithms"),
            PanelId::StacBrowser => write!(f, "STAC"),
            PanelId::Console => write!(f, "Console"),
            PanelId::DataManager => write!(f, "Data"),
            PanelId::Layers => write!(f, "Layers"),
            PanelId::Properties => write!(f, "Properties"),
        }
    }
}

/// Create the initial dock layout (SAGA-style).
pub fn create_dock_state() -> DockState<PanelId> {
    // Start with the map canvas + 3D view as tabbed main surface
    let mut dock_state = DockState::new(vec![PanelId::MapCanvas, PanelId::View3D]);

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
        vec![PanelId::AlgoTree, PanelId::Properties, PanelId::StacBrowser],
    );

    // Split right sidebar: top (AlgoTree+Properties+StacBrowser) and bottom (DataManager+Layers)
    let [_right_top, _right_bottom] = dock_state.main_surface_mut().split_below(
        right,
        0.5,
        vec![PanelId::DataManager, PanelId::Layers],
    );

    dock_state
}
