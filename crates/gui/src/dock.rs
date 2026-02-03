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
}

impl std::fmt::Display for PanelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PanelId::MapCanvas => write!(f, "Map"),
            PanelId::AlgoTree => write!(f, "Algorithms"),
            PanelId::Console => write!(f, "Console"),
        }
    }
}

/// Create the initial dock layout.
pub fn create_dock_state() -> DockState<PanelId> {
    // Start with the map canvas as the main surface
    let mut dock_state = DockState::new(vec![PanelId::MapCanvas]);

    // Split: main area (top) and console (bottom) — 75% / 25%
    let [_top, _bottom] = dock_state.main_surface_mut().split_below(
        NodeIndex::root(),
        0.75,
        vec![PanelId::Console],
    );

    // Split top area: map canvas (left) and algo tree (right) — 72% / 28%
    let [_map, _algo] = dock_state.main_surface_mut().split_right(
        _top,
        0.72,
        vec![PanelId::AlgoTree],
    );

    dock_state
}
