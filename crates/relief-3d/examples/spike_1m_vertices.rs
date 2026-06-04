//! M1 acceptance spike — 1024×1024 = 1,048,576 vertex mesh.
//!
//! Run from the repo root:
//!
//! ```bash
//! cargo run --release -p surtgis-relief-3d --example spike_1m_vertices
//! ```
//!
//! Acceptance bar: ≥60 FPS sustained on the dev machine. This is the
//! production-workload validation the 2D spec failed to do (§12.6 of
//! `SPEC_SURTGIS_RELIEF.md`). If this drops below 60 FPS, the wgpu
//! pipeline approach is the wrong architecture and we must address it
//! in M1 rather than after every milestone has inherited the trap.

use surtgis_relief_3d::mesh::{cosine_test_heights, grid_mesh};
#[cfg(not(target_arch = "wasm32"))]
use surtgis_relief_3d::native::run_spike;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    const SIDE: usize = 1024;
    eprintln!(
        "relief-3d M1 spike: {}×{} grid ({} vertices, {} triangles)",
        SIDE,
        SIDE,
        SIDE * SIDE,
        (SIDE - 1) * (SIDE - 1) * 2
    );
    let (verts, idx) = grid_mesh(SIDE, SIDE, 1.5, cosine_test_heights(SIDE, SIDE));
    eprintln!("FPS bar: ≥60 sustained. ESC or close-window to exit. Camera orbits on its own.");
    if let Err(e) = run_spike(verts, idx, "surtgis-relief-3d — M1 spike (1M vertices)") {
        eprintln!("spike error: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
