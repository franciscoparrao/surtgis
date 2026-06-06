//! P4-M1 acceptance spike — 4096 × 4096 = 16,777,216 cell synthetic DEM.
//!
//! Run from the repo root:
//!
//!   cargo run --release -p surtgis-relief-3d --example spike_lod_4k
//!
//! Acceptance: ≥ 30 FPS sustained while orbiting the camera. Logs the
//! visible-chunk count alongside the FPS so a regression in culling
//! shows up immediately.
//!
//! Why synthetic and not a real DEM: the §12.6 lesson keeps biting.
//! A real 4K DEM would tie this spike to a specific tile we'd have
//! to commit to the repo (and pay the LFS cost), and a procedural
//! source makes it trivial to scale up to 8K later for the M3 / M4
//! follow-ups.

use std::time::Instant;

use surtgis_core::raster::Raster;
use surtgis_relief_3d::lod::{LodParams, QuadtreeMesh};
#[cfg(not(target_arch = "wasm32"))]
use surtgis_relief_3d::native::{CameraMode, TextureSource, run_lod_viewer_with_mode};

const SIDE: usize = 4096;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    eprintln!(
        "spike_lod_4k: {SIDE}×{SIDE} synthetic DEM ({} cells)",
        SIDE * SIDE
    );

    let t0 = Instant::now();
    let dem = synthetic_dem(SIDE);
    eprintln!("DEM generated in {:.2}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let params = LodParams::default();
    let mesh = QuadtreeMesh::from_dem(&dem, 0.45, params.clone());
    eprintln!(
        "quadtree built in {:.2}s: {} chunks × {} LOD levels, \
         triangle counts {:?}",
        t0.elapsed().as_secs_f64(),
        mesh.chunks.len(),
        mesh.lod_levels,
        mesh.triangle_counts()
    );
    eprintln!(
        "CPU per-chunk total: {:.1} MB; GPU pool capacity 192 MB (M3c lazy upload)",
        mesh.cpu_bytes() as f64 / 1.0e6
    );

    eprintln!("FPS bar: ≥ 30 sustained. ESC or close window to exit.");
    if let Err(e) = run_lod_viewer_with_mode(
        mesh,
        params,
        TextureSource::Checker,
        "surtgis-relief-3d — P4 M1 spike (4K DEM)",
        CameraMode::AutoOrbit,
    ) {
        eprintln!("spike error: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}

/// Procedural landscape: a sum of cosines (slow-frequency ridges) + a
/// bit of higher-frequency detail + a single broad dome. Gives the
/// LOD pipeline something with real relief at every scale, so the
/// distance-band selection has consequences.
fn synthetic_dem(side: usize) -> Raster<f64> {
    let mut dem = Raster::<f64>::new(side, side);
    let inv = 1.0 / side as f64;
    let cx = side as f64 * 0.5;
    let cy = side as f64 * 0.5;
    for r in 0..side {
        for c in 0..side {
            let u = c as f64 * inv * std::f64::consts::TAU * 4.0;
            let v = r as f64 * inv * std::f64::consts::TAU * 3.0;
            let ridges = 220.0 * (u.cos() + v.cos());
            let detail = 60.0 * ((u * 5.0).sin() + (v * 5.0).cos());
            // Broad dome falling off with squared distance from the centre,
            // gives the camera something to orbit around at all distances.
            let dx = c as f64 - cx;
            let dy = r as f64 - cy;
            let dome = 1200.0 * (-(dx * dx + dy * dy) * inv * inv * 4.0).exp();
            dem.set(r, c, 200.0 + ridges + detail + dome).unwrap();
        }
    }
    dem
}
