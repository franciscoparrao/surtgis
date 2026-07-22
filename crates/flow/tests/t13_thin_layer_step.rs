//! T13 (spec v1.1 §7): thin layer over a bed step larger than the flow
//! depth — the regime where plain Audusse truncates the driving force to
//! O(g·h²/Δx) and a film on a steep slope cannot accelerate (the EXP1/EXP2
//! deficit GEODEO measured on Macul at 30 m cells).
//!
//! Setup: 0.05 m uniform film on a 30° plane discretised at 30 m cells
//! (bed step ≈ 17.3 m ≫ h), frictionless. Acceptance: after 10 substeps a
//! mid-domain cell moves at u ≥ 0.8·g·sinθ·t (with Chen & Noelle the
//! depth-averaged force is g·tanθ ≥ g·sinθ; with Audusse u stays at the
//! dam-break scale √(g·h) ≈ 0.7 m/s — orders of magnitude below the gate).

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 5;
const COLS: usize = 40;
const DX: f64 = 30.0;
const H_FILM: f32 = 0.05;

#[test]
fn t13_thin_film_on_steep_slope_accelerates() {
    let slope = 30.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            dem_data[r * COLS + c] = ((COLS as f64 * DX - x) * slope) as f32;
        }
    }
    let transform = GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(dem_data, ROWS, COLS).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::filled(ROWS, COLS, H_FILM);
    release.set_transform(transform);

    let params = VoellmyParams {
        mu: 0.0,
        xi: f32::INFINITY,
        v_stop: 0.0,
    };
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    };
    let mut sim = Simulation::new(&dem, &release, params, config).unwrap();

    // At least 10 substeps, in short physical chunks: the film accelerates
    // hard (that is the point), so the CFL substep shrinks quickly and a
    // large single request would take hundreds of thousands of substeps.
    let mut substeps = 0u32;
    while substeps < 10 {
        substeps += sim.step(2.0).unwrap();
    }
    let t = sim.time();

    // Mid-domain, middle row: u must reflect the full slope force.
    let s = sim.state();
    let i = (ROWS / 2) * COLS + COLS / 2;
    let h = f64::from(s.h[i]);
    assert!(h > 1e-3, "film dried out at the probe cell (h = {h})");
    let u = f64::from(s.hu[i]) / h;
    let g_sin = 9.80665 * 30.0f64.to_radians().sin();
    eprintln!(
        "T13: after {substeps} substeps (t = {t:.1} s): u = {u:.1} m/s vs gate 0.8·g·sinθ·t = {:.1} m/s",
        0.8 * g_sin * t
    );
    assert!(
        u >= 0.8 * g_sin * t,
        "u = {u:.2} m/s < 0.8·g·sinθ·t = {:.2} m/s — thin-layer driving force still truncated (spec T13)",
        0.8 * g_sin * t
    );
    // Sanity: it should also not exceed the frictionless ballistic bound
    // g·tanθ·t by more than FP/transport slack.
    let g_tan = 9.80665 * slope;
    assert!(
        u <= 1.05 * g_tan * t,
        "u = {u:.2} m/s exceeds the ballistic bound {:.2} m/s — overdriven",
        g_tan * t
    );
}
