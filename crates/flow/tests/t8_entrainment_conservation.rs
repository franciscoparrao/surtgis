//! T8 (spec v1.1 §7): mass conservation with the entrainment source term.
//!
//! T4 valley (closed NoData borders) plus an erodible strip along the
//! channel (e_max = 2 m), K = 1e-3, 5000 substeps, with static-yield
//! deposits forming during the run (interaction coverage, spec v1.1 §2.5).
//! Acceptance: |V_flow − V₀ − V_eroded| / V₀ < 1e-6 throughout, and
//! V_eroded > 0 (the term actually acted).

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{EntrainmentParams, Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 100;
const COLS: usize = 150;
const DX: f64 = 10.0;

fn build() -> (Raster<f32>, Raster<f32>, Raster<f32>) {
    let down = 20.0f64.to_radians().tan();
    let side = 10.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    let mut emax_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            let y = (r as f64 + 0.5) * DX;
            let dy = (y - ROWS as f64 * DX / 2.0).abs();
            let z = (COLS as f64 * DX - x) * down + dy * side;
            dem_data[r * COLS + c] = if r == 0 || c == 0 || r == ROWS - 1 || c == COLS - 1 {
                f32::NAN // closed border
            } else {
                z as f32
            };
            // Erodible strip: 2 m of loose material along the valley axis,
            // downstream of the release.
            if dy < 60.0 && (50..140).contains(&c) {
                emax_data[r * COLS + c] = 2.0;
            }
        }
    }
    let mut rel_data = vec![0.0f32; ROWS * COLS];
    for r in 38..63 {
        for c in 10..50 {
            rel_data[r * COLS + c] = 5.0;
        }
    }
    let transform = GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(dem_data, ROWS, COLS).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::from_vec(rel_data, ROWS, COLS).unwrap();
    release.set_transform(transform);
    let mut emax = Raster::from_vec(emax_data, ROWS, COLS).unwrap();
    emax.set_transform(transform);
    (dem, release, emax)
}

#[test]
fn t8_mass_conserved_with_entrainment_source() {
    let (dem, release, emax) = build();
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    };
    let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config).unwrap();
    sim.set_erodible(&emax, EntrainmentParams::default())
        .unwrap();
    let mass0 = sim.total_mass();
    assert!((mass0 - 500_000.0).abs() < 1.0);

    let mut substeps = 0u32;
    let mut worst = 0.0f64;
    while substeps < 5000 {
        substeps += sim.step(2.0).unwrap();
        // Invariant continuously (every render step): flow gains exactly
        // what the bed lost.
        let residual = (sim.total_mass() - mass0 - sim.total_eroded()).abs() / mass0;
        worst = worst.max(residual);
        assert!(
            residual < 1e-6,
            "conservation residual {residual:e} at {substeps} substeps (spec T8)"
        );
    }

    let eroded = sim.total_eroded();
    eprintln!(
        "T8: {substeps} substeps, eroded {eroded:.0} m³, worst residual {worst:.3e} (gate 1e-6)"
    );
    assert!(eroded > 0.0, "entrainment never acted (spec T8)");
    // Per-cell law: e bounded by e_max everywhere, exact.
    let e = sim.eroded_depth();
    assert_eq!(e.len(), ROWS * COLS);
    for (i, &ei) in e.iter().enumerate() {
        assert!(
            (0.0..=2.0).contains(&ei),
            "e[{i}] = {ei} outside [0, e_max]"
        );
    }
}
