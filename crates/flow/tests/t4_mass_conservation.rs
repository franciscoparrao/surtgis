//! T4 (spec §7): mass conservation over a long run with closed borders.
//!
//! Debris-flow scenario on a V-shaped valley (synthetic stand-in for the
//! Macul quebrada — the real-DEM run belongs to the out-of-CI scientific
//! validation, §7 last paragraph), release 500,000 m³, 5000 substeps, all
//! borders closed with a NoData ring so no mass can leave the domain.
//! Acceptance: |Δmass|/mass₀ < 1e-6.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 100;
const COLS: usize = 150;
const DX: f64 = 10.0;

#[test]
fn t4_mass_is_conserved_with_closed_borders() {
    // Valley: 20° main slope descending east, 10° side slopes funnelling
    // into the centreline; 1-cell NoData ring closes every border.
    let down = 20.0f64.to_radians().tan();
    let side = 10.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            let y = (r as f64 + 0.5) * DX;
            let z = (COLS as f64 * DX - x) * down + (y - ROWS as f64 * DX / 2.0).abs() * side;
            dem_data[r * COLS + c] = if r == 0 || c == 0 || r == ROWS - 1 || c == COLS - 1 {
                f32::NAN // closed border
            } else {
                z as f32
            };
        }
    }
    // Release: 25 rows x 40 cols x 5 m x (10 m)^2 = 500,000 m³ near the head
    // of the valley.
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

    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    };
    let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config).unwrap();
    let mass0 = sim.total_mass();
    assert!((mass0 - 500_000.0).abs() < 1.0, "release is {mass0} m³");

    let mut substeps = 0u32;
    while substeps < 5000 {
        substeps += sim.step(2.0).unwrap();
    }

    let drift = (sim.total_mass() - mass0).abs() / mass0;
    eprintln!(
        "T4: {substeps} substeps, mass {mass0:.1} -> {:.1} m³, relative drift = {drift:.3e} (gate 1e-6)",
        sim.total_mass()
    );
    assert!(
        drift < 1e-6,
        "relative mass drift {drift:e} >= 1e-6 (spec T4)"
    );

    // With every border closed nothing may wet the ring's inner neighbours'
    // walls... but the ring itself must have stayed dry (solid cells carry
    // no state by construction).
    let h = &sim.state().h;
    for c in 0..COLS {
        assert_eq!(h[c], 0.0);
        assert_eq!(h[(ROWS - 1) * COLS + c], 0.0);
    }
    for r in 0..ROWS {
        assert_eq!(h[r * COLS], 0.0);
        assert_eq!(h[r * COLS + COLS - 1], 0.0);
    }
}
