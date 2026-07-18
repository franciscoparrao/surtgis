//! T6 (spec §7): non-negativity and NaN robustness.
//!
//! Release block over the rim of a NoData cliff (reflective wall) and over a
//! domain corner (transmissive outflow), on a 30° slope, 2000 steps with
//! realistic Voellmy friction. Acceptance: min(h) >= 0 exact and zero
//! NaN/Inf over the whole run.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 100;
const COLS: usize = 100;
const DX: f64 = 5.0;

#[test]
fn t6_no_negative_depths_no_nan() {
    // 30° plane dipping east, so the flow runs into the canyon wall and out
    // through the eastern/corner edges.
    let slope = 30.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            let mut z = (COLS as f64 * DX - x) * slope;
            // NoData canyon: vertical strip, upper half of the domain.
            if (60..70).contains(&c) && r < 60 {
                z = f64::NAN;
            }
            dem_data[r * COLS + c] = z as f32;
        }
    }
    // Release: 8 m block straddling the canyon rim (cells inside the NoData
    // strip are ignored by construction) + 5 m block on the NW domain corner.
    let mut rel_data = vec![0.0f32; ROWS * COLS];
    for r in 20..30 {
        for c in 50..65 {
            rel_data[r * COLS + c] = 8.0;
        }
    }
    for r in 0..8 {
        for c in 0..8 {
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

    // Drive until at least 2000 substeps have run; every step must succeed
    // (a Diverged error would fail the unwrap) and the state must stay
    // exactly non-negative and finite throughout.
    let mut substeps = 0u32;
    let mut checks = 0u32;
    while substeps < 2000 {
        substeps += sim.step(1.0).unwrap();
        let s = sim.state();
        let mut h_min = f32::INFINITY;
        for i in 0..s.h.len() {
            h_min = h_min.min(s.h[i]);
            assert!(
                s.h[i].is_finite() && s.hu[i].is_finite() && s.hv[i].is_finite(),
                "non-finite state at cell {i} after {substeps} substeps"
            );
        }
        assert!(h_min >= 0.0, "negative depth {h_min:e} (spec T6)");
        checks += 1;
    }
    eprintln!("T6: {substeps} substeps, {checks} full-state checks, min(h) >= 0 and all finite");

    // Sanity: the release actually flowed. 214 release cells; several times
    // that must have been wetted on the way downslope (with the static-yield
    // detention the deposit is compact: ~1250 cells in practice).
    let arrivals = sim.arrival_times();
    let wetted = arrivals.iter().filter(|t| t.is_finite()).count();
    assert!(
        wetted > 800,
        "only {wetted} cells ever wetted; the flow did not propagate"
    );
}
