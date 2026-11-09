//! T1 (spec §7): lake at rest over a sinusoidal bed (well-balancedness).
//!
//! Bed z(x, y) sinusoidal with emerged islands, water surface flat at H;
//! no friction; 1000 steps. Acceptance: max |u|, |v| < 1e-6 m/s and relative
//! mass change < 1e-12. This is the direct check that the hydrostatic
//! reconstruction (Audusse et al. 2004) balances the bed source term exactly,
//! including at wet/dry shorelines.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 100;
const COLS: usize = 120;
const DX: f64 = 10.0;
/// Flat water surface elevation [m].
const SURFACE: f64 = 10.0;

fn bed(r: usize, c: usize) -> f64 {
    let x = (c as f64 + 0.5) * DX;
    let y = (r as f64 + 0.5) * DX;
    // 6 ± 6 m: peaks up to 12 m emerge above the 10 m surface as islands.
    let z = 6.0 + 6.0 * (x / 130.0).sin() * (y / 170.0).cos();
    // Quantise to multiples of 2^-10 m so that both z and SURFACE - z are
    // exactly representable in f32: the discrete initial condition is then an
    // exact lake at rest, and the test measures the scheme's balance error
    // rather than the f32 rounding of an inexact initial surface.
    (z * 1024.0).round() / 1024.0
}

#[test]
fn t1_lake_at_rest_stays_at_rest() {
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    let mut rel_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let z = bed(r, c);
            dem_data[r * COLS + c] = z as f32;
            // Depth chosen so the free surface is exactly flat where wet.
            rel_data[r * COLS + c] = (SURFACE - z).max(0.0) as f32;
        }
    }
    let transform = GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(dem_data, ROWS, COLS).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::from_vec(rel_data, ROWS, COLS).unwrap();
    release.set_transform(transform);

    let params = VoellmyParams {
        mu: 0.0,
        xi: f32::INFINITY,
        v_stop: 0.0,
    };
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 2000,
    };
    let mut sim = Simulation::new(&dem, &release, params, config).unwrap();
    let mass0 = sim.total_mass();
    assert!(mass0 > 0.0);

    // Deepest water ~10 m -> c ~ 9.9 m/s -> stable dt ~ 0.45*10/9.9 ~ 0.45 s.
    // step(0.4) is below that, so each call is exactly one substep.
    let mut substeps = 0u32;
    for _ in 0..1000 {
        substeps += sim.step(0.4).unwrap();
    }
    assert!(
        substeps >= 1000,
        "expected >= 1000 substeps, got {substeps}"
    );

    let state = sim.state();
    let h_dry = 1e-3f32;
    let mut max_vel = 0.0f64;
    for i in 0..state.h.len() {
        let h = f64::from(state.h[i]);
        if h < f64::from(h_dry) {
            continue;
        }
        let u = (f64::from(state.hu[i]) / h).abs();
        let v = (f64::from(state.hv[i]) / h).abs();
        max_vel = max_vel.max(u).max(v);
    }
    let drift = (sim.total_mass() - mass0).abs() / mass0;
    eprintln!(
        "T1 lake at rest: max|u|,|v| = {max_vel:.3e} m/s (gate 1e-6), mass drift = {drift:.3e} (gate 1e-12)"
    );
    assert!(
        max_vel < 1e-6,
        "max velocity {max_vel:e} >= 1e-6 m/s (spec T1)"
    );
    assert!(
        drift < 1e-12,
        "relative mass drift {drift:e} >= 1e-12 (spec T1)"
    );
}
