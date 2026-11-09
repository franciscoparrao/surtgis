//! T2 (spec §7): dam break over a dry bed vs the Ritter analytical solution.
//!
//! 1D channel (1000×3 grid, Δx = 1 m), h_L = 1 m, h_R = 0 (dry), no
//! friction. Acceptance: L1 error of h vs the analytical profile < 0.02 m at
//! t = 10 s.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const G: f64 = 9.80665;
const COLS: usize = 1000;
const ROWS: usize = 3;
const DX: f64 = 1.0;
const DAM_X: f64 = 500.0;
const H_L: f64 = 1.0;
const T_END: f64 = 10.0;

fn channel(fill: impl Fn(usize) -> f32) -> Raster<f32> {
    let mut data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            data[r * COLS + c] = fill(c);
        }
    }
    let mut raster = Raster::from_vec(data, ROWS, COLS).unwrap();
    raster.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    raster
}

/// Ritter (1892) solution for a dam break over a dry, frictionless bed.
/// `x` is measured from the dam.
fn ritter_h(x: f64, t: f64) -> f64 {
    let c_l = (G * H_L).sqrt();
    if x <= -c_l * t {
        H_L
    } else if x < 2.0 * c_l * t {
        let v = 2.0 * c_l - x / t;
        v * v / (9.0 * G)
    } else {
        0.0
    }
}

#[test]
fn t2_dam_break_dry_bed_matches_ritter() {
    let dem = channel(|_| 0.0);
    let release = channel(|c| {
        if ((c as f64) + 0.5) < DAM_X {
            H_L as f32
        } else {
            0.0
        }
    });

    let params = VoellmyParams {
        mu: 0.0,
        xi: f32::INFINITY,
        v_stop: 0.0,
    };
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 10_000,
    };
    let mut sim = Simulation::new(&dem, &release, params, config).unwrap();
    let mass0 = sim.total_mass();

    let substeps = sim.step(T_END as f32).unwrap();
    assert!(substeps > 0);
    assert!((sim.time() - T_END).abs() < 1e-6);

    // L1 (mean absolute) error of h against the analytical profile, middle row.
    let h = &sim.state().h;
    let mut l1 = 0.0f64;
    for c in 0..COLS {
        let x = (c as f64 + 0.5) - DAM_X;
        let num = f64::from(h[COLS + c]); // row 1
        l1 += (num - ritter_h(x, T_END)).abs();
    }
    l1 /= COLS as f64;
    eprintln!("T2 Ritter: L1 = {l1:.5} m (gate 0.02 m), {substeps} substeps");
    assert!(l1 < 0.02, "L1 error {l1:.4} m >= 0.02 m (spec T2)");

    // The channel is symmetric across its 3 rows: they must stay identical.
    for c in 0..COLS {
        assert_eq!(h[c], h[COLS + c], "row 0 != row 1 at col {c}");
        assert_eq!(h[2 * COLS + c], h[COLS + c], "row 2 != row 1 at col {c}");
    }

    // Nothing has reached the domain ends by t = 10 s: mass is conserved.
    let drift = (sim.total_mass() - mass0).abs() / mass0;
    assert!(drift < 1e-5, "relative mass drift {drift:e} >= 1e-5");

    // No negative depths, no NaN (T6 property, cheap to assert here too).
    assert!(h.iter().all(|&v| v >= 0.0 && v.is_finite()));
}
