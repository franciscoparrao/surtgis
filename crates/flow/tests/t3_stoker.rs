//! T3 (spec §7): dam break over a wet bed vs the Stoker analytical solution.
//!
//! 1D channel (1000×3, Δx = 1 m), h_L = 1 m, h_R = 0.1 m, no friction.
//! Acceptance at t = 10 s: L1 error of h < 0.015 m and shock position within
//! ±2 cells of the analytical shock.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const G: f64 = 9.80665;
const COLS: usize = 1000;
const ROWS: usize = 3;
const DX: f64 = 1.0;
const DAM_X: f64 = 500.0;
const H_L: f64 = 1.0;
const H_R: f64 = 0.1;
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

/// Stoker middle state: depth h2 of the constant region between the
/// rarefaction and the shock, from matching the rarefaction relation
/// `u2 = 2(√(g·hL) − √(g·h2))` with the shock jump relation
/// `u2 = (h2 − hR)·√(g/2·(h2 + hR)/(h2·hR))` by bisection.
fn stoker_middle() -> (f64, f64, f64) {
    let c_l = (G * H_L).sqrt();
    let f = |h2: f64| {
        2.0 * (c_l - (G * h2).sqrt()) - (h2 - H_R) * ((G / 2.0) * (h2 + H_R) / (h2 * H_R)).sqrt()
    };
    let (mut lo, mut hi) = (H_R, H_L);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let h2 = 0.5 * (lo + hi);
    let u2 = 2.0 * (c_l - (G * h2).sqrt());
    let shock_speed = h2 * u2 / (h2 - H_R);
    (h2, u2, shock_speed)
}

/// Full Stoker profile at time `t`; `x` measured from the dam.
fn stoker_h(x: f64, t: f64, h2: f64, u2: f64, s: f64) -> f64 {
    let c_l = (G * H_L).sqrt();
    let c2 = (G * h2).sqrt();
    if x <= -c_l * t {
        H_L
    } else if x <= (u2 - c2) * t {
        let v = 2.0 * c_l - x / t;
        v * v / (9.0 * G)
    } else if x <= s * t {
        h2
    } else {
        H_R
    }
}

#[test]
fn t3_dam_break_wet_bed_matches_stoker() {
    let dem = channel(|_| 0.0);
    let release = channel(|c| {
        if ((c as f64) + 0.5) < DAM_X {
            H_L as f32
        } else {
            H_R as f32
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
    sim.step(T_END as f32).unwrap();

    let (h2, u2, shock_speed) = stoker_middle();
    let h = &sim.state().h;

    // L1 (mean absolute) error against the analytical profile, middle row.
    let mut l1 = 0.0f64;
    for c in 0..COLS {
        let x = (c as f64 + 0.5) - DAM_X;
        let num = f64::from(h[COLS + c]);
        l1 += (num - stoker_h(x, T_END, h2, u2, shock_speed)).abs();
    }
    l1 /= COLS as f64;
    eprintln!("T3 Stoker: L1 = {l1:.5} m (gate 0.015 m), h2 = {h2:.4} m, S = {shock_speed:.4} m/s");
    assert!(l1 < 0.015, "L1 error {l1:.4} m >= 0.015 m (spec T3)");

    // Shock position: last cell (west→east) still above the half-jump depth.
    let half_jump = (h2 + H_R) / 2.0;
    let mut shock_col = None;
    for c in (0..COLS).rev() {
        if f64::from(h[COLS + c]) > half_jump {
            shock_col = Some(c);
            break;
        }
    }
    let x_num = shock_col.expect("no shock found") as f64 + 1.0 - DAM_X; // east face of that cell
    let x_exact = shock_speed * T_END;
    assert!(
        (x_num - x_exact).abs() <= 2.0 * DX,
        "shock at x={x_num} m, analytical {x_exact:.2} m: off by more than 2 cells (spec T3)"
    );
}
