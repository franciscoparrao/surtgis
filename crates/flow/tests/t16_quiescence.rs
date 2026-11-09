//! T16: the quiescence diagnostic `Simulation::mass_fraction_at_rest`.
//!
//! Runout calibration is only meaningful on a flow that has stopped —
//! measuring extent at a fixed simulated time instead reports where the
//! front happened to be at that instant, and for any target extent there
//! is a (friction, cut-off time) pair that "reproduces" it. This test pins
//! the diagnostic that lets a caller detect rest instead of guessing.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 60;
const COLS: usize = 120;
const DX: f64 = 10.0;

/// Slope descending east that flattens into a closed basin, so a
/// high-friction flow can actually come to rest inside the domain.
fn ramp_into_flat() -> Raster<f32> {
    let mut data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = c as f64 * DX;
            // 25° for the first half, flat afterwards
            let z = if x < 600.0 {
                800.0 - x * 25.0f64.to_radians().tan()
            } else {
                800.0 - 600.0 * 25.0f64.to_radians().tan()
            };
            let border = r == 0 || c == 0 || r == ROWS - 1 || c == COLS - 1;
            data[r * COLS + c] = if border { f32::NAN } else { z as f32 };
        }
    }
    let mut dem = Raster::from_vec(data, ROWS, COLS).unwrap();
    dem.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    dem
}

fn release() -> Raster<f32> {
    let mut data = vec![0.0f32; ROWS * COLS];
    for r in 25..35 {
        for c in 5..15 {
            data[r * COLS + c] = 4.0;
        }
    }
    let mut rel = Raster::from_vec(data, ROWS, COLS).unwrap();
    rel.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    rel
}

fn config() -> SolverConfig {
    SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 200_000,
    }
}

fn sim(mu: f32) -> Simulation {
    let params = VoellmyParams {
        mu,
        xi: 200.0,
        ..VoellmyParams::default()
    };
    Simulation::new(&ramp_into_flat(), &release(), params, config()).unwrap()
}

#[test]
fn t16_at_rest_before_release_and_after_deposition() {
    let mut s = sim(0.1); // mobile enough to be clearly in motion early on

    // Initial state: zero velocity everywhere, so everything is at rest.
    assert_eq!(s.mass_fraction_at_rest(0.5), 1.0);

    // Mid-flight a substantial share of the volume is moving.
    s.step(10.0).unwrap();
    let moving = s.mass_fraction_at_rest(0.5);
    assert!(
        moving < 0.9,
        "flow should be substantially in motion shortly after release, \
         got {moving:.3} at rest"
    );

    // Given enough time in the closed basin it deposits and the fraction
    // returns to ~1 — the signal a calibration run should wait for.
    for _ in 0..90 {
        s.step(10.0).unwrap();
    }
    let settled = s.mass_fraction_at_rest(0.5);
    assert!(
        settled > 0.99,
        "flow should have come to rest, got {settled:.3}"
    );
}

#[test]
fn t16_rest_is_reached_and_then_holds() {
    // The property a stopping criterion depends on: once the flow settles,
    // it stays settled — so "resting fraction >= f" is a usable halting
    // test rather than a value that oscillates across the threshold.
    //
    // (Note the fraction is NOT monotone in friction: in a closed basin a
    // low-friction flow arrives and levels out sooner than a high-friction
    // one still creeping down the ramp. Rest is about the state of motion,
    // not about how mobile the rheology is.)
    let mut s = sim(0.1);
    let mut first_rest: Option<usize> = None;
    for i in 0..80 {
        s.step(10.0).unwrap();
        let f = s.mass_fraction_at_rest(0.5);
        if first_rest.is_none() && f > 0.99 {
            first_rest = Some(i);
        }
        if let Some(k) = first_rest {
            assert!(
                f > 0.98,
                "flow left rest after settling at step {k} (step {i}: {f:.3})"
            );
        }
    }
    assert!(
        first_rest.is_some(),
        "flow never reached rest within 800 s of simulated time"
    );
}

#[test]
fn t16_threshold_is_monotone_and_bounded() {
    let mut s = sim(0.1);
    s.step(20.0).unwrap();
    let mut prev = -1.0;
    for &vt in &[0.01f32, 0.1, 1.0, 10.0, 1e6] {
        let f = s.mass_fraction_at_rest(vt);
        assert!((0.0..=1.0).contains(&f), "fraction {f} out of [0,1]");
        assert!(
            f >= prev - 1e-12,
            "raising the threshold must not lower the resting fraction \
             ({f} after {prev})"
        );
        prev = f;
    }
    // An enormous threshold classifies everything as at rest.
    assert_eq!(s.mass_fraction_at_rest(1e9), 1.0);
}
