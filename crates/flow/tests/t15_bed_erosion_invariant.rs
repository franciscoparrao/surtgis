//! T15 (audit R4): the bed always satisfies `z = z₀ − e` exactly, and
//! `update_dem` (live barriers, spec §4) composes with erosion instead of
//! silently un-eroding the channel.
//!
//! The old commit path did `z -= Δe` in f32: with bed elevations of
//! 500-900 m the ulp is 3-6e-5 m, so sub-ulp increments were credited to
//! `h`/`e`/`eroded_volume` but never actually left the bed — the flow
//! gained mass the bed never lost, and nothing checked the invariant.
//! The bed is now derived (`z = round(z₀ − e)`, one rounding, no
//! accumulation), which this test pins down bitwise.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{EntrainmentParams, Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 60;
const COLS: usize = 100;
const DX: f64 = 10.0;

/// 20° valley descending east at high absolute elevation (~600-900 m) so
/// the f32 ulp of `z` is orders of magnitude above tiny Δe increments —
/// the exact regime where the old running subtraction lost mass. Closed
/// NoData ring so T15a runs on a sealed domain.
fn high_valley_dem() -> Raster<f32> {
    let down = 20.0f64.to_radians().tan();
    let side = 10.0f64.to_radians().tan();
    let mut data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            let y = (r as f64 + 0.5) * DX;
            let z =
                600.0 + (COLS as f64 * DX - x) * down + (y - ROWS as f64 * DX / 2.0).abs() * side;
            data[r * COLS + c] = if r == 0 || c == 0 || r == ROWS - 1 || c == COLS - 1 {
                f32::NAN
            } else {
                z as f32
            };
        }
    }
    let mut dem = Raster::from_vec(data, ROWS, COLS).unwrap();
    dem.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    dem
}

fn release() -> Raster<f32> {
    let mut data = vec![0.0f32; ROWS * COLS];
    for r in 20..40 {
        for c in 5..25 {
            data[r * COLS + c] = 3.0;
        }
    }
    let mut rel = Raster::from_vec(data, ROWS, COLS).unwrap();
    rel.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    rel
}

fn erodible() -> Raster<f32> {
    let mut data = vec![1.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            if r == 0 || c == 0 || r == ROWS - 1 || c == COLS - 1 {
                data[r * COLS + c] = 0.0;
            }
        }
    }
    let mut e = Raster::from_vec(data, ROWS, COLS).unwrap();
    e.set_transform(GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX));
    e
}

fn config() -> SolverConfig {
    SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    }
}

/// Max |z − round(z₀ − e)| over the grid — must be exactly 0: the bed is
/// derived with the same single-rounding expression.
fn max_invariant_error(sim: &Simulation, dem0: &Raster<f32>) -> f32 {
    let e = sim.eroded_depth();
    let mut worst = 0.0f32;
    for (i, &z0) in dem0.data().iter().enumerate() {
        if !z0.is_finite() {
            continue;
        }
        let expected = if e[i] > 0.0 {
            (f64::from(z0) - f64::from(e[i])) as f32
        } else {
            z0
        };
        let z = sim.grid().bed()[i];
        worst = worst.max((z - expected).abs());
    }
    worst
}

#[test]
fn t15_bed_equals_z0_minus_e_bitwise() {
    let dem0 = high_valley_dem();
    let mut sim = Simulation::new(&dem0, &release(), VoellmyParams::default(), config()).unwrap();
    sim.set_erodible(&erodible(), EntrainmentParams::default())
        .unwrap();
    let mass0 = sim.total_mass();

    for _ in 0..20 {
        sim.step(2.0).unwrap();
    }
    assert!(sim.total_eroded() > 0.0, "no erosion happened at all");

    // e never exceeds e_max (the audit's ~1-ulp overshoot is clamped).
    for &e in sim.eroded_depth() {
        assert!(e <= 1.0, "eroded depth {e} exceeds e_max = 1.0");
    }

    let err = max_invariant_error(&sim, &dem0);
    eprintln!(
        "T15a: eroded {:.1} m³ over {} steps, max |z - (z0 - e)| = {err:e}",
        sim.total_eroded(),
        20
    );
    assert_eq!(err, 0.0, "bed invariant z = z0 - e violated by {err}");

    // Closed domain: what the flow gained, the bed lost — the sub-ulp leak
    // made the left side grow without the right side moving.
    let gained = sim.total_mass() - mass0;
    let lost = sim.total_eroded();
    let rel = (gained - lost).abs() / mass0;
    assert!(
        rel < 1e-4,
        "flow gained {gained:.2} m³ but bed lost {lost:.2} m³ (rel {rel:e})"
    );
}

#[test]
fn t15_update_dem_preserves_erosion_under_a_live_barrier() {
    // The GEODEO live-barrier path had zero tests. Contract: after
    // update_dem with entrainment active, the new DEM is the new z₀ and
    // the accumulated erosion is re-applied on top.
    let dem0 = high_valley_dem();
    let mut sim = Simulation::new(&dem0, &release(), VoellmyParams::default(), config()).unwrap();
    sim.set_erodible(&erodible(), EntrainmentParams::default())
        .unwrap();

    for _ in 0..10 {
        sim.step(2.0).unwrap();
    }
    let eroded_before = sim.total_eroded();
    assert!(eroded_before > 0.0, "no erosion before the barrier");

    // Raise a 5 m barrier across the valley, downstream of the release.
    let mut barrier = dem0.clone();
    for r in 1..ROWS - 1 {
        for c in 60..63 {
            let v = barrier.get(r, c).unwrap();
            barrier.set(r, c, v + 5.0).unwrap();
        }
    }
    sim.update_dem(&barrier).unwrap();

    // The bed must reflect BOTH the barrier and the prior erosion.
    let err = max_invariant_error(&sim, &barrier);
    assert_eq!(
        err, 0.0,
        "after update_dem the bed no longer satisfies z = z0_new - e"
    );

    // And the run keeps going: stepping over the modified bed works, the
    // erosion ledger is monotone, and the invariant still holds.
    for _ in 0..10 {
        sim.step(2.0).unwrap();
    }
    assert!(sim.total_eroded() >= eroded_before);
    let err = max_invariant_error(&sim, &barrier);
    assert_eq!(err, 0.0, "bed invariant broken after post-barrier stepping");
}
