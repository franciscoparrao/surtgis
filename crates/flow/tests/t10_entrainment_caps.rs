//! T10 (spec v1.1 §7): the four erosion caps and the mass budget under a
//! deliberately pathological configuration.
//!
//! K = 1.0 (three orders of magnitude past the recommended range) on a
//! steep plane with a heterogeneous erodible field: the per-cell cap
//! (e ≤ e_max, exact), the rate cap (Δe ≤ rate_max·Δt over any interval),
//! the budget invariant and NaN-freeness must all hold — bounded by
//! construction, not by parameter goodwill.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{EntrainmentParams, Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 60;
const COLS: usize = 200;
const DX: f64 = 5.0;

#[test]
fn t10_caps_hold_under_pathological_k() {
    let slope = 25.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    let mut emax_data = vec![0.0f32; ROWS * COLS];
    let mut rel_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            dem_data[r * COLS + c] = ((COLS as f64 * DX - x) * slope) as f32;
            // Erodible field: deterministic heterogeneous pattern 0..5 m.
            emax_data[r * COLS + c] = ((r * 7 + c * 13) % 11) as f32 * 0.5;
            if (20..40).contains(&r) && (10..40).contains(&c) {
                rel_data[r * COLS + c] = 8.0;
            }
        }
    }
    let transform = GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(dem_data, ROWS, COLS).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::from_vec(rel_data, ROWS, COLS).unwrap();
    release.set_transform(transform);
    let mut emax_r = Raster::from_vec(emax_data.clone(), ROWS, COLS).unwrap();
    emax_r.set_transform(transform);

    let mut params = EntrainmentParams::default();
    params.k = 1.0; // pathological on purpose
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    };
    let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config).unwrap();
    sim.set_erodible(&emax_r, params).unwrap();
    let mass0 = sim.total_mass();
    let budget: f64 = emax_data.iter().map(|&v| f64::from(v)).sum::<f64>() * DX * DX;

    let mut substeps = 0u32;
    let mut e_prev = vec![0.0f32; ROWS * COLS];
    let mut t_prev = 0.0f64;
    while substeps < 2000 {
        // Every step must succeed: the caps keep the pathological K bounded,
        // so the budget invariant never trips (it trips only on solver bugs).
        substeps += sim.step(1.0).unwrap();
        let e = sim.eroded_depth();
        let dt_int = sim.time() - t_prev;
        for i in 0..e.len() {
            assert!(e[i].is_finite() && e[i] >= 0.0);
            assert!(
                e[i] <= emax_data[i],
                "e[{i}] = {} > e_max = {} (per-cell cap, spec T10)",
                e[i],
                emax_data[i]
            );
            // Rate cap integrated over the interval (rate_max default
            // 0.05). The absolute slack covers f32 representation of the
            // two e snapshots (values up to 5 m -> ulp ~5e-7 each); any
            // real cap breach would exceed this by orders of magnitude.
            let de = f64::from(e[i]) - f64::from(e_prev[i]);
            assert!(
                de <= 0.05 * dt_int + 2e-6,
                "cell {i} eroded {de} m in {dt_int} s (rate cap, spec T10)"
            );
        }
        e_prev.copy_from_slice(e);
        t_prev = sim.time();
        let s = sim.state();
        assert!(s.h.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    let eroded = sim.total_eroded();
    eprintln!(
        "T10: {substeps} substeps at K=1.0 — eroded {eroded:.0} of {budget:.0} m³ budget, mass {:.0} -> {:.0} m³, all caps held",
        mass0,
        sim.total_mass()
    );
    assert!(eroded > 0.0);
    assert!(
        eroded <= budget * (1.0 + 1e-6),
        "budget exceeded (spec T10)"
    );
}

#[test]
fn t10b_set_erodible_rejected_after_step_and_bad_inputs() {
    let mut dem: Raster<f32> = Raster::new(20, 20);
    dem.set_transform(GeoTransform::new(0.0, 200.0, 10.0, -10.0));
    let mut release: Raster<f32> = Raster::new(20, 20);
    release.set_transform(*dem.transform());
    release.set(10, 10, 2.0).unwrap();
    let mut emax: Raster<f32> = Raster::new(20, 20);
    emax.set_transform(*dem.transform());

    let mut sim = Simulation::new(
        &dem,
        &release,
        VoellmyParams::default(),
        SolverConfig::default(),
    )
    .unwrap();

    // Negative erodible depth -> rejected.
    let mut bad = emax.clone();
    bad.set(5, 5, -1.0).unwrap();
    assert!(
        sim.set_erodible(&bad, EntrainmentParams::default())
            .is_err()
    );

    // Bad params -> rejected.
    let mut p = EntrainmentParams::default();
    p.f_max = 0.0;
    assert!(sim.set_erodible(&emax, p).is_err());

    // After the first step -> AlreadyStepped (spec v1.1 §4).
    sim.step(1.0).unwrap();
    assert!(matches!(
        sim.set_erodible(&emax, EntrainmentParams::default()),
        Err(surtgis_flow::FlowError::AlreadyStepped { .. })
    ));
}
