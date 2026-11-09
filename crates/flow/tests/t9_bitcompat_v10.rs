//! T9 (spec v1.1 §7): bit-compatibility with v1.0.
//!
//! The same scenario run (a) without `set_erodible` and (b) with an
//! all-zero erodible raster must produce bit-identical states: opting in
//! with a zero budget is a strict no-op, so the v1.0 behaviour is
//! preserved exactly unless material is actually erodible.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{EntrainmentParams, FlowState, Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 100;
const COLS: usize = 150;
const DX: f64 = 10.0;
const SUBSTEPS: u32 = 1200;

fn build() -> (Raster<f32>, Raster<f32>) {
    let down = 20.0f64.to_radians().tan();
    let side = 10.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            let y = (r as f64 + 0.5) * DX;
            let z = (COLS as f64 * DX - x) * down + (y - ROWS as f64 * DX / 2.0).abs() * side;
            dem_data[r * COLS + c] = if r == 0 || c == 0 || r == ROWS - 1 || c == COLS - 1 {
                f32::NAN
            } else {
                z as f32
            };
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
    (dem, release)
}

fn run(with_zero_erodible: bool) -> (FlowState, Vec<f32>, f64) {
    let (dem, release) = build();
    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    };
    let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config).unwrap();
    if with_zero_erodible {
        let mut zeros: Raster<f32> = Raster::new(ROWS, COLS);
        zeros.set_transform(*dem.transform());
        sim.set_erodible(&zeros, EntrainmentParams::default())
            .unwrap();
    }
    let mut substeps = 0u32;
    while substeps < SUBSTEPS {
        substeps += sim.step(1.0).unwrap();
    }
    (
        sim.state().clone(),
        sim.arrival_times().to_vec(),
        sim.total_mass(),
    )
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn t9_zero_erodible_is_bitwise_identical_to_plain_run() {
    let (s_plain, a_plain, m_plain) = run(false);
    let (s_zero, a_zero, m_zero) = run(true);

    assert_eq!(bits(&s_plain.h), bits(&s_zero.h), "h differs (spec T9)");
    assert_eq!(bits(&s_plain.hu), bits(&s_zero.hu), "hu differs (spec T9)");
    assert_eq!(bits(&s_plain.hv), bits(&s_zero.hv), "hv differs (spec T9)");
    assert_eq!(bits(&a_plain), bits(&a_zero), "arrivals differ (spec T9)");
    assert_eq!(
        m_plain.to_bits(),
        m_zero.to_bits(),
        "mass differs (spec T9)"
    );
    eprintln!("T9: zero-erodible run bitwise identical to plain v1.0 run ({SUBSTEPS} substeps)");
}
