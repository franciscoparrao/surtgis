//! T11 (spec v1.1 §7): bitwise determinism with entrainment active —
//! extension of T7 to the eroded state. Two 8-thread runs must match
//! bitwise, and a 3-thread run as well (thread-count independence by
//! construction, as in v1.0).

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{EntrainmentParams, FlowState, Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 80;
const COLS: usize = 100;
const DX: f64 = 10.0;
const SUBSTEPS: u32 = 400;

fn build() -> (Raster<f32>, Raster<f32>, Raster<f32>) {
    let down = 20.0f64.to_radians().tan();
    let side = 10.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    let mut rel_data = vec![0.0f32; ROWS * COLS];
    let mut emax_data = vec![0.0f32; ROWS * COLS];
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            let y = (r as f64 + 0.5) * DX;
            let dy = (y - ROWS as f64 * DX / 2.0).abs();
            let z = (COLS as f64 * DX - x) * down + dy * side;
            dem_data[r * COLS + c] = if r == 0 || c == 0 || r == ROWS - 1 || c == COLS - 1 {
                f32::NAN
            } else {
                z as f32
            };
            if (30..50).contains(&r) && (8..40).contains(&c) {
                rel_data[r * COLS + c] = 5.0;
            }
            if dy < 80.0 && c >= 30 {
                emax_data[r * COLS + c] = 1.5;
            }
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

fn run(threads: usize) -> (FlowState, Vec<f32>, f64, f64) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let (dem, release, emax) = build();
        let config = SolverConfig {
            cfl: 0.45,
            h_dry: 1e-3,
            max_substeps: 100_000,
        };
        let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config).unwrap();
        sim.set_erodible(&emax, EntrainmentParams::default())
            .unwrap();
        let mut substeps = 0u32;
        while substeps < SUBSTEPS {
            substeps += sim.step(1.0).unwrap();
        }
        (
            sim.state().clone(),
            sim.eroded_depth().to_vec(),
            sim.total_mass(),
            sim.total_eroded(),
        )
    })
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn t11_entrainment_runs_are_bitwise_deterministic() {
    let (s1, e1, m1, v1) = run(8);
    let (s2, e2, m2, v2) = run(8);
    assert_eq!(bits(&s1.h), bits(&s2.h), "h differs between 8-thread runs");
    assert_eq!(bits(&s1.hu), bits(&s2.hu));
    assert_eq!(bits(&s1.hv), bits(&s2.hv));
    assert_eq!(bits(&e1), bits(&e2), "eroded depth differs between runs");
    assert_eq!(m1.to_bits(), m2.to_bits());
    assert_eq!(v1.to_bits(), v2.to_bits(), "total eroded differs");
    assert!(v1 > 0.0, "entrainment never acted — test is vacuous");

    let (s3, e3, m3, v3) = run(3);
    assert_eq!(bits(&s1.h), bits(&s3.h), "h differs across thread counts");
    assert_eq!(bits(&s1.hu), bits(&s3.hu));
    assert_eq!(bits(&s1.hv), bits(&s3.hv));
    assert_eq!(bits(&e1), bits(&e3), "eroded depth differs across threads");
    assert_eq!(m1.to_bits(), m3.to_bits());
    assert_eq!(v1.to_bits(), v3.to_bits());
    eprintln!(
        "T11: bitwise identical across runs and thread counts; eroded {v1:.0} m³ in {SUBSTEPS} substeps"
    );
}
