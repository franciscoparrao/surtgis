//! T7 (spec §7): bitwise determinism under rayon.
//!
//! Two runs of the same valley scenario (T4 geometry, shortened) on an
//! 8-thread pool must produce bit-identical states. The solver is in fact
//! deterministic independently of thread count (per-cell accumulation in
//! fixed order, exact max-reduction for the CFL step), so a third run on a
//! 3-thread pool is also compared — a stronger guarantee than the spec asks.

use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{FlowState, Simulation, SolverConfig, VoellmyParams};

const ROWS: usize = 80;
const COLS: usize = 100;
const DX: f64 = 10.0;
const SUBSTEPS: u32 = 400;

fn build_rasters() -> (Raster<f32>, Raster<f32>) {
    let down = 20.0f64.to_radians().tan();
    let side = 10.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; ROWS * COLS];
    let mut rel_data = vec![0.0f32; ROWS * COLS];
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
            if (30..50).contains(&r) && (8..40).contains(&c) {
                rel_data[r * COLS + c] = 5.0;
            }
        }
    }
    let transform = GeoTransform::new(0.0, ROWS as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(dem_data, ROWS, COLS).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::from_vec(rel_data, ROWS, COLS).unwrap();
    release.set_transform(transform);
    (dem, release)
}

fn run_scenario(threads: usize) -> (FlowState, Vec<f32>, f64) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let (dem, release) = build_rasters();
        let config = SolverConfig {
            cfl: 0.45,
            h_dry: 1e-3,
            max_substeps: 100_000,
        };
        let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config).unwrap();
        let mut substeps = 0u32;
        while substeps < SUBSTEPS {
            substeps += sim.step(1.0).unwrap();
        }
        (
            sim.state().clone(),
            sim.arrival_times().to_vec(),
            sim.total_mass(),
        )
    })
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn t7_two_runs_8_threads_are_bitwise_identical() {
    let (s1, a1, m1) = run_scenario(8);
    let (s2, a2, m2) = run_scenario(8);

    // memcmp-equivalent: compare the raw bit patterns (spec T7).
    assert_eq!(bits(&s1.h), bits(&s2.h), "h differs between runs");
    assert_eq!(bits(&s1.hu), bits(&s2.hu), "hu differs between runs");
    assert_eq!(bits(&s1.hv), bits(&s2.hv), "hv differs between runs");
    assert_eq!(bits(&a1), bits(&a2), "arrival times differ between runs");
    assert_eq!(
        m1.to_bits(),
        m2.to_bits(),
        "total mass differs between runs"
    );

    // Stronger than spec: thread-count independence by construction.
    let (s3, a3, m3) = run_scenario(3);
    assert_eq!(bits(&s1.h), bits(&s3.h), "h differs across thread counts");
    assert_eq!(
        bits(&s1.hu),
        bits(&s3.hu),
        "hu differs across thread counts"
    );
    assert_eq!(
        bits(&s1.hv),
        bits(&s3.hv),
        "hv differs across thread counts"
    );
    assert_eq!(bits(&a1), bits(&a3), "arrivals differ across thread counts");
    assert_eq!(
        m1.to_bits(),
        m3.to_bits(),
        "mass differs across thread counts"
    );

    eprintln!(
        "T7: 8-thread runs bitwise identical; 3-thread run also identical (stronger than spec)"
    );
}
