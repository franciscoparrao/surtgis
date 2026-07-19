//! Performance benchmark (spec §7): 1024×1024 grid, ~15% wet cells.
//!
//! Reports substeps/s; the spec target is 30+ substeps/s on 8 cores (M5),
//! 20+ at the end of M3. Does not gate merges — the number is reported in
//! each PR and regressions over 10% require justification.

use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use surtgis_core::{GeoTransform, Raster};
use surtgis_flow::{Simulation, SolverConfig, VoellmyParams};

const N: usize = 1024;
const DX: f64 = 5.0;

/// Inclined plane with a circular release covering ~15% of the domain,
/// pre-run for 20 substeps so the wet area is fully in motion.
fn build_sim() -> Simulation {
    let slope = 15.0f64.to_radians().tan();
    let mut dem_data = vec![0.0f32; N * N];
    for r in 0..N {
        for c in 0..N {
            let x = (c as f64 + 0.5) * DX;
            dem_data[r * N + c] = ((N as f64 * DX - x) * slope) as f32;
        }
    }
    // 15% of N² cells: radius = sqrt(0.15/pi)*N ~ 0.2185*N.
    let radius = 0.2185 * N as f64;
    let centre = N as f64 / 2.0;
    let mut rel_data = vec![0.0f32; N * N];
    for r in 0..N {
        for c in 0..N {
            let dr = r as f64 + 0.5 - centre;
            let dc = c as f64 + 0.5 - centre;
            if (dr * dr + dc * dc).sqrt() <= radius {
                rel_data[r * N + c] = 3.0;
            }
        }
    }
    let transform = GeoTransform::new(0.0, N as f64 * DX, DX, -DX);
    let mut dem = Raster::from_vec(dem_data, N, N).unwrap();
    dem.set_transform(transform);
    let mut release = Raster::from_vec(rel_data, N, N).unwrap();
    release.set_transform(transform);

    let config = SolverConfig {
        cfl: 0.45,
        h_dry: 1e-3,
        max_substeps: 100_000,
    };
    let mut sim = Simulation::new(&dem, &release, VoellmyParams::default(), config).unwrap();
    let mut warmup = 0u32;
    while warmup < 20 {
        warmup += sim.step(0.5).unwrap();
    }
    sim
}

fn bench_substeps(c: &mut Criterion) {
    // Fixed, deterministic workload: every iteration starts from the same
    // warmed-up state (fresh sim per iteration via iter_batched), so the
    // measurement does not drift as the wet area grows. The substep count
    // per step(0.5) is deterministic; substeps/s = n / (time per iter).
    let n_substeps = {
        let mut sim = build_sim();
        sim.step(0.5).unwrap()
    };
    eprintln!("workload: step(0.5) = {n_substeps} substeps from the warmed state");

    let mut group = c.benchmark_group("flow");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(2));
    group.bench_function("step_0.5s_1024x1024_15pct_wet", |b| {
        b.iter_batched(
            build_sim,
            |mut sim| {
                let n = sim.step(black_box(0.5)).unwrap();
                assert_eq!(n, n_substeps, "workload changed");
                sim
            },
            criterion::BatchSize::PerIteration,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_substeps);
criterion_main!(benches);
