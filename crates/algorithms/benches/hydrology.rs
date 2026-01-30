//! Benchmarks for hydrology algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use surtgis_algorithms::hydrology::{
    fill_sinks, flow_accumulation, flow_direction, FillSinksParams,
};
use surtgis_core::{GeoTransform, Raster};

/// Create a DEM with a basin shape: higher edges sloping toward center outlet
fn create_basin_dem(size: usize) -> Raster<f64> {
    let mut dem = Raster::new(size, size);
    dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
    let center = size as f64 / 2.0;
    for row in 0..size {
        for col in 0..size {
            let dx = col as f64 - center;
            let dy = row as f64 - center;
            let dist = (dx * dx + dy * dy).sqrt();
            // Bowl shape + small noise to avoid flat areas
            let noise = ((row * 7 + col * 13) % 17) as f64 * 0.01;
            dem.set(row, col, dist + noise).unwrap();
        }
    }
    dem
}

fn bench_fill_sinks(c: &mut Criterion) {
    let mut group = c.benchmark_group("hydrology/fill_sinks");
    for size in [128, 256, 512, 1024] {
        let dem = create_basin_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                fill_sinks(
                    black_box(&dem),
                    FillSinksParams { min_slope: 0.01 },
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_flow_direction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hydrology/flow_direction");
    for size in [256, 512, 1024, 2048] {
        let dem = create_basin_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| flow_direction(black_box(&dem)).unwrap())
        });
    }
    group.finish();
}

fn bench_flow_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("hydrology/flow_accumulation");
    for size in [256, 512, 1024, 2048] {
        let dem = create_basin_dem(size);
        let fdir = flow_direction(&dem).unwrap();
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| flow_accumulation(black_box(&fdir)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fill_sinks,
    bench_flow_direction,
    bench_flow_accumulation,
);
criterion_main!(benches);
