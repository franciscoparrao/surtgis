//! Benchmarks for terrain algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use surtgis_algorithms::terrain::{slope, SlopeParams};
use surtgis_core::{GeoTransform, Raster};

fn create_dem(size: usize) -> Raster<f64> {
    let mut dem = Raster::new(size, size);
    dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));

    // Create a varied surface (combination of planes and noise-like pattern)
    for row in 0..size {
        for col in 0..size {
            let base = (row + col) as f64;
            let variation = ((row * 7 + col * 13) % 100) as f64 / 10.0;
            dem.set(row, col, base + variation).unwrap();
        }
    }
    dem
}

fn bench_slope(c: &mut Criterion) {
    let mut group = c.benchmark_group("slope");

    for size in [256, 512, 1024, 2048].iter() {
        let dem = create_dem(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                slope(black_box(&dem), SlopeParams::default()).unwrap()
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_slope);
criterion_main!(benches);
