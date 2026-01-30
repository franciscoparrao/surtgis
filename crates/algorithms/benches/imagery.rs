//! Benchmarks for imagery algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use surtgis_algorithms::imagery::{band_math_binary, ndvi, BandMathOp};
use surtgis_core::{GeoTransform, Raster};

fn create_band(size: usize, base: f64) -> Raster<f64> {
    let mut r = Raster::new(size, size);
    r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
    for row in 0..size {
        for col in 0..size {
            let v = base + ((row * 7 + col * 13) % 200) as f64;
            r.set(row, col, v).unwrap();
        }
    }
    r
}

fn bench_ndvi(c: &mut Criterion) {
    let mut group = c.benchmark_group("imagery/ndvi");
    for size in [256, 512, 1024, 2048] {
        let nir = create_band(size, 300.0);
        let red = create_band(size, 100.0);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| ndvi(black_box(&nir), black_box(&red)).unwrap())
        });
    }
    group.finish();
}

fn bench_band_math(c: &mut Criterion) {
    let mut group = c.benchmark_group("imagery/band_math");
    for size in [256, 512, 1024, 2048] {
        let a = create_band(size, 100.0);
        let b_raster = create_band(size, 50.0);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                band_math_binary(black_box(&a), black_box(&b_raster), BandMathOp::Add).unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ndvi, bench_band_math);
criterion_main!(benches);
