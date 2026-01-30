//! Benchmarks for morphology algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use surtgis_algorithms::morphology::{
    closing, dilate, erode, gradient, opening, StructuringElement,
};
use surtgis_core::{GeoTransform, Raster};

fn create_test_raster(size: usize) -> Raster<f64> {
    let mut r = Raster::new(size, size);
    r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
    // Varied surface with some structure
    for row in 0..size {
        for col in 0..size {
            let v = ((row * 7 + col * 13) % 256) as f64;
            r.set(row, col, v).unwrap();
        }
    }
    r
}

fn bench_erode(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology/erode");
    let se = StructuringElement::Square(1);
    for size in [256, 512, 1024, 2048] {
        let raster = create_test_raster(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| erode(black_box(&raster), &se).unwrap())
        });
    }
    group.finish();
}

fn bench_dilate(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology/dilate");
    let se = StructuringElement::Square(1);
    for size in [256, 512, 1024, 2048] {
        let raster = create_test_raster(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| dilate(black_box(&raster), &se).unwrap())
        });
    }
    group.finish();
}

fn bench_opening(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology/opening");
    let se = StructuringElement::Square(1);
    for size in [256, 512, 1024, 2048] {
        let raster = create_test_raster(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| opening(black_box(&raster), &se).unwrap())
        });
    }
    group.finish();
}

fn bench_closing(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology/closing");
    let se = StructuringElement::Square(1);
    for size in [256, 512, 1024, 2048] {
        let raster = create_test_raster(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| closing(black_box(&raster), &se).unwrap())
        });
    }
    group.finish();
}

fn bench_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology/gradient");
    let se = StructuringElement::Square(1);
    for size in [256, 512, 1024, 2048] {
        let raster = create_test_raster(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| gradient(black_box(&raster), &se).unwrap())
        });
    }
    group.finish();
}

fn bench_radius_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology/erode_radius");
    let raster = create_test_raster(1024);
    for radius in [1, 2, 3, 5, 8] {
        let se = StructuringElement::Square(radius);
        group.bench_with_input(BenchmarkId::from_parameter(radius), &radius, |b, _| {
            b.iter(|| erode(black_box(&raster), &se).unwrap())
        });
    }
    group.finish();
}

fn bench_se_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("morphology/erode_shapes");
    let raster = create_test_raster(1024);
    let shapes: Vec<(&str, StructuringElement)> = vec![
        ("square_3", StructuringElement::Square(1)),
        ("cross_3", StructuringElement::Cross(1)),
        ("disk_3", StructuringElement::Disk(1)),
        ("square_5", StructuringElement::Square(2)),
        ("disk_5", StructuringElement::Disk(2)),
    ];
    for (name, se) in &shapes {
        group.bench_with_input(BenchmarkId::new("shape", name), name, |b, _| {
            b.iter(|| erode(black_box(&raster), se).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_erode,
    bench_dilate,
    bench_opening,
    bench_closing,
    bench_gradient,
    bench_radius_scaling,
    bench_se_shapes,
);
criterion_main!(benches);
