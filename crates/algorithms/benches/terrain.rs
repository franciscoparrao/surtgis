//! Benchmarks for terrain algorithms

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use surtgis_algorithms::terrain::{
    aspect, curvature, hillshade, landform_classification, slope, tpi, tri, AspectOutput,
    CurvatureParams, CurvatureType, HillshadeParams, LandformParams, SlopeParams, TpiParams,
    TriParams,
};
use surtgis_core::{GeoTransform, Raster};

fn create_dem(size: usize) -> Raster<f64> {
    let mut dem = Raster::new(size, size);
    dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
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
    let mut group = c.benchmark_group("terrain/slope");
    for size in [256, 512, 1024, 2048] {
        let dem = create_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| slope(black_box(&dem), SlopeParams::default()).unwrap())
        });
    }
    group.finish();
}

fn bench_aspect(c: &mut Criterion) {
    let mut group = c.benchmark_group("terrain/aspect");
    for size in [256, 512, 1024, 2048] {
        let dem = create_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| aspect(black_box(&dem), AspectOutput::Degrees).unwrap())
        });
    }
    group.finish();
}

fn bench_hillshade(c: &mut Criterion) {
    let mut group = c.benchmark_group("terrain/hillshade");
    for size in [256, 512, 1024, 2048] {
        let dem = create_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| hillshade(black_box(&dem), HillshadeParams::default()).unwrap())
        });
    }
    group.finish();
}

fn bench_curvature(c: &mut Criterion) {
    let mut group = c.benchmark_group("terrain/curvature");
    for size in [256, 512, 1024, 2048] {
        let dem = create_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                curvature(
                    black_box(&dem),
                    CurvatureParams {
                        curvature_type: CurvatureType::General,
                        ..Default::default()
                    },
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_tpi(c: &mut Criterion) {
    let mut group = c.benchmark_group("terrain/tpi");
    for size in [256, 512, 1024, 2048] {
        let dem = create_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tpi(black_box(&dem), TpiParams { radius: 1 }).unwrap())
        });
    }
    group.finish();
}

fn bench_tri(c: &mut Criterion) {
    let mut group = c.benchmark_group("terrain/tri");
    for size in [256, 512, 1024, 2048] {
        let dem = create_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| tri(black_box(&dem), TriParams { radius: 1 }).unwrap())
        });
    }
    group.finish();
}

fn bench_landform(c: &mut Criterion) {
    let mut group = c.benchmark_group("terrain/landform");
    // Landform is more expensive (computes TPI at 2 scales + slope), use smaller sizes
    for size in [128, 256, 512, 1024] {
        let dem = create_dem(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                landform_classification(
                    black_box(&dem),
                    LandformParams {
                        small_radius: 3,
                        large_radius: 10,
                        tpi_threshold: 1.0,
                        slope_threshold: 6.0,
                    },
                )
                .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_slope,
    bench_aspect,
    bench_hillshade,
    bench_curvature,
    bench_tpi,
    bench_tri,
    bench_landform,
);
criterion_main!(benches);
