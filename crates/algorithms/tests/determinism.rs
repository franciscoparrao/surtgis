//! Determinism contract: the same inputs give bit-identical outputs run to
//! run and under any number of worker threads.
//!
//! Two things could break this and both are covered. (1) A parallel
//! reduction whose combination order depends on the thread count: the
//! suite runs under a 1-thread and a 4-thread pool. (2) Iterating a
//! `HashMap` into a numeric result: `std`'s hasher is seeded per
//! instance, so the suite runs twice in the same process and the outputs
//! must match. Every output is compared through the same BLAKE3 digest
//! that `surtgis verify` uses, so "identical" means the bytes on disk.

#![cfg(feature = "parallel")]

use std::collections::BTreeMap;

use surtgis_algorithms::classification::{KmeansParams, kmeans_raster};
use surtgis_algorithms::hydrology::{
    FillSinksParams, HandParams, StreamNetworkParams, fill_sinks, flow_accumulation,
    flow_direction, hand, stream_network,
};
use surtgis_algorithms::landscape::{
    DiversityParams, landscape_metrics, shannon_diversity, simpson_diversity,
};
use surtgis_algorithms::segmentation::{FelzenszwalbParams, SlicParams, felzenszwalb, slic};
use surtgis_algorithms::statistics::{
    FocalParams, FocalStatistic, focal_statistics, global_morans_i, zonal_statistics,
};
use surtgis_algorithms::terrain::{
    AspectOutput, CurvatureParams, GeomorphonParams, HillshadeParams, SlopeParams, SlopeUnits,
    TpiParams, ViewshedParams, aspect, curvature, geomorphons, hillshade, slope, tpi, twi,
    viewshed,
};
use surtgis_algorithms::texture::{GlcmParams, haralick_glcm};
use surtgis_core::io::raster_data_hash;
use surtgis_core::{GeoTransform, Raster};

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Two hills, a valley, a closed depression and cell-level noise: enough
/// relief for every algorithm to produce non-trivial output, and enough
/// ties (integer-quantised classes) to exercise tie-breaking.
fn dem(rows: usize, cols: usize) -> Raster<f64> {
    let mut r = Raster::<f64>::new(rows, cols);
    let mut seed = 7u64;
    for i in 0..rows {
        for j in 0..cols {
            let (y, x) = (i as f64 / rows as f64, j as f64 / cols as f64);
            let hills = 60.0 * (-((x - 0.3).powi(2) + (y - 0.4).powi(2)) / 0.03).exp()
                + 45.0 * (-((x - 0.75).powi(2) + (y - 0.6).powi(2)) / 0.02).exp();
            let pit = -15.0 * (-((x - 0.5).powi(2) + (y - 0.5).powi(2)) / 0.003).exp();
            let noise = 0.8 * lcg(&mut seed);
            r.data_mut()[[i, j]] = 300.0 + hills + pit + 20.0 * y + noise;
        }
    }
    r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
    r.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
    r.set_nodata(Some(-9999.0));
    r
}

/// Five classes quantised from elevation (many exact ties).
fn classes(dem: &Raster<f64>) -> Raster<f64> {
    let mut c = dem.clone();
    c.data_mut()
        .mapv_inplace(|v| ((v - 300.0) / 25.0).floor().clamp(0.0, 4.0));
    c.set_nodata(None);
    c
}

fn zones(dem: &Raster<f64>) -> Raster<i32> {
    let (rows, cols) = dem.shape();
    let mut z = Raster::<i32>::new(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            z.data_mut()[[i, j]] = ((i / 15) * 3 + j / 20) as i32;
        }
    }
    z.set_transform(*dem.transform());
    z
}

fn h64(r: &Raster<f64>) -> String {
    raster_data_hash(&[r]).data_blake3
}
fn h8(r: &Raster<u8>) -> String {
    raster_data_hash(&[r]).data_blake3
}
fn h32(r: &Raster<i32>) -> String {
    raster_data_hash(&[r]).data_blake3
}

/// Every algorithm in the contract, as `(name, digest-or-debug)`.
fn suite(dem: &Raster<f64>, cls: &Raster<f64>, zn: &Raster<i32>) -> Vec<(&'static str, String)> {
    let (rows, cols) = dem.shape();
    let mut out = Vec::new();

    out.push(("slope", h64(&slope(dem, SlopeParams::default()).unwrap())));
    out.push(("aspect", h64(&aspect(dem, AspectOutput::Degrees).unwrap())));
    out.push((
        "hillshade",
        h64(&hillshade(dem, HillshadeParams::default()).unwrap()),
    ));
    out.push((
        "curvature",
        h64(&curvature(dem, CurvatureParams::default()).unwrap()),
    ));
    out.push(("tpi", h64(&tpi(dem, TpiParams::default()).unwrap())));
    out.push((
        "geomorphons",
        h8(&geomorphons(dem, {
            let mut p = GeomorphonParams::default();
            p.radius = 6;
            p
        })
        .unwrap()),
    ));
    out.push((
        "viewshed",
        h8(&viewshed(dem, {
            let mut p = ViewshedParams::default();
            p.observer_row = rows / 2;
            p.observer_col = cols / 3;
            p
        })
        .unwrap()),
    ));

    let filled = fill_sinks(dem, FillSinksParams { min_slope: 0.0 }).unwrap();
    out.push(("fill_sinks", h64(&filled)));
    let dir = flow_direction(&filled).unwrap();
    out.push(("flow_direction", h8(&dir)));
    let acc = flow_accumulation(&dir).unwrap();
    out.push(("flow_accumulation", h64(&acc)));
    out.push((
        "stream_network",
        h8(&stream_network(
            &acc,
            StreamNetworkParams {
                threshold: 40.0,
                ..StreamNetworkParams::default()
            },
        )
        .unwrap()),
    ));
    out.push((
        "hand",
        h64(&hand(&filled, &dir, &acc, {
            let mut p = HandParams::default();
            p.stream_threshold = 40.0;
            p
        })
        .unwrap()),
    ));
    let slope_rad = slope(&filled, {
        let mut p = SlopeParams::default();
        p.units = SlopeUnits::Radians;
        p
    })
    .unwrap();
    out.push(("twi", h64(&twi(&acc, &slope_rad).unwrap())));

    for (name, stat) in [
        ("focal_mean", FocalStatistic::Mean),
        ("focal_stddev", FocalStatistic::StdDev),
    ] {
        out.push((
            name,
            h64(&focal_statistics(dem, {
                let mut p = FocalParams::default();
                p.radius = 3;
                p.statistic = stat;
                p.circular = true;
                p
            })
            .unwrap()),
        ));
    }
    out.push(("morans_i", format!("{:?}", global_morans_i(dem).unwrap())));
    let zs: BTreeMap<i32, String> = zonal_statistics(dem, zn)
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, format!("{v:?}")))
        .collect();
    out.push(("zonal_statistics", format!("{zs:?}")));

    let dp = DiversityParams {
        radius: 3,
        ..DiversityParams::default()
    };
    out.push((
        "shannon_diversity",
        h64(&shannon_diversity(cls, dp.clone()).unwrap()),
    ));
    out.push((
        "simpson_diversity",
        h64(&simpson_diversity(cls, dp).unwrap()),
    ));
    out.push((
        "landscape_metrics",
        format!("{:?}", landscape_metrics(cls).unwrap()),
    ));

    out.push((
        "kmeans",
        h64(&kmeans_raster(dem, KmeansParams::default()).unwrap()),
    ));
    out.push((
        "slic",
        h32(&slic(
            &[dem, cls],
            SlicParams {
                n_segments: 40,
                ..SlicParams::default()
            },
        )
        .unwrap()),
    ));
    out.push((
        "felzenszwalb",
        h32(&felzenszwalb(&[dem, cls], FelzenszwalbParams::default()).unwrap()),
    ));
    out.push((
        "glcm",
        h64(&haralick_glcm(dem, GlcmParams::default()).unwrap()),
    ));
    out
}

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap()
}

fn assert_same(a: &[(&str, String)], b: &[(&str, String)], what: &str) {
    assert_eq!(a.len(), b.len());
    let diff: Vec<&str> = a
        .iter()
        .zip(b)
        .filter(|(x, y)| x.1 != y.1)
        .map(|(x, _)| x.0)
        .collect();
    assert!(diff.is_empty(), "{what}: outputs differ for {diff:?}");
}

#[test]
fn repeated_runs_in_one_process_are_bit_identical() {
    let d = dem(90, 120);
    let (c, z) = (classes(&d), zones(&d));
    let p = pool(4);
    let first = p.install(|| suite(&d, &c, &z));
    let second = p.install(|| suite(&d, &c, &z));
    assert_same(&first, &second, "run 1 vs run 2");
}

#[test]
fn thread_count_does_not_change_results() {
    let d = dem(90, 120);
    let (c, z) = (classes(&d), zones(&d));
    let one = pool(1).install(|| suite(&d, &c, &z));
    let four = pool(4).install(|| suite(&d, &c, &z));
    assert_same(&one, &four, "1 thread vs 4 threads");
}
