//! Benchmark sky_view_factor on dem_filled.tif. Used as the baseline
//! reference for the P-ambient sprint.
//!
//! Run from the repo root:
//!   cargo run --release -p surtgis-algorithms --example bench_svf

use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::terrain::{SvfParams, sky_view_factor, sky_view_factor_fast};
use surtgis_core::io::read_geotiff;

const N_REPS: usize = 3;

fn main() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("locate repo root")
        .to_path_buf();
    let path = repo.join("dem_filled.tif");
    let dem = read_geotiff::<f64, _>(&path, None).expect("read DEM");
    let (rows, cols) = dem.shape();
    eprintln!("DEM: {rows} x {cols} ({} cells)", rows * cols);

    // Warm-up
    let _ = sky_view_factor(
        &dem,
        SvfParams {
            radius: 20,
            directions: 16,
        },
    )
    .unwrap();

    eprintln!("warm-up complete; timing {N_REPS} reps");
    let mut samples = Vec::with_capacity(N_REPS);
    for r in 1..=N_REPS {
        let t = Instant::now();
        let _svf = sky_view_factor(
            &dem,
            SvfParams {
                radius: 20,
                directions: 16,
            },
        )
        .unwrap();
        let s = t.elapsed().as_secs_f64();
        eprintln!("  rep {r}: {s:.3}s");
        samples.push(s);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_orig = samples[samples.len() / 2];
    eprintln!("median (sky_view_factor):      {med_orig:.3}s");

    // Fast version.
    let _ = sky_view_factor_fast(
        &dem,
        SvfParams {
            radius: 20,
            directions: 16,
        },
    )
    .unwrap();
    let mut s2 = Vec::with_capacity(N_REPS);
    for r in 1..=N_REPS {
        let t = Instant::now();
        let _svf = sky_view_factor_fast(
            &dem,
            SvfParams {
                radius: 20,
                directions: 16,
            },
        )
        .unwrap();
        let s = t.elapsed().as_secs_f64();
        eprintln!("  fast rep {r}: {s:.3}s");
        s2.push(s);
    }
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med_fast = s2[s2.len() / 2];
    eprintln!("median (sky_view_factor_fast): {med_fast:.3}s");
    eprintln!("speedup: {:.2}×", med_orig / med_fast);

    // Equivalence sample: max abs delta over the whole DEM.
    let orig = sky_view_factor(
        &dem,
        SvfParams {
            radius: 20,
            directions: 16,
        },
    )
    .unwrap();
    let fast = sky_view_factor_fast(
        &dem,
        SvfParams {
            radius: 20,
            directions: 16,
        },
    )
    .unwrap();
    let mut max_abs = 0f64;
    let mut sum_abs = 0f64;
    let mut n = 0usize;
    for (a, b) in orig.data().iter().zip(fast.data().iter()) {
        if a.is_nan() || b.is_nan() {
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d;
        n += 1;
    }
    let mean_abs = sum_abs / n.max(1) as f64;
    eprintln!("equivalence over {n} cells: max abs delta = {max_abs:.3e}, mean = {mean_abs:.3e}");
}
