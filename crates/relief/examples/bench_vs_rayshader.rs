//! M2 acceptance benchmark: surtgis-relief vs rayshader on `dem_filled.tif`.
//!
//! Mirrors `benchmarks/rayshader_baseline.R` parameter-for-parameter so the
//! two numbers are directly comparable.
//!
//! Run from the repo root:
//!
//! ```bash
//! cargo run --release -p surtgis-relief --example bench_vs_rayshader
//! ```

use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::terrain::HillshadeParams;
use surtgis_core::io::read_geotiff;
use surtgis_relief::{RayShadeParams, ray_shade, sphere_shade};

const N_REPS: usize = 5;

// Same sun config as benchmarks/rayshader_baseline.R: azimuth 315°,
// anglebreaks seq(40, 50, 1) (11 altitude samples), so we emit
// 11 SunSamples at altitude_deg = 40..50 inclusive and a single azimuth.
fn rayshader_equivalent_params() -> RayShadeParams {
    let mut suns = Vec::with_capacity(11);
    for i in 0..11 {
        suns.push(surtgis_relief::SunSample {
            azimuth_deg: 315.0,
            altitude_deg: 40.0 + i as f64,
        });
    }
    RayShadeParams {
        suns,
        radius: 850, // ~max(rows, cols) for rayshader-equivalent full-grid trace
    }
}

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("locate repo root")
        .to_path_buf();
    let dem_path = repo_root.join("dem_filled.tif");

    println!("DEM: {}", dem_path.display());
    let dem = read_geotiff::<f64, _>(&dem_path, None).expect("read DEM");
    let (rows, cols) = dem.shape();
    println!("shape: {} x {} ({} cells)", rows, cols, rows * cols);
    println!("reps : {} timed (+ 1 warmup)", N_REPS);
    println!("config: 11 sun samples (rayshader anglebreaks seq(40, 50, 1)), az 315°");

    let ray_params = rayshader_equivalent_params();
    let sphere_params = {
        let mut p = HillshadeParams::default();
        p.azimuth = 315.0;
        p.altitude = 45.0;
        p.z_factor = 1.0;
        p.normalized = true;
        p
    };

    // Warmup
    println!("\n[warmup]");
    let _ = ray_shade(&dem, &ray_params).expect("ray_shade warmup");
    let _ = sphere_shade(&dem, sphere_params.clone()).expect("sphere_shade warmup");

    let mut ray_times = Vec::with_capacity(N_REPS);
    let mut sphere_times = Vec::with_capacity(N_REPS);
    let mut total_times = Vec::with_capacity(N_REPS);

    for rep in 1..=N_REPS {
        let t0 = Instant::now();
        let _ray = ray_shade(&dem, &ray_params).expect("ray_shade");
        let t_ray = t0.elapsed().as_secs_f64();

        let t0 = Instant::now();
        let _sph = sphere_shade(&dem, sphere_params.clone()).expect("sphere_shade");
        let t_sph = t0.elapsed().as_secs_f64();

        let total = t_ray + t_sph;
        println!(
            "[rep {}/{}] ray={:.3}s  sphere={:.3}s  total={:.3}s",
            rep, N_REPS, t_ray, t_sph, total
        );
        ray_times.push(t_ray);
        sphere_times.push(t_sph);
        total_times.push(total);
    }

    fn summarise(name: &str, xs: &mut Vec<f64>) {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        let median = xs[n / 2];
        let q1 = xs[n / 4];
        let q3 = xs[(3 * n) / 4];
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
        let sd = var.sqrt();
        println!(
            "  {:<13}: median={:.3}s  IQR=[{:.3}, {:.3}]  mean={:.3}s  sd={:.3}s",
            name, median, q1, q3, mean, sd
        );
    }

    println!("\n--- summary ---");
    summarise("ray_shade", &mut ray_times);
    summarise("sphere_shade", &mut sphere_times);
    summarise("TOTAL", &mut total_times);

    let our_median = {
        let mut v = total_times.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let rayshader_median = 1.80;
    let speedup = rayshader_median / our_median;
    println!(
        "\nM2 acceptance:\n  rayshader baseline TOTAL median: {:.3}s\n  surtgis-relief TOTAL median:    {:.3}s\n  speedup: {:.2}x\n  target  : >= 2.00x  ({})\n  stretch : >= 4.00x  ({})",
        rayshader_median,
        our_median,
        speedup,
        if speedup >= 2.0 { "PASS" } else { "FAIL" },
        if speedup >= 4.0 {
            "PASS"
        } else {
            "(stretch not met)"
        },
    );
}
