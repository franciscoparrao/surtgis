//! Probe horizon variants and radii to find the right ray_shade strategy.
//!
//! Runs:
//!   - horizon_angle_map (non-fast) at radius 50, 100, 200
//!   - horizon_angle_map_fast at radius 100, 500, with near_distance 32
//!   - horizon_angles precompute with 12 directions
//!
//! Outputs wall-clock for one azimuth (315°), then multiplies by 11 to
//! project the multi-sun cost.

use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::terrain::{
    HorizonParams, horizon_angle_map, horizon_angle_map_fast, horizon_angles,
};
use surtgis_core::io::read_geotiff;

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf();
    let dem = read_geotiff::<f64, _>(&repo_root.join("dem_filled.tif"), None).unwrap();
    let (rows, cols) = dem.shape();
    println!("DEM: {rows} x {cols} ({} cells)", rows * cols);
    println!("All times are after a warmup call.\n");

    // Warmup the JIT-equivalent (cache, rayon thread pool start, etc.)
    let _ = horizon_angle_map(&dem, 0.0, 100).unwrap();

    macro_rules! time {
        ($name:expr, $call:expr) => {{
            let t = Instant::now();
            let _r = $call;
            let s = t.elapsed().as_secs_f64();
            println!("  {:<50} : {:>7.3}s  (x11 = {:>7.3}s)", $name, s, s * 11.0);
        }};
    }

    println!("== horizon_angle_map (non-fast) ==");
    time!(
        "radius=50",
        horizon_angle_map(&dem, 315f64.to_radians(), 50)
    );
    time!(
        "radius=100",
        horizon_angle_map(&dem, 315f64.to_radians(), 100)
    );
    time!(
        "radius=200",
        horizon_angle_map(&dem, 315f64.to_radians(), 200)
    );
    time!(
        "radius=500",
        horizon_angle_map(&dem, 315f64.to_radians(), 500)
    );

    println!("\n== horizon_angle_map_fast (LOD pyramid) ==");
    time!(
        "radius=100, near=32",
        horizon_angle_map_fast(&dem, 315f64.to_radians(), 100, 32)
    );
    time!(
        "radius=500, near=32",
        horizon_angle_map_fast(&dem, 315f64.to_radians(), 500, 32)
    );
    time!(
        "radius=1000, near=32",
        horizon_angle_map_fast(&dem, 315f64.to_radians(), 1000, 32)
    );

    println!("\n== horizon_angles precompute (one shot for many azimuths) ==");
    for &(r, d) in &[(100usize, 12usize), (100, 36), (500, 12), (500, 36)] {
        let t = Instant::now();
        let _h = horizon_angles(
            &dem,
            HorizonParams {
                radius: r,
                directions: d,
            },
        )
        .unwrap();
        let s = t.elapsed().as_secs_f64();
        println!(
            "  radius={:<4} dirs={:<3}  : {:>7.3}s   (gives all azimuths in one call)",
            r, d, s
        );
    }
}
