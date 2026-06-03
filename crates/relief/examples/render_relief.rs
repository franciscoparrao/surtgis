//! End-to-end relief render through [`ReliefBuilder`].
//!
//! Reads `dem_filled.tif` from the repo root, builds a full relief stack
//! (terrain base + sphere shade + ray-traced shadows + ambient occlusion)
//! and writes a PNG to `output/relief_demo.png`.
//!
//! Run from the repo root:
//!
//! ```bash
//! cargo run --release -p surtgis-relief --example render_relief
//! ```

use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::terrain::HillshadeParams;
use surtgis_core::io::read_geotiff;
use surtgis_relief::{
    ColorScheme, RayShadeParams, ReliefBuilder, ambient_shade, ray_shade, sphere_shade,
};

fn main() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("locate repo root")
        .to_path_buf();
    let dem_path = repo_root.join("dem_filled.tif");
    let out_path = repo_root.join("output/relief_demo.png");

    println!("DEM: {}", dem_path.display());
    let dem = read_geotiff::<f64, _>(&dem_path, None).expect("read DEM");
    let (rows, cols) = dem.shape();
    println!("shape: {} x {}", rows, cols);

    let t0 = Instant::now();
    let sphere = sphere_shade(
        &dem,
        HillshadeParams {
            azimuth: 315.0,
            altitude: 45.0,
            z_factor: 1.0,
            normalized: true,
        },
    )
    .expect("sphere_shade");
    println!("sphere_shade: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let shadow = ray_shade(
        &dem,
        &RayShadeParams::with_soft_shadow_altitude(315.0, 40.0, 50.0, 11),
    )
    .expect("ray_shade");
    println!("ray_shade   : {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let ambient = ambient_shade(&dem, 30).expect("ambient_shade");
    println!("ambient_shade: {:.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let img = ReliefBuilder::new(&dem)
        .base_colormap(ColorScheme::Terrain)
        .add_shade(sphere, 0.6)
        .add_shadow(shadow, 0.7)
        .add_ambient(ambient, 0.3)
        .render()
        .expect("render");
    println!("compose     : {:.3}s", t0.elapsed().as_secs_f64());

    if let Some(parent) = out_path.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    img.save_png(&out_path).expect("save png");
    println!("wrote {}", out_path.display());
}
