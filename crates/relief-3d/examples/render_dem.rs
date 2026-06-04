//! M2 acceptance: render a real DEM interactively.
//!
//! Loads `dem_filled.tif` from the repo root, runs the rayshader-style
//! 2D recipe through `surtgis-relief::ReliefBuilder` to produce an
//! RGBA texture, then renders the displaced + textured mesh in a
//! winit window with the OrbitCamera.
//!
//! Run from the repo root:
//!
//! ```bash
//! cargo run --release -p surtgis-relief-3d --example render_dem
//! ```
//!
//! Controls:
//!   - Left-drag       — rotate
//!   - Right-drag      — pan
//!   - Scroll wheel    — zoom
//!   - Close window    — exit

use std::path::PathBuf;
use std::time::Instant;

use surtgis_algorithms::terrain::HillshadeParams;
use surtgis_core::io::read_geotiff;
use surtgis_relief::{
    ColorScheme, RayShadeParams, ReliefBuilder, ambient_shade, ray_shade, sphere_shade,
};
use surtgis_relief_3d::mesh::from_dem;
#[cfg(not(target_arch = "wasm32"))]
use surtgis_relief_3d::native::run_viewer;

const VERTICAL_EXAGGERATION: f32 = 0.45;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    env_logger::init();
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("locate repo root")
        .to_path_buf();
    let dem_path = repo_root.join("dem_filled.tif");

    eprintln!("loading DEM: {}", dem_path.display());
    let dem = read_geotiff::<f64, _>(&dem_path, None).expect("read DEM");
    let (rows, cols) = dem.shape();
    eprintln!(
        "DEM shape: {rows}x{cols} ({} vertices, {} triangles)",
        rows * cols,
        (rows - 1) * (cols - 1) * 2,
    );

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
    let shadow = ray_shade(
        &dem,
        &RayShadeParams::with_soft_shadow_altitude(315.0, 40.0, 50.0, 11),
    )
    .expect("ray_shade");
    let ao = ambient_shade(&dem, 20).expect("ambient_shade");
    let img = ReliefBuilder::new(&dem)
        .base_colormap(ColorScheme::Terrain)
        .add_shade(sphere, 0.6)
        .add_shadow(shadow, 0.7)
        .add_ambient(ao, 0.3)
        .render()
        .expect("compose");
    eprintln!(
        "relief composite computed in {:.2}s",
        t0.elapsed().as_secs_f64()
    );

    let t0 = Instant::now();
    let (verts, idx) = from_dem(&dem, VERTICAL_EXAGGERATION);
    eprintln!(
        "mesh built in {:.2}s — {} vertices, vertical exaggeration {}",
        t0.elapsed().as_secs_f64(),
        verts.len(),
        VERTICAL_EXAGGERATION
    );

    let width = img.width as u32;
    let height = img.height as u32;
    if let Err(e) = run_viewer(
        verts,
        idx,
        img.pixels,
        width,
        height,
        "surtgis-relief-3d — render_dem (M2)",
    ) {
        eprintln!("viewer error: {e}");
        std::process::exit(1);
    }
}

#[cfg(target_arch = "wasm32")]
fn main() {}
