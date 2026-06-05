//! `surtgis relief-3d` — headless 3D PNG screenshot.
//!
//! Same 2D rayshader recipe as `surtgis relief`, then the resulting
//! RGBA layer is draped on a displaced DEM mesh and rendered into a
//! `width × height` offscreen wgpu texture. No window, no display
//! required — runs on CI / containers / SSH.

use anyhow::{Context, Result, anyhow};
use std::path::Path;

use surtgis_algorithms::terrain::HillshadeParams;
use surtgis_core::io::read_geotiff;
use surtgis_relief::{
    ColorScheme, RayShadeParams, ReliefBuilder, RgbaImage, ambient_shade, ray_shade, save_png,
    sphere_shade,
};
use surtgis_relief_3d::headless::{HeadlessConfig, render_to_rgba};

#[allow(clippy::too_many_arguments)]
pub fn handle(
    input: &Path,
    output: &Path,
    colormap: &str,
    width: u32,
    height: u32,
    sun_azimuth: f64,
    sun_altitude: f64,
    shadows: bool,
    soft: usize,
    ambient: bool,
    vertical_exaggeration: f32,
    camera_azimuth: f32,
    camera_polar: f32,
    camera_distance: f32,
    haze: f32,
) -> Result<()> {
    let scheme = parse_scheme(colormap)?;

    let dem = read_geotiff::<f64, _>(input, None)
        .with_context(|| format!("read DEM: {}", input.display()))?;
    let (rows, cols) = dem.shape();

    // 2D recipe — same as `surtgis relief`.
    let sphere = sphere_shade(
        &dem,
        HillshadeParams {
            azimuth: sun_azimuth,
            altitude: sun_altitude,
            z_factor: 1.0,
            normalized: true,
        },
    )
    .context("sphere_shade failed")?;

    let mut builder = ReliefBuilder::new(&dem)
        .base_colormap(scheme)
        .add_shade(sphere, 0.6);

    if shadows {
        let params = if soft > 1 {
            let low = (sun_altitude - 5.0).max(0.5);
            let high = (sun_altitude + 5.0).min(89.0);
            RayShadeParams::with_soft_shadow_altitude(sun_azimuth, low, high, soft)
        } else {
            RayShadeParams {
                suns: vec![surtgis_relief::SunSample::new(sun_azimuth, sun_altitude)],
                radius: 0,
            }
        };
        let params = RayShadeParams {
            suns: params.suns,
            radius: rows.max(cols),
        };
        let shadow = ray_shade(&dem, &params).context("ray_shade failed")?;
        builder = builder.add_shadow(shadow, 0.7);
    }
    if ambient {
        let ao = ambient_shade(&dem, 30).context("ambient_shade failed")?;
        builder = builder.add_ambient(ao, 0.3);
    }

    let texture = builder.render().context("composite render failed")?;

    // Headless 3D render.
    let cfg = HeadlessConfig {
        width,
        height,
        sun_azimuth_deg: sun_azimuth as f32,
        sun_altitude_deg: sun_altitude as f32,
        ambient: 0.4,
        vertical_scale: 1.0,
        vertical_exaggeration,
        camera_azimuth_deg: camera_azimuth,
        camera_polar_deg: camera_polar,
        camera_distance,
        fov_deg: 45.0,
        haze_density: haze.clamp(0.0, 1.0),
        haze_rgb: [0.78, 0.83, 0.88],
    };
    let rgba =
        render_to_rgba(&dem, &texture, &cfg).map_err(|e| anyhow!("3D render failed: {e}"))?;

    // PNG output via the surtgis-colormap encoder.
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir: {}", parent.display()))?;
    }
    let img = RgbaImage::from_rgba(width as usize, height as usize, rgba)
        .map_err(|e| anyhow!("RgbaImage::from_rgba: {e}"))?;
    save_png(output, width, height, &img.pixels)
        .with_context(|| format!("write PNG: {}", output.display()))?;
    tracing::info!("wrote {}", output.display());
    Ok(())
}

fn parse_scheme(s: &str) -> Result<ColorScheme> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "terrain" => ColorScheme::Terrain,
        "divergent" => ColorScheme::Divergent,
        "grayscale" | "greyscale" => ColorScheme::Grayscale,
        "ndvi" => ColorScheme::Ndvi,
        "blue-white-red" | "bwr" => ColorScheme::BlueWhiteRed,
        "geomorphons" => ColorScheme::Geomorphons,
        "water" => ColorScheme::Water,
        "accumulation" => ColorScheme::Accumulation,
        "imhof1" | "imhof" => ColorScheme::Imhof1,
        "imhof2" => ColorScheme::Imhof2,
        "imhof3" => ColorScheme::Imhof3,
        "imhof4" => ColorScheme::Imhof4,
        "bw1" => ColorScheme::Bw1,
        "bw2" => ColorScheme::Bw2,
        "desert-dry" | "desert" => ColorScheme::DesertDry,
        "pastel" => ColorScheme::Pastel,
        other => {
            return Err(anyhow!(
                "unknown colormap '{other}'. Valid: terrain, divergent, grayscale, ndvi, bwr, geomorphons, water, accumulation, imhof1..imhof4, bw1, bw2, desert, pastel"
            ));
        }
    })
}
