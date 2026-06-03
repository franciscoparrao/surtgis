//! `surtgis relief` — full rayshader-style shaded relief composite.
//!
//! Stages (each optional via flags):
//!   1. Coloured base via colormap of the DEM.
//!   2. Normal-based sphere shade (always on).
//!   3. Ray-traced cast shadows (`--shadows`, with `--soft N` samples).
//!   4. Ambient occlusion via SVF (`--ambient`).
//!   5. Water mask (`--water`).
//!
//! Writes a PNG. Native target only — the `image` crate dependency lives
//! behind `cfg(not(target_arch = "wasm32"))` in `surtgis-colormap`.

use anyhow::{Context, Result, anyhow};
use std::path::Path;

use surtgis_algorithms::terrain::HillshadeParams;
use surtgis_core::io::read_geotiff;
use surtgis_relief::{
    ColorScheme, RayShadeParams, ReliefBuilder, SunSample, WaterParams, ambient_shade,
    detect_water, ray_shade, sphere_shade,
};

#[allow(clippy::too_many_arguments)]
pub fn handle(
    input: &Path,
    output: &Path,
    colormap: &str,
    sun_azimuth: f64,
    sun_altitude: f64,
    shadows: bool,
    soft: usize,
    ambient: bool,
    water: bool,
    z_factor: f64,
    radius: usize,
) -> Result<()> {
    let scheme = parse_scheme(colormap)?;

    let dem = read_geotiff::<f64, _>(input, None)
        .with_context(|| format!("read DEM: {}", input.display()))?;

    let sphere_params = HillshadeParams {
        azimuth: sun_azimuth,
        altitude: sun_altitude,
        z_factor,
        normalized: true,
    };
    let sphere = sphere_shade(&dem, sphere_params).context("sphere_shade failed")?;

    let mut builder = ReliefBuilder::new(&dem)
        .base_colormap(scheme)
        .add_shade(sphere, 0.6);

    if shadows {
        let params = if soft > 1 {
            // Soft-shadow penumbra over an altitude window of ±5° centred
            // on the sun altitude, with `soft` samples. Keeps a single
            // azimuth so the amortised ray_shade path kicks in.
            let low = (sun_altitude - 5.0).max(0.5);
            let high = (sun_altitude + 5.0).min(89.0);
            RayShadeParams::with_soft_shadow_altitude(sun_azimuth, low, high, soft)
        } else {
            RayShadeParams {
                suns: vec![SunSample::new(sun_azimuth, sun_altitude)],
                radius: 0, // overwritten below
            }
        };
        let params = RayShadeParams {
            suns: params.suns,
            radius: cast_radius(&dem),
        };
        let shadow = ray_shade(&dem, &params).context("ray_shade failed")?;
        builder = builder.add_shadow(shadow, 0.7);
    }

    if ambient {
        let ao = ambient_shade(&dem, radius).context("ambient_shade failed")?;
        builder = builder.add_ambient(ao, 0.3);
    }

    if water {
        let mask = detect_water(&dem, &WaterParams::default()).context("detect_water failed")?;
        builder = builder.add_water(mask, ColorScheme::Water);
    }

    let img = builder.render().context("composite render failed")?;

    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir: {}", parent.display()))?;
    }
    img.save_png(output)
        .with_context(|| format!("write PNG: {}", output.display()))?;
    tracing::info!("wrote {}", output.display());
    Ok(())
}

/// Lower-case parser for the `--colormap` flag. Mirrors
/// [`ColorScheme::ALL`] / [`ColorScheme::name`] but with the names users
/// actually type at the shell.
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
        other => {
            return Err(anyhow!(
                "unknown colormap '{other}'. Valid: terrain, divergent, grayscale, ndvi, bwr, geomorphons, water, accumulation"
            ));
        }
    })
}

/// Pick a cast-shadow radius proportional to the DEM extent. `max(rows,
/// cols)` matches the rayshader convention of "trace until you've crossed
/// the heightmap".
fn cast_radius(dem: &surtgis_core::raster::Raster<f64>) -> usize {
    let (rows, cols) = dem.shape();
    rows.max(cols)
}
