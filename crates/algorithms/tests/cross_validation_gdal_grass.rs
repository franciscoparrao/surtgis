//! Cross-validation of SurtGIS slope/aspect against GDAL and GRASS GIS.
//!
//! Compares pixel-by-pixel outputs on the Andes Chile 30m DEM (UTM Zone 19S).
//!
//! The original DEM (EPSG:4326) is reprojected to UTM (EPSG:32719) so that
//! cellsize is in meters, avoiding geographic-CRS scale ambiguities.
//!
//! Reference outputs generated with:
//! ```bash
//! gdalwarp -t_srs EPSG:32719 -r bilinear andes_chile_30m.tif andes_chile_30m_utm.tif
//! gdaldem slope andes_chile_30m_utm.tif gdal_slope.tif
//! gdaldem aspect andes_chile_30m_utm.tif gdal_aspect.tif
//! grass -c andes_chile_30m_utm.tif grassdb --exec bash -c "
//!   r.in.gdal input=andes_chile_30m_utm.tif output=dem
//!   r.slope.aspect elevation=dem slope=slope aspect=aspect -n
//!   r.out.gdal input=slope output=grass_slope.tif
//!   r.out.gdal input=aspect output=grass_aspect.tif
//! "
//! ```
//!
//! All three tools use Horn's (1981) 3x3 finite-difference method.
//!
//! Acceptance criteria (task 4.7):
//! - Slope RMSE < 0.5° vs GDAL and GRASS
//! - Aspect RMSE < 1.0° vs GDAL and GRASS (angular distance)

use surtgis_algorithms::terrain::{aspect, slope, AspectOutput, SlopeParams, SlopeUnits};
use surtgis_core::io::read_geotiff;
use surtgis_core::raster::Raster;
use std::path::{Path, PathBuf};

// ── Fixture paths (relative to workspace root) ────────────────────────

const DEM_UTM: &str = "tests/fixtures/andes_chile_30m_utm.tif";
const GDAL_SLOPE: &str = "tests/fixtures/gdal_slope.tif";
const GDAL_ASPECT: &str = "tests/fixtures/gdal_aspect.tif";
const GRASS_SLOPE: &str = "tests/fixtures/grass_slope.tif";
const GRASS_ASPECT: &str = "tests/fixtures/grass_aspect.tif";

fn workspace_root() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().to_path_buf()
}

fn load_raster(relative: &str) -> Option<Raster<f64>> {
    let path = workspace_root().join(relative);
    if !path.exists() {
        eprintln!("SKIPPING: fixture not found at {}", path.display());
        return None;
    }
    Some(read_geotiff::<f64, _>(&path, None).expect(&format!("failed to read {relative}")))
}

macro_rules! require {
    ($path:expr) => {
        match load_raster($path) {
            Some(r) => r,
            None => return,
        }
    };
}

// ── Statistics ─────────────────────────────────────────────────────────

struct ComparisonStats {
    n: usize,
    rmse: f64,
    mae: f64,
    max_abs_err: f64,
    mean_bias: f64,
}

/// Compare two rasters pixel-by-pixel, skipping 1-pixel border and nodata.
fn compare_rasters(
    a: &Raster<f64>,
    b: &Raster<f64>,
    is_nodata_a: impl Fn(f64) -> bool,
    is_nodata_b: impl Fn(f64) -> bool,
    angular: bool,
) -> ComparisonStats {
    let (rows, cols) = a.shape();
    assert_eq!(a.shape(), b.shape(), "shape mismatch");

    let mut sum_sq = 0.0;
    let mut sum_abs = 0.0;
    let mut sum_diff = 0.0;
    let mut max_abs = 0.0_f64;
    let mut n = 0usize;

    for row in 1..rows - 1 {
        for col in 1..cols - 1 {
            let va = a.get(row, col).unwrap();
            let vb = b.get(row, col).unwrap();

            if is_nodata_a(va) || is_nodata_b(vb) {
                continue;
            }

            let diff = if angular {
                let d = (va - vb).abs();
                if d > 180.0 { 360.0 - d } else { d }
            } else {
                va - vb
            };

            sum_sq += diff * diff;
            sum_abs += diff.abs();
            sum_diff += diff;
            max_abs = max_abs.max(diff.abs());
            n += 1;
        }
    }

    let nf = n as f64;
    ComparisonStats {
        n,
        rmse: (sum_sq / nf).sqrt(),
        mae: sum_abs / nf,
        max_abs_err: max_abs,
        mean_bias: sum_diff / nf,
    }
}

fn print_stats(label: &str, s: &ComparisonStats) {
    eprintln!(
        "  {label}: n={}, RMSE={:.4}°, MAE={:.4}°, MaxErr={:.2}°, Bias={:.6}°",
        s.n, s.rmse, s.mae, s.max_abs_err, s.mean_bias
    );
}

// ── Nodata predicates ──────────────────────────────────────────────────

/// SurtGIS slope: NaN = nodata/edge
fn is_slope_nodata(v: f64) -> bool {
    v.is_nan()
}

/// SurtGIS aspect: NaN or -1.0 = nodata/flat/edge
fn is_aspect_nodata_surtgis(v: f64) -> bool {
    v.is_nan() || v < 0.0
}

/// GDAL: NaN or -9999 = nodata; -1 = flat (aspect)
fn is_gdal_nodata(v: f64) -> bool {
    v.is_nan() || (v + 9999.0).abs() < 1.0
}

fn is_gdal_aspect_nodata(v: f64) -> bool {
    v.is_nan() || (v + 9999.0).abs() < 1.0 || v < 0.0
}

/// GRASS: NaN = nodata; also -9999 can appear from reprojection edges
fn is_grass_nodata(v: f64) -> bool {
    v.is_nan() || (v + 9999.0).abs() < 1.0
}

fn is_grass_aspect_nodata(v: f64) -> bool {
    v.is_nan() || (v + 9999.0).abs() < 1.0 || v < 0.0
}

// ── Slope tests ────────────────────────────────────────────────────────

#[test]
fn slope_vs_gdal() {
    let dem = require!(DEM_UTM);
    let gdal_ref = require!(GDAL_SLOPE);

    let surtgis_out = slope(
        &dem,
        SlopeParams { units: SlopeUnits::Degrees, z_factor: 1.0 },
    )
    .expect("surtgis slope failed");

    eprintln!("\n── Slope: SurtGIS vs GDAL (Horn, UTM) ──");
    let stats = compare_rasters(
        &surtgis_out, &gdal_ref,
        is_slope_nodata, is_gdal_nodata,
        false,
    );
    print_stats("slope vs GDAL", &stats);

    assert!(stats.n > 400_000, "Too few valid pixels: {}", stats.n);
    assert!(
        stats.rmse < 0.5,
        "Slope RMSE vs GDAL = {:.4}° exceeds 0.5° threshold",
        stats.rmse
    );
}

#[test]
fn slope_vs_grass() {
    let dem = require!(DEM_UTM);
    let grass_ref = require!(GRASS_SLOPE);

    let surtgis_out = slope(
        &dem,
        SlopeParams { units: SlopeUnits::Degrees, z_factor: 1.0 },
    )
    .expect("surtgis slope failed");

    eprintln!("\n── Slope: SurtGIS vs GRASS (Horn, UTM) ──");
    let stats = compare_rasters(
        &surtgis_out, &grass_ref,
        is_slope_nodata, is_grass_nodata,
        false,
    );
    print_stats("slope vs GRASS", &stats);

    assert!(stats.n > 400_000, "Too few valid pixels: {}", stats.n);
    assert!(
        stats.rmse < 0.5,
        "Slope RMSE vs GRASS = {:.4}° exceeds 0.5° threshold",
        stats.rmse
    );
}

// ── Aspect tests ───────────────────────────────────────────────────────

#[test]
fn aspect_vs_gdal() {
    let dem = require!(DEM_UTM);
    let gdal_ref = require!(GDAL_ASPECT);

    let surtgis_out = aspect(&dem, AspectOutput::Degrees).expect("surtgis aspect failed");

    eprintln!("\n── Aspect: SurtGIS vs GDAL (Horn, UTM) ──");
    let stats = compare_rasters(
        &surtgis_out, &gdal_ref,
        is_aspect_nodata_surtgis, is_gdal_aspect_nodata,
        true,
    );
    print_stats("aspect vs GDAL", &stats);

    assert!(stats.n > 400_000, "Too few valid pixels: {}", stats.n);
    assert!(
        stats.rmse < 1.0,
        "Aspect RMSE vs GDAL = {:.4}° exceeds 1.0° threshold",
        stats.rmse
    );
}

#[test]
fn aspect_vs_grass() {
    let dem = require!(DEM_UTM);
    let grass_ref = require!(GRASS_ASPECT);

    let surtgis_out = aspect(&dem, AspectOutput::Degrees).expect("surtgis aspect failed");

    eprintln!("\n── Aspect: SurtGIS vs GRASS (Horn -n, UTM) ──");
    let stats = compare_rasters(
        &surtgis_out, &grass_ref,
        is_aspect_nodata_surtgis, is_grass_aspect_nodata,
        true,
    );
    print_stats("aspect vs GRASS", &stats);

    assert!(stats.n > 400_000, "Too few valid pixels: {}", stats.n);
    assert!(
        stats.rmse < 1.0,
        "Aspect RMSE vs GRASS = {:.4}° exceeds 1.0° threshold",
        stats.rmse
    );
}

// ── Control: GDAL vs GRASS (both are "ground truth") ──────────────────

#[test]
fn control_gdal_vs_grass_slope() {
    let gdal_ref = require!(GDAL_SLOPE);
    let grass_ref = require!(GRASS_SLOPE);

    eprintln!("\n── Control: GDAL vs GRASS slope ──");
    let stats = compare_rasters(
        &gdal_ref, &grass_ref,
        is_gdal_nodata, is_grass_nodata,
        false,
    );
    print_stats("GDAL vs GRASS slope", &stats);

    assert!(
        stats.rmse < 0.01,
        "GDAL vs GRASS slope RMSE = {:.6}° (expected < 0.01°)",
        stats.rmse
    );
}

#[test]
fn control_gdal_vs_grass_aspect() {
    let gdal_ref = require!(GDAL_ASPECT);
    let grass_ref = require!(GRASS_ASPECT);

    eprintln!("\n── Control: GDAL vs GRASS aspect ──");
    let stats = compare_rasters(
        &gdal_ref, &grass_ref,
        is_gdal_aspect_nodata, is_grass_aspect_nodata,
        true,
    );
    print_stats("GDAL vs GRASS aspect", &stats);

    assert!(
        stats.rmse < 0.5,
        "GDAL vs GRASS aspect RMSE = {:.4}° (expected < 0.5°)",
        stats.rmse
    );
}
