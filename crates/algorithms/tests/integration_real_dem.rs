//! Integration tests using a real DEM (Copernicus 30m, Andes, Chile).
//!
//! These tests require the fixture file `tests/fixtures/andes_chile_30m.tif`
//! at the workspace root. If the file doesn't exist, all tests are skipped.
//!
//! The DEM covers a 720×720 pixel area of the Chilean Andes near Santiago,
//! with elevations ranging from ~2858m to ~5981m at 30m resolution.

use surtgis_algorithms::hydrology::{fill_sinks, flow_accumulation, flow_direction, FillSinksParams};
use surtgis_algorithms::statistics::{focal_statistics, FocalParams, FocalStatistic};
use surtgis_algorithms::terrain::{
    aspect, curvature, hillshade, slope, twi, AspectOutput, CurvatureParams, CurvatureType,
    HillshadeParams, SlopeParams, SlopeUnits,
};
use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_core::raster::Raster;
use std::path::Path;

/// Path to the DEM fixture relative to workspace root.
const DEM_FILENAME: &str = "tests/fixtures/andes_chile_30m.tif";

/// Resolve the DEM path from either workspace root or crate directory.
fn dem_path() -> std::path::PathBuf {
    // cargo test sets CARGO_MANIFEST_DIR to the crate directory.
    // The fixture lives at the workspace root.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent().unwrap().parent().unwrap();
    workspace_root.join(DEM_FILENAME)
}

/// Helper: load the DEM or skip the test if the fixture doesn't exist.
fn load_dem() -> Option<Raster<f64>> {
    let path = dem_path();
    if !path.exists() {
        eprintln!("SKIPPING: fixture not found at {}", path.display());
        return None;
    }
    Some(read_geotiff::<f64, _>(&path, None).expect("failed to read DEM fixture"))
}

/// Macro to skip tests when the DEM fixture is missing.
macro_rules! require_dem {
    () => {
        match load_dem() {
            Some(dem) => dem,
            None => return,
        }
    };
}

// ---------------------------------------------------------------------------
// I/O roundtrip
// ---------------------------------------------------------------------------

#[test]
fn io_roundtrip() {
    let dem = require_dem!();
    let (rows, cols) = dem.shape();

    assert!(rows > 0 && cols > 0, "DEM should have positive dimensions");
    assert_eq!((rows, cols), (720, 720), "expected 720×720 DEM");

    // Write to a temp file and re-read
    let tmp = tempfile::NamedTempFile::with_suffix(".tif").unwrap();
    write_geotiff(&dem, tmp.path(), None).expect("write failed");

    let reloaded: Raster<f64> = read_geotiff(tmp.path(), None).expect("re-read failed");
    assert_eq!(reloaded.shape(), dem.shape());

    // Compare a sample of values
    for r in [0, 100, 359, 500, 719] {
        for c in [0, 100, 359, 500, 719] {
            let orig = dem.get(r, c).unwrap();
            let copy = reloaded.get(r, c).unwrap();
            if orig.is_nan() {
                assert!(copy.is_nan(), "pixel ({r},{c}): expected NaN");
            } else {
                assert!(
                    (orig - copy).abs() < 0.01,
                    "pixel ({r},{c}): orig={orig}, copy={copy}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Terrain suite
// ---------------------------------------------------------------------------

#[test]
fn terrain_slope() {
    let dem = require_dem!();
    let result = slope(
        &dem,
        SlopeParams {
            units: SlopeUnits::Degrees,
            z_factor: 1.0,
        },
    )
    .expect("slope failed");

    assert_eq!(result.shape(), dem.shape());

    // Sanity: all valid slope values in [0, 90]
    let data = result.data();
    let mut valid = 0usize;
    for &v in data.iter() {
        if v.is_nan() {
            continue;
        }
        assert!(v >= 0.0, "slope should be >= 0, got {v}");
        assert!(v <= 90.0, "slope should be <= 90°, got {v}");
        valid += 1;
    }
    assert!(valid > 0, "should have valid slope values");

    // In mountainous Andes terrain, expect at least some steep slopes
    let stats = result.statistics();
    assert!(
        stats.max.unwrap() > 30.0,
        "Andes DEM should have slopes > 30°, max={}",
        stats.max.unwrap()
    );
}

#[test]
fn terrain_aspect() {
    let dem = require_dem!();
    let result = aspect(&dem, AspectOutput::Degrees).expect("aspect failed");

    assert_eq!(result.shape(), dem.shape());

    let data = result.data();
    let mut valid = 0usize;
    for &v in data.iter() {
        if v.is_nan() || v < 0.0 {
            // flat areas may get -1 or NaN
            continue;
        }
        assert!(v >= 0.0, "aspect should be >= 0, got {v}");
        assert!(v <= 360.0, "aspect should be <= 360°, got {v}");
        valid += 1;
    }
    assert!(valid > 0, "should have valid aspect values");
}

#[test]
fn terrain_hillshade() {
    let dem = require_dem!();
    let result = hillshade(&dem, HillshadeParams::default()).expect("hillshade failed");

    assert_eq!(result.shape(), dem.shape());

    let data = result.data();
    let mut valid = 0usize;
    for &v in data.iter() {
        if v.is_nan() {
            continue;
        }
        assert!(v >= 0.0, "hillshade should be >= 0, got {v}");
        assert!(v <= 255.0, "hillshade should be <= 255, got {v}");
        valid += 1;
    }
    assert!(valid > 0, "should have valid hillshade values");
}

// ---------------------------------------------------------------------------
// Curvature suite
// ---------------------------------------------------------------------------

#[test]
fn curvature_general() {
    let dem = require_dem!();
    let result = curvature(&dem, CurvatureParams::default()).expect("curvature failed");

    assert_eq!(result.shape(), dem.shape());

    let data = result.data();
    let mut valid = 0usize;
    for &v in data.iter() {
        if v.is_nan() {
            continue;
        }
        assert!(v.is_finite(), "curvature should be finite, got {v}");
        valid += 1;
    }
    assert!(valid > 0, "should have valid curvature values");
}

#[test]
fn curvature_profile() {
    let dem = require_dem!();
    let result = curvature(
        &dem,
        CurvatureParams {
            curvature_type: CurvatureType::Profile,
            ..Default::default()
        },
    )
    .expect("profile curvature failed");

    assert_eq!(result.shape(), dem.shape());

    let data = result.data();
    let valid = data.iter().filter(|v| !v.is_nan() && v.is_finite()).count();
    assert!(valid > 0, "should have valid profile curvature values");
}

#[test]
fn curvature_plan() {
    let dem = require_dem!();
    let result = curvature(
        &dem,
        CurvatureParams {
            curvature_type: CurvatureType::Plan,
            ..Default::default()
        },
    )
    .expect("plan curvature failed");

    assert_eq!(result.shape(), dem.shape());

    let data = result.data();
    let valid = data.iter().filter(|v| !v.is_nan() && v.is_finite()).count();
    assert!(valid > 0, "should have valid plan curvature values");
}

// ---------------------------------------------------------------------------
// Hydrology suite
// ---------------------------------------------------------------------------

#[test]
fn hydrology_fill_sinks() {
    let dem = require_dem!();
    let filled = fill_sinks(&dem, FillSinksParams::default()).expect("fill_sinks failed");

    assert_eq!(filled.shape(), dem.shape());

    // Filled DEM should be >= original DEM everywhere (sinks raised)
    let (rows, cols) = dem.shape();
    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            let orig = dem.get(r, c).unwrap();
            let fill = filled.get(r, c).unwrap();
            if orig.is_nan() || fill.is_nan() {
                continue;
            }
            assert!(
                fill >= orig - 1e-6,
                "filled ({r},{c}) = {fill} < original {orig}"
            );
        }
    }
}

#[test]
fn hydrology_flow_pipeline() {
    let dem = require_dem!();

    // Step 1: Fill sinks
    let filled = fill_sinks(&dem, FillSinksParams::default()).expect("fill_sinks failed");

    // Step 2: Flow direction
    let fdir = flow_direction(&filled).expect("flow_direction failed");
    assert_eq!(fdir.shape(), dem.shape());

    // Direction values should be 0-8
    let fdir_data = fdir.data();
    for &v in fdir_data.iter() {
        assert!(v <= 8, "flow direction should be 0-8, got {v}");
    }

    // Step 3: Flow accumulation
    let facc = flow_accumulation(&fdir).expect("flow_accumulation failed");
    assert_eq!(facc.shape(), dem.shape());

    // Accumulation should be >= 0
    let facc_data = facc.data();
    for &v in facc_data.iter() {
        if v.is_nan() {
            continue;
        }
        assert!(v >= 0.0, "flow_accumulation should be >= 0, got {v}");
    }

    // Max accumulation should be significant in a 720×720 raster
    let stats = facc.statistics();
    assert!(
        stats.max.unwrap() > 100.0,
        "max flow accumulation should be > 100, got {}",
        stats.max.unwrap()
    );
}

#[test]
fn hydrology_twi() {
    let dem = require_dem!();

    // Full pipeline: fill → flow_dir → flow_acc → slope_rad → TWI
    let filled = fill_sinks(&dem, FillSinksParams::default()).expect("fill");
    let fdir = flow_direction(&filled).expect("fdir");
    let facc = flow_accumulation(&fdir).expect("facc");
    let slope_rad = slope(
        &dem,
        SlopeParams {
            units: SlopeUnits::Radians,
            z_factor: 1.0,
        },
    )
    .expect("slope_rad");

    let result = twi(&facc, &slope_rad).expect("TWI failed");
    assert_eq!(result.shape(), dem.shape());

    let data = result.data();
    let mut valid = 0usize;
    for &v in data.iter() {
        if v.is_nan() {
            continue;
        }
        assert!(v.is_finite(), "TWI should be finite, got {v}");
        valid += 1;
    }
    assert!(valid > 0, "should have valid TWI values");
}

// ---------------------------------------------------------------------------
// Statistics suite
// ---------------------------------------------------------------------------

#[test]
fn statistics_focal_mean() {
    let dem = require_dem!();
    let result = focal_statistics(
        &dem,
        FocalParams {
            radius: 2,
            statistic: FocalStatistic::Mean,
            circular: false,
        },
    )
    .expect("focal_mean failed");

    assert_eq!(result.shape(), dem.shape());

    // Focal mean should be close to original values (smoothing effect)
    let dem_stats = dem.statistics();
    let focal_stats = result.statistics();

    // Mean of focal result should be close to mean of original
    let dem_mean = dem_stats.mean.unwrap();
    let focal_mean = focal_stats.mean.unwrap();
    assert!(
        (dem_mean - focal_mean).abs() < 50.0,
        "focal mean ({focal_mean}) should be close to DEM mean ({dem_mean})"
    );
}

#[test]
fn statistics_focal_std() {
    let dem = require_dem!();
    let result = focal_statistics(
        &dem,
        FocalParams {
            radius: 2,
            statistic: FocalStatistic::StdDev,
            circular: false,
        },
    )
    .expect("focal_std failed");

    assert_eq!(result.shape(), dem.shape());

    // Standard deviation should be non-negative
    let data = result.data();
    for &v in data.iter() {
        if v.is_nan() {
            continue;
        }
        assert!(v >= 0.0, "focal std should be >= 0, got {v}");
        assert!(v.is_finite(), "focal std should be finite, got {v}");
    }
}

// ---------------------------------------------------------------------------
// No NaN/Inf in outputs (comprehensive sanity)
// ---------------------------------------------------------------------------

#[test]
fn no_inf_in_outputs() {
    let dem = require_dem!();

    let slope_out = slope(
        &dem,
        SlopeParams {
            units: SlopeUnits::Degrees,
            z_factor: 1.0,
        },
    )
    .expect("slope");
    let aspect_out = aspect(&dem, AspectOutput::Degrees).expect("aspect");
    let hs_out = hillshade(&dem, HillshadeParams::default()).expect("hillshade");
    let curv_out = curvature(&dem, CurvatureParams::default()).expect("curvature");

    for (name, raster) in [
        ("slope", &slope_out),
        ("aspect", &aspect_out),
        ("hillshade", &hs_out),
        ("curvature", &curv_out),
    ] {
        for &v in raster.data().iter() {
            assert!(
                !v.is_infinite(),
                "{name} output contains Inf"
            );
        }
    }
}
