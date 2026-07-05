//! Regression test for REG-1.
//!
//! Sprint 3 (PR #68) wired the batch path's per-row latitude correction for
//! geographic (lon/lat) CRSs (`CellSizes::for_dem`/`at_row`) into `slope()`,
//! `aspect()`, `hillshade()`, and `multidirectional_hillshade()`, but never
//! propagated it to the corresponding streaming `WindowAlgorithm`
//! implementations. Since the CLI auto-selects streaming for large files,
//! the same EPSG:4326 DEM produced correct output via the batch functions
//! and silently wrong output via the CLI.
//!
//! The fix threads a `GeoRowContext` (built by `StripProcessor` from the
//! raster's CRS + `GeoTransform`) into a new `WindowAlgorithm::process_chunk_geo`
//! method, which the four affected algorithms override to re-derive their
//! per-row metric cell size exactly like the batch path does.
//!
//! These tests drive the *real* `StripProcessor` (real GeoTIFF read/write,
//! not an emulation) end-to-end on a small synthetic geographic DEM and
//! assert the output matches the batch function bit-for-bit (interior
//! cells only — batch and streaming legitimately differ at the border by
//! kernel radius, which is unrelated to this fix).

use surtgis_algorithms::terrain::{
    AspectOutput, AspectStreaming, HillshadeParams, HillshadeStreaming, MultiHillshadeParams,
    MultiHillshadeStreaming, SlopeParams, SlopeStreaming, aspect, hillshade,
    multidirectional_hillshade, slope,
};
use surtgis_core::crs::CRS;
use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_core::raster::Raster;
use surtgis_core::{GeoTransform, StripProcessor, WindowAlgorithm};

const N: usize = 24;
const PX: f64 = 0.001; // ~100 m per pixel at the equator
const LAT0: f64 = -45.0; // far enough south that the correction is material
const CHUNK_ROWS: usize = 5; // forces several strip seams inside the raster

/// Synthetic geographic DEM (EPSG:4326): irregular-ish tilted surface so
/// both gradient components are non-zero everywhere.
fn geographic_dem() -> Raster<f64> {
    let mut dem = Raster::new(N, N);
    dem.set_transform(GeoTransform::new(-71.0, LAT0, PX, -PX));
    dem.set_crs(Some(CRS::from_epsg(4326)));
    for r in 0..N {
        for c in 0..N {
            let z = (r as f64) * 15.0 + (c as f64) * 20.0 + ((r * 3 + c) % 5) as f64 * 2.0;
            dem.set(r, c, z).unwrap();
        }
    }
    dem
}

/// Round-trip `dem` through a real GeoTIFF file and process it with
/// `StripProcessor`, returning the output raster.
fn run_streaming<A: WindowAlgorithm>(dem: &Raster<f64>, algo: &A) -> Raster<f64> {
    let dir = tempfile::tempdir().unwrap();
    let in_path = dir.path().join("in.tif");
    let out_path = dir.path().join("out.tif");
    write_geotiff(dem, &in_path, None).unwrap();

    StripProcessor::new(CHUNK_ROWS)
        .process(&in_path, &out_path, algo, false)
        .unwrap();

    read_geotiff::<f64, _>(&out_path, None).unwrap()
}

/// Compare batch vs streaming on interior cells (skip the 1-cell border,
/// which both paths NaN-out for unrelated reasons).
///
/// Tolerance is deliberately loose (not `1e-9`): `run_streaming` round-trips
/// the DEM through a real GeoTIFF file, and this crate's native writer
/// always stores samples as `f32` (see `native::encode_geotiff`), so a
/// couple of ULPs of `f32` rounding noise (~1e-7 relative) is expected and
/// is unrelated to the REG-1 correction under test. A real regression
/// (e.g. the latitude correction silently not engaging) produces errors
/// many orders of magnitude larger than this — often changing the slope by
/// tens of degrees — so this tolerance still catches it easily.
fn assert_interior_matches(batch: &Raster<f64>, streaming: &Raster<f64>, label: &str) {
    let (rows, cols) = batch.shape();
    let mut compared = 0;
    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            let b = batch.get(r, c).unwrap();
            let s = streaming.get(r, c).unwrap();
            if b.is_nan() && s.is_nan() {
                continue;
            }
            let tol = 1e-3 + 1e-4 * b.abs();
            assert!(
                (b - s).abs() < tol,
                "{label}: batch/streaming mismatch at ({r},{c}): batch={b} streaming={s}"
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "{label}: no interior cells were compared");
}

#[test]
fn reg1_slope_streaming_matches_batch_on_geographic_crs() {
    let dem = geographic_dem();
    let params = SlopeParams::default();
    let batch = slope(&dem, params.clone()).unwrap();
    let algo = SlopeStreaming {
        units: params.units,
        z_factor: params.z_factor,
    };
    let streaming = run_streaming(&dem, &algo);
    assert_interior_matches(&batch, &streaming, "slope");
}

#[test]
fn reg1_aspect_streaming_matches_batch_on_geographic_crs() {
    let dem = geographic_dem();
    let batch = aspect(&dem, AspectOutput::Degrees).unwrap();
    let algo = AspectStreaming {
        output_format: AspectOutput::Degrees,
    };
    let streaming = run_streaming(&dem, &algo);
    assert_interior_matches(&batch, &streaming, "aspect");
}

#[test]
fn reg1_hillshade_streaming_matches_batch_on_geographic_crs() {
    let dem = geographic_dem();
    let params = HillshadeParams::default();
    let batch = hillshade(&dem, params.clone()).unwrap();
    let algo = HillshadeStreaming {
        azimuth: params.azimuth,
        altitude: params.altitude,
        z_factor: params.z_factor,
    };
    let streaming = run_streaming(&dem, &algo);
    assert_interior_matches(&batch, &streaming, "hillshade");
}

#[test]
fn reg1_multidirectional_hillshade_streaming_matches_batch_on_geographic_crs() {
    let dem = geographic_dem();
    let params = MultiHillshadeParams::default();
    let batch = multidirectional_hillshade(&dem, params.clone()).unwrap();
    let algo = MultiHillshadeStreaming {
        altitude: params.altitude,
        z_factor: params.z_factor,
        normalized: params.normalized,
    };
    let streaming = run_streaming(&dem, &algo);
    assert_interior_matches(&batch, &streaming, "multidirectional_hillshade");
}

/// Sanity check that the correction actually engages: a *projected* (no
/// CRS) DEM with the same transform must NOT match a geographic run at
/// this latitude — dx shrinks with latitude in the geographic case, so
/// the slope magnitude differs. This guards against a vacuous "fix" where
/// `geo_ctx` is threaded through but never actually changes the math.
#[test]
fn reg1_geographic_correction_actually_changes_the_result() {
    let mut projected_dem = geographic_dem();
    projected_dem.set_crs(None); // same transform, but no CRS => no correction

    let geo_dem = geographic_dem();

    let params = SlopeParams::default();
    let projected_batch = slope(&projected_dem, params.clone()).unwrap();
    let geo_batch = slope(&geo_dem, params).unwrap();

    let (rows, cols) = geo_batch.shape();
    let r = rows / 2;
    let c = cols / 2;
    let projected_val = projected_batch.get(r, c).unwrap();
    let geo_val = geo_batch.get(r, c).unwrap();
    assert!(
        (projected_val - geo_val).abs() > 1.0,
        "expected geographic correction to materially change slope: projected={projected_val} geo={geo_val}"
    );
}
