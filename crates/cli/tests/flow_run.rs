//! End-to-end test of `surtgis flow run` (spec surtgis-flow v1.0 §8):
//! spawns the real binary on a synthetic valley DEM and asserts the output
//! contract — h_t####.tif frames, optional velocity frames, arrival raster
//! and manifest.json with the normative field set.
#![cfg(feature = "flow")]

use assert_cmd::Command;
use std::path::{Path, PathBuf};
use surtgis_core::{GeoTransform, Raster};

const ROWS: usize = 40;
const COLS: usize = 60;
const DX: f64 = 10.0;

/// Inclined V-valley DEM + release block, written as f32 GeoTIFFs.
fn synth_inputs(dir: &Path) -> (PathBuf, PathBuf) {
    let down = 15.0f64.to_radians().tan();
    let side = 8.0f64.to_radians().tan();
    let mut dem: Raster<f32> = Raster::new(ROWS, COLS);
    let mut release: Raster<f32> = Raster::new(ROWS, COLS);
    let transform = GeoTransform::new(500_000.0, 6_200_000.0, DX, -DX);
    for r in 0..ROWS {
        for c in 0..COLS {
            let x = (c as f64 + 0.5) * DX;
            let y = (r as f64 + 0.5) * DX;
            let z = (COLS as f64 * DX - x) * down + (y - ROWS as f64 * DX / 2.0).abs() * side;
            dem.set(r, c, z as f32).unwrap();
            if (15..25).contains(&r) && (5..15).contains(&c) {
                release.set(r, c, 3.0).unwrap();
            }
        }
    }
    dem.set_transform(transform);
    release.set_transform(transform);
    let dem_path = dir.join("dem.tif");
    let rel_path = dir.join("release.tif");
    surtgis_core::io::write_geotiff(&dem, &dem_path, None).unwrap();
    surtgis_core::io::write_geotiff(&release, &rel_path, None).unwrap();
    (dem_path, rel_path)
}

#[test]
fn flow_run_end_to_end_produces_frames_and_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let (dem, release) = synth_inputs(tmp.path());
    let outdir = tmp.path().join("out");
    let arrival = tmp.path().join("arrival.tif");

    Command::cargo_bin("surtgis")
        .unwrap()
        .args(["flow", "run"])
        .arg(&dem)
        .arg(&release)
        .arg(&outdir)
        .args(["--mu", "0.15", "--xi", "200"])
        .args(["--duration", "6", "--output-interval", "2"])
        .arg("--dump-velocity")
        .arg("--arrival")
        .arg(&arrival)
        .assert()
        .success();

    // Frames: t = 0, 2, 4, 6 -> 4 h-frames (+ u/v with --dump-velocity).
    for frame in 0..=3 {
        for prefix in ["h", "u", "v"] {
            let f = outdir.join(format!("{prefix}_t{frame:04}.tif"));
            assert!(f.is_file(), "missing frame {}", f.display());
        }
    }
    assert!(
        !outdir.join("h_t0004.tif").exists(),
        "unexpected extra frame"
    );
    assert!(arrival.is_file(), "missing arrival raster");

    // A frame must round-trip as a georeferenced raster on the DEM grid.
    let h3: Raster<f32> = surtgis_core::io::read_geotiff(outdir.join("h_t0003.tif"), None).unwrap();
    assert_eq!(h3.shape(), (ROWS, COLS));
    assert!((h3.transform().origin_x - 500_000.0).abs() < 1e-6);
    // Some material must still be present and must have moved downslope.
    let total: f64 = h3.data().iter().map(|&v| f64::from(v.max(0.0))).sum();
    assert!(total > 0.0, "all mass vanished by t=6 s");

    // Manifest: normative field set (spec §8).
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(outdir.join("manifest.json")).unwrap())
            .unwrap();
    for key in [
        "crs",
        "origin",
        "cellsize",
        "dt_output",
        "n_frames",
        "duration",
        "row0",
        "mu",
        "xi",
        "units",
    ] {
        assert!(manifest.get(key).is_some(), "manifest missing key {key}");
    }
    assert_eq!(manifest["n_frames"], 4);
    assert_eq!(manifest["cellsize"], DX);
    assert_eq!(manifest["dt_output"], 2.0);
    // GEODEO-requested fields (2026-07-19): explicit duration and
    // self-describing row orientation.
    assert_eq!(manifest["duration"], 6.0);
    assert_eq!(manifest["row0"], "north");
    assert_eq!(manifest["origin"][0], 500_000.0);
    assert_eq!(manifest["units"]["h"], "m");
}

#[test]
fn flow_run_with_entrainment_writes_erosion_and_manifest_v2() {
    let tmp = tempfile::tempdir().unwrap();
    let (dem, release) = synth_inputs(tmp.path());
    // 1 m of erodible material everywhere (the run is short: the flow only
    // advances a couple of cells, so the erodible field must include the
    // release area itself for erosion to be observable).
    let emax: Raster<f32> = {
        let mut r = Raster::filled(ROWS, COLS, 1.0f32);
        r.set_transform(GeoTransform::new(500_000.0, 6_200_000.0, DX, -DX));
        r
    };
    let emax_path = tmp.path().join("erodible.tif");
    surtgis_core::io::write_geotiff(&emax, &emax_path, None).unwrap();
    let outdir = tmp.path().join("out_ent");

    Command::cargo_bin("surtgis")
        .unwrap()
        .args(["flow", "run"])
        .arg(&dem)
        .arg(&release)
        .arg(&outdir)
        .args(["--duration", "6", "--output-interval", "2"])
        .arg("--erodible")
        .arg(&emax_path)
        .args(["--entrainment-k", "1e-2"])
        .arg("--dump-erosion")
        .assert()
        .success();

    for frame in 0..=3 {
        assert!(outdir.join(format!("h_t{frame:04}.tif")).is_file());
        assert!(
            outdir.join(format!("e_t{frame:04}.tif")).is_file(),
            "missing erosion frame {frame}"
        );
    }
    // Erosion must actually have happened by the last frame.
    let e3: Raster<f32> = surtgis_core::io::read_geotiff(outdir.join("e_t0003.tif"), None).unwrap();
    let eroded: f64 = e3
        .data()
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| f64::from(v))
        .sum();
    assert!(eroded > 0.0, "no erosion recorded in e_t0003.tif");

    // Manifest v2: version tag + entrainment block (spec v1.1 §5).
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(outdir.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest["manifest_version"], 2);
    assert_eq!(manifest["units"]["e"], "m");
    let ent = &manifest["entrainment"];
    for key in ["k", "rate_max", "v_entr_min", "f_max", "total_eroded_m3"] {
        assert!(ent.get(key).is_some(), "entrainment block missing {key}");
    }
    assert!(ent["total_eroded_m3"].as_f64().unwrap() > 0.0);

    // Without --erodible the manifest stays byte-level v1: no version field.
    let outdir_plain = tmp.path().join("out_plain");
    Command::cargo_bin("surtgis")
        .unwrap()
        .args(["flow", "run"])
        .arg(&dem)
        .arg(&release)
        .arg(&outdir_plain)
        .args(["--duration", "2", "--output-interval", "2"])
        .assert()
        .success();
    let plain: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(outdir_plain.join("manifest.json")).unwrap())
            .unwrap();
    assert!(
        plain.get("manifest_version").is_none(),
        "v1 manifest must stay untagged"
    );
    assert!(plain.get("entrainment").is_none());
}

#[test]
fn flow_run_rejects_mismatched_grids() {
    let tmp = tempfile::tempdir().unwrap();
    let (dem, _) = synth_inputs(tmp.path());
    // Release on a different grid size.
    let mut bad: Raster<f32> = Raster::new(10, 10);
    bad.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));
    let bad_path = tmp.path().join("bad_release.tif");
    surtgis_core::io::write_geotiff(&bad, &bad_path, None).unwrap();

    Command::cargo_bin("surtgis")
        .unwrap()
        .args(["flow", "run"])
        .arg(&dem)
        .arg(&bad_path)
        .arg(tmp.path().join("out2"))
        .args(["--duration", "2", "--output-interval", "1"])
        .assert()
        .failure();
}
