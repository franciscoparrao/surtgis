//! End-to-end smoke tests: spawn the actual `surtgis` binary via
//! `assert_cmd` and assert on exit code + stdout/stderr, instead of only
//! unit-testing handler functions in-process. Catches wiring bugs (clap
//! arg definitions, exit code plumbing, completions generation) that
//! in-process tests can't see.

use assert_cmd::Command;
use predicates::prelude::*;
use std::path::{Path, PathBuf};
use surtgis_core::{GeoTransform, Raster};

/// Build a small synthetic DEM (a uniform east-west ramp) and write it as
/// a GeoTIFF into `dir`, returning its path. Not committed as a fixture —
/// generated fresh per test so these tests need no external assets and
/// run in every CI job, not just the ones that download fixtures.
fn synth_dem(dir: &Path) -> PathBuf {
    let rows = 20;
    let cols = 20;
    let mut dem: Raster<f64> = Raster::new(rows, cols);
    dem.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));
    for r in 0..rows {
        for c in 0..cols {
            dem.set(r, c, (c as f64) * 2.0).unwrap();
        }
    }
    let path = dir.join("dem.tif");
    surtgis_core::io::write_geotiff(&dem, &path, None).unwrap();
    path
}

fn surtgis_cmd() -> Command {
    Command::cargo_bin("surtgis").unwrap()
}

#[test]
fn help_succeeds_with_exit_0() {
    surtgis_cmd()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("surtgis"));
}

#[test]
fn info_on_valid_raster_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let dem = synth_dem(dir.path());
    surtgis_cmd()
        .arg("info")
        .arg(&dem)
        .assert()
        .success()
        .stdout(predicate::str::contains("Dimensions: 20 x 20"));
}

/// A missing input file must exit with the dedicated not-found code (3),
/// not the generic failure code (1) — see `EXIT_NOT_FOUND` in main.rs.
/// This is the behavioral contract a calling script would rely on to
/// distinguish "bad path" from "computation failed" without parsing
/// stderr text.
#[test]
fn info_on_missing_file_exits_not_found() {
    surtgis_cmd()
        .arg("info")
        .arg("/nonexistent/path/does-not-exist-surtgis-smoke.tif")
        .assert()
        .code(3)
        .stderr(predicate::str::contains("Error"));
}

/// An unrecognized subcommand is a clap usage error: exit code 2,
/// unrelated to (and unaffected by) our custom not-found/failure codes.
#[test]
fn unrecognized_subcommand_exits_2() {
    surtgis_cmd()
        .arg("not-a-real-subcommand")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("Usage"));
}

#[test]
fn terrain_slope_end_to_end_produces_output() {
    let dir = tempfile::tempdir().unwrap();
    let dem = synth_dem(dir.path());
    let output = dir.path().join("slope.tif");

    surtgis_cmd()
        .args(["terrain", "slope"])
        .arg(&dem)
        .arg(&output)
        .assert()
        .success();

    assert!(output.exists(), "slope output file was not created");
    // Sanity: the CLI's own `info` can read back what it just wrote.
    surtgis_cmd()
        .arg("info")
        .arg(&output)
        .assert()
        .success()
        .stdout(predicate::str::contains("Dimensions"));
}

/// This contract only holds for the default (native-reader) build. With
/// the optional `gdal` feature enabled, a missing file surfaces as GDAL's
/// opaque `Error::Gdal(String)` (no `ErrorKind`, just a NULL-pointer
/// message from the C API), which `exit_code_for` can't structurally
/// classify as not-found — so this test is skipped under `--all-features`.
/// CI's main test job runs with default features, where this passes.
#[test]
#[cfg_attr(
    feature = "gdal",
    ignore = "GDAL backend reports missing files as an opaque string error, not io::ErrorKind::NotFound"
)]
fn terrain_slope_on_missing_input_exits_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let output = dir.path().join("slope.tif");
    surtgis_cmd()
        .args(["terrain", "slope"])
        .arg("/nonexistent/dem-smoke-test.tif")
        .arg(&output)
        .assert()
        .code(3);
}

#[test]
fn completions_bash_generates_nonempty_script() {
    surtgis_cmd()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(predicate::str::contains("_surtgis"));
}

#[test]
fn completions_zsh_generates_nonempty_script() {
    surtgis_cmd()
        .args(["completions", "zsh"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_fish_generates_nonempty_script() {
    surtgis_cmd()
        .args(["completions", "fish"])
        .assert()
        .success()
        .stdout(predicate::str::is_empty().not());
}

#[test]
fn completions_rejects_unknown_shell() {
    surtgis_cmd()
        .args(["completions", "not-a-shell"])
        .assert()
        .code(2);
}

// ─── Provenance ────────────────────────────────────────────────────────

/// Every output carries a record of how it was made; `provenance` shows
/// it, `verify` recomputes the output digest and re-hashes the inputs.
#[test]
fn outputs_carry_verifiable_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let dem = synth_dem(dir.path());
    let out = dir.path().join("slope.tif");
    surtgis_cmd()
        .args(["terrain", "slope"])
        .arg(&dem)
        .arg(&out)
        .assert()
        .success();

    surtgis_cmd()
        .arg("provenance")
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Operation: terrain slope"))
        .stdout(predicate::str::contains("dem.tif"))
        .stdout(predicate::str::contains("blake3:"));

    surtgis_cmd()
        .args(["provenance", "--json"])
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "\"schema\": \"surtgis-provenance/1\"",
        ));

    surtgis_cmd()
        .arg("verify")
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("Output  OK"))
        .stdout(predicate::str::contains("Input   OK"));

    // Touching the input breaks the lineage check but not the output one.
    let mut d: Raster<f64> = surtgis_core::io::read_geotiff(&dem, None).unwrap();
    d.set(0, 0, 123.0).unwrap();
    surtgis_core::io::write_geotiff(&d, &dem, None).unwrap();
    surtgis_cmd()
        .arg("verify")
        .arg(&out)
        .assert()
        .failure()
        .stdout(predicate::str::contains("Output  OK"))
        .stdout(predicate::str::contains("Input   MISMATCH"));

    // The digest follows the pixels: rewriting the output with one pixel
    // changed (same record passed in) yields a different recorded digest,
    // so a file whose pixels were altered behind the record cannot verify.
    let before = surtgis_core::io::read_provenance(&out).unwrap().unwrap();
    let mut o: Raster<f64> = surtgis_core::io::read_geotiff(&out, None).unwrap();
    o.set(1, 1, 99.0).unwrap();
    surtgis_core::io::write_geotiff_with_provenance(&o, &out, None, &before).unwrap();
    let after = surtgis_core::io::read_provenance(&out).unwrap().unwrap();
    assert_ne!(
        before.output.unwrap().data_blake3,
        after.output.unwrap().data_blake3
    );
}

/// `--no-provenance` (and the env var) writes plain files.
#[test]
fn no_provenance_flag_writes_plain_files() {
    let dir = tempfile::tempdir().unwrap();
    let dem = synth_dem(dir.path());
    let out = dir.path().join("plain.tif");
    surtgis_cmd()
        .args(["--no-provenance", "terrain", "slope"])
        .arg(&dem)
        .arg(&out)
        .assert()
        .success();
    assert!(surtgis_core::io::read_provenance(&out).unwrap().is_none());
    surtgis_cmd()
        .arg("provenance")
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("no provenance record"));
    surtgis_cmd().arg("verify").arg(&out).assert().failure();

    let out2 = dir.path().join("plain2.tif");
    surtgis_cmd()
        .env("SURTGIS_NO_PROVENANCE", "1")
        .args(["terrain", "slope"])
        .arg(&dem)
        .arg(&out2)
        .assert()
        .success();
    assert!(surtgis_core::io::read_provenance(&out2).unwrap().is_none());
}

// ─── Embeddings ────────────────────────────────────────────────────────

/// A 4-band "embedding" stack: west and east halves point in different
/// directions in vector space.
fn synth_embeddings(dir: &Path) -> PathBuf {
    let (rows, cols) = (20, 20);
    let dirs = [[1.0, 0.2, 0.0, 0.1], [0.0, 0.1, 1.0, 0.3]];
    let mut bands: Vec<Raster<f64>> = (0..4).map(|_| Raster::<f64>::new(rows, cols)).collect();
    for r in 0..rows {
        for c in 0..cols {
            let d = if c < cols / 2 { dirs[0] } else { dirs[1] };
            for (k, b) in bands.iter_mut().enumerate() {
                b.data_mut()[[r, c]] = 100.0 * d[k] + ((r * 7 + c * 3) % 5) as f64;
            }
        }
    }
    for b in &mut bands {
        b.set_transform(GeoTransform::new(350_000.0, 6_300_000.0, 10.0, -10.0));
        b.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
    }
    let path = dir.join("emb.tif");
    let refs: Vec<&Raster<f64>> = bands.iter().collect();
    surtgis_core::io::write_geotiff_stack(
        &refs,
        None,
        &path,
        &surtgis_core::io::GeoTiffOptions::default(),
    )
    .unwrap();
    path
}

#[test]
fn embeddings_similarity_pca_norm_info() {
    let dir = tempfile::tempdir().unwrap();
    let emb = synth_embeddings(dir.path());

    surtgis_cmd()
        .args(["embeddings", "info"])
        .arg(&emb)
        .assert()
        .success()
        .stdout(predicate::str::contains("Bands (dimensions): 4"))
        .stdout(predicate::str::contains("L2 norm"));

    // Similarity to a western cell: west ≈ 1, east clearly lower.
    let sim = dir.path().join("sim.tif");
    surtgis_cmd()
        .args(["embeddings", "similarity"])
        .arg(&emb)
        .arg(&sim)
        .args(["--ref-cell", "5,3"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Reference: cell (5, 3)"));
    let s: Raster<f64> = surtgis_core::io::read_geotiff(&sim, None).unwrap();
    assert!(s.data()[[5, 3]] > 0.999);
    assert!(s.data()[[5, 15]] < 0.8, "east {}", s.data()[[5, 15]]);
    // The same reference as lon/lat (centre of cell (5,3)) and as a vector.
    let gt = GeoTransform::new(350_000.0, 6_300_000.0, 10.0, -10.0);
    let (x, y) = gt.pixel_to_geo(3, 5);
    let (lon, lat) = surtgis_core::warp::Transformer::new(32719, 4326)
        .unwrap()
        .forward(x, y)
        .unwrap();
    let sim2 = dir.path().join("sim2.tif");
    surtgis_cmd()
        .args(["embeddings", "similarity"])
        .arg(&emb)
        .arg(&sim2)
        .args(["--ref-lonlat", &format!("{lon},{lat}")])
        .assert()
        .success();
    let s2: Raster<f64> = surtgis_core::io::read_geotiff(&sim2, None).unwrap();
    assert_eq!(s.data(), s2.data());
    surtgis_cmd()
        .args(["embeddings", "similarity"])
        .arg(&emb)
        .arg(dir.path().join("sim3.tif"))
        .args(["--ref-vec", "1,0.2,0,0.1", "--metric", "euclidean"])
        .assert()
        .success();
    // Exactly one reference flag.
    surtgis_cmd()
        .args(["embeddings", "similarity"])
        .arg(&emb)
        .arg(dir.path().join("x.tif"))
        .assert()
        .failure();

    // PCA: 3-band stack, model saved and reusable.
    let pca = dir.path().join("pca.tif");
    let model = dir.path().join("pca.json");
    surtgis_cmd()
        .args(["embeddings", "pca"])
        .arg(&emb)
        .arg(&pca)
        .args(["--components", "3", "--save-model"])
        .arg(&model)
        .assert()
        .success()
        .stdout(predicate::str::contains("PC1: eigenvalue"));
    let pcs: Vec<Raster<f64>> = surtgis_core::io::read_geotiff_bands(&pca).unwrap();
    assert_eq!(pcs.len(), 3);
    assert!(model.exists());
    surtgis_cmd()
        .args(["embeddings", "pca"])
        .arg(&emb)
        .arg(dir.path().join("pca2.tif"))
        .args(["--model"])
        .arg(&model)
        .assert()
        .success();
    let pcs2: Vec<Raster<f64>> =
        surtgis_core::io::read_geotiff_bands(dir.path().join("pca2.tif")).unwrap();
    assert_eq!(pcs[0].data(), pcs2[0].data());

    let n = dir.path().join("norm.tif");
    surtgis_cmd()
        .args(["embeddings", "norm"])
        .arg(&emb)
        .arg(&n)
        .assert()
        .success();
    let nr: Raster<f64> = surtgis_core::io::read_geotiff(&n, None).unwrap();
    assert!(nr.data()[[0, 0]] > 100.0);

    // Every output carries provenance naming the stack.
    surtgis_cmd().arg("verify").arg(&sim).assert().success();
}

/// A multi-band feature raster (an embedding stack) expands to one
/// feature per band in `extract`, next to ordinary single-band features.
#[test]
fn extract_expands_multiband_features() {
    let dir = tempfile::tempdir().unwrap();
    let feats = dir.path().join("features");
    std::fs::create_dir_all(&feats).unwrap();
    let emb = synth_embeddings(&feats);
    // One single-band feature on the same grid.
    let mut single: Raster<f64> = Raster::new(20, 20);
    single.set_transform(GeoTransform::new(350_000.0, 6_300_000.0, 10.0, -10.0));
    single.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
    single.data_mut().fill(7.0);
    surtgis_core::io::write_geotiff(&single, feats.join("single.tif"), None).unwrap();
    let _ = emb;
    // Two labelled points in the raster CRS.
    let points = dir.path().join("pts.geojson");
    std::fs::write(
        &points,
        r#"{"type":"FeatureCollection","features":[
          {"type":"Feature","properties":{"cls":1},"geometry":{"type":"Point","coordinates":[350035.0,6299945.0]}},
          {"type":"Feature","properties":{"cls":2},"geometry":{"type":"Point","coordinates":[350155.0,6299945.0]}}]}"#,
    )
    .unwrap();
    let out = dir.path().join("table.csv");
    surtgis_cmd()
        .args(["extract", "--features-dir"])
        .arg(&feats)
        .arg("--points")
        .arg(&points)
        .args(["--target", "cls"])
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("4 feature(s)"));
    let csv = std::fs::read_to_string(&out).unwrap();
    let header = csv.lines().next().unwrap();
    for col in ["emb:b1", "emb:b2", "emb:b3", "emb:b4", "single"] {
        assert!(header.contains(col), "header {header}");
    }
    assert_eq!(csv.lines().count(), 3, "{csv}");
}
