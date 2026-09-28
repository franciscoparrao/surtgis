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
