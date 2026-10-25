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
    // The same reference as lon/lat (centre of cell (5,3)): needs the
    // `projections` feature of the CLI (off in the no-default-features job).
    if cfg!(feature = "projections") {
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
    }
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

// ─── issues_surtgis.md (riesgo_tal_tal, 2026-09-28) ─────────────────────

/// #1: a Float32 file whose GDAL_NODATA is the f64 decimal terra writes
/// must have its nodata recognised (stats exclude it, `info` names it).
#[test]
fn float32_terra_nodata_is_recognised() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("terra.tif");
    let nd = -3.39999999999999996e38_f64;
    let mut r: Raster<f32> = Raster::new(4, 4);
    r.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
    r.data_mut().fill(10.0);
    r.data_mut()[[0, 0]] = nd as f32;
    r.set_nodata(Some(nd as f32));
    surtgis_core::io::write_geotiff(&r, &path, None).unwrap();
    surtgis_cmd()
        .arg("info")
        .arg(&path)
        .assert()
        .success()
        .stdout(predicate::str::contains("file tag; NaN in memory"))
        .stdout(predicate::str::contains("Min: 10.0000"));
}

/// #3: rasterize streams by strips and never reads the reference pixels;
/// #4: an ESRI .prj naming the UTM zone equals the raster's EPSG code.
#[test]
fn rasterize_streams_and_accepts_esri_prj() {
    let dir = tempfile::tempdir().unwrap();
    let reference = dir.path().join("ref.tif");
    let mut r: Raster<f64> = Raster::new(1100, 40);
    r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
    r.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
    surtgis_core::io::write_geotiff(&r, &reference, None).unwrap();
    // A polygon covering rows 100..600 (strip boundary at 512 inside it),
    // columns 10..30, with a CRS as ESRI WKT (no AUTHORITY).
    let geojson = dir.path().join("poly.geojson");
    std::fs::write(
        &geojson,
        r#"{"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"PROJCS[\"WGS_1984_UTM_Zone_19S\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Transverse_Mercator\"],UNIT[\"Meter\",1.0]]"}},"features":[{"type":"Feature","properties":{"cls":7},"geometry":{"type":"Polygon","coordinates":[[[500100,6299000],[500300,6299000],[500300,6294000],[500100,6294000],[500100,6299000]]]}}]}"#,
    )
    .unwrap();
    let out = dir.path().join("mask.tif");
    surtgis_cmd()
        .arg("rasterize")
        .arg(&geojson)
        .arg(&out)
        .arg("--reference")
        .arg(&reference)
        .args(["--attribute", "cls"])
        .assert()
        .success()
        .stdout(predicate::str::contains("3 strips"));
    let m: Raster<f32> = surtgis_core::io::read_geotiff(&out, None).unwrap();
    assert_eq!(m.shape(), (1100, 40));
    assert_eq!(m.data()[[300, 20]], 7.0);
    assert_eq!(m.data()[[511, 20]], 7.0);
    assert_eq!(m.data()[[512, 20]], 7.0);
    assert!(m.data()[[50, 20]].is_nan());
    assert!(m.data()[[700, 20]].is_nan());
    assert!(m.data()[[300, 5]].is_nan());
    let back = surtgis_core::io::read_geotiff::<f32, _>(&out, None).unwrap();
    assert_eq!(back.crs().and_then(|c| c.epsg()), Some(32719));
}

/// #5: `reclassify` takes negative bounds and can fill nodata;
/// `imagery calc` has comparisons, if() and isnan(); `--output-dtype f32`.
#[test]
fn reclassify_calc_and_output_dtype_ergonomics() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("v.tif");
    let mut r: Raster<f64> = Raster::new(3, 3);
    r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
    for (i, v) in r.data_mut().iter_mut().enumerate() {
        *v = i as f64 * 0.25 - 1.0; // -1 .. 1
    }
    r.data_mut()[[2, 2]] = f64::NAN;
    r.set_nodata(Some(f64::NAN));
    surtgis_core::io::write_geotiff(&r, &src, None).unwrap();

    let out = dir.path().join("cls.tif");
    surtgis_cmd()
        .args(["imagery", "reclassify"])
        .arg(&src)
        .arg(&out)
        .args(["--class", "-0.5,0.5,1", "--default", "-9", "--fill-nodata"])
        .assert()
        .success();
    let c: Raster<f64> = surtgis_core::io::read_geotiff(&out, None).unwrap();
    assert_eq!(c.data()[[1, 1]], 1.0); // 0.0 in [-0.5, 0.5)
    assert_eq!(c.data()[[0, 0]], -9.0); // -1.0 outside
    assert_eq!(c.data()[[2, 2]], -9.0); // nodata filled

    let calc = dir.path().join("calc.tif");
    surtgis_cmd()
        .args([
            "imagery",
            "calc",
            "-e",
            "if(isnan(A), 0, if(A > 0, 1, 0))",
            "-b",
        ])
        .arg(format!("A={}", src.display()))
        .arg(&calc)
        .assert()
        .success();
    let k: Raster<f64> = surtgis_core::io::read_geotiff(&calc, None).unwrap();
    assert_eq!(k.data()[[2, 2]], 0.0);
    assert_eq!(k.data()[[2, 1]], 1.0);
    assert_eq!(k.data()[[0, 0]], 0.0);

    let slope32 = dir.path().join("slope32.tif");
    surtgis_cmd()
        .args(["--output-dtype", "f32", "terrain", "slope"])
        .arg(&src)
        .arg(&slope32)
        .assert()
        .success();
    let any = surtgis_core::io::read_geotiff_any(&slope32, None).unwrap();
    assert_eq!(any.dtype(), surtgis_core::DataType::F32);
}

/// #7: `resample` streams — same values as the in-memory resampler,
/// nearest and bilinear, across strip boundaries, with nodata.
#[test]
fn resample_streams_and_matches_in_memory() {
    let dir = tempfile::tempdir().unwrap();
    let src_path = dir.path().join("src.tif");
    let ref_path = dir.path().join("ref.tif");
    // 10 m source, 400×300, smooth field with a nodata block.
    let mut src: Raster<f64> = Raster::new(400, 300);
    src.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
    src.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
    for r in 0..400 {
        for c in 0..300 {
            src.data_mut()[[r, c]] = (r as f64 * 0.7).sin() * 50.0 + c as f64 * 0.3;
        }
    }
    for r in 100..140 {
        for c in 50..90 {
            src.data_mut()[[r, c]] = f64::NAN;
        }
    }
    src.set_nodata(Some(f64::NAN));
    surtgis_core::io::write_geotiff(&src, &src_path, None).unwrap();
    // 37 m reference grid, offset so cells straddle source pixels.
    let mut reference: Raster<f64> = Raster::new(105, 78);
    reference.set_transform(GeoTransform::new(500_015.0, 6_299_990.0, 37.0, -37.0));
    reference.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
    surtgis_core::io::write_geotiff(&reference, &ref_path, None).unwrap();

    for method in ["nearest", "bilinear"] {
        let out = dir.path().join(format!("out_{method}.tif"));
        surtgis_cmd()
            .arg("resample")
            .arg(&src_path)
            .arg(&out)
            .arg("--reference")
            .arg(&ref_path)
            .args(["--method", method])
            .assert()
            .success()
            .stdout(predicate::str::contains("strips of"));
        let got: Raster<f32> = surtgis_core::io::read_geotiff(&out, None).unwrap();
        assert_eq!(got.shape(), (105, 78));
        let m = if method == "nearest" {
            surtgis_core::ResampleMethod::NearestNeighbor
        } else {
            surtgis_core::ResampleMethod::Bilinear
        };
        let want = surtgis_core::resample_to_grid(&src, &reference, m).unwrap();
        let mut valid = 0;
        for r in 0..105 {
            for c in 0..78 {
                let (g, w) = (got.data()[[r, c]] as f64, want.data()[[r, c]]);
                match (g.is_nan(), w.is_nan()) {
                    (true, true) => {}
                    (false, false) => {
                        valid += 1;
                        assert!(
                            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                            "{method} ({r},{c}): {g} vs {w}"
                        );
                    }
                    _ => panic!("{method} ({r},{c}): nodata mismatch {g} vs {w}"),
                }
            }
        }
        assert!(valid > 7000, "{method}: {valid} valid cells");
        assert_eq!(got.transform().origin_x, 500_015.0);
        assert_eq!(got.crs().and_then(|c| c.epsg()), Some(32719));
    }
}

/// #8: `imagery calc` and `reclassify` stream by strips; the result equals
/// the in-memory evaluation (5 bands, strips forced to 7 rows).
#[test]
fn calc_and_reclassify_stream_by_strips() {
    use surtgis_algorithms::imagery::index_builder;
    let dir = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    let mut rasters = Vec::new();
    for k in 0..5 {
        let mut r: Raster<f64> = Raster::new(50, 30);
        r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
        r.set_crs(Some(surtgis_core::CRS::from_epsg(32719)));
        for i in 0..50 {
            for j in 0..30 {
                r.data_mut()[[i, j]] = ((i * 7 + j * 3 + k * 11) % 13) as f64 * 0.1;
            }
        }
        if k == 2 {
            r.data_mut()[[20, 5]] = f64::NAN;
            r.set_nodata(Some(f64::NAN));
        }
        let p = dir.path().join(format!("b{k}.tif"));
        surtgis_core::io::write_geotiff(&r, &p, None).unwrap();
        paths.push(p);
        rasters.push(r);
    }
    let out = dir.path().join("mean.tif");
    let mut cmd = surtgis_cmd();
    cmd.env("SURTGIS_STRIP_ROWS", "7")
        .args(["imagery", "calc", "-e", "(A+B+C+D+E)/5"]);
    for (k, p) in paths.iter().enumerate() {
        let name = ["A", "B", "C", "D", "E"][k];
        cmd.arg("-b").arg(format!("{name}={}", p.display()));
    }
    cmd.arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("8 strips, 5 bands"));
    let got: Raster<f32> = surtgis_core::io::read_geotiff(&out, None).unwrap();
    let refs: std::collections::HashMap<&str, &Raster<f64>> = ["A", "B", "C", "D", "E"]
        .iter()
        .copied()
        .zip(rasters.iter())
        .collect();
    let want = index_builder("(A+B+C+D+E)/5", &refs).unwrap();
    for i in 0..50 {
        for j in 0..30 {
            let (g, w) = (got.data()[[i, j]] as f64, want.data()[[i, j]]);
            assert!(
                (g.is_nan() && w.is_nan()) || (g - w).abs() < 1e-6,
                "({i},{j}): {g} vs {w}"
            );
        }
    }
    assert!(got.data()[[20, 5]].is_nan());
    assert_eq!(got.transform().origin_y, 6_300_000.0);
    assert_eq!(got.crs().and_then(|c| c.epsg()), Some(32719));

    // Misaligned input is refused, not silently combined.
    let mut off: Raster<f64> = Raster::new(50, 30);
    off.set_transform(GeoTransform::new(500_005.0, 6_300_000.0, 10.0, -10.0));
    let off_path = dir.path().join("off.tif");
    surtgis_core::io::write_geotiff(&off, &off_path, None).unwrap();
    surtgis_cmd()
        .args(["imagery", "calc", "-e", "A+B", "-b"])
        .arg(format!("A={}", paths[0].display()))
        .arg("-b")
        .arg(format!("B={}", off_path.display()))
        .arg(dir.path().join("x.tif"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("same grid"));

    // Reclassify streams too: strip boundaries invisible in the classes.
    let cls = dir.path().join("cls.tif");
    surtgis_cmd()
        .env("SURTGIS_STRIP_ROWS", "7")
        .args(["imagery", "reclassify"])
        .arg(&paths[0])
        .arg(&cls)
        .args(["--class", "0,0.5,1", "--class", "0.5,2,2", "--default", "0"])
        .assert()
        .success()
        .stdout(predicate::str::contains("8 strips"));
    let c: Raster<f32> = surtgis_core::io::read_geotiff(&cls, None).unwrap();
    for i in 0..50 {
        for j in 0..30 {
            let v = rasters[0].data()[[i, j]];
            let want = if v < 0.5 { 1.0 } else { 2.0 };
            assert_eq!(c.data()[[i, j]], want, "({i},{j}) v={v}");
        }
    }
}

/// #10: paths with ñ/accents (Chañaral, Ñuble) must not break the
/// provenance record, and the written file must verify.
#[test]
fn non_ascii_paths_write_and_verify() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("chañaral");
    std::fs::create_dir_all(&src_dir).unwrap();
    let dem = synth_dem(&src_dir);
    let out = dir.path().join("salida_ñ").join("pendiente_Aysén.tif");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    surtgis_cmd()
        .args(["terrain", "slope"])
        .arg(&dem)
        .arg(&out)
        .assert()
        .success();
    surtgis_cmd()
        .arg("verify")
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("chañaral"));
}

/// #11: `--bbox` with negative (west/south) coordinates parses without
/// the `--bbox=` form.
#[test]
fn clip_bbox_accepts_negative_coordinates_without_equals() {
    let dir = tempfile::tempdir().unwrap();
    let mut r: Raster<f64> = Raster::new(30, 40);
    r.set_transform(GeoTransform::new(-71.0, -25.0, 0.01, -0.01));
    r.set_crs(Some(surtgis_core::CRS::from_epsg(4326)));
    let input = dir.path().join("geo.tif");
    surtgis_core::io::write_geotiff(&r, &input, None).unwrap();
    let out = dir.path().join("clip.tif");
    surtgis_cmd()
        .arg("clip")
        .arg(&input)
        .args(["--bbox", "-70.95,-25.15,-70.90,-25.10"])
        .arg(&out)
        .assert()
        .success();
    let c: Raster<f64> = surtgis_core::io::read_geotiff(&out, None).unwrap();
    assert!(c.rows() > 0 && c.rows() < 30 && c.cols() > 0 && c.cols() < 40);
}

/// #12: weighted flow accumulation; constant weights scale the count.
#[test]
fn flow_accumulation_weights_and_include_self() {
    let dir = tempfile::tempdir().unwrap();
    let dem = synth_dem(dir.path());
    let fdir = dir.path().join("fdir.tif");
    surtgis_cmd()
        .args(["hydrology", "flow-direction"])
        .arg(&dem)
        .arg(&fdir)
        .assert()
        .success();
    let mut w: Raster<f32> = Raster::new(20, 20);
    w.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
    w.data_mut().fill(2.5);
    let wpath = dir.path().join("w.tif");
    surtgis_core::io::write_geotiff(&w, &wpath, None).unwrap();

    let count = dir.path().join("count.tif");
    let weighted = dir.path().join("weighted.tif");
    surtgis_cmd()
        .args(["hydrology", "flow-accumulation"])
        .arg(&fdir)
        .arg(&count)
        .assert()
        .success();
    surtgis_cmd()
        .args(["hydrology", "flow-accumulation"])
        .arg(&fdir)
        .arg(&weighted)
        .arg("--weights")
        .arg(&wpath)
        .arg("--include-self")
        .assert()
        .success()
        .stdout(predicate::str::contains("Weighted flow accumulation"));
    let n: Raster<f64> = surtgis_core::io::read_geotiff(&count, None).unwrap();
    let a: Raster<f64> = surtgis_core::io::read_geotiff(&weighted, None).unwrap();
    assert!(n.data().iter().any(|&v| v > 5.0), "flow must converge");
    for (cnt, acc) in n.data().iter().zip(a.data()) {
        assert!(
            (acc - 2.5 * (cnt + 1.0)).abs() < 1e-9,
            "{acc} vs 2.5*({cnt}+1)"
        );
    }

    // A weight raster on another grid is refused.
    let mut off: Raster<f32> = Raster::new(20, 21);
    off.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
    let off_path = dir.path().join("off.tif");
    surtgis_core::io::write_geotiff(&off, &off_path, None).unwrap();
    surtgis_cmd()
        .args(["hydrology", "flow-accumulation"])
        .arg(&fdir)
        .arg(dir.path().join("x.tif"))
        .arg("--weights")
        .arg(&off_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("resample it first"));
}
