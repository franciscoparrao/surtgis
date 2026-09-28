//! Planar (INTERLEAVE=BAND), ZSTD, bottom-up COGs — the AlphaEarth
//! embedding layout — read through the COG reader on a local file, and
//! checked against the same window stored north-up, pixel-interleaved
//! and DEFLATE, decoded by the independent `tiff`-crate reader in core.
//!
//! Both fixtures are `gdal_translate` cuts of a real AlphaEarth tile
//! (2024, UTM 19S, 8 of 64 bands, 96×80 px, 32-px tiles); the north-up
//! one went through `gdalwarp`.

#![cfg(feature = "native")]

use surtgis_cloud::blocking::CogReaderBlocking;
use surtgis_cloud::{BBox, CogReaderOptions};
use surtgis_core::Raster;
use surtgis_core::io::read_geotiff_bands;

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn planar_zstd_bottom_up_matches_north_up_pixel_interleaved() {
    check_planar(&fixture("aef_planar_bottomup_zstd.tif"));
}

/// The same window as a BigTIFF (magic 43, 64-bit offsets, 20-byte IFD
/// entries): the layout of the published 3.6 GB AlphaEarth tiles.
#[test]
fn bigtiff_planar_zstd_bottom_up_reads_the_same() {
    check_planar(&fixture("aef_planar_bottomup_zstd_bigtiff.tif"));
}

fn check_planar(planar: &str) {
    let northup = fixture("aef_northup_pixel_deflate.tif");

    // Oracle: the north-up file through core's native reader.
    let truth: Vec<Raster<f64>> = read_geotiff_bands(&northup).unwrap();
    assert_eq!(truth.len(), 8);
    let gt_truth = *truth[0].transform();
    assert!(gt_truth.pixel_height < 0.0);

    // The planar bottom-up file through the COG reader (local backend).
    let mut r = CogReaderBlocking::open(planar, CogReaderOptions::default()).unwrap();
    let m = r.metadata();
    assert_eq!(r.bands(), 8);
    assert_eq!((m.width, m.height), (96, 80));
    assert_eq!(m.nodata, Some(-128.0));
    // Normalised to north-up: same origin and pixel size as the warped file.
    assert!((m.geo_transform.origin_x - gt_truth.origin_x).abs() < 1e-6);
    assert!((m.geo_transform.origin_y - gt_truth.origin_y).abs() < 1e-6);
    assert!((m.geo_transform.pixel_height - gt_truth.pixel_height).abs() < 1e-9);

    let (min_x, min_y, max_x, max_y) = m.geo_transform.bounds(96, 80);
    let got: Vec<Raster<f64>> = r
        .read_bbox_bands(&BBox::new(min_x, min_y, max_x, max_y), None)
        .unwrap();
    assert_eq!(got.len(), 8);
    for (b, (g, t)) in got.iter().zip(&truth).enumerate() {
        assert_eq!(g.shape(), t.shape(), "band {b} shape");
        let mut diffs = 0usize;
        let mut valid = 0usize;
        for (x, y) in g.data().iter().zip(t.data()) {
            let (xn, yn) = (x.is_nan() || *x == -128.0, y.is_nan() || *y == -128.0);
            if xn && yn {
                continue;
            }
            valid += 1;
            if (x - y).abs() > 0.0 {
                diffs += 1;
            }
        }
        assert!(valid > 1000, "band {b}: {valid} valid cells");
        assert_eq!(diffs, 0, "band {b}: {diffs} of {valid} cells differ");
    }

    // A sub-window in the middle: same values as the oracle's crop.
    let sub = BBox::new(
        min_x + 20.0 * 10.0,
        max_y - 50.0 * 10.0,
        min_x + 60.0 * 10.0,
        max_y - 10.0 * 10.0,
    );
    let win: Vec<Raster<f64>> = r.read_bbox_bands(&sub, None).unwrap();
    assert_eq!(win[0].shape(), (40, 40));
    let (col0, row0) = win[0].transform().geo_to_pixel(sub.min_x, sub.max_y);
    assert!(col0.abs() < 1e-6 && row0.abs() < 1e-6);
    for (b, w) in win.iter().enumerate() {
        for rr in 0..40 {
            for cc in 0..40 {
                let a = w.data()[[rr, cc]];
                let t = truth[b].data()[[rr + 10, cc + 20]];
                let both_nodata = (a.is_nan() || a == -128.0) && (t.is_nan() || t == -128.0);
                assert!(both_nodata || a == t, "band {b} ({rr},{cc}): {a} vs {t}");
            }
        }
    }
}
