//! End-to-end regression test for issue #123: `surtgis reproject` must
//! reproject *every* band of a multi-band GeoTIFF and preserve the
//! sample type — a u8 RGB orthophoto comes back as u8 RGB, not as a
//! single float grey band.
#![cfg(feature = "projections")]

use assert_cmd::Command;
use std::path::Path;
use surtgis_core::io::{read_geotiff_any, read_geotiff_bands, write_geotiff_multiband};
use surtgis_core::raster::DataType;
use surtgis_core::{GeoTransform, Raster, crs::CRS};

const ROWS: usize = 80;
const COLS: usize = 120;

/// RGB u8 raster in UTM 19S with per-band signatures: R constant 200,
/// G a horizontal gradient, B constant 50.
fn synth_rgb(path: &Path) {
    let transform = GeoTransform::new(340_000.0, 6_290_000.0, 30.0, -30.0);
    let mut bands: Vec<Raster<u8>> = Vec::new();
    for band in 0..3u8 {
        let mut r: Raster<u8> = Raster::new(ROWS, COLS);
        for row in 0..ROWS {
            for col in 0..COLS {
                let v = match band {
                    0 => 200,
                    1 => (col * 255 / (COLS - 1)) as u8,
                    _ => 50,
                };
                r.set(row, col, v).unwrap();
            }
        }
        r.set_transform(transform);
        r.set_crs(Some(CRS::from_epsg(32719)));
        bands.push(r);
    }
    let refs: Vec<&Raster<u8>> = bands.iter().collect();
    write_geotiff_multiband(&refs, path, None).unwrap();
}

#[test]
fn rgb_u8_reprojects_all_bands_and_stays_u8() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("rgb.tif");
    let output = dir.path().join("rgb_4326.tif");
    synth_rgb(&input);

    Command::new(env!("CARGO_BIN_EXE_surtgis"))
        .args([
            "reproject",
            input.to_str().unwrap(),
            output.to_str().unwrap(),
            "--to",
            "EPSG:4326",
        ])
        .assert()
        .success();

    // Sample type preserved: Byte, not Float32/64.
    let probe = read_geotiff_any(&output, Some(0)).unwrap();
    assert_eq!(probe.dtype(), DataType::U8, "u8 in must be u8 out");

    // All three bands present, each with its own signature intact.
    let bands: Vec<Raster<f64>> = read_geotiff_bands(&output).unwrap();
    assert_eq!(bands.len(), 3, "RGB input must produce 3 output bands");

    let (rows, cols) = bands[0].shape();
    assert!(rows > 10 && cols > 10);
    // Interior pixel well inside the footprint: R=200, B=50 exactly;
    // G is the gradient, so just check it is strictly between R and B
    // ordering-wise on both halves of the image.
    let (rc, cc) = (rows / 2, cols / 2);
    assert_eq!(bands[0].get(rc, cc).unwrap(), 200.0, "red band lost");
    assert_eq!(bands[2].get(rc, cc).unwrap(), 50.0, "blue band lost");
    let g_left = bands[1].get(rc, cols / 4).unwrap();
    let g_right = bands[1].get(rc, 3 * cols / 4).unwrap();
    assert!(
        g_right > g_left + 50.0,
        "green gradient not preserved: left {g_left}, right {g_right}"
    );

    // Georeference converted to the target CRS.
    let epsg = bands[0].crs().and_then(|c| c.epsg());
    assert_eq!(epsg, Some(4326));
}
