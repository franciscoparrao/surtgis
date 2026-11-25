//! COG reader on a `RasterPixelIsPoint` file (GTRasterTypeGeoKey 1025 = 2):
//! the tiepoint marks the centre of the first pixel and must be moved to
//! the corner, as GDAL does by default. Read as the corner, Landsat
//! Collection 2 windows came out half a pixel to the north-west with their
//! origin on a pixel centre (reported from strandline, 2026-11).
//!
//! Fixture: 20×12 Float32 ramp (value = row·100 + col), EPSG:32619, 30 m,
//! 16×16 tiles, written by GDAL 3.11 with `AREA_OR_POINT=Point`: tiepoint
//! (637860, 4500000), `gdalinfo` origin (637845, 4500015).

#![cfg(feature = "native")]

use surtgis_cloud::blocking::CogReaderBlocking;
use surtgis_cloud::{BBox, CogReaderOptions};
use surtgis_core::Raster;

fn fixture() -> String {
    format!(
        "{}/tests/fixtures/pixel_is_point.tif",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
fn metadata_has_the_gdal_corner_origin() {
    let r = CogReaderBlocking::open(&fixture(), CogReaderOptions::default()).unwrap();
    let gt = r.metadata().geo_transform;
    assert_eq!(
        (gt.origin_x, gt.origin_y, gt.pixel_width, gt.pixel_height),
        (637845.0, 4500015.0, 30.0, -30.0)
    );
}

/// A window from column 3, row 2: its origin is on the pixel grid (an odd
/// multiple of 15) and its first cell is pixel (row 2, col 3) = 203.
#[test]
fn read_bbox_window_is_on_the_corner_grid() {
    let mut r = CogReaderBlocking::open(&fixture(), CogReaderOptions::default()).unwrap();
    let (x0, y0) = (637845.0 + 3.0 * 30.0, 4500015.0 - 2.0 * 30.0);
    let bb = BBox::new(
        x0 + 1.0,
        y0 - 4.0 * 30.0 + 1.0,
        x0 + 5.0 * 30.0 - 1.0,
        y0 - 1.0,
    );
    let w: Raster<f64> = r.read_bbox(&bb, None).unwrap();
    let gt = w.transform();
    assert_eq!((gt.origin_x, gt.origin_y), (x0, y0), "{gt:?}");
    assert_eq!(gt.origin_x % 30.0, 15.0);
    assert_eq!(w.get(0, 0).unwrap(), 203.0);
    assert_eq!(w.get(1, 2).unwrap(), 305.0);
}
