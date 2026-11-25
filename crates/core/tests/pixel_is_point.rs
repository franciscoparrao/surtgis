//! GeoTIFFs tagged `GTRasterTypeGeoKey = RasterPixelIsPoint` (key 1025 = 2)
//! put their tiepoint at the centre of the first pixel. Every reader must
//! move it to the corner convention of `GeoTransform`, as GDAL does by
//! default; reading the tiepoint as the corner shifted Landsat Collection 2
//! half a pixel to the north-west (reported from strandline, 2026-11).
//!
//! Fixture: 20×12 Float32 ramp (value = row·100 + col), EPSG:32619, 30 m,
//! striped, written by GDAL 3.11 with `AREA_OR_POINT=Point`. It stores the
//! tiepoint (637860, 4500000) and key 1025 = 2; `gdalinfo` reports the
//! corner origin (637845, 4500015), an odd multiple of 15 as in Landsat.

use std::path::PathBuf;

use surtgis_core::GeoTransform;
use surtgis_core::io::{
    StripReader, read_geotiff_any_from_buffer, read_geotiff_bands_from_buffer,
    read_geotiff_from_buffer, window::geotiff_info, write_geotiff,
};
use surtgis_core::raster::Raster;

const POINT: &[u8] = include_bytes!("fixtures/pixel_is_point.tif");

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pixel_is_point.tif")
}

/// The corner origin GDAL reports for the fixture.
fn assert_gdal_origin(gt: &GeoTransform) {
    assert_eq!(
        (gt.origin_x, gt.origin_y, gt.pixel_width, gt.pixel_height),
        (637845.0, 4500015.0, 30.0, -30.0),
        "{gt:?}"
    );
}

#[test]
fn every_reader_moves_the_tiepoint_to_the_corner() {
    let r: Raster<f64> = read_geotiff_from_buffer(POINT, None).unwrap();
    assert_gdal_origin(r.transform());
    let bands: Vec<Raster<f64>> = read_geotiff_bands_from_buffer(POINT).unwrap();
    assert_gdal_origin(bands[0].transform());
    let any = read_geotiff_any_from_buffer(POINT, None).unwrap();
    assert_gdal_origin(any.transform());
    assert_gdal_origin(&geotiff_info(fixture()).unwrap().transform);
    assert_gdal_origin(StripReader::open(&fixture()).unwrap().transform());
}

#[test]
fn first_pixel_centre_is_the_tiepoint() {
    let r: Raster<f64> = read_geotiff_from_buffer(POINT, None).unwrap();
    assert_eq!(r.transform().pixel_to_geo(0, 0), (637860.0, 4500000.0));
    assert_eq!(r.get(0, 0).unwrap(), 0.0);
    assert_eq!(r.get(2, 3).unwrap(), 203.0);
}

/// Written back, the raster is tagged PixelIsArea with the corner origin:
/// re-reading it gives the same georeferencing, not a second shift.
#[test]
fn write_then_read_keeps_the_georeferencing() {
    let r: Raster<f64> = read_geotiff_from_buffer(POINT, None).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("area.tif");
    write_geotiff(&r, &out, None).unwrap();
    let back: Raster<f64> = surtgis_core::io::read_geotiff(&out, None).unwrap();
    assert_gdal_origin(back.transform());
}
