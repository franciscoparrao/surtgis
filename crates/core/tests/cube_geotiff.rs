//! End-to-end: three dated GeoTIFFs on disk → Cube → aligned iteration.

use surtgis_core::cube::Cube;
use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_core::{GeoTransform, Raster};

#[test]
fn cube_from_three_dated_geotiffs() {
    let dir = tempfile::tempdir().unwrap();
    let times = vec![1_704_067_200i64, 1_706_745_600, 1_709_251_200]; // 2024-01/02/03-01
    let (rows, cols) = (32usize, 48usize);

    // Write three monthly slices with a known per-time offset
    let mut paths = Vec::new();
    for (i, t) in times.iter().enumerate() {
        let mut r: Raster<f64> = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 30.0, -30.0));
        r.set_nodata(Some(f64::NAN));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, (i * 1000 + row * cols + col) as f64)
                    .unwrap();
            }
        }
        let path = dir.path().join(format!("ndvi_{}.tif", t));
        write_geotiff(&r, &path, None).unwrap();
        paths.push(path);
    }

    // Read back and stack
    let slices: Vec<Raster<f64>> = paths
        .iter()
        .map(|p| read_geotiff::<f64, _>(p, None).unwrap())
        .collect();
    let cube = Cube::from_slices(times.clone(), vec!["ndvi".into()], slices).unwrap();

    assert_eq!(cube.shape(), (rows, cols));
    assert_eq!(cube.times(), &times[..]);

    // Pixel series in time order: +1000 per slice
    let series: Vec<f64> = cube.pixel_series(10, 20, 0).unwrap().collect();
    assert_eq!(series, vec![500.0, 1500.0, 2500.0]);

    // Chunked iteration covers the grid and stays aligned
    let mut covered = 0usize;
    for chunk in cube.chunks(10) {
        let v0 = chunk.views[0][[0, 0]];
        let v1 = chunk.views[1][[0, 0]];
        let v2 = chunk.views[2][[0, 0]];
        assert_eq!(v1 - v0, 1000.0);
        assert_eq!(v2 - v1, 1000.0);
        covered += chunk.rows;
    }
    assert_eq!(covered, rows);
}
