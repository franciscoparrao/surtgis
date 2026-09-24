//! XYZ tile arithmetic for the `WebMercatorQuad` tile matrix set
//! (EPSG:3857, 256×256 tiles), the scheme MapLibre, Leaflet, OpenLayers
//! and QGIS-XYZ consume.

use surtgis_core::raster::GeoTransform;
use surtgis_core::warp::{Bounds, GridSpec};

/// Tile edge in pixels.
pub const TILE_SIZE: usize = 256;
/// EPSG code of the tile matrix set.
pub const TILE_EPSG: u32 = 3857;
/// Half the world width in Web Mercator metres (π · 6 378 137).
pub const ORIGIN: f64 = 20_037_508.342_789_244;

/// Ground resolution of one pixel at zoom `z`, in metres at the equator.
pub fn resolution(z: u8) -> f64 {
    2.0 * ORIGIN / ((1u64 << z) as f64 * TILE_SIZE as f64)
}

/// Web Mercator bounds of tile `(z, x, y)`.
pub fn bounds(z: u8, x: u32, y: u32) -> Bounds {
    let span = resolution(z) * TILE_SIZE as f64;
    let min_x = -ORIGIN + x as f64 * span;
    let max_y = ORIGIN - y as f64 * span;
    Bounds {
        min_x,
        min_y: max_y - span,
        max_x: min_x + span,
        max_y,
    }
}

/// Whether `(x, y)` addresses a tile at zoom `z`.
pub fn is_valid(z: u8, x: u32, y: u32) -> bool {
    z <= 30 && (x as u64) < (1u64 << z) && (y as u64) < (1u64 << z)
}

/// Target grid for tile `(z, x, y)` grown by `gutter` pixels on every side
/// (the focal kernel's support), so the algorithm can be computed on the
/// grown window and cropped back to the tile.
pub fn grid(z: u8, x: u32, y: u32, gutter: usize) -> GridSpec {
    let res = resolution(z);
    let b = bounds(z, x, y);
    let g = gutter as f64 * res;
    GridSpec {
        transform: GeoTransform::new(b.min_x - g, b.max_y + g, res, -res),
        rows: TILE_SIZE + 2 * gutter,
        cols: TILE_SIZE + 2 * gutter,
        epsg: TILE_EPSG,
    }
}

/// Tile column/row containing `(lon, lat)` at zoom `z`.
pub fn tile_for(lon: f64, lat: f64, z: u8) -> (u32, u32) {
    let n = (1u64 << z) as f64;
    let x = ((lon + 180.0) / 360.0 * n).floor();
    let lat_r = lat.to_radians();
    let y =
        ((1.0 - (lat_r.tan() + 1.0 / lat_r.cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor();
    (x.clamp(0.0, n - 1.0) as u32, y.clamp(0.0, n - 1.0) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_zero_is_the_whole_world() {
        let b = bounds(0, 0, 0);
        assert!((b.min_x + ORIGIN).abs() < 1e-6 && (b.max_x - ORIGIN).abs() < 1e-6);
        assert!((b.min_y + ORIGIN).abs() < 1e-6 && (b.max_y - ORIGIN).abs() < 1e-6);
        assert!((resolution(0) - 2.0 * ORIGIN / 256.0).abs() < 1e-9);
    }

    #[test]
    fn zoom_one_quadrants_tile_the_world() {
        let nw = bounds(1, 0, 0);
        let se = bounds(1, 1, 1);
        assert!((nw.max_x).abs() < 1e-6 && (nw.min_y).abs() < 1e-6);
        assert!((se.min_x).abs() < 1e-6 && (se.max_y).abs() < 1e-6);
        assert!(is_valid(1, 1, 1) && !is_valid(1, 2, 0));
    }

    #[test]
    fn grid_with_gutter_grows_symmetrically() {
        let g = grid(10, 300, 600, 3);
        let b = bounds(10, 300, 600);
        let res = resolution(10);
        assert_eq!(g.rows, 262);
        assert_eq!(g.cols, 262);
        assert!((g.transform.origin_x - (b.min_x - 3.0 * res)).abs() < 1e-6);
        assert!((g.transform.origin_y - (b.max_y + 3.0 * res)).abs() < 1e-6);
        let gb = g.bounds();
        assert!((gb.max_x - (b.max_x + 3.0 * res)).abs() < 1e-6);
        assert!((gb.min_y - (b.min_y - 3.0 * res)).abs() < 1e-6);
    }

    /// The tile found for a lon/lat contains that point once projected.
    #[test]
    fn tile_for_contains_the_projected_point() {
        use surtgis_core::warp::Transformer;
        let tf = Transformer::new(4326, TILE_EPSG).unwrap();
        for (lon, lat, z) in [
            (-70.6693, -33.4489, 10u8),
            (151.2, -33.9, 7),
            (10.0, 60.0, 4),
        ] {
            let (x, y) = tile_for(lon, lat, z);
            let (mx, my) = tf.forward(lon, lat).unwrap();
            let b = bounds(z, x, y);
            assert!(b.min_x <= mx && mx <= b.max_x, "x {mx} not in {b:?}");
            assert!(b.min_y <= my && my <= b.max_y, "y {my} not in {b:?}");
        }
        assert_eq!(tile_for(0.0, 0.0, 1), (1, 1));
    }
}
