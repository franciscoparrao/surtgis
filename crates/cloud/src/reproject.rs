//! Pure-Rust WGS84 ↔ UTM reprojection (Snyder 1987, USGS formulas).
//!
//! Covers EPSG 326xx (UTM North) and 327xx (UTM South), which handles
//! Sentinel-2, Landsat, and most satellite imagery. No external C dependencies
//! (no libproj), so it works on WASM targets.

use crate::tile_index::BBox;

// ── WGS84 ellipsoid constants ────────────────────────────────────────────

const A: f64 = 6_378_137.0; // semi-major axis (m)
const F: f64 = 1.0 / 298.257_223_563; // flattening
const E2: f64 = 2.0 * F - F * F; // eccentricity squared
const E_PRIME2: f64 = E2 / (1.0 - E2); // second eccentricity squared
const K0: f64 = 0.9996; // UTM scale factor
const FALSE_EASTING: f64 = 500_000.0;
const FALSE_NORTHING_SOUTH: f64 = 10_000_000.0;

// ── Public API ───────────────────────────────────────────────────────────

/// Reproject a WGS84 bbox to the target EPSG CRS.
///
/// Returns the original bbox unchanged if the target is WGS84 (4326) or an
/// unsupported CRS.
pub fn reproject_bbox_to_cog(bbox: &BBox, target_epsg: u32) -> BBox {
    if is_wgs84(target_epsg) {
        return *bbox;
    }

    let Some((zone, north)) = parse_utm_epsg(target_epsg) else {
        // Unsupported CRS — return original bbox and let CogReader handle it.
        #[cfg(feature = "native")]
        eprintln!(
            "warning: unsupported target CRS EPSG:{}, skipping bbox reprojection",
            target_epsg
        );
        return *bbox;
    };

    // Transform all four corners and take the envelope.
    // This handles the non-linear distortion of the UTM projection better
    // than transforming only min/max.
    let corners = [
        (bbox.min_x, bbox.min_y),
        (bbox.min_x, bbox.max_y),
        (bbox.max_x, bbox.min_y),
        (bbox.max_x, bbox.max_y),
    ];

    let mut min_e = f64::MAX;
    let mut min_n = f64::MAX;
    let mut max_e = f64::MIN;
    let mut max_n = f64::MIN;

    for &(lon, lat) in &corners {
        let (e, n) = wgs84_to_utm(lon, lat, zone, north);
        min_e = min_e.min(e);
        min_n = min_n.min(n);
        max_e = max_e.max(e);
        max_n = max_n.max(n);
    }

    BBox::new(min_e, min_n, max_e, max_n)
}

/// Check if an EPSG code represents WGS84 geographic.
pub fn is_wgs84(epsg: u32) -> bool {
    epsg == 4326
}

/// Parse an EPSG code into UTM zone info: `Some((zone, is_north))`.
///
/// - EPSG 326xx → zone xx, North hemisphere
/// - EPSG 327xx → zone xx, South hemisphere
pub fn parse_utm_epsg(epsg: u32) -> Option<(u32, bool)> {
    if (32601..=32660).contains(&epsg) {
        Some((epsg - 32600, true))
    } else if (32701..=32760).contains(&epsg) {
        Some((epsg - 32700, false))
    } else {
        None
    }
}

// ── Core projection (Snyder 1987, USGS Prof. Paper 1395, pp. 61-64) ─────

/// Convert WGS84 (longitude, latitude) in degrees to UTM (easting, northing)
/// in metres for the given zone and hemisphere.
fn wgs84_to_utm(lon_deg: f64, lat_deg: f64, zone: u32, north: bool) -> (f64, f64) {
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();

    // Central meridian of the zone
    let lon0 = ((zone as f64 - 1.0) * 6.0 - 180.0 + 3.0).to_radians();

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();
    let tan_lat = lat.tan();

    let n = A / (1.0 - E2 * sin_lat * sin_lat).sqrt();
    let t = tan_lat * tan_lat;
    let c = E_PRIME2 * cos_lat * cos_lat;
    let a_coeff = cos_lat * (lon - lon0);

    // Meridional arc length M (Snyder eq. 3-21)
    let m = meridional_arc(lat);

    let a2 = a_coeff * a_coeff;
    let a4 = a2 * a2;
    let a6 = a4 * a2;

    // Easting (Snyder eq. 8-9)
    let easting = K0 * n
        * (a_coeff
            + (1.0 - t + c) * a2 * a_coeff / 6.0
            + (5.0 - 18.0 * t + t * t + 72.0 * c - 58.0 * E_PRIME2)
                * a4
                * a_coeff
                / 120.0)
        + FALSE_EASTING;

    // Northing (Snyder eq. 8-10)
    let northing = K0
        * (m
            + n
                * tan_lat
                * (a2 / 2.0
                    + (5.0 - t + 9.0 * c + 4.0 * c * c) * a4 / 24.0
                    + (61.0 - 58.0 * t + t * t + 600.0 * c - 330.0 * E_PRIME2) * a6 / 720.0));

    let northing = if north {
        northing
    } else {
        northing + FALSE_NORTHING_SOUTH
    };

    (easting, northing)
}

/// Meridional arc from equator to latitude `lat` (radians).
/// Snyder eq. 3-21.
fn meridional_arc(lat: f64) -> f64 {
    let e2 = E2;
    let e4 = e2 * e2;
    let e6 = e4 * e2;

    A * ((1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0) * lat
        - (3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0) * (2.0 * lat).sin()
        + (15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0) * (4.0 * lat).sin()
        - (35.0 * e6 / 3072.0) * (6.0 * lat).sin())
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: assert two values are within `tol` of each other.
    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        let diff = (a - b).abs();
        assert!(
            diff < tol,
            "{msg}: expected {b}, got {a}, diff {diff} exceeds tolerance {tol}"
        );
    }

    #[test]
    fn parse_utm_north() {
        assert_eq!(parse_utm_epsg(32630), Some((30, true)));
        assert_eq!(parse_utm_epsg(32601), Some((1, true)));
        assert_eq!(parse_utm_epsg(32660), Some((60, true)));
    }

    #[test]
    fn parse_utm_south() {
        assert_eq!(parse_utm_epsg(32721), Some((21, false)));
        assert_eq!(parse_utm_epsg(32701), Some((1, false)));
        assert_eq!(parse_utm_epsg(32760), Some((60, false)));
    }

    #[test]
    fn parse_utm_invalid() {
        assert_eq!(parse_utm_epsg(4326), None);
        assert_eq!(parse_utm_epsg(3857), None);
        assert_eq!(parse_utm_epsg(32600), None); // zone 0 invalid
        assert_eq!(parse_utm_epsg(32661), None); // zone 61 invalid
        assert_eq!(parse_utm_epsg(32700), None);
    }

    #[test]
    fn is_wgs84_test() {
        assert!(is_wgs84(4326));
        assert!(!is_wgs84(32630));
        assert!(!is_wgs84(3857));
    }

    // Reference values from pyproj (PROJ 9.x):
    //   from pyproj import Transformer
    //   t = Transformer.from_crs(4326, 32630, always_xy=True)
    //   t.transform(-3.7037, 40.4168) → (440298.94, 4474257.31)
    #[test]
    fn madrid_wgs84_to_utm30n() {
        let (e, n) = wgs84_to_utm(-3.7037, 40.4168, 30, true);
        assert_close(e, 440_298.94, 1.0, "easting");
        assert_close(n, 4_474_257.31, 1.0, "northing");
    }

    // Buenos Aires: (-58.3816, -34.6037) → UTM 21S (EPSG:32721)
    //   t = Transformer.from_crs(4326, 32721, always_xy=True)
    //   t.transform(-58.3816, -34.6037) → (373317.50, 6170036.17)
    #[test]
    fn buenos_aires_wgs84_to_utm21s() {
        let (e, n) = wgs84_to_utm(-58.3816, -34.6037, 21, false);
        assert_close(e, 373_317.50, 1.0, "easting");
        assert_close(n, 6_170_036.17, 1.0, "northing");
    }

    // Equator at zone 30 central meridian (-3°): easting should be 500000
    #[test]
    fn equator_central_meridian() {
        let (e, n) = wgs84_to_utm(-3.0, 0.0, 30, true);
        assert_close(e, 500_000.0, 0.01, "easting at CM");
        assert_close(n, 0.0, 0.01, "northing at equator");
    }

    #[test]
    fn reproject_bbox_wgs84_noop() {
        let bbox = BBox::new(-3.75, 40.40, -3.70, 40.45);
        let result = reproject_bbox_to_cog(&bbox, 4326);
        assert!((result.min_x - bbox.min_x).abs() < f64::EPSILON);
        assert!((result.min_y - bbox.min_y).abs() < f64::EPSILON);
        assert!((result.max_x - bbox.max_x).abs() < f64::EPSILON);
        assert!((result.max_y - bbox.max_y).abs() < f64::EPSILON);
    }

    #[test]
    fn reproject_bbox_unknown_epsg_noop() {
        let bbox = BBox::new(-3.75, 40.40, -3.70, 40.45);
        let result = reproject_bbox_to_cog(&bbox, 3857);
        // Unknown → return original
        assert!((result.min_x - bbox.min_x).abs() < f64::EPSILON);
    }

    #[test]
    fn reproject_bbox_madrid_utm30n() {
        let bbox = BBox::new(-3.75, 40.40, -3.70, 40.45);
        let result = reproject_bbox_to_cog(&bbox, 32630);

        // Result should be in UTM metres, not degrees
        assert!(result.min_x > 100_000.0, "easting should be in metres");
        assert!(result.min_y > 4_000_000.0, "northing should be in metres");
        assert!(result.max_x > result.min_x);
        assert!(result.max_y > result.min_y);

        // Width should be roughly 4km (0.05° lon at 40°N ≈ 4.3 km)
        let width = result.max_x - result.min_x;
        assert!(width > 3_000.0 && width < 6_000.0, "width ~4km, got {width}");

        // Height should be roughly 5.5km (0.05° lat ≈ 5.5 km)
        let height = result.max_y - result.min_y;
        assert!(
            height > 4_000.0 && height < 7_000.0,
            "height ~5.5km, got {height}"
        );
    }

    #[test]
    fn reproject_bbox_southern_hemisphere() {
        // Bbox around Buenos Aires
        let bbox = BBox::new(-58.40, -34.62, -58.35, -34.58);
        let result = reproject_bbox_to_cog(&bbox, 32721);

        assert!(result.min_x > 100_000.0, "easting in metres");
        // In UTM south, northing is offset by 10_000_000
        assert!(result.min_y > 6_000_000.0, "northing with south offset");
        assert!(result.max_x > result.min_x);
        assert!(result.max_y > result.min_y);
    }
}
