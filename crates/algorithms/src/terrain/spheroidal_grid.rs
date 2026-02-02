//! Spheroidal Grid Support for Global DEMs
//!
//! Florinsky (2017b, 2025 §4.3, Ch. 16): Implements geodetic inverse problem
//! and variable cell area for processing DEMs on spheroidal (geographic)
//! coordinate systems. Required for correct processing of SRTM, Copernicus
//! GLO-30, ETOPO, etc.
//!
//! Key formulas:
//! - Cell dimensions vary with latitude: dx = R·cos(φ)·Δλ, dy = R·Δφ
//! - Partial derivatives adjusted for non-uniform spacing
//! - Geodetic inverse problem (Vincenty 1975) for accurate distances
//!
//! Reference:
//! Florinsky, I.V. (2025). Digital Terrain Analysis. §4.3, Ch. 16.
//! Vincenty, T. (1975). Direct and inverse solutions of geodesics.

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};
use ndarray::Array2;
use crate::maybe_rayon::*;


/// WGS84 ellipsoid parameters
const WGS84_A: f64 = 6_378_137.0;         // semi-major axis (m)
const WGS84_F: f64 = 1.0 / 298.257_223_563; // flattening

/// Parameters for spheroidal grid computations
#[derive(Debug, Clone)]
pub struct SpheroidalParams {
    /// Semi-major axis in meters. Default: WGS84 (6378137.0)
    pub semi_major: f64,
    /// Flattening. Default: WGS84 (1/298.257223563)
    pub flattening: f64,
}

impl Default for SpheroidalParams {
    fn default() -> Self {
        Self {
            semi_major: WGS84_A,
            flattening: WGS84_F,
        }
    }
}

/// Grid cell dimensions at a given latitude on the spheroid
#[derive(Debug, Clone, Copy)]
pub struct CellDimensions {
    /// East-West cell size in meters
    pub dx: f64,
    /// North-South cell size in meters
    pub dy: f64,
    /// Cell area in m²
    pub area: f64,
}

/// Compute cell dimensions at given latitude for a geographic grid.
///
/// # Arguments
/// * `latitude_deg` — Latitude in degrees
/// * `d_lon` — Grid spacing in longitude (degrees)
/// * `d_lat` — Grid spacing in latitude (degrees)
/// * `params` — Spheroid parameters
///
/// # Returns
/// [`CellDimensions`] with dx, dy, and area in meters
pub fn cell_dimensions(
    latitude_deg: f64,
    d_lon: f64,
    d_lat: f64,
    params: &SpheroidalParams,
) -> CellDimensions {
    let lat = latitude_deg.to_radians();
    let a = params.semi_major;
    let f = params.flattening;
    let e2 = 2.0 * f - f * f; // first eccentricity squared

    let sin_lat = lat.sin();
    let cos_lat = lat.cos();

    // Radius of curvature in the prime vertical (N)
    let n = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();

    // Radius of curvature in the meridional plane (M)
    let m = a * (1.0 - e2) / (1.0 - e2 * sin_lat * sin_lat).powf(1.5);

    let dx = n * cos_lat * d_lon.to_radians();
    let dy = m * d_lat.to_radians();
    let area = dx.abs() * dy.abs();

    CellDimensions { dx: dx.abs(), dy: dy.abs(), area }
}

/// Compute cell dimension rasters for a geographic DEM.
///
/// Creates dx (East-West) and dy (North-South) rasters where each cell
/// contains its true ground dimension in meters.
///
/// # Arguments
/// * `dem` — Input DEM in geographic coordinates (lon/lat)
/// * `params` — Spheroid parameters
///
/// # Returns
/// (dx_raster, dy_raster) with cell dimensions in meters
pub fn geographic_cell_sizes(
    dem: &Raster<f64>,
    params: SpheroidalParams,
) -> Result<(Raster<f64>, Raster<f64>)> {
    let (rows, cols) = dem.shape();
    let tf = dem.transform();

    // Check if this looks like a geographic grid
    let pixel_w = tf.pixel_width.abs();
    if pixel_w > 1.0 {
        return Err(Error::Algorithm(
            "DEM appears to be in projected coordinates (pixel width > 1°). \
             Use this function only with geographic (lon/lat) DEMs.".into()
        ));
    }

    let d_lon = tf.pixel_width.abs();
    let d_lat = tf.pixel_height.abs();

    let dx_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            // Latitude at this row center
            let lat = tf.origin_y + (row as f64 + 0.5) * tf.pixel_height;
            let dims = cell_dimensions(lat, d_lon, d_lat, &params);
            vec![dims.dx; cols]
        })
        .collect();

    let dy_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let lat = tf.origin_y + (row as f64 + 0.5) * tf.pixel_height;
            let dims = cell_dimensions(lat, d_lon, d_lat, &params);
            vec![dims.dy; cols]
        })
        .collect();

    let mut dx_raster = dem.with_same_meta::<f64>(rows, cols);
    *dx_raster.data_mut() = Array2::from_shape_vec((rows, cols), dx_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    let mut dy_raster = dem.with_same_meta::<f64>(rows, cols);
    *dy_raster.data_mut() = Array2::from_shape_vec((rows, cols), dy_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok((dx_raster, dy_raster))
}

/// Vincenty inverse formula: compute geodesic distance between two points.
///
/// # Arguments
/// * `lat1`, `lon1` — Point 1 in radians
/// * `lat2`, `lon2` — Point 2 in radians
/// * `params` — Spheroid parameters
///
/// # Returns
/// Distance in meters
pub fn vincenty_distance(
    lat1: f64, lon1: f64,
    lat2: f64, lon2: f64,
    params: &SpheroidalParams,
) -> f64 {
    let a = params.semi_major;
    let f = params.flattening;
    let b = a * (1.0 - f);

    let u1 = ((1.0 - f) * lat1.tan()).atan();
    let u2 = ((1.0 - f) * lat2.tan()).atan();
    let l = lon2 - lon1;

    let sin_u1 = u1.sin();
    let cos_u1 = u1.cos();
    let sin_u2 = u2.sin();
    let cos_u2 = u2.cos();

    let mut lambda = l;

    for _ in 0..100 {
        let sin_lambda = lambda.sin();
        let cos_lambda = lambda.cos();

        let sin_sigma = ((cos_u2 * sin_lambda).powi(2)
            + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2)).sqrt();

        if sin_sigma < 1e-15 {
            return 0.0; // Co-incident points
        }

        let cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
        let sigma = sin_sigma.atan2(cos_sigma);

        let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
        let cos2_alpha = 1.0 - sin_alpha * sin_alpha;

        let cos_2sigma_m = if cos2_alpha > 1e-15 {
            cos_sigma - 2.0 * sin_u1 * sin_u2 / cos2_alpha
        } else {
            0.0
        };

        let c = f / 16.0 * cos2_alpha * (4.0 + f * (4.0 - 3.0 * cos2_alpha));
        let lambda_prev = lambda;
        lambda = l + (1.0 - c) * f * sin_alpha
            * (sigma + c * sin_sigma
                * (cos_2sigma_m + c * cos_sigma
                    * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)));

        if (lambda - lambda_prev).abs() < 1e-12 {
            // Converged
            let u2_val = cos2_alpha * (a * a - b * b) / (b * b);
            let big_a = 1.0 + u2_val / 16384.0
                * (4096.0 + u2_val * (-768.0 + u2_val * (320.0 - 175.0 * u2_val)));
            let big_b = u2_val / 1024.0
                * (256.0 + u2_val * (-128.0 + u2_val * (74.0 - 47.0 * u2_val)));
            let delta_sigma = big_b * sin_sigma
                * (cos_2sigma_m + big_b / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m * cos_2sigma_m)
                        - big_b / 6.0 * cos_2sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos_2sigma_m * cos_2sigma_m)));

            return b * big_a * (sigma - delta_sigma);
        }
    }

    // Failed to converge (antipodal points) — use spherical approximation
    a * ((sin_u1 * sin_u2 + cos_u1 * cos_u2 * l.cos())
        .max(-1.0).min(1.0)).acos()
}

/// Compute slope on a geographic (spheroidal) grid using variable cell sizes.
///
/// Unlike standard slope which assumes uniform cell size, this uses the
/// actual ground distances which vary with latitude.
///
/// # Arguments
/// * `dem` — Input DEM in geographic coordinates
/// * `params` — Spheroid parameters
///
/// # Returns
/// Raster<f64> with slope in radians
pub fn slope_geographic(
    dem: &Raster<f64>,
    params: SpheroidalParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let tf = dem.transform();

    let d_lon = tf.pixel_width.abs();
    let d_lat = tf.pixel_height.abs();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            if row == 0 || row >= rows - 1 {
                return row_data;
            }

            let lat = tf.origin_y + (row as f64 + 0.5) * tf.pixel_height;
            let dims = cell_dimensions(lat, d_lon, d_lat, &params);
            let dx = dims.dx;
            let dy = dims.dy;

            for col in 1..cols - 1 {
                let a = unsafe { dem.get_unchecked(row - 1, col - 1) };
                let b = unsafe { dem.get_unchecked(row - 1, col) };
                let c = unsafe { dem.get_unchecked(row - 1, col + 1) };
                let d = unsafe { dem.get_unchecked(row, col - 1) };
                let f = unsafe { dem.get_unchecked(row, col + 1) };
                let g = unsafe { dem.get_unchecked(row + 1, col - 1) };
                let h = unsafe { dem.get_unchecked(row + 1, col) };
                let i = unsafe { dem.get_unchecked(row + 1, col + 1) };

                if [a, b, c, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    continue;
                }

                let dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / (8.0 * dx);
                let dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / (8.0 * dy);

                row_data[col] = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_cell_dimensions_equator() {
        let dims = cell_dimensions(0.0, 1.0 / 3600.0, 1.0 / 3600.0, &SpheroidalParams::default());
        // 1 arcsecond at equator ≈ 30.87m (E-W) and ≈ 30.72m (N-S)
        assert!(dims.dx > 29.0 && dims.dx < 32.0,
            "Equator 1\" dx should be ~30.87m, got {:.2}", dims.dx);
        assert!(dims.dy > 29.0 && dims.dy < 32.0,
            "Equator 1\" dy should be ~30.72m, got {:.2}", dims.dy);
    }

    #[test]
    fn test_cell_dimensions_latitude_60() {
        // At 60°N, dx should be ~half of equator (cos(60°) = 0.5)
        let eq = cell_dimensions(0.0, 1.0 / 3600.0, 1.0 / 3600.0, &SpheroidalParams::default());
        let at60 = cell_dimensions(60.0, 1.0 / 3600.0, 1.0 / 3600.0, &SpheroidalParams::default());

        let ratio = at60.dx / eq.dx;
        assert!(
            (ratio - 0.5).abs() < 0.02,
            "dx ratio at 60° should be ~0.5, got {:.4}", ratio
        );
        // dy should be similar (slightly larger due to oblate spheroid)
        assert!(
            (at60.dy - eq.dy).abs() / eq.dy < 0.02,
            "dy should vary less with latitude"
        );
    }

    #[test]
    fn test_vincenty_known_distance() {
        // London (51.5°N, 0°W) to Paris (48.85°N, 2.35°E)
        let d = vincenty_distance(
            51.5_f64.to_radians(), 0.0,
            48.85_f64.to_radians(), 2.35_f64.to_radians(),
            &SpheroidalParams::default(),
        );
        // Known distance ≈ 340 km
        assert!(
            (d / 1000.0 - 340.0).abs() < 10.0,
            "London-Paris should be ~340km, got {:.1}km", d / 1000.0
        );
    }

    #[test]
    fn test_vincenty_zero_distance() {
        let d = vincenty_distance(
            45.0_f64.to_radians(), 10.0_f64.to_radians(),
            45.0_f64.to_radians(), 10.0_f64.to_radians(),
            &SpheroidalParams::default(),
        );
        assert!(d < 0.001, "Same point should have 0 distance, got {}", d);
    }

    #[test]
    fn test_slope_geographic() {
        // Geographic DEM with 1 arcsecond spacing
        let n = 11;
        let d = 1.0 / 3600.0; // 1 arcsecond
        let mut dem = Raster::new(n, n);
        dem.set_transform(GeoTransform::new(10.0, 45.0 + n as f64 * d, d, -d));

        // South-sloping plane
        for row in 0..n {
            for col in 0..n {
                dem.set(row, col, (n - row) as f64 * 10.0).unwrap();
            }
        }

        let result = slope_geographic(&dem, SpheroidalParams::default()).unwrap();
        let center_slope = result.get(5, 5).unwrap();
        assert!(
            center_slope > 0.01 && center_slope < 1.0,
            "Geographic slope should be reasonable, got {:.4} rad ({:.1}°)",
            center_slope, center_slope.to_degrees()
        );
    }

    #[test]
    fn test_geographic_cell_sizes() {
        let n = 5;
        let d = 1.0 / 3600.0;
        let mut dem = Raster::filled(n, n, 100.0_f64);
        dem.set_transform(GeoTransform::new(10.0, 45.0 + n as f64 * d, d, -d));

        let (dx, dy) = geographic_cell_sizes(&dem, SpheroidalParams::default()).unwrap();
        let dx_val = dx.get(2, 2).unwrap();
        let dy_val = dy.get(2, 2).unwrap();

        // At ~45°N, 1 arcsecond ≈ 21.8m (E-W) and 30.8m (N-S)
        assert!(dx_val > 15.0 && dx_val < 30.0,
            "dx at 45°N should be ~21m, got {:.2}", dx_val);
        assert!(dy_val > 28.0 && dy_val < 33.0,
            "dy at 45°N should be ~30.8m, got {:.2}", dy_val);
    }
}
