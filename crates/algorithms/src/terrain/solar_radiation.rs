//! Solar Radiation Model
//!
//! Computes direct (beam), diffuse, and total solar radiation considering
//! terrain slope, aspect, and topographic shadows.
//! Simplified model inspired by Hofierka & Šúri (2002) / r.sun.

use ndarray::Array2;
use rayon::prelude::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use std::f64::consts::PI;

/// Parameters for solar radiation model
#[derive(Debug, Clone)]
pub struct SolarParams {
    /// Day of year (1-365)
    pub day: u32,
    /// Latitude in degrees (for single-latitude approximation)
    pub latitude: f64,
    /// Atmospheric transmittance (default 0.7, clear sky)
    pub transmittance: f64,
    /// Diffuse proportion of global radiation (default 0.3)
    pub diffuse_proportion: f64,
    /// Solar constant in W/m² (default 1367.0)
    pub solar_constant: f64,
    /// Time step in hours for daily integration (default 0.5)
    pub time_step: f64,
}

impl Default for SolarParams {
    fn default() -> Self {
        Self {
            day: 172, // Summer solstice (June 21)
            latitude: 45.0,
            transmittance: 0.7,
            diffuse_proportion: 0.3,
            solar_constant: 1367.0,
            time_step: 0.5,
        }
    }
}

/// Solar radiation output
#[derive(Debug)]
pub struct SolarRadiationResult {
    /// Direct (beam) radiation in Wh/m²/day
    pub beam: Raster<f64>,
    /// Diffuse radiation in Wh/m²/day
    pub diffuse: Raster<f64>,
    /// Total (beam + diffuse) radiation in Wh/m²/day
    pub total: Raster<f64>,
}

/// Compute daily solar radiation on terrain
///
/// Integrates solar radiation over the day considering:
/// - Solar geometry (declination, hour angle)
/// - Terrain slope and aspect
/// - Self-shadowing (slope faces away from sun)
/// - Atmospheric transmittance
///
/// Note: This is a simplified model. It does NOT compute cast shadows
/// from surrounding terrain (use viewshed for that). For full shadow
/// modeling, combine with horizon angle computation.
///
/// # Arguments
/// * `slope_rad` - Slope in radians
/// * `aspect_rad` - Aspect in radians (0=N, π/2=E, π=S, 3π/2=W)
/// * `params` - Solar parameters
///
/// # Returns
/// SolarRadiationResult with beam, diffuse, and total radiation rasters
pub fn solar_radiation(
    slope_rad: &Raster<f64>,
    aspect_rad: &Raster<f64>,
    params: SolarParams,
) -> Result<SolarRadiationResult> {
    let (rows_s, cols_s) = slope_rad.shape();
    let (rows_a, cols_a) = aspect_rad.shape();

    if rows_s != rows_a || cols_s != cols_a {
        return Err(Error::SizeMismatch {
            er: rows_s, ec: cols_s,
            ar: rows_a, ac: cols_a,
        });
    }

    if params.day == 0 || params.day > 365 {
        return Err(Error::Algorithm("Day must be 1-365".into()));
    }

    let rows = rows_s;
    let cols = cols_s;

    // Solar declination (Spencer 1971)
    let gamma = 2.0 * PI * (params.day as f64 - 1.0) / 365.0;
    let declination = 0.006918 - 0.399912 * gamma.cos() + 0.070257 * gamma.sin()
        - 0.006758 * (2.0 * gamma).cos() + 0.000907 * (2.0 * gamma).sin()
        - 0.002697 * (3.0 * gamma).cos() + 0.00148 * (3.0 * gamma).sin();

    let lat_rad = params.latitude.to_radians();

    // Hour angle at sunrise/sunset for flat terrain
    let cos_omega_s = -(lat_rad.tan() * declination.tan());
    let omega_s = if cos_omega_s < -1.0 {
        PI // Polar day
    } else if cos_omega_s > 1.0 {
        0.0 // Polar night
    } else {
        cos_omega_s.acos()
    };

    // Time steps from sunrise to sunset
    let sunrise_hour = 12.0 - omega_s.to_degrees() / 15.0;
    let sunset_hour = 12.0 + omega_s.to_degrees() / 15.0;

    let num_steps = ((sunset_hour - sunrise_hour) / params.time_step).ceil() as usize;
    let dt = params.time_step;

    // Process each cell
    let result_data: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN); cols];

            for col in 0..cols {
                let slp = unsafe { slope_rad.get_unchecked(row, col) };
                let asp = unsafe { aspect_rad.get_unchecked(row, col) };

                if slp.is_nan() || asp.is_nan() {
                    continue;
                }

                let mut beam_daily = 0.0;
                let mut diffuse_daily = 0.0;

                for step in 0..=num_steps {
                    let hour = sunrise_hour + step as f64 * dt;
                    if hour > sunset_hour { break; }

                    let omega = (hour - 12.0) * 15.0_f64.to_radians();

                    // Solar altitude angle
                    let sin_alt = lat_rad.sin() * declination.sin()
                        + lat_rad.cos() * declination.cos() * omega.cos();

                    if sin_alt <= 0.0 { continue; } // Below horizon

                    let alt = sin_alt.asin();

                    // Solar azimuth
                    let cos_az = (declination.sin() - lat_rad.sin() * sin_alt)
                        / (lat_rad.cos() * alt.cos());
                    let az = if omega > 0.0 {
                        2.0 * PI - cos_az.clamp(-1.0, 1.0).acos()
                    } else {
                        cos_az.clamp(-1.0, 1.0).acos()
                    };

                    // Incidence angle on sloped surface
                    let cos_inc = sin_alt * slp.cos()
                        + alt.cos() * slp.sin() * (az - asp).cos();

                    // Direct beam on surface (only if sun hits the surface)
                    if cos_inc > 0.0 {
                        // Atmospheric path length (air mass)
                        let air_mass = 1.0 / (sin_alt + 0.50572 * (alt.to_degrees() + 6.07995).powf(-1.6364));
                        let beam_normal = params.solar_constant * params.transmittance.powf(air_mass);
                        beam_daily += beam_normal * cos_inc * dt;
                    }

                    // Diffuse (isotropic sky model, corrected for slope)
                    let i0 = params.solar_constant * sin_alt;
                    let svf_approx = (1.0 + slp.cos()) / 2.0; // Simplified SVF for slope
                    diffuse_daily += i0 * params.diffuse_proportion * svf_approx * dt;
                }

                row_data[col] = (beam_daily, diffuse_daily);
            }

            row_data
        })
        .collect();

    // Build output rasters
    let mut beam = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut diffuse = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut total = slope_rad.with_same_meta::<f64>(rows, cols);

    beam.set_nodata(Some(f64::NAN));
    diffuse.set_nodata(Some(f64::NAN));
    total.set_nodata(Some(f64::NAN));

    let mut beam_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut diff_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut total_data = Array2::from_elem((rows, cols), f64::NAN);

    for row in 0..rows {
        for col in 0..cols {
            let (b, d) = result_data[row * cols + col];
            if !b.is_nan() {
                beam_data[(row, col)] = b;
                diff_data[(row, col)] = d;
                total_data[(row, col)] = b + d;
            }
        }
    }

    *beam.data_mut() = beam_data;
    *diffuse.data_mut() = diff_data;
    *total.data_mut() = total_data;

    Ok(SolarRadiationResult { beam, diffuse, total })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_solar_flat() {
        let mut slope_r = Raster::filled(5, 5, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(5, 5, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let result = solar_radiation(&slope_r, &aspect_r, SolarParams::default()).unwrap();
        let total = result.total.get(2, 2).unwrap();
        assert!(total > 0.0, "Flat terrain should receive radiation, got {}", total);
    }

    #[test]
    fn test_solar_south_vs_north() {
        // South-facing slope should receive more radiation (northern hemisphere)
        let mut slope_r = Raster::filled(5, 5, 0.5_f64); // ~28.6°
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let mut south_aspect = Raster::filled(5, 5, PI); // South
        south_aspect.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let mut north_aspect = Raster::filled(5, 5, 0.0_f64); // North
        north_aspect.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let south_result = solar_radiation(&slope_r, &south_aspect, SolarParams {
            latitude: 45.0,
            ..Default::default()
        }).unwrap();
        let north_result = solar_radiation(&slope_r, &north_aspect, SolarParams {
            latitude: 45.0,
            ..Default::default()
        }).unwrap();

        let south_total = south_result.total.get(2, 2).unwrap();
        let north_total = north_result.total.get(2, 2).unwrap();

        assert!(
            south_total > north_total,
            "South slope should get more sun than north: {} vs {}",
            south_total, north_total
        );
    }

    #[test]
    fn test_solar_winter_vs_summer() {
        let mut slope_r = Raster::filled(3, 3, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(3, 3, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        let summer = solar_radiation(&slope_r, &aspect_r, SolarParams {
            day: 172, latitude: 45.0, ..Default::default()
        }).unwrap();
        let winter = solar_radiation(&slope_r, &aspect_r, SolarParams {
            day: 355, latitude: 45.0, ..Default::default()
        }).unwrap();

        let s = summer.total.get(1, 1).unwrap();
        let w = winter.total.get(1, 1).unwrap();

        assert!(s > w, "Summer should have more radiation than winter: {} vs {}", s, w);
    }

    #[test]
    fn test_solar_invalid_day() {
        let slope_r = Raster::filled(3, 3, 0.0_f64);
        let aspect_r = Raster::filled(3, 3, 0.0_f64);
        assert!(solar_radiation(&slope_r, &aspect_r, SolarParams {
            day: 0, ..Default::default()
        }).is_err());
    }
}
