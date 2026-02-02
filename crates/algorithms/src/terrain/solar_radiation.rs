//! Solar Radiation Model
//!
//! Computes direct (beam), diffuse, and total solar radiation considering
//! terrain slope, aspect, and topographic shadows.
//! Simplified model inspired by Hofierka & Šúri (2002) / r.sun.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use std::f64::consts::PI;

use super::horizon_angles::HorizonAngles;

/// Diffuse radiation model
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum DiffuseModel {
    /// Isotropic sky model: uniform diffuse from all sky directions.
    /// Simple but underestimates near-sun and near-horizon brightness.
    #[default]
    Isotropic,
    /// Klucher (1979) anisotropic model: corrects for circumsolar
    /// brightening and horizon brightening. 5–15% improvement in
    /// diffuse component accuracy.
    Klucher,
    /// Perez et al. (2002) anisotropic model: most accurate diffuse model.
    /// Uses clearness index and sky brightness to select from 8 sky condition
    /// bins, modeling circumsolar and horizon brightness separately.
    /// Requires `ghi` (Global Horizontal Irradiance) and `dhi` (Diffuse
    /// Horizontal Irradiance) to compute ε and Δ. When these are not available,
    /// they are estimated from `transmittance` and `diffuse_proportion`.
    Perez,
}


/// Parameters for solar radiation model
#[derive(Debug, Clone)]
pub struct SolarParams {
    /// Day of year (1-365)
    pub day: u32,
    /// Latitude in degrees (for single-latitude approximation)
    pub latitude: f64,
    /// Atmospheric transmittance (default 0.7, clear sky).
    /// Used when `linke_turbidity` is None.
    pub transmittance: f64,
    /// Diffuse proportion of global radiation (default 0.3)
    pub diffuse_proportion: f64,
    /// Solar constant in W/m² (default 1367.0)
    pub solar_constant: f64,
    /// Time step in hours for daily integration (default 0.5)
    pub time_step: f64,
    /// Diffuse radiation model (default: Isotropic)
    pub diffuse_model: DiffuseModel,
    /// Linke turbidity factor (default: None → use simple transmittance).
    /// Typical values: 2.0 (very clear), 3.0 (clear), 4.0 (hazy), 5+ (polluted).
    /// When set, uses Kasten (1996) atmospheric model instead of transmittance.
    pub linke_turbidity: Option<f64>,
    /// Ground albedo for reflected radiation (default: 0.2 for vegetation).
    /// Set 0.7 for snow, 0.1 for water, 0.0 to disable reflected component.
    pub albedo: f64,
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
            diffuse_model: DiffuseModel::Isotropic,
            linke_turbidity: None,
            albedo: 0.2,
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
    /// Reflected radiation in Wh/m²/day (terrain-reflected component)
    pub reflected: Raster<f64>,
    /// Total (beam + diffuse + reflected) radiation in Wh/m²/day
    pub total: Raster<f64>,
}

/// Compute beam normal irradiance for a given air mass using either
/// simple transmittance or Linke turbidity (Kasten 1996) model.
fn beam_normal_irradiance(solar_constant: f64, air_mass: f64, transmittance: f64, linke_turbidity: Option<f64>) -> f64 {
    if let Some(tl) = linke_turbidity {
        // Kasten (1996) Rayleigh optical depth
        let m = air_mass;
        let delta_r = 1.0 / (6.6296 + 1.7513 * m - 0.1202 * m.powi(2)
            + 0.0065 * m.powi(3) - 0.00013 * m.powi(4));
        solar_constant * (-0.8662 * tl * delta_r).exp()
    } else {
        solar_constant * transmittance.powf(air_mass)
    }
}

// Perez et al. (1990) model coefficients for 8 sky brightness bins.
// Each row: [f11, f12, f13, f21, f22, f23]
// Bins are selected by clearness index ε.
const PEREZ_COEFFS: [[f64; 6]; 8] = [
    // ε bin 1: 1.000 ≤ ε < 1.065 (overcast)
    [-0.008, 0.588, -0.062, -0.060, 0.072, -0.022],
    // ε bin 2: 1.065 ≤ ε < 1.230
    [ 0.130, 0.683, -0.151, -0.019, 0.066, -0.029],
    // ε bin 3: 1.230 ≤ ε < 1.500
    [ 0.330, 0.487, -0.221,  0.055,-0.064, -0.026],
    // ε bin 4: 1.500 ≤ ε < 1.950
    [ 0.568, 0.187, -0.295,  0.109,-0.152, -0.014],
    // ε bin 5: 1.950 ≤ ε < 2.800
    [ 0.873,-0.392, -0.362,  0.226,-0.462,  0.001],
    // ε bin 6: 2.800 ≤ ε < 4.500
    [ 1.133,-1.237, -0.412,  0.288,-0.823,  0.056],
    // ε bin 7: 4.500 ≤ ε < 6.200
    [ 1.060,-1.600, -0.359,  0.264,-1.127,  0.131],
    // ε bin 8: ε ≥ 6.200 (clear)
    [ 0.678,-0.327, -0.250,  0.156,-1.377,  0.251],
];

/// ε bin boundaries for Perez model
const PEREZ_EPS_BINS: [f64; 8] = [1.0, 1.065, 1.23, 1.5, 1.95, 2.8, 4.5, 6.2];

/// Select Perez sky brightness bin from clearness index ε
fn perez_bin(epsilon: f64) -> usize {
    for i in (0..8).rev() {
        if epsilon >= PEREZ_EPS_BINS[i] {
            return i;
        }
    }
    0
}

/// Compute Perez anisotropic diffuse radiation on a tilted surface.
///
/// Returns diffuse irradiance on tilted surface.
#[allow(clippy::too_many_arguments)]
fn perez_diffuse(
    dhi: f64, _ghi: f64, dni: f64,
    theta_z: f64, cos_inc: f64,
    slope: f64, air_mass: f64,
    solar_constant: f64,
) -> f64 {
    if dhi < 1e-6 {
        return 0.0;
    }

    // Clearness index ε
    let epsilon = ((dhi + dni) / dhi + 5.535e-6 * theta_z.powi(3))
        / (1.0 + 5.535e-6 * theta_z.powi(3));

    // Sky brightness Δ
    let i0_ext = solar_constant; // extraterrestrial on horizontal
    let delta = dhi * air_mass / i0_ext;

    let bin = perez_bin(epsilon);
    let c = &PEREZ_COEFFS[bin];

    // Circumsolar brightness coefficient F1
    let f1 = (c[0] + c[1] * delta + c[2] * theta_z).max(0.0);
    // Horizon brightness coefficient F2
    let f2 = c[3] + c[4] * delta + c[5] * theta_z;

    // a = max(0, cos θ_i), b = max(cos 85°, cos θ_z)
    let a = cos_inc.max(0.0);
    let b = theta_z.cos().max(85.0_f64.to_radians().cos());

    // Perez diffuse on tilted surface
    dhi * ((1.0 - f1) * (1.0 + slope.cos()) / 2.0 + f1 * a / b + f2 * slope.sin())
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

    // Precompute daily GHI on flat terrain for reflected radiation
    let mut ghi_flat_daily = 0.0;
    for step in 0..=num_steps {
        let hour = sunrise_hour + step as f64 * dt;
        if hour > sunset_hour { break; }
        let omega = (hour - 12.0) * 15.0_f64.to_radians();
        let sin_alt = lat_rad.sin() * declination.sin()
            + lat_rad.cos() * declination.cos() * omega.cos();
        if sin_alt <= 0.0 { continue; }
        let alt = sin_alt.asin();
        let air_mass = 1.0 / (sin_alt + 0.50572 * (alt.to_degrees() + 6.07995).powf(-1.6364));
        let beam_n = beam_normal_irradiance(params.solar_constant, air_mass, params.transmittance, params.linke_turbidity);
        let beam_horiz = beam_n * sin_alt;
        let dhi = params.solar_constant * sin_alt * params.diffuse_proportion;
        ghi_flat_daily += (beam_horiz + dhi) * dt;
    }

    // Process each cell
    let result_data: Vec<(f64, f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN, f64::NAN); cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
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
                        let air_mass = 1.0 / (sin_alt + 0.50572 * (alt.to_degrees() + 6.07995).powf(-1.6364));
                        let beam_normal = beam_normal_irradiance(
                            params.solar_constant, air_mass, params.transmittance, params.linke_turbidity,
                        );
                        beam_daily += beam_normal * cos_inc * dt;
                    }

                    // Diffuse radiation
                    let i0 = params.solar_constant * sin_alt;
                    let ghi = i0; // Global horizontal irradiance approximation
                    let dhi = ghi * params.diffuse_proportion;
                    let svf_approx = (1.0 + slp.cos()) / 2.0;

                    let diffuse_inst = match params.diffuse_model {
                        DiffuseModel::Isotropic => {
                            dhi * svf_approx
                        }
                        DiffuseModel::Klucher => {
                            let f_clear = if ghi > 1e-6 {
                                1.0 - (dhi / ghi).powi(2)
                            } else {
                                0.0
                            };
                            let theta_z = (PI / 2.0) - alt;
                            let sin3_tz = theta_z.sin().powi(3);
                            let cos_theta = cos_inc.max(0.0);
                            let cos2_theta = cos_theta * cos_theta;
                            let half_slope_sin3 = (slp / 2.0).sin().powi(3);

                            dhi * svf_approx
                                * (1.0 + f_clear * cos2_theta * sin3_tz)
                                * (1.0 + f_clear * half_slope_sin3)
                        }
                        DiffuseModel::Perez => {
                            let theta_z = (PI / 2.0) - alt;
                            let air_mass = 1.0 / (sin_alt + 0.50572
                                * (alt.to_degrees() + 6.07995).powf(-1.6364));
                            let dni = if sin_alt > 0.01 {
                                (ghi - dhi) / sin_alt
                            } else {
                                0.0
                            };
                            perez_diffuse(dhi, ghi, dni, theta_z, cos_inc,
                                slp, air_mass, params.solar_constant)
                        }
                    };

                    diffuse_daily += diffuse_inst * dt;
                }

                // Reflected radiation: albedo × GHI_flat × ground_view_factor
                let reflected_daily = params.albedo * ghi_flat_daily * (1.0 - slp.cos()) / 2.0;

                *row_data_col = (beam_daily, diffuse_daily, reflected_daily);
            }

            row_data
        })
        .collect();

    // Build output rasters
    let mut beam = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut diffuse = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut reflected = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut total = slope_rad.with_same_meta::<f64>(rows, cols);

    beam.set_nodata(Some(f64::NAN));
    diffuse.set_nodata(Some(f64::NAN));
    reflected.set_nodata(Some(f64::NAN));
    total.set_nodata(Some(f64::NAN));

    let mut beam_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut diff_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut refl_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut total_data = Array2::from_elem((rows, cols), f64::NAN);

    for row in 0..rows {
        for col in 0..cols {
            let (b, d, r) = result_data[row * cols + col];
            if !b.is_nan() {
                beam_data[(row, col)] = b;
                diff_data[(row, col)] = d;
                refl_data[(row, col)] = r;
                total_data[(row, col)] = b + d + r;
            }
        }
    }

    *beam.data_mut() = beam_data;
    *diffuse.data_mut() = diff_data;
    *reflected.data_mut() = refl_data;
    *total.data_mut() = total_data;

    Ok(SolarRadiationResult { beam, diffuse, reflected, total })
}

/// Compute daily solar radiation with topographic shadow casting.
///
/// Like [`solar_radiation`], but uses precomputed [`HorizonAngles`] to
/// determine if each cell is in cast shadow from surrounding terrain.
/// When the sun's altitude is below the horizon angle at the sun's azimuth,
/// the cell receives no direct beam radiation for that timestep.
///
/// This eliminates 10–40% overestimation in mountainous terrain compared
/// to the self-shadowing-only model.
///
/// # Arguments
/// * `slope_rad` — Slope in radians
/// * `aspect_rad` — Aspect in radians (0=N, π/2=E, π=S, 3π/2=W)
/// * `params` — Solar parameters
/// * `horizon` — Precomputed horizon angles (from [`super::horizon_angles::horizon_angles`])
///
/// # Returns
/// SolarRadiationResult with beam, diffuse, and total radiation rasters
pub fn solar_radiation_shadowed(
    slope_rad: &Raster<f64>,
    aspect_rad: &Raster<f64>,
    params: SolarParams,
    horizon: &HorizonAngles,
) -> Result<SolarRadiationResult> {
    let (rows_s, cols_s) = slope_rad.shape();
    let (rows_a, cols_a) = aspect_rad.shape();

    if rows_s != rows_a || cols_s != cols_a {
        return Err(Error::SizeMismatch {
            er: rows_s, ec: cols_s,
            ar: rows_a, ac: cols_a,
        });
    }

    let (h_rows, h_cols) = horizon.shape();
    if rows_s != h_rows || cols_s != h_cols {
        return Err(Error::SizeMismatch {
            er: rows_s, ec: cols_s,
            ar: h_rows, ac: h_cols,
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
        PI
    } else if cos_omega_s > 1.0 {
        0.0
    } else {
        cos_omega_s.acos()
    };

    let sunrise_hour = 12.0 - omega_s.to_degrees() / 15.0;
    let sunset_hour = 12.0 + omega_s.to_degrees() / 15.0;

    let num_steps = ((sunset_hour - sunrise_hour) / params.time_step).ceil() as usize;
    let dt = params.time_step;

    // Precompute daily GHI on flat terrain for reflected radiation
    let mut ghi_flat_daily = 0.0;
    for step in 0..=num_steps {
        let hour = sunrise_hour + step as f64 * dt;
        if hour > sunset_hour { break; }
        let omega = (hour - 12.0) * 15.0_f64.to_radians();
        let sin_alt = lat_rad.sin() * declination.sin()
            + lat_rad.cos() * declination.cos() * omega.cos();
        if sin_alt <= 0.0 { continue; }
        let alt = sin_alt.asin();
        let air_mass = 1.0 / (sin_alt + 0.50572 * (alt.to_degrees() + 6.07995).powf(-1.6364));
        let beam_n = beam_normal_irradiance(params.solar_constant, air_mass, params.transmittance, params.linke_turbidity);
        let beam_horiz = beam_n * sin_alt;
        let dhi = params.solar_constant * sin_alt * params.diffuse_proportion;
        ghi_flat_daily += (beam_horiz + dhi) * dt;
    }

    let result_data: Vec<(f64, f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN, f64::NAN); cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
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

                    if sin_alt <= 0.0 { continue; }

                    let alt = sin_alt.asin();

                    // Solar azimuth (0=N, clockwise)
                    let cos_az = (declination.sin() - lat_rad.sin() * sin_alt)
                        / (lat_rad.cos() * alt.cos());
                    let az = if omega > 0.0 {
                        2.0 * PI - cos_az.clamp(-1.0, 1.0).acos()
                    } else {
                        cos_az.clamp(-1.0, 1.0).acos()
                    };

                    // Shadow check: compare sun altitude with horizon angle
                    let horizon_at_sun = horizon.interpolate(row, col, az);
                    let in_shadow = !horizon_at_sun.is_nan() && alt < horizon_at_sun;

                    // Incidence angle on sloped surface
                    let cos_inc = sin_alt * slp.cos()
                        + alt.cos() * slp.sin() * (az - asp).cos();

                    // Direct beam (only if sun hits surface AND not in cast shadow)
                    if cos_inc > 0.0 && !in_shadow {
                        let air_mass = 1.0 / (sin_alt + 0.50572 * (alt.to_degrees() + 6.07995).powf(-1.6364));
                        let beam_normal = beam_normal_irradiance(
                            params.solar_constant, air_mass, params.transmittance, params.linke_turbidity,
                        );
                        beam_daily += beam_normal * cos_inc * dt;
                    }

                    // Diffuse radiation (still received in shadow, reduced by SVF)
                    let i0 = params.solar_constant * sin_alt;
                    let ghi = i0;
                    let dhi = ghi * params.diffuse_proportion;
                    let svf_approx = (1.0 + slp.cos()) / 2.0;

                    let diffuse_inst = match params.diffuse_model {
                        DiffuseModel::Isotropic => {
                            dhi * svf_approx
                        }
                        DiffuseModel::Klucher => {
                            let f_clear = if ghi > 1e-6 {
                                1.0 - (dhi / ghi).powi(2)
                            } else {
                                0.0
                            };
                            let theta_z = (PI / 2.0) - alt;
                            let sin3_tz = theta_z.sin().powi(3);
                            let cos_theta = cos_inc.max(0.0);
                            let cos2_theta = cos_theta * cos_theta;
                            let half_slope_sin3 = (slp / 2.0).sin().powi(3);

                            dhi * svf_approx
                                * (1.0 + f_clear * cos2_theta * sin3_tz)
                                * (1.0 + f_clear * half_slope_sin3)
                        }
                        DiffuseModel::Perez => {
                            let theta_z = (PI / 2.0) - alt;
                            let air_mass = 1.0 / (sin_alt + 0.50572
                                * (alt.to_degrees() + 6.07995).powf(-1.6364));
                            let dni = if sin_alt > 0.01 {
                                (ghi - dhi) / sin_alt
                            } else {
                                0.0
                            };
                            perez_diffuse(dhi, ghi, dni, theta_z, cos_inc,
                                slp, air_mass, params.solar_constant)
                        }
                    };

                    diffuse_daily += diffuse_inst * dt;
                }

                // Reflected radiation
                let reflected_daily = params.albedo * ghi_flat_daily * (1.0 - slp.cos()) / 2.0;

                *row_data_col = (beam_daily, diffuse_daily, reflected_daily);
            }

            row_data
        })
        .collect();

    // Build output rasters
    let mut beam = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut diffuse = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut reflected = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut total = slope_rad.with_same_meta::<f64>(rows, cols);

    beam.set_nodata(Some(f64::NAN));
    diffuse.set_nodata(Some(f64::NAN));
    reflected.set_nodata(Some(f64::NAN));
    total.set_nodata(Some(f64::NAN));

    let mut beam_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut diff_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut refl_data = Array2::from_elem((rows, cols), f64::NAN);
    let mut total_data = Array2::from_elem((rows, cols), f64::NAN);

    for row in 0..rows {
        for col in 0..cols {
            let (b, d, r) = result_data[row * cols + col];
            if !b.is_nan() {
                beam_data[(row, col)] = b;
                diff_data[(row, col)] = d;
                refl_data[(row, col)] = r;
                total_data[(row, col)] = b + d + r;
            }
        }
    }

    *beam.data_mut() = beam_data;
    *diffuse.data_mut() = diff_data;
    *reflected.data_mut() = refl_data;
    *total.data_mut() = total_data;

    Ok(SolarRadiationResult { beam, diffuse, reflected, total })
}

/// Representative day of year for each month (Klein 1977).
/// These days have declination closest to the monthly average.
const KLEIN_REPRESENTATIVE_DAYS: [u32; 12] = [
    17,  // Jan
    47,  // Feb
    75,  // Mar
    105, // Apr
    135, // May
    162, // Jun
    198, // Jul
    228, // Aug
    258, // Sep
    288, // Oct
    318, // Nov
    344, // Dec
];

/// Days in each month (non-leap year)
const DAYS_PER_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Monthly solar radiation result
#[derive(Debug)]
pub struct MonthlySolarResult {
    /// Monthly radiation per month (Wh/m²/month), indexed 0..12
    pub months: Vec<SolarRadiationResult>,
    /// Annual total (Wh/m²/year) — sum of all 12 months
    pub annual: SolarRadiationResult,
}

/// Compute monthly and annual solar radiation.
///
/// Uses Klein (1977) representative days: for each month, computes
/// daily radiation on the representative day and multiplies by the
/// number of days in that month. Annual is the sum of all 12 months.
///
/// # Arguments
/// * `slope_rad` — Slope in radians
/// * `aspect_rad` — Aspect in radians
/// * `params` — Solar parameters (day field is ignored; each month's representative day is used)
pub fn solar_radiation_annual(
    slope_rad: &Raster<f64>,
    aspect_rad: &Raster<f64>,
    params: SolarParams,
) -> Result<MonthlySolarResult> {
    let (rows, cols) = slope_rad.shape();
    let mut months = Vec::with_capacity(12);

    // Accumulators for annual totals
    let mut ann_beam_data = Array2::from_elem((rows, cols), 0.0_f64);
    let mut ann_diff_data = Array2::from_elem((rows, cols), 0.0_f64);
    let mut ann_refl_data = Array2::from_elem((rows, cols), 0.0_f64);
    let mut ann_total_data = Array2::from_elem((rows, cols), 0.0_f64);
    let mut has_data = Array2::from_elem((rows, cols), false);

    for month in 0..12 {
        let mut month_params = params.clone();
        month_params.day = KLEIN_REPRESENTATIVE_DAYS[month];

        let daily = solar_radiation(slope_rad, aspect_rad, month_params)?;
        let n_days = DAYS_PER_MONTH[month] as f64;

        // Scale daily → monthly and accumulate annual
        let mut m_beam_data = Array2::from_elem((rows, cols), f64::NAN);
        let mut m_diff_data = Array2::from_elem((rows, cols), f64::NAN);
        let mut m_refl_data = Array2::from_elem((rows, cols), f64::NAN);
        let mut m_total_data = Array2::from_elem((rows, cols), f64::NAN);

        for r in 0..rows {
            for c in 0..cols {
                let b = daily.beam.data()[[r, c]];
                if !b.is_nan() {
                    let d = daily.diffuse.data()[[r, c]];
                    let rf = daily.reflected.data()[[r, c]];
                    let t = daily.total.data()[[r, c]];

                    m_beam_data[[r, c]] = b * n_days;
                    m_diff_data[[r, c]] = d * n_days;
                    m_refl_data[[r, c]] = rf * n_days;
                    m_total_data[[r, c]] = t * n_days;

                    ann_beam_data[[r, c]] += b * n_days;
                    ann_diff_data[[r, c]] += d * n_days;
                    ann_refl_data[[r, c]] += rf * n_days;
                    ann_total_data[[r, c]] += t * n_days;
                    has_data[[r, c]] = true;
                }
            }
        }

        let mut m_beam = slope_rad.with_same_meta::<f64>(rows, cols);
        let mut m_diff = slope_rad.with_same_meta::<f64>(rows, cols);
        let mut m_refl = slope_rad.with_same_meta::<f64>(rows, cols);
        let mut m_total = slope_rad.with_same_meta::<f64>(rows, cols);
        m_beam.set_nodata(Some(f64::NAN));
        m_diff.set_nodata(Some(f64::NAN));
        m_refl.set_nodata(Some(f64::NAN));
        m_total.set_nodata(Some(f64::NAN));
        *m_beam.data_mut() = m_beam_data;
        *m_diff.data_mut() = m_diff_data;
        *m_refl.data_mut() = m_refl_data;
        *m_total.data_mut() = m_total_data;

        months.push(SolarRadiationResult {
            beam: m_beam,
            diffuse: m_diff,
            reflected: m_refl,
            total: m_total,
        });
    }

    // Set NaN where no data
    for r in 0..rows {
        for c in 0..cols {
            if !has_data[[r, c]] {
                ann_beam_data[[r, c]] = f64::NAN;
                ann_diff_data[[r, c]] = f64::NAN;
                ann_refl_data[[r, c]] = f64::NAN;
                ann_total_data[[r, c]] = f64::NAN;
            }
        }
    }

    let mut a_beam = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut a_diff = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut a_refl = slope_rad.with_same_meta::<f64>(rows, cols);
    let mut a_total = slope_rad.with_same_meta::<f64>(rows, cols);
    a_beam.set_nodata(Some(f64::NAN));
    a_diff.set_nodata(Some(f64::NAN));
    a_refl.set_nodata(Some(f64::NAN));
    a_total.set_nodata(Some(f64::NAN));
    *a_beam.data_mut() = ann_beam_data;
    *a_diff.data_mut() = ann_diff_data;
    *a_refl.data_mut() = ann_refl_data;
    *a_total.data_mut() = ann_total_data;

    Ok(MonthlySolarResult {
        months,
        annual: SolarRadiationResult {
            beam: a_beam,
            diffuse: a_diff,
            reflected: a_refl,
            total: a_total,
        },
    })
}

// ---------------------------------------------------------------------------
// Corripio (2003) Vectorial Algebra Solar Geometry
// ---------------------------------------------------------------------------
// Replaces scalar trigonometry with rotation matrices on 3D unit vectors.
// Eliminates trigonometric edge-cases at poles and equinoxes.

/// 3D vector type for Corripio vectorial calculations
#[derive(Debug, Clone, Copy)]
pub struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    fn dot(self, other: Self) -> f64 { self.x * other.x + self.y * other.y + self.z * other.z }
    #[allow(dead_code)]
    fn length(self) -> f64 { self.dot(self).sqrt() }
    #[allow(dead_code)]
    fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-15 { return Self::new(0.0, 0.0, 1.0); }
        Self::new(self.x / l, self.y / l, self.z / l)
    }
}


/// Rotate vector around Z axis by angle θ
fn rotate_z(v: Vec3, theta: f64) -> Vec3 {
    let c = theta.cos();
    let s = theta.sin();
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

/// Compute solar position as a unit vector using vectorial algebra (Corripio 2003).
///
/// Returns (sun_vector, sin_altitude) where sun_vector is in the local
/// East-North-Up (ENU) coordinate system.
///
/// # Arguments
/// * `day` — Day of year (1–365)
/// * `hour` — Solar hour (0–24)
/// * `latitude` — Latitude in radians
pub fn solar_vector(day: u32, hour: f64, latitude: f64) -> (Vec3, f64) {
    // Solar declination
    let decl = 23.45_f64.to_radians()
        * ((360.0 * (284.0 + day as f64) / 365.0).to_radians().sin());

    // Hour angle (radians): 0 at noon, positive afternoon
    let hour_angle = (hour - 12.0) * 15.0_f64.to_radians();

    let sin_lat = latitude.sin();
    let cos_lat = latitude.cos();
    let sin_decl = decl.sin();
    let cos_decl = decl.cos();
    let cos_ha = hour_angle.cos();
    let sin_ha = hour_angle.sin();

    // Solar altitude
    let sin_alt = sin_lat * sin_decl + cos_lat * cos_decl * cos_ha;
    let alt = sin_alt.asin();
    let cos_alt = alt.cos();

    // Solar azimuth from North (clockwise)
    // sin(A) = -cos(decl) * sin(ha) / cos(alt)
    // cos(A) = (sin(alt)*sin(lat) - sin(decl)) / (cos(alt)*cos(lat))
    let (sin_az, cos_az) = if cos_alt > 1e-10 && cos_lat > 1e-10 {
        let sa = -cos_decl * sin_ha / cos_alt;
        let ca = (sin_alt * sin_lat - sin_decl) / (cos_alt * cos_lat);
        (sa, ca)
    } else {
        (0.0, 1.0)
    };

    // Build sun unit vector in ENU (x=East, y=North, z=Up)
    let sun = Vec3::new(
        cos_alt * sin_az,  // East component
        cos_alt * cos_az,  // North component
        sin_alt,           // Up component
    );

    (sun, sin_alt)
}

/// Compute the surface normal vector from slope and aspect.
///
/// # Arguments
/// * `slope` — Slope in radians
/// * `aspect` — Aspect in radians (0=N, π/2=E, π=S, 3π/2=W)
///
/// # Returns
/// Unit normal vector in ENU coordinates
pub fn surface_normal(slope: f64, aspect: f64) -> Vec3 {
    // Rotate around Y axis by slope (tilt south)
    let cs = slope.cos();
    let ss = slope.sin();
    let tilted = Vec3::new(ss, 0.0, cs);

    // Rotate around Z axis by aspect (reorient to actual facing direction)
    // Aspect 0 = North, π/2 = East → rotation angle = aspect (from N clockwise)
    rotate_z(tilted, aspect)
}

/// Compute cosine of incidence angle using vectorial algebra (Corripio 2003).
///
/// # Arguments
/// * `sun` — Sun unit vector (ENU)
/// * `normal` — Surface normal unit vector (ENU)
///
/// # Returns
/// cos(θ_i) = dot(sun, normal), clamped to [0, 1]
pub fn cos_incidence_vectorial(sun: Vec3, normal: Vec3) -> f64 {
    sun.dot(normal).clamp(0.0, 1.0)
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
    fn test_solar_klucher_vs_isotropic() {
        // Klucher should produce different (generally higher) diffuse
        // than isotropic on sloped surfaces
        let mut slope_r = Raster::filled(5, 5, 0.5_f64); // ~28.6°
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(5, 5, PI); // South
        aspect_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let iso = solar_radiation(&slope_r, &aspect_r, SolarParams {
            diffuse_model: DiffuseModel::Isotropic,
            ..Default::default()
        }).unwrap();
        let klu = solar_radiation(&slope_r, &aspect_r, SolarParams {
            diffuse_model: DiffuseModel::Klucher,
            ..Default::default()
        }).unwrap();

        let iso_diff = iso.diffuse.get(2, 2).unwrap();
        let klu_diff = klu.diffuse.get(2, 2).unwrap();

        // Klucher adds circumsolar+horizon brightening → should differ
        assert!(
            (iso_diff - klu_diff).abs() > 0.1,
            "Klucher should differ from isotropic: iso={:.1}, klu={:.1}",
            iso_diff, klu_diff
        );
    }

    #[test]
    fn test_reflected_zero_on_flat() {
        // Flat terrain: ground view factor = 0 → no reflected radiation
        let mut slope_r = Raster::filled(5, 5, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(5, 5, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let result = solar_radiation(&slope_r, &aspect_r, SolarParams {
            albedo: 0.2,
            ..Default::default()
        }).unwrap();

        let refl = result.reflected.get(2, 2).unwrap();
        assert!(refl.abs() < 1e-10, "Reflected on flat should be 0, got {}", refl);
    }

    #[test]
    fn test_reflected_positive_on_slope() {
        // Sloped terrain should receive reflected radiation
        let mut slope_r = Raster::filled(5, 5, 0.5_f64); // ~28.6°
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(5, 5, PI);
        aspect_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let result = solar_radiation(&slope_r, &aspect_r, SolarParams {
            albedo: 0.2,
            ..Default::default()
        }).unwrap();

        let refl = result.reflected.get(2, 2).unwrap();
        assert!(refl > 0.0, "Reflected on slope should be positive, got {}", refl);

        // Higher albedo → more reflected
        let result_snow = solar_radiation(&slope_r, &aspect_r, SolarParams {
            albedo: 0.7,
            ..Default::default()
        }).unwrap();
        let refl_snow = result_snow.reflected.get(2, 2).unwrap();
        assert!(
            refl_snow > refl,
            "Snow albedo (0.7) should reflect more than vegetation (0.2): {} vs {}",
            refl_snow, refl
        );
    }

    #[test]
    fn test_linke_turbidity_reduces_beam() {
        // High Linke turbidity should reduce beam radiation
        let mut slope_r = Raster::filled(5, 5, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(5, 5, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let clear = solar_radiation(&slope_r, &aspect_r, SolarParams {
            linke_turbidity: Some(2.0), // very clear
            ..Default::default()
        }).unwrap();
        let hazy = solar_radiation(&slope_r, &aspect_r, SolarParams {
            linke_turbidity: Some(5.0), // polluted/hazy
            ..Default::default()
        }).unwrap();

        let beam_clear = clear.beam.get(2, 2).unwrap();
        let beam_hazy = hazy.beam.get(2, 2).unwrap();
        assert!(
            beam_clear > beam_hazy,
            "Clear sky (TL=2) should have more beam than hazy (TL=5): {} vs {}",
            beam_clear, beam_hazy
        );
    }

    #[test]
    fn test_linke_vs_simple_transmittance() {
        // Linke turbidity and simple transmittance should produce different results
        let mut slope_r = Raster::filled(5, 5, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(5, 5, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let simple = solar_radiation(&slope_r, &aspect_r, SolarParams {
            linke_turbidity: None,
            transmittance: 0.7,
            ..Default::default()
        }).unwrap();
        let linke = solar_radiation(&slope_r, &aspect_r, SolarParams {
            linke_turbidity: Some(3.0),
            ..Default::default()
        }).unwrap();

        let beam_s = simple.beam.get(2, 2).unwrap();
        let beam_l = linke.beam.get(2, 2).unwrap();

        // They should produce different values
        assert!(
            (beam_s - beam_l).abs() > 1.0,
            "Linke and simple should differ: simple={:.1}, linke={:.1}",
            beam_s, beam_l
        );
    }

    #[test]
    fn test_solar_invalid_day() {
        let slope_r = Raster::filled(3, 3, 0.0_f64);
        let aspect_r = Raster::filled(3, 3, 0.0_f64);
        assert!(solar_radiation(&slope_r, &aspect_r, SolarParams {
            day: 0, ..Default::default()
        }).is_err());
    }

    #[test]
    fn test_shadowed_reduces_beam() {
        // Flat terrain with shadow casting from a valley DEM
        // Valley cell should get less beam than ridge cell
        use crate::terrain::horizon_angles::{horizon_angles, HorizonParams};

        let size = 21;
        // V-shaped valley: center column is lowest
        let mut dem = Raster::new(size, size);
        dem.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        for row in 0..size {
            for col in 0..size {
                let dist = (col as f64 - 10.0).abs();
                dem.set(row, col, dist * 20.0).unwrap(); // steep sides
            }
        }

        let mut slope_r = Raster::filled(size, size, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        let mut aspect_r = Raster::filled(size, size, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));

        let horizon = horizon_angles(&dem, HorizonParams {
            radius: 10,
            directions: 36,
        }).unwrap();

        let params = SolarParams {
            latitude: 45.0,
            ..Default::default()
        };

        let no_shadow = solar_radiation(&slope_r, &aspect_r, params.clone()).unwrap();
        let with_shadow = solar_radiation_shadowed(
            &slope_r, &aspect_r, params, &horizon,
        ).unwrap();

        // Valley center (col=10) should have reduced beam with shadows
        let beam_no = no_shadow.beam.get(10, 10).unwrap();
        let beam_sh = with_shadow.beam.get(10, 10).unwrap();

        assert!(
            beam_sh < beam_no,
            "Shadow casting should reduce beam: without={:.1}, with={:.1}",
            beam_no, beam_sh
        );

        // Ridge top (col=0 or col=20) should have similar or same beam
        // (nothing blocks them from above)
    }

    #[test]
    fn test_shadowed_flat_same_as_unshadowed() {
        // On perfectly flat terrain, shadow casting should not change results
        use crate::terrain::horizon_angles::{horizon_angles, HorizonParams};

        let mut dem = Raster::filled(11, 11, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));

        let mut slope_r = Raster::filled(11, 11, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(11, 11, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));

        let horizon = horizon_angles(&dem, HorizonParams {
            radius: 5,
            directions: 36,
        }).unwrap();

        let params = SolarParams::default();

        let no_shadow = solar_radiation(&slope_r, &aspect_r, params.clone()).unwrap();
        let with_shadow = solar_radiation_shadowed(
            &slope_r, &aspect_r, params, &horizon,
        ).unwrap();

        let ns = no_shadow.total.get(5, 5).unwrap();
        let ws = with_shadow.total.get(5, 5).unwrap();

        assert!(
            (ns - ws).abs() < 0.1,
            "Flat terrain: shadowed should match unshadowed: no={:.1}, sh={:.1}",
            ns, ws
        );
    }

    #[test]
    fn test_annual_12_months() {
        let mut slope_r = Raster::filled(3, 3, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(3, 3, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        let result = solar_radiation_annual(&slope_r, &aspect_r, SolarParams {
            latitude: 45.0,
            ..Default::default()
        }).unwrap();

        assert_eq!(result.months.len(), 12);
        // Annual should be positive
        let annual = result.annual.total.get(1, 1).unwrap();
        assert!(annual > 0.0, "Annual should be positive, got {}", annual);
    }

    #[test]
    fn test_annual_summer_month_gt_winter() {
        let mut slope_r = Raster::filled(3, 3, 0.0_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(3, 3, 0.0_f64);
        aspect_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        let result = solar_radiation_annual(&slope_r, &aspect_r, SolarParams {
            latitude: 45.0,
            ..Default::default()
        }).unwrap();

        let june = result.months[5].total.get(1, 1).unwrap(); // June
        let dec = result.months[11].total.get(1, 1).unwrap(); // December

        assert!(
            june > dec,
            "June should have more radiation than December at lat 45: June={:.0}, Dec={:.0}",
            june, dec
        );
    }

    #[test]
    fn test_annual_sum_of_months() {
        let mut slope_r = Raster::filled(3, 3, 0.3_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(3, 3, PI);
        aspect_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        let result = solar_radiation_annual(&slope_r, &aspect_r, SolarParams::default()).unwrap();

        let annual = result.annual.total.get(1, 1).unwrap();
        let month_sum: f64 = result.months.iter()
            .map(|m| m.total.get(1, 1).unwrap())
            .sum();

        assert!(
            (annual - month_sum).abs() < 1.0,
            "Annual should equal sum of months: annual={:.0}, sum={:.0}",
            annual, month_sum
        );
    }

    // ===================================================================
    // P3-SO1: Perez anisotropic diffuse model
    // ===================================================================

    #[test]
    fn test_perez_vs_isotropic() {
        let mut slope_r = Raster::filled(5, 5, 0.5_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(5, 5, PI);
        aspect_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let iso = solar_radiation(&slope_r, &aspect_r, SolarParams {
            diffuse_model: DiffuseModel::Isotropic,
            ..Default::default()
        }).unwrap();
        let perez = solar_radiation(&slope_r, &aspect_r, SolarParams {
            diffuse_model: DiffuseModel::Perez,
            ..Default::default()
        }).unwrap();

        let iso_d = iso.diffuse.get(2, 2).unwrap();
        let perez_d = perez.diffuse.get(2, 2).unwrap();

        assert!(
            (iso_d - perez_d).abs() > 0.01,
            "Perez should differ from isotropic: iso={:.1}, perez={:.1}",
            iso_d, perez_d
        );
    }

    #[test]
    fn test_perez_positive_on_slope() {
        let mut slope_r = Raster::filled(3, 3, 0.3_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut aspect_r = Raster::filled(3, 3, PI);
        aspect_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        let result = solar_radiation(&slope_r, &aspect_r, SolarParams {
            diffuse_model: DiffuseModel::Perez,
            ..Default::default()
        }).unwrap();

        let d = result.diffuse.get(1, 1).unwrap();
        assert!(d > 0.0, "Perez diffuse should be positive, got {:.1}", d);
    }

    #[test]
    fn test_perez_bin_selection() {
        assert_eq!(perez_bin(1.0), 0);
        assert_eq!(perez_bin(1.5), 3);
        assert_eq!(perez_bin(6.5), 7);
        assert_eq!(perez_bin(3.0), 5);
    }

    // ===================================================================
    // P3-SO2: Corripio vectorial algebra
    // ===================================================================

    #[test]
    fn test_solar_vector_noon_equinox() {
        // At equinox (day ~80), noon, latitude 0:
        // Sun should be nearly overhead → z ≈ 1
        let (sv, sin_alt) = solar_vector(80, 12.0, 0.0);
        assert!(
            sin_alt > 0.9,
            "Equinox noon at equator: sin_alt should be ~1, got {:.4}",
            sin_alt
        );
        assert!(sv.z > 0.9, "Sun should be nearly overhead");
    }

    #[test]
    fn test_solar_vector_sunrise() {
        // At lat 45, day 172 (summer solstice), early morning → sin_alt ≈ 0
        let (_, sin_alt) = solar_vector(172, 5.0, 45.0_f64.to_radians());
        assert!(
            sin_alt < 0.3,
            "Early morning sin_alt should be small, got {:.4}",
            sin_alt
        );
    }

    #[test]
    fn test_solar_vector_night() {
        // Midnight → sin_alt < 0
        let (_, sin_alt) = solar_vector(172, 0.0, 45.0_f64.to_radians());
        assert!(
            sin_alt < 0.0,
            "Midnight sin_alt should be negative, got {:.4}",
            sin_alt
        );
    }

    #[test]
    fn test_surface_normal_flat() {
        let n = surface_normal(0.0, 0.0);
        assert!(
            (n.z - 1.0).abs() < 0.01,
            "Flat surface normal should be (0,0,1), got z={:.4}",
            n.z
        );
    }

    #[test]
    fn test_cos_incidence_vectorial_overhead() {
        let sun = Vec3::new(0.0, 0.0, 1.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let ci = cos_incidence_vectorial(sun, normal);
        assert!(
            (ci - 1.0).abs() < 0.001,
            "Overhead sun on flat surface: cos_inc should be 1, got {:.4}",
            ci
        );
    }

    #[test]
    fn test_cos_incidence_vectorial_grazing() {
        let sun = Vec3::new(1.0, 0.0, 0.0); // sun on horizon, east
        let normal = Vec3::new(0.0, 0.0, 1.0); // flat surface
        let ci = cos_incidence_vectorial(sun, normal);
        assert!(
            ci.abs() < 0.001,
            "Grazing sun on flat surface: cos_inc should be ~0, got {:.4}",
            ci
        );
    }
}
