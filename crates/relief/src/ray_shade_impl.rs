//! Ray-traced cast shadows via [`crate::cast_shadow_ray_mask`].
//!
//! Single-sun: one early-exit ray march per cell, marching toward the sun
//! along the cell's terrain and recording whether the ray clears every
//! occluder it encounters before reaching the search radius.
//!
//! Multi-sun (rayshader-style soft shadow penumbra): average the per-sun
//! binary lit/shadow masks. Each sample triggers one
//! `cast_shadow_ray_mask` call; with early exit, per-call cost is
//! ~3–4× lower than going through `horizon_angle_map` (which has to find
//! the *maximum* occlusion angle, not just *any* occluder).

use crate::shadow_ray::{cast_shadow_ray_mask, horizon_tan_map};
use crate::{ReliefError, Result};
use ndarray::Array2;
use surtgis_core::raster::Raster;

/// Azimuths within this many radians are treated as "the same" for the
/// amortised path. Tighter than 1° to keep the heuristic conservative.
const SHARED_AZIMUTH_TOL_RAD: f64 = 0.5_f64 * std::f64::consts::PI / 180.0;

/// A single sun position.
#[derive(Debug, Clone, Copy)]
pub struct SunSample {
    /// Azimuth in degrees clockwise from North.
    pub azimuth_deg: f64,
    /// Altitude above the horizon, in degrees.
    pub altitude_deg: f64,
}

impl SunSample {
    /// Create a sun position from its azimuth (degrees clockwise from North)
    /// and altitude (degrees above the horizon).
    pub fn new(azimuth_deg: f64, altitude_deg: f64) -> Self {
        Self {
            azimuth_deg,
            altitude_deg,
        }
    }

    fn altitude_rad(&self) -> f64 {
        self.altitude_deg.to_radians()
    }

    /// Convert to math-radians (0 = N, clockwise) — same convention as
    /// `horizon_angle_map`.
    fn azimuth_rad(&self) -> f64 {
        self.azimuth_deg.to_radians()
    }
}

/// Parameters for [`ray_shade`].
#[derive(Debug, Clone)]
pub struct RayShadeParams {
    /// Sun position(s). Multiple samples = soft shadows.
    pub suns: Vec<SunSample>,
    /// Maximum ray-march distance in cells. Past this distance the ray is
    /// assumed unoccluded. For rayshader-equivalent quality use
    /// `max(rows, cols)`; for speed at the cost of "missing" the longest
    /// cast shadows, set a lower value.
    pub radius: usize,
}

impl Default for RayShadeParams {
    /// Nominal sun at 315° azimuth, 45° altitude, single sample (hard
    /// shadows), radius 1000 cells. The default radius is intended to be
    /// "long enough for typical 1–2 K² DEMs"; for larger DEMs override
    /// it to `max(rows, cols)`.
    fn default() -> Self {
        Self {
            suns: vec![SunSample::new(315.0, 45.0)],
            radius: 1000,
        }
    }
}

impl RayShadeParams {
    /// Build params with `n` evenly-spaced sun samples spanning
    /// `±half_spread_deg` in azimuth around `nominal`, all at the same
    /// altitude. This is the rayshader-equivalent soft-shadow recipe.
    pub fn with_soft_shadow(
        nominal_azimuth_deg: f64,
        altitude_deg: f64,
        n_samples: usize,
        half_spread_deg: f64,
    ) -> Self {
        let mut suns = Vec::with_capacity(n_samples.max(1));
        if n_samples <= 1 {
            suns.push(SunSample::new(nominal_azimuth_deg, altitude_deg));
        } else {
            let step = 2.0 * half_spread_deg / (n_samples - 1) as f64;
            for i in 0..n_samples {
                let az = nominal_azimuth_deg - half_spread_deg + step * i as f64;
                suns.push(SunSample::new(az, altitude_deg));
            }
        }
        Self { suns, radius: 1000 }
    }

    /// Soft shadow over altitude (matches rayshader's `anglebreaks` =
    /// `seq(low, high, by=1)` pattern more closely than azimuth spread).
    /// All samples share the same azimuth.
    pub fn with_soft_shadow_altitude(
        azimuth_deg: f64,
        low_alt_deg: f64,
        high_alt_deg: f64,
        n_samples: usize,
    ) -> Self {
        let mut suns = Vec::with_capacity(n_samples.max(1));
        if n_samples <= 1 {
            suns.push(SunSample::new(
                azimuth_deg,
                (low_alt_deg + high_alt_deg) * 0.5,
            ));
        } else {
            let step = (high_alt_deg - low_alt_deg) / (n_samples - 1) as f64;
            for i in 0..n_samples {
                suns.push(SunSample::new(azimuth_deg, low_alt_deg + step * i as f64));
            }
        }
        Self { suns, radius: 1000 }
    }
}

/// Ray-traced cast shadows.
///
/// For each cell, returns intensity in `[0, 1]` where `1` = fully lit and
/// `0` = fully in shadow. Soft-shadow penumbra arises when
/// `params.suns.len() > 1`: the binary lit/shadow mask is averaged over
/// the samples.
///
/// NaN cells in the DEM pass through as NaN in the output.
///
/// # Errors
///
/// - [`ReliefError::InvalidParam`] if `params.suns` is empty.
/// - [`ReliefError::Algorithm`] if the underlying horizon-angle computation
///   fails.
pub fn ray_shade(dem: &Raster<f64>, params: &RayShadeParams) -> Result<Raster<f64>> {
    if params.suns.is_empty() {
        return Err(ReliefError::InvalidParam(
            "params.suns must contain at least one sample".into(),
        ));
    }

    // Fast path: all sun samples share an azimuth (the rayshader recipe).
    // Compute one horizon-tan map and threshold each altitude in O(N).
    if shared_azimuth(&params.suns) {
        return ray_shade_amortised(dem, params);
    }

    ray_shade_per_sun(dem, params)
}

/// Returns true iff every sun sample's azimuth is within
/// `SHARED_AZIMUTH_TOL_RAD` of the first.
fn shared_azimuth(suns: &[SunSample]) -> bool {
    if suns.len() <= 1 {
        return true;
    }
    let a0 = suns[0].azimuth_rad();
    suns.iter()
        .skip(1)
        .all(|s| (s.azimuth_rad() - a0).abs() <= SHARED_AZIMUTH_TOL_RAD)
}

/// Single-azimuth amortisation: one horizon_tan_map call + one
/// thresholding per altitude. Equivalent to `ray_shade_per_sun` when
/// every sun shares an azimuth (the rayshader anglebreaks pattern).
fn ray_shade_amortised(dem: &Raster<f64>, params: &RayShadeParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let n_samples = params.suns.len() as f64;
    let inv_n = 1.0 / n_samples;

    let azimuth_rad = params.suns[0].azimuth_rad();
    let horizon = horizon_tan_map(dem, azimuth_rad, params.radius)?;
    let horizon_data = horizon.data();

    let tan_alts: Vec<f64> = params.suns.iter().map(|s| s.altitude_rad().tan()).collect();

    let mut acc = vec![0.0f64; rows * cols];
    for (i, &h) in horizon_data.iter().enumerate() {
        if h.is_nan() {
            acc[i] = f64::NAN;
            continue;
        }
        let mut lit_count = 0usize;
        for &t in &tan_alts {
            if t >= h {
                lit_count += 1;
            }
        }
        acc[i] = lit_count as f64 * inv_n;
    }

    let mut out = Raster::new(rows, cols);
    out.set_transform(*dem.transform());
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() =
        Array2::from_shape_vec((rows, cols), acc).map_err(|e| ReliefError::Shape(e.to_string()))?;
    Ok(out)
}

/// Per-sun ray-march fallback (general case, varying azimuths).
fn ray_shade_per_sun(dem: &Raster<f64>, params: &RayShadeParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let n_samples = params.suns.len() as f64;
    let inv_n = 1.0 / n_samples;

    let mut acc = vec![0.0f64; rows * cols];

    for sun in &params.suns {
        let mask = cast_shadow_ray_mask(dem, sun.azimuth_rad(), sun.altitude_rad(), params.radius)?;
        for (i, v) in mask.data().iter().enumerate() {
            if v.is_nan() {
                acc[i] = f64::NAN;
                continue;
            }
            if !acc[i].is_nan() {
                acc[i] += v * inv_n;
            }
        }
    }

    let mut out = Raster::new(rows, cols);
    out.set_transform(*dem.transform());
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() =
        Array2::from_shape_vec((rows, cols), acc).map_err(|e| ReliefError::Shape(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_dem(rows: usize, cols: usize, elev: f64) -> Raster<f64> {
        let mut dem = Raster::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                dem.set(r, c, elev).unwrap();
            }
        }
        dem
    }

    fn ridge_dem(rows: usize, cols: usize, peak_col: usize, height: f64) -> Raster<f64> {
        let mut dem = Raster::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let d = (c as f64 - peak_col as f64).abs();
                let e = (height - d).max(0.0);
                dem.set(r, c, e).unwrap();
            }
        }
        dem
    }

    #[test]
    fn errors_on_empty_suns() {
        let dem = flat_dem(4, 4, 100.0);
        let params = RayShadeParams {
            suns: vec![],
            radius: 4,
        };
        assert!(matches!(
            ray_shade(&dem, &params),
            Err(ReliefError::InvalidParam(_))
        ));
    }

    #[test]
    fn flat_dem_high_sun_is_fully_lit() {
        let dem = flat_dem(8, 8, 100.0);
        let params = RayShadeParams {
            suns: vec![SunSample::new(315.0, 60.0)],
            radius: 4,
        };
        let out = ray_shade(&dem, &params).unwrap();
        // Every interior cell should be lit.
        for r in 1..7 {
            for c in 1..7 {
                let v = out.get(r, c).unwrap();
                assert!(v.is_nan() || (v - 1.0).abs() < 1e-9, "cell ({r},{c}) = {v}");
            }
        }
    }

    #[test]
    fn ridge_shadows_the_lit_side() {
        // Ridge peaking at the centre. With sun coming from the east (low
        // altitude), cells west of the ridge sit in shadow; cells east of
        // it are lit.
        let dem = ridge_dem(8, 16, 8, 5.0);
        let params = RayShadeParams {
            suns: vec![SunSample::new(90.0, 5.0)], // sun in the east, very low
            radius: 16,
        };
        let out = ray_shade(&dem, &params).unwrap();

        // Tally lit cells on each side of the ridge for an interior row.
        let row = 4;
        let mut east_lit = 0;
        let mut east_total = 0;
        let mut west_lit = 0;
        let mut west_total = 0;
        for c in 0..16 {
            let v = out.get(row, c).unwrap();
            if v.is_nan() {
                continue;
            }
            if c > 9 {
                east_total += 1;
                if v > 0.5 {
                    east_lit += 1;
                }
            } else if c < 7 {
                west_total += 1;
                if v > 0.5 {
                    west_lit += 1;
                }
            }
        }
        let east_frac = east_lit as f64 / east_total.max(1) as f64;
        let west_frac = west_lit as f64 / west_total.max(1) as f64;
        assert!(
            east_frac > west_frac,
            "east lit-fraction ({east_frac}) should exceed west ({west_frac})"
        );
    }

    #[test]
    fn amortised_matches_per_sun_on_shared_azimuth() {
        // The fast path (shared azimuth) and the general path must agree
        // numerically when run on the same suns. Mid-altitude ridge so we
        // get a mix of lit, partial, and shadowed cells.
        let dem = ridge_dem(8, 16, 8, 5.0);
        let params = RayShadeParams::with_soft_shadow_altitude(90.0, 3.0, 8.0, 6);
        let amortised = ray_shade_amortised(&dem, &params).unwrap();
        let per_sun = ray_shade_per_sun(&dem, &params).unwrap();
        for r in 0..8 {
            for c in 0..16 {
                let a = amortised.get(r, c).unwrap();
                let b = per_sun.get(r, c).unwrap();
                if a.is_nan() || b.is_nan() {
                    assert_eq!(a.is_nan(), b.is_nan(), "nan mismatch at ({r},{c})");
                    continue;
                }
                assert!((a - b).abs() < 1e-9, "({r},{c}) amortised={a} per_sun={b}");
            }
        }
    }

    #[test]
    fn shared_azimuth_detection_works() {
        let single = vec![SunSample::new(315.0, 45.0)];
        assert!(shared_azimuth(&single));

        let shared = (0..11)
            .map(|i| SunSample::new(315.0, 40.0 + i as f64))
            .collect::<Vec<_>>();
        assert!(shared_azimuth(&shared));

        let spread = RayShadeParams::with_soft_shadow(315.0, 45.0, 5, 10.0).suns;
        assert!(!shared_azimuth(&spread), "5° spread should NOT be shared");
    }

    #[test]
    fn soft_shadow_intensity_between_zero_and_one() {
        // Multiple sun samples close to grazing should produce a partial
        // mask: at least one cell should land in (0, 1).
        let dem = ridge_dem(8, 16, 8, 5.0);
        let params = RayShadeParams::with_soft_shadow(90.0, 5.0, 8, 5.0);
        let out = ray_shade(&dem, &params).unwrap();
        let mut saw_partial = false;
        for v in out.data().iter() {
            if !v.is_nan() && *v > 0.05 && *v < 0.95 {
                saw_partial = true;
                break;
            }
        }
        assert!(
            saw_partial,
            "expected at least one cell with partial shadow"
        );
    }
}
