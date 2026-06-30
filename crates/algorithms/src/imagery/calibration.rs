//! Radiometric calibration: DN ↔ reflectance plus DOS1 atmospheric
//! correction.
//!
//! These are the pure-math primitives. Callers provide the
//! calibration coefficients explicitly (typically read from the
//! sensor metadata: `MTL.txt` for Landsat, `MTD_MSIL1C.xml` for
//! Sentinel-2, or the STAC item properties). Metadata parsing is
//! intentionally out of scope here — pipeline code that already
//! consumes STAC can pass the coefficients straight through.
//!
//! References:
//! - Landsat 8/9 Collection 2 Level-1 Data Format Control Book
//!   (USGS, LSDS-1822) — TOA reflectance: `ρ = (M_p · DN + A_p) /
//!   sin(SUN_ELEVATION)`.
//! - Landsat Collection 2 Level-2 Science Product Guide
//!   (USGS, LSDS-1619) — surface reflectance: `ρ = 0.0000275 · DN
//!   − 0.2`.
//! - Sentinel-2 Products Specification Document
//!   (ESA, S2-PDGS-TAS-DI-PSD) — L1C TOA: `ρ = (DN + offset) /
//!   quantification_value`. From baseline 04.00 (2022-01-25),
//!   `offset = −1000` for harmonised products.
//! - Chavez, P.S. (1988). An improved dark-object subtraction
//!   technique for atmospheric scattering correction of multispectral
//!   data. Remote Sensing of Environment 24(3), 459–479.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

// ─── Landsat 8/9 Collection 2 Level-1 → TOA reflectance ─────────────

/// Per-band coefficients for Landsat 8/9 Collection 2 Level-1 → TOA reflectance.
#[derive(Debug, Clone, Copy)]
pub struct LandsatToaParams {
    /// `M_p` from MTL: `REFLECTANCE_MULT_BAND_x` (typically 2e-5).
    pub multiplicative: f64,
    /// `A_p` from MTL: `REFLECTANCE_ADD_BAND_x` (typically −0.1).
    pub additive: f64,
    /// `SUN_ELEVATION` from MTL, in degrees (acquisition geometry).
    pub sun_elevation_deg: f64,
}

/// Landsat 8/9 Collection 2 Level-1 DN → TOA reflectance.
///
/// `ρ_TOA = (M_p · DN + A_p) / sin(SUN_ELEVATION)`. The sin term
/// corrects for solar zenith. NaN cells are preserved. Sentinel
/// nodata (if set on the input raster) is converted to NaN in the
/// output and `output.nodata` is set to `Some(NaN)`.
pub fn dn_to_toa_landsat(raster: &Raster<f64>, params: LandsatToaParams) -> Result<Raster<f64>> {
    if params.sun_elevation_deg <= 0.0 || params.sun_elevation_deg > 90.0 {
        return Err(Error::Algorithm(
            "sun_elevation_deg must be in (0, 90]".into(),
        ));
    }
    let sin_elev = params.sun_elevation_deg.to_radians().sin();
    let nodata = raster.nodata();
    let m = params.multiplicative;
    let a = params.additive;
    map_per_pixel(raster, nodata, |dn| (m * dn + a) / sin_elev)
}

// ─── Landsat Collection 2 Level-2 → surface reflectance ─────────────

/// Landsat Collection 2 Level-2 DN → surface reflectance (fixed
/// USGS coefficients: scale `2.75e-5`, offset `−0.2`). Out-of-range
/// values (`DN == 0` is "fill" per USGS) become NaN.
pub fn dn_to_surface_reflectance_landsat_c2(raster: &Raster<f64>) -> Result<Raster<f64>> {
    let nodata = raster.nodata();
    map_per_pixel(raster, nodata, |dn| {
        if dn <= 0.0 {
            f64::NAN
        } else {
            0.0000275 * dn - 0.2
        }
    })
}

// ─── Sentinel-2 → TOA / BOA reflectance ─────────────────────────────

/// Quantification value and offset for Sentinel-2 → TOA/BOA reflectance.
#[derive(Debug, Clone, Copy)]
pub struct S2ReflectanceParams {
    /// `BOA_QUANTIFICATION_VALUE` (L2A) or `QUANTIFICATION_VALUE`
    /// (L1C) from `MTD_MSIL*.xml`. Defaults to `10000`.
    pub quantification_value: f64,
    /// `BOA_ADD_OFFSET` per band (introduced in PSD baseline 04.00,
    /// 2022-01-25). Pre-baseline products: `0`. Post-baseline
    /// harmonised products: `−1000`.
    pub offset: f64,
}

impl Default for S2ReflectanceParams {
    fn default() -> Self {
        Self {
            quantification_value: 10000.0,
            offset: 0.0,
        }
    }
}

/// Sentinel-2 DN → reflectance. Applies to both L1C (TOA) and L2A
/// (BOA) — the formula is the same, only the metadata source for
/// `quantification_value` and `offset` differs.
///
/// `ρ = (DN + offset) / quantification_value`. NaN passthrough.
pub fn dn_to_reflectance_s2(
    raster: &Raster<f64>,
    params: S2ReflectanceParams,
) -> Result<Raster<f64>> {
    if params.quantification_value <= 0.0 {
        return Err(Error::Algorithm("quantification_value must be > 0".into()));
    }
    let nodata = raster.nodata();
    let q = params.quantification_value;
    let off = params.offset;
    map_per_pixel(raster, nodata, |dn| (dn + off) / q)
}

// ─── DOS1 (dark-object subtraction) ────────────────────────────────

/// Parameters for DOS1 (dark-object subtraction) atmospheric correction.
#[derive(Debug, Clone, Copy)]
pub struct Dos1Params {
    /// Quantile used to estimate the per-band dark-object value.
    /// The default `0.001` (= the 0.1 % darkest pixel) is robust
    /// against outliers from sensor artefacts. Set to `0.0` to use
    /// the absolute minimum.
    pub dark_object_quantile: f64,
}

impl Default for Dos1Params {
    fn default() -> Self {
        Self {
            dark_object_quantile: 0.001,
        }
    }
}

/// Dark-object subtraction (DOS1, simplified per-band variant).
///
/// Estimates path radiance as the per-band `dark_object_quantile`
/// percentile of the finite values and subtracts it from every
/// pixel, clamping the result to be non-negative. This is the
/// simplified DOS1 variant most often used in operational
/// preprocessing — the user is expected to apply it **before** TOA
/// → DN conversion or apply it to DN data directly. Full Chavez
/// 1988 DOS1 additionally rescales by `E_SUN / cos(θ_z) · π · d²`
/// and is out of scope here — chain with `dn_to_toa_landsat` /
/// `dn_to_reflectance_s2` afterwards for an equivalent effect.
///
/// NaN cells are preserved. Returns an error if the raster has no
/// finite values to estimate the dark object.
pub fn dos1(raster: &Raster<f64>, params: Dos1Params) -> Result<Raster<f64>> {
    if !(0.0..=1.0).contains(&params.dark_object_quantile) {
        return Err(Error::Algorithm(
            "dark_object_quantile must be in [0, 1]".into(),
        ));
    }
    let nodata = raster.nodata();
    let (rows, cols) = raster.shape();

    // Collect finite values to estimate the quantile.
    let mut finite: Vec<f64> = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { raster.get_unchecked(row, col) };
            if !is_nodata(v, nodata) {
                finite.push(v);
            }
        }
    }
    if finite.is_empty() {
        return Err(Error::Algorithm(
            "DOS1: raster has no finite values to estimate dark object".into(),
        ));
    }
    finite.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((finite.len() as f64 - 1.0) * params.dark_object_quantile).round() as usize;
    let dark = finite[idx.min(finite.len() - 1)];

    map_per_pixel(raster, nodata, |v| (v - dark).max(0.0))
}

// ─── Shared helpers ────────────────────────────────────────────────

#[inline]
fn is_nodata(val: f64, nodata: Option<f64>) -> bool {
    if val.is_nan() {
        return true;
    }
    if let Some(nd) = nodata
        && !nd.is_nan()
        && (val - nd).abs() < f64::EPSILON
    {
        return true;
    }
    false
}

fn map_per_pixel<F>(raster: &Raster<f64>, nodata: Option<f64>, f: F) -> Result<Raster<f64>>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let (rows, cols) = raster.shape();
    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let v = unsafe { raster.get_unchecked(row, col) };
                if is_nodata(v, nodata) {
                    continue;
                }
                let r = f(v);
                *out = if r.is_nan() { f64::NAN } else { r };
            }
            row_data
        })
        .collect();
    let mut out = raster.with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn filled(rows: usize, cols: usize, v: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, v);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn landsat_toa_reference_value() {
        // Reference value: USGS Landsat 8 example.
        //   M_p = 2e-5, A_p = -0.1, SUN_ELEVATION = 60°, DN = 10000
        //   ρ = (2e-5 · 10000 + (-0.1)) / sin(60°)
        //     = 0.1 / 0.8660254...
        //     ≈ 0.11547005383792515
        let r = filled(4, 4, 10000.0);
        let result = dn_to_toa_landsat(
            &r,
            LandsatToaParams {
                multiplicative: 2e-5,
                additive: -0.1,
                sun_elevation_deg: 60.0,
            },
        )
        .unwrap();
        let v = result.get(2, 2).unwrap();
        let expected = 0.1 / (60.0_f64.to_radians()).sin();
        assert!(
            (v - expected).abs() < 1e-12,
            "expected {}, got {}",
            expected,
            v
        );
    }

    #[test]
    fn landsat_toa_rejects_invalid_sun_elevation() {
        let r = filled(2, 2, 100.0);
        assert!(
            dn_to_toa_landsat(
                &r,
                LandsatToaParams {
                    multiplicative: 1.0,
                    additive: 0.0,
                    sun_elevation_deg: 0.0,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn landsat_c2_sr_reference_value() {
        // Reference: DN = 20000 → ρ = 2.75e-5 · 20000 − 0.2 = 0.35.
        let r = filled(3, 3, 20000.0);
        let result = dn_to_surface_reflectance_landsat_c2(&r).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!((v - 0.35).abs() < 1e-12, "expected 0.35, got {}", v);
    }

    #[test]
    fn landsat_c2_sr_zero_dn_becomes_nan() {
        // DN == 0 is "fill" per USGS spec → NaN in output.
        let r = filled(2, 2, 0.0);
        let result = dn_to_surface_reflectance_landsat_c2(&r).unwrap();
        assert!(result.get(0, 0).unwrap().is_nan());
    }

    #[test]
    fn s2_reflectance_reference_value() {
        // Pre-baseline product: offset = 0, quantification = 10000.
        // DN = 2500 → ρ = 2500 / 10000 = 0.25.
        let r = filled(3, 3, 2500.0);
        let result = dn_to_reflectance_s2(&r, S2ReflectanceParams::default()).unwrap();
        assert!((result.get(0, 0).unwrap() - 0.25).abs() < 1e-12);
    }

    #[test]
    fn s2_reflectance_post_baseline_offset() {
        // Post-baseline 04.00 (2022-01-25): offset = -1000.
        // DN = 2500 → ρ = (2500 - 1000) / 10000 = 0.15.
        let r = filled(3, 3, 2500.0);
        let result = dn_to_reflectance_s2(
            &r,
            S2ReflectanceParams {
                quantification_value: 10000.0,
                offset: -1000.0,
            },
        )
        .unwrap();
        assert!((result.get(0, 0).unwrap() - 0.15).abs() < 1e-12);
    }

    #[test]
    fn s2_reflectance_rejects_zero_quantification() {
        let r = filled(2, 2, 1000.0);
        assert!(
            dn_to_reflectance_s2(
                &r,
                S2ReflectanceParams {
                    quantification_value: 0.0,
                    offset: 0.0,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn nan_passthrough_all_calibrators() {
        let mut r = Raster::new(3, 3);
        r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        for row in 0..3 {
            for col in 0..3 {
                r.set(row, col, 1000.0).unwrap();
            }
        }
        r.set(1, 1, f64::NAN).unwrap();
        let a = dn_to_toa_landsat(
            &r,
            LandsatToaParams {
                multiplicative: 2e-5,
                additive: -0.1,
                sun_elevation_deg: 60.0,
            },
        )
        .unwrap();
        let b = dn_to_surface_reflectance_landsat_c2(&r).unwrap();
        let c = dn_to_reflectance_s2(&r, S2ReflectanceParams::default()).unwrap();
        for out in &[a, b, c] {
            assert!(out.get(1, 1).unwrap().is_nan());
            assert!(out.nodata().is_some_and(|nd| nd.is_nan()));
        }
    }

    #[test]
    fn dos1_dark_cell_goes_to_zero() {
        // Min value 50, max value 200. With quantile=0.0 we expect
        // the absolute minimum to become 0 and the max to become 150.
        let mut r = Raster::new(4, 4);
        r.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        for row in 0..4 {
            for col in 0..4 {
                r.set(row, col, 100.0 + (row + col) as f64 * 10.0).unwrap();
            }
        }
        r.set(0, 0, 50.0).unwrap();
        r.set(3, 3, 200.0).unwrap();
        let result = dos1(
            &r,
            Dos1Params {
                dark_object_quantile: 0.0,
            },
        )
        .unwrap();
        assert!((result.get(0, 0).unwrap() - 0.0).abs() < 1e-12);
        assert!((result.get(3, 3).unwrap() - 150.0).abs() < 1e-12);
    }

    #[test]
    fn dos1_preserves_monotonicity() {
        // After subtracting a constant, the rank order of pixels
        // is preserved (clamped at 0 from below).
        let mut r = Raster::new(5, 5);
        r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                r.set(row, col, (row * 5 + col) as f64 + 100.0).unwrap();
            }
        }
        let result = dos1(&r, Dos1Params::default()).unwrap();
        for row in 0..5 {
            for col in 0..5 {
                let raw = (row * 5 + col) as f64 + 100.0;
                let val = result.get(row, col).unwrap();
                if raw <= 100.0 {
                    assert!(val < 1e-9);
                } else {
                    assert!(val >= 0.0);
                }
            }
        }
        // Order is preserved at the boundary between any two cells.
        for row in 0..5 {
            for col in 0..4 {
                let a = result.get(row, col).unwrap();
                let b = result.get(row, col + 1).unwrap();
                assert!(b >= a, "monotonicity broken at ({},{})", row, col);
            }
        }
    }

    #[test]
    fn dos1_errors_on_all_nan() {
        let mut r = Raster::new(3, 3);
        r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        for row in 0..3 {
            for col in 0..3 {
                r.set(row, col, f64::NAN).unwrap();
            }
        }
        assert!(dos1(&r, Dos1Params::default()).is_err());
    }

    #[test]
    fn dos1_nan_passthrough() {
        let mut r = Raster::new(4, 4);
        r.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        for row in 0..4 {
            for col in 0..4 {
                r.set(row, col, 100.0 + (row + col) as f64).unwrap();
            }
        }
        r.set(1, 1, f64::NAN).unwrap();
        let result = dos1(&r, Dos1Params::default()).unwrap();
        assert!(result.get(1, 1).unwrap().is_nan());
        assert!(result.nodata().is_some_and(|nd| nd.is_nan()));
    }
}
