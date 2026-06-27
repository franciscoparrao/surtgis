//! SAR (Synthetic Aperture Radar) water and flood mapping primitives.
//!
//! A minimal, dependency-free SAR toolkit aimed at Sentinel-1-style dual-pol
//! flood mapping. It deliberately reuses the existing imagery and temporal
//! machinery for the heavy lifting:
//!
//! - **Multi-temporal compositing** (min / median backscatter to suppress
//!   transient bright returns) → use [`crate::temporal::temporal_min`] /
//!   [`crate::imagery::median_composite`].
//! - **Amplitude change detection** (flood = backscatter drop vs. a dry
//!   reference) → use [`crate::imagery::raster_difference`] or
//!   [`crate::imagery::ir_mad`] on the backscatter bands.
//!
//! This module adds the SAR-specific pieces that were missing:
//!
//! - [`linear_to_db`] / [`db_to_linear`] — backscatter unit conversion, needed
//!   because water-detection thresholds are conventionally expressed in dB.
//! - [`dual_pol_water_index`] — the normalized co-/cross-pol difference
//!   `(VV - VH) / (VV + VH)`, which is elevated over smooth open water.
//! - [`sar_water_mask`] — threshold a backscatter (or index) raster into a
//!   binary water/flood mask.
//!
//! A speckle filter (Lee / refined Lee) is intentionally **not** included here;
//! it is tracked as a follow-up. Pre-filtering with a speckle filter improves
//! these results but is not required to obtain a usable first-pass mask,
//! especially after multi-temporal compositing.

use crate::imagery::normalized_difference;
use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// u8 mask value for non-water cells.
pub const SAR_NON_WATER: u8 = 0;
/// u8 mask value for water / flooded cells.
pub const SAR_WATER: u8 = 1;
/// u8 mask value for nodata cells.
pub const SAR_NODATA: u8 = 255;

#[inline]
fn is_nodata(v: f64, nodata: Option<f64>) -> bool {
    v.is_nan() || nodata.map(|nd| v == nd).unwrap_or(false)
}

/// Map a single-band f64 raster elementwise (row-parallel), returning a new
/// f64 raster whose nodata is `NaN`.
fn map_f64<F>(src: &Raster<f64>, f: F) -> Result<Raster<f64>>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    let (rows, cols) = src.shape();
    let nodata = src.nodata();
    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let v = unsafe { src.get_unchecked(row, col) };
                if is_nodata(v, nodata) {
                    continue;
                }
                *out = f(v);
            }
            row_data
        })
        .collect();

    let mut output = src.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

/// Convert linear-power backscatter (σ⁰, γ⁰ …) to decibels: `10·log₁₀(x)`.
///
/// Non-positive or nodata cells become `NaN`. Already-calibrated products from
/// Planetary Computer (`sentinel-1-rtc`) are in linear power and should be
/// converted with this before applying dB thresholds.
pub fn linear_to_db(backscatter: &Raster<f64>) -> Result<Raster<f64>> {
    map_f64(backscatter, |x| {
        if x <= 0.0 { f64::NAN } else { 10.0 * x.log10() }
    })
}

/// Convert decibel backscatter back to linear power: `10^(x/10)`.
pub fn db_to_linear(backscatter_db: &Raster<f64>) -> Result<Raster<f64>> {
    map_f64(backscatter_db, |x| 10.0_f64.powf(x / 10.0))
}

/// Dual-polarisation water index: `(co_pol - cross_pol) / (co_pol + cross_pol)`.
///
/// For Sentinel-1 pass `co_pol = VV`, `cross_pol = VH` (linear power). Smooth
/// open water depolarises weakly, so VH collapses faster than VV and the index
/// rises toward +1 over water while staying low over rough land. Use
/// [`sar_water_mask`] with a positive threshold and `water_below = false` to
/// extract water from this index.
///
/// This is the SAR analogue of the optical normalized-difference water indices
/// and shares their numerics via [`normalized_difference`].
pub fn dual_pol_water_index(co_pol: &Raster<f64>, cross_pol: &Raster<f64>) -> Result<Raster<f64>> {
    normalized_difference(co_pol, cross_pol)
}

/// Threshold a backscatter or index raster into a binary water/flood mask.
///
/// # Arguments
/// * `raster` - input backscatter (e.g. VV in dB) or a water index.
/// * `threshold` - decision boundary.
/// * `water_below` - if `true`, cells `< threshold` are water (the usual case
///   for backscatter: open water is specular and returns little energy); if
///   `false`, cells `> threshold` are water (the case for a water *index* that
///   is high over water).
///
/// Returns a `u8` raster using [`SAR_WATER`], [`SAR_NON_WATER`] and
/// [`SAR_NODATA`] (with the raster nodata set to [`SAR_NODATA`]). `NaN` /
/// nodata input cells map to [`SAR_NODATA`].
pub fn sar_water_mask(
    raster: &Raster<f64>,
    threshold: f64,
    water_below: bool,
) -> Result<Raster<u8>> {
    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();
    let data: Vec<u8> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![SAR_NODATA; cols];
            for (col, out) in row_data.iter_mut().enumerate() {
                let v = unsafe { raster.get_unchecked(row, col) };
                if is_nodata(v, nodata) {
                    continue;
                }
                let is_water = if water_below {
                    v < threshold
                } else {
                    v > threshold
                };
                *out = if is_water { SAR_WATER } else { SAR_NON_WATER };
            }
            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(SAR_NODATA));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn r(data: Vec<f64>, rows: usize, cols: usize) -> Raster<f64> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut x = Raster::from_array(arr);
        x.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        x
    }

    #[test]
    fn test_linear_db_roundtrip() {
        let lin = r(vec![1.0, 0.1, 0.01, 10.0], 2, 2);
        let db = linear_to_db(&lin).unwrap();
        // 10*log10(1)=0, 0.1->-10, 0.01->-20, 10->10
        assert!((db.get(0, 0).unwrap() - 0.0).abs() < 1e-9);
        assert!((db.get(0, 1).unwrap() + 10.0).abs() < 1e-9);
        assert!((db.get(1, 0).unwrap() + 20.0).abs() < 1e-9);
        assert!((db.get(1, 1).unwrap() - 10.0).abs() < 1e-9);
        // round-trip
        let back = db_to_linear(&db).unwrap();
        for (a, b) in lin.data().iter().zip(back.data().iter()) {
            assert!((a - b).abs() < 1e-9);
        }
    }

    #[test]
    fn test_linear_to_db_nonpositive_is_nan() {
        let lin = r(vec![0.0, -1.0, 0.5, f64::NAN], 2, 2);
        let db = linear_to_db(&lin).unwrap();
        assert!(db.get(0, 0).unwrap().is_nan());
        assert!(db.get(0, 1).unwrap().is_nan());
        assert!(!db.get(1, 0).unwrap().is_nan());
        assert!(db.get(1, 1).unwrap().is_nan());
    }

    #[test]
    fn test_dual_pol_index_water_high() {
        // Water: VV moderate, VH near zero -> index near +1.
        // Land: VV ~ VH -> index near 0.
        let vv = r(vec![0.1, 0.2, 0.2, 0.2], 2, 2);
        let vh = r(vec![0.001, 0.18, 0.2, 0.19], 2, 2);
        let idx = dual_pol_water_index(&vv, &vh).unwrap();
        assert!(idx.get(0, 0).unwrap() > 0.9, "water cell should be near +1");
        assert!(idx.get(1, 0).unwrap().abs() < 1e-6, "equal pol -> 0");
    }

    #[test]
    fn test_sar_water_mask_below() {
        // Backscatter in dB: water is low (< -17 dB typical for VV).
        let vv_db = r(vec![-20.0, -5.0, -18.0, -3.0], 2, 2);
        let mask = sar_water_mask(&vv_db, -17.0, true).unwrap();
        assert_eq!(mask.get(0, 0).unwrap(), SAR_WATER); // -20 < -17
        assert_eq!(mask.get(0, 1).unwrap(), SAR_NON_WATER); // -5
        assert_eq!(mask.get(1, 0).unwrap(), SAR_WATER); // -18
        assert_eq!(mask.get(1, 1).unwrap(), SAR_NON_WATER); // -3
    }

    #[test]
    fn test_sar_water_mask_above_and_nodata() {
        // Index mode: water is HIGH; include a NaN cell.
        let idx = r(vec![0.8, 0.1, f64::NAN, 0.6], 2, 2);
        let mask = sar_water_mask(&idx, 0.5, false).unwrap();
        assert_eq!(mask.get(0, 0).unwrap(), SAR_WATER);
        assert_eq!(mask.get(0, 1).unwrap(), SAR_NON_WATER);
        assert_eq!(mask.get(1, 0).unwrap(), SAR_NODATA);
        assert_eq!(mask.get(1, 1).unwrap(), SAR_WATER);
        assert_eq!(mask.nodata(), Some(SAR_NODATA));
    }
}
