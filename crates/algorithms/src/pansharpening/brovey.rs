//! Brovey transform pansharpening.
//!
//! Reference: Gillespie, A. R., Kahle, A. B., & Walker, R. E.
//! (1987). "Color enhancement of highly correlated images. II.
//! Channel ratio and 'chromaticity' transformation techniques."
//! Remote Sensing of Environment 22(3), 343–365.
//!
//! Algorithm:
//!
//!   P_synth(x) = mean_b(MS_b(x))
//!   out_b(x)   = MS_b(x) · pan(x) / P_synth(x)
//!
//! `P_synth` is the synthetic low-resolution pan built by averaging
//! the input MS bands at every pixel. The per-band ratio preserves
//! the original spectral relationships while injecting the high
//! spatial frequencies from the real pan. Fast and simple; can
//! oversaturate radiometric values when the synthetic and real pan
//! differ strongly.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Apply the Brovey transform.
///
/// `pan` and every `ms` band must live on the same grid (shape,
/// geotransform and EPSG-comparable CRS — the MS bands are assumed
/// to be already resampled to the pan grid). Pixels where any input
/// is NaN, or where `P_synth ≤ 0`, are emitted as NaN. Returns one
/// output band per input MS band.
pub fn brovey(pan: &Raster<f64>, ms: &[&Raster<f64>]) -> Result<Vec<Raster<f64>>> {
    if ms.is_empty() {
        return Err(Error::Algorithm("Brovey needs at least one MS band".into()));
    }
    let mut all: Vec<&Raster<f64>> = Vec::with_capacity(ms.len() + 1);
    all.push(pan);
    all.extend_from_slice(ms);
    surtgis_core::raster::check_aligned(&all)?;
    let (rows, cols) = pan.shape();
    let n_bands = ms.len();
    let n_px = rows * cols;
    let inv_n = 1.0 / n_bands as f64;

    // Per-pixel scratch for output bands.
    let mut out_data: Vec<Vec<f64>> = (0..n_bands).map(|_| vec![f64::NAN; n_px]).collect();

    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            let pan_v = unsafe { pan.get_unchecked(row, col) };
            if !pan_v.is_finite() {
                continue;
            }
            let mut synth = 0.0;
            let mut any_nan = false;
            let mut band_vals: Vec<f64> = Vec::with_capacity(n_bands);
            for b in ms.iter().take(n_bands) {
                let v = unsafe { b.get_unchecked(row, col) };
                if !v.is_finite() {
                    any_nan = true;
                    break;
                }
                synth += v;
                band_vals.push(v);
            }
            if any_nan {
                continue;
            }
            synth *= inv_n;
            if synth <= 0.0 {
                continue; // Division by zero / sign-flipping guard
            }
            let factor = pan_v / synth;
            for (b, &v) in band_vals.iter().enumerate() {
                out_data[b][p] = v * factor;
            }
        }
    }

    let mut outputs = Vec::with_capacity(n_bands);
    for band_data in out_data.into_iter() {
        let mut out = pan.with_same_meta::<f64>(rows, cols);
        out.set_nodata(Some(f64::NAN));
        *out.data_mut() = Array2::from_shape_vec((rows, cols), band_data)
            .map_err(|e| Error::Other(e.to_string()))?;
        outputs.push(out);
    }
    Ok(outputs)
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
    fn pan_equal_to_synth_preserves_ms() {
        // pan = mean(MS) → factor = 1 → output equals MS bit-for-bit.
        let ms0 = filled(5, 5, 100.0);
        let ms1 = filled(5, 5, 200.0);
        let ms2 = filled(5, 5, 300.0);
        let pan = filled(5, 5, 200.0); // mean(100, 200, 300) = 200
        let result = brovey(&pan, &[&ms0, &ms1, &ms2]).unwrap();
        assert_eq!(result.len(), 3);
        for row in 0..5 {
            for col in 0..5 {
                assert!((result[0].get(row, col).unwrap() - 100.0).abs() < 1e-12);
                assert!((result[1].get(row, col).unwrap() - 200.0).abs() < 1e-12);
                assert!((result[2].get(row, col).unwrap() - 300.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn pan_double_synth_doubles_all_bands() {
        let ms0 = filled(3, 3, 50.0);
        let ms1 = filled(3, 3, 150.0);
        let pan = filled(3, 3, 200.0); // 2 × mean(50, 150)
        let result = brovey(&pan, &[&ms0, &ms1]).unwrap();
        assert!((result[0].get(0, 0).unwrap() - 100.0).abs() < 1e-12);
        assert!((result[1].get(0, 0).unwrap() - 300.0).abs() < 1e-12);
    }

    #[test]
    fn nan_in_any_input_propagates_to_all_outputs() {
        let mut ms0 = filled(2, 2, 100.0);
        ms0.set(0, 0, f64::NAN).unwrap();
        let ms1 = filled(2, 2, 200.0);
        let pan = filled(2, 2, 150.0);
        let result = brovey(&pan, &[&ms0, &ms1]).unwrap();
        for b in &result {
            assert!(b.get(0, 0).unwrap().is_nan());
        }
        assert!(result[0].nodata().is_some_and(|nd| nd.is_nan()));
    }

    #[test]
    fn zero_synth_emits_nan() {
        // mean(MS) = 0 → division would explode → emit NaN.
        let ms0 = filled(2, 2, 0.0);
        let ms1 = filled(2, 2, 0.0);
        let pan = filled(2, 2, 10.0);
        let result = brovey(&pan, &[&ms0, &ms1]).unwrap();
        for b in &result {
            for row in 0..2 {
                for col in 0..2 {
                    assert!(b.get(row, col).unwrap().is_nan());
                }
            }
        }
    }

    #[test]
    fn rejects_empty_ms() {
        let pan = filled(3, 3, 1.0);
        assert!(brovey(&pan, &[]).is_err());
    }

    #[test]
    fn rejects_mismatched_shape() {
        let pan = filled(3, 3, 1.0);
        let ms = filled(3, 4, 1.0);
        assert!(brovey(&pan, &[&ms]).is_err());
    }
}
