//! Gram-Schmidt pansharpening.
//!
//! Reference: Laben, C. A., & Brower, B. V. (2000). "Process for
//! enhancing the spatial resolution of multispectral imagery using
//! pan-sharpening." US Patent 6,011,875. The patent expired in
//! January 2018; the method is in the public domain.
//!
//! Algorithm:
//!
//! 1. Build the synthetic low-resolution pan `P_lr` as the per-pixel
//!    mean of the MS bands.
//! 2. Run Gram-Schmidt on the sequence `(P_lr, MS_1, ..., MS_B)`,
//!    centred per band. Record the regression coefficients
//!    `φ_{k,i} = <a_k, GS_i> / <GS_i, GS_i>` (here `a_k =
//!    MS_k − μ_{MS_k}` and `GS_i` is the centred GS vector at
//!    position `i`).
//! 3. Histogram-match the real high-resolution pan to `P_lr`:
//!    `pan' = (pan − μ_pan) · σ(P_lr) / σ(pan)`.
//! 4. Substitute `GS_0 ← pan'` and invert. Because the inverse
//!    `a_k = GS_k + Σ_{i<k} φ_{k,i} · GS_i` uses the same
//!    coefficients, and `GS_i` for `i ≥ 1` is unchanged, the
//!    pansharpened band is
//!
//!      MS'_k = GS_k + μ_{MS_k} + φ_{k,0} · pan'
//!            + Σ_{i=1..k-1} φ_{k,i} · GS_i .
//!
//! The injection of `pan'` into the first GS slot transfers the
//! spatial frequencies of the real pan into every MS band while
//! preserving the per-band spectral signature (mean) and the
//! between-band correlations encoded in `φ`.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Apply Gram-Schmidt pansharpening (Laben & Brower 2000).
///
/// `pan` and every `ms` band must live on the same grid (shape,
/// geotransform and EPSG-comparable CRS — the MS bands are assumed
/// to be already resampled to the pan grid). Pixels where any input
/// is NaN are emitted as NaN in every output band.
pub fn gram_schmidt(pan: &Raster<f64>, ms: &[&Raster<f64>]) -> Result<Vec<Raster<f64>>> {
    if ms.is_empty() {
        return Err(Error::Algorithm(
            "Gram-Schmidt needs at least one MS band".into(),
        ));
    }
    let mut all: Vec<&Raster<f64>> = Vec::with_capacity(ms.len() + 1);
    all.push(pan);
    all.extend_from_slice(ms);
    surtgis_core::raster::check_aligned(&all)?;
    let (rows, cols) = pan.shape();
    let n_bands = ms.len();
    let n_px = rows * cols;
    let inv_b = 1.0 / n_bands as f64;

    // Identify valid pixels (no NaN in pan or any MS band).
    let mut valid_mask = vec![false; n_px];
    let mut n_valid = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            let pan_v = unsafe { pan.get_unchecked(row, col) };
            if !pan_v.is_finite() {
                continue;
            }
            let mut ok = true;
            for b in ms.iter() {
                if !unsafe { b.get_unchecked(row, col) }.is_finite() {
                    ok = false;
                    break;
                }
            }
            if ok {
                valid_mask[p] = true;
                n_valid += 1;
            }
        }
    }
    if n_valid < 2 {
        return Err(Error::Algorithm(
            "Gram-Schmidt: need ≥2 valid pixels".into(),
        ));
    }
    let inv_nv = 1.0 / n_valid as f64;

    // Per-band means.
    let mut means = vec![0.0f64; n_bands];
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            for (k, b) in ms.iter().enumerate() {
                means[k] += unsafe { b.get_unchecked(row, col) };
            }
        }
    }
    for m in means.iter_mut() {
        *m *= inv_nv;
    }

    // Synthetic low-res pan = mean of MS per pixel, centred over
    // valid pixels.
    let mut p_lr = vec![f64::NAN; n_px];
    let mut p_lr_sum = 0.0;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            let mut s = 0.0;
            for b in ms.iter() {
                s += unsafe { b.get_unchecked(row, col) };
            }
            p_lr[p] = s * inv_b;
            p_lr_sum += p_lr[p];
        }
    }
    let p_lr_mean = p_lr_sum * inv_nv;

    // Centred Gram-Schmidt sequence: gs[0] = P_lr − μ_{P_lr},
    // gs[k] for k=1..=B is built sequentially from MS_k centred.
    // We carry only the *centred* GS values; the inversion later
    // adds the MS-band mean back to recover absolute reflectance.
    let mut gs: Vec<Vec<f64>> = (0..=n_bands).map(|_| vec![f64::NAN; n_px]).collect();
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            gs[0][p] = p_lr[p] - p_lr_mean;
        }
    }
    let mut gs_dot_self: Vec<f64> = vec![0.0; n_bands + 1];
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            gs_dot_self[0] += gs[0][p] * gs[0][p];
        }
    }

    // phi[k][i] = <a_k, GS_i> / <GS_i, GS_i>, k in 1..=B, i < k.
    let mut phi: Vec<Vec<f64>> = (0..=n_bands).map(|k| vec![0.0; k]).collect();

    for k in 1..=n_bands {
        // Compute phi[k][i] for i = 0..k.
        // a_k = MS_{k-1} - means[k-1]
        for i in 0..k {
            let denom = gs_dot_self[i].max(1e-30);
            let mut num = 0.0;
            for row in 0..rows {
                for col in 0..cols {
                    let p = row * cols + col;
                    if !valid_mask[p] {
                        continue;
                    }
                    let a = unsafe { ms[k - 1].get_unchecked(row, col) } - means[k - 1];
                    num += a * gs[i][p];
                }
            }
            phi[k][i] = num / denom;
        }
        // Build GS_k = a_k - Σ phi[k][i] · GS_i.
        for row in 0..rows {
            for col in 0..cols {
                let p = row * cols + col;
                if !valid_mask[p] {
                    continue;
                }
                let a = unsafe { ms[k - 1].get_unchecked(row, col) } - means[k - 1];
                let mut v = a;
                for i in 0..k {
                    v -= phi[k][i] * gs[i][p];
                }
                gs[k][p] = v;
            }
        }
        // Update <GS_k, GS_k> for the next iteration.
        let mut s = 0.0;
        for row in 0..rows {
            for col in 0..cols {
                let p = row * cols + col;
                if !valid_mask[p] {
                    continue;
                }
                s += gs[k][p] * gs[k][p];
            }
        }
        gs_dot_self[k] = s;
    }

    // Histogram-match pan to P_lr (== GS_0 after centring).
    let mut pan_sum = 0.0;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            pan_sum += unsafe { pan.get_unchecked(row, col) };
        }
    }
    let pan_mean = pan_sum * inv_nv;
    let mut pan_var = 0.0;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            let d = unsafe { pan.get_unchecked(row, col) } - pan_mean;
            pan_var += d * d;
        }
    }
    pan_var *= inv_nv;
    let pan_std = pan_var.sqrt().max(1e-12);
    let p_lr_std = (gs_dot_self[0] * inv_nv).sqrt().max(1e-12);
    let pan_scale = p_lr_std / pan_std;
    let mut pan_matched = vec![f64::NAN; n_px];
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            pan_matched[p] = (unsafe { pan.get_unchecked(row, col) } - pan_mean) * pan_scale;
        }
    }

    // Invert: MS'_k = GS_k + μ_{MS_k} + Σ_{i<k} phi[k][i] · GS'_i
    // with GS'_0 = pan_matched, GS'_i = GS_i for i ≥ 1.
    let mut out_data: Vec<Vec<f64>> = (0..n_bands).map(|_| vec![f64::NAN; n_px]).collect();
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            for k in 1..=n_bands {
                let mut s = gs[k][p];
                // i = 0 uses the substituted pan.
                s += phi[k][0] * pan_matched[p];
                for i in 1..k {
                    s += phi[k][i] * gs[i][p];
                }
                out_data[k - 1][p] = s + means[k - 1];
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

    fn from_grid(grid: &[&[f64]]) -> Raster<f64> {
        let rows = grid.len();
        let cols = grid[0].len();
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for (row, row_vals) in grid.iter().enumerate() {
            for (col, &v) in row_vals.iter().enumerate() {
                r.set(row, col, v).unwrap();
            }
        }
        r
    }

    fn filled(rows: usize, cols: usize, v: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, v);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn pan_equal_to_synth_recovers_ms() {
        // When pan equals the synthetic P_lr exactly, the matched
        // pan equals the centred P_lr, and the GS inverse should
        // reconstruct the MS bands bit-for-bit (no spatial info to
        // inject).
        let ms0 = from_grid(&[
            &[10.0, 30.0, 50.0, 70.0],
            &[20.0, 40.0, 60.0, 80.0],
            &[15.0, 35.0, 55.0, 75.0],
            &[25.0, 45.0, 65.0, 85.0],
        ]);
        let ms1 = from_grid(&[
            &[20.0, 50.0, 80.0, 110.0],
            &[30.0, 60.0, 90.0, 120.0],
            &[25.0, 55.0, 85.0, 115.0],
            &[35.0, 65.0, 95.0, 125.0],
        ]);
        // pan = (ms0 + ms1) / 2.
        let mut pan = Raster::new(4, 4);
        pan.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        for row in 0..4 {
            for col in 0..4 {
                let v = (ms0.get(row, col).unwrap() + ms1.get(row, col).unwrap()) * 0.5;
                pan.set(row, col, v).unwrap();
            }
        }
        let result = gram_schmidt(&pan, &[&ms0, &ms1]).unwrap();
        for row in 0..4 {
            for col in 0..4 {
                let d0 = (result[0].get(row, col).unwrap() - ms0.get(row, col).unwrap()).abs();
                let d1 = (result[1].get(row, col).unwrap() - ms1.get(row, col).unwrap()).abs();
                assert!(d0 < 1e-9, "MS0 drifted at ({},{}): {}", row, col, d0);
                assert!(d1 < 1e-9, "MS1 drifted at ({},{}): {}", row, col, d1);
            }
        }
    }

    #[test]
    fn pansharpening_preserves_per_band_mean() {
        // Property: per-band mean of the pansharpened output should
        // equal the per-band mean of the input MS. The centred
        // injection of pan' has zero mean, so the band-mean stays.
        let ms0 = from_grid(&[
            &[10.0, 30.0, 50.0, 70.0],
            &[20.0, 40.0, 60.0, 80.0],
            &[15.0, 35.0, 55.0, 75.0],
            &[25.0, 45.0, 65.0, 85.0],
        ]);
        let ms1 = from_grid(&[
            &[8.0, 28.0, 48.0, 68.0],
            &[18.0, 38.0, 58.0, 78.0],
            &[13.0, 33.0, 53.0, 73.0],
            &[23.0, 43.0, 63.0, 83.0],
        ]);
        // pan is independent of MS but well-correlated.
        let pan = from_grid(&[
            &[11.0, 31.0, 49.0, 71.0],
            &[19.0, 41.0, 59.0, 79.0],
            &[16.0, 34.0, 56.0, 74.0],
            &[24.0, 46.0, 64.0, 84.0],
        ]);
        let result = gram_schmidt(&pan, &[&ms0, &ms1]).unwrap();
        let mean = |r: &Raster<f64>| {
            let mut s = 0.0;
            for row in 0..4 {
                for col in 0..4 {
                    s += r.get(row, col).unwrap();
                }
            }
            s / 16.0
        };
        let m0_in = mean(&ms0);
        let m1_in = mean(&ms1);
        let m0_out = mean(&result[0]);
        let m1_out = mean(&result[1]);
        assert!(
            (m0_in - m0_out).abs() < 1e-6,
            "MS0 mean drift: {} → {}",
            m0_in,
            m0_out
        );
        assert!(
            (m1_in - m1_out).abs() < 1e-6,
            "MS1 mean drift: {} → {}",
            m1_in,
            m1_out
        );
    }

    #[test]
    fn nan_propagates_to_all_bands() {
        let mut ms0 = filled(4, 4, 50.0);
        for col in 0..4 {
            for row in 0..4 {
                ms0.set(row, col, 50.0 + col as f64 * 10.0).unwrap();
            }
        }
        ms0.set(0, 0, f64::NAN).unwrap();
        let mut ms1 = filled(4, 4, 100.0);
        for col in 0..4 {
            for row in 0..4 {
                ms1.set(row, col, 100.0 + col as f64 * 15.0).unwrap();
            }
        }
        let mut pan = filled(4, 4, 75.0);
        for col in 0..4 {
            for row in 0..4 {
                pan.set(row, col, 75.0 + col as f64 * 12.0).unwrap();
            }
        }
        let result = gram_schmidt(&pan, &[&ms0, &ms1]).unwrap();
        for b in &result {
            assert!(b.get(0, 0).unwrap().is_nan());
        }
        assert!(result[0].nodata().is_some_and(|nd| nd.is_nan()));
    }

    #[test]
    fn rejects_empty_ms() {
        let pan = filled(3, 3, 1.0);
        assert!(gram_schmidt(&pan, &[]).is_err());
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let pan = filled(3, 3, 1.0);
        let ms = filled(3, 4, 1.0);
        assert!(gram_schmidt(&pan, &[&ms]).is_err());
    }
}
