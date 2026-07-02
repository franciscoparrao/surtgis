//! PCA-based pansharpening.
//!
//! Reference: Chavez, P. S., Sides, S. C., & Anderson, J. A.
//! (1991). "Comparison of three different methods to merge
//! multiresolution and multispectral data: Landsat TM and SPOT
//! panchromatic." Photogrammetric Engineering & Remote Sensing
//! 57(3), 295–303.
//!
//! Algorithm:
//!
//! 1. Stack the upsampled MS bands into an `(N × B)` matrix `X`.
//! 2. Centre per band: `X_c = X − μ` where `μ ∈ ℝ^B`.
//! 3. Build the `(B × B)` covariance `C = (X_c^T · X_c) / N` and
//!    diagonalise it with the Jacobi solver from
//!    `classification::pca::jacobi_eigen`. Sort eigenvectors by
//!    eigenvalue descending → `V ∈ ℝ^{B × B}`.
//! 4. Project: `Y = X_c · V` (the principal components).
//! 5. Replace `Y[:, 0]` with a histogram-matched panchromatic band
//!    (zero-mean, scaled to match the std of the first PC).
//! 6. Invert: `X' = Y' · V^T + μ`.
//!
//! Works for any number of bands. The first PC is the
//! highest-variance linear combination of the MS bands — when MS
//! and pan are well-correlated this captures the intensity content
//! and substituting the pan injects the high spatial frequencies.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use crate::classification::pca::jacobi_eigen;

/// PCA pansharpening. NaN cells in any input are excluded from the
/// covariance / statistics computation and emitted as NaN in every
/// output band.
pub fn pca_pansharpen(pan: &Raster<f64>, ms: &[&Raster<f64>]) -> Result<Vec<Raster<f64>>> {
    if ms.is_empty() {
        return Err(Error::Algorithm(
            "PCA pansharpening needs at least one MS band".into(),
        ));
    }
    // Pan and MS bands must live on the same grid (shape,
    // geotransform and EPSG-comparable CRS): MS is assumed to be
    // already resampled to the pan grid.
    let mut all: Vec<&Raster<f64>> = Vec::with_capacity(ms.len() + 1);
    all.push(pan);
    all.extend_from_slice(ms);
    surtgis_core::raster::check_aligned(&all)?;
    let (rows, cols) = pan.shape();
    let n_bands = ms.len();
    let n_px = rows * cols;

    // Collect valid pixels (no NaN in pan or any MS band) and
    // compute per-band means.
    let mut valid_mask = vec![false; n_px];
    let mut means = vec![0.0f64; n_bands];
    let mut n_valid = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            let pan_v = unsafe { pan.get_unchecked(row, col) };
            if !pan_v.is_finite() {
                continue;
            }
            let mut any_nan = false;
            for b in ms.iter() {
                let v = unsafe { b.get_unchecked(row, col) };
                if !v.is_finite() {
                    any_nan = true;
                    break;
                }
            }
            if any_nan {
                continue;
            }
            valid_mask[p] = true;
            n_valid += 1;
            for (k, b) in ms.iter().enumerate() {
                means[k] += unsafe { b.get_unchecked(row, col) };
            }
        }
    }
    if n_valid < 2 {
        return Err(Error::Algorithm(
            "PCA pansharpening: need ≥2 valid pixels to estimate covariance".into(),
        ));
    }
    let inv_nv = 1.0 / n_valid as f64;
    for m in means.iter_mut() {
        *m *= inv_nv;
    }

    // Covariance over valid pixels.
    let mut cov = vec![vec![0.0f64; n_bands]; n_bands];
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            for k1 in 0..n_bands {
                let v1 = unsafe { ms[k1].get_unchecked(row, col) } - means[k1];
                for k2 in k1..n_bands {
                    let v2 = unsafe { ms[k2].get_unchecked(row, col) } - means[k2];
                    cov[k1][k2] += v1 * v2;
                }
            }
        }
    }
    for k1 in 0..n_bands {
        for k2 in k1..n_bands {
            cov[k1][k2] *= inv_nv;
            cov[k2][k1] = cov[k1][k2];
        }
    }

    // Eigendecomposition + sort descending.
    let (eigenvalues, eigenvectors) = jacobi_eigen(&cov, n_bands)?;
    let mut order: Vec<usize> = (0..n_bands).collect();
    order.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
    // V[k][j] = component k of the j-th sorted eigenvector.
    let v: Vec<Vec<f64>> = (0..n_bands)
        .map(|k| (0..n_bands).map(|j| eigenvectors[k][order[j]]).collect())
        .collect();

    // Project: compute PC1 per valid pixel + first-PC mean/std for
    // histogram matching. We only need PC1's stats; the other PCs
    // stream straight to the inversion stage.
    let mut pc1 = vec![0.0f64; n_px];
    let mut pc1_sum = 0.0;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            let mut s = 0.0;
            for k in 0..n_bands {
                s += (unsafe { ms[k].get_unchecked(row, col) } - means[k]) * v[k][0];
            }
            pc1[p] = s;
            pc1_sum += s;
        }
    }
    let pc1_mean = pc1_sum * inv_nv;
    let mut pc1_var = 0.0;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            let d = pc1[p] - pc1_mean;
            pc1_var += d * d;
        }
    }
    pc1_var *= inv_nv;
    let pc1_std = pc1_var.sqrt().max(1e-12);

    // Match pan to PC1: zero-mean, then scale to PC1 std + offset.
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
    let pan_scale = pc1_std / pan_std;

    // Invert: x'_b = Σ_j Y'[j] · V[b][j] + μ_b, where Y'[0] is the
    // matched pan and Y'[j>0] is the original projection.
    // Computing Y[j] (j>0) on the fly avoids materialising the full
    // N×B PC matrix.
    let mut out_data: Vec<Vec<f64>> = (0..n_bands).map(|_| vec![f64::NAN; n_px]).collect();
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            // Pan'(centred) — substitutes Y[:, 0].
            let pan_c = (unsafe { pan.get_unchecked(row, col) } - pan_mean) * pan_scale;
            // Original Y[1..B].
            let mut y_rest = vec![0.0f64; n_bands - 1];
            for (j, slot) in y_rest.iter_mut().enumerate() {
                let comp = j + 1;
                let mut s = 0.0;
                for k in 0..n_bands {
                    s += (unsafe { ms[k].get_unchecked(row, col) } - means[k]) * v[k][comp];
                }
                *slot = s;
            }
            // Back-project.
            for b in 0..n_bands {
                let mut s = pan_c * v[b][0];
                for j in 0..n_bands - 1 {
                    s += y_rest[j] * v[b][j + 1];
                }
                out_data[b][p] = s + means[b];
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
    fn pan_matching_first_pc_is_lossless() {
        // 2 bands with a clear first PC direction. Use pan = MS0
        // (which is itself the highest-variance band). After
        // pansharpening, the output should be very close to MS.
        let ms0 = from_grid(&[
            &[10.0, 30.0, 50.0, 70.0],
            &[20.0, 40.0, 60.0, 80.0],
            &[15.0, 35.0, 55.0, 75.0],
            &[25.0, 45.0, 65.0, 85.0],
        ]);
        let ms1 = from_grid(&[
            &[11.0, 31.0, 51.0, 71.0],
            &[21.0, 41.0, 61.0, 81.0],
            &[16.0, 36.0, 56.0, 76.0],
            &[26.0, 46.0, 66.0, 86.0],
        ]);
        let pan = from_grid(&[
            &[10.5, 30.5, 50.5, 70.5],
            &[20.5, 40.5, 60.5, 80.5],
            &[15.5, 35.5, 55.5, 75.5],
            &[25.5, 45.5, 65.5, 85.5],
        ]);
        let result = pca_pansharpen(&pan, &[&ms0, &ms1]).unwrap();
        assert_eq!(result.len(), 2);
        // When pan ≈ mean(MS) (collinear with PC1), reconstruction
        // should keep the MS values within a small tolerance.
        let mut diff_sum = 0.0;
        for row in 0..4 {
            for col in 0..4 {
                let d0 = (result[0].get(row, col).unwrap() - ms0.get(row, col).unwrap()).abs();
                let d1 = (result[1].get(row, col).unwrap() - ms1.get(row, col).unwrap()).abs();
                diff_sum += d0 + d1;
            }
        }
        let mean_abs_diff = diff_sum / (16.0 * 2.0);
        // The pan is essentially MS0 + 0.5 (very small offset) and
        // since the PCA captures the gradient direction, the
        // reconstruction stays within a few units.
        assert!(
            mean_abs_diff < 5.0,
            "PCA reconstruction drifted too far: {}",
            mean_abs_diff
        );
    }

    #[test]
    fn nan_propagates_to_all_bands() {
        let mut ms0 = filled(4, 4, 50.0);
        ms0.set(0, 0, f64::NAN).unwrap();
        let ms1 = filled(4, 4, 100.0);
        // Need spatial variation so covariance is not zero.
        let mut ms1 = ms1;
        for col in 0..4 {
            for row in 0..4 {
                ms1.set(row, col, 100.0 + col as f64 * 10.0).unwrap();
            }
        }
        let mut pan = filled(4, 4, 75.0);
        for col in 0..4 {
            for row in 0..4 {
                pan.set(row, col, 75.0 + col as f64 * 5.0).unwrap();
            }
        }
        let result = pca_pansharpen(&pan, &[&ms0, &ms1]).unwrap();
        for b in &result {
            assert!(b.get(0, 0).unwrap().is_nan());
        }
        assert!(result[0].nodata().is_some_and(|nd| nd.is_nan()));
    }

    #[test]
    fn rejects_empty_ms() {
        let pan = filled(3, 3, 1.0);
        assert!(pca_pansharpen(&pan, &[]).is_err());
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let pan = filled(3, 3, 1.0);
        let ms = filled(3, 4, 1.0);
        assert!(pca_pansharpen(&pan, &[&ms]).is_err());
    }
}
