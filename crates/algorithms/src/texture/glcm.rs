//! Gray-Level Co-occurrence Matrix (GLCM) texture features
//!
//! Computes Haralick texture measures from a GLCM built at each pixel's
//! neighborhood. Values are quantized to `n_levels` gray levels.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Available GLCM texture measures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlcmTexture {
    /// Angular Second Moment (energy) — uniformity
    Energy,
    /// Contrast — local intensity variation
    Contrast,
    /// Homogeneity (Inverse Difference Moment)
    Homogeneity,
    /// Correlation — linear dependency of gray levels
    Correlation,
    /// Entropy — randomness/disorder
    Entropy,
    /// Dissimilarity — weighted difference
    Dissimilarity,
}

/// Parameters for GLCM computation
#[derive(Debug, Clone)]
pub struct GlcmParams {
    /// Window radius for GLCM computation
    pub radius: usize,
    /// Number of quantization levels (default: 32)
    pub n_levels: usize,
    /// Distance for co-occurrence (default: 1)
    pub distance: usize,
    /// Which texture measure to compute
    pub texture: GlcmTexture,
}

impl Default for GlcmParams {
    fn default() -> Self {
        Self {
            radius: 3,
            n_levels: 32,
            distance: 1,
            texture: GlcmTexture::Contrast,
        }
    }
}

/// Compute a single GLCM-based texture feature.
///
/// For each pixel, builds a GLCM from the surrounding window using
/// 4 directions (0°, 45°, 90°, 135°) and averages the result.
///
/// # Arguments
/// * `raster` - Input raster (continuous values quantized internally)
/// * `params` - GLCM parameters
pub fn haralick_glcm(raster: &Raster<f64>, params: GlcmParams) -> Result<Raster<f64>> {
    let textures = [params.texture];
    let mut results = haralick_glcm_multi(raster, &params, &textures)?;
    Ok(results.pop().unwrap())
}

/// Compute multiple GLCM texture features in a single pass.
///
/// The GLCM is the dominant cost per pixel; computing N features is
/// effectively the same cost as one. This entry point builds the
/// GLCM once per window and evaluates every requested feature on it,
/// so requesting N features runs ~N× faster than calling
/// `haralick_glcm` N times.
///
/// The `texture` field of `params` is ignored — the `textures` slice
/// determines which features are produced. Returns one raster per
/// requested feature, in the same order.
pub fn haralick_glcm_multi(
    raster: &Raster<f64>,
    params: &GlcmParams,
    textures: &[GlcmTexture],
) -> Result<Vec<Raster<f64>>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("GLCM radius must be > 0".into()));
    }
    if params.n_levels < 2 {
        return Err(Error::Algorithm("GLCM n_levels must be >= 2".into()));
    }
    if textures.is_empty() {
        return Err(Error::Algorithm(
            "haralick_glcm_multi needs at least one texture".into(),
        ));
    }

    let (rows, cols) = raster.shape();

    // Find min/max for quantization
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    for r in 0..rows {
        for c in 0..cols {
            let v = unsafe { raster.get_unchecked(r, c) };
            if v.is_finite() {
                vmin = vmin.min(v);
                vmax = vmax.max(v);
            }
        }
    }

    if vmin >= vmax {
        return Err(Error::Algorithm(
            "Raster has no value range for GLCM".into(),
        ));
    }

    let range = vmax - vmin;
    let n = params.n_levels;
    let d = params.distance as isize;
    let r = params.radius as isize;
    let n_tex = textures.len();

    // Direction offsets: 0°, 45°, 90°, 135°
    let directions: [(isize, isize); 4] = [(0, d), (-d, d), (-d, 0), (-d, -d)];

    // Each parallel chunk emits `n_tex` rows of length `cols`,
    // interleaved feature-major (feature 0 cols, feature 1 cols, ...).
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_block = vec![f64::NAN; n_tex * cols];
            let mut glcm = vec![0.0; n * n];

            for col in 0..cols {
                // Build GLCM from window
                glcm.fill(0.0);
                let mut total = 0.0;

                for dir in &directions {
                    for dr in -r..=r {
                        for dc in -r..=r {
                            let r1 = row as isize + dr;
                            let c1 = col as isize + dc;
                            let r2 = r1 + dir.0;
                            let c2 = c1 + dir.1;

                            if r1 >= 0
                                && c1 >= 0
                                && (r1 as usize) < rows
                                && (c1 as usize) < cols
                                && r2 >= 0
                                && c2 >= 0
                                && (r2 as usize) < rows
                                && (c2 as usize) < cols
                            {
                                let v1 = unsafe { raster.get_unchecked(r1 as usize, c1 as usize) };
                                let v2 = unsafe { raster.get_unchecked(r2 as usize, c2 as usize) };

                                if v1.is_finite() && v2.is_finite() {
                                    let i = quantize(v1, vmin, range, n);
                                    let j = quantize(v2, vmin, range, n);
                                    glcm[i * n + j] += 1.0;
                                    glcm[j * n + i] += 1.0; // Symmetric
                                    total += 2.0;
                                }
                            }
                        }
                    }
                }

                if total < 1.0 {
                    continue;
                }

                // Normalize GLCM
                for v in &mut glcm {
                    *v /= total;
                }

                // Evaluate every requested feature on the shared GLCM.
                for (k, &tex) in textures.iter().enumerate() {
                    row_block[k * cols + col] = compute_texture(&glcm, n, tex);
                }
            }

            row_block
        })
        .collect();

    // Slice the interleaved block back into one Raster per feature.
    let mut outputs = Vec::with_capacity(n_tex);
    let row_stride = n_tex * cols;
    for k in 0..n_tex {
        let mut buf = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            let start = row * row_stride + k * cols;
            buf.extend_from_slice(&output_data[start..start + cols]);
        }
        let mut out = raster.with_same_meta::<f64>(rows, cols);
        out.set_nodata(Some(f64::NAN));
        *out.data_mut() =
            Array2::from_shape_vec((rows, cols), buf).map_err(|e| Error::Other(e.to_string()))?;
        outputs.push(out);
    }

    Ok(outputs)
}

fn quantize(value: f64, vmin: f64, range: f64, n_levels: usize) -> usize {
    let normalized = (value - vmin) / range;
    let level = (normalized * (n_levels - 1) as f64).round() as usize;
    level.min(n_levels - 1)
}

fn compute_texture(glcm: &[f64], n: usize, texture: GlcmTexture) -> f64 {
    match texture {
        GlcmTexture::Energy => glcm.iter().map(|p| p * p).sum(),
        GlcmTexture::Contrast => {
            let mut val = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let diff = (i as f64 - j as f64).powi(2);
                    val += glcm[i * n + j] * diff;
                }
            }
            val
        }
        GlcmTexture::Homogeneity => {
            let mut val = 0.0;
            for i in 0..n {
                for j in 0..n {
                    val += glcm[i * n + j] / (1.0 + (i as f64 - j as f64).abs());
                }
            }
            val
        }
        GlcmTexture::Correlation => {
            // μ_i, μ_j, σ_i, σ_j
            let mut mu_i = 0.0;
            let mut mu_j = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let p = glcm[i * n + j];
                    mu_i += i as f64 * p;
                    mu_j += j as f64 * p;
                }
            }
            let mut sig_i = 0.0;
            let mut sig_j = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let p = glcm[i * n + j];
                    sig_i += (i as f64 - mu_i).powi(2) * p;
                    sig_j += (j as f64 - mu_j).powi(2) * p;
                }
            }
            sig_i = sig_i.sqrt();
            sig_j = sig_j.sqrt();

            if sig_i < 1e-15 || sig_j < 1e-15 {
                return 0.0;
            }

            let mut corr = 0.0;
            for i in 0..n {
                for j in 0..n {
                    corr += glcm[i * n + j] * (i as f64 - mu_i) * (j as f64 - mu_j);
                }
            }
            corr / (sig_i * sig_j)
        }
        GlcmTexture::Entropy => {
            let mut val = 0.0;
            for p in glcm {
                if *p > 0.0 {
                    val -= p * p.ln();
                }
            }
            val
        }
        GlcmTexture::Dissimilarity => {
            let mut val = 0.0;
            for i in 0..n {
                for j in 0..n {
                    val += glcm[i * n + j] * (i as f64 - j as f64).abs();
                }
            }
            val
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn uniform_raster(size: usize, val: f64) -> Raster<f64> {
        let mut r = Raster::filled(size, size, val);
        r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        r
    }

    fn gradient_raster(size: usize) -> Raster<f64> {
        let mut r = Raster::new(size, size);
        r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        for row in 0..size {
            for col in 0..size {
                r.set(row, col, (row * size + col) as f64).unwrap();
            }
        }
        r
    }

    #[test]
    fn test_glcm_energy_uniform() {
        // Uniform raster should have no value range
        let r = uniform_raster(10, 5.0);
        let result = haralick_glcm(
            &r,
            GlcmParams {
                texture: GlcmTexture::Energy,
                ..Default::default()
            },
        );
        assert!(result.is_err()); // No range
    }

    #[test]
    fn test_glcm_contrast_gradient() {
        let r = gradient_raster(20);
        let result = haralick_glcm(
            &r,
            GlcmParams {
                texture: GlcmTexture::Contrast,
                radius: 2,
                n_levels: 16,
                ..Default::default()
            },
        )
        .unwrap();

        let v = result.get(10, 10).unwrap();
        assert!(v > 0.0, "Gradient should have contrast > 0, got {}", v);
    }

    #[test]
    fn test_glcm_entropy_gradient() {
        let r = gradient_raster(20);
        let result = haralick_glcm(
            &r,
            GlcmParams {
                texture: GlcmTexture::Entropy,
                radius: 2,
                n_levels: 16,
                ..Default::default()
            },
        )
        .unwrap();

        let v = result.get(10, 10).unwrap();
        assert!(v > 0.0, "Gradient should have entropy > 0, got {}", v);
    }

    #[test]
    fn reference_binary_3x3() {
        // Hand-computed reference. 3x3 raster, n_levels=2, radius=1
        // (window covers the whole raster), distance=1, symmetric,
        // averaged over the 4 standard directions:
        //
        //   0 0 1
        //   0 1 1
        //   0 0 1
        //
        // The normalised GLCM at (1,1) is:
        //   p(0,0) = 0.300   p(0,1) = 0.225
        //   p(1,0) = 0.225   p(1,1) = 0.250
        //
        // Reference texture values:
        //   Energy        = Σp²            = 0.25375
        //   Contrast      = Σp·(i-j)²      = 0.45
        //   Homogeneity   = Σp/(1+|i-j|)   = 0.775
        //   Dissimilarity = Σp·|i-j|       = 0.45
        //   Entropy       = -Σp·ln(p)      ≈ 1.37907
        //   Correlation   ≈ 0.09773
        let mut r = Raster::new(3, 3);
        r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let g = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]];
        for row in 0..3 {
            for col in 0..3 {
                r.set(row, col, g[row][col]).unwrap();
            }
        }
        let base = GlcmParams {
            radius: 1,
            n_levels: 2,
            distance: 1,
            texture: GlcmTexture::Contrast,
        };
        let textures = [
            GlcmTexture::Energy,
            GlcmTexture::Contrast,
            GlcmTexture::Homogeneity,
            GlcmTexture::Dissimilarity,
            GlcmTexture::Entropy,
            GlcmTexture::Correlation,
        ];
        let expected = [0.25375, 0.45, 0.775, 0.45, 1.37907_f64, 0.09773_f64];
        let result = haralick_glcm_multi(&r, &base, &textures).unwrap();
        for (k, (&tex, &exp)) in textures.iter().zip(expected.iter()).enumerate() {
            let v = result[k].get(1, 1).unwrap();
            // Entropy/correlation are floating-point reductions; widen
            // tolerance to 1e-4 for them. Closed-form features hit 1e-12.
            let tol = match tex {
                GlcmTexture::Entropy | GlcmTexture::Correlation => 1e-4,
                _ => 1e-12,
            };
            assert!(
                (v - exp).abs() < tol,
                "{:?}: expected {}, got {} (tol {})",
                tex,
                exp,
                v,
                tol
            );
        }
    }

    #[test]
    fn multi_matches_per_feature() {
        // Multi-pass output must equal per-feature outputs cell-by-cell.
        let r = gradient_raster(20);
        let base = GlcmParams {
            radius: 2,
            n_levels: 16,
            distance: 1,
            texture: GlcmTexture::Contrast, // ignored by multi
        };
        let textures = [
            GlcmTexture::Energy,
            GlcmTexture::Contrast,
            GlcmTexture::Homogeneity,
            GlcmTexture::Correlation,
            GlcmTexture::Entropy,
            GlcmTexture::Dissimilarity,
        ];
        let multi = haralick_glcm_multi(&r, &base, &textures).unwrap();
        assert_eq!(multi.len(), textures.len());
        for (k, &tex) in textures.iter().enumerate() {
            let single = haralick_glcm(
                &r,
                GlcmParams {
                    texture: tex,
                    ..base.clone()
                },
            )
            .unwrap();
            for row in 0..20 {
                for col in 0..20 {
                    let a = multi[k].get(row, col).unwrap();
                    let b = single.get(row, col).unwrap();
                    if a.is_nan() {
                        assert!(
                            b.is_nan(),
                            "multi NaN but single finite at {}, {}",
                            row,
                            col
                        );
                    } else {
                        assert!(
                            (a - b).abs() < 1e-12,
                            "feature {:?} mismatch at ({}, {}): multi={} single={}",
                            tex,
                            row,
                            col,
                            a,
                            b
                        );
                    }
                }
            }
        }
    }
}
