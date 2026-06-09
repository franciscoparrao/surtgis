//! Inter-tile colour balancing for mosaics.
//!
//! Two policies, single-band:
//!
//! - [`histogram_match`] — empirical CDF matching. For every
//!   source pixel `v`, find its rank in the sorted finite source
//!   values, then emit the value at the same rank in the sorted
//!   reference. This is non-linear; matches the full distribution
//!   shape (not just `(μ, σ)`). Best when source and reference are
//!   captured under noticeably different conditions (different sun
//!   angle, atmosphere, sensor gain).
//!
//! - [`moment_match`] — linear `(μ, σ)` matching:
//!   `v' = (v − μ_src) · (σ_ref / σ_src) + μ_ref`. Cheap, preserves
//!   monotonicity, fits well when source and reference distributions
//!   are roughly Gaussian-shaped (typical for radiometrically
//!   calibrated reflectance bands).
//!
//! Both operate per band — apply once per spectral band you want
//! to colour-balance. NaN cells pass through unchanged.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Histogram (empirical CDF) matching. Source pixels are remapped
/// so their rank-ordered distribution matches the reference's.
pub fn histogram_match(source: &Raster<f64>, reference: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = source.shape();

    let mut src_sorted: Vec<f64> = source
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let mut ref_sorted: Vec<f64> = reference
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if src_sorted.is_empty() {
        return Err(Error::Algorithm(
            "histogram_match: source has no finite values".into(),
        ));
    }
    if ref_sorted.is_empty() {
        return Err(Error::Algorithm(
            "histogram_match: reference has no finite values".into(),
        ));
    }
    src_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_src = src_sorted.len();
    let n_ref = ref_sorted.len();

    let mut out_data = vec![f64::NAN; rows * cols];
    let denom = (n_src - 1).max(1) as f64;
    let ref_max = (n_ref - 1) as f64;
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { source.get_unchecked(row, col) };
            if !v.is_finite() {
                continue;
            }
            // Rank in source (≥ v gets the upper half — partition_point
            // returns the first index whose value is ≥ v).
            let rank = src_sorted.partition_point(|x| *x < v) as f64;
            // Linear interpolation between adjacent reference values
            // gives smoother output than nearest-rank.
            let pos = (rank / denom) * ref_max;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(n_ref - 1);
            let frac = pos - lo as f64;
            out_data[row * cols + col] =
                ref_sorted[lo] * (1.0 - frac) + ref_sorted[hi] * frac;
        }
    }

    let mut out = source.with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() = Array2::from_shape_vec((rows, cols), out_data)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(out)
}

/// Linear `(μ, σ)` matching. Output mean and stddev equal the
/// reference's. Monotonic in the source value.
pub fn moment_match(source: &Raster<f64>, reference: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = source.shape();
    let (mu_src, sigma_src) = mean_std(source.data().iter().copied());
    let (mu_ref, sigma_ref) = mean_std(reference.data().iter().copied());
    if sigma_src < 1e-12 {
        return Err(Error::Algorithm(
            "moment_match: source has zero variance".into(),
        ));
    }
    let scale = sigma_ref / sigma_src;

    let mut out_data = vec![f64::NAN; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { source.get_unchecked(row, col) };
            if !v.is_finite() {
                continue;
            }
            out_data[row * cols + col] = (v - mu_src) * scale + mu_ref;
        }
    }
    let mut out = source.with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() = Array2::from_shape_vec((rows, cols), out_data)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(out)
}

fn mean_std<I: Iterator<Item = f64>>(it: I) -> (f64, f64) {
    let mut n = 0usize;
    let mut s = 0.0;
    let mut s2 = 0.0;
    for v in it {
        if v.is_finite() {
            n += 1;
            s += v;
            s2 += v * v;
        }
    }
    if n == 0 {
        return (0.0, 0.0);
    }
    let mu = s / n as f64;
    let var = (s2 / n as f64) - mu * mu;
    (mu, var.max(0.0).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn ramp(rows: usize, cols: usize, scale: f64, offset: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, offset + scale * (row + col) as f64).unwrap();
            }
        }
        r
    }

    #[test]
    fn histogram_match_identity_when_distributions_equal() {
        // src and ref are the same → output ≈ src.
        let src = ramp(10, 10, 1.0, 0.0);
        let result = histogram_match(&src, &src).unwrap();
        for row in 0..10 {
            for col in 0..10 {
                let s = src.get(row, col).unwrap();
                let r = result.get(row, col).unwrap();
                assert!(
                    (s - r).abs() < 1e-9,
                    "identity drift at ({},{}): {} vs {}",
                    row,
                    col,
                    s,
                    r
                );
            }
        }
    }

    #[test]
    fn histogram_match_recovers_reference_distribution() {
        // Source = [0..100), reference = [200..300). After matching
        // the source should have stats matching the reference.
        let mut src = Raster::new(10, 10);
        src.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let mut refr = Raster::new(10, 10);
        refr.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for i in 0..100 {
            let row = i / 10;
            let col = i % 10;
            src.set(row, col, i as f64).unwrap();
            refr.set(row, col, 200.0 + i as f64).unwrap();
        }
        let result = histogram_match(&src, &refr).unwrap();
        let (mu_out, sigma_out) = mean_std(result.data().iter().copied());
        let (mu_ref, sigma_ref) = mean_std(refr.data().iter().copied());
        assert!(
            (mu_out - mu_ref).abs() < 0.5,
            "mean drift: {} vs {}",
            mu_out,
            mu_ref
        );
        assert!(
            (sigma_out - sigma_ref).abs() < 0.5,
            "sigma drift: {} vs {}",
            sigma_out,
            sigma_ref
        );
    }

    #[test]
    fn moment_match_exact_mean_and_std() {
        let src = ramp(20, 20, 1.0, 0.0);
        let refr = ramp(20, 20, 3.0, 100.0);
        let result = moment_match(&src, &refr).unwrap();
        let (mu_out, sigma_out) = mean_std(result.data().iter().copied());
        let (mu_ref, sigma_ref) = mean_std(refr.data().iter().copied());
        assert!(
            (mu_out - mu_ref).abs() < 1e-9,
            "mean: {} vs {}",
            mu_out,
            mu_ref
        );
        assert!(
            (sigma_out - sigma_ref).abs() < 1e-9,
            "sigma: {} vs {}",
            sigma_out,
            sigma_ref
        );
    }

    #[test]
    fn moment_match_rejects_zero_variance_source() {
        let mut src = Raster::filled(5, 5, 1.0);
        src.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let refr = ramp(5, 5, 1.0, 0.0);
        assert!(moment_match(&src, &refr).is_err());
    }

    #[test]
    fn nan_passthrough_histogram() {
        let mut src = ramp(8, 8, 1.0, 0.0);
        let refr = ramp(8, 8, 2.0, 0.0);
        src.set(0, 0, f64::NAN).unwrap();
        let result = histogram_match(&src, &refr).unwrap();
        assert!(result.get(0, 0).unwrap().is_nan());
        assert!(result.nodata().is_some_and(|nd| nd.is_nan()));
    }

    #[test]
    fn nan_passthrough_moment() {
        let mut src = ramp(8, 8, 1.0, 0.0);
        let refr = ramp(8, 8, 2.0, 0.0);
        src.set(3, 4, f64::NAN).unwrap();
        let result = moment_match(&src, &refr).unwrap();
        assert!(result.get(3, 4).unwrap().is_nan());
    }
}
