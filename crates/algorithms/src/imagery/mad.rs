//! Multivariate Alteration Detection (MAD) and Iteratively
//! Reweighted MAD (IR-MAD) for bitemporal change detection.
//!
//! References:
//! - Nielsen, A. A., Conradsen, K., & Simpson, J. J. (1998).
//!   "Multivariate Alteration Detection (MAD) and MAF
//!   Post-Processing in Multispectral, Bitemporal Image Data: New
//!   Approaches to Change Detection Studies." Remote Sensing of
//!   Environment 64(1), 1–19.
//! - Nielsen, A. A. (2007). "The Regularized Iteratively Reweighted
//!   MAD Method for Change Detection in Multi- and Hyperspectral
//!   Data." IEEE Transactions on Image Processing 16(2), 463–478.
//!
//! ## Algorithm sketch
//!
//! Given two `B`-band rasters `T1, T2` of the same scene at two
//! times, we look for the linear combinations
//!
//!   U_i = a_iᵀ T1,    V_i = b_iᵀ T2
//!
//! that are *maximally correlated* across the two dates — this is
//! the canonical correlation analysis (CCA) problem. The MAD
//! variates are the differences
//!
//!   MAD_i = U_i − V_i,    i = 1..B
//!
//! Because U_i, V_i are the components of maximum agreement, the
//! MAD_i isolate the parts of the spectrum that *changed*. The
//! first MAD typically dominates the change signal, with
//! decreasing magnitude as `i` grows. Each MAD_i has variance
//! `2·(1 − ρ_i)` where `ρ_i` is the i-th canonical correlation
//! returned alongside the rasters.
//!
//! IR-MAD (Nielsen 2007) refines this by iteratively re-estimating
//! the joint covariance over pixels weighted by `1 − F_χ²(T², B)`,
//! where T² is the standardised squared norm of the MAD vector at
//! each pixel. Pixels that look "unchanged" (small T²) get higher
//! weights, so the covariance is increasingly estimated on the
//! no-change population — the canonical solution. Convergence
//! gives much sharper change maps than one-shot MAD.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use crate::classification::pca::jacobi_eigen;

// ─── Public API ────────────────────────────────────────────────────

/// One-shot MAD result.
#[derive(Debug)]
pub struct MadResult {
    /// The `B` MAD variate rasters, sorted in descending order of
    /// `(1 − ρ_i)` — i.e. MAD_1 captures the largest change signal.
    pub mad: Vec<Raster<f64>>,
    /// Canonical correlations `ρ_i ∈ [0, 1]`, matching `mad` order.
    /// `ρ_i = 1` means the i-th canonical components U_i and V_i are
    /// perfectly correlated (no change in that direction);
    /// `ρ_i = 0` means total anti-correlation.
    pub correlations: Vec<f64>,
}

/// Parameters controlling the iteratively re-weighted MAD (IR-MAD) iteration.
#[derive(Debug, Clone)]
pub struct IrMadParams {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence threshold on the max canonical-correlation
    /// change between iterations.
    pub tol: f64,
    /// Regularisation added to the diagonal of `S_XX, S_YY` to
    /// guard against singular covariance (Nielsen 2007 §IV.B).
    /// In units of the per-band variance. `0.0` disables.
    pub regularisation: f64,
}

impl Default for IrMadParams {
    fn default() -> Self {
        Self {
            max_iter: 25,
            tol: 1e-3,
            regularisation: 0.0,
        }
    }
}

/// Output of an IR-MAD change-detection run.
#[derive(Debug)]
pub struct IrMadResult {
    /// One MAD variate raster per input band.
    pub mad: Vec<Raster<f64>>,
    /// Canonical correlation for each MAD variate.
    pub correlations: Vec<f64>,
    /// Per-pixel "no-change" probability from the chi-square test
    /// on the standardised MAD norm — `0` = clear change, `1` =
    /// clear no-change. NaN where any input is NaN.
    pub weights: Raster<f64>,
    /// Iterations actually run (≤ `params.max_iter`).
    pub n_iter: usize,
}

/// One-shot MAD: solve the CCA on the unweighted joint covariance,
/// emit `B` MAD variates and `B` canonical correlations.
pub fn mad(t1: &[&Raster<f64>], t2: &[&Raster<f64>]) -> Result<MadResult> {
    let (rows, cols, b) = validate_inputs(t1, t2)?;
    let valid_mask = build_valid_mask(t1, t2, rows, cols);
    let n_valid = valid_mask.iter().filter(|v| **v).count();
    if n_valid < 2 {
        return Err(Error::Algorithm("MAD: need ≥2 valid pixels".into()));
    }
    let weights: Option<&[f64]> = None;
    let (a, b_vec, rhos) = solve_cca(t1, t2, &valid_mask, weights, 0.0, b)?;
    let mad_rasters = project_mad(t1, t2, &a, &b_vec, rows, cols, b, &valid_mask)?;
    Ok(MadResult {
        mad: mad_rasters,
        correlations: rhos,
    })
}

/// IR-MAD: iteratively reweighted MAD (Nielsen 2007). Repeats MAD
/// with covariances re-estimated from the weighted samples until
/// the canonical correlations stabilise.
pub fn ir_mad(
    t1: &[&Raster<f64>],
    t2: &[&Raster<f64>],
    params: IrMadParams,
) -> Result<IrMadResult> {
    let (rows, cols, b) = validate_inputs(t1, t2)?;
    let valid_mask = build_valid_mask(t1, t2, rows, cols);
    let n_px = rows * cols;
    let n_valid = valid_mask.iter().filter(|v| **v).count();
    if n_valid < 2 {
        return Err(Error::Algorithm("IR-MAD: need ≥2 valid pixels".into()));
    }

    let mut weights: Vec<f64> = vec![1.0; n_px];
    let mut prev_rhos: Vec<f64> = vec![0.0; b];
    let mut a = vec![0.0; b * b];
    let mut bv = vec![0.0; b * b];
    let mut rhos = vec![0.0; b];
    let mut n_iter = 0;

    for it in 1..=params.max_iter {
        n_iter = it;
        let (na, nb, nr) = solve_cca(
            t1,
            t2,
            &valid_mask,
            Some(&weights),
            params.regularisation,
            b,
        )?;
        a = na;
        bv = nb;
        rhos = nr;

        // Compute MAD variates per valid pixel for the χ² weight.
        // MAD_i variance = 2·(1 − ρ_i); standardise by that.
        let mut new_weights = vec![0.0; n_px];
        for row in 0..rows {
            for col in 0..cols {
                let p = row * cols + col;
                if !valid_mask[p] {
                    continue;
                }
                let mut t2_stat = 0.0;
                for i in 0..b {
                    let mut u = 0.0;
                    let mut v = 0.0;
                    for k in 0..b {
                        u += a[i * b + k] * t1[k].get(row, col).unwrap();
                        v += bv[i * b + k] * t2[k].get(row, col).unwrap();
                    }
                    let mad_i = u - v;
                    let var = (2.0 * (1.0 - rhos[i])).max(1e-12);
                    t2_stat += mad_i * mad_i / var;
                }
                new_weights[p] = 1.0 - chi2_cdf(t2_stat, b);
            }
        }
        weights = new_weights;

        // Convergence: max change in any canonical correlation.
        let max_delta = rhos
            .iter()
            .zip(prev_rhos.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        prev_rhos = rhos.clone();
        if max_delta < params.tol {
            break;
        }
    }

    let mad_rasters = project_mad(t1, t2, &a, &bv, rows, cols, b, &valid_mask)?;
    // Pack weights as a raster.
    let mut w_data = vec![f64::NAN; n_px];
    for (i, &w) in weights.iter().enumerate() {
        if valid_mask[i] {
            w_data[i] = w;
        }
    }
    let mut w_ras = t1[0].with_same_meta::<f64>(rows, cols);
    w_ras.set_nodata(Some(f64::NAN));
    *w_ras.data_mut() =
        Array2::from_shape_vec((rows, cols), w_data).map_err(|e| Error::Other(e.to_string()))?;

    Ok(IrMadResult {
        mad: mad_rasters,
        correlations: rhos,
        weights: w_ras,
        n_iter,
    })
}

// ─── Shared CCA core ────────────────────────────────────────────────

fn validate_inputs(t1: &[&Raster<f64>], t2: &[&Raster<f64>]) -> Result<(usize, usize, usize)> {
    if t1.is_empty() {
        return Err(Error::Algorithm("MAD: need ≥1 band".into()));
    }
    if t1.len() != t2.len() {
        return Err(Error::Algorithm(format!(
            "MAD: t1 has {} bands, t2 has {}",
            t1.len(),
            t2.len()
        )));
    }
    let (rows, cols) = t1[0].shape();
    for r in t1.iter().chain(t2.iter()) {
        if r.shape() != (rows, cols) {
            return Err(Error::Algorithm("MAD: all bands must share shape".into()));
        }
    }
    Ok((rows, cols, t1.len()))
}

fn build_valid_mask(
    t1: &[&Raster<f64>],
    t2: &[&Raster<f64>],
    rows: usize,
    cols: usize,
) -> Vec<bool> {
    let n = rows * cols;
    let mut mask = vec![true; n];
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            for r in t1.iter().chain(t2.iter()) {
                if !unsafe { r.get_unchecked(row, col) }.is_finite() {
                    mask[p] = false;
                    break;
                }
            }
        }
    }
    mask
}

/// Solve CCA on `(t1, t2)` over the `valid_mask` pixels with
/// optional per-pixel weights and optional diagonal regularisation.
/// Returns `(a_flat, b_flat, rhos)` where `a_flat[i*B + k]` is the
/// k-th coefficient of the i-th canonical variate (and likewise for
/// `b_flat`). Both are sorted in descending order of `(1 − ρ)`.
fn solve_cca(
    t1: &[&Raster<f64>],
    t2: &[&Raster<f64>],
    valid_mask: &[bool],
    weights: Option<&[f64]>,
    regularisation: f64,
    b: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let (rows, cols) = t1[0].shape();

    // Weighted means.
    let mut mu_x = vec![0.0; b];
    let mut mu_y = vec![0.0; b];
    let mut w_sum = 0.0;
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            let w = weights.map(|ws| ws[p]).unwrap_or(1.0);
            for k in 0..b {
                mu_x[k] += w * t1[k].get(row, col).unwrap();
                mu_y[k] += w * t2[k].get(row, col).unwrap();
            }
            w_sum += w;
        }
    }
    if w_sum < 1e-9 {
        return Err(Error::Algorithm("MAD: total weight ≈ 0".into()));
    }
    for k in 0..b {
        mu_x[k] /= w_sum;
        mu_y[k] /= w_sum;
    }

    // Weighted covariance blocks: S_XX (B×B), S_YY (B×B), S_XY (B×B).
    let mut s_xx = vec![vec![0.0; b]; b];
    let mut s_yy = vec![vec![0.0; b]; b];
    let mut s_xy = vec![vec![0.0; b]; b];
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            let w = weights.map(|ws| ws[p]).unwrap_or(1.0);
            for i in 0..b {
                let xi = t1[i].get(row, col).unwrap() - mu_x[i];
                let yi = t2[i].get(row, col).unwrap() - mu_y[i];
                for j in i..b {
                    let xj = t1[j].get(row, col).unwrap() - mu_x[j];
                    let yj = t2[j].get(row, col).unwrap() - mu_y[j];
                    s_xx[i][j] += w * xi * xj;
                    s_yy[i][j] += w * yi * yj;
                }
                for j in 0..b {
                    let yj = t2[j].get(row, col).unwrap() - mu_y[j];
                    s_xy[i][j] += w * xi * yj;
                }
            }
        }
    }
    let inv_w = 1.0 / w_sum;
    for i in 0..b {
        for j in i..b {
            s_xx[i][j] *= inv_w;
            s_yy[i][j] *= inv_w;
            if j > i {
                s_xx[j][i] = s_xx[i][j];
                s_yy[j][i] = s_yy[i][j];
            }
        }
        for j in 0..b {
            s_xy[i][j] *= inv_w;
        }
    }

    // Diagonal regularisation (Nielsen 2007 eq. 23).
    if regularisation > 0.0 {
        for i in 0..b {
            let r = regularisation * s_xx[i][i].max(1e-12);
            s_xx[i][i] += r;
            let r = regularisation * s_yy[i][i].max(1e-12);
            s_yy[i][i] += r;
        }
    }

    // Cholesky: S_XX = L_X L_Xᵀ, S_YY = L_Y L_Yᵀ.
    let l_x = cholesky_lower(&s_xx, b)?;
    let l_y = cholesky_lower(&s_yy, b)?;

    // M = L_X⁻¹ · S_XY · L_Y⁻ᵀ. We compute it column by column:
    //   step 1: solve L_X · A = S_XY  for A = L_X⁻¹ S_XY (B×B)
    //   step 2: solve L_Y · M_row = A_col  ... done per row instead
    // Equivalent: M[i][j] = (L_X⁻¹ S_XY L_Y⁻ᵀ)[i][j]. Practical
    // path: A = solve_lower(L_X, S_XY) col-wise, then
    // Mᵀ = solve_lower(L_Y, Aᵀ) col-wise → transpose.
    let mut a_mat = vec![vec![0.0; b]; b]; // A = L_X⁻¹ S_XY
    for j in 0..b {
        let col: Vec<f64> = (0..b).map(|i| s_xy[i][j]).collect();
        let sol = solve_lower_tri(&l_x, &col, b);
        for i in 0..b {
            a_mat[i][j] = sol[i];
        }
    }
    let mut m_mat = vec![vec![0.0; b]; b]; // M = A · L_Y⁻ᵀ = (L_Y⁻¹ Aᵀ)ᵀ
    for i in 0..b {
        let row: Vec<f64> = (0..b).map(|j| a_mat[i][j]).collect();
        let sol = solve_lower_tri(&l_y, &row, b);
        for j in 0..b {
            m_mat[i][j] = sol[j];
        }
    }

    // Solve symmetric eigenproblem on MᵀM (B×B). Eigenvalues are
    // the squared canonical correlations; eigenvectors are V.
    let mut mtm = vec![vec![0.0; b]; b];
    for i in 0..b {
        for j in 0..b {
            let mut s = 0.0;
            for k in 0..b {
                s += m_mat[k][i] * m_mat[k][j];
            }
            mtm[i][j] = s;
        }
    }
    let (eigvals, eigvecs) = jacobi_eigen(&mtm, b)?;
    let mut order: Vec<usize> = (0..b).collect();
    // Sort descending by eigenvalue == ρ²: largest ρ → "no change",
    // but the convention used downstream (project_mad) reverses
    // this so MAD_1 = smallest ρ = max change. We'll sort ascending
    // here so the *output* ordering aligns with "MAD_1 = max change".
    order.sort_by(|&i, &j| eigvals[i].partial_cmp(&eigvals[j]).unwrap());
    let rhos: Vec<f64> = order
        .iter()
        .map(|&i| eigvals[i].max(0.0).sqrt().min(1.0))
        .collect();
    let v: Vec<Vec<f64>> = (0..b)
        .map(|k| (0..b).map(|j| eigvecs[k][order[j]]).collect())
        .collect();

    // U = M · V · diag(1/ρ); a_i = L_X⁻ᵀ u_i, b_i = L_Y⁻ᵀ v_i.
    let mut u = vec![vec![0.0; b]; b];
    for j in 0..b {
        for i in 0..b {
            let mut s = 0.0;
            for k in 0..b {
                s += m_mat[i][k] * v[k][j];
            }
            let inv = if rhos[j] > 1e-9 { 1.0 / rhos[j] } else { 0.0 };
            u[i][j] = s * inv;
        }
    }
    let mut a_flat = vec![0.0; b * b];
    let mut b_flat = vec![0.0; b * b];
    for j in 0..b {
        let u_col: Vec<f64> = (0..b).map(|i| u[i][j]).collect();
        let v_col: Vec<f64> = (0..b).map(|i| v[i][j]).collect();
        let a_col = solve_upper_tri_transposed(&l_x, &u_col, b);
        let bv_col = solve_upper_tri_transposed(&l_y, &v_col, b);
        for k in 0..b {
            a_flat[j * b + k] = a_col[k];
            b_flat[j * b + k] = bv_col[k];
        }
    }
    Ok((a_flat, b_flat, rhos))
}

fn project_mad(
    t1: &[&Raster<f64>],
    t2: &[&Raster<f64>],
    a: &[f64],
    bv: &[f64],
    rows: usize,
    cols: usize,
    b: usize,
    valid_mask: &[bool],
) -> Result<Vec<Raster<f64>>> {
    let n_px = rows * cols;
    let mut out: Vec<Vec<f64>> = (0..b).map(|_| vec![f64::NAN; n_px]).collect();
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid_mask[p] {
                continue;
            }
            for i in 0..b {
                let mut u = 0.0;
                let mut v = 0.0;
                for k in 0..b {
                    u += a[i * b + k] * t1[k].get(row, col).unwrap();
                    v += bv[i * b + k] * t2[k].get(row, col).unwrap();
                }
                out[i][p] = u - v;
            }
        }
    }
    let mut rasters = Vec::with_capacity(b);
    for band_data in out.into_iter() {
        let mut r = t1[0].with_same_meta::<f64>(rows, cols);
        r.set_nodata(Some(f64::NAN));
        *r.data_mut() = Array2::from_shape_vec((rows, cols), band_data)
            .map_err(|e| Error::Other(e.to_string()))?;
        rasters.push(r);
    }
    Ok(rasters)
}

// ─── Linear-algebra helpers (small B×B, no LAPACK) ──────────────────

/// Cholesky-Banachiewicz factorisation of a symmetric positive-
/// definite matrix `a` into lower-triangular `L` with
/// `a = L · Lᵀ`.
fn cholesky_lower(a: &[Vec<f64>], n: usize) -> Result<Vec<Vec<f64>>> {
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(Error::Algorithm(format!(
                        "Cholesky failed: non-positive diagonal at {} ({})",
                        i, s
                    )));
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    Ok(l)
}

/// Forward substitution: solve `L · x = b` for `x` where `L` is
/// lower triangular.
fn solve_lower_tri(l: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// Solve `Lᵀ · x = b` for `x` where `L` is lower triangular
/// (equivalently: upper-triangular back-substitution on `Lᵀ`).
fn solve_upper_tri_transposed(l: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

// ─── χ²(k) CDF (no rand / no special-function crate) ────────────────

/// Lower incomplete gamma function P(s, x) computed via the series
/// expansion for `x ≤ s + 1` and the continued-fraction expansion
/// otherwise. Sufficient precision for the IR-MAD weight (well below
/// 1e-6 on the regimes that matter).
fn lower_incomplete_gamma(s: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < s + 1.0 {
        // Series expansion.
        let mut term = 1.0 / s;
        let mut sum = term;
        let mut a = s;
        for _ in 0..200 {
            a += 1.0;
            term *= x / a;
            sum += term;
            if term.abs() < sum.abs() * 1e-15 {
                break;
            }
        }
        sum * (-x + s * x.ln() - gamma_ln(s)).exp()
    } else {
        // Continued fraction (Numerical Recipes).
        let mut b = x + 1.0 - s;
        let mut c = 1e30;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..200 {
            let an = -(i as f64) * (i as f64 - s);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-30 {
                d = 1e-30;
            }
            c = b + an / c;
            if c.abs() < 1e-30 {
                c = 1e-30;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let q = h * (-x + s * x.ln() - gamma_ln(s)).exp();
        1.0 - q
    }
}

/// Lanczos approximation to log Γ(x), accurate to ~1e-14.
fn gamma_ln(x: f64) -> f64 {
    let cof = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000000000190015;
    for c in &cof {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

/// χ²(k) cumulative distribution function.
fn chi2_cdf(x: f64, k: usize) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    lower_incomplete_gamma(k as f64 / 2.0, x / 2.0)
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn raster_from_grid(grid: &[&[f64]]) -> Raster<f64> {
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

    fn ramp(rows: usize, cols: usize, sx: f64, sy: f64, offset: f64) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, offset + sx * col as f64 + sy * row as f64)
                    .unwrap();
            }
        }
        r
    }

    #[test]
    fn cholesky_roundtrip_2x2() {
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let l = cholesky_lower(&a, 2).unwrap();
        // L · Lᵀ == a.
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0;
                for k in 0..2 {
                    s += l[i][k] * l[j][k];
                }
                assert!((s - a[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cholesky_rejects_non_positive_definite() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]]; // det < 0
        assert!(cholesky_lower(&a, 2).is_err());
    }

    #[test]
    fn solve_lower_tri_roundtrip() {
        let l = vec![vec![2.0, 0.0], vec![3.0, 4.0]];
        // L · x = b for b = [4, 14] → x = [2, 2]
        let x = solve_lower_tri(&l, &[4.0, 14.0], 2);
        assert!((x[0] - 2.0).abs() < 1e-12);
        assert!((x[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn chi2_cdf_known_values() {
        // χ²(2) at x=0 -> 0; at x=2 -> 1 - e⁻¹ ≈ 0.6321
        assert!((chi2_cdf(0.0, 2) - 0.0).abs() < 1e-12);
        let cdf2 = chi2_cdf(2.0, 2);
        assert!(
            (cdf2 - (1.0 - (-1.0_f64).exp())).abs() < 1e-9,
            "χ²(2) at x=2 ≈ 0.632; got {}",
            cdf2
        );
        // χ²(4) at x=∞ -> 1
        assert!((chi2_cdf(100.0, 4) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn mad_identity_returns_near_zero_variates() {
        // T1 == T2 → MAD variates ≈ 0 and canonical correlations ≈ 1.
        let r0 = ramp(10, 10, 1.0, 0.5, 0.0);
        let r1 = ramp(10, 10, 0.5, 1.0, 100.0);
        let result = mad(&[&r0, &r1], &[&r0, &r1]).unwrap();
        assert_eq!(result.mad.len(), 2);
        for rho in &result.correlations {
            assert!(
                (*rho - 1.0).abs() < 1e-9,
                "rho should be 1 for identical inputs, got {}",
                rho
            );
        }
        // MAD variates near zero (machine-precision after subtraction).
        for v in &result.mad {
            for row in 0..10 {
                for col in 0..10 {
                    let m = v.get(row, col).unwrap();
                    assert!(m.abs() < 1e-9, "MAD at ({},{}) = {}", row, col, m);
                }
            }
        }
    }

    #[test]
    fn mad_detects_planted_change() {
        // Two-band scene with linearly INDEPENDENT bands (one
        // varies with col, the other with row²) so S_XX is full
        // rank. T2 = T1 except in a 4x4 block where band 1 is
        // boosted by +50. MAD_1 must light up exactly that block.
        let mut r0_t1 = Raster::new(20, 20);
        r0_t1.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        let mut r1_t1 = Raster::new(20, 20);
        r1_t1.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                r0_t1.set(row, col, col as f64).unwrap();
                r1_t1.set(row, col, (row as f64).powi(2) * 0.1).unwrap();
            }
        }
        let mut r0_t2 = r0_t1.clone();
        let r1_t2 = r1_t1.clone();
        for row in 8..12 {
            for col in 8..12 {
                r0_t2
                    .set(row, col, r0_t2.get(row, col).unwrap() + 50.0)
                    .unwrap();
            }
        }
        let result = mad(&[&r0_t1, &r1_t1], &[&r0_t2, &r1_t2]).unwrap();
        // MAD_1 magnitude inside the change block must dominate the
        // outside-block magnitude.
        let mut inside = 0.0_f64;
        let mut outside = 0.0_f64;
        let mut n_in = 0;
        let mut n_out = 0;
        for row in 0..20 {
            for col in 0..20 {
                let m = result.mad[0].get(row, col).unwrap().abs();
                if (8..12).contains(&row) && (8..12).contains(&col) {
                    inside += m;
                    n_in += 1;
                } else {
                    outside += m;
                    n_out += 1;
                }
            }
        }
        let mean_in = inside / n_in as f64;
        let mean_out = outside / n_out as f64;
        assert!(
            mean_in > 3.0 * mean_out,
            "MAD_1 should light up the change block: in={}, out={}",
            mean_in,
            mean_out
        );
    }

    #[test]
    fn mad_canonical_correlations_in_unit_interval() {
        let r0_t1 = ramp(15, 15, 1.0, 0.5, 0.0);
        let r1_t1 = ramp(15, 15, 0.5, 1.0, 50.0);
        // T2 = T1 + small Gaussian-like noise (deterministic).
        let mut r0_t2 = r0_t1.clone();
        let mut r1_t2 = r1_t1.clone();
        for row in 0..15 {
            for col in 0..15 {
                let n = ((row * 7 + col * 13) % 11) as f64 - 5.0;
                r0_t2
                    .set(row, col, r0_t2.get(row, col).unwrap() + 0.3 * n)
                    .unwrap();
                r1_t2
                    .set(row, col, r1_t2.get(row, col).unwrap() + 0.2 * n)
                    .unwrap();
            }
        }
        let result = mad(&[&r0_t1, &r1_t1], &[&r0_t2, &r1_t2]).unwrap();
        for rho in &result.correlations {
            assert!(
                (0.0..=1.0001).contains(rho),
                "canonical correlation out of [0,1]: {}",
                rho
            );
        }
    }

    #[test]
    fn mad_nan_propagates_to_all_outputs() {
        let mut r0 = ramp(8, 8, 1.0, 0.5, 0.0);
        let r1 = ramp(8, 8, 0.5, 1.0, 50.0);
        r0.set(2, 2, f64::NAN).unwrap();
        let result = mad(&[&r0, &r1], &[&r0, &r1]).unwrap();
        for m in &result.mad {
            assert!(m.get(2, 2).unwrap().is_nan());
        }
    }

    #[test]
    fn ir_mad_converges_and_emphasises_change_block() {
        let mut r0_t1 = Raster::new(20, 20);
        r0_t1.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        let mut r1_t1 = Raster::new(20, 20);
        r1_t1.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                r0_t1.set(row, col, col as f64).unwrap();
                r1_t1.set(row, col, (row as f64).powi(2) * 0.1).unwrap();
            }
        }
        let mut r0_t2 = r0_t1.clone();
        let r1_t2 = r1_t1.clone();
        for row in 8..12 {
            for col in 8..12 {
                r0_t2
                    .set(row, col, r0_t2.get(row, col).unwrap() + 50.0)
                    .unwrap();
            }
        }
        let res = ir_mad(
            &[&r0_t1, &r1_t1],
            &[&r0_t2, &r1_t2],
            IrMadParams {
                max_iter: 15,
                tol: 1e-4,
                regularisation: 1e-6,
            },
        )
        .unwrap();
        assert!(res.n_iter <= 15);
        // Inside the change block, weight (= no-change probability)
        // should be low; outside, high.
        let mut w_in = 0.0;
        let mut w_out = 0.0;
        let mut n_in = 0;
        let mut n_out = 0;
        for row in 0..20 {
            for col in 0..20 {
                let w = res.weights.get(row, col).unwrap();
                if (8..12).contains(&row) && (8..12).contains(&col) {
                    w_in += w;
                    n_in += 1;
                } else {
                    w_out += w;
                    n_out += 1;
                }
            }
        }
        let mean_w_in = w_in / n_in as f64;
        let mean_w_out = w_out / n_out as f64;
        assert!(
            mean_w_out > mean_w_in,
            "IR-MAD weights: outside ({}) should exceed inside ({})",
            mean_w_out,
            mean_w_in
        );
    }

    #[test]
    fn rejects_mismatched_band_counts() {
        let r = ramp(5, 5, 1.0, 0.0, 0.0);
        assert!(mad(&[&r, &r], &[&r]).is_err());
    }

    #[test]
    fn rejects_empty_inputs() {
        assert!(mad(&[], &[]).is_err());
    }
}
