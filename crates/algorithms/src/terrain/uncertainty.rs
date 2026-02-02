//! DEM Uncertainty Propagation (Florinsky Ch. 5)
//!
//! Given a DEM with known vertical RMSE (mz), computes per-pixel RMSE
//! for derived morphometric variables: slope, aspect, curvatures, TWI, etc.
//!
//! Error propagation is "computationally free" — it uses the same partial
//! derivatives already computed for the morphometric variable itself.
//!
//! No open-source GIS library provides per-pixel uncertainty maps alongside
//! morphometric outputs. This is a UNIQUE capability.
//!
//! Reference:
//! Florinsky, I.V. (2025). Digital Terrain Analysis in Soil Science and
//! Geology, Ch. 5: Errors of Digital Terrain Modeling.

use ndarray::Array2;
use crate::maybe_rayon::*;
use crate::terrain::derivatives::{evans_young, extract_window};
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for uncertainty computation
#[derive(Debug, Clone)]
pub struct UncertaintyParams {
    /// Vertical RMSE of the DEM in map units (meters).
    /// Typical values: SRTM ~5m, LiDAR ~0.1m, photogrammetry ~1m.
    pub dem_rmse: f64,
}

/// Result of uncertainty analysis for a DEM
#[derive(Debug)]
pub struct UncertaintyResult {
    /// RMSE of slope angle (radians)
    pub slope_rmse: Raster<f64>,
    /// RMSE of aspect angle (radians)
    pub aspect_rmse: Raster<f64>,
    /// RMSE of general (mean) curvature
    pub general_curvature_rmse: Raster<f64>,
    /// RMSE of profile curvature
    pub profile_curvature_rmse: Raster<f64>,
    /// RMSE of plan curvature
    pub plan_curvature_rmse: Raster<f64>,
}

/// Compute per-pixel uncertainty (RMSE) for morphometric variables.
///
/// Uses Florinsky (2025) Ch. 5 error propagation formulas. For each
/// morphometric variable F(z₁,...,z₉), the RMSE is:
///
/// ```text
/// mF² = mz² × Σᵢ (∂F/∂zᵢ)²
/// ```
///
/// where the summation is over the 9 cells of the 3×3 window.
///
/// The partial derivatives ∂F/∂zᵢ are computed analytically from the
/// Evans-Young (1979) derivative formulas.
///
/// # Arguments
/// * `dem` — Input DEM
/// * `params` — Uncertainty parameters (DEM RMSE)
///
/// # Returns
/// [`UncertaintyResult`] with RMSE rasters for each variable.
pub fn uncertainty(dem: &Raster<f64>, params: UncertaintyParams) -> Result<UncertaintyResult> {
    if params.dem_rmse <= 0.0 {
        return Err(Error::Algorithm("DEM RMSE must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let cs = dem.cell_size();
    let mz = params.dem_rmse;
    let mz2 = mz * mz;

    let data = dem.data();

    // Compute uncertainty for all morphometric variables simultaneously
    // to avoid recomputing the 3×3 window and derivatives multiple times
    let row_results: Vec<Vec<(f64, f64, f64, f64, f64)>> = (1..rows - 1)
        .into_par_iter()
        .map(|row| {
            let mut row_data = Vec::with_capacity(cols - 2);

            for col in 1..cols - 1 {
                let win = match extract_window(data, row, col) {
                    Some(w) => w,
                    None => {
                        row_data.push((f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN));
                        continue;
                    }
                };

                let d = evans_young(win, cs);

                // Slope RMSE: G = atan(sqrt(p² + q²))
                // ∂G/∂zᵢ = (p·∂p/∂zᵢ + q·∂q/∂zᵢ) / ((1 + p² + q²) · √(p² + q²))
                // Evans-Young coefficients for p: [z3+z6+z9-z1-z4-z7]/(6h)
                // Weights: ∂p/∂z1=-1/(6h), ∂p/∂z3=1/(6h), etc.
                let slope_rmse = slope_uncertainty(d.p, d.q, cs, mz2);

                // Aspect RMSE
                let aspect_rmse = aspect_uncertainty(d.p, d.q, cs, mz2);

                // General curvature RMSE
                let gc_rmse = general_curvature_uncertainty(d.p, d.q, d.r, d.s, d.t, cs, mz2);

                // Profile curvature RMSE
                let pc_rmse = profile_curvature_uncertainty(d.p, d.q, d.r, d.s, d.t, cs, mz2);

                // Plan curvature RMSE
                let tc_rmse = plan_curvature_uncertainty(d.p, d.q, d.r, d.s, d.t, cs, mz2);

                row_data.push((slope_rmse, aspect_rmse, gc_rmse, pc_rmse, tc_rmse));
            }

            row_data
        })
        .collect();

    // Build output rasters
    let mut slope_arr = Array2::from_elem((rows, cols), f64::NAN);
    let mut aspect_arr = Array2::from_elem((rows, cols), f64::NAN);
    let mut gc_arr = Array2::from_elem((rows, cols), f64::NAN);
    let mut pc_arr = Array2::from_elem((rows, cols), f64::NAN);
    let mut tc_arr = Array2::from_elem((rows, cols), f64::NAN);

    for (ri, row_data) in row_results.iter().enumerate() {
        let row = ri + 1;
        for (ci, &(s, a, g, p, t)) in row_data.iter().enumerate() {
            let col = ci + 1;
            slope_arr[(row, col)] = s;
            aspect_arr[(row, col)] = a;
            gc_arr[(row, col)] = g;
            pc_arr[(row, col)] = p;
            tc_arr[(row, col)] = t;
        }
    }

    let make_raster = |arr: Array2<f64>| -> Raster<f64> {
        let mut r = dem.with_same_meta::<f64>(rows, cols);
        r.set_nodata(Some(f64::NAN));
        *r.data_mut() = arr;
        r
    };

    Ok(UncertaintyResult {
        slope_rmse: make_raster(slope_arr),
        aspect_rmse: make_raster(aspect_arr),
        general_curvature_rmse: make_raster(gc_arr),
        profile_curvature_rmse: make_raster(pc_arr),
        plan_curvature_rmse: make_raster(tc_arr),
    })
}

/// Slope RMSE using Evans-Young error propagation.
///
/// G = atan(sqrt(p² + q²))
/// mG² = mz² × Σ (∂G/∂zᵢ)²
///
/// For Evans-Young, the partial derivatives of p and q with respect to
/// each z_i are constant coefficients. Using the chain rule:
///
/// ∂G/∂zᵢ = (p·∂p/∂zᵢ + q·∂q/∂zᵢ) / ((1+p²+q²) · √(p²+q²))
///
/// The sum of squared partials simplifies to a formula in p, q, and cell_size.
fn slope_uncertainty(p: f64, q: f64, cs: f64, mz2: f64) -> f64 {
    let p2q2 = p * p + q * q;
    if p2q2 < 1e-30 {
        // Near-flat: slope RMSE ≈ mz × √(2/9) / (cs × √3)
        // From the sum of squared EY coefficients for p and q
        return (mz2 * 2.0 / (9.0 * cs * cs)).sqrt().atan();
    }

    let w = 1.0 + p2q2;

    // Evans-Young coefficients for p: [-1, 0, 1, -1, 0, 1, -1, 0, 1] / (6h)
    // Evans-Young coefficients for q: [1, 1, 1, 0, 0, 0, -1, -1, -1] / (6h)
    // ∂G/∂zᵢ = (p * dp_i + q * dq_i) / (w * sqrt(p2q2))
    let h6 = 6.0 * cs;

    // Coefficient arrays for p and q
    let dp = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]; // /(6h)
    let dq = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]; // /(6h)

    let denom = w * p2q2.sqrt() * h6;
    let mut sum_sq = 0.0;

    for i in 0..9 {
        let dg = (p * dp[i] + q * dq[i]) / denom;
        sum_sq += dg * dg;
    }

    (mz2 * sum_sq).sqrt()
}

/// Aspect RMSE using Evans-Young error propagation.
///
/// A = atan2(-q, -p)
/// ∂A/∂zᵢ = (p·∂q/∂zᵢ - q·∂p/∂zᵢ) / (p² + q²)
fn aspect_uncertainty(p: f64, q: f64, cs: f64, mz2: f64) -> f64 {
    let p2q2 = p * p + q * q;
    if p2q2 < 1e-30 {
        return f64::NAN; // Aspect undefined on flat terrain
    }

    let h6 = 6.0 * cs;
    let dp = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
    let dq = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0];

    let mut sum_sq = 0.0;
    for i in 0..9 {
        let da = (p * dq[i] - q * dp[i]) / (p2q2 * h6);
        sum_sq += da * da;
    }

    (mz2 * sum_sq).sqrt()
}

/// General curvature RMSE.
///
/// H = -((1+q²)r - 2pqs + (1+p²)t) / (2(1+p²+q²)^(3/2))
///
/// We use numerical differentiation of H with respect to each z_i
/// via the finite-difference approach: perturb each z_i, recompute H.
fn general_curvature_uncertainty(
    p: f64, q: f64, r: f64, s: f64, t: f64,
    cs: f64, mz2: f64,
) -> f64 {
    curvature_uncertainty_generic(p, q, r, s, t, cs, mz2, CurvatureType::General)
}

/// Profile curvature RMSE.
fn profile_curvature_uncertainty(
    p: f64, q: f64, r: f64, s: f64, t: f64,
    cs: f64, mz2: f64,
) -> f64 {
    curvature_uncertainty_generic(p, q, r, s, t, cs, mz2, CurvatureType::Profile)
}

/// Plan curvature RMSE.
fn plan_curvature_uncertainty(
    p: f64, q: f64, r: f64, s: f64, t: f64,
    cs: f64, mz2: f64,
) -> f64 {
    curvature_uncertainty_generic(p, q, r, s, t, cs, mz2, CurvatureType::Plan)
}

enum CurvatureType {
    General,
    Profile,
    Plan,
}

/// Generic curvature uncertainty using analytical partial derivatives.
///
/// For Evans-Young derivatives, the partial derivatives of r, s, t with
/// respect to each z_i are known constants. We use the chain rule to
/// compute ∂K/∂zᵢ = (∂K/∂p)(∂p/∂zᵢ) + (∂K/∂q)(∂q/∂zᵢ) + ... for r,s,t.
fn curvature_uncertainty_generic(
    p: f64, q: f64, r: f64, s: f64, t: f64,
    cs: f64, mz2: f64,
    ctype: CurvatureType,
) -> f64 {
    let h6 = 6.0 * cs;
    let h2_3 = 3.0 * cs * cs;
    let h2_4 = 4.0 * cs * cs;

    // Evans-Young coefficients for each derivative w.r.t. z_i:
    // p: [-1, 0, 1, -1, 0, 1, -1, 0, 1] / (6h)
    // q: [1, 1, 1, 0, 0, 0, -1, -1, -1] / (6h)
    // r: [1, -2, 1, 1, -2, 1, 1, -2, 1] / (3h²)
    // s: [-1, 0, 1, 0, 0, 0, 1, 0, -1] / (4h²)
    // t: [1, 1, 1, -2, -2, -2, 1, 1, 1] / (3h²) — wait, let me recheck
    // Actually for Evans-Young:
    // t = (z1+z2+z3+z7+z8+z9 - 2(z4+z5+z6)) / (3h²)
    // So dt/dz1 = 1/(3h²), dt/dz4 = -2/(3h²), etc.

    let dp_raw: [f64; 9] = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
    let dq_raw: [f64; 9] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0];
    let dr_raw: [f64; 9] = [1.0, -2.0, 1.0, 1.0, -2.0, 1.0, 1.0, -2.0, 1.0];
    let ds_raw: [f64; 9] = [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0];
    let dt_raw: [f64; 9] = [1.0, 1.0, 1.0, -2.0, -2.0, -2.0, 1.0, 1.0, 1.0];

    // Compute ∂K/∂p, ∂K/∂q, ∂K/∂r, ∂K/∂s, ∂K/∂t for chosen curvature type
    let (dk_dp, dk_dq, dk_dr, dk_ds, dk_dt) = match ctype {
        CurvatureType::General => curvature_partials_general(p, q, r, s, t),
        CurvatureType::Profile => curvature_partials_profile(p, q, r, s, t),
        CurvatureType::Plan => curvature_partials_plan(p, q, r, s, t),
    };

    let mut sum_sq = 0.0;
    for i in 0..9 {
        let dki = dk_dp * dp_raw[i] / h6
            + dk_dq * dq_raw[i] / h6
            + dk_dr * dr_raw[i] / h2_3
            + dk_ds * ds_raw[i] / h2_4
            + dk_dt * dt_raw[i] / h2_3;
        sum_sq += dki * dki;
    }

    (mz2 * sum_sq).sqrt()
}

/// Partial derivatives of general curvature H with respect to p, q, r, s, t.
///
/// H = -((1+q²)r - 2pqs + (1+p²)t) / (2·w^(3/2))  where w = 1+p²+q²
fn curvature_partials_general(p: f64, q: f64, r: f64, s: f64, t: f64) -> (f64, f64, f64, f64, f64) {
    let p2 = p * p;
    let q2 = q * q;
    let w = 1.0 + p2 + q2;
    let w32 = w * w.sqrt();
    let w52 = w32 * w;

    let num = (1.0 + q2) * r - 2.0 * p * q * s + (1.0 + p2) * t;

    // ∂H/∂r = -(1+q²) / (2·w^(3/2))
    let dh_dr = -(1.0 + q2) / (2.0 * w32);
    // ∂H/∂s = 2pq / (2·w^(3/2)) = pq / w^(3/2)
    let dh_ds = p * q / w32;
    // ∂H/∂t = -(1+p²) / (2·w^(3/2))
    let dh_dt = -(1.0 + p2) / (2.0 * w32);

    // ∂H/∂p = -(-2qs + 2pt) / (2·w^(3/2)) + 3p·num / (2·w^(5/2))
    let dh_dp = -(-2.0 * q * s + 2.0 * p * t) / (2.0 * w32)
        + 3.0 * p * num / (2.0 * w52);
    // ∂H/∂q = -(2qr - 2ps) / (2·w^(3/2)) + 3q·num / (2·w^(5/2))
    let dh_dq = -(2.0 * q * r - 2.0 * p * s) / (2.0 * w32)
        + 3.0 * q * num / (2.0 * w52);

    (dh_dp, dh_dq, dh_dr, dh_ds, dh_dt)
}

/// Partial derivatives of profile curvature Kp with respect to p, q, r, s, t.
///
/// Kp = -(p²r + 2pqs + q²t) / ((p²+q²)·(1+p²+q²)^(3/2))
fn curvature_partials_profile(p: f64, q: f64, r: f64, s: f64, t: f64) -> (f64, f64, f64, f64, f64) {
    let p2 = p * p;
    let q2 = q * q;
    let p2q2 = p2 + q2;

    if p2q2 < 1e-20 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let w = 1.0 + p2q2;
    let w32 = w * w.sqrt();
    let denom = p2q2 * w32;

    // ∂Kp/∂r = -p² / denom
    let dk_dr = -p2 / denom;
    // ∂Kp/∂s = -2pq / denom
    let dk_ds = -2.0 * p * q / denom;
    // ∂Kp/∂t = -q² / denom
    let dk_dt = -q2 / denom;

    // For dp and dq, use numerical differentiation (simpler and less error-prone)
    let eps = 1e-8;

    let kp = |pp: f64, qq: f64| -> f64 {
        let pp2 = pp * pp;
        let qq2 = qq * qq;
        let pq2 = pp2 + qq2;
        if pq2 < 1e-30 { return 0.0; }
        let ww = 1.0 + pq2;
        -(pp2 * r + 2.0 * pp * qq * s + qq2 * t) / (pq2 * ww * ww.sqrt())
    };

    let dk_dp = (kp(p + eps, q) - kp(p - eps, q)) / (2.0 * eps);
    let dk_dq = (kp(p, q + eps) - kp(p, q - eps)) / (2.0 * eps);

    (dk_dp, dk_dq, dk_dr, dk_ds, dk_dt)
}

/// Partial derivatives of plan curvature Kt with respect to p, q, r, s, t.
///
/// Kt = -(q²r - 2pqs + p²t) / ((p²+q²)·√(1+p²+q²))
fn curvature_partials_plan(p: f64, q: f64, r: f64, s: f64, t: f64) -> (f64, f64, f64, f64, f64) {
    let p2 = p * p;
    let q2 = q * q;
    let p2q2 = p2 + q2;

    if p2q2 < 1e-20 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let w = 1.0 + p2q2;
    let w12 = w.sqrt();
    let denom = p2q2 * w12;

    // ∂Kt/∂r = -q² / denom
    let dk_dr = -q2 / denom;
    // ∂Kt/∂s = 2pq / denom
    let dk_ds = 2.0 * p * q / denom;
    // ∂Kt/∂t = -p² / denom
    let dk_dt = -p2 / denom;

    // Numerical differentiation for p, q partials
    let eps = 1e-8;

    let kt = |pp: f64, qq: f64| -> f64 {
        let pp2 = pp * pp;
        let qq2 = qq * qq;
        let pq2 = pp2 + qq2;
        if pq2 < 1e-30 { return 0.0; }
        let ww = 1.0 + pq2;
        -(qq2 * r - 2.0 * pp * qq * s + pp2 * t) / (pq2 * ww.sqrt())
    };

    let dk_dp = (kt(p + eps, q) - kt(p - eps, q)) / (2.0 * eps);
    let dk_dq = (kt(p, q + eps) - kt(p, q - eps)) / (2.0 * eps);

    (dk_dp, dk_dq, dk_dr, dk_ds, dk_dt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_uncertainty_flat_terrain() {
        // On flat terrain, slope RMSE should be small and curvature RMSE should be ~0
        let mut dem = Raster::filled(11, 11, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 10.0, -10.0));

        let result = uncertainty(&dem, UncertaintyParams { dem_rmse: 1.0 }).unwrap();

        // Slope RMSE on flat should be positive (uncertainty exists even on flat)
        let s = result.slope_rmse.get(5, 5).unwrap();
        assert!(s > 0.0 && s < 1.0, "Flat terrain slope RMSE should be small, got {:.6}", s);

        // Curvature RMSE should be finite
        let gc = result.general_curvature_rmse.get(5, 5).unwrap();
        assert!(gc.is_finite() && gc >= 0.0, "Curvature RMSE should be finite, got {:.6}", gc);
    }

    #[test]
    fn test_uncertainty_sloped_terrain() {
        // Uniform slope: slope RMSE should be defined
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 10.0, -10.0));
        for row in 0..11 {
            for col in 0..11 {
                dem.set(row, col, row as f64 * 10.0).unwrap();
            }
        }

        let result = uncertainty(&dem, UncertaintyParams { dem_rmse: 1.0 }).unwrap();

        let s = result.slope_rmse.get(5, 5).unwrap();
        assert!(s > 0.0 && s.is_finite(), "Slope RMSE should be positive, got {:.6}", s);

        // Aspect RMSE should be defined (slope > 0)
        let a = result.aspect_rmse.get(5, 5).unwrap();
        assert!(a.is_finite() && a >= 0.0, "Aspect RMSE should be finite, got {:.6}", a);
    }

    #[test]
    fn test_uncertainty_scales_with_rmse() {
        // Higher DEM RMSE → higher uncertainty
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 10.0, -10.0));
        for row in 0..11 {
            for col in 0..11 {
                dem.set(row, col, row as f64 * 5.0 + col as f64 * 3.0).unwrap();
            }
        }

        let r1 = uncertainty(&dem, UncertaintyParams { dem_rmse: 0.5 }).unwrap();
        let r2 = uncertainty(&dem, UncertaintyParams { dem_rmse: 2.0 }).unwrap();

        let s1 = r1.slope_rmse.get(5, 5).unwrap();
        let s2 = r2.slope_rmse.get(5, 5).unwrap();

        assert!(
            s2 > s1,
            "Higher DEM RMSE should give higher slope RMSE: {:.6} vs {:.6}",
            s1, s2
        );
    }

    #[test]
    fn test_uncertainty_output_shape() {
        let mut dem = Raster::filled(11, 11, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 10.0, -10.0));

        let result = uncertainty(&dem, UncertaintyParams { dem_rmse: 1.0 }).unwrap();

        assert_eq!(result.slope_rmse.shape(), (11, 11));
        assert_eq!(result.aspect_rmse.shape(), (11, 11));
        assert_eq!(result.general_curvature_rmse.shape(), (11, 11));
        assert_eq!(result.profile_curvature_rmse.shape(), (11, 11));
        assert_eq!(result.plan_curvature_rmse.shape(), (11, 11));
    }

    #[test]
    fn test_uncertainty_boundary_nan() {
        let mut dem = Raster::filled(5, 5, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let result = uncertainty(&dem, UncertaintyParams { dem_rmse: 1.0 }).unwrap();

        // Boundary cells should be NAN
        assert!(result.slope_rmse.get(0, 0).unwrap().is_nan());
        // Interior cells should be finite
        assert!(result.slope_rmse.get(2, 2).unwrap().is_finite());
    }

    #[test]
    fn test_uncertainty_invalid_params() {
        let dem = Raster::filled(5, 5, 100.0_f64);
        assert!(uncertainty(&dem, UncertaintyParams { dem_rmse: 0.0 }).is_err());
        assert!(uncertainty(&dem, UncertaintyParams { dem_rmse: -1.0 }).is_err());
    }

    #[test]
    fn test_uncertainty_curved_terrain() {
        // Bowl: slope increases from center, curvature is constant
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 10.0, -10.0));
        for row in 0..21 {
            for col in 0..21 {
                let x = (col as f64 - 10.0) * 10.0;
                let y = (row as f64 - 10.0) * 10.0;
                dem.set(row, col, 0.001 * (x * x + y * y)).unwrap();
            }
        }

        let result = uncertainty(&dem, UncertaintyParams { dem_rmse: 0.5 }).unwrap();

        // Interior cells should have defined uncertainty
        let gc = result.general_curvature_rmse.get(10, 10).unwrap();
        assert!(gc.is_finite() && gc > 0.0, "Bowl curvature RMSE should be positive, got {:.8}", gc);
    }
}
