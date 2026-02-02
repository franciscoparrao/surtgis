//! Complete system of 12 Florinsky curvatures
//!
//! Implements the full curvature system from Florinsky (2025) Chapter 2:
//!
//! **3 independent curvatures**: H (mean), E (difference), M (unsphericity)
//! **6 simple curvatures**: kmin, kmax, kh, kv, khe, kve
//! **3 total curvatures**: K (Gaussian), Ka (accumulation), Kr (ring)
//!
//! Plus rotor (rot) and Laplacian (∇²z).
//!
//! All curvatures are computed from partial derivatives (p, q, r, s, t)
//! using the **full formulas** (with √(1+p²+q²) denominators), not the
//! simplified Z&T approximations.
//!
//! Key relationships (Florinsky §2.4):
//! ```text
//!   kmin = H - M          kmax = H + M
//!   kh   = H - E          kv   = H + E
//!   khe  = M - E          kve  = M + E
//!   K    = H² - M²        Ka   = H² - E²        Kr = M² - E²
//! ```
//!
//! Reference: Florinsky, I.V. (2025) "Digital Terrain Analysis" 3rd ed.,
//! Chapter 2: Morphometric Variables

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Available curvature types in the Florinsky system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvancedCurvatureType {
    /// Mean curvature H — average of principal curvatures
    MeanH,
    /// Gaussian curvature K — product of principal curvatures
    GaussianK,
    /// Unsphericity M = √(H²−K) — deviation from sphere
    UnsphericitytM,
    /// Difference curvature E = (kv−kh)/2 — flow asymmetry
    DifferenceE,
    /// Minimal curvature kmin = H − M
    MinimalKmin,
    /// Maximal curvature kmax = H + M
    MaximalKmax,
    /// Horizontal (plan) curvature kh — contour-line curvature
    HorizontalKh,
    /// Vertical (profile) curvature kv — flow-line curvature
    VerticalKv,
    /// Horizontal excess khe = M − E
    HorizontalExcessKhe,
    /// Vertical excess kve = M + E
    VerticalExcessKve,
    /// Accumulation curvature Ka = kh·kv
    AccumulationKa,
    /// Ring curvature Kr = M² − E²
    RingKr,
    /// Rotor — flow-line torsion
    Rotor,
    /// Laplacian ∇²z = r + t
    Laplacian,
}

/// Compute a specific curvature from the Florinsky 12-curvature system
///
/// Uses Evans-Young partial derivatives with full (non-simplified)
/// curvature formulas including the √(1+p²+q²) denominators.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `curv_type` - Which curvature to compute
///
/// # Returns
/// Raster with curvature values
pub fn advanced_curvatures(
    dem: &Raster<f64>,
    curv_type: AdvancedCurvatureType,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cs = dem.cell_size();
    let nodata = dem.nodata();
    let cs2 = cs * cs;
    let cs6 = 6.0 * cs;
    let cs2_3 = 3.0 * cs2;
    let cs2_4 = 4.0 * cs2;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let z5 = unsafe { dem.get_unchecked(row, col) };
                if z5.is_nan() || nodata.map_or(false, |nd| (z5 - nd).abs() < f64::EPSILON) {
                    continue;
                }
                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    continue;
                }

                let z1 = unsafe { dem.get_unchecked(row - 1, col - 1) };
                let z2 = unsafe { dem.get_unchecked(row - 1, col) };
                let z3 = unsafe { dem.get_unchecked(row - 1, col + 1) };
                let z4 = unsafe { dem.get_unchecked(row, col - 1) };
                let z6 = unsafe { dem.get_unchecked(row, col + 1) };
                let z7 = unsafe { dem.get_unchecked(row + 1, col - 1) };
                let z8 = unsafe { dem.get_unchecked(row + 1, col) };
                let z9 = unsafe { dem.get_unchecked(row + 1, col + 1) };

                if [z1, z2, z3, z4, z6, z7, z8, z9].iter().any(|v| v.is_nan()) {
                    continue;
                }

                // Evans-Young partial derivatives (weighted LS on 3×3)
                let p = (z3 + z6 + z9 - z1 - z4 - z7) / cs6;
                let q = (z1 + z2 + z3 - z7 - z8 - z9) / cs6;
                let r = (z1 + z3 + z4 + z6 + z7 + z9 - 2.0 * (z2 + z5 + z8)) / cs2_3;
                let s = (z3 + z7 - z1 - z9) / cs2_4;
                let t = (z1 + z2 + z3 + z7 + z8 + z9 - 2.0 * (z4 + z5 + z6)) / cs2_3;

                let p2 = p * p;
                let q2 = q * q;
                let p2q2 = p2 + q2;
                let w = 1.0 + p2q2; // 1 + p² + q²

                row_data[col] = compute_curvature(curv_type, p, q, r, s, t, p2, q2, p2q2, w);
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

/// Compute all 14 curvatures at once (single pass, more efficient)
///
/// Returns a struct with all curvature rasters.
pub fn all_curvatures(dem: &Raster<f64>) -> Result<AllCurvatures> {
    let (rows, cols) = dem.shape();
    let cs = dem.cell_size();
    let nodata = dem.nodata();
    let cs2 = cs * cs;
    let cs6 = 6.0 * cs;
    let cs2_3 = 3.0 * cs2;
    let cs2_4 = 4.0 * cs2;

    // Pre-allocate all 14 output vectors
    let all: Vec<[f64; 14]> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![[f64::NAN; 14]; cols];

            for col in 0..cols {
                let z5 = unsafe { dem.get_unchecked(row, col) };
                if z5.is_nan() || nodata.map_or(false, |nd| (z5 - nd).abs() < f64::EPSILON) {
                    continue;
                }
                if row == 0 || row == rows - 1 || col == 0 || col == cols - 1 {
                    continue;
                }

                let z1 = unsafe { dem.get_unchecked(row - 1, col - 1) };
                let z2 = unsafe { dem.get_unchecked(row - 1, col) };
                let z3 = unsafe { dem.get_unchecked(row - 1, col + 1) };
                let z4 = unsafe { dem.get_unchecked(row, col - 1) };
                let z6 = unsafe { dem.get_unchecked(row, col + 1) };
                let z7 = unsafe { dem.get_unchecked(row + 1, col - 1) };
                let z8 = unsafe { dem.get_unchecked(row + 1, col) };
                let z9 = unsafe { dem.get_unchecked(row + 1, col + 1) };

                if [z1, z2, z3, z4, z6, z7, z8, z9].iter().any(|v| v.is_nan()) {
                    continue;
                }

                let p = (z3 + z6 + z9 - z1 - z4 - z7) / cs6;
                let q = (z1 + z2 + z3 - z7 - z8 - z9) / cs6;
                let r = (z1 + z3 + z4 + z6 + z7 + z9 - 2.0 * (z2 + z5 + z8)) / cs2_3;
                let s = (z3 + z7 - z1 - z9) / cs2_4;
                let t = (z1 + z2 + z3 + z7 + z8 + z9 - 2.0 * (z4 + z5 + z6)) / cs2_3;

                let p2 = p * p;
                let q2 = q * q;
                let p2q2 = p2 + q2;
                let w = 1.0 + p2q2;
                let w_sqrt = w.sqrt();
                let w_32 = w * w_sqrt;

                // Full Florinsky formulas
                let mean_h = -((1.0 + q2) * r - 2.0 * p * q * s + (1.0 + p2) * t) / (2.0 * w_32);
                let gauss_k = (r * t - s * s) / (w * w);

                let (kh, kv) = if p2q2 < 1e-20 {
                    (0.0, 0.0)
                } else {
                    let kh_val = -(q2 * r - 2.0 * p * q * s + p2 * t) / (p2q2 * w_sqrt);
                    let kv_val = -(p2 * r + 2.0 * p * q * s + q2 * t) / (p2q2 * w_32);
                    (kh_val, kv_val)
                };

                let disc = mean_h * mean_h - gauss_k;
                let m_val = if disc > 0.0 { disc.sqrt() } else { 0.0 };
                let e_val = (kv - kh) / 2.0;

                let kmin = mean_h - m_val;
                let kmax = mean_h + m_val;
                let khe = m_val - e_val;
                let kve = m_val + e_val;
                let ka = kh * kv;
                let kr = m_val * m_val - e_val * e_val;

                let rot = if p2q2 < 1e-20 {
                    0.0
                } else {
                    ((p2 - q2) * s - p * q * (r - t)) / p2q2.powf(1.5)
                };

                let laplacian = r + t;

                row_data[col] = [
                    mean_h, gauss_k, m_val, e_val,
                    kmin, kmax, kh, kv,
                    khe, kve, ka, kr,
                    rot, laplacian,
                ];
            }

            row_data
        })
        .collect();

    // Split into individual rasters
    let mut rasters: Vec<Vec<f64>> = (0..14).map(|_| Vec::with_capacity(rows * cols)).collect();
    for pixel in &all {
        for (i, r) in rasters.iter_mut().enumerate() {
            r.push(pixel[i]);
        }
    }

    let make_raster = |data: Vec<f64>| -> Result<Raster<f64>> {
        let mut output = dem.with_same_meta::<f64>(rows, cols);
        output.set_nodata(Some(f64::NAN));
        *output.data_mut() = Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(output)
    };

    Ok(AllCurvatures {
        mean_h: make_raster(rasters.remove(0))?,
        gaussian_k: make_raster(rasters.remove(0))?,
        unsphericity_m: make_raster(rasters.remove(0))?,
        difference_e: make_raster(rasters.remove(0))?,
        minimal_kmin: make_raster(rasters.remove(0))?,
        maximal_kmax: make_raster(rasters.remove(0))?,
        horizontal_kh: make_raster(rasters.remove(0))?,
        vertical_kv: make_raster(rasters.remove(0))?,
        horizontal_excess_khe: make_raster(rasters.remove(0))?,
        vertical_excess_kve: make_raster(rasters.remove(0))?,
        accumulation_ka: make_raster(rasters.remove(0))?,
        ring_kr: make_raster(rasters.remove(0))?,
        rotor: make_raster(rasters.remove(0))?,
        laplacian: make_raster(rasters.remove(0))?,
    })
}

/// All 14 curvature rasters from a single computation pass
#[derive(Debug)]
pub struct AllCurvatures {
    /// Mean curvature H
    pub mean_h: Raster<f64>,
    /// Gaussian curvature K
    pub gaussian_k: Raster<f64>,
    /// Unsphericity M = √(H²−K)
    pub unsphericity_m: Raster<f64>,
    /// Difference curvature E = (kv−kh)/2
    pub difference_e: Raster<f64>,
    /// Minimal principal curvature kmin
    pub minimal_kmin: Raster<f64>,
    /// Maximal principal curvature kmax
    pub maximal_kmax: Raster<f64>,
    /// Horizontal (plan) curvature kh
    pub horizontal_kh: Raster<f64>,
    /// Vertical (profile) curvature kv
    pub vertical_kv: Raster<f64>,
    /// Horizontal excess curvature khe
    pub horizontal_excess_khe: Raster<f64>,
    /// Vertical excess curvature kve
    pub vertical_excess_kve: Raster<f64>,
    /// Accumulation curvature Ka = kh·kv
    pub accumulation_ka: Raster<f64>,
    /// Ring curvature Kr = M²−E²
    pub ring_kr: Raster<f64>,
    /// Rotor (flow-line torsion)
    pub rotor: Raster<f64>,
    /// Laplacian ∇²z = r + t
    pub laplacian: Raster<f64>,
}

// Private helper: compute one curvature value from pre-computed derivatives
#[inline]
fn compute_curvature(
    curv_type: AdvancedCurvatureType,
    p: f64, q: f64, r: f64, s: f64, t: f64,
    p2: f64, q2: f64, p2q2: f64, w: f64,
) -> f64 {
    let w_sqrt = w.sqrt();
    let w_32 = w * w_sqrt;

    match curv_type {
        AdvancedCurvatureType::MeanH => {
            -((1.0 + q2) * r - 2.0 * p * q * s + (1.0 + p2) * t) / (2.0 * w_32)
        }
        AdvancedCurvatureType::GaussianK => {
            (r * t - s * s) / (w * w)
        }
        AdvancedCurvatureType::Laplacian => {
            r + t
        }
        AdvancedCurvatureType::Rotor => {
            if p2q2 < 1e-20 { return 0.0; }
            ((p2 - q2) * s - p * q * (r - t)) / p2q2.powf(1.5)
        }
        _ => {
            // Curvatures that need kh, kv, H, K, M, E
            let mean_h = -((1.0 + q2) * r - 2.0 * p * q * s + (1.0 + p2) * t) / (2.0 * w_32);
            let gauss_k = (r * t - s * s) / (w * w);

            let (kh, kv) = if p2q2 < 1e-20 {
                (0.0, 0.0)
            } else {
                let kh_val = -(q2 * r - 2.0 * p * q * s + p2 * t) / (p2q2 * w_sqrt);
                let kv_val = -(p2 * r + 2.0 * p * q * s + q2 * t) / (p2q2 * w_32);
                (kh_val, kv_val)
            };

            let disc = mean_h * mean_h - gauss_k;
            let m_val = if disc > 0.0 { disc.sqrt() } else { 0.0 };
            let e_val = (kv - kh) / 2.0;

            match curv_type {
                AdvancedCurvatureType::HorizontalKh => kh,
                AdvancedCurvatureType::VerticalKv => kv,
                AdvancedCurvatureType::UnsphericitytM => m_val,
                AdvancedCurvatureType::DifferenceE => e_val,
                AdvancedCurvatureType::MinimalKmin => mean_h - m_val,
                AdvancedCurvatureType::MaximalKmax => mean_h + m_val,
                AdvancedCurvatureType::HorizontalExcessKhe => m_val - e_val,
                AdvancedCurvatureType::VerticalExcessKve => m_val + e_val,
                AdvancedCurvatureType::AccumulationKa => kh * kv,
                AdvancedCurvatureType::RingKr => m_val * m_val - e_val * e_val,
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    /// Parabolic bowl z = x² + y²
    fn bowl() -> Raster<f64> {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let x = col as f64 - 10.0;
                let y = row as f64 - 10.0;
                dem.set(row, col, x * x + y * y).unwrap();
            }
        }
        dem
    }

    /// Saddle z = x² - y²
    fn saddle() -> Raster<f64> {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let x = col as f64 - 10.0;
                let y = row as f64 - 10.0;
                dem.set(row, col, x * x - y * y).unwrap();
            }
        }
        dem
    }

    /// Tilted plane z = row + col
    fn plane() -> Raster<f64> {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_mean_curvature_bowl() {
        let dem = bowl();
        let result = advanced_curvatures(&dem, AdvancedCurvatureType::MeanH).unwrap();
        let val = result.get(10, 10).unwrap();
        // At center of bowl (flat spot): H should be close to -2
        // z = x²+y², Hessian = diag(2,2), p=q=0 at center → H = -(2+2)/2 = -2
        assert!(
            (val - (-2.0)).abs() < 0.1,
            "Expected H ≈ -2 at bowl center, got {}",
            val
        );
    }

    #[test]
    fn test_gaussian_curvature_bowl() {
        let dem = bowl();
        let result = advanced_curvatures(&dem, AdvancedCurvatureType::GaussianK).unwrap();
        let val = result.get(10, 10).unwrap();
        // K = r*t - s² / (1+p²+q²)². At center: K = 2*2/1 = 4
        assert!(
            (val - 4.0).abs() < 0.5,
            "Expected K ≈ 4 at bowl center, got {}",
            val
        );
    }

    #[test]
    fn test_gaussian_curvature_saddle() {
        let dem = saddle();
        let result = advanced_curvatures(&dem, AdvancedCurvatureType::GaussianK).unwrap();
        let val = result.get(10, 10).unwrap();
        // Saddle: r=2, t=-2, s=0 → K = 2*(-2)-0 = -4
        assert!(
            val < 0.0,
            "Expected K < 0 for saddle, got {}",
            val
        );
    }

    #[test]
    fn test_unsphericity_bowl() {
        let dem = bowl();
        let result = advanced_curvatures(&dem, AdvancedCurvatureType::UnsphericitytM).unwrap();
        let val = result.get(10, 10).unwrap();
        // Spherical surface: kmin=kmax → M=0
        assert!(
            val.abs() < 0.1,
            "Expected M ≈ 0 for bowl (spherical), got {}",
            val
        );
    }

    #[test]
    fn test_unsphericity_saddle() {
        let dem = saddle();
        let result = advanced_curvatures(&dem, AdvancedCurvatureType::UnsphericitytM).unwrap();
        let val = result.get(10, 10).unwrap();
        // Saddle: very non-spherical → M > 0
        assert!(
            val > 0.0,
            "Expected M > 0 for saddle, got {}",
            val
        );
    }

    #[test]
    fn test_laplacian_bowl() {
        let dem = bowl();
        let result = advanced_curvatures(&dem, AdvancedCurvatureType::Laplacian).unwrap();
        let val = result.get(10, 10).unwrap();
        // z = x²+y² → ∇²z = 2+2 = 4
        assert!(
            (val - 4.0).abs() < 0.5,
            "Expected Laplacian ≈ 4, got {}",
            val
        );
    }

    #[test]
    fn test_rotor_plane() {
        let dem = plane();
        let result = advanced_curvatures(&dem, AdvancedCurvatureType::Rotor).unwrap();
        let val = result.get(5, 5).unwrap();
        // Plane: zero rotor
        assert!(
            val.abs() < 1e-10,
            "Expected rotor ≈ 0 for plane, got {}",
            val
        );
    }

    #[test]
    fn test_all_curvatures_returns_all() {
        let dem = bowl();
        let all = all_curvatures(&dem).unwrap();
        // Check values at center
        let h = all.mean_h.get(10, 10).unwrap();
        let k = all.gaussian_k.get(10, 10).unwrap();
        let m = all.unsphericity_m.get(10, 10).unwrap();
        assert!(!h.is_nan());
        assert!(!k.is_nan());
        assert!(!m.is_nan());
        // Ka = K + Kr (Florinsky Theorem 2.7) at an off-center point
        // where p,q ≠ 0 (center has p=q=0 → kh,kv indeterminate)
        let ka = all.accumulation_ka.get(8, 8).unwrap();
        let k_off = all.gaussian_k.get(8, 8).unwrap();
        let kr = all.ring_kr.get(8, 8).unwrap();
        if !ka.is_nan() && !k_off.is_nan() && !kr.is_nan() {
            assert!(
                (ka - (k_off + kr)).abs() < 0.1,
                "Ka ({}) should equal K ({}) + Kr ({}) = {}",
                ka, k_off, kr, k_off + kr
            );
        }
    }

    #[test]
    fn test_accumulation_curvature_theorem() {
        // Ka = K + Kr must hold everywhere
        let dem = saddle();
        let all = all_curvatures(&dem).unwrap();
        for row in 2..19 {
            for col in 2..19 {
                let ka = all.accumulation_ka.get(row, col).unwrap();
                let k = all.gaussian_k.get(row, col).unwrap();
                let kr = all.ring_kr.get(row, col).unwrap();
                if !ka.is_nan() && !k.is_nan() && !kr.is_nan() {
                    assert!(
                        (ka - (k + kr)).abs() < 0.01,
                        "Ka ≠ K + Kr at ({},{}): {} vs {}",
                        row, col, ka, k + kr
                    );
                }
            }
        }
    }

    #[test]
    fn test_kmin_kmax_from_h_m() {
        let dem = saddle();
        let all = all_curvatures(&dem).unwrap();
        let h = all.mean_h.get(10, 10).unwrap();
        let m = all.unsphericity_m.get(10, 10).unwrap();
        let kmin = all.minimal_kmin.get(10, 10).unwrap();
        let kmax = all.maximal_kmax.get(10, 10).unwrap();
        assert!(
            (kmin - (h - m)).abs() < 1e-10,
            "kmin should equal H-M"
        );
        assert!(
            (kmax - (h + m)).abs() < 1e-10,
            "kmax should equal H+M"
        );
    }

    #[test]
    fn test_excess_curvatures_non_negative() {
        let dem = saddle();
        let all = all_curvatures(&dem).unwrap();
        for row in 1..20 {
            for col in 1..20 {
                let khe = all.horizontal_excess_khe.get(row, col).unwrap();
                let kve = all.vertical_excess_kve.get(row, col).unwrap();
                if !khe.is_nan() {
                    assert!(khe >= -1e-10, "khe should be ≥ 0, got {} at ({},{})", khe, row, col);
                }
                if !kve.is_nan() {
                    assert!(kve >= -1e-10, "kve should be ≥ 0, got {} at ({},{})", kve, row, col);
                }
            }
        }
    }
}
