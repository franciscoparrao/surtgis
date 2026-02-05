//! Surface curvature from DEMs
//!
//! Calculates profile, plan and general (mean) curvature using second-order
//! partial derivatives estimated from a 3×3 neighborhood.
//!
//! ```text
//! a b c      z1 z2 z3
//! d e f  ->  z4 z5 z6
//! g h i      z7 z8 z9
//! ```
//!
//! Supports two derivative estimation methods:
//!
//! - **Evans-Young** (default): Weighted least-squares on all 9 cells.
//!   More robust — recommended by Florinsky (2025).
//! - **Zevenbergen-Thorne**: Central differences on cardinal/diagonal neighbors.
//!   Legacy method, available for backward compatibility.
//!
//! And two curvature formula variants:
//!
//! - **Full** (default): Includes √(1+p²+q²) denominators.
//!   Correct on steep terrain.
//! - **Simplified**: Omits denominators. Only valid for gentle slopes (<10°).
//!   Error >15% at 30°, >30% at 45°.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// Which curvature to compute
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CurvatureType {
    /// General (mean) curvature: -(d²z/dx² + d²z/dy²) / 2
    #[default]
    General,
    /// Profile curvature: along the direction of maximum slope
    Profile,
    /// Plan (tangential) curvature: perpendicular to the slope direction
    Plan,
}

/// Method for estimating partial derivatives from the 3×3 neighborhood
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DerivativeMethod {
    /// Evans-Young (1979/1978): weighted least-squares using all 9 cells.
    /// Recommended by Florinsky (2025) as the standard 3×3 approach.
    #[default]
    EvansYoung,
    /// Zevenbergen & Thorne (1987): central differences.
    /// Legacy method; less robust but widely used historically.
    ZevenbergenThorne,
}

/// Whether to use full or simplified curvature formulas
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CurvatureFormula {
    /// Full formulas with √(1+p²+q²) denominators (Florinsky 2025).
    /// Correct for all slopes. Recommended.
    #[default]
    Full,
    /// Simplified formulas omitting denominators (Z&T 1987).
    /// Only valid for gentle terrain (<10°). Error >15% at 30°.
    Simplified,
}

/// Parameters for curvature calculation
#[derive(Debug, Clone)]
pub struct CurvatureParams {
    /// Type of curvature to compute
    pub curvature_type: CurvatureType,
    /// Method for partial derivative estimation
    pub method: DerivativeMethod,
    /// Full vs simplified curvature formulas
    pub formula: CurvatureFormula,
    /// Z-factor for unit conversion (default 1.0)
    pub z_factor: f64,
}

impl Default for CurvatureParams {
    fn default() -> Self {
        Self {
            curvature_type: CurvatureType::General,
            method: DerivativeMethod::EvansYoung,
            formula: CurvatureFormula::Full,
            z_factor: 1.0,
        }
    }
}

/// Curvature algorithm
#[derive(Debug, Clone, Default)]
pub struct Curvature;

impl Algorithm for Curvature {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = CurvatureParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Curvature"
    }

    fn description(&self) -> &'static str {
        "Calculate surface curvature from a DEM (Zevenbergen & Thorne 1987)"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        curvature(&input, params)
    }
}

/// Calculate surface curvature from a DEM
///
/// Supports Evans-Young (default) and Zevenbergen-Thorne derivative methods,
/// and full (default) or simplified curvature formulas.
///
/// Positive values indicate concave surfaces (profile: decelerating flow;
/// plan: converging flow). Negative values indicate convex surfaces.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Curvature parameters (type, method, formula, z_factor)
///
/// # Returns
/// Raster with curvature values (1/m by default)
pub fn curvature(dem: &Raster<f64>, params: CurvatureParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cs = dem.cell_size() * params.z_factor;
    let nodata = dem.nodata();
    let cs2 = cs * cs;
    let method = params.method;
    let formula = params.formula;

    // Precompute method-specific constants
    let two_cs = 2.0 * cs;
    let cs6 = 6.0 * cs;
    let cs2_3 = 3.0 * cs2;
    let cs2_4 = 4.0 * cs2;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let z5 = unsafe { dem.get_unchecked(row, col) };
                if z5.is_nan() || nodata.is_some_and(|nd| (z5 - nd).abs() < f64::EPSILON) {
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

                // Compute partial derivatives based on method
                let (p, q, r, s, t) = match method {
                    DerivativeMethod::EvansYoung => {
                        let p = (z3 + z6 + z9 - z1 - z4 - z7) / cs6;
                        let q = (z1 + z2 + z3 - z7 - z8 - z9) / cs6;
                        let r = (z1 + z3 + z4 + z6 + z7 + z9
                            - 2.0 * (z2 + z5 + z8))
                            / cs2_3;
                        let s = (z3 + z7 - z1 - z9) / cs2_4;
                        let t = (z1 + z2 + z3 + z7 + z8 + z9
                            - 2.0 * (z4 + z5 + z6))
                            / cs2_3;
                        (p, q, r, s, t)
                    }
                    DerivativeMethod::ZevenbergenThorne => {
                        let p = (z6 - z4) / two_cs;
                        let q = (z2 - z8) / two_cs;
                        let r = (z4 - 2.0 * z5 + z6) / cs2;
                        let s = (z3 - z1 - z9 + z7) / (4.0 * cs2);
                        let t = (z2 - 2.0 * z5 + z8) / cs2;
                        (p, q, r, s, t)
                    }
                };

                let p2 = p * p;
                let q2 = q * q;
                let p2q2 = p2 + q2;

                *row_data_col = match (params.curvature_type, formula) {
                    // --- Full formulas (with denominators) ---
                    (CurvatureType::General, CurvatureFormula::Full) => {
                        let w = 1.0 + p2q2;
                        -((1.0 + q2) * r - 2.0 * p * q * s + (1.0 + p2) * t)
                            / (2.0 * w * w.sqrt())
                    }
                    (CurvatureType::Profile, CurvatureFormula::Full) => {
                        if p2q2 < 1e-20 {
                            0.0
                        } else {
                            let w = 1.0 + p2q2;
                            -(p2 * r + 2.0 * p * q * s + q2 * t)
                                / (p2q2 * w * w.sqrt())
                        }
                    }
                    (CurvatureType::Plan, CurvatureFormula::Full) => {
                        if p2q2 < 1e-20 {
                            0.0
                        } else {
                            let w_sqrt = (1.0 + p2q2).sqrt();
                            -(q2 * r - 2.0 * p * q * s + p2 * t)
                                / (p2q2 * w_sqrt)
                        }
                    }
                    // --- Simplified formulas (legacy Z&T, no denominators) ---
                    (CurvatureType::General, CurvatureFormula::Simplified) => {
                        -(r + t) / 2.0
                    }
                    (CurvatureType::Profile, CurvatureFormula::Simplified) => {
                        if p2q2 < 1e-20 {
                            0.0
                        } else {
                            -(p2 * r + 2.0 * p * q * s + q2 * t) / p2q2
                        }
                    }
                    (CurvatureType::Plan, CurvatureFormula::Simplified) => {
                        if p2q2 < 1e-20 {
                            0.0
                        } else {
                            -(q2 * r - 2.0 * p * q * s + p2 * t) / p2q2
                        }
                    }
                };
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

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    /// Tilted plane z = row + col → constant first derivatives, zero second derivatives
    fn tilted_plane() -> Raster<f64> {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, (r + c) as f64).unwrap();
            }
        }
        dem
    }

    /// Parabolic bowl z = x² + y² → positive curvature (concave up)
    fn bowl() -> Raster<f64> {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..21 {
            for c in 0..21 {
                let x = c as f64 - 10.0;
                let y = r as f64 - 10.0;
                dem.set(r, c, x * x + y * y).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_general_curvature_flat_plane() {
        let dem = tilted_plane();
        let result = curvature(&dem, CurvatureParams::default()).unwrap();
        // Plane has zero curvature everywhere
        let val = result.get(5, 5).unwrap();
        assert!(val.abs() < 1e-10, "Expected ~0 curvature for plane, got {}", val);
    }

    #[test]
    fn test_general_curvature_bowl() {
        let dem = bowl();
        let result = curvature(&dem, CurvatureParams {
            curvature_type: CurvatureType::General,
            ..Default::default()
        }).unwrap();

        // z = x² + y² → d²z/dx²=2, d²z/dy²=2 → general = -(2+2)/2 = -2
        let val = result.get(10, 10).unwrap();
        assert!(
            (val - (-2.0)).abs() < 1e-6,
            "Expected -2.0 general curvature for bowl center, got {}", val
        );
    }

    #[test]
    fn test_profile_curvature_plane() {
        let dem = tilted_plane();
        let result = curvature(&dem, CurvatureParams {
            curvature_type: CurvatureType::Profile,
            ..Default::default()
        }).unwrap();

        let val = result.get(5, 5).unwrap();
        assert!(val.abs() < 1e-10, "Expected ~0 profile curvature for plane, got {}", val);
    }

    #[test]
    fn test_plan_curvature_plane() {
        let dem = tilted_plane();
        let result = curvature(&dem, CurvatureParams {
            curvature_type: CurvatureType::Plan,
            ..Default::default()
        }).unwrap();

        let val = result.get(5, 5).unwrap();
        assert!(val.abs() < 1e-10, "Expected ~0 plan curvature for plane, got {}", val);
    }

    #[test]
    fn test_curvature_nodata_border() {
        let dem = bowl();
        let result = curvature(&dem, CurvatureParams::default()).unwrap();
        // Borders should be NaN
        assert!(result.get(0, 0).unwrap().is_nan());
        assert!(result.get(0, 10).unwrap().is_nan());
    }
}
