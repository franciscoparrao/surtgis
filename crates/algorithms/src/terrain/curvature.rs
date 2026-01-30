//! Surface curvature from DEMs
//!
//! Calculates profile, plan and general (mean) curvature using second-order
//! partial derivatives estimated from a 3x3 neighborhood (Zevenbergen & Thorne 1987).
//!
//! ```text
//! a b c      z1 z2 z3
//! d e f  ->  z4 z5 z6
//! g h i      z7 z8 z9
//! ```
//!
//! The surface at the center cell is approximated by a bivariate quadratic:
//!   Z = Ax²y² + Bx²y + Cxy² + Dx² + Ey² + Fxy + Gx + Hy + I
//!
//! Partial derivatives used:
//!   p  = dz/dx  = (z6 - z4) / (2*cs)
//!   q  = dz/dy  = (z2 - z8) / (2*cs)
//!   r  = d²z/dx² = (z4 - 2*z5 + z6) / cs²
//!   s  = d²z/dxdy = (z3 - z1 - z9 + z7) / (4*cs²)
//!   t  = d²z/dy² = (z2 - 2*z5 + z8) / cs²
//!
//! Curvatures:
//!   General  = -(r + t) / 2
//!   Profile  = -(r*p² + 2*s*p*q + t*q²) / (p² + q²)      (along steepest descent)
//!   Plan     = -(r*q² - 2*s*p*q + t*p²) / (p² + q²)      (perpendicular to descent)

use ndarray::Array2;
use rayon::prelude::*;
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

/// Parameters for curvature calculation
#[derive(Debug, Clone)]
pub struct CurvatureParams {
    /// Type of curvature to compute
    pub curvature_type: CurvatureType,
    /// Z-factor for unit conversion (default 1.0)
    pub z_factor: f64,
}

impl Default for CurvatureParams {
    fn default() -> Self {
        Self {
            curvature_type: CurvatureType::General,
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
/// Uses Zevenbergen & Thorne (1987) method.  Positive values indicate concave
/// surfaces, negative values indicate convex surfaces.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Curvature parameters (type, z_factor)
///
/// # Returns
/// Raster with curvature values (1/m by default)
pub fn curvature(dem: &Raster<f64>, params: CurvatureParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cs = dem.cell_size() * params.z_factor;
    let nodata = dem.nodata();
    let cs2 = cs * cs;
    let two_cs = 2.0 * cs;

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

                let p = (z6 - z4) / two_cs;
                let q = (z2 - z8) / two_cs;
                let r = (z4 - 2.0 * z5 + z6) / cs2;
                let s = (z3 - z1 - z9 + z7) / (4.0 * cs2);
                let t = (z2 - 2.0 * z5 + z8) / cs2;

                row_data[col] = match params.curvature_type {
                    CurvatureType::General => {
                        -(r + t) / 2.0
                    }
                    CurvatureType::Profile => {
                        let p2q2 = p * p + q * q;
                        if p2q2 < 1e-20 {
                            0.0 // flat area
                        } else {
                            -(r * p * p + 2.0 * s * p * q + t * q * q) / p2q2
                        }
                    }
                    CurvatureType::Plan => {
                        let p2q2 = p * p + q * q;
                        if p2q2 < 1e-20 {
                            0.0
                        } else {
                            -(r * q * q - 2.0 * s * p * q + t * p * p) / p2q2
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
            z_factor: 1.0,
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
            z_factor: 1.0,
        }).unwrap();

        let val = result.get(5, 5).unwrap();
        assert!(val.abs() < 1e-10, "Expected ~0 profile curvature for plane, got {}", val);
    }

    #[test]
    fn test_plan_curvature_plane() {
        let dem = tilted_plane();
        let result = curvature(&dem, CurvatureParams {
            curvature_type: CurvatureType::Plan,
            z_factor: 1.0,
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
