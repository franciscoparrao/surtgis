//! LS-Factor for RUSLE soil erosion model (McCool 1987, Desmet & Govers 1996)
//!
//! The LS-Factor combines slope length (L) and slope steepness (S)
//! into a single dimensionless factor used in the Revised Universal
//! Soil Loss Equation (RUSLE). It quantifies the effect of topography
//! on erosion potential.
//!
//! LS = (A_s / 22.13)^m * (sin(beta) / 0.0896)^n
//!
//! where A_s = flow_accumulation * cell_size (specific catchment area per unit contour length),
//! 22.13 m is the standard USLE plot length, and 0.0896 = sin(5.14 deg) is the
//! standard USLE plot slope gradient.
//!
//! References:
//! - McCool, D.K. et al. (1987). Revised slope steepness factor for the USLE.
//! - Desmet, P.J.J. & Govers, G. (1996). A GIS procedure for automatically
//!   calculating the USLE LS factor on topographically complex landscape units.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for LS-Factor calculation
#[derive(Debug, Clone)]
pub struct LsFactorParams {
    /// Cell size in meters (default 1.0)
    pub cell_size: f64,
    /// Exponent for the slope length factor (default 0.4)
    pub m_exponent: f64,
    /// Exponent for the slope steepness factor (default 1.3)
    pub n_exponent: f64,
}

impl Default for LsFactorParams {
    fn default() -> Self {
        Self {
            cell_size: 1.0,
            m_exponent: 0.4,
            n_exponent: 1.3,
        }
    }
}

/// Compute LS-Factor from flow accumulation and slope.
///
/// LS = (A_s / 22.13)^m * (sin(beta) / 0.0896)^n
///
/// where A_s = flow_acc * cell_size (specific catchment area approximation).
///
/// # Arguments
/// * `flow_acc` - Flow accumulation raster (cell count)
/// * `slope_rad` - Slope raster in radians
/// * `params` - LS-Factor parameters
///
/// # Returns
/// Raster with LS-Factor values (dimensionless)
pub fn ls_factor(
    flow_acc: &Raster<f64>,
    slope_rad: &Raster<f64>,
    params: LsFactorParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = flow_acc.shape();
    let (sr, sc) = slope_rad.shape();

    if rows != sr || cols != sc {
        return Err(Error::SizeMismatch {
            er: rows,
            ec: cols,
            ar: sr,
            ac: sc,
        });
    }

    let nodata_acc = flow_acc.nodata();
    let nodata_slp = slope_rad.nodata();
    let cell_size = params.cell_size;
    let m = params.m_exponent;
    let n = params.n_exponent;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let acc = unsafe { flow_acc.get_unchecked(row, col) };
                let slp = unsafe { slope_rad.get_unchecked(row, col) };

                // Skip nodata
                if acc.is_nan()
                    || slp.is_nan()
                    || nodata_acc.is_some_and(|nd| (acc - nd).abs() < f64::EPSILON)
                    || nodata_slp.is_some_and(|nd| (slp - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                // Specific catchment area (per unit contour length)
                let a_s = acc * cell_size;
                let sin_beta = slp.sin();

                // LS = (A_s / 22.13)^m * (sin(beta) / 0.0896)^n
                let l_component = (a_s / 22.13).powf(m);
                let s_component = (sin_beta / 0.0896).powf(n);

                row_data[col] = l_component * s_component;
            }
            row_data
        })
        .collect();

    let mut output = flow_acc.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_ls_factor_flat_surface() {
        // Flat surface: slope = 0 → sin(0) = 0 → S component = 0 → LS ≈ 0
        let mut flow_acc = Raster::filled(5, 5, 100.0);
        flow_acc.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let mut slope = Raster::filled(5, 5, 0.0);
        slope.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let result = ls_factor(
            &flow_acc,
            &slope,
            LsFactorParams {
                cell_size: 10.0,
                ..Default::default()
            },
        )
        .unwrap();

        let val = result.get(2, 2).unwrap();
        assert!(
            val.abs() < 1e-10,
            "LS-Factor should be ~0 for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_ls_factor_tilted_plane() {
        // Uniform slope with uniform flow accumulation → uniform LS > 0
        let slope_deg = 10.0_f64;
        let slope_rad = slope_deg.to_radians();

        let mut flow_acc = Raster::filled(5, 5, 50.0);
        flow_acc.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let mut slope = Raster::filled(5, 5, slope_rad);
        slope.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let params = LsFactorParams {
            cell_size: 10.0,
            ..Default::default()
        };
        let result = ls_factor(&flow_acc, &slope, params).unwrap();

        // All cells should have the same positive LS value
        let val_a = result.get(1, 1).unwrap();
        let val_b = result.get(3, 3).unwrap();
        assert!(val_a > 0.0, "LS should be positive on slope, got {}", val_a);
        assert!(
            (val_a - val_b).abs() < 1e-10,
            "Uniform inputs should give uniform LS: {} vs {}",
            val_a,
            val_b
        );

        // Check approximate expected value
        // A_s = 50 * 10 = 500, sin(10deg) ≈ 0.1736
        // LS = (500/22.13)^0.4 * (0.1736/0.0896)^1.3
        let expected_l = (500.0 / 22.13_f64).powf(0.4);
        let expected_s = (slope_rad.sin() / 0.0896).powf(1.3);
        let expected = expected_l * expected_s;
        assert!(
            (val_a - expected).abs() < 1e-6,
            "Expected LS ≈ {}, got {}",
            expected,
            val_a
        );
    }

    #[test]
    fn test_ls_factor_dimension_mismatch() {
        let flow_acc = Raster::<f64>::new(5, 5);
        let slope = Raster::<f64>::new(3, 3);
        let result = ls_factor(&flow_acc, &slope, LsFactorParams::default());
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}
