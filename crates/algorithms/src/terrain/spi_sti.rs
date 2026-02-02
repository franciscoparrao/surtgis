//! Stream Power Index (SPI) and Sediment Transport Index (STI)
//!
//! SPI = A_s × tan(β)
//! STI = (m+1) × (A_s / 22.13)^m × (sin(β) / 0.0896)^n
//!
//! where A_s = specific catchment area and β = slope angle.
//! Default USLE factors: m=0.4, n=1.3

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Sediment Transport Index
#[derive(Debug, Clone)]
pub struct StiParams {
    /// USLE slope-length exponent (default 0.4)
    pub m: f64,
    /// USLE slope steepness exponent (default 1.3)
    pub n: f64,
}

impl Default for StiParams {
    fn default() -> Self {
        Self { m: 0.4, n: 1.3 }
    }
}

/// Compute Stream Power Index
///
/// SPI = A_s × tan(β)
/// where A_s = (flow_acc + 1) × cell_size
///
/// # Arguments
/// * `flow_acc` - Flow accumulation (cell counts)
/// * `slope_rad` - Slope in radians
pub fn spi(flow_acc: &Raster<f64>, slope_rad: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows_a, cols_a) = flow_acc.shape();
    let (rows_s, cols_s) = slope_rad.shape();

    if rows_a != rows_s || cols_a != cols_s {
        return Err(Error::SizeMismatch {
            er: rows_a, ec: cols_a,
            ar: rows_s, ac: cols_s,
        });
    }

    let rows = rows_a;
    let cols = cols_a;
    let cell_size = flow_acc.cell_size();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let acc = unsafe { flow_acc.get_unchecked(row, col) };
                let slp = unsafe { slope_rad.get_unchecked(row, col) };

                if acc.is_nan() || slp.is_nan() {
                    continue;
                }

                let a_s = (acc + 1.0) * cell_size;
                *row_data_col = a_s * slp.tan();
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

/// Compute Sediment Transport Index
///
/// STI = (m+1) × (A_s / 22.13)^m × (sin(β) / 0.0896)^n
///
/// Based on Moore & Burch (1986) modification of the USLE length-slope factor.
///
/// # Arguments
/// * `flow_acc` - Flow accumulation (cell counts)
/// * `slope_rad` - Slope in radians
/// * `params` - STI parameters (m, n exponents)
pub fn sti(
    flow_acc: &Raster<f64>,
    slope_rad: &Raster<f64>,
    params: StiParams,
) -> Result<Raster<f64>> {
    let (rows_a, cols_a) = flow_acc.shape();
    let (rows_s, cols_s) = slope_rad.shape();

    if rows_a != rows_s || cols_a != cols_s {
        return Err(Error::SizeMismatch {
            er: rows_a, ec: cols_a,
            ar: rows_s, ac: cols_s,
        });
    }

    let rows = rows_a;
    let cols = cols_a;
    let cell_size = flow_acc.cell_size();
    let m = params.m;
    let n = params.n;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let acc = unsafe { flow_acc.get_unchecked(row, col) };
                let slp = unsafe { slope_rad.get_unchecked(row, col) };

                if acc.is_nan() || slp.is_nan() {
                    continue;
                }

                let a_s = (acc + 1.0) * cell_size;
                let length_factor = (a_s / 22.13).powf(m);
                let slope_factor = (slp.sin() / 0.0896).powf(n);

                *row_data_col = (m + 1.0) * length_factor * slope_factor;
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

    fn make_rasters() -> (Raster<f64>, Raster<f64>) {
        let mut acc = Raster::new(5, 5);
        acc.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut slp = Raster::new(5, 5);
        slp.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                acc.set(row, col, 50.0).unwrap();
                slp.set(row, col, 0.2).unwrap(); // ~11.5°
            }
        }
        (acc, slp)
    }

    #[test]
    fn test_spi_positive() {
        let (acc, slp) = make_rasters();
        let result = spi(&acc, &slp).unwrap();
        let v = result.get(2, 2).unwrap();
        assert!(v > 0.0, "SPI should be positive, got {}", v);
        // A_s = 51*1 = 51, tan(0.2) ≈ 0.2027, SPI ≈ 10.34
        assert!((v - 10.34).abs() < 0.1, "SPI should be ~10.34, got {}", v);
    }

    #[test]
    fn test_sti_positive() {
        let (acc, slp) = make_rasters();
        let result = sti(&acc, &slp, StiParams::default()).unwrap();
        let v = result.get(2, 2).unwrap();
        assert!(v > 0.0, "STI should be positive, got {}", v);
    }

    #[test]
    fn test_spi_flat_is_zero() {
        let mut acc = Raster::filled(3, 3, 10.0_f64);
        acc.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut slp = Raster::filled(3, 3, 0.0_f64);
        slp.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        let result = spi(&acc, &slp).unwrap();
        let v = result.get(1, 1).unwrap();
        assert!(v.abs() < 1e-10, "SPI on flat should be 0, got {}", v);
    }

    #[test]
    fn test_dimension_mismatch() {
        let acc: Raster<f64> = Raster::new(5, 5);
        let slp: Raster<f64> = Raster::new(3, 3);
        assert!(spi(&acc, &slp).is_err());
        assert!(sti(&acc, &slp, StiParams::default()).is_err());
    }
}
