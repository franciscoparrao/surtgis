//! Topographic Wetness Index (TWI)
//!
//! TWI = ln(a / tan(β))
//! where a = specific catchment area (flow accumulation * cell area / cell width)
//! and β = slope in radians.
//!
//! High TWI values indicate areas prone to saturation.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Compute Topographic Wetness Index
///
/// # Arguments
/// * `flow_acc` - Flow accumulation raster (cell counts)
/// * `slope_rad` - Slope in radians
///
/// # Returns
/// TWI raster. NaN where slope ≈ 0 (flat areas get capped TWI).
pub fn twi(flow_acc: &Raster<f64>, slope_rad: &Raster<f64>) -> Result<Raster<f64>> {
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

    // Minimum slope to avoid ln(inf); ~0.001 rad ≈ 0.057°
    let min_slope = 0.001_f64;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let acc = unsafe { flow_acc.get_unchecked(row, col) };
                let slp = unsafe { slope_rad.get_unchecked(row, col) };

                if acc.is_nan() || slp.is_nan() {
                    continue;
                }

                // Specific catchment area: (acc + 1) * cell_size
                let sca = (acc + 1.0) * cell_size;
                let beta = slp.max(min_slope);

                row_data[col] = (sca / beta.tan()).ln();
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
    fn test_twi_basic() {
        let mut flow_acc = Raster::new(5, 5);
        flow_acc.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let mut slope_r = Raster::new(5, 5);
        slope_r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                flow_acc.set(row, col, 10.0).unwrap();
                slope_r.set(row, col, 0.1).unwrap(); // ~5.7°
            }
        }

        let result = twi(&flow_acc, &slope_r).unwrap();
        let v = result.get(2, 2).unwrap();
        // sca = (10+1)*1 = 11, tan(0.1) = 0.1003, TWI = ln(11/0.1003) ≈ 4.70
        assert!(v > 4.0 && v < 5.5, "TWI should be ~4.7, got {}", v);
    }

    #[test]
    fn test_twi_high_acc_low_slope() {
        let mut flow_acc = Raster::filled(3, 3, 1000.0_f64);
        flow_acc.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));
        let mut slope_r = Raster::filled(3, 3, 0.01_f64);
        slope_r.set_transform(GeoTransform::new(0.0, 3.0, 1.0, -1.0));

        let result = twi(&flow_acc, &slope_r).unwrap();
        let v = result.get(1, 1).unwrap();
        // High accumulation + low slope = high TWI (wet area)
        assert!(v > 10.0, "Should be very wet area, got {}", v);
    }

    #[test]
    fn test_twi_dimension_mismatch() {
        let acc: Raster<f64> = Raster::new(5, 5);
        let slp: Raster<f64> = Raster::new(3, 3);
        assert!(twi(&acc, &slp).is_err());
    }
}
