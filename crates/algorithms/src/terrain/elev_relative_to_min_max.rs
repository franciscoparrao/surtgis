//! Elevation Relative to Min-Max
//!
//! Normalizes elevation to the global range [0, 1]:
//!
//!   output = (elev - elev_min) / (elev_max - elev_min)
//!
//! 0 = lowest point in the DEM, 1 = highest point.
//!
//! Reference: WhiteboxTools `ElevRelativeToMinMax`

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Compute globally normalized elevation.
///
/// Returns a raster where each pixel is `(z - z_min) / (z_max - z_min)`,
/// ranging from 0 (lowest) to 1 (highest).
pub fn elev_relative_to_min_max(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Pass 1: global min/max
    let mut global_min = f64::INFINITY;
    let mut global_max = f64::NEG_INFINITY;
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { dem.get_unchecked(row, col) };
            if v.is_nan() || nodata.is_some_and(|nd| (v - nd).abs() < f64::EPSILON) {
                continue;
            }
            if v < global_min { global_min = v; }
            if v > global_max { global_max = v; }
        }
    }

    let range = global_max - global_min;

    // Pass 2: normalize
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let v = unsafe { dem.get_unchecked(row, col) };
                if v.is_nan() || nodata.is_some_and(|nd| (v - nd).abs() < f64::EPSILON) {
                    continue;
                }
                if range < f64::EPSILON {
                    row_data[col] = 0.0;
                } else {
                    row_data[col] = (v - global_min) / range;
                }
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

    #[test]
    fn test_elev_relative_bounds() {
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, (r * 10 + c) as f64).unwrap();
            }
        }

        let result = elev_relative_to_min_max(&dem).unwrap();

        // Min cell (0,0) = 0 → normalized = 0
        let v_min = result.get(0, 0).unwrap();
        assert!((v_min - 0.0).abs() < 1e-10, "Min should be 0, got {}", v_min);

        // Max cell (9,9) = 99 → normalized = 1
        let v_max = result.get(9, 9).unwrap();
        assert!((v_max - 1.0).abs() < 1e-10, "Max should be 1, got {}", v_max);
    }

    #[test]
    fn test_elev_relative_flat() {
        let mut dem = Raster::filled(5, 5, 42.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let result = elev_relative_to_min_max(&dem).unwrap();
        let v = result.get(2, 2).unwrap();
        assert!((v - 0.0).abs() < 1e-10, "Flat DEM should yield 0, got {}", v);
    }

    #[test]
    fn test_elev_relative_linear() {
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for r in 0..5 {
            for c in 0..5 {
                dem.set(r, c, r as f64 * 100.0).unwrap();
            }
        }

        let result = elev_relative_to_min_max(&dem).unwrap();
        // Row 2 = 200, range = [0, 400] → 200/400 = 0.5
        let v = result.get(2, 0).unwrap();
        assert!((v - 0.5).abs() < 1e-10, "Mid-range should be 0.5, got {}", v);
    }
}
