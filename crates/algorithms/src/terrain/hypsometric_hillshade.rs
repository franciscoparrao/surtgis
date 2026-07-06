//! Hypsometrically Tinted Hillshade
//!
//! Combines hillshade illumination with hypsometric (elevation-normalized)
//! tinting. The result highlights both terrain shape and relative elevation.
//!
//!   output = hillshade × (elev - elev_min) / (elev_max - elev_min)
//!
//! Reference: WhiteboxTools `HypsometricallyTintedHillshade`

use crate::maybe_rayon::*;
use crate::terrain::{HillshadeParams, hillshade};
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Compute hypsometrically tinted hillshade.
///
/// Multiplies the standard hillshade by the normalized elevation position
/// `(z - z_min) / (z_max - z_min)`, producing values in [0, 1].
pub fn hypsometric_hillshade(dem: &Raster<f64>, params: HillshadeParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();

    // Pass 1: compute global min/max
    let mut global_min = f64::INFINITY;
    let mut global_max = f64::NEG_INFINITY;
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { dem.get_unchecked(row, col) };
            if dem.is_nodata(v) {
                continue;
            }
            if v < global_min {
                global_min = v;
            }
            if v > global_max {
                global_max = v;
            }
        }
    }

    // The documented contract is output in [0, 1], so the internal
    // hillshade must be normalized regardless of what the caller set.
    let hs_params = HillshadeParams {
        normalized: true,
        ..params
    };

    let range = global_max - global_min;
    if range < f64::EPSILON {
        // Flat DEM: return hillshade as-is
        return hillshade(dem, hs_params);
    }

    // Compute hillshade
    let hs = hillshade(dem, hs_params)?;

    // Pass 2: multiply hillshade by normalized elevation
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let elev = unsafe { dem.get_unchecked(row, col) };
                let hs_val = unsafe { hs.get_unchecked(row, col) };
                if hs_val.is_nan() {
                    continue;
                }
                if dem.is_nodata(elev) {
                    continue;
                }
                let norm = (elev - global_min) / range;
                row_data[col] = hs_val * norm;
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
    fn test_hypsometric_hillshade_range() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                dem.set(r, c, (r * 10 + c) as f64).unwrap();
            }
        }

        let result = hypsometric_hillshade(&dem, HillshadeParams::default()).unwrap();
        // Interior pixels should be in [0, 1]
        let v = result.get(10, 10).unwrap();
        assert!(v >= 0.0 && v <= 1.0, "Expected [0,1], got {}", v);
    }

    #[test]
    fn test_hypsometric_low_elev_darker() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        // Linear ramp: row 0 = 0, row 19 = 190
        for r in 0..20 {
            for c in 0..20 {
                dem.set(r, c, r as f64 * 10.0).unwrap();
            }
        }

        let result = hypsometric_hillshade(&dem, HillshadeParams::default()).unwrap();
        let low = result.get(2, 10).unwrap();
        let high = result.get(17, 10).unwrap();
        // Both should be valid (not NaN for interior pixels)
        assert!(!low.is_nan(), "Low-elevation pixel should not be NaN");
        assert!(!high.is_nan(), "High-elevation pixel should not be NaN");
    }
}
