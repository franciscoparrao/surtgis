//! Relative Aspect
//!
//! Computes the angular difference between local aspect and regional
//! (smoothed) aspect:
//!
//!   output = |aspect_local - aspect_regional| mod 180
//!
//! Regional aspect is derived from a Gaussian-smoothed version of the DEM.
//! High values indicate local terrain deviating from regional trend.
//!
//! - 0° = local aspect matches regional direction
//! - 180° = local aspect opposes regional direction
//!
//! Reference: WhiteboxTools `RelativeAspect`

use ndarray::Array2;
use crate::maybe_rayon::*;
use crate::terrain::{
    aspect, AspectOutput,
    gaussian_smoothing, GaussianSmoothingParams,
};
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for relative aspect.
#[derive(Debug, Clone)]
pub struct RelativeAspectParams {
    /// Sigma for Gaussian smoothing to derive regional aspect (default 50).
    pub sigma: f64,
}

impl Default for RelativeAspectParams {
    fn default() -> Self {
        Self { sigma: 50.0 }
    }
}

/// Compute relative aspect.
pub fn relative_aspect(
    dem: &Raster<f64>,
    params: RelativeAspectParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Compute local aspect (degrees, 0-360)
    let local_aspect = aspect(dem, AspectOutput::Degrees)?;

    // Smooth DEM and compute regional aspect
    let radius = (params.sigma * 3.0).ceil() as usize;
    let smoothed = gaussian_smoothing(dem, GaussianSmoothingParams {
        sigma: params.sigma,
        radius,
    })?;
    let regional_aspect = aspect(&smoothed, AspectOutput::Degrees)?;

    // Compute angular difference
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let elev = unsafe { dem.get_unchecked(row, col) };
                if elev.is_nan() || nodata.is_some_and(|nd| (elev - nd).abs() < f64::EPSILON) {
                    continue;
                }

                let la = unsafe { local_aspect.get_unchecked(row, col) };
                let ra = unsafe { regional_aspect.get_unchecked(row, col) };

                if la.is_nan() || ra.is_nan() {
                    continue;
                }

                // Angular difference modulo 180 (undirected)
                let mut diff = (la - ra).abs();
                if diff > 180.0 {
                    diff = 360.0 - diff;
                }
                row_data[col] = diff;
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
    fn test_uniform_slope_low_relative() {
        // Planar slope: local and regional aspect should agree → low relative aspect
        let mut dem = Raster::new(50, 50);
        dem.set_transform(GeoTransform::new(0.0, 50.0, 1.0, -1.0));
        for r in 0..50 {
            for c in 0..50 {
                dem.set(r, c, r as f64 * 10.0).unwrap();
            }
        }

        let result = relative_aspect(&dem, RelativeAspectParams { sigma: 5.0 }).unwrap();
        let v = result.get(25, 25).unwrap();
        if !v.is_nan() {
            assert!(v < 10.0, "Uniform slope should have low relative aspect, got {}", v);
        }
    }

    #[test]
    fn test_range_zero_to_180() {
        let mut dem = Raster::new(50, 50);
        dem.set_transform(GeoTransform::new(0.0, 50.0, 1.0, -1.0));
        for r in 0..50 {
            for c in 0..50 {
                let x = c as f64 - 25.0;
                let y = r as f64 - 25.0;
                dem.set(r, c, x * x + y * y + (x * 0.3).sin() * 30.0).unwrap();
            }
        }

        let result = relative_aspect(&dem, RelativeAspectParams { sigma: 5.0 }).unwrap();
        for r in 10..40 {
            for c in 10..40 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v >= -0.01 && v <= 180.01,
                        "Relative aspect should be [0,180], got {} at ({},{})", v, r, c);
                }
            }
        }
    }
}
