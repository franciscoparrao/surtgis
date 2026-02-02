//! Topographic Position Index (TPI)
//!
//! TPI measures the difference between the elevation of a cell and the mean
//! elevation of its surrounding neighborhood:
//!
//!   TPI = z_center - mean(z_neighbors)
//!
//! - Positive TPI → cell is higher than surroundings (ridge, hilltop)
//! - Negative TPI → cell is lower than surroundings (valley, depression)
//! - Near-zero TPI → cell is at the same level (flat area or mid-slope)
//!
//! Supports configurable neighborhood radius for multi-scale analysis.
//! Reference: Weiss (2001) "Topographic Position and Landforms Analysis"

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// Parameters for TPI calculation
#[derive(Debug, Clone)]
pub struct TpiParams {
    /// Neighborhood radius in cells (default 1 → 3x3 window)
    /// Radius 2 → 5x5, radius 5 → 11x11, etc.
    pub radius: usize,
}

impl Default for TpiParams {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

/// TPI algorithm
#[derive(Debug, Clone, Default)]
pub struct Tpi;

impl Algorithm for Tpi {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = TpiParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "TPI"
    }

    fn description(&self) -> &'static str {
        "Topographic Position Index: elevation relative to neighborhood mean"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        tpi(&input, params)
    }
}

/// Calculate Topographic Position Index
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - TPI parameters (neighborhood radius)
///
/// # Returns
/// Raster with TPI values (same units as input elevation)
pub fn tpi(dem: &Raster<f64>, params: TpiParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let nodata = dem.nodata();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let center = unsafe { dem.get_unchecked(row, col) };
                if center.is_nan() || nodata.map_or(false, |nd| (center - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                // Skip cells where the neighborhood extends outside the raster
                let ri = row as isize;
                let ci = col as isize;
                if ri < r || ri >= rows as isize - r || ci < r || ci >= cols as isize - r {
                    continue;
                }

                let mut sum = 0.0;
                let mut count = 0u32;

                for dr in -r..=r {
                    for dc in -r..=r {
                        if dr == 0 && dc == 0 {
                            continue; // exclude center
                        }
                        let nr = (ri + dr) as usize;
                        let nc = (ci + dc) as usize;
                        let nv = unsafe { dem.get_unchecked(nr, nc) };
                        if !nv.is_nan() && nodata.map_or(true, |nd| (nv - nd).abs() >= f64::EPSILON)
                        {
                            sum += nv;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    row_data[col] = center - sum / count as f64;
                }
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), output_data).map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_tpi_flat_surface() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = tpi(&dem, TpiParams::default()).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(
            val.abs() < 1e-10,
            "Expected TPI ~0 for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_tpi_peak() {
        // Center peak: 100 surrounded by 50
        let mut dem = Raster::filled(5, 5, 50.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        dem.set(2, 2, 100.0).unwrap();

        let result = tpi(&dem, TpiParams { radius: 1 }).unwrap();
        let val = result.get(2, 2).unwrap();
        // TPI = 100 - 50 = 50
        assert!(
            (val - 50.0).abs() < 1e-10,
            "Expected TPI=50 for peak, got {}",
            val
        );
    }

    #[test]
    fn test_tpi_valley() {
        // Center depression: 10 surrounded by 50
        let mut dem = Raster::filled(5, 5, 50.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        dem.set(2, 2, 10.0).unwrap();

        let result = tpi(&dem, TpiParams { radius: 1 }).unwrap();
        let val = result.get(2, 2).unwrap();
        // TPI = 10 - 50 = -40
        assert!(
            (val - (-40.0)).abs() < 1e-10,
            "Expected TPI=-40 for valley, got {}",
            val
        );
    }

    #[test]
    fn test_tpi_larger_radius() {
        let mut dem = Raster::filled(11, 11, 50.0);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        dem.set(5, 5, 100.0).unwrap();

        // radius=1: only immediate neighbors
        let r1 = tpi(&dem, TpiParams { radius: 1 }).unwrap();
        // radius=2: 5x5 window
        let r2 = tpi(&dem, TpiParams { radius: 2 }).unwrap();

        let v1 = r1.get(5, 5).unwrap();
        let v2 = r2.get(5, 5).unwrap();

        // Both positive (peak), but r2 should be larger (more 50s in average)
        assert!(v1 > 0.0 && v2 > 0.0);
        assert!(
            v2 > v1 - 1.0, // v2 ≈ 100 - 50 = 50 (only 1 non-50 in neighborhood)
            "Larger radius should still detect peak"
        );
    }
}
