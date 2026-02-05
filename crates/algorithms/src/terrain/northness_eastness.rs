//! Northness and Eastness from DEMs
//!
//! Decompose aspect into continuous directional components:
//! - **Northness** = cos(aspect) → ranges from -1 (south) to +1 (north)
//! - **Eastness** = sin(aspect) → ranges from -1 (west) to +1 (east)
//!
//! These circular decompositions avoid the discontinuity at 0°/360° that makes
//! raw aspect problematic for statistical analysis and modeling.
//!
//! Reference: Stage (1976) "An expression for the effect of aspect, slope,
//! and habitat type on tree growth"

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use super::aspect::{aspect, AspectOutput};

/// Calculate northness from a DEM
///
/// `northness = cos(aspect)` where aspect is in radians (0 = North, clockwise).
///
/// - +1.0 = north-facing slope
/// - -1.0 = south-facing slope
/// -  0.0 = east or west-facing slope
/// - NaN = flat or border cell
///
/// # Arguments
/// * `dem` - Input DEM raster
pub fn northness(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let aspect_raster = aspect(dem, AspectOutput::Radians)?;
    let (rows, cols) = aspect_raster.shape();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let a = unsafe { aspect_raster.get_unchecked(row, col) };
                // aspect returns -1.0 for flat/nodata cells
                if a < 0.0 || a.is_nan() {
                    continue;
                }
                *row_data_col = a.cos();
            }
            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

/// Calculate eastness from a DEM
///
/// `eastness = sin(aspect)` where aspect is in radians (0 = North, clockwise).
///
/// - +1.0 = east-facing slope
/// - -1.0 = west-facing slope
/// -  0.0 = north or south-facing slope
/// - NaN = flat or border cell
///
/// # Arguments
/// * `dem` - Input DEM raster
pub fn eastness(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let aspect_raster = aspect(dem, AspectOutput::Radians)?;
    let (rows, cols) = aspect_raster.shape();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let a = unsafe { aspect_raster.get_unchecked(row, col) };
                if a < 0.0 || a.is_nan() {
                    continue;
                }
                *row_data_col = a.sin();
            }
            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

/// Calculate both northness and eastness from a DEM in a single pass
///
/// More efficient than calling `northness()` and `eastness()` separately
/// since aspect is computed only once.
///
/// # Returns
/// Tuple of (northness, eastness) rasters
pub fn northness_eastness(dem: &Raster<f64>) -> Result<(Raster<f64>, Raster<f64>)> {
    let aspect_raster = aspect(dem, AspectOutput::Radians)?;
    let (rows, cols) = aspect_raster.shape();

    let pairs: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN); cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let a = unsafe { aspect_raster.get_unchecked(row, col) };
                if a < 0.0 || a.is_nan() {
                    continue;
                }
                *row_data_col = (a.cos(), a.sin());
            }
            row_data
        })
        .collect();

    let (north_data, east_data): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let mut north = dem.with_same_meta::<f64>(rows, cols);
    north.set_nodata(Some(f64::NAN));
    *north.data_mut() = Array2::from_shape_vec((rows, cols), north_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    let mut east = dem.with_same_meta::<f64>(rows, cols);
    east.set_nodata(Some(f64::NAN));
    *east.data_mut() = Array2::from_shape_vec((rows, cols), east_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok((north, east))
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn north_slope_dem() -> Raster<f64> {
        // Slopes down to north: higher rows have higher elevation
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, row as f64).unwrap();
            }
        }
        dem
    }

    fn east_slope_dem() -> Raster<f64> {
        // Slopes down to east: higher cols have lower elevation
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, -(col as f64)).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_northness_north_slope() {
        let dem = north_slope_dem();
        let result = northness(&dem).unwrap();
        let val = result.get(5, 5).unwrap();
        // North-facing slope → northness close to +1
        assert!(val > 0.9, "Expected northness ~1.0, got {}", val);
    }

    #[test]
    fn test_eastness_east_slope() {
        let dem = east_slope_dem();
        let result = eastness(&dem).unwrap();
        let val = result.get(5, 5).unwrap();
        // East-facing slope → eastness close to +1
        assert!(val > 0.9, "Expected eastness ~1.0, got {}", val);
    }

    #[test]
    fn test_northness_eastness_combined() {
        let dem = north_slope_dem();
        let (north, east) = northness_eastness(&dem).unwrap();
        let n = north.get(5, 5).unwrap();
        let e = east.get(5, 5).unwrap();
        // n² + e² should be ≈ 1 for non-flat cells
        let sum_sq = n * n + e * e;
        assert!(
            (sum_sq - 1.0).abs() < 0.01,
            "n²+e² should be ~1.0, got {} (n={}, e={})",
            sum_sq,
            n,
            e
        );
    }

    #[test]
    fn test_flat_surface_is_nan() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let result = northness(&dem).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(val.is_nan(), "Flat surface should produce NaN, got {}", val);
    }
}
