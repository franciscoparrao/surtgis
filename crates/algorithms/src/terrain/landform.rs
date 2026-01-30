//! Landform classification from DEMs
//!
//! Classifies terrain into landform categories using TPI at two scales
//! combined with slope, following Weiss (2001) and Jenness (2006).
//!
//! The method computes TPI at a small (local) and large (regional) scale,
//! then standardizes each (z-score), and classifies based on thresholds:
//!
//! | Small TPI | Large TPI | Slope | Landform          | Code |
//! |-----------|-----------|-------|-------------------|------|
//! | high +    | high +    |  any  | Ridge / hilltop   |  1   |
//! | high +    | near 0    |  any  | Upper slope       |  2   |
//! | high +    | low -     |  any  | Upland drainage   |  3   |
//! | near 0    | high +    | steep | Open slope        |  4   |
//! | near 0    | high +    | gentle| U-shaped valley   |  5   |
//! | near 0    | near 0    | steep | Plain / flat      |  6   |
//! | near 0    | near 0    | gentle| Mid-slope         |  7   |
//! | near 0    | low -     |  any  | Upper drainage    |  8   |
//! | low -     | high +    |  any  | Mid-slope valley  |  9   |
//! | low -     | near 0    |  any  | Shallow valley    | 10   |
//! | low -     | low -     |  any  | Valley / lowland  | 11   |
//!
//! The output raster has integer codes 1–11 (as f64). NaN for nodata.
//!
//! Reference: Weiss, A. (2001) "Topographic Position and Landforms Analysis"
//!            Jenness, J. (2006) "Topographic Position Index extension for ArcView"

use ndarray::Array2;
use rayon::prelude::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

use crate::terrain::slope::{slope, SlopeParams, SlopeUnits};
use crate::terrain::tpi::{tpi, TpiParams};

/// Landform class codes
pub mod class {
    pub const RIDGE: f64 = 1.0;
    pub const UPPER_SLOPE: f64 = 2.0;
    pub const UPLAND_DRAINAGE: f64 = 3.0;
    pub const OPEN_SLOPE: f64 = 4.0;
    pub const U_SHAPED_VALLEY: f64 = 5.0;
    pub const PLAIN: f64 = 6.0;
    pub const MID_SLOPE: f64 = 7.0;
    pub const UPPER_DRAINAGE: f64 = 8.0;
    pub const MID_SLOPE_VALLEY: f64 = 9.0;
    pub const SHALLOW_VALLEY: f64 = 10.0;
    pub const VALLEY: f64 = 11.0;
}

/// Parameters for landform classification
#[derive(Debug, Clone)]
pub struct LandformParams {
    /// Small-scale TPI radius in cells (default 3)
    pub small_radius: usize,
    /// Large-scale TPI radius in cells (default 10)
    pub large_radius: usize,
    /// Threshold for standardized TPI (z-score) to be considered "high" or "low"
    /// Default 1.0 (one standard deviation)
    pub tpi_threshold: f64,
    /// Slope threshold (degrees) separating "gentle" from "steep"
    /// Default 6.0 degrees
    pub slope_threshold: f64,
}

impl Default for LandformParams {
    fn default() -> Self {
        Self {
            small_radius: 3,
            large_radius: 10,
            tpi_threshold: 1.0,
            slope_threshold: 6.0,
        }
    }
}

/// Landform classification algorithm
#[derive(Debug, Clone, Default)]
pub struct Landform;

impl Algorithm for Landform {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = LandformParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Landform Classification"
    }

    fn description(&self) -> &'static str {
        "Classify terrain into landform categories using multi-scale TPI and slope"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        landform_classification(&input, params)
    }
}

/// Classify terrain into landform categories
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Classification parameters
///
/// # Returns
/// Raster with landform class codes (1–11 as f64)
pub fn landform_classification(dem: &Raster<f64>, params: LandformParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();

    // Step 1: Compute TPI at two scales
    let tpi_small = tpi(dem, TpiParams { radius: params.small_radius })?;
    let tpi_large = tpi(dem, TpiParams { radius: params.large_radius })?;

    // Step 2: Compute slope in degrees
    let slope_deg = slope(dem, SlopeParams {
        units: SlopeUnits::Degrees,
        z_factor: 1.0,
    })?;

    // Step 3: Standardize TPI values (z-score)
    let (small_mean, small_std) = mean_std(&tpi_small);
    let (large_mean, large_std) = mean_std(&tpi_large);

    let threshold = params.tpi_threshold;
    let slope_thresh = params.slope_threshold;

    // Step 4: Classify in parallel
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let sv = unsafe { tpi_small.get_unchecked(row, col) };
                let lv = unsafe { tpi_large.get_unchecked(row, col) };
                let sl = unsafe { slope_deg.get_unchecked(row, col) };

                if sv.is_nan() || lv.is_nan() || sl.is_nan() {
                    continue;
                }

                // Standardize
                let sz = if small_std > 1e-10 {
                    (sv - small_mean) / small_std
                } else {
                    0.0
                };
                let lz = if large_std > 1e-10 {
                    (lv - large_mean) / large_std
                } else {
                    0.0
                };

                // Classify
                let s_high = sz > threshold;
                let s_low = sz < -threshold;
                let l_high = lz > threshold;
                let l_low = lz < -threshold;
                let gentle = sl < slope_thresh;

                row_data[col] = if s_high && l_high {
                    class::RIDGE
                } else if s_high && !l_high && !l_low {
                    class::UPPER_SLOPE
                } else if s_high && l_low {
                    class::UPLAND_DRAINAGE
                } else if !s_high && !s_low && l_high && !gentle {
                    class::OPEN_SLOPE
                } else if !s_high && !s_low && l_high && gentle {
                    class::U_SHAPED_VALLEY
                } else if !s_high && !s_low && !l_high && !l_low && !gentle {
                    class::MID_SLOPE
                } else if !s_high && !s_low && !l_high && !l_low && gentle {
                    class::PLAIN
                } else if !s_high && !s_low && l_low {
                    class::UPPER_DRAINAGE
                } else if s_low && l_high {
                    class::MID_SLOPE_VALLEY
                } else if s_low && !l_high && !l_low {
                    class::SHALLOW_VALLEY
                } else {
                    // s_low && l_low
                    class::VALLEY
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

/// Compute mean and standard deviation of valid (non-NaN) values
fn mean_std(raster: &Raster<f64>) -> (f64, f64) {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0u64;

    for &v in raster.data().iter() {
        if !v.is_nan() {
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
    }

    if count == 0 {
        return (0.0, 0.0);
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - mean * mean;
    let std = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    (mean, std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn mountain_dem() -> Raster<f64> {
        // 50x50 DEM: cone-shaped mountain centered at (25,25)
        let mut dem = Raster::new(50, 50);
        dem.set_transform(GeoTransform::new(0.0, 50.0, 1.0, -1.0));
        for r in 0..50 {
            for c in 0..50 {
                let dx = c as f64 - 25.0;
                let dy = r as f64 - 25.0;
                let dist = (dx * dx + dy * dy).sqrt();
                let elev = (25.0 - dist).max(0.0) * 10.0; // peak = 250
                dem.set(r, c, elev).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_landform_classification_runs() {
        let dem = mountain_dem();
        let result = landform_classification(&dem, LandformParams {
            small_radius: 2,
            large_radius: 5,
            tpi_threshold: 1.0,
            slope_threshold: 6.0,
        }).unwrap();

        // Peak should be a high-TPI class (ridge or upper slope)
        let peak_class = result.get(25, 25).unwrap();
        assert!(
            !peak_class.is_nan(),
            "Peak should have a class, got NaN"
        );
        assert!(
            peak_class >= 1.0 && peak_class <= 11.0,
            "Class should be 1-11, got {}",
            peak_class
        );
    }

    #[test]
    fn test_landform_flat_area() {
        let mut dem = Raster::filled(50, 50, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 50.0, 1.0, -1.0));

        let result = landform_classification(&dem, LandformParams {
            small_radius: 2,
            large_radius: 5,
            tpi_threshold: 1.0,
            slope_threshold: 6.0,
        }).unwrap();

        // Flat surface → plain (class 6)
        let center = result.get(25, 25).unwrap();
        assert!(
            !center.is_nan(),
            "Center should have a class"
        );
        assert!(
            (center - class::PLAIN).abs() < 0.1,
            "Flat surface should be PLAIN (6), got {}",
            center
        );
    }

    #[test]
    fn test_landform_all_classes_valid() {
        let dem = mountain_dem();
        let result = landform_classification(&dem, LandformParams {
            small_radius: 2,
            large_radius: 5,
            tpi_threshold: 0.5,
            slope_threshold: 10.0,
        }).unwrap();

        // All valid cells should have class codes 1–11
        for r in 0..50 {
            for c in 0..50 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(
                        v >= 1.0 && v <= 11.0,
                        "Invalid class {} at ({}, {})",
                        v, r, c
                    );
                }
            }
        }
    }
}
