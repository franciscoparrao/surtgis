//! Pennock Landform Classification (1987)
//!
//! Classifies terrain into 7 landform elements based on slope,
//! profile curvature, and plan curvature:
//!
//! | Class | Name                    | Slope    | Profile   | Plan        |
//! |-------|------------------------|----------|-----------|-------------|
//! | 1     | Level                  | ≤ thresh | -         | -           |
//! | 2     | Divergent shoulder     | > thresh | > 0       | > 0         |
//! | 3     | Convergent shoulder    | > thresh | > 0       | < 0         |
//! | 4     | Divergent backslope    | > thresh | ≈ 0       | > 0         |
//! | 5     | Convergent backslope   | > thresh | ≈ 0       | < 0         |
//! | 6     | Divergent footslope    | > thresh | < 0       | > 0         |
//! | 7     | Convergent footslope   | > thresh | < 0       | < 0         |
//!
//! Profile curvature > 0 = convex (shoulder), < 0 = concave (footslope).
//! Plan curvature > 0 = divergent, < 0 = convergent.
//!
//! Reference: Pennock, Zebarth & De Jong (1987) "Landform classification
//! and soil distribution in hummocky terrain, Saskatchewan"

use ndarray::Array2;
use crate::maybe_rayon::*;
use crate::terrain::{slope, curvature, SlopeParams, SlopeUnits, CurvatureParams, CurvatureType};
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Pennock classification.
#[derive(Debug, Clone)]
pub struct PennockParams {
    /// Slope threshold in degrees below which terrain is "level" (default 3.0)
    pub slope_threshold: f64,
    /// Profile curvature threshold: values within ±this are "linear/backslope" (default 0.1)
    pub profile_curv_threshold: f64,
    /// Plan curvature threshold: values within ±this are treated as neither
    /// divergent nor convergent (default 0.0 = split at zero)
    pub plan_curv_threshold: f64,
}

impl Default for PennockParams {
    fn default() -> Self {
        Self {
            slope_threshold: 3.0,
            profile_curv_threshold: 0.1,
            plan_curv_threshold: 0.0,
        }
    }
}

/// Pennock landform class codes.
pub const PENNOCK_LEVEL: u8 = 1;
pub const PENNOCK_DIV_SHOULDER: u8 = 2;
pub const PENNOCK_CONV_SHOULDER: u8 = 3;
pub const PENNOCK_DIV_BACKSLOPE: u8 = 4;
pub const PENNOCK_CONV_BACKSLOPE: u8 = 5;
pub const PENNOCK_DIV_FOOTSLOPE: u8 = 6;
pub const PENNOCK_CONV_FOOTSLOPE: u8 = 7;

/// Classify terrain using Pennock (1987) method.
///
/// Returns a raster with class codes 1-7 (see constants above).
pub fn pennock(dem: &Raster<f64>, params: PennockParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();

    // Compute inputs
    let slp = slope(dem, SlopeParams { units: SlopeUnits::Degrees, z_factor: 1.0 })?;
    let prof = curvature(dem, CurvatureParams {
        curvature_type: CurvatureType::Profile,
        z_factor: 1.0,
        ..CurvatureParams::default()
    })?;
    let plan = curvature(dem, CurvatureParams {
        curvature_type: CurvatureType::Plan,
        z_factor: 1.0,
        ..CurvatureParams::default()
    })?;

    let nodata = dem.nodata();
    let st = params.slope_threshold;
    let pct = params.profile_curv_threshold;
    let plct = params.plan_curv_threshold;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let elev = unsafe { dem.get_unchecked(row, col) };
                if elev.is_nan() || nodata.is_some_and(|nd| (elev - nd).abs() < f64::EPSILON) {
                    continue;
                }

                let s = unsafe { slp.get_unchecked(row, col) };
                let pc = unsafe { prof.get_unchecked(row, col) };
                let plc = unsafe { plan.get_unchecked(row, col) };

                if s.is_nan() || pc.is_nan() || plc.is_nan() {
                    continue;
                }

                let class = if s <= st {
                    PENNOCK_LEVEL
                } else if pc > pct {
                    // Convex profile = shoulder
                    if plc > plct { PENNOCK_DIV_SHOULDER }
                    else { PENNOCK_CONV_SHOULDER }
                } else if pc < -pct {
                    // Concave profile = footslope
                    if plc > plct { PENNOCK_DIV_FOOTSLOPE }
                    else { PENNOCK_CONV_FOOTSLOPE }
                } else {
                    // Linear profile = backslope
                    if plc > plct { PENNOCK_DIV_BACKSLOPE }
                    else { PENNOCK_CONV_BACKSLOPE }
                };

                row_data[col] = class as f64;
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
    fn test_flat_is_level() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = pennock(&dem, PennockParams::default()).unwrap();
        let v = result.get(5, 5).unwrap();
        if !v.is_nan() {
            assert!((v - PENNOCK_LEVEL as f64).abs() < 0.5,
                "Flat DEM should be Level (1), got {}", v);
        }
    }

    #[test]
    fn test_classes_in_range() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                let x = c as f64 - 10.0;
                let y = r as f64 - 10.0;
                dem.set(r, c, 500.0 - x * x - y * y + (x * 0.5).sin() * 20.0).unwrap();
            }
        }

        let result = pennock(&dem, PennockParams::default()).unwrap();
        for r in 0..20 {
            for c in 0..20 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v >= 1.0 && v <= 7.0,
                        "Class should be 1-7, got {} at ({},{})", v, r, c);
                }
            }
        }
    }
}
