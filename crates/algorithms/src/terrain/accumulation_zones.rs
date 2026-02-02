//! Accumulation/dispersion zone classification
//!
//! Classifies terrain into accumulation and dispersion zones based on
//! the signs of horizontal (plan) and vertical (profile) curvatures:
//!
//! | kh (plan) | kv (profile) | Zone              | Code |
//! |-----------|-------------|-------------------|------|
//! | kh < 0    | kv < 0      | Accumulation      | 1    |
//! | kh < 0    | kv > 0      | Transitional acc. | 2    |
//! | kh > 0    | kv < 0      | Transitional disp.| 3    |
//! | kh > 0    | kv > 0      | Dispersion        | 4    |
//!
//! Accumulation zones (kh<0 ∧ kv<0) concentrate both overland and
//! subsurface flow — these are the areas most prone to saturation,
//! erosion, and nutrient transport.
//!
//! Reference: Florinsky, I.V. (2025) "Digital Terrain Analysis" 3rd ed.,
//! Chapter 2.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use super::curvature::{curvature, CurvatureParams, CurvatureType};

/// Zone classification codes
pub const ZONE_ACCUMULATION: f64 = 1.0;
pub const ZONE_TRANSITIONAL_ACC: f64 = 2.0;
pub const ZONE_TRANSITIONAL_DISP: f64 = 3.0;
pub const ZONE_DISPERSION: f64 = 4.0;

/// Classify terrain into accumulation and dispersion zones
///
/// Computes plan (horizontal) and profile (vertical) curvature, then
/// classifies each cell by the sign combination.
///
/// # Arguments
/// * `dem` - Input DEM raster
///
/// # Returns
/// Raster with zone codes (1-4), NaN for borders/nodata
pub fn accumulation_zones(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let kh = curvature(
        dem,
        CurvatureParams {
            curvature_type: CurvatureType::Plan,
            ..Default::default()
        },
    )?;
    let kv = curvature(
        dem,
        CurvatureParams {
            curvature_type: CurvatureType::Profile,
            ..Default::default()
        },
    )?;

    let (rows, cols) = dem.shape();

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let h = unsafe { kh.get_unchecked(row, col) };
                let v = unsafe { kv.get_unchecked(row, col) };
                if h.is_nan() || v.is_nan() {
                    continue;
                }

                row_data[col] = if h < 0.0 && v < 0.0 {
                    ZONE_ACCUMULATION
                } else if h < 0.0 && v >= 0.0 {
                    ZONE_TRANSITIONAL_ACC
                } else if h >= 0.0 && v < 0.0 {
                    ZONE_TRANSITIONAL_DISP
                } else {
                    ZONE_DISPERSION
                };
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

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_accumulation_bowl() {
        // Bowl z = x²+y²: plan curvature < 0 (concave), profile < 0 (concave)
        // at the center → accumulation zone
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let x = col as f64 - 10.0;
                let y = row as f64 - 10.0;
                dem.set(row, col, x * x + y * y).unwrap();
            }
        }

        let result = accumulation_zones(&dem).unwrap();
        // Check that we get valid zone codes for interior cells
        let val = result.get(10, 10).unwrap();
        assert!(
            !val.is_nan(),
            "Interior cell should have a zone code"
        );
        assert!(
            val >= 1.0 && val <= 4.0,
            "Zone code should be 1-4, got {}",
            val
        );
    }

    #[test]
    fn test_accumulation_flat_surface() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = accumulation_zones(&dem).unwrap();
        let val = result.get(5, 5).unwrap();
        // Flat surface: both curvatures = 0 → dispersion zone (kh≥0, kv≥0)
        assert!(
            (val - ZONE_DISPERSION).abs() < 1e-10,
            "Flat surface should be dispersion (4), got {}",
            val
        );
    }

    #[test]
    fn test_all_zone_codes_valid() {
        // Saddle surface z = x²-y² should produce varied zones
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let x = col as f64 - 10.0;
                let y = row as f64 - 10.0;
                dem.set(row, col, x * x - y * y).unwrap();
            }
        }

        let result = accumulation_zones(&dem).unwrap();
        for row in 1..20 {
            for col in 1..20 {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    assert!(
                        val >= 1.0 && val <= 4.0,
                        "Invalid zone code {} at ({}, {})",
                        val,
                        row,
                        col
                    );
                }
            }
        }
    }
}
