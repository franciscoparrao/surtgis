//! Hypsometric Integral: elevation distribution within watersheds
//!
//! The Hypsometric Integral (HI) describes the distribution of elevations
//! within a drainage basin. It is computed as:
//!
//! HI = (mean_elev - min_elev) / (max_elev - min_elev)
//!
//! HI ranges from 0 to 1:
//! - HI > 0.6: young, inequilibrium stage (convex hypsometric curve)
//! - HI 0.35–0.6: mature, equilibrium stage (S-shaped curve)
//! - HI < 0.35: old, monadnock stage (concave curve)
//!
//! Reference:
//! Strahler, A.N. (1952). Hypsometric (area-altitude) analysis of erosional
//! topography. *Geological Society of America Bulletin*, 63(11), 1117–1142.

use std::collections::HashMap;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Compute hypsometric integral for each watershed.
///
/// HI = (mean_elev - min_elev) / (max_elev - min_elev)
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `watersheds` - Watershed ID raster (i32, 0 or negative = no watershed)
///
/// # Returns
/// HashMap mapping watershed_id to hypsometric integral value
pub fn hypsometric_integral(
    dem: &Raster<f64>,
    watersheds: &Raster<i32>,
) -> Result<HashMap<i32, f64>> {
    let (rows, cols) = dem.shape();
    let (wr, wc) = watersheds.shape();

    if rows != wr || cols != wc {
        return Err(Error::SizeMismatch {
            er: rows,
            ec: cols,
            ar: wr,
            ac: wc,
        });
    }

    let nodata_dem = dem.nodata();

    // Accumulate min, max, sum, count per watershed in a single pass
    struct WatershedStats {
        min: f64,
        max: f64,
        sum: f64,
        count: u64,
    }

    let mut stats: HashMap<i32, WatershedStats> = HashMap::new();

    for row in 0..rows {
        for col in 0..cols {
            let ws_id = unsafe { watersheds.get_unchecked(row, col) };
            if ws_id <= 0 {
                continue; // Skip no-watershed cells
            }

            let z = unsafe { dem.get_unchecked(row, col) };
            if z.is_nan() || nodata_dem.is_some_and(|nd| (z - nd).abs() < f64::EPSILON) {
                continue;
            }

            let entry = stats.entry(ws_id).or_insert(WatershedStats {
                min: f64::INFINITY,
                max: f64::NEG_INFINITY,
                sum: 0.0,
                count: 0,
            });

            if z < entry.min {
                entry.min = z;
            }
            if z > entry.max {
                entry.max = z;
            }
            entry.sum += z;
            entry.count += 1;
        }
    }

    // Compute HI for each watershed
    let mut result = HashMap::new();
    for (ws_id, ws_stats) in &stats {
        if ws_stats.count == 0 {
            continue;
        }
        let range = ws_stats.max - ws_stats.min;
        if range.abs() < 1e-10 {
            // Flat watershed: HI = 0.5 by convention
            result.insert(*ws_id, 0.5);
        } else {
            let mean = ws_stats.sum / ws_stats.count as f64;
            let hi = (mean - ws_stats.min) / range;
            result.insert(*ws_id, hi);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_hypsometric_uniform_slope() {
        // Linear slope: elevations 0,1,2,...,9 → mean=4.5, min=0, max=9
        // HI = (4.5 - 0) / (9 - 0) = 0.5
        let rows = 1;
        let cols = 10;
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let mut ws = Raster::<i32>::new(rows, cols);
        ws.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for col in 0..cols {
            dem.set(0, col, col as f64).unwrap();
            ws.set(0, col, 1).unwrap();
        }

        let result = hypsometric_integral(&dem, &ws).unwrap();
        let hi = result[&1];
        assert!(
            (hi - 0.5).abs() < 1e-6,
            "Linear slope should have HI=0.5, got {}",
            hi
        );
    }

    #[test]
    fn test_hypsometric_concave() {
        // Concave profile: most area at low elevations
        // Elevations: 0, 0, 0, 0, 0, 0, 0, 0, 5, 10
        // mean = 1.5, min = 0, max = 10 → HI = 0.15
        let rows = 1;
        let cols = 10;
        let elevs = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 10.0];
        let arr = Array2::from_shape_vec((rows, cols), elevs).unwrap();
        let mut dem = Raster::from_array(arr);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let ws_data = vec![1_i32; 10];
        let ws_arr = Array2::from_shape_vec((rows, cols), ws_data).unwrap();
        let mut ws = Raster::from_array(ws_arr);
        ws.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = hypsometric_integral(&dem, &ws).unwrap();
        let hi = result[&1];
        assert!(
            hi < 0.35,
            "Concave profile should have low HI (<0.35), got {}",
            hi
        );
    }

    #[test]
    fn test_hypsometric_multiple_watersheds() {
        let rows = 2;
        let cols = 4;
        // Watershed 1: elevations 0, 1, 2, 3 → mean=1.5, HI=0.5
        // Watershed 2: elevations 10, 10, 10, 20 → mean=12.5, HI=0.25
        let dem_data = vec![0.0, 1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 20.0];
        let ws_data = vec![1, 1, 1, 1, 2, 2, 2, 2];

        let dem_arr = Array2::from_shape_vec((rows, cols), dem_data).unwrap();
        let ws_arr = Array2::from_shape_vec((rows, cols), ws_data).unwrap();
        let mut dem = Raster::from_array(dem_arr);
        dem.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        let mut ws = Raster::from_array(ws_arr);
        ws.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));

        let result = hypsometric_integral(&dem, &ws).unwrap();
        assert_eq!(result.len(), 2);

        let hi1 = result[&1];
        let hi2 = result[&2];
        assert!(
            (hi1 - 0.5).abs() < 1e-6,
            "Watershed 1 HI should be 0.5, got {}",
            hi1
        );
        assert!(
            (hi2 - 0.25).abs() < 1e-6,
            "Watershed 2 HI should be 0.25, got {}",
            hi2
        );
    }

    #[test]
    fn test_hypsometric_dimension_mismatch() {
        let dem = Raster::<f64>::new(5, 5);
        let ws = Raster::<i32>::new(3, 3);
        let result = hypsometric_integral(&dem, &ws);
        assert!(result.is_err(), "Should error on dimension mismatch");
    }
}
