//! Elevation Above Pit
//!
//! Computes the depth of each cell within its local depression (sink):
//!
//!   output = DEM_filled - DEM_original
//!
//! Cells on the surface (not in a depression) have a value of 0.
//! Cells inside a depression show how deep they sit below the spill point.
//!
//! This is equivalent to WhiteboxTools `ElevAbovePit` / `DepthInSink`.
//!
//! Uses the Priority-Flood algorithm (Barnes 2014) for depression filling.

use ndarray::Array2;
use crate::maybe_rayon::*;
use crate::hydrology::{priority_flood, PriorityFloodParams};
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Compute elevation above pit (depth-in-sink).
///
/// Returns a raster where each pixel is the difference between the
/// hydrologically filled DEM and the original DEM. Non-sink cells = 0.
pub fn elev_above_pit(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Fill depressions using priority-flood with small epsilon for flat resolution
    let filled = priority_flood(dem, PriorityFloodParams { epsilon: 1e-5 })?;

    // Subtract: filled - original
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let orig = unsafe { dem.get_unchecked(row, col) };
                let fill = unsafe { filled.get_unchecked(row, col) };
                if orig.is_nan() || fill.is_nan() {
                    continue;
                }
                if nodata.is_some_and(|nd| (orig - nd).abs() < f64::EPSILON) {
                    continue;
                }
                row_data[col] = fill - orig;
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
    fn test_no_pits() {
        // Plane: no depressions → all zeros
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, r as f64 * 10.0).unwrap();
            }
        }

        let result = elev_above_pit(&dem).unwrap();
        for r in 0..10 {
            for c in 0..10 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v.abs() < 1e-10, "No-pit DEM should have 0 depth, got {} at ({},{})", v, r, c);
                }
            }
        }
    }

    #[test]
    fn test_single_pit() {
        // Create a bowl: edges = 10, center = 5
        let mut dem = Raster::filled(5, 5, 10.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        dem.set(2, 2, 5.0).unwrap();

        let result = elev_above_pit(&dem).unwrap();
        let pit_depth = result.get(2, 2).unwrap();
        // Filled should be 10 (surrounding), original is 5, depth = 5
        assert!((pit_depth - 5.0).abs() < 1e-10, "Pit depth should be 5, got {}", pit_depth);
    }

    #[test]
    fn test_non_negative() {
        // All values should be >= 0 (filled >= original)
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for r in 0..10 {
            for c in 0..10 {
                let x = c as f64 - 5.0;
                let y = r as f64 - 5.0;
                dem.set(r, c, x * x + y * y + (x * 3.0).sin() * 5.0).unwrap();
            }
        }

        let result = elev_above_pit(&dem).unwrap();
        for r in 0..10 {
            for c in 0..10 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v >= -1e-10, "Depth should be >= 0, got {} at ({},{})", v, r, c);
                }
            }
        }
    }
}
