//! Valley Depth: vertical distance from each cell to the interpolated ridge surface
//!
//! Valley depth is computed by inverting the DEM and filling depressions on the
//! inverted surface. The filled inverted surface corresponds to the ridge envelope.
//! The difference between the ridge envelope and the original DEM gives the
//! valley depth for each cell.
//!
//! Algorithm:
//! 1. Invert DEM: inverted = max_elev - dem
//! 2. Fill depressions on inverted DEM (identifies ridges)
//! 3. Valley depth = filled_inverted - inverted = ridge_surface - dem
//!
//! Valley depth is 0 at ridges and increases toward valley bottoms.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use crate::hydrology::{priority_flood, PriorityFloodParams};

/// Compute valley depth by inverting the DEM and filling depressions.
///
/// # Arguments
/// * `dem` - Input DEM raster
///
/// # Returns
/// Raster with valley depth values (meters). Ridges have depth ≈ 0,
/// valleys have positive depth values.
pub fn valley_depth(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Step 1: Find maximum elevation (ignoring nodata)
    let mut max_elev = f64::NEG_INFINITY;
    for row in 0..rows {
        for col in 0..cols {
            let val = unsafe { dem.get_unchecked(row, col) };
            if val.is_nan() || nodata.is_some_and(|nd| (val - nd).abs() < f64::EPSILON) {
                continue;
            }
            if val > max_elev {
                max_elev = val;
            }
        }
    }

    if max_elev.is_infinite() {
        return Err(Error::Other("DEM contains no valid elevation data".into()));
    }

    // Step 2: Create inverted DEM: inverted = max_elev - dem
    let mut inverted_data = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    for row in 0..rows {
        for col in 0..cols {
            let val = unsafe { dem.get_unchecked(row, col) };
            if val.is_nan() || nodata.is_some_and(|nd| (val - nd).abs() < f64::EPSILON) {
                continue;
            }
            inverted_data[(row, col)] = max_elev - val;
        }
    }

    let mut inverted = dem.with_same_meta::<f64>(rows, cols);
    inverted.set_nodata(Some(f64::NAN));
    *inverted.data_mut() = inverted_data;

    // Step 3: Fill depressions on the inverted DEM
    let filled_inverted = priority_flood(&inverted, PriorityFloodParams { epsilon: 0.0 })?;

    // Step 4: Valley depth = filled_inverted - inverted
    // This equals (ridge_surface_inverted - inverted) = (ridge_surface - dem)
    let mut output_data = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    for row in 0..rows {
        for col in 0..cols {
            let inv = inverted.data()[(row, col)];
            let filled = unsafe { filled_inverted.get_unchecked(row, col) };
            if inv.is_nan() || filled.is_nan() {
                continue;
            }
            let depth = filled - inv;
            // Clamp small negative values from floating-point errors
            output_data[(row, col)] = if depth < 0.0 { 0.0 } else { depth };
        }
    }

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_valley_depth_v_shape() {
        // V-shaped valley with a south-draining gradient so the
        // inverted DEM has proper depressions that can be filled.
        // elevation = |col - center| + small southward slope
        let rows = 7;
        let cols = 11;
        let mut dem = Raster::new(rows, cols);
        dem.set_transform(GeoTransform::new(0.0, cols as f64, 1.0, -1.0));

        for row in 0..rows {
            for col in 0..cols {
                let dist_from_center = (col as f64 - 5.0).abs();
                // Add a gentle southward slope so boundary cells are lower,
                // allowing the inverted DEM to drain properly
                let along_slope = (rows - 1 - row) as f64 * 0.1;
                dem.set(row, col, dist_from_center + along_slope).unwrap();
            }
        }

        let result = valley_depth(&dem).unwrap();

        // Interior center column should have the largest depth
        let center_depth = result.get(3, 5).unwrap();
        // Edge columns (ridges) should have smaller depth
        let edge_depth = result.get(3, 0).unwrap();

        assert!(
            !center_depth.is_nan(),
            "Center should have valid valley depth"
        );
        assert!(
            !edge_depth.is_nan(),
            "Edge should have valid valley depth"
        );
        // Center should have >= edge depth (valley is deeper)
        assert!(
            center_depth >= edge_depth - 0.01,
            "Valley center should have >= ridge depth: center={}, edge={}",
            center_depth,
            edge_depth
        );
    }

    #[test]
    fn test_valley_depth_flat_surface() {
        // Flat surface: all depths should be ~0
        let mut dem = Raster::filled(5, 5, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let result = valley_depth(&dem).unwrap();

        for row in 0..5 {
            for col in 0..5 {
                let val = result.get(row, col).unwrap();
                assert!(
                    val.abs() < 1e-6,
                    "Flat surface should have 0 valley depth at ({},{}), got {}",
                    row,
                    col,
                    val
                );
            }
        }
    }
}
