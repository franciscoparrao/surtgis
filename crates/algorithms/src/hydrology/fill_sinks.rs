//! Sink filling for hydrological analysis
//!
//! Implements the Planchon-Darboux (2001) algorithm for filling
//! depressions in a DEM to ensure continuous flow paths.
//!
//! Reference:
//! Planchon, O., Darboux, F. (2001). A fast, simple and versatile algorithm
//! to fill the depressions of digital elevation models.
//! Catena, 46(2-3), 159-176.

use ndarray::Array2;
use surtgis_core::Result;
use surtgis_core::raster::Raster;

/// Parameters for sink filling
#[derive(Debug, Clone)]
pub struct FillSinksParams {
    /// Minimum slope to enforce between cells (prevents flat areas)
    /// Set to 0.0 to allow flat areas after filling.
    ///
    /// D7 fix: the old default (0.01, i.e. 1%) was far more aggressive
    /// than SAGA/WhiteboxTools and inconsistent with this crate's own
    /// `priority_flood::PriorityFloodParams` (`epsilon: 1e-5`). The
    /// default now matches `priority_flood`'s epsilon exactly for
    /// cross-algorithm consistency.
    pub min_slope: f64,
}

impl Default for FillSinksParams {
    fn default() -> Self {
        Self { min_slope: 1e-5 }
    }
}

// Row-major scan order (not D8-code order): tie-breaking in the fill
// depends on the historical neighbor visit order. See `d8` module docs.
use super::d8::{SCAN_DISTANCE as D8_DISTANCES, SCAN_OFFSETS as D8_OFFSETS};

/// Fill depressions in a DEM using the Planchon-Darboux (2001) algorithm.
///
/// This algorithm ensures that every cell has a downslope path to the
/// edge of the DEM, which is required for flow direction and accumulation.
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - Fill parameters (minimum slope enforcement)
///
/// # Returns
/// A new raster with all depressions filled
pub fn fill_sinks(dem: &Raster<f64>, params: FillSinksParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let epsilon = params.min_slope * cell_size;

    // Step 1: Initialize output surface W
    // W(c) = DEM(c) if c is on the border
    // W(c) = very large value otherwise
    let big_value = f64::MAX / 2.0;
    let mut w = Array2::from_elem((rows, cols), big_value);

    // Init pass: nodata cells preserve NaN; physical-border AND
    // nodata-adjacent cells initialise to DEM value (they act as
    // drainage exits); everything else stays at big_value.
    //
    // Why nodata-adjacent counts as boundary: when reprojection leaves
    // NaN curtains between the physical border and a valid interior
    // (typical of WGS84 -> UTM rotated grids), interior cells that
    // see only NaN neighbours have no drainage path under the original
    // init scheme, so they stay at big_value and contaminate
    // every downstream tool. Treating NaN as a drainage exit matches
    // GIS convention and what priority_flood already does.
    let is_nodata = |val: f64| -> bool { dem.is_nodata(val) };

    for row in 0..rows {
        for col in 0..cols {
            let val = unsafe { dem.get_unchecked(row, col) };

            if is_nodata(val) {
                w[(row, col)] = f64::NAN;
                continue;
            }

            let physical_border = row == 0 || row == rows - 1 || col == 0 || col == cols - 1;

            let mut nodata_adjacent = false;
            if !physical_border {
                for (dr, dc) in D8_OFFSETS.iter() {
                    let nr = (row as isize + dr) as usize;
                    let nc = (col as isize + dc) as usize;
                    let nv = unsafe { dem.get_unchecked(nr, nc) };
                    if is_nodata(nv) {
                        nodata_adjacent = true;
                        break;
                    }
                }
            }

            if physical_border || nodata_adjacent {
                w[(row, col)] = val;
            }
        }
    }

    // Step 2: Iteratively lower W until stable
    // A cell's W value can be lowered if a neighbor n satisfies:
    //   W(c) > W(n) + epsilon
    // then: W(c) = max(DEM(c), W(n) + epsilon_d)
    // where epsilon_d accounts for diagonal distance
    let mut changed = true;
    while changed {
        changed = false;

        // Forward pass: top-left to bottom-right
        for row in 1..rows - 1 {
            for col in 1..cols - 1 {
                let dem_val = unsafe { dem.get_unchecked(row, col) };

                // Skip nodata
                if dem.is_nodata(dem_val) {
                    continue;
                }

                if w[(row, col)] > dem_val {
                    for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                        let nr = (row as isize + dr) as usize;
                        let nc = (col as isize + dc) as usize;
                        let eps_d = epsilon * D8_DISTANCES[idx];

                        let wn = w[(nr, nc)];
                        if wn.is_nan() || wn >= big_value {
                            continue;
                        }

                        let new_val = wn + eps_d;
                        if dem_val >= new_val {
                            w[(row, col)] = dem_val;
                            changed = true;
                            break;
                        }
                        if w[(row, col)] > new_val {
                            w[(row, col)] = new_val;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Backward pass: bottom-right to top-left
        for row in (1..rows - 1).rev() {
            for col in (1..cols - 1).rev() {
                let dem_val = unsafe { dem.get_unchecked(row, col) };

                if dem.is_nodata(dem_val) {
                    continue;
                }

                if w[(row, col)] > dem_val {
                    for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                        let nr = (row as isize + dr) as usize;
                        let nc = (col as isize + dc) as usize;
                        let eps_d = epsilon * D8_DISTANCES[idx];

                        let wn = w[(nr, nc)];
                        if wn.is_nan() || wn >= big_value {
                            continue;
                        }

                        let new_val = wn + eps_d;
                        if dem_val >= new_val {
                            w[(row, col)] = dem_val;
                            changed = true;
                            break;
                        }
                        if w[(row, col)] > new_val {
                            w[(row, col)] = new_val;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    debug_assert!(
        w.iter().all(|&v| v.is_nan() || v < big_value),
        "fill-sinks: some interior cells did not drain"
    );

    let mut output = dem.like(0.0);
    *output.data_mut() = w;
    // Normalise nodata metadata to NaN regardless of input sentinel,
    // matching the data array (sentinel cells were converted to NaN
    // during init).
    output.set_nodata(Some(f64::NAN));

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn create_dem_with_sink() -> Raster<f64> {
        // 7x7 DEM with a depression in the center
        //
        // 9 9 9 9 9 9 9
        // 9 8 8 8 8 8 9
        // 9 8 7 7 7 8 9
        // 9 8 7 3 7 8 9   <- center cell is a sink (3 < 7)
        // 9 8 7 7 7 8 9
        // 9 8 8 8 8 8 9
        // 9 9 9 9 9 9 9
        let mut dem = Raster::new(7, 7);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));

        let values = [
            9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 8.0, 7.0,
            7.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 3.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 7.0, 7.0, 8.0,
            9.0, 9.0, 8.0, 8.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
        ];

        for (idx, &val) in values.iter().enumerate() {
            let row = idx / 7;
            let col = idx % 7;
            dem.set(row, col, val).unwrap();
        }

        dem
    }

    #[test]
    fn test_fill_sinks_raises_depression() {
        let dem = create_dem_with_sink();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // The center cell (3,3) had value 3.0, surrounded by 7.0
        // After filling, it should be >= 7.0
        let center = filled.get(3, 3).unwrap();
        assert!(
            center >= 7.0,
            "Sink at (3,3) should be filled to >= 7.0, got {}",
            center
        );
    }

    #[test]
    fn test_fill_sinks_preserves_border() {
        let dem = create_dem_with_sink();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // Border values should remain unchanged
        assert_eq!(filled.get(0, 0).unwrap(), 9.0);
        assert_eq!(filled.get(0, 3).unwrap(), 9.0);
        assert_eq!(filled.get(6, 6).unwrap(), 9.0);
    }

    #[test]
    fn test_fill_sinks_with_outlet() {
        // 5x5 DEM with an outlet gap in the border
        // All border = 10, except (4,2)=2 (outlet)
        // Center (2,2)=1 (sink below outlet)
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        for row in 0..5 {
            for col in 0..5 {
                let is_border = row == 0 || row == 4 || col == 0 || col == 4;
                dem.set(row, col, if is_border { 10.0 } else { 5.0 })
                    .unwrap();
            }
        }
        dem.set(2, 2, 1.0).unwrap(); // Sink
        dem.set(4, 2, 2.0).unwrap(); // Low outlet on border

        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // Sink should be filled but only up to the outlet level
        let center = filled.get(2, 2).unwrap();
        assert!(
            center >= 1.0 && center <= 5.0,
            "Center should be filled but not above interior level, got {}",
            center
        );

        // Interior cells that aren't sinks should be unchanged
        let side = filled.get(1, 1).unwrap();
        assert_eq!(
            side, 5.0,
            "Non-sink interior should be preserved, got {}",
            side
        );
    }

    #[test]
    fn test_fill_sinks_no_change_on_clean_dem() {
        // A sloped plane has no sinks
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        for row in 0..10 {
            for col in 0..10 {
                // Slope toward bottom-right corner
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }

        let filled = fill_sinks(&dem, FillSinksParams::default()).unwrap();

        // Should be essentially unchanged
        for row in 0..10 {
            for col in 0..10 {
                let orig = dem.get(row, col).unwrap();
                let fill = filled.get(row, col).unwrap();
                assert!(
                    fill >= orig,
                    "Filled value should be >= original at ({}, {})",
                    row,
                    col
                );
            }
        }
    }

    #[test]
    fn test_fill_sinks_with_min_slope() {
        let dem = create_dem_with_sink();
        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.01 }).unwrap();

        // With min_slope, filled values should create a slight gradient
        let center = filled.get(3, 3).unwrap();
        let neighbor = filled.get(3, 4).unwrap();

        // Center should be slightly lower than its neighbor (or equal)
        // because we enforce a minimum slope
        assert!(
            center <= neighbor + 0.1,
            "Min slope should create gradient: center={}, neighbor={}",
            center,
            neighbor
        );
    }

    #[test]
    fn test_fill_sinks_preserves_interior_nan() {
        // 5x5 sloped plane with one NaN cell and one sink next to it.
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }
        dem.set(2, 2, f64::NAN).unwrap();
        // Sink directly adjacent to the NaN cell.
        dem.set(2, 3, 0.5).unwrap();

        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // NaN preserved.
        assert!(
            filled.get(2, 2).unwrap().is_nan(),
            "NaN cell at (2,2) must be preserved"
        );
        // Sink filled to a finite, reasonable value (never big_value).
        let sink = filled.get(2, 3).unwrap();
        assert!(
            sink.is_finite() && sink.abs() < 1e100,
            "Sink at (2,3) should be finite and not big_value, got {}",
            sink
        );
        // Sink filled UP (>= original) to at least the level of a non-NaN neighbour.
        assert!(sink >= 0.5, "Sink should be raised, got {}", sink);
        // No cell anywhere should be big_value-leaked.
        for row in 0..5 {
            for col in 0..5 {
                let v = filled.get(row, col).unwrap();
                assert!(
                    v.is_nan() || v.abs() < 1e100,
                    "big_value leak at ({}, {}) = {}",
                    row,
                    col,
                    v
                );
            }
        }
    }

    #[test]
    fn test_fill_sinks_with_nodata_curtain() {
        // 9x9 DEM with 3x3 NaN blocks at each corner, leaving a
        // cross-shaped valid region with a sink at the very centre.
        // This is the synthetic analogue of a reprojected UTM grid:
        // NaN curtains seal off most border-to-interior drainage paths.
        let mut dem = Raster::new(9, 9);
        dem.set_transform(GeoTransform::new(0.0, 9.0, 1.0, -1.0));
        for row in 0..9 {
            for col in 0..9 {
                dem.set(row, col, 10.0).unwrap();
            }
        }
        // Four 3x3 NaN corners.
        for (r0, c0) in [(0, 0), (0, 6), (6, 0), (6, 6)] {
            for dr in 0..3 {
                for dc in 0..3 {
                    dem.set(r0 + dr, c0 + dc, f64::NAN).unwrap();
                }
            }
        }
        // Centre sink.
        dem.set(4, 4, 2.0).unwrap();

        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // All NaN corners preserved.
        for (r0, c0) in [(0, 0), (0, 6), (6, 0), (6, 6)] {
            for dr in 0..3 {
                for dc in 0..3 {
                    let v = filled.get(r0 + dr, c0 + dc).unwrap();
                    assert!(
                        v.is_nan(),
                        "Corner NaN cell ({}, {}) was modified to {}",
                        r0 + dr,
                        c0 + dc,
                        v
                    );
                }
            }
        }
        // Sink at centre filled.
        let centre = filled.get(4, 4).unwrap();
        assert!(
            centre.is_finite() && centre >= 2.0 && centre <= 10.0,
            "Centre sink (4,4) should be filled to a finite value in [2, 10], got {}",
            centre
        );
        // No big_value leak anywhere.
        for row in 0..9 {
            for col in 0..9 {
                let v = filled.get(row, col).unwrap();
                assert!(
                    v.is_nan() || v.abs() < 1e100,
                    "big_value leak at ({}, {}) = {}",
                    row,
                    col,
                    v
                );
            }
        }
        // Output nodata metadata is NaN.
        assert!(
            filled.nodata().is_some_and(|nd| nd.is_nan()),
            "Output nodata metadata should be Some(NaN)"
        );
    }

    #[test]
    fn test_fill_sinks_explicit_nodata_sentinel() {
        // Sentinel-based nodata (-9999) should be converted to NaN in output.
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (row + col) as f64).unwrap();
            }
        }
        dem.set(2, 2, -9999.0).unwrap();
        dem.set_nodata(Some(-9999.0));

        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();

        // Sentinel cell becomes NaN in output (uniform convention).
        assert!(
            filled.get(2, 2).unwrap().is_nan(),
            "Sentinel nodata cell should be NaN in output"
        );
        // Output nodata metadata is NaN, not the original sentinel.
        assert!(
            filled.nodata().is_some_and(|nd| nd.is_nan()),
            "Output nodata metadata should be Some(NaN), not the sentinel"
        );
        // No big_value leak.
        for row in 0..5 {
            for col in 0..5 {
                let v = filled.get(row, col).unwrap();
                assert!(
                    v.is_nan() || v.abs() < 1e100,
                    "big_value leak at ({}, {}) = {}",
                    row,
                    col,
                    v
                );
            }
        }
    }
}
