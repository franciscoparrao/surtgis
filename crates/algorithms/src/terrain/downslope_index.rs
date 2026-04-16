//! Downslope Index (Hjerdt et al. 2004)
//!
//! For each cell, follows the path of steepest descent until the
//! cumulative elevation drop exceeds a given threshold `d` meters.
//! Reports the horizontal distance traveled.
//!
//! Short distances = steep terrain (drops quickly)
//! Long distances = gentle terrain (travels far before dropping)
//!
//! Reference: Hjerdt, K.N., et al. (2004) "A new topographic index to
//! quantify downslope controls on local drainage"

use ndarray::Array2;
use crate::maybe_rayon::*;
use crate::hydrology::{priority_flood, PriorityFloodParams};
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for downslope index.
#[derive(Debug, Clone)]
pub struct DownslopeIndexParams {
    /// Elevation drop threshold in DEM units (default 2.0 meters).
    pub drop: f64,
}

impl Default for DownslopeIndexParams {
    fn default() -> Self {
        Self { drop: 2.0 }
    }
}

/// 8-connected neighbor offsets
const D8_OFFSETS: [(isize, isize); 8] = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
];

/// Distance factor for each offset (1.0 for cardinal, sqrt(2) for diagonal)
const D8_DIST: [f64; 8] = [
    std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
    1.0,                           1.0,
    std::f64::consts::SQRT_2, 1.0, std::f64::consts::SQRT_2,
];

/// Compute downslope index.
///
/// Follows the steepest descent path on a hydrologically filled DEM
/// until the cumulative elevation drop (from the original DEM) exceeds
/// the threshold. Reports the horizontal distance in map units (meters).
pub fn downslope_index(
    dem: &Raster<f64>,
    params: DownslopeIndexParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let drop_threshold = params.drop;
    let nodata = dem.nodata();
    let cell_size = dem.cell_size();
    let max_steps = rows.max(cols) * 2; // safety limit

    // Fill DEM so steepest descent path doesn't get stuck in pits
    let filled = priority_flood(dem, PriorityFloodParams { epsilon: 1e-5 })?;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let start_elev = unsafe { dem.get_unchecked(row, col) };
                if start_elev.is_nan()
                    || nodata.is_some_and(|nd| (start_elev - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                let mut cur_r = row;
                let mut cur_c = col;
                let mut distance = 0.0;

                for _ in 0..max_steps {
                    // Check drop on ORIGINAL DEM
                    let orig_elev = unsafe { dem.get_unchecked(cur_r, cur_c) };
                    if !orig_elev.is_nan() && start_elev - orig_elev >= drop_threshold {
                        break;
                    }

                    // Follow steepest descent on FILLED DEM (avoids pits)
                    let cur_filled = unsafe { filled.get_unchecked(cur_r, cur_c) };
                    let mut best_slope = 0.0;
                    let mut best_idx: Option<usize> = None;

                    for (idx, &(dr, dc)) in D8_OFFSETS.iter().enumerate() {
                        let nr = cur_r as isize + dr;
                        let nc = cur_c as isize + dc;
                        if nr < 0 || nr >= rows as isize || nc < 0 || nc >= cols as isize {
                            continue;
                        }
                        let nv = unsafe { filled.get_unchecked(nr as usize, nc as usize) };
                        if nv.is_nan() {
                            continue;
                        }
                        let elev_drop = cur_filled - nv;
                        if elev_drop > 0.0 {
                            let s = elev_drop / (D8_DIST[idx] * cell_size);
                            if s > best_slope {
                                best_slope = s;
                                best_idx = Some(idx);
                            }
                        }
                    }

                    match best_idx {
                        Some(idx) => {
                            let (dr, dc) = D8_OFFSETS[idx];
                            distance += D8_DIST[idx] * cell_size;
                            cur_r = (cur_r as isize + dr) as usize;
                            cur_c = (cur_c as isize + dc) as usize;
                        }
                        None => break, // boundary
                    }
                }

                *out = distance;
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
    fn test_steep_slope_short_distance() {
        // 10m drop per cell, cell_size=1.0 → drop=2.0 reached in <1 cell
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                dem.set(r, c, (19 - r) as f64 * 10.0).unwrap();
            }
        }

        let result = downslope_index(&dem, DownslopeIndexParams { drop: 2.0 }).unwrap();
        let v = result.get(10, 10).unwrap();
        // cell_size=1.0, steep slope → distance in meters, should be small
        assert!(v <= 2.0, "Steep slope should reach drop quickly, got {}", v);
    }

    #[test]
    fn test_gentle_slope_long_distance() {
        // 0.5m drop per cell, cell_size=1.0 → need 4 cells = 4m to reach drop=2.0
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                dem.set(r, c, (19 - r) as f64 * 0.5).unwrap();
            }
        }

        let result = downslope_index(&dem, DownslopeIndexParams { drop: 2.0 }).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!(v >= 3.0, "Gentle slope should need more distance, got {}", v);
    }

    #[test]
    fn test_non_negative() {
        let mut dem = Raster::new(15, 15);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));
        for r in 0..15 {
            for c in 0..15 {
                let x = c as f64 - 7.0;
                let y = r as f64 - 7.0;
                dem.set(r, c, 100.0 - x * x - y * y).unwrap();
            }
        }

        let result = downslope_index(&dem, DownslopeIndexParams::default()).unwrap();
        for r in 0..15 {
            for c in 0..15 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v >= 0.0, "Distance should be >= 0, got {} at ({},{})", v, r, c);
                }
            }
        }
    }
}
