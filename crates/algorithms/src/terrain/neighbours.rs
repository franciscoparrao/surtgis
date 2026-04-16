//! Elevation Neighbour Statistics (3×3 window)
//!
//! Computes directional elevation change statistics in the immediate
//! 3×3 neighbourhood:
//!
//! - `max_downslope_elev_change`: max(center - neighbor) for neighbors below center
//! - `min_downslope_elev_change`: min(center - neighbor) for neighbors below center
//! - `max_upslope_elev_change`: max(neighbor - center) for neighbors above center
//! - `num_downslope_neighbours`: count of neighbors with elevation < center
//! - `num_upslope_neighbours`: count of neighbors with elevation > center
//!
//! Reference: WhiteboxTools `MaxDownslopeElevChange`, `MinDownslopeElevChange`,
//! `MaxUpslopeElevChange`, `NumDownslopeNeighbours`, `NumUpslopeNeighbours`

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// All 5 neighbour statistics in a single pass.
pub struct NeighbourStatsResult {
    pub max_downslope_change: Raster<f64>,
    pub min_downslope_change: Raster<f64>,
    pub max_upslope_change: Raster<f64>,
    pub num_downslope: Raster<f64>,
    pub num_upslope: Raster<f64>,
}

/// Compute all 5 neighbour elevation statistics in one pass.
pub fn neighbour_stats(dem: &Raster<f64>) -> Result<NeighbourStatsResult> {
    let (rows, cols) = dem.shape();
    let nodata = dem.nodata();

    // Each row produces 5 values per pixel
    let output_data: Vec<[f64; 5]> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![[f64::NAN; 5]; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { dem.get_unchecked(row, col) };
                if center.is_nan()
                    || nodata.is_some_and(|nd| (center - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                if row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1 {
                    continue;
                }

                let mut max_down = f64::NEG_INFINITY;
                let mut min_down = f64::INFINITY;
                let mut max_up = f64::NEG_INFINITY;
                let mut n_down = 0.0_f64;
                let mut n_up = 0.0_f64;
                let mut has_down = false;
                let mut has_up = false;

                for dr in -1_isize..=1 {
                    for dc in -1_isize..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = (row as isize + dr) as usize;
                        let nc = (col as isize + dc) as usize;
                        let nv = unsafe { dem.get_unchecked(nr, nc) };
                        if nv.is_nan()
                            || nodata.is_some_and(|nd| (nv - nd).abs() < f64::EPSILON)
                        {
                            continue;
                        }

                        if nv < center {
                            // Downslope neighbour
                            let change = center - nv;
                            if change > max_down { max_down = change; }
                            if change < min_down { min_down = change; }
                            n_down += 1.0;
                            has_down = true;
                        } else if nv > center {
                            // Upslope neighbour
                            let change = nv - center;
                            if change > max_up { max_up = change; }
                            n_up += 1.0;
                            has_up = true;
                        }
                    }
                }

                out[0] = if has_down { max_down } else { 0.0 };
                out[1] = if has_down { min_down } else { 0.0 };
                out[2] = if has_up { max_up } else { 0.0 };
                out[3] = n_down;
                out[4] = n_up;
            }

            row_data
        })
        .collect();

    let make_raster = |idx: usize| -> Result<Raster<f64>> {
        let data: Vec<f64> = output_data.iter().map(|v| v[idx]).collect();
        let mut out = dem.with_same_meta::<f64>(rows, cols);
        out.set_nodata(Some(f64::NAN));
        *out.data_mut() = Array2::from_shape_vec((rows, cols), data)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(out)
    };

    Ok(NeighbourStatsResult {
        max_downslope_change: make_raster(0)?,
        min_downslope_change: make_raster(1)?,
        max_upslope_change: make_raster(2)?,
        num_downslope: make_raster(3)?,
        num_upslope: make_raster(4)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_flat_surface() {
        let mut dem = Raster::filled(5, 5, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));

        let result = neighbour_stats(&dem).unwrap();
        let v = result.max_downslope_change.get(2, 2).unwrap();
        assert!((v - 0.0).abs() < 1e-10, "Flat: max_down should be 0, got {}", v);
        let n = result.num_downslope.get(2, 2).unwrap();
        assert!((n - 0.0).abs() < 1e-10, "Flat: n_down should be 0, got {}", n);
    }

    #[test]
    fn test_peak() {
        let mut dem = Raster::filled(5, 5, 50.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        dem.set(2, 2, 100.0).unwrap();

        let result = neighbour_stats(&dem).unwrap();
        let max_d = result.max_downslope_change.get(2, 2).unwrap();
        assert!((max_d - 50.0).abs() < 1e-10, "Peak: max downslope should be 50, got {}", max_d);
        let n_d = result.num_downslope.get(2, 2).unwrap();
        assert!((n_d - 8.0).abs() < 1e-10, "Peak: all 8 neighbors downslope, got {}", n_d);
        let n_u = result.num_upslope.get(2, 2).unwrap();
        assert!((n_u - 0.0).abs() < 1e-10, "Peak: no upslope neighbors, got {}", n_u);
    }

    #[test]
    fn test_valley() {
        let mut dem = Raster::filled(5, 5, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        dem.set(2, 2, 50.0).unwrap();

        let result = neighbour_stats(&dem).unwrap();
        let max_u = result.max_upslope_change.get(2, 2).unwrap();
        assert!((max_u - 50.0).abs() < 1e-10, "Valley: max upslope should be 50, got {}", max_u);
        let n_u = result.num_upslope.get(2, 2).unwrap();
        assert!((n_u - 8.0).abs() < 1e-10, "Valley: all 8 neighbors upslope, got {}", n_u);
    }

    #[test]
    fn test_slope_gradient() {
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for r in 0..5 {
            for c in 0..5 {
                dem.set(r, c, r as f64 * 10.0).unwrap();
            }
        }

        let result = neighbour_stats(&dem).unwrap();
        let n_d = result.num_downslope.get(2, 2).unwrap();
        // Row above (3 cells) are lower → 3 downslope
        assert!((n_d - 3.0).abs() < 1e-10, "Slope: 3 downslope neighbors, got {}", n_d);
        let n_u = result.num_upslope.get(2, 2).unwrap();
        assert!((n_u - 3.0).abs() < 1e-10, "Slope: 3 upslope neighbors, got {}", n_u);
    }
}
