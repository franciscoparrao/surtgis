//! Viewshed analysis
//!
//! Determines which cells are visible from one or more observer points
//! using line-of-sight ray tracing on a DEM.

use ndarray::Array2;
use rayon::prelude::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for viewshed analysis
#[derive(Debug, Clone)]
pub struct ViewshedParams {
    /// Observer row position
    pub observer_row: usize,
    /// Observer column position
    pub observer_col: usize,
    /// Observer height above ground (meters, default 1.7)
    pub observer_height: f64,
    /// Target height above ground (meters, default 0.0)
    pub target_height: f64,
    /// Maximum radius in cells (0 = unlimited)
    pub max_radius: usize,
}

impl Default for ViewshedParams {
    fn default() -> Self {
        Self {
            observer_row: 0,
            observer_col: 0,
            observer_height: 1.7,
            target_height: 0.0,
            max_radius: 0,
        }
    }
}

/// Compute viewshed from a single observer point
///
/// Uses Bresenham-like line-of-sight algorithm along rays from the
/// observer to each cell on the perimeter, marking visible cells.
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Observer position and height parameters
///
/// # Returns
/// Raster<u8> where 1 = visible, 0 = not visible
pub fn viewshed(dem: &Raster<f64>, params: ViewshedParams) -> Result<Raster<u8>> {
    let (rows, cols) = dem.shape();

    if params.observer_row >= rows || params.observer_col >= cols {
        return Err(Error::IndexOutOfBounds {
            row: params.observer_row,
            col: params.observer_col,
            rows,
            cols,
        });
    }

    let cell_size = dem.cell_size();
    let obs_z = unsafe { dem.get_unchecked(params.observer_row, params.observer_col) }
        + params.observer_height;

    if obs_z.is_nan() {
        return Err(Error::Algorithm("Observer is on NaN cell".into()));
    }

    let max_r = if params.max_radius > 0 {
        params.max_radius
    } else {
        rows.max(cols)
    };

    // Generate target points on the perimeter of the search area
    let obs_r = params.observer_row as isize;
    let obs_c = params.observer_col as isize;

    // Collect perimeter targets
    let mut targets: Vec<(isize, isize)> = Vec::new();
    let r = max_r as isize;

    // Top and bottom rows
    for c in (obs_c - r)..=(obs_c + r) {
        targets.push((obs_r - r, c));
        targets.push((obs_r + r, c));
    }
    // Left and right columns (excluding corners)
    for row in (obs_r - r + 1)..=(obs_r + r - 1) {
        targets.push((row, obs_c - r));
        targets.push((row, obs_c + r));
    }

    // Process rays in parallel - each ray traces from observer to perimeter target
    let visibility_maps: Vec<Vec<(usize, usize)>> = targets
        .par_iter()
        .map(|&(tr, tc)| {
            trace_ray(dem, obs_r, obs_c, obs_z, tr, tc,
                      params.target_height, cell_size, rows, cols)
        })
        .collect();

    // Merge visibility results
    let mut output_data = Array2::<u8>::zeros((rows, cols));
    output_data[(params.observer_row, params.observer_col)] = 1;

    for visible_cells in &visibility_maps {
        for &(r, c) in visible_cells {
            output_data[(r, c)] = 1;
        }
    }

    let mut output = dem.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = output_data;

    Ok(output)
}

/// Trace a ray from observer to target, returning visible cells
fn trace_ray(
    dem: &Raster<f64>,
    obs_r: isize, obs_c: isize, obs_z: f64,
    target_r: isize, target_c: isize,
    target_height: f64,
    cell_size: f64,
    rows: usize, cols: usize,
) -> Vec<(usize, usize)> {
    let mut visible = Vec::new();
    let mut max_angle = f64::NEG_INFINITY;

    // Bresenham-style stepping
    let dr = target_r - obs_r;
    let dc = target_c - obs_c;
    let steps = dr.unsigned_abs().max(dc.unsigned_abs());

    if steps == 0 {
        return visible;
    }

    let step_r = dr as f64 / steps as f64;
    let step_c = dc as f64 / steps as f64;

    for s in 1..=steps {
        let cr = (obs_r as f64 + step_r * s as f64).round() as isize;
        let cc = (obs_c as f64 + step_c * s as f64).round() as isize;

        if cr < 0 || cc < 0 || (cr as usize) >= rows || (cc as usize) >= cols {
            break;
        }

        let r = cr as usize;
        let c = cc as usize;

        let z = unsafe { dem.get_unchecked(r, c) };
        if z.is_nan() {
            break;
        }

        let drow = (r as f64 - obs_r as f64) * cell_size;
        let dcol = (c as f64 - obs_c as f64) * cell_size;
        let dist = (drow * drow + dcol * dcol).sqrt();

        if dist < f64::EPSILON {
            continue;
        }

        let target_z = z + target_height;
        let angle = (target_z - obs_z) / dist;

        if angle >= max_angle {
            visible.push((r, c));
            max_angle = angle;
        }
    }

    visible
}

/// Compute viewshed from multiple observer points
///
/// Returns cumulative visibility: each cell's value indicates how many
/// observers can see it.
pub fn viewshed_multiple(
    dem: &Raster<f64>,
    observers: &[(usize, usize)],
    observer_height: f64,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let mut cumulative = Raster::filled(rows, cols, 0.0_f64);
    cumulative.set_transform(*dem.transform());

    for &(obs_row, obs_col) in observers {
        let params = ViewshedParams {
            observer_row: obs_row,
            observer_col: obs_col,
            observer_height,
            target_height: 0.0,
            max_radius: 0,
        };
        let vs = viewshed(dem, params)?;
        for row in 0..rows {
            for col in 0..cols {
                let v = unsafe { vs.get_unchecked(row, col) };
                if v > 0 {
                    let current = unsafe { cumulative.get_unchecked(row, col) };
                    cumulative.set(row, col, current + 1.0).unwrap();
                }
            }
        }
    }

    Ok(cumulative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_viewshed_flat() {
        // Flat terrain: everything should be visible
        let mut dem = Raster::filled(20, 20, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        let result = viewshed(&dem, ViewshedParams {
            observer_row: 10,
            observer_col: 10,
            observer_height: 1.7,
            ..Default::default()
        }).unwrap();

        // Center and nearby cells should be visible
        assert_eq!(result.get(10, 10).unwrap(), 1);
        assert_eq!(result.get(10, 15).unwrap(), 1);
        assert_eq!(result.get(5, 10).unwrap(), 1);
    }

    #[test]
    fn test_viewshed_blocked() {
        // Wall between observer and target
        let mut dem = Raster::filled(20, 20, 0.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));

        // Create a wall at column 10
        for row in 0..20 {
            dem.set(row, 10, 1000.0).unwrap();
        }

        let result = viewshed(&dem, ViewshedParams {
            observer_row: 10,
            observer_col: 5,
            observer_height: 1.7,
            ..Default::default()
        }).unwrap();

        // Cells behind the wall should not be visible
        assert_eq!(result.get(10, 15).unwrap(), 0, "Cell behind wall should be hidden");
        // Cells before the wall should be visible
        assert_eq!(result.get(10, 8).unwrap(), 1, "Cell before wall should be visible");
    }

    #[test]
    fn test_viewshed_observer_visible() {
        let mut dem = Raster::filled(10, 10, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = viewshed(&dem, ViewshedParams {
            observer_row: 5,
            observer_col: 5,
            observer_height: 0.0,
            ..Default::default()
        }).unwrap();
        assert_eq!(result.get(5, 5).unwrap(), 1, "Observer should always be visible");
    }

    #[test]
    fn test_viewshed_invalid_observer() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        let result = viewshed(&dem, ViewshedParams {
            observer_row: 20,
            observer_col: 5,
            ..Default::default()
        });
        assert!(result.is_err());
    }
}
