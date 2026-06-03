//! Per-cell shadow ray-march with early exit.
//!
//! This is **the** primitive `ray_shade` needs. The terrain crate already
//! offers [`horizon_angle_map`](surtgis_algorithms::terrain::horizon_angle_map),
//! but that function computes the *maximum* elevation angle to an occluder
//! along a fixed azimuth at every cell — it walks the whole ray even after
//! finding occluders. The shadow query is much cheaper: at every step we
//! only need to know whether the ray has been blocked yet, and we can
//! early-exit the moment we find any single occluder. Skipping that
//! distinction is what makes pre-M2 timings show
//! `horizon_angle_map × 11 ≈ 5 s` while rayshader's full
//! `ray_shade` lands at ~1.24 s on the same DEM.
//!
//! Algorithm — for each interior cell with starting elevation `z0`:
//!   ```text
//!   step along (dr, dc) by 1 cell;
//!   distance_world = step_index * cell_size;
//!   required_ray_height = z0 + distance_world * tan(sun_altitude);
//!   if terrain_at(stepped_cell) > required_ray_height: SHADOWED, exit;
//!   ```
//!
//! Border / NaN cells are returned as NaN.

use ndarray::Array2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use surtgis_core::raster::Raster;

use crate::Result;

/// Per-cell binary shadow mask: `1.0` = lit, `0.0` = in shadow.
///
/// `azimuth_rad`: 0 = N, π/2 = E (clockwise), matching the SurtGIS
/// hillshade / horizon-angle convention.
///
/// `altitude_rad`: 0 = grazing, π/2 = overhead.
///
/// `radius`: maximum ray-march distance in cells. Past this distance the
/// ray is assumed unoccluded. For a typical 600×600 DEM, a radius of
/// `max(rows, cols)` is the rayshader-equivalent setting (~850 cells).
pub fn cast_shadow_ray_mask(
    dem: &Raster<f64>,
    azimuth_rad: f64,
    altitude_rad: f64,
    radius: usize,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let tan_alt = altitude_rad.tan();

    // Direction vector matching horizon_angle_map's convention:
    //   N = (-1,  0), E = ( 0,  1)
    let dr = -azimuth_rad.cos();
    let dc = azimuth_rad.sin();

    // Pre-compute the per-step height increment so the inner loop just
    // does additions rather than multiplications.
    let height_step = cell_size * tan_alt;
    let rows_f = rows as f64;
    let cols_f = cols as f64;

    // Per-row parallel pass. Each row produces a `cols`-long vector of
    // intensities, which we then assemble into the output.
    let row_results: Vec<Vec<f64>> = (0..rows)
        .into_par_iter()
        .map(|row| {
            let mut out_row = vec![1.0f64; cols];

            for col in 0..cols {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    out_row[col] = f64::NAN;
                    continue;
                }

                // Incremental position + required-height state. Avoids the
                // per-step `step as f64 * dr/dc/tan_alt` multiplications
                // that dominated the previous implementation.
                let mut nr_f = row as f64;
                let mut nc_f = col as f64;
                let mut required = z0;

                let mut shadowed = false;
                for _ in 0..radius {
                    nr_f += dr;
                    nc_f += dc;
                    required += height_step;
                    if nr_f < 0.0 || nc_f < 0.0 || nr_f >= rows_f || nc_f >= cols_f {
                        break;
                    }
                    let z = unsafe { dem.get_unchecked(nr_f as usize, nc_f as usize) };
                    if z.is_nan() {
                        // Treat nodata as non-occluder; rayshader does the
                        // same when stepping outside the heightmap.
                        continue;
                    }
                    if z > required {
                        shadowed = true;
                        break;
                    }
                }
                if shadowed {
                    out_row[col] = 0.0;
                }
            }

            out_row
        })
        .collect();

    let mut data = Vec::with_capacity(rows * cols);
    for r in row_results {
        data.extend(r);
    }
    let mut out = Raster::new(rows, cols);
    out.set_transform(*dem.transform());
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| crate::ReliefError::Shape(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ridge_dem(rows: usize, cols: usize, peak_col: usize, height: f64) -> Raster<f64> {
        let mut dem = Raster::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let d = (c as f64 - peak_col as f64).abs();
                let e = (height - d).max(0.0);
                dem.set(r, c, e).unwrap();
            }
        }
        dem
    }

    #[test]
    fn flat_dem_overhead_sun_lit_everywhere() {
        let mut dem = Raster::new(8, 8);
        for r in 0..8 {
            for c in 0..8 {
                dem.set(r, c, 50.0).unwrap();
            }
        }
        let out = cast_shadow_ray_mask(&dem, 0.0, 80f64.to_radians(), 10).unwrap();
        for v in out.data().iter() {
            assert!(v.is_nan() || (*v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn ridge_shadows_the_lit_side_with_grazing_east_sun() {
        // Ridge at column 8. Sun in the east (azimuth 90°) at very low
        // altitude (5°). Cells west of the ridge sit behind it from the
        // sun's POV → shadowed. Cells east are lit.
        let dem = ridge_dem(4, 16, 8, 5.0);
        let out = cast_shadow_ray_mask(&dem, 90f64.to_radians(), 5f64.to_radians(), 16).unwrap();
        // Spot-check a few specific cells.
        let east = out.get(2, 12).unwrap();
        let west = out.get(2, 3).unwrap();
        assert!(east > 0.5, "east of ridge should be lit, got {east}");
        assert!(west < 0.5, "west of ridge should be in shadow, got {west}");
    }
}
