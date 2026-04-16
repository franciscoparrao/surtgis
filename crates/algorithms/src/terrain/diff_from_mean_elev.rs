//! Difference from Mean Elevation
//!
//! Computes the difference between a cell's elevation and the mean
//! elevation of its focal neighborhood:
//!
//!   output = z_center - mean(z_neighbors)
//!
//! Unlike DEV (`dev.rs`), this is NOT normalized by standard deviation,
//! so the output is in the same units as the DEM (meters).
//!
//! - Positive → cell is above local average
//! - Negative → cell is below local average
//!
//! Reference: WhiteboxTools `DiffFromMeanElev`

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for difference from mean elevation.
#[derive(Debug, Clone)]
pub struct DiffFromMeanParams {
    /// Neighborhood radius in cells (default 10 → 21×21 window).
    pub radius: usize,
}

impl Default for DiffFromMeanParams {
    fn default() -> Self {
        Self { radius: 10 }
    }
}

/// Compute difference from mean elevation.
pub fn diff_from_mean_elev(dem: &Raster<f64>, params: DiffFromMeanParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let nodata = dem.nodata();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { dem.get_unchecked(row, col) };
                if center.is_nan()
                    || nodata.is_some_and(|nd| (center - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                let ri = row as isize;
                let ci = col as isize;
                if ri < r || ri >= rows as isize - r || ci < r || ci >= cols as isize - r {
                    continue;
                }

                let mut sum = 0.0;
                let mut count = 0u32;

                for dr in -r..=r {
                    for dc in -r..=r {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = (ri + dr) as usize;
                        let nc = (ci + dc) as usize;
                        let nv = unsafe { dem.get_unchecked(nr, nc) };
                        if !nv.is_nan()
                            && nodata.is_none_or(|nd| (nv - nd).abs() >= f64::EPSILON)
                        {
                            sum += nv;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    *out = center - sum / count as f64;
                }
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

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming difference from mean elevation implementing `WindowAlgorithm`.
#[derive(Debug, Clone)]
pub struct DiffFromMeanStreaming {
    pub radius: usize,
}

impl Default for DiffFromMeanStreaming {
    fn default() -> Self {
        Self { radius: 10 }
    }
}

impl surtgis_core::WindowAlgorithm for DiffFromMeanStreaming {
    fn kernel_radius(&self) -> usize {
        self.radius
    }

    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        _cell_size_x: f64,
        _cell_size_y: f64,
    ) {
        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = self.radius;
        let r_i = radius as isize;

        for r in 0..out_rows {
            let ir = r + radius;
            if ir < radius || ir + radius >= in_rows {
                for c in 0..cols {
                    output[[r, c]] = f64::NAN;
                }
                continue;
            }

            for c in 0..cols {
                if c < radius || c + radius >= cols {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let center = input[[ir, c]];
                if center.is_nan()
                    || nodata.map_or(false, |nd| (center - nd).abs() < f64::EPSILON)
                {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let mut sum = 0.0;
                let mut count = 0u32;
                let ci = c as isize;

                for dr in -r_i..=r_i {
                    for dc in -r_i..=r_i {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = (ir as isize + dr) as usize;
                        let nc = (ci + dc) as usize;
                        let nv = input[[nr, nc]];
                        if !nv.is_nan()
                            && nodata.map_or(true, |nd| (nv - nd).abs() >= f64::EPSILON)
                        {
                            sum += nv;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    output[[r, c]] = center - sum / count as f64;
                } else {
                    output[[r, c]] = f64::NAN;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_diff_flat_surface() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = diff_from_mean_elev(&dem, DiffFromMeanParams { radius: 1 }).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(val.abs() < 1e-10, "Flat surface should be 0, got {}", val);
    }

    #[test]
    fn test_diff_peak() {
        let mut dem = Raster::filled(7, 7, 50.0);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        dem.set(3, 3, 100.0).unwrap();

        let result = diff_from_mean_elev(&dem, DiffFromMeanParams { radius: 1 }).unwrap();
        let val = result.get(3, 3).unwrap();
        // center = 100, neighbors all 50, mean = 50, diff = 50
        assert!((val - 50.0).abs() < 1e-10, "Peak diff should be 50, got {}", val);
    }

    #[test]
    fn test_diff_valley() {
        let mut dem = Raster::filled(7, 7, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        dem.set(3, 3, 50.0).unwrap();

        let result = diff_from_mean_elev(&dem, DiffFromMeanParams { radius: 1 }).unwrap();
        let val = result.get(3, 3).unwrap();
        assert!((val - (-50.0)).abs() < 1e-10, "Valley diff should be -50, got {}", val);
    }

    #[test]
    fn test_diff_units_match_dem() {
        // DiffFromMean should be in same units as DEM (not dimensionless like DEV)
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for row in 0..11 {
            for col in 0..11 {
                dem.set(row, col, row as f64 * 10.0).unwrap();
            }
        }

        let result = diff_from_mean_elev(&dem, DiffFromMeanParams { radius: 2 }).unwrap();
        let val = result.get(5, 5).unwrap();
        // Center = 50, but result should be in meters, not dimensionless
        assert!(val.abs() < 1.0, "Slope ramp should have near-zero diff from mean, got {}", val);
    }
}
