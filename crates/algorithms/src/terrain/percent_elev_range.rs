//! Percent Elevation Range
//!
//! Computes the relative position of a cell's elevation within the
//! local (focal) elevation range:
//!
//!   output = (z - z_min_window) / (z_max_window - z_min_window) × 100
//!
//! - 0% → cell is at the local minimum (valley bottom)
//! - 100% → cell is at the local maximum (ridge top)
//!
//! Similar to `relative_slope_position` but computed directly from
//! a focal window on the DEM, without requiring HAND.
//!
//! Reference: WhiteboxTools `PercentElevRange`

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for percent elevation range.
#[derive(Debug, Clone)]
pub struct PercentElevRangeParams {
    /// Neighborhood radius in cells (default 10 → 21×21 window).
    pub radius: usize,
}

impl Default for PercentElevRangeParams {
    fn default() -> Self {
        Self { radius: 10 }
    }
}

/// Compute percent elevation range.
pub fn percent_elev_range(
    dem: &Raster<f64>,
    params: PercentElevRangeParams,
) -> Result<Raster<f64>> {
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

                let mut local_min = f64::INFINITY;
                let mut local_max = f64::NEG_INFINITY;

                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = (ri + dr) as usize;
                        let nc = (ci + dc) as usize;
                        let nv = unsafe { dem.get_unchecked(nr, nc) };
                        if !nv.is_nan()
                            && nodata.is_none_or(|nd| (nv - nd).abs() >= f64::EPSILON)
                        {
                            if nv < local_min { local_min = nv; }
                            if nv > local_max { local_max = nv; }
                        }
                    }
                }

                let range = local_max - local_min;
                if range < f64::EPSILON {
                    *out = 0.0;
                } else {
                    *out = (center - local_min) / range * 100.0;
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

/// Streaming percent elevation range implementing `WindowAlgorithm`.
#[derive(Debug, Clone)]
pub struct PercentElevRangeStreaming {
    pub radius: usize,
}

impl Default for PercentElevRangeStreaming {
    fn default() -> Self {
        Self { radius: 10 }
    }
}

impl surtgis_core::WindowAlgorithm for PercentElevRangeStreaming {
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

                let mut local_min = f64::INFINITY;
                let mut local_max = f64::NEG_INFINITY;
                let ci = c as isize;

                for dr in -r_i..=r_i {
                    for dc in -r_i..=r_i {
                        let nr = (ir as isize + dr) as usize;
                        let nc = (ci + dc) as usize;
                        let nv = input[[nr, nc]];
                        if !nv.is_nan()
                            && nodata.map_or(true, |nd| (nv - nd).abs() >= f64::EPSILON)
                        {
                            if nv < local_min { local_min = nv; }
                            if nv > local_max { local_max = nv; }
                        }
                    }
                }

                let range = local_max - local_min;
                if range < f64::EPSILON {
                    output[[r, c]] = 0.0;
                } else {
                    output[[r, c]] = (center - local_min) / range * 100.0;
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
    fn test_percent_range_flat() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = percent_elev_range(&dem, PercentElevRangeParams { radius: 1 }).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!((val - 0.0).abs() < 1e-10, "Flat should be 0%, got {}", val);
    }

    #[test]
    fn test_percent_range_peak() {
        let mut dem = Raster::filled(7, 7, 50.0);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        dem.set(3, 3, 100.0).unwrap();

        let result = percent_elev_range(&dem, PercentElevRangeParams { radius: 1 }).unwrap();
        let val = result.get(3, 3).unwrap();
        // center=100, min=50, max=100, (100-50)/(100-50)*100 = 100%
        assert!((val - 100.0).abs() < 1e-10, "Peak should be 100%, got {}", val);
    }

    #[test]
    fn test_percent_range_valley() {
        let mut dem = Raster::filled(7, 7, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        dem.set(3, 3, 50.0).unwrap();

        let result = percent_elev_range(&dem, PercentElevRangeParams { radius: 1 }).unwrap();
        let val = result.get(3, 3).unwrap();
        // center=50, min=50, max=100, (50-50)/(100-50)*100 = 0%
        assert!((val - 0.0).abs() < 1e-10, "Valley should be 0%, got {}", val);
    }

    #[test]
    fn test_percent_range_midpoint() {
        let mut dem = Raster::new(7, 7);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        // Row-based ramp: 0,10,20,...,60
        for r in 0..7 {
            for c in 0..7 {
                dem.set(r, c, r as f64 * 10.0).unwrap();
            }
        }

        let result = percent_elev_range(&dem, PercentElevRangeParams { radius: 2 }).unwrap();
        let val = result.get(3, 3).unwrap();
        // center=30, min=10, max=50, (30-10)/(50-10)*100 = 50%
        assert!((val - 50.0).abs() < 1e-10, "Midpoint should be 50%, got {}", val);
    }
}
