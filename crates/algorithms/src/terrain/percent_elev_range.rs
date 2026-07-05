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

use crate::maybe_rayon::*;
use ndarray::Array2;
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

/// Per-cell kernel shared by the batch (`percent_elev_range`) and streaming
/// (`PercentElevRangeStreaming`) paths.
///
/// Computes `(center - min) / (max - min) × 100` over the `(2r+1)²` window
/// centered at `(row, col)` (center cell included), skipping NaN/nodata
/// values. Returns 0.0 when the local range is (near) zero.
///
/// The caller must guarantee that the full window lies inside `data`.
#[inline]
fn percent_elev_range_kernel(
    data: &Array2<f64>,
    row: usize,
    col: usize,
    r: isize,
    nodata: Option<f64>,
) -> f64 {
    debug_assert!(row as isize >= r && (row as isize) < data.nrows() as isize - r);
    debug_assert!(col as isize >= r && (col as isize) < data.ncols() as isize - r);

    let center = data[[row, col]];
    let mut local_min = f64::INFINITY;
    let mut local_max = f64::NEG_INFINITY;

    for dr in -r..=r {
        for dc in -r..=r {
            let nr = (row as isize + dr) as usize;
            let nc = (col as isize + dc) as usize;
            // SAFETY: the caller guarantees the full window is in bounds.
            let nv = unsafe { *data.uget((nr, nc)) };
            if !nv.is_nan() && nodata.is_none_or(|nd| nv != nd) {
                if nv < local_min {
                    local_min = nv;
                }
                if nv > local_max {
                    local_max = nv;
                }
            }
        }
    }

    let range = local_max - local_min;
    if range < f64::EPSILON {
        0.0
    } else {
        (center - local_min) / range * 100.0
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
    let data = dem.data();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { dem.get_unchecked(row, col) };
                if dem.is_nodata(center) {
                    continue;
                }

                let ri = row as isize;
                let ci = col as isize;
                if ri < r || ri >= rows as isize - r || ci < r || ci >= cols as isize - r {
                    continue;
                }

                *out = percent_elev_range_kernel(data, row, col, r, nodata);
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
    /// Neighborhood radius in cells.
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
        let radius = self.radius;
        let r_i = radius as isize;

        output
            .as_slice_mut()
            .expect("process_chunk output must be in standard layout")
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, out_row)| {
                let ir = r + radius;
                if ir < radius || ir + radius >= in_rows {
                    out_row.fill(f64::NAN);
                    return;
                }

                for (c, out_v) in out_row.iter_mut().enumerate() {
                    if c < radius || c + radius >= cols {
                        *out_v = f64::NAN;
                        continue;
                    }

                    let center = input[[ir, c]];
                    if center.is_nan() || nodata.is_some_and(|nd| center == nd) {
                        *out_v = f64::NAN;
                        continue;
                    }

                    *out_v = percent_elev_range_kernel(input, ir, c, r_i, nodata);
                }
            });
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
        assert!(
            (val - 100.0).abs() < 1e-10,
            "Peak should be 100%, got {}",
            val
        );
    }

    #[test]
    fn test_percent_range_valley() {
        let mut dem = Raster::filled(7, 7, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        dem.set(3, 3, 50.0).unwrap();

        let result = percent_elev_range(&dem, PercentElevRangeParams { radius: 1 }).unwrap();
        let val = result.get(3, 3).unwrap();
        // center=50, min=50, max=100, (50-50)/(100-50)*100 = 0%
        assert!(
            (val - 0.0).abs() < 1e-10,
            "Valley should be 0%, got {}",
            val
        );
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
        assert!(
            (val - 50.0).abs() < 1e-10,
            "Midpoint should be 50%, got {}",
            val
        );
    }
}
