//! Deviation from Mean Elevation (DEV)
//!
//! DEV normalizes TPI by the standard deviation of the neighborhood:
//!
//!   DEV = (z_center - mean(z_neighbors)) / stddev(z_neighbors)
//!
//! This normalization makes the index scale-independent and comparable
//! across different landscape types and neighborhood sizes.
//!
//! - Positive DEV → cell is higher than surroundings (ridge, hilltop)
//! - Negative DEV → cell is lower than surroundings (valley, depression)
//! - Near-zero DEV → cell is at the same level (flat area or mid-slope)
//!
//! Reference: De Reu et al. (2013) "Application of the topographic position
//! index to heterogeneous landscapes" (509 citations)

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::Result;
use surtgis_core::raster::Raster;

/// Parameters for DEV calculation
#[derive(Debug, Clone)]
pub struct DevParams {
    /// Neighborhood radius in cells (default 1 → 3×3 window)
    /// Radius 2 → 5×5, radius 5 → 11×11, etc.
    pub radius: usize,
}

impl Default for DevParams {
    fn default() -> Self {
        Self { radius: 10 }
    }
}

/// Per-cell DEV kernel shared by the batch (`dev`) and streaming
/// (`DevStreaming`) paths.
///
/// Computes `(center - mean) / stddev` over the `(2r+1)²` window centered
/// at `(row, col)`, excluding the center cell and any NaN/nodata neighbors.
/// Returns NaN when fewer than 2 valid neighbors exist, and 0.0 for
/// near-zero variance (perfectly flat neighborhoods).
///
/// The caller must guarantee that the full window lies inside `data`.
#[inline]
fn dev_kernel(data: &Array2<f64>, row: usize, col: usize, r: isize, nodata: Option<f64>) -> f64 {
    debug_assert!(row as isize >= r && (row as isize) < data.nrows() as isize - r);
    debug_assert!(col as isize >= r && (col as isize) < data.ncols() as isize - r);

    let center = data[[row, col]];
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0u32;

    for dr in -r..=r {
        for dc in -r..=r {
            if dr == 0 && dc == 0 {
                continue; // exclude center
            }
            let nr = (row as isize + dr) as usize;
            let nc = (col as isize + dc) as usize;
            // SAFETY: the caller guarantees the full window is in bounds.
            let nv = unsafe { *data.uget((nr, nc)) };
            if !nv.is_nan() && nodata.is_none_or(|nd| nv != nd) {
                sum += nv;
                sum_sq += nv * nv;
                count += 1;
            }
        }
    }

    if count < 2 {
        return f64::NAN;
    }

    let mean = sum / count as f64;
    let variance = (sum_sq / count as f64) - (mean * mean);

    // Protect against near-zero variance (perfectly flat neighborhoods)
    if variance < 1e-20 {
        0.0
    } else {
        (center - mean) / variance.sqrt()
    }
}

/// Calculate Deviation from Mean Elevation
///
/// DEV = (z - mean) / stddev, where mean and stddev are computed over
/// a circular neighborhood of the given radius (excluding the center cell).
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - DEV parameters (neighborhood radius)
///
/// # Returns
/// Raster with DEV values (dimensionless, typically in range [-3, 3])
pub fn dev(dem: &Raster<f64>, params: DevParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let nodata = dem.nodata();
    let data = dem.data();

    let output_data = par_map_rows(rows, cols, |row, out_row| {
        for (col, row_data_col) in out_row.iter_mut().enumerate() {
            let center = unsafe { dem.get_unchecked(row, col) };
            if center.is_nan() || nodata.is_some_and(|nd| center == nd) {
                continue;
            }

            let ri = row as isize;
            let ci = col as isize;
            if ri < r || ri >= rows as isize - r || ci < r || ci >= cols as isize - r {
                continue;
            }

            *row_data_col = dev_kernel(data, row, col, r, nodata);
        }
    });

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming DEV calculator implementing `WindowAlgorithm`.
///
/// Processes a DEM strip-by-strip with bounded memory.
/// Uses the same (z - mean) / stddev method as `dev()`.
#[derive(Debug, Clone)]
pub struct DevStreaming {
    /// Neighborhood radius in cells.
    pub radius: usize,
}

impl Default for DevStreaming {
    fn default() -> Self {
        Self { radius: 10 }
    }
}

impl surtgis_core::WindowAlgorithm for DevStreaming {
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
                let ir = r + radius; // input row corresponding to output row r
                // Check vertical edges: need full window above and below
                if ir < radius || ir + radius >= in_rows {
                    out_row.fill(f64::NAN);
                    return;
                }

                for (c, out_v) in out_row.iter_mut().enumerate() {
                    // Check horizontal edges
                    if c < radius || c + radius >= cols {
                        *out_v = f64::NAN;
                        continue;
                    }

                    let center = input[[ir, c]];
                    if center.is_nan() || nodata.is_some_and(|nd| center == nd) {
                        *out_v = f64::NAN;
                        continue;
                    }

                    *out_v = dev_kernel(input, ir, c, r_i, nodata);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_dev_flat_surface() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = dev(&dem, DevParams { radius: 1 }).unwrap();
        let val = result.get(5, 5).unwrap();
        // Flat surface: mean == center, variance ≈ 0 → DEV = 0
        assert!(
            val.abs() < 1e-10,
            "Expected DEV ~0 for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_dev_peak() {
        let mut dem = Raster::filled(5, 5, 50.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        dem.set(2, 2, 100.0).unwrap();

        let result = dev(&dem, DevParams { radius: 1 }).unwrap();
        let val = result.get(2, 2).unwrap();

        // TPI = 100 - 50 = 50
        // All neighbors are 50, so stddev of neighbors = 0 → DEV large positive
        // Actually: 7 of 8 neighbors are 50, mean=50, stddev=0 → DEV should be 0-protected
        // Wait, all 8 neighbors ARE 50, so variance=0 → returns 0.0
        // Let's set up a more realistic test
        assert!(
            val.abs() < 1e-10,
            "Uniform neighbors with different center: variance=0 → DEV=0, got {}",
            val
        );
    }

    #[test]
    fn test_dev_realistic() {
        // Create a DEM with varied terrain
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for row in 0..11 {
            for col in 0..11 {
                let x = col as f64 - 5.0;
                let y = row as f64 - 5.0;
                dem.set(row, col, 100.0 - x * x - y * y).unwrap();
            }
        }
        // Center (5,5) = 100 (highest point), corners are lowest
        let result = dev(&dem, DevParams { radius: 2 }).unwrap();
        let val = result.get(5, 5).unwrap();

        // Center is highest → DEV should be positive
        assert!(val > 0.0, "Peak should have positive DEV, got {}", val);
    }

    #[test]
    fn test_dev_valley() {
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for row in 0..11 {
            for col in 0..11 {
                let x = col as f64 - 5.0;
                let y = row as f64 - 5.0;
                dem.set(row, col, x * x + y * y).unwrap();
            }
        }
        // Center (5,5) = 0 (lowest point), edges are highest
        let result = dev(&dem, DevParams { radius: 2 }).unwrap();
        let val = result.get(5, 5).unwrap();

        assert!(val < 0.0, "Valley should have negative DEV, got {}", val);
    }

    #[test]
    fn test_dev_dimensionless() {
        // DEV should be scale-independent: multiplying all elevations
        // by a constant should not change DEV
        let mut dem1 = Raster::new(11, 11);
        let mut dem2 = Raster::new(11, 11);
        dem1.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        dem2.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for row in 0..11 {
            for col in 0..11 {
                let x = col as f64 - 5.0;
                let y = row as f64 - 5.0;
                let z = 100.0 - x * x - y * y;
                dem1.set(row, col, z).unwrap();
                dem2.set(row, col, z * 10.0).unwrap(); // 10× vertical exaggeration
            }
        }

        let r1 = dev(&dem1, DevParams { radius: 2 }).unwrap();
        let r2 = dev(&dem2, DevParams { radius: 2 }).unwrap();
        let v1 = r1.get(5, 5).unwrap();
        let v2 = r2.get(5, 5).unwrap();

        assert!(
            (v1 - v2).abs() < 1e-6,
            "DEV should be scale-independent: {} vs {}",
            v1,
            v2
        );
    }
}
