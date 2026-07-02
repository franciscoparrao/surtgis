//! Convergence Index
//!
//! Measures convergence/divergence of overland flow.
//! Similar to plan curvature but smoother. Negative values indicate
//! convergent flow (valleys), positive values indicate divergent flow (ridges).

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for convergence index
#[derive(Debug, Clone)]
pub struct ConvergenceParams {
    /// Neighborhood radius in cells (default 1)
    pub radius: usize,
}

impl Default for ConvergenceParams {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

/// Compute Convergence Index
///
/// For each cell, computes the average deviation of aspect directions
/// of surrounding cells from the direction toward the center cell.
/// Values range from -100 (pure convergence) to +100 (pure divergence).
///
/// Based on Kiss (2004) / Köthe & Lehmeier (1996).
///
/// # Arguments
/// * `dem` - Input DEM
/// * `params` - Neighborhood radius
pub fn convergence_index(dem: &Raster<f64>, params: ConvergenceParams) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Radius must be > 0".into()));
    }

    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let data = dem.data();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let z0 = data[[row, col]];
                if z0.is_nan() {
                    continue;
                }

                *row_data_col = convergence_kernel(data, row, col, r);
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

/// Compute aspect at a single cell (degrees, 0=N, clockwise) using Horn's
/// method. Returns NaN for edge cells, NaN stencils, or flat areas.
///
/// Shared by the batch (`convergence_index`) and streaming
/// (`ConvergenceStreaming`) paths.
fn compute_aspect_at(data: &Array2<f64>, row: usize, col: usize) -> f64 {
    let (rows, cols) = data.dim();
    if row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1 {
        return f64::NAN;
    }

    let a = data[[row - 1, col - 1]];
    let b = data[[row - 1, col]];
    let c = data[[row - 1, col + 1]];
    let d = data[[row, col - 1]];
    let f = data[[row, col + 1]];
    let g = data[[row + 1, col - 1]];
    let h = data[[row + 1, col]];
    let i = data[[row + 1, col + 1]];

    if [a, b, c, d, f, g, h, i].iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }

    let dz_dx = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) / 8.0;
    let dz_dy = ((g + 2.0 * h + i) - (a + 2.0 * b + c)) / 8.0;

    if dz_dx.abs() < f64::EPSILON && dz_dy.abs() < f64::EPSILON {
        return f64::NAN; // Flat, no aspect
    }

    let mut aspect = (-dz_dx).atan2(dz_dy).to_degrees();
    if aspect < 0.0 {
        aspect += 360.0;
    }
    aspect
}

/// Per-cell convergence index kernel shared by the batch
/// (`convergence_index`) and streaming (`ConvergenceStreaming`) paths.
///
/// For each neighbor in the `(2r+1)²` window (center excluded), computes
/// the deviation between the neighbor's aspect and the direction toward
/// the center cell, then scales the mean deviation to [-100, 100].
/// Neighbors outside `data` or without a defined aspect are skipped.
/// Returns NaN when no neighbor has a defined aspect.
#[inline]
fn convergence_kernel(data: &Array2<f64>, row: usize, col: usize, r: isize) -> f64 {
    let (rows, cols) = data.dim();
    let mut sum_deviation = 0.0;
    let mut count = 0;

    for dr in -r..=r {
        for dc in -r..=r {
            if dr == 0 && dc == 0 {
                continue;
            }

            let nr = row as isize + dr;
            let nc = col as isize + dc;

            if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                continue;
            }

            // Compute aspect at neighbor
            let aspect_n = compute_aspect_at(data, nr as usize, nc as usize);
            if aspect_n.is_nan() {
                continue;
            }

            // Direction from neighbor toward center (azimuth from north)
            // dr,dc are offsets from center to neighbor; reverse and convert
            // East component = -dc, North component = dr (row goes south = -north)
            let dir_to_center = (-(dc as f64)).atan2(dr as f64);
            let dir_deg = dir_to_center.to_degrees();
            let dir_deg = if dir_deg < 0.0 {
                dir_deg + 360.0
            } else {
                dir_deg
            };

            // Angular difference
            let mut diff = (aspect_n - dir_deg).abs();
            if diff > 180.0 {
                diff = 360.0 - diff;
            }

            // Convergence: aspect points toward center (diff near 0)
            // Divergence: aspect points away (diff near 180)
            sum_deviation += diff;
            count += 1;
        }
    }

    if count > 0 {
        let avg_deviation = sum_deviation / count as f64;
        // Scale to [-100, 100]: 0° → -100 (convergent), 90° → 0, 180° → +100 (divergent)
        (avg_deviation / 90.0 - 1.0) * 100.0
    } else {
        f64::NAN
    }
}

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming Convergence Index calculator implementing `WindowAlgorithm`.
///
/// Processes a DEM strip-by-strip with bounded memory.
/// Uses the same Kiss (2004) / Köthe & Lehmeier (1996) method as
/// `convergence_index()`.
///
/// The kernel radius is `radius + 1` because aspect computation at each
/// neighbor requires a 3x3 Horn window around that neighbor.
#[derive(Debug, Clone)]
pub struct ConvergenceStreaming {
    /// Neighborhood radius in cells.
    pub radius: usize,
}

impl Default for ConvergenceStreaming {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl surtgis_core::WindowAlgorithm for ConvergenceStreaming {
    fn kernel_radius(&self) -> usize {
        // Need radius for neighborhood + 1 for Horn's method at border neighbors
        self.radius + 1
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
        let kr = self.radius + 1; // kernel radius (the value returned by kernel_radius)
        let r = self.radius as isize;

        output
            .as_slice_mut()
            .expect("process_chunk output must be in standard layout")
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row_out, out_row)| {
                let ir = row_out + kr; // input row corresponding to output row

                // Check vertical edges: need full kernel above and below
                if ir < kr || ir + kr >= in_rows {
                    out_row.fill(f64::NAN);
                    return;
                }

                for (c, out_v) in out_row.iter_mut().enumerate() {
                    // Check horizontal edges
                    if c < kr || c + kr >= cols {
                        *out_v = f64::NAN;
                        continue;
                    }

                    let z0 = input[[ir, c]];
                    if z0.is_nan() || nodata.is_some_and(|nd| (z0 - nd).abs() < f64::EPSILON) {
                        *out_v = f64::NAN;
                        continue;
                    }

                    *out_v = convergence_kernel(input, ir, c, r);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_convergence_pit() {
        // Bowl shape: all aspects point inward → convergent
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for row in 0..11 {
            for col in 0..11 {
                let dx = col as f64 - 5.0;
                let dy = row as f64 - 5.0;
                dem.set(row, col, dx * dx + dy * dy).unwrap();
            }
        }

        let result = convergence_index(&dem, ConvergenceParams::default()).unwrap();
        let center = result.get(5, 5).unwrap();
        assert!(
            center < 0.0,
            "Bowl center should be convergent, got {}",
            center
        );
    }

    #[test]
    fn test_convergence_peak() {
        // Peak: all aspects point outward → divergent
        let mut dem = Raster::new(11, 11);
        dem.set_transform(GeoTransform::new(0.0, 11.0, 1.0, -1.0));
        for row in 0..11 {
            for col in 0..11 {
                let dx = col as f64 - 5.0;
                let dy = row as f64 - 5.0;
                dem.set(row, col, 100.0 - (dx * dx + dy * dy)).unwrap();
            }
        }

        let result = convergence_index(&dem, ConvergenceParams::default()).unwrap();
        let center = result.get(5, 5).unwrap();
        assert!(
            center > 0.0,
            "Peak center should be divergent, got {}",
            center
        );
    }

    #[test]
    fn test_convergence_radius_zero() {
        let dem = Raster::filled(5, 5, 100.0_f64);
        assert!(convergence_index(&dem, ConvergenceParams { radius: 0 }).is_err());
    }
}
