//! Average Normal Vector Angular Deviation
//!
//! Measures the mean angular deviation of surface normal vectors from
//! their local mean in a focal window:
//!
//! 1. For each pixel in window, compute surface normal n = (-dz/dx, -dz/dy, 1) normalized
//! 2. Compute mean normal n̄ in the window
//! 3. Deviation = mean(arccos(n · n̄)) in degrees
//!
//! High values indicate rough, variable terrain surfaces.
//!
//! Reference: WhiteboxTools `AverageNormalVectorAngularDeviation`

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for normal vector angular deviation.
#[derive(Debug, Clone)]
pub struct NormalDeviationParams {
    /// Neighborhood radius in cells (default 3 → 7×7 window).
    pub radius: usize,
}

impl Default for NormalDeviationParams {
    fn default() -> Self {
        Self { radius: 3 }
    }
}

/// Per-cell kernel shared by the batch (`normal_vector_deviation`) and
/// streaming (`NormalDeviationStreaming`) paths.
///
/// Computes the mean angular deviation (degrees) of the unit surface
/// normals (Horn 1981 gradients scaled by `8 · cell_size`) of every valid
/// pixel in the `(2r+1)²` window centered at `(row, col)` from their mean
/// normal. NaN/nodata pixels and pixels whose 3×3 Horn stencil would leave
/// `data` are skipped. Returns NaN when fewer than 2 normals are available
/// or the mean normal degenerates.
///
/// The caller must guarantee that the full window plus a 1-cell stencil
/// margin lies inside `data` (border of `r + 1`).
#[inline]
fn normal_deviation_kernel(
    data: &Array2<f64>,
    row: usize,
    col: usize,
    r: isize,
    cell_size: f64,
    nodata: Option<f64>,
) -> f64 {
    let (rows, cols) = data.dim();

    // Collect normals in window
    let mut normals: Vec<[f64; 3]> = Vec::new();
    let mut mean_nx = 0.0;
    let mut mean_ny = 0.0;
    let mut mean_nz = 0.0;

    for dr in -r..=r {
        for dc in -r..=r {
            let nr = (row as isize + dr) as usize;
            let nc = (col as isize + dc) as usize;

            let nv = data[[nr, nc]];
            if nv.is_nan() || nodata.is_some_and(|nd| (nv - nd).abs() < f64::EPSILON) {
                continue;
            }
            if nr == 0 || nr >= rows - 1 || nc == 0 || nc >= cols - 1 {
                continue;
            }

            // Horn's method with cell_size
            let dzdx = (data[[nr - 1, nc + 1]] + 2.0 * data[[nr, nc + 1]] + data[[nr + 1, nc + 1]]
                - data[[nr - 1, nc - 1]]
                - 2.0 * data[[nr, nc - 1]]
                - data[[nr + 1, nc - 1]])
                / (8.0 * cell_size);
            let dzdy = (data[[nr + 1, nc - 1]] + 2.0 * data[[nr + 1, nc]] + data[[nr + 1, nc + 1]]
                - data[[nr - 1, nc - 1]]
                - 2.0 * data[[nr - 1, nc]]
                - data[[nr - 1, nc + 1]])
                / (8.0 * cell_size);

            // Normal: (-dz/dx, -dz/dy, 1) normalized
            let nx = -dzdx;
            let ny = -dzdy;
            let nz = 1.0;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            let nx = nx / len;
            let ny = ny / len;
            let nz = nz / len;

            mean_nx += nx;
            mean_ny += ny;
            mean_nz += nz;
            normals.push([nx, ny, nz]);
        }
    }

    let count = normals.len();
    if count < 2 {
        return f64::NAN;
    }

    // Normalize mean vector
    let n = count as f64;
    mean_nx /= n;
    mean_ny /= n;
    mean_nz /= n;
    let mean_len = (mean_nx * mean_nx + mean_ny * mean_ny + mean_nz * mean_nz).sqrt();
    if mean_len < 1e-10 {
        return f64::NAN;
    }
    mean_nx /= mean_len;
    mean_ny /= mean_len;
    mean_nz /= mean_len;

    // Mean angular deviation
    let mut sum_angle = 0.0;
    for normal in &normals {
        let dot =
            (normal[0] * mean_nx + normal[1] * mean_ny + normal[2] * mean_nz).clamp(-1.0, 1.0);
        sum_angle += dot.acos();
    }

    (sum_angle / count as f64).to_degrees()
}

/// Compute average normal vector angular deviation.
pub fn normal_vector_deviation(
    dem: &Raster<f64>,
    params: NormalDeviationParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let nodata = dem.nodata();
    let data = dem.data();

    let cell_size = dem.cell_size();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { dem.get_unchecked(row, col) };
                if center.is_nan() || nodata.is_some_and(|nd| (center - nd).abs() < f64::EPSILON) {
                    continue;
                }

                let ri = row as isize;
                let ci = col as isize;
                let border = r + 1; // need 3x3 stencil at window edges
                if ri < border
                    || ri >= rows as isize - border
                    || ci < border
                    || ci >= cols as isize - border
                {
                    continue;
                }

                *out = normal_deviation_kernel(data, row, col, r, cell_size, nodata);
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

/// Streaming normal vector angular deviation implementing `WindowAlgorithm`.
#[derive(Debug, Clone)]
pub struct NormalDeviationStreaming {
    /// Neighborhood radius in cells.
    pub radius: usize,
}

impl surtgis_core::WindowAlgorithm for NormalDeviationStreaming {
    fn kernel_radius(&self) -> usize {
        self.radius + 1
    }

    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        cell_size_x: f64,
        _cell_size_y: f64,
    ) {
        let (in_rows, cols) = input.dim();
        let kr = self.radius + 1;
        let r = self.radius as isize;

        output
            .as_slice_mut()
            .expect("process_chunk output must be in standard layout")
            .par_chunks_mut(cols)
            .enumerate()
            .for_each(|(row, out_row)| {
                let ir = row + kr;
                if ir < kr || ir + kr >= in_rows {
                    out_row.fill(f64::NAN);
                    return;
                }

                for (c, out_v) in out_row.iter_mut().enumerate() {
                    if c < kr || c + kr >= cols {
                        *out_v = f64::NAN;
                        continue;
                    }

                    let center = input[[ir, c]];
                    if center.is_nan()
                        || nodata.is_some_and(|nd| (center - nd).abs() < f64::EPSILON)
                    {
                        *out_v = f64::NAN;
                        continue;
                    }

                    *out_v = normal_deviation_kernel(input, ir, c, r, cell_size_x, nodata);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_flat_zero_deviation() {
        let mut dem = Raster::filled(15, 15, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));

        let result = normal_vector_deviation(&dem, NormalDeviationParams { radius: 2 }).unwrap();
        let v = result.get(7, 7).unwrap();
        if !v.is_nan() {
            assert!(v < 0.01, "Flat surface should have ~0 deviation, got {}", v);
        }
    }

    #[test]
    fn test_uniform_slope_low_deviation() {
        let mut dem = Raster::new(15, 15);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));
        for r in 0..15 {
            for c in 0..15 {
                dem.set(r, c, r as f64 * 10.0 + c as f64 * 5.0).unwrap();
            }
        }

        let result = normal_vector_deviation(&dem, NormalDeviationParams { radius: 2 }).unwrap();
        let v = result.get(7, 7).unwrap();
        assert!(
            v < 1.0,
            "Uniform slope should have low deviation, got {}",
            v
        );
    }

    #[test]
    fn test_bowl_higher_deviation() {
        // Bowl: normals radiate outward in all directions → high deviation
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                let x = c as f64 - 10.0;
                let y = r as f64 - 10.0;
                dem.set(r, c, x * x + y * y).unwrap();
            }
        }

        let result = normal_vector_deviation(&dem, NormalDeviationParams { radius: 4 }).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!(v > 1.0, "Bowl center should have high deviation, got {}", v);
    }
}
