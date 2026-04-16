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

use ndarray::Array2;
use crate::maybe_rayon::*;
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

/// Compute average normal vector angular deviation.
pub fn normal_vector_deviation(
    dem: &Raster<f64>,
    params: NormalDeviationParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let nodata = dem.nodata();

    let cell_size = dem.cell_size();

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
                let border = r + 1; // need 3x3 stencil at window edges
                if ri < border || ri >= rows as isize - border
                    || ci < border || ci >= cols as isize - border
                {
                    continue;
                }

                // Collect normals in window
                let mut normals: Vec<[f64; 3]> = Vec::new();
                let mut mean_nx = 0.0;
                let mut mean_ny = 0.0;
                let mut mean_nz = 0.0;

                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = (ri + dr) as usize;
                        let nc = (ci + dc) as usize;

                        let nv = unsafe { dem.get_unchecked(nr, nc) };
                        if nv.is_nan()
                            || nodata.is_some_and(|nd| (nv - nd).abs() < f64::EPSILON)
                        {
                            continue;
                        }
                        if nr == 0 || nr >= rows - 1 || nc == 0 || nc >= cols - 1 {
                            continue;
                        }

                        // Horn's method with cell_size
                        let z = |r: usize, c: usize| unsafe { dem.get_unchecked(r, c) };
                        let dzdx = (
                            z(nr - 1, nc + 1) + 2.0 * z(nr, nc + 1) + z(nr + 1, nc + 1)
                            - z(nr - 1, nc - 1) - 2.0 * z(nr, nc - 1) - z(nr + 1, nc - 1)
                        ) / (8.0 * cell_size);
                        let dzdy = (
                            z(nr + 1, nc - 1) + 2.0 * z(nr + 1, nc) + z(nr + 1, nc + 1)
                            - z(nr - 1, nc - 1) - 2.0 * z(nr - 1, nc) - z(nr - 1, nc + 1)
                        ) / (8.0 * cell_size);

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
                    continue;
                }

                // Normalize mean vector
                let n = count as f64;
                mean_nx /= n;
                mean_ny /= n;
                mean_nz /= n;
                let mean_len = (mean_nx * mean_nx + mean_ny * mean_ny + mean_nz * mean_nz).sqrt();
                if mean_len < 1e-10 {
                    continue;
                }
                mean_nx /= mean_len;
                mean_ny /= mean_len;
                mean_nz /= mean_len;

                // Mean angular deviation
                let mut sum_angle = 0.0;
                for normal in &normals {
                    let dot = (normal[0] * mean_nx + normal[1] * mean_ny + normal[2] * mean_nz)
                        .clamp(-1.0, 1.0);
                    sum_angle += dot.acos();
                }

                *out = (sum_angle / count as f64).to_degrees();
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
        _cell_size_x: f64,
        _cell_size_y: f64,
    ) {
        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let kr = self.radius + 1;
        let r = self.radius as isize;

        for row in 0..out_rows {
            let ir = row + kr;
            if ir < kr || ir + kr >= in_rows {
                for c in 0..cols {
                    output[[row, c]] = f64::NAN;
                }
                continue;
            }

            for c in 0..cols {
                if c < kr || c + kr >= cols {
                    output[[row, c]] = f64::NAN;
                    continue;
                }

                let center = input[[ir, c]];
                if center.is_nan()
                    || nodata.map_or(false, |nd| (center - nd).abs() < f64::EPSILON)
                {
                    output[[row, c]] = f64::NAN;
                    continue;
                }

                let mut normals: Vec<[f64; 3]> = Vec::new();
                let mut mean_nx = 0.0;
                let mut mean_ny = 0.0;
                let mut mean_nz = 0.0;

                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = (ir as isize + dr) as usize;
                        let nc = (c as isize + dc) as usize;

                        let nv = input[[nr, nc]];
                        if nv.is_nan()
                            || nodata.map_or(false, |nd| (nv - nd).abs() < f64::EPSILON)
                        {
                            continue;
                        }
                        if nr == 0 || nr + 1 >= in_rows || nc == 0 || nc + 1 >= cols {
                            continue;
                        }

                        let dzdx = (
                            input[[nr - 1, nc + 1]] + 2.0 * input[[nr, nc + 1]] + input[[nr + 1, nc + 1]]
                            - input[[nr - 1, nc - 1]] - 2.0 * input[[nr, nc - 1]] - input[[nr + 1, nc - 1]]
                        ) / 8.0;
                        let dzdy = (
                            input[[nr + 1, nc - 1]] + 2.0 * input[[nr + 1, nc]] + input[[nr + 1, nc + 1]]
                            - input[[nr - 1, nc - 1]] - 2.0 * input[[nr - 1, nc]] - input[[nr - 1, nc + 1]]
                        ) / 8.0;

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
                    output[[row, c]] = f64::NAN;
                    continue;
                }

                let n = count as f64;
                mean_nx /= n;
                mean_ny /= n;
                mean_nz /= n;
                let mean_len = (mean_nx * mean_nx + mean_ny * mean_ny + mean_nz * mean_nz).sqrt();
                if mean_len < 1e-10 {
                    output[[row, c]] = f64::NAN;
                    continue;
                }
                mean_nx /= mean_len;
                mean_ny /= mean_len;
                mean_nz /= mean_len;

                let mut sum_angle = 0.0;
                for normal in &normals {
                    let dot = (normal[0] * mean_nx + normal[1] * mean_ny + normal[2] * mean_nz)
                        .clamp(-1.0, 1.0);
                    sum_angle += dot.acos();
                }

                output[[row, c]] = (sum_angle / count as f64).to_degrees();
            }
        }
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
        assert!(v < 1.0, "Uniform slope should have low deviation, got {}", v);
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
