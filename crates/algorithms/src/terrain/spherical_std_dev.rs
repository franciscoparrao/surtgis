//! Spherical Standard Deviation of Surface Normals
//!
//! Measures the dispersion of surface normal vectors in a focal window
//! using spherical statistics:
//!
//!   σ_s = sqrt(-2 × ln(R̄))
//!
//! where R̄ is the mean resultant length of the unit normal vectors.
//! This is a monotonic transformation of VRM (Vector Ruggedness Measure)
//! into angular units.
//!
//! - 0 = perfectly uniform normals (flat surface)
//! - Higher values = more dispersed normals (rougher terrain)
//!
//! Reference: WhiteboxTools `SphericalStdDevOfNormals`
//! Fisher, Lewis & Embleton (1987) Statistical Analysis of Spherical Data

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for spherical standard deviation.
#[derive(Debug, Clone)]
pub struct SphericalStdDevParams {
    /// Neighborhood radius in cells (default 3 → 7×7 window).
    pub radius: usize,
}

impl Default for SphericalStdDevParams {
    fn default() -> Self {
        Self { radius: 3 }
    }
}

/// Compute spherical standard deviation of surface normals.
pub fn spherical_std_dev(
    dem: &Raster<f64>,
    params: SphericalStdDevParams,
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
                let border = r + 1;
                if ri < border || ri >= rows as isize - border
                    || ci < border || ci >= cols as isize - border
                {
                    continue;
                }

                let mut sum_nx = 0.0;
                let mut sum_ny = 0.0;
                let mut sum_nz = 0.0;
                let mut count = 0u32;

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

                        let z = |r: usize, c: usize| unsafe { dem.get_unchecked(r, c) };
                        let dzdx = (
                            z(nr - 1, nc + 1) + 2.0 * z(nr, nc + 1) + z(nr + 1, nc + 1)
                            - z(nr - 1, nc - 1) - 2.0 * z(nr, nc - 1) - z(nr + 1, nc - 1)
                        ) / (8.0 * cell_size);
                        let dzdy = (
                            z(nr + 1, nc - 1) + 2.0 * z(nr + 1, nc) + z(nr + 1, nc + 1)
                            - z(nr - 1, nc - 1) - 2.0 * z(nr - 1, nc) - z(nr - 1, nc + 1)
                        ) / (8.0 * cell_size);

                        let nx = -dzdx;
                        let ny = -dzdy;
                        let nz = 1.0;
                        let len = (nx * nx + ny * ny + nz * nz).sqrt();

                        sum_nx += nx / len;
                        sum_ny += ny / len;
                        sum_nz += nz / len;
                        count += 1;
                    }
                }

                if count < 2 {
                    continue;
                }

                let n = count as f64;
                let r_bar = ((sum_nx / n).powi(2) + (sum_ny / n).powi(2) + (sum_nz / n).powi(2)).sqrt();

                if r_bar >= 1.0 {
                    // Perfect alignment
                    *out = 0.0;
                } else if r_bar < 1e-10 {
                    // Complete dispersion
                    *out = std::f64::consts::PI; // max theoretical value
                } else {
                    *out = (-2.0 * r_bar.ln()).sqrt();
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

/// Streaming spherical standard deviation implementing `WindowAlgorithm`.
#[derive(Debug, Clone)]
pub struct SphericalStdDevStreaming {
    pub radius: usize,
}

impl surtgis_core::WindowAlgorithm for SphericalStdDevStreaming {
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

                let mut sum_nx = 0.0;
                let mut sum_ny = 0.0;
                let mut sum_nz = 0.0;
                let mut count = 0u32;

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

                        sum_nx += nx / len;
                        sum_ny += ny / len;
                        sum_nz += nz / len;
                        count += 1;
                    }
                }

                if count < 2 {
                    output[[row, c]] = f64::NAN;
                    continue;
                }

                let n = count as f64;
                let r_bar = ((sum_nx / n).powi(2) + (sum_ny / n).powi(2) + (sum_nz / n).powi(2)).sqrt();

                if r_bar >= 1.0 {
                    output[[row, c]] = 0.0;
                } else if r_bar < 1e-10 {
                    output[[row, c]] = std::f64::consts::PI;
                } else {
                    output[[row, c]] = (-2.0 * r_bar.ln()).sqrt();
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
    fn test_flat_zero() {
        let mut dem = Raster::filled(15, 15, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));

        let result = spherical_std_dev(&dem, SphericalStdDevParams { radius: 2 }).unwrap();
        let v = result.get(7, 7).unwrap();
        if !v.is_nan() {
            assert!(v < 0.01, "Flat surface should have σ ≈ 0, got {}", v);
        }
    }

    #[test]
    fn test_uniform_slope_low() {
        let mut dem = Raster::new(15, 15);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));
        for r in 0..15 {
            for c in 0..15 {
                dem.set(r, c, r as f64 * 10.0).unwrap();
            }
        }

        let result = spherical_std_dev(&dem, SphericalStdDevParams { radius: 2 }).unwrap();
        let v = result.get(7, 7).unwrap();
        assert!(v < 0.1, "Uniform slope should have low σ, got {}", v);
    }

    #[test]
    fn test_non_negative() {
        let mut dem = Raster::new(15, 15);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));
        for r in 0..15 {
            for c in 0..15 {
                let x = c as f64 - 7.0;
                let y = r as f64 - 7.0;
                dem.set(r, c, x * x + y * y).unwrap();
            }
        }

        let result = spherical_std_dev(&dem, SphericalStdDevParams { radius: 2 }).unwrap();
        for r in 4..11 {
            for c in 4..11 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v >= 0.0, "σ should be >= 0, got {} at ({},{})", v, r, c);
                }
            }
        }
    }
}
