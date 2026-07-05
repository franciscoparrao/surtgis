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

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::Result;
use surtgis_core::raster::Raster;

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

/// Per-cell kernel shared by the batch (`spherical_std_dev`) and streaming
/// (`SphericalStdDevStreaming`) paths.
///
/// Computes `sqrt(-2 · ln(R̄))` from the unit surface normals (Horn 1981
/// gradients scaled by `8 · cell_size`) of every valid pixel in the
/// `(2r+1)²` window centered at `(row, col)`. NaN/nodata pixels and pixels
/// whose 3×3 Horn stencil would leave `data` are skipped. Returns NaN when
/// fewer than 2 normals are available, 0.0 for perfect alignment and π for
/// complete dispersion.
///
/// The caller must guarantee that the full window plus a 1-cell stencil
/// margin lies inside `data` (border of `r + 1`).
#[inline]
fn spherical_std_dev_kernel(
    data: &Array2<f64>,
    row: usize,
    col: usize,
    r: isize,
    cell_size: f64,
    nodata: Option<f64>,
) -> f64 {
    let (rows, cols) = data.dim();
    let mut sum_nx = 0.0;
    let mut sum_ny = 0.0;
    let mut sum_nz = 0.0;
    let mut count = 0u32;

    for dr in -r..=r {
        for dc in -r..=r {
            let nr = (row as isize + dr) as usize;
            let nc = (col as isize + dc) as usize;

            let nv = data[[nr, nc]];
            if nv.is_nan() || nodata.is_some_and(|nd| nv == nd) {
                continue;
            }
            if nr == 0 || nr >= rows - 1 || nc == 0 || nc >= cols - 1 {
                continue;
            }

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
        return f64::NAN;
    }

    let n = count as f64;
    let r_bar = ((sum_nx / n).powi(2) + (sum_ny / n).powi(2) + (sum_nz / n).powi(2)).sqrt();

    if r_bar >= 1.0 {
        // Perfect alignment
        0.0
    } else if r_bar < 1e-10 {
        // Complete dispersion
        std::f64::consts::PI // max theoretical value
    } else {
        (-2.0 * r_bar.ln()).sqrt()
    }
}

/// Compute spherical standard deviation of surface normals.
pub fn spherical_std_dev(dem: &Raster<f64>, params: SphericalStdDevParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let nodata = dem.nodata();
    let data = dem.data();

    let cell_size = dem.cell_size();

    let output_data = par_map_rows(rows, cols, |row, out_row| {
        for (col, out) in out_row.iter_mut().enumerate() {
            let center = unsafe { dem.get_unchecked(row, col) };
            if center.is_nan() || nodata.is_some_and(|nd| center == nd) {
                continue;
            }

            let ri = row as isize;
            let ci = col as isize;
            let border = r + 1;
            if ri < border
                || ri >= rows as isize - border
                || ci < border
                || ci >= cols as isize - border
            {
                continue;
            }

            *out = spherical_std_dev_kernel(data, row, col, r, cell_size, nodata);
        }
    });

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = output_data;

    Ok(output)
}

// ─── Streaming implementation ──────────────────────────────────────────

/// Streaming spherical standard deviation implementing `WindowAlgorithm`.
#[derive(Debug, Clone)]
pub struct SphericalStdDevStreaming {
    /// Neighborhood radius in cells.
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
                    if center.is_nan() || nodata.is_some_and(|nd| center == nd) {
                        *out_v = f64::NAN;
                        continue;
                    }

                    *out_v = spherical_std_dev_kernel(input, ir, c, r, cell_size_x, nodata);
                }
            });
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
