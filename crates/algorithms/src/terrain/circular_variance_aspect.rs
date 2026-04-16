//! Circular Variance of Aspect
//!
//! Measures the dispersion of aspect directions within a focal window
//! using circular statistics:
//!
//!   CV = 1 - R̄
//!
//! where R̄ = sqrt(mean(cos(a))² + mean(sin(a))²) is the mean resultant
//! length of the aspect angles.
//!
//! - 0 = uniform aspect (all pixels face the same direction)
//! - 1 = maximum dispersion (aspects point in all directions)
//!
//! Reference: Mardia (1972) circular statistics.
//! WhiteboxTools `CircularVarianceOfAspect`

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for circular variance of aspect.
#[derive(Debug, Clone)]
pub struct CircularVarianceParams {
    /// Neighborhood radius in cells (default 3 → 7×7 window).
    pub radius: usize,
}

impl Default for CircularVarianceParams {
    fn default() -> Self {
        Self { radius: 3 }
    }
}

/// Compute circular variance of aspect.
///
/// Internally computes aspect via the Horn (1981) method from the DEM
/// for each pixel in the focal window, then applies circular statistics.
pub fn circular_variance_aspect(
    dem: &Raster<f64>,
    params: CircularVarianceParams,
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
                // Need radius + 1 for computing aspect (3x3 stencil) at window edges
                let border = r + 1;
                if ri < border || ri >= rows as isize - border
                    || ci < border || ci >= cols as isize - border
                {
                    continue;
                }

                let mut sum_cos = 0.0;
                let mut sum_sin = 0.0;
                let mut count = 0u32;

                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = (ri + dr) as usize;
                        let nc = (ci + dc) as usize;

                        // Compute aspect at (nr, nc) using Horn's 3x3 stencil
                        let nv = unsafe { dem.get_unchecked(nr, nc) };
                        if nv.is_nan()
                            || nodata.is_some_and(|nd| (nv - nd).abs() < f64::EPSILON)
                        {
                            continue;
                        }

                        // Check that 3x3 stencil around (nr,nc) is valid
                        if nr == 0 || nr >= rows - 1 || nc == 0 || nc >= cols - 1 {
                            continue;
                        }

                        // Horn's method for dz/dx and dz/dy (cell_size cancels for aspect)
                        let z = |r: usize, c: usize| unsafe { dem.get_unchecked(r, c) };
                        let dzdx = (
                            z(nr - 1, nc + 1) + 2.0 * z(nr, nc + 1) + z(nr + 1, nc + 1)
                            - z(nr - 1, nc - 1) - 2.0 * z(nr, nc - 1) - z(nr + 1, nc - 1)
                        );
                        let dzdy = (
                            z(nr + 1, nc - 1) + 2.0 * z(nr + 1, nc) + z(nr + 1, nc + 1)
                            - z(nr - 1, nc - 1) - 2.0 * z(nr - 1, nc) - z(nr - 1, nc + 1)
                        );

                        // Flat check
                        if dzdx.abs() < 1e-10 && dzdy.abs() < 1e-10 {
                            continue; // Skip flat areas (undefined aspect)
                        }

                        let aspect_rad = (-dzdx).atan2(-dzdy);

                        sum_cos += aspect_rad.cos();
                        sum_sin += aspect_rad.sin();
                        count += 1;
                    }
                }

                if count < 2 {
                    continue;
                }

                let mean_cos = sum_cos / count as f64;
                let mean_sin = sum_sin / count as f64;
                let r_bar = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();

                *out = 1.0 - r_bar;
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

/// Streaming circular variance of aspect implementing `WindowAlgorithm`.
#[derive(Debug, Clone)]
pub struct CircularVarianceStreaming {
    pub radius: usize,
}

impl Default for CircularVarianceStreaming {
    fn default() -> Self {
        Self { radius: 3 }
    }
}

impl surtgis_core::WindowAlgorithm for CircularVarianceStreaming {
    fn kernel_radius(&self) -> usize {
        // Need extra 1 for aspect computation at window edges
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
        let kr = self.radius + 1; // kernel_radius
        let r = self.radius as isize;

        for row in 0..out_rows {
            let ir = row + kr; // center in input
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

                let mut sum_cos = 0.0;
                let mut sum_sin = 0.0;
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

                        if dzdx.abs() < 1e-10 && dzdy.abs() < 1e-10 {
                            continue;
                        }

                        let aspect_rad = (-dzdx).atan2(-dzdy);
                        sum_cos += aspect_rad.cos();
                        sum_sin += aspect_rad.sin();
                        count += 1;
                    }
                }

                if count < 2 {
                    output[[row, c]] = f64::NAN;
                    continue;
                }

                let mean_cos = sum_cos / count as f64;
                let mean_sin = sum_sin / count as f64;
                let r_bar = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();
                output[[row, c]] = 1.0 - r_bar;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_uniform_slope_low_variance() {
        // Uniform planar slope → all aspects identical → CV ≈ 0
        let mut dem = Raster::new(15, 15);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));
        for r in 0..15 {
            for c in 0..15 {
                dem.set(r, c, r as f64 * 10.0 + c as f64 * 5.0).unwrap();
            }
        }

        let result = circular_variance_aspect(&dem, CircularVarianceParams { radius: 2 }).unwrap();
        let v = result.get(7, 7).unwrap();
        assert!(v < 0.05, "Uniform slope should have CV ≈ 0, got {}", v);
    }

    #[test]
    fn test_range_zero_to_one() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                let x = c as f64 - 10.0;
                let y = r as f64 - 10.0;
                dem.set(r, c, x * x + y * y).unwrap(); // Bowl: aspects radiate outward
            }
        }

        let result = circular_variance_aspect(&dem, CircularVarianceParams { radius: 3 }).unwrap();
        for r in 5..15 {
            for c in 5..15 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v >= -0.001 && v <= 1.001,
                        "CV should be [0,1], got {} at ({},{})", v, r, c);
                }
            }
        }
    }

    #[test]
    fn test_bowl_high_variance() {
        // Bowl: aspects radiate outward in all directions → high CV near center
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                let x = c as f64 - 10.0;
                let y = r as f64 - 10.0;
                dem.set(r, c, x * x + y * y).unwrap();
            }
        }

        let result = circular_variance_aspect(&dem, CircularVarianceParams { radius: 4 }).unwrap();
        let v = result.get(10, 10).unwrap();
        assert!(v > 0.5, "Bowl center should have high CV, got {}", v);
    }
}
