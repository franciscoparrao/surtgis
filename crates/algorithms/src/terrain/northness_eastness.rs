//! Northness and Eastness from DEMs
//!
//! Decompose aspect into continuous directional components:
//! - **Northness** = cos(aspect) → ranges from -1 (south) to +1 (north)
//! - **Eastness** = sin(aspect) → ranges from -1 (west) to +1 (east)
//!
//! These circular decompositions avoid the discontinuity at 0°/360° that makes
//! raw aspect problematic for statistical analysis and modeling.
//!
//! Reference: Stage (1976) "An expression for the effect of aspect, slope,
//! and habitat type on tree growth"

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::Result;
use surtgis_core::raster::Raster;

use super::aspect::{AspectOutput, aspect};

/// Calculate northness from a DEM
///
/// `northness = cos(aspect)` where aspect is in radians (0 = North, clockwise).
///
/// - +1.0 = north-facing slope
/// - -1.0 = south-facing slope
/// -  0.0 = east or west-facing slope
/// - NaN = flat or border cell
///
/// # Arguments
/// * `dem` - Input DEM raster
pub fn northness(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let aspect_raster = aspect(dem, AspectOutput::Radians)?;
    let (rows, cols) = aspect_raster.shape();
    let aspect_data = aspect_raster.data();

    let data = par_map_rows(rows, cols, |row, out_row| {
        for (col, out_val) in out_row.iter_mut().enumerate() {
            let a = aspect_data[[row, col]];
            // aspect returns NaN for flat/nodata cells; `a < 0.0` is
            // defensive leftover from when it used a `-1.0` sentinel.
            if a < 0.0 || a.is_nan() {
                continue;
            }
            *out_val = a.cos();
        }
    });

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = data;
    Ok(output)
}

/// Calculate eastness from a DEM
///
/// `eastness = sin(aspect)` where aspect is in radians (0 = North, clockwise).
///
/// - +1.0 = east-facing slope
/// - -1.0 = west-facing slope
/// -  0.0 = north or south-facing slope
/// - NaN = flat or border cell
///
/// # Arguments
/// * `dem` - Input DEM raster
pub fn eastness(dem: &Raster<f64>) -> Result<Raster<f64>> {
    let aspect_raster = aspect(dem, AspectOutput::Radians)?;
    let (rows, cols) = aspect_raster.shape();
    let aspect_data = aspect_raster.data();

    let data = par_map_rows(rows, cols, |row, out_row| {
        for (col, out_val) in out_row.iter_mut().enumerate() {
            let a = aspect_data[[row, col]];
            if a < 0.0 || a.is_nan() {
                continue;
            }
            *out_val = a.sin();
        }
    });

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = data;
    Ok(output)
}

/// Calculate both northness and eastness from a DEM in a single pass
///
/// More efficient than calling `northness()` and `eastness()` separately
/// since aspect is computed only once.
///
/// # Returns
/// Tuple of (northness, eastness) rasters
pub fn northness_eastness(dem: &Raster<f64>) -> Result<(Raster<f64>, Raster<f64>)> {
    let aspect_raster = aspect(dem, AspectOutput::Radians)?;
    let (rows, cols) = aspect_raster.shape();
    let aspect_data = aspect_raster.data();

    // Two independent passes over the precomputed aspect raster (one per
    // trig component) rather than one pass emitting `(cos, sin)` pairs —
    // `par_map_rows` preallocates a single `Array2` output, so a tuple
    // output isn't a fit. Aspect itself (the expensive part) is still
    // computed only once, above.
    let north_data = par_map_rows(rows, cols, |row, out_row| {
        for (col, out_val) in out_row.iter_mut().enumerate() {
            let a = aspect_data[[row, col]];
            if a < 0.0 || a.is_nan() {
                continue;
            }
            *out_val = a.cos();
        }
    });
    let east_data = par_map_rows(rows, cols, |row, out_row| {
        for (col, out_val) in out_row.iter_mut().enumerate() {
            let a = aspect_data[[row, col]];
            if a < 0.0 || a.is_nan() {
                continue;
            }
            *out_val = a.sin();
        }
    });

    let mut north = dem.with_same_meta::<f64>(rows, cols);
    north.set_nodata(Some(f64::NAN));
    *north.data_mut() = north_data;

    let mut east = dem.with_same_meta::<f64>(rows, cols);
    east.set_nodata(Some(f64::NAN));
    *east.data_mut() = east_data;

    Ok((north, east))
}

// ─── Streaming implementations ─────────────────────────────────────────

/// Per-cell aspect kernel shared by the streaming northness/eastness paths.
///
/// Computes the compass-bearing aspect (radians, 0 = North, clockwise) from
/// Horn (1981) gradients on the 3×3 window centered at `(row, col)`.
/// Gradients are left unnormalised — the `8 · cell_size` factor cancels in
/// `atan2`. Returns NaN for NaN/nodata centers, NaN stencils, or flat cells
/// (zero gradient), matching the sentinel handling of the batch `aspect()`
/// path (which cannot be shared directly because it lives in `aspect.rs`).
///
/// The caller must guarantee that `(row ± 1, col ± 1)` lies inside `data`.
#[inline]
fn aspect_rad_kernel(data: &Array2<f64>, row: usize, col: usize, nodata: Option<f64>) -> f64 {
    const FLAT_THRESHOLD: f64 = 1e-10;

    let e = data[[row, col]];
    if e.is_nan() || nodata.is_some_and(|nd| e == nd) {
        return f64::NAN;
    }

    let a = data[[row - 1, col - 1]];
    let b = data[[row - 1, col]];
    let cv = data[[row - 1, col + 1]];
    let d = data[[row, col - 1]];
    let f = data[[row, col + 1]];
    let g = data[[row + 1, col - 1]];
    let h = data[[row + 1, col]];
    let i = data[[row + 1, col + 1]];

    if [a, b, cv, d, f, g, h, i].iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }

    // Horn's method for gradients (unnormalised — the 8*cs factor
    // cancels out in atan2 so we can skip it for aspect)
    let dz_dx = (cv + 2.0 * f + i) - (a + 2.0 * d + g);
    let dz_dy = (g + 2.0 * h + i) - (a + 2.0 * b + cv);

    if dz_dx.abs() < FLAT_THRESHOLD && dz_dy.abs() < FLAT_THRESHOLD {
        return f64::NAN;
    }

    // Compass bearing (0=North, clockwise)
    (-dz_dx).atan2(dz_dy)
}

/// Shared `process_chunk` body for the streaming northness/eastness
/// calculators: applies `component` (cos for northness, sin for eastness)
/// to the aspect of every interior cell, NaN elsewhere.
fn process_chunk_aspect_component(
    input: &Array2<f64>,
    output: &mut Array2<f64>,
    nodata: Option<f64>,
    component: fn(f64) -> f64,
) {
    let (in_rows, cols) = input.dim();
    let radius = 1;

    output
        .as_slice_mut()
        .expect("process_chunk output must be in standard layout")
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, out_row)| {
            let ir = r + radius;
            if ir == 0 || ir >= in_rows - 1 {
                out_row.fill(f64::NAN);
                return;
            }

            for (c, out_v) in out_row.iter_mut().enumerate() {
                if c == 0 || c >= cols - 1 {
                    *out_v = f64::NAN;
                    continue;
                }

                *out_v = component(aspect_rad_kernel(input, ir, c, nodata));
            }
        });
}

/// Streaming northness calculator implementing `WindowAlgorithm`.
///
/// Computes Horn (1981) gradients and returns `cos(aspect)`.
/// Flat cells (zero gradient) produce NaN.
#[derive(Debug, Clone, Copy, Default)]
pub struct NorthnessStreaming;

impl surtgis_core::WindowAlgorithm for NorthnessStreaming {
    fn kernel_radius(&self) -> usize {
        1 // 3×3 kernel
    }

    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        _cell_size_x: f64,
        _cell_size_y: f64,
    ) {
        process_chunk_aspect_component(input, output, nodata, f64::cos);
    }
}

/// Streaming eastness calculator implementing `WindowAlgorithm`.
///
/// Computes Horn (1981) gradients and returns `sin(aspect)`.
/// Flat cells (zero gradient) produce NaN.
#[derive(Debug, Clone, Copy, Default)]
pub struct EastnessStreaming;

impl surtgis_core::WindowAlgorithm for EastnessStreaming {
    fn kernel_radius(&self) -> usize {
        1 // 3×3 kernel
    }

    fn process_chunk(
        &self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        nodata: Option<f64>,
        _cell_size_x: f64,
        _cell_size_y: f64,
    ) {
        process_chunk_aspect_component(input, output, nodata, f64::sin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn north_slope_dem() -> Raster<f64> {
        // Slopes down to north: higher rows have higher elevation
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, row as f64).unwrap();
            }
        }
        dem
    }

    fn east_slope_dem() -> Raster<f64> {
        // Slopes down to east: higher cols have lower elevation
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, -(col as f64)).unwrap();
            }
        }
        dem
    }

    #[test]
    fn test_northness_north_slope() {
        let dem = north_slope_dem();
        let result = northness(&dem).unwrap();
        let val = result.get(5, 5).unwrap();
        // North-facing slope → northness close to +1
        assert!(val > 0.9, "Expected northness ~1.0, got {}", val);
    }

    #[test]
    fn test_eastness_east_slope() {
        let dem = east_slope_dem();
        let result = eastness(&dem).unwrap();
        let val = result.get(5, 5).unwrap();
        // East-facing slope → eastness close to +1
        assert!(val > 0.9, "Expected eastness ~1.0, got {}", val);
    }

    #[test]
    fn test_northness_eastness_combined() {
        let dem = north_slope_dem();
        let (north, east) = northness_eastness(&dem).unwrap();
        let n = north.get(5, 5).unwrap();
        let e = east.get(5, 5).unwrap();
        // n² + e² should be ≈ 1 for non-flat cells
        let sum_sq = n * n + e * e;
        assert!(
            (sum_sq - 1.0).abs() < 0.01,
            "n²+e² should be ~1.0, got {} (n={}, e={})",
            sum_sq,
            n,
            e
        );
    }

    #[test]
    fn test_flat_surface_is_nan() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let result = northness(&dem).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(val.is_nan(), "Flat surface should produce NaN, got {}", val);
    }
}
