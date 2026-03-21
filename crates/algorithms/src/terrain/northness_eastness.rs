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

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

use super::aspect::{aspect, AspectOutput};

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

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let a = aspect_data[[row, col]];
                // aspect returns -1.0 for flat/nodata cells
                if a < 0.0 || a.is_nan() {
                    continue;
                }
                *row_data_col = a.cos();
            }
            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
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

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let a = aspect_data[[row, col]];
                if a < 0.0 || a.is_nan() {
                    continue;
                }
                *row_data_col = a.sin();
            }
            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
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

    let pairs: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN); cols];
            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let a = aspect_data[[row, col]];
                if a < 0.0 || a.is_nan() {
                    continue;
                }
                *row_data_col = (a.cos(), a.sin());
            }
            row_data
        })
        .collect();

    let (north_data, east_data): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

    let mut north = dem.with_same_meta::<f64>(rows, cols);
    north.set_nodata(Some(f64::NAN));
    *north.data_mut() = Array2::from_shape_vec((rows, cols), north_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    let mut east = dem.with_same_meta::<f64>(rows, cols);
    east.set_nodata(Some(f64::NAN));
    *east.data_mut() = Array2::from_shape_vec((rows, cols), east_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok((north, east))
}

// ─── Streaming implementations ─────────────────────────────────────────

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
        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        const FLAT_THRESHOLD: f64 = 1e-10;

        for r in 0..out_rows {
            let ir = r + radius;
            if ir == 0 || ir >= in_rows - 1 {
                for c in 0..cols {
                    output[[r, c]] = f64::NAN;
                }
                continue;
            }

            for c in 0..cols {
                if c == 0 || c >= cols - 1 {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let e = input[[ir, c]];
                if e.is_nan()
                    || nodata.map_or(false, |nd| (e - nd).abs() < f64::EPSILON)
                {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let a = input[[ir - 1, c - 1]];
                let b = input[[ir - 1, c]];
                let cv = input[[ir - 1, c + 1]];
                let d = input[[ir, c - 1]];
                let f = input[[ir, c + 1]];
                let g = input[[ir + 1, c - 1]];
                let h = input[[ir + 1, c]];
                let i = input[[ir + 1, c + 1]];

                if [a, b, cv, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                // Horn's method for gradients (unnormalised — the 8*cs factor
                // cancels out in atan2 so we can skip it for aspect)
                let dz_dx = (cv + 2.0 * f + i) - (a + 2.0 * d + g);
                let dz_dy = (g + 2.0 * h + i) - (a + 2.0 * b + cv);

                if dz_dx.abs() < FLAT_THRESHOLD && dz_dy.abs() < FLAT_THRESHOLD {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                // Compass bearing (0=North, clockwise)
                let aspect_rad = (-dz_dx).atan2(dz_dy);
                output[[r, c]] = aspect_rad.cos();
            }
        }
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
        let (in_rows, cols) = input.dim();
        let out_rows = output.nrows();
        let radius = 1;

        const FLAT_THRESHOLD: f64 = 1e-10;

        for r in 0..out_rows {
            let ir = r + radius;
            if ir == 0 || ir >= in_rows - 1 {
                for c in 0..cols {
                    output[[r, c]] = f64::NAN;
                }
                continue;
            }

            for c in 0..cols {
                if c == 0 || c >= cols - 1 {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let e = input[[ir, c]];
                if e.is_nan()
                    || nodata.map_or(false, |nd| (e - nd).abs() < f64::EPSILON)
                {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let a = input[[ir - 1, c - 1]];
                let b = input[[ir - 1, c]];
                let cv = input[[ir - 1, c + 1]];
                let d = input[[ir, c - 1]];
                let f = input[[ir, c + 1]];
                let g = input[[ir + 1, c - 1]];
                let h = input[[ir + 1, c]];
                let i = input[[ir + 1, c + 1]];

                if [a, b, cv, d, f, g, h, i].iter().any(|v| v.is_nan()) {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let dz_dx = (cv + 2.0 * f + i) - (a + 2.0 * d + g);
                let dz_dy = (g + 2.0 * h + i) - (a + 2.0 * b + cv);

                if dz_dx.abs() < FLAT_THRESHOLD && dz_dy.abs() < FLAT_THRESHOLD {
                    output[[r, c]] = f64::NAN;
                    continue;
                }

                let aspect_rad = (-dz_dx).atan2(dz_dy);
                output[[r, c]] = aspect_rad.sin();
            }
        }
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
