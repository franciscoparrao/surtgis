//! Landscape diversity indices using moving windows
//!
//! Computes landscape ecology metrics on categorical (class) rasters.
//! Values are rounded to the nearest integer for class identification.

use std::collections::HashMap;

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for landscape diversity analysis
#[derive(Debug, Clone)]
pub struct DiversityParams {
    /// Window radius (actual window size = 2*radius + 1)
    pub radius: usize,
    /// Whether to use circular window (default: false = square)
    pub circular: bool,
}

impl Default for DiversityParams {
    fn default() -> Self {
        Self {
            radius: 3,
            circular: false,
        }
    }
}

/// Shannon Diversity Index (H')
///
/// `H' = -sum(pi * ln(pi))` where pi is the proportion of class i in the window.
///
/// Measures information entropy — higher values indicate more diverse
/// landscapes. H' = 0 when only one class is present.
///
/// # Arguments
/// * `raster` - Categorical raster (values rounded to integer classes)
/// * `params` - Window parameters
pub fn shannon_diversity(raster: &Raster<f64>, params: DiversityParams) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Diversity radius must be > 0".into()));
    }

    let (rows, cols) = raster.shape();
    let offsets = build_offsets(params.radius, params.circular);

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let counts = collect_class_counts(raster, row, col, rows, cols, &offsets);
                if counts.is_empty() {
                    continue;
                }

                let total: usize = counts.values().sum();
                let total_f = total as f64;
                let mut h = 0.0;

                for &count in counts.values() {
                    let pi = count as f64 / total_f;
                    if pi > 0.0 {
                        h -= pi * pi.ln();
                    }
                }

                *out = h;
            }

            row_data
        })
        .collect();

    build_output(raster, rows, cols, output_data)
}

/// Simpson Diversity Index (1 - D)
///
/// `D = sum(pi^2)`, result = `1 - D`
///
/// Probability that two randomly selected cells belong to different classes.
/// Range [0, 1): 0 = single class, approaches 1 = many equally abundant classes.
///
/// # Arguments
/// * `raster` - Categorical raster (values rounded to integer classes)
/// * `params` - Window parameters
pub fn simpson_diversity(raster: &Raster<f64>, params: DiversityParams) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Diversity radius must be > 0".into()));
    }

    let (rows, cols) = raster.shape();
    let offsets = build_offsets(params.radius, params.circular);

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let counts = collect_class_counts(raster, row, col, rows, cols, &offsets);
                if counts.is_empty() {
                    continue;
                }

                let total: usize = counts.values().sum();
                let total_f = total as f64;
                let mut d = 0.0;

                for &count in counts.values() {
                    let pi = count as f64 / total_f;
                    d += pi * pi;
                }

                *out = 1.0 - d;
            }

            row_data
        })
        .collect();

    build_output(raster, rows, cols, output_data)
}

/// Patch Density
///
/// Number of distinct contiguous patches (4-connected) within the moving window,
/// normalized by window area (patches per cell).
///
/// Higher values indicate more fragmented landscapes.
///
/// # Arguments
/// * `raster` - Categorical raster (values rounded to integer classes)
/// * `params` - Window parameters
pub fn patch_density(raster: &Raster<f64>, params: DiversityParams) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Patch density radius must be > 0".into()));
    }

    let (rows, cols) = raster.shape();
    let r = params.radius as isize;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, out) in row_data.iter_mut().enumerate() {
                let win_size = (2 * params.radius + 1) as usize;
                let mut grid = vec![f64::NAN; win_size * win_size];
                let mut valid_count = 0usize;

                // Extract window into local grid
                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = row as isize + dr;
                        let nc = col as isize + dc;

                        if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                            let v = unsafe { raster.get_unchecked(nr as usize, nc as usize) };
                            if !v.is_nan() {
                                let lr = (dr + r) as usize;
                                let lc = (dc + r) as usize;
                                grid[lr * win_size + lc] = v.round();
                                valid_count += 1;
                            }
                        }
                    }
                }

                if valid_count == 0 {
                    continue;
                }

                // Count patches using flood-fill (4-connected)
                let mut visited = vec![false; win_size * win_size];
                let mut patches = 0usize;

                for i in 0..win_size {
                    for j in 0..win_size {
                        let idx = i * win_size + j;
                        if !visited[idx] && !grid[idx].is_nan() {
                            patches += 1;
                            flood_fill(&grid, &mut visited, win_size, i, j, grid[idx]);
                        }
                    }
                }

                *out = patches as f64 / valid_count as f64;
            }

            row_data
        })
        .collect();

    build_output(raster, rows, cols, output_data)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn build_offsets(radius: usize, circular: bool) -> Vec<(isize, isize)> {
    let r = radius as isize;
    let mut offsets = Vec::new();

    if circular {
        let r_sq = (radius * radius) as isize;
        for dr in -r..=r {
            for dc in -r..=r {
                if dr * dr + dc * dc <= r_sq {
                    offsets.push((dr, dc));
                }
            }
        }
    } else {
        for dr in -r..=r {
            for dc in -r..=r {
                offsets.push((dr, dc));
            }
        }
    }

    offsets
}

fn collect_class_counts(
    raster: &Raster<f64>,
    row: usize,
    col: usize,
    rows: usize,
    cols: usize,
    offsets: &[(isize, isize)],
) -> HashMap<i64, usize> {
    let mut counts = HashMap::new();

    for &(dr, dc) in offsets {
        let nr = row as isize + dr;
        let nc = col as isize + dc;

        if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
            let v = unsafe { raster.get_unchecked(nr as usize, nc as usize) };
            if !v.is_nan() {
                let class = v.round() as i64;
                *counts.entry(class).or_insert(0) += 1;
            }
        }
    }

    counts
}

fn flood_fill(grid: &[f64], visited: &mut [bool], size: usize, r: usize, c: usize, class: f64) {
    let mut stack = vec![(r, c)];

    while let Some((cr, cc)) = stack.pop() {
        let idx = cr * size + cc;
        if visited[idx] {
            continue;
        }
        if grid[idx].is_nan() || (grid[idx] - class).abs() > 0.5 {
            continue;
        }

        visited[idx] = true;

        // 4-connected neighbors
        if cr > 0 { stack.push((cr - 1, cc)); }
        if cr + 1 < size { stack.push((cr + 1, cc)); }
        if cc > 0 { stack.push((cr, cc - 1)); }
        if cc + 1 < size { stack.push((cr, cc + 1)); }
    }
}

fn build_output(
    template: &Raster<f64>,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
) -> Result<Raster<f64>> {
    let mut output = template.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn class_raster() -> Raster<f64> {
        // 10x10 raster with 4 quadrants: class 1,2,3,4
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                let class = match (row < 5, col < 5) {
                    (true, true) => 1.0,
                    (true, false) => 2.0,
                    (false, true) => 3.0,
                    (false, false) => 4.0,
                };
                r.set(row, col, class).unwrap();
            }
        }
        r
    }

    fn uniform_class_raster() -> Raster<f64> {
        let mut r = Raster::filled(10, 10, 1.0);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        r
    }

    #[test]
    fn test_shannon_uniform() {
        let r = uniform_class_raster();
        let result = shannon_diversity(&r, DiversityParams::default()).unwrap();
        let v = result.get(5, 5).unwrap();
        assert!(v.abs() < 1e-10, "Shannon of uniform should be 0, got {}", v);
    }

    #[test]
    fn test_shannon_diverse() {
        let r = class_raster();
        // Window centered at (4,4) with radius=1 should see classes 1,2,3,4
        let result = shannon_diversity(&r, DiversityParams { radius: 1, circular: false }).unwrap();
        let v = result.get(4, 4).unwrap();
        // 4 classes equally: H = -4 * (0.25 * ln(0.25)) = ln(4) ≈ 1.386
        // But window may not be perfectly split... let's just check > 0
        assert!(v > 0.0, "Shannon at class boundary should be > 0, got {}", v);
    }

    #[test]
    fn test_simpson_uniform() {
        let r = uniform_class_raster();
        let result = simpson_diversity(&r, DiversityParams::default()).unwrap();
        let v = result.get(5, 5).unwrap();
        assert!(v.abs() < 1e-10, "Simpson of uniform should be 0, got {}", v);
    }

    #[test]
    fn test_simpson_diverse() {
        let r = class_raster();
        let result = simpson_diversity(&r, DiversityParams { radius: 1, circular: false }).unwrap();
        let v = result.get(4, 4).unwrap();
        assert!(v > 0.0, "Simpson at class boundary should be > 0, got {}", v);
    }

    #[test]
    fn test_patch_density_uniform() {
        let r = uniform_class_raster();
        let result = patch_density(&r, DiversityParams::default()).unwrap();
        let v = result.get(5, 5).unwrap();
        // Uniform = 1 patch / n cells
        let expected = 1.0 / 49.0; // 7x7 window
        assert!((v - expected).abs() < 1e-10, "Patch density of uniform should be 1/n, got {}", v);
    }

    #[test]
    fn test_patch_density_fragmented() {
        // Checkerboard pattern (maximum fragmentation)
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                r.set(row, col, ((row + col) % 2) as f64).unwrap();
            }
        }

        let result = patch_density(&r, DiversityParams { radius: 1, circular: false }).unwrap();
        let v = result.get(5, 5).unwrap();
        // Checkerboard with radius=1: each cell is its own patch in 4-connected
        // 9 cells / 9 cells = 1.0 (assuming all are isolated patches)
        assert!(v > 0.5, "Checkerboard should have high patch density, got {}", v);
    }

    #[test]
    fn test_diversity_radius_zero() {
        let r = uniform_class_raster();
        assert!(shannon_diversity(&r, DiversityParams { radius: 0, circular: false }).is_err());
        assert!(simpson_diversity(&r, DiversityParams { radius: 0, circular: false }).is_err());
        assert!(patch_density(&r, DiversityParams { radius: 0, circular: false }).is_err());
    }
}
