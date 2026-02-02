//! Focal (moving window) statistics
//!
//! Computes statistics within a moving window centered on each cell.
//! Supports: Mean, StdDev, Min, Max, Range, Sum, Count, Median, Percentile.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Available focal statistics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FocalStatistic {
    /// Arithmetic mean
    Mean,
    /// Standard deviation (population)
    StdDev,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Range (max - min)
    Range,
    /// Sum of values
    Sum,
    /// Count of valid (non-NaN) values
    Count,
    /// Median value
    Median,
    /// Percentile (0-100)
    Percentile(f64),
}

/// Parameters for focal statistics
#[derive(Debug, Clone)]
pub struct FocalParams {
    /// Window radius (actual window size = 2*radius + 1)
    pub radius: usize,
    /// Statistic to compute
    pub statistic: FocalStatistic,
    /// Whether to use circular window (default: false = square)
    pub circular: bool,
}

impl Default for FocalParams {
    fn default() -> Self {
        Self {
            radius: 1,
            statistic: FocalStatistic::Mean,
            circular: false,
        }
    }
}

/// Compute focal statistics on a raster
///
/// Applies a moving window of the specified radius and computes the
/// requested statistic for all valid cells within the window.
///
/// # Arguments
/// * `raster` - Input raster
/// * `params` - Focal parameters (radius, statistic, circular)
///
/// # Returns
/// Raster with the computed statistic at each cell
pub fn focal_statistics(raster: &Raster<f64>, params: FocalParams) -> Result<Raster<f64>> {
    if params.radius == 0 {
        return Err(Error::Algorithm("Focal radius must be > 0".into()));
    }

    if let FocalStatistic::Percentile(p) = params.statistic
        && !(0.0..=100.0).contains(&p) {
            return Err(Error::Algorithm("Percentile must be between 0 and 100".into()));
        }

    let (rows, cols) = raster.shape();
    let r = params.radius as isize;

    // Precompute circular mask offsets if needed
    let offsets: Vec<(isize, isize)> = if params.circular {
        let r_sq = (params.radius * params.radius) as isize;
        let mut offs = Vec::new();
        for dr in -r..=r {
            for dc in -r..=r {
                if dr * dr + dc * dc <= r_sq {
                    offs.push((dr, dc));
                }
            }
        }
        offs
    } else {
        let mut offs = Vec::with_capacity(((2 * r + 1) * (2 * r + 1)) as usize);
        for dr in -r..=r {
            for dc in -r..=r {
                offs.push((dr, dc));
            }
        }
        offs
    };

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                // Collect valid neighbor values
                let mut values: Vec<f64> = Vec::with_capacity(offsets.len());

                for &(dr, dc) in &offsets {
                    let nr = row as isize + dr;
                    let nc = col as isize + dc;

                    if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                        let v = unsafe { raster.get_unchecked(nr as usize, nc as usize) };
                        if !v.is_nan() {
                            values.push(v);
                        }
                    }
                }

                if values.is_empty() {
                    continue;
                }

                *row_data_col = compute_statistic(&mut values, &params.statistic);
            }

            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

fn compute_statistic(values: &mut [f64], stat: &FocalStatistic) -> f64 {
    let n = values.len() as f64;

    match stat {
        FocalStatistic::Mean => {
            values.iter().sum::<f64>() / n
        }
        FocalStatistic::StdDev => {
            let mean = values.iter().sum::<f64>() / n;
            let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
            var.sqrt()
        }
        FocalStatistic::Min => {
            values.iter().cloned().fold(f64::INFINITY, f64::min)
        }
        FocalStatistic::Max => {
            values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        }
        FocalStatistic::Range => {
            let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            max - min
        }
        FocalStatistic::Sum => {
            values.iter().sum::<f64>()
        }
        FocalStatistic::Count => {
            n
        }
        FocalStatistic::Median => {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mid = values.len() / 2;
            if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            }
        }
        FocalStatistic::Percentile(p) => {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = (p / 100.0 * (values.len() - 1) as f64).round() as usize;
            values[idx.min(values.len() - 1)]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn uniform_raster(size: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(size, size, value);
        r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        r
    }

    fn gradient_raster(size: usize) -> Raster<f64> {
        let mut r = Raster::new(size, size);
        r.set_transform(GeoTransform::new(0.0, size as f64, 1.0, -1.0));
        for row in 0..size {
            for col in 0..size {
                r.set(row, col, (row * size + col) as f64).unwrap();
            }
        }
        r
    }

    #[test]
    fn test_focal_mean_uniform() {
        let r = uniform_raster(10, 5.0);
        let result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::Mean,
            circular: false,
        }).unwrap();
        let v = result.get(5, 5).unwrap();
        assert!((v - 5.0).abs() < 1e-10, "Mean of uniform should be 5.0, got {}", v);
    }

    #[test]
    fn test_focal_min_max() {
        let r = gradient_raster(10);
        let min_result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::Min,
            circular: false,
        }).unwrap();
        let max_result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::Max,
            circular: false,
        }).unwrap();

        let min_v = min_result.get(5, 5).unwrap();
        let max_v = max_result.get(5, 5).unwrap();
        // Cell (5,5) = 55, neighbors span (4,4)=44 to (6,6)=66
        assert!((min_v - 44.0).abs() < 1e-10);
        assert!((max_v - 66.0).abs() < 1e-10);
    }

    #[test]
    fn test_focal_std_uniform() {
        let r = uniform_raster(10, 5.0);
        let result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::StdDev,
            circular: false,
        }).unwrap();
        let v = result.get(5, 5).unwrap();
        assert!(v.abs() < 1e-10, "StdDev of uniform should be 0, got {}", v);
    }

    #[test]
    fn test_focal_range() {
        let r = gradient_raster(10);
        let result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::Range,
            circular: false,
        }).unwrap();
        let v = result.get(5, 5).unwrap();
        assert!((v - 22.0).abs() < 1e-10, "Range should be 22, got {}", v);
    }

    #[test]
    fn test_focal_sum() {
        let r = uniform_raster(10, 1.0);
        let result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::Sum,
            circular: false,
        }).unwrap();
        // Interior cell: 3x3 = 9 cells
        let v = result.get(5, 5).unwrap();
        assert!((v - 9.0).abs() < 1e-10, "Sum should be 9, got {}", v);
    }

    #[test]
    fn test_focal_count() {
        let r = uniform_raster(10, 1.0);
        let result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::Count,
            circular: false,
        }).unwrap();
        let v = result.get(5, 5).unwrap();
        assert!((v - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_focal_median() {
        let r = gradient_raster(10);
        let result = focal_statistics(&r, FocalParams {
            radius: 1,
            statistic: FocalStatistic::Median,
            circular: false,
        }).unwrap();
        // Median of 3x3 window around (5,5)=55 should be 55
        let v = result.get(5, 5).unwrap();
        assert!((v - 55.0).abs() < 1e-10, "Median should be 55, got {}", v);
    }

    #[test]
    fn test_focal_circular() {
        let r = uniform_raster(10, 1.0);
        let result = focal_statistics(&r, FocalParams {
            radius: 2,
            statistic: FocalStatistic::Count,
            circular: true,
        }).unwrap();
        // Circular window r=2: offsets with dr²+dc² <= 4
        // (0,0),(±1,0),(0,±1),(±1,±1),(±2,0),(0,±2) = 13 cells
        let v = result.get(5, 5).unwrap();
        assert!((v - 13.0).abs() < 1e-10, "Circular r=2 should have 13 cells, got {}", v);
    }

    #[test]
    fn test_focal_radius_zero_error() {
        let r = uniform_raster(5, 1.0);
        let result = focal_statistics(&r, FocalParams {
            radius: 0,
            statistic: FocalStatistic::Mean,
            circular: false,
        });
        assert!(result.is_err());
    }
}
