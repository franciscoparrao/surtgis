//! Per-pixel temporal statistics across a raster time series.
//!
//! Each function takes a slice of co-registered rasters and computes
//! a single output raster where each pixel is a statistic across the
//! temporal dimension.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// All temporal statistics in a single pass.
pub struct TemporalStats {
    pub mean: Raster<f64>,
    pub std: Raster<f64>,
    pub min: Raster<f64>,
    pub max: Raster<f64>,
    pub count: Raster<f64>,
}

fn validate_stack(rasters: &[&Raster<f64>]) -> Result<(usize, usize)> {
    if rasters.is_empty() {
        return Err(Error::Other("temporal statistics require at least 1 raster".into()));
    }
    let (rows, cols) = rasters[0].shape();
    for r in rasters.iter().skip(1) {
        if r.shape() != (rows, cols) {
            return Err(Error::SizeMismatch {
                er: rows, ec: cols, ar: r.rows(), ac: r.cols(),
            });
        }
    }
    Ok((rows, cols))
}

fn collect_valid(rasters: &[&Raster<f64>], row: usize, col: usize) -> Vec<f64> {
    let mut vals = Vec::with_capacity(rasters.len());
    for r in rasters {
        let v = unsafe { r.get_unchecked(row, col) };
        if v.is_finite() {
            vals.push(v);
        }
    }
    vals
}

/// Per-pixel mean across time.
pub fn temporal_mean(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let (rows, cols) = validate_stack(rasters)?;
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    out.as_slice_mut().unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            for col in 0..cols {
                let vals = collect_valid(rasters, row, col);
                if !vals.is_empty() {
                    out_row[col] = vals.iter().sum::<f64>() / vals.len() as f64;
                }
            }
        });

    let mut result = Raster::from_array(out);
    result.set_transform(rasters[0].transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

/// Per-pixel standard deviation across time.
pub fn temporal_std(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let (rows, cols) = validate_stack(rasters)?;
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    out.as_slice_mut().unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            for col in 0..cols {
                let vals = collect_valid(rasters, row, col);
                if vals.len() >= 2 {
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
                    out_row[col] = var.sqrt();
                }
            }
        });

    let mut result = Raster::from_array(out);
    result.set_transform(rasters[0].transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

/// Per-pixel minimum across time.
pub fn temporal_min(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let (rows, cols) = validate_stack(rasters)?;
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    out.as_slice_mut().unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            for col in 0..cols {
                let vals = collect_valid(rasters, row, col);
                if let Some(&min) = vals.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                    out_row[col] = min;
                }
            }
        });

    let mut result = Raster::from_array(out);
    result.set_transform(rasters[0].transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

/// Per-pixel maximum across time.
pub fn temporal_max(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let (rows, cols) = validate_stack(rasters)?;
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    out.as_slice_mut().unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            for col in 0..cols {
                let vals = collect_valid(rasters, row, col);
                if let Some(&max) = vals.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                    out_row[col] = max;
                }
            }
        });

    let mut result = Raster::from_array(out);
    result.set_transform(rasters[0].transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

/// Per-pixel count of valid (non-NaN) observations across time.
pub fn temporal_count(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let (rows, cols) = validate_stack(rasters)?;
    let mut out = Array2::<f64>::from_elem((rows, cols), 0.0);

    out.as_slice_mut().unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            for col in 0..cols {
                let count = collect_valid(rasters, row, col).len();
                out_row[col] = count as f64;
            }
        });

    let mut result = Raster::from_array(out);
    result.set_transform(rasters[0].transform().clone());
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

/// Per-pixel percentile (0-100) across time.
pub fn temporal_percentile(rasters: &[&Raster<f64>], percentile: f64) -> Result<Raster<f64>> {
    if !(0.0..=100.0).contains(&percentile) {
        return Err(Error::Other(format!("percentile must be 0-100, got {}", percentile)));
    }
    let (rows, cols) = validate_stack(rasters)?;
    let mut out = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    out.as_slice_mut().unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            for col in 0..cols {
                let mut vals = collect_valid(rasters, row, col);
                if !vals.is_empty() {
                    vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    let idx = (percentile / 100.0) * (vals.len() - 1) as f64;
                    let lo = idx.floor() as usize;
                    let hi = idx.ceil() as usize;
                    if lo == hi {
                        out_row[col] = vals[lo];
                    } else {
                        let frac = idx - lo as f64;
                        out_row[col] = vals[lo] * (1.0 - frac) + vals[hi] * frac;
                    }
                }
            }
        });

    let mut result = Raster::from_array(out);
    result.set_transform(rasters[0].transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

/// Compute all temporal statistics in a single pass (more efficient than calling each separately).
pub fn temporal_stats(rasters: &[&Raster<f64>]) -> Result<TemporalStats> {
    let (rows, cols) = validate_stack(rasters)?;
    let n = rasters.len();

    let mut mean_arr = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    let mut std_arr = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    let mut min_arr = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    let mut max_arr = Array2::<f64>::from_elem((rows, cols), f64::NAN);
    let mut count_arr = Array2::<f64>::from_elem((rows, cols), 0.0);

    // Process row by row (can't easily parallelize across 5 output arrays)
    let total = rows * cols;
    let mut mean_flat = vec![f64::NAN; total];
    let mut std_flat = vec![f64::NAN; total];
    let mut min_flat = vec![f64::NAN; total];
    let mut max_flat = vec![f64::NAN; total];
    let mut count_flat = vec![0.0f64; total];

    // Zip 5 slices together for parallel processing
    let chunk_size = cols;
    mean_flat.par_chunks_mut(chunk_size)
        .zip(std_flat.par_chunks_mut(chunk_size))
        .zip(min_flat.par_chunks_mut(chunk_size))
        .zip(max_flat.par_chunks_mut(chunk_size))
        .zip(count_flat.par_chunks_mut(chunk_size))
        .enumerate()
        .for_each(|(row, ((((mean_row, std_row), min_row), max_row), count_row))| {
            let mut vals = Vec::with_capacity(n);
            for col in 0..cols {
                vals.clear();
                for r in rasters {
                    let v = unsafe { r.get_unchecked(row, col) };
                    if v.is_finite() {
                        vals.push(v);
                    }
                }
                count_row[col] = vals.len() as f64;
                if !vals.is_empty() {
                    let sum: f64 = vals.iter().sum();
                    let m = sum / vals.len() as f64;
                    mean_row[col] = m;
                    min_row[col] = vals.iter().cloned().fold(f64::INFINITY, f64::min);
                    max_row[col] = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    if vals.len() >= 2 {
                        let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
                        std_row[col] = var.sqrt();
                    }
                }
            }
        });

    mean_arr.as_slice_mut().unwrap().copy_from_slice(&mean_flat);
    std_arr.as_slice_mut().unwrap().copy_from_slice(&std_flat);
    min_arr.as_slice_mut().unwrap().copy_from_slice(&min_flat);
    max_arr.as_slice_mut().unwrap().copy_from_slice(&max_flat);
    count_arr.as_slice_mut().unwrap().copy_from_slice(&count_flat);

    let make_raster = |arr: Array2<f64>, nodata: bool| {
        let mut r = Raster::from_array(arr);
        r.set_transform(rasters[0].transform().clone());
        if nodata { r.set_nodata(Some(f64::NAN)); }
        if let Some(crs) = rasters[0].crs() {
            r.set_crs(Some(crs.clone()));
        }
        r
    };

    Ok(TemporalStats {
        mean: make_raster(mean_arr, true),
        std: make_raster(std_arr, true),
        min: make_raster(min_arr, true),
        max: make_raster(max_arr, true),
        count: make_raster(count_arr, false),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_raster(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_temporal_mean() {
        let r1 = make_raster(vec![vec![10.0, 20.0], vec![30.0, f64::NAN]]);
        let r2 = make_raster(vec![vec![20.0, 40.0], vec![f64::NAN, f64::NAN]]);
        let r3 = make_raster(vec![vec![30.0, 60.0], vec![50.0, f64::NAN]]);

        let result = temporal_mean(&[&r1, &r2, &r3]).unwrap();
        let d = result.data();

        assert!((d[[0, 0]] - 20.0).abs() < 1e-10); // mean(10,20,30)
        assert!((d[[0, 1]] - 40.0).abs() < 1e-10); // mean(20,40,60)
        assert!((d[[1, 0]] - 40.0).abs() < 1e-10); // mean(30,50)
        assert!(d[[1, 1]].is_nan()); // all NaN
    }

    #[test]
    fn test_temporal_std() {
        let r1 = make_raster(vec![vec![10.0]]);
        let r2 = make_raster(vec![vec![20.0]]);
        let r3 = make_raster(vec![vec![30.0]]);

        let result = temporal_std(&[&r1, &r2, &r3]).unwrap();
        assert!((result.data()[[0, 0]] - 10.0).abs() < 1e-10); // std(10,20,30) = 10
    }

    #[test]
    fn test_temporal_min_max() {
        let r1 = make_raster(vec![vec![10.0, 50.0]]);
        let r2 = make_raster(vec![vec![5.0, 30.0]]);
        let r3 = make_raster(vec![vec![20.0, 40.0]]);

        let min = temporal_min(&[&r1, &r2, &r3]).unwrap();
        let max = temporal_max(&[&r1, &r2, &r3]).unwrap();

        assert!((min.data()[[0, 0]] - 5.0).abs() < 1e-10);
        assert!((max.data()[[0, 0]] - 20.0).abs() < 1e-10);
        assert!((min.data()[[0, 1]] - 30.0).abs() < 1e-10);
        assert!((max.data()[[0, 1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_count() {
        let r1 = make_raster(vec![vec![1.0, f64::NAN]]);
        let r2 = make_raster(vec![vec![2.0, f64::NAN]]);
        let r3 = make_raster(vec![vec![f64::NAN, f64::NAN]]);

        let result = temporal_count(&[&r1, &r2, &r3]).unwrap();
        assert!((result.data()[[0, 0]] - 2.0).abs() < 1e-10);
        assert!((result.data()[[0, 1]] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_percentile() {
        let r1 = make_raster(vec![vec![10.0]]);
        let r2 = make_raster(vec![vec![20.0]]);
        let r3 = make_raster(vec![vec![30.0]]);
        let r4 = make_raster(vec![vec![40.0]]);

        let p0 = temporal_percentile(&[&r1, &r2, &r3, &r4], 0.0).unwrap();
        let p50 = temporal_percentile(&[&r1, &r2, &r3, &r4], 50.0).unwrap();
        let p100 = temporal_percentile(&[&r1, &r2, &r3, &r4], 100.0).unwrap();

        assert!((p0.data()[[0, 0]] - 10.0).abs() < 1e-10);
        assert!((p50.data()[[0, 0]] - 25.0).abs() < 1e-10); // interp between 20 and 30
        assert!((p100.data()[[0, 0]] - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_stats_all() {
        let r1 = make_raster(vec![vec![10.0, 20.0]]);
        let r2 = make_raster(vec![vec![20.0, 40.0]]);
        let r3 = make_raster(vec![vec![30.0, 60.0]]);

        let stats = temporal_stats(&[&r1, &r2, &r3]).unwrap();

        assert!((stats.mean.data()[[0, 0]] - 20.0).abs() < 1e-10);
        assert!((stats.min.data()[[0, 0]] - 10.0).abs() < 1e-10);
        assert!((stats.max.data()[[0, 0]] - 30.0).abs() < 1e-10);
        assert!((stats.count.data()[[0, 0]] - 3.0).abs() < 1e-10);
        assert!((stats.std.data()[[0, 0]] - 10.0).abs() < 1e-10);
    }
}
