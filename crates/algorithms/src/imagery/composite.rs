//! Temporal compositing operations for multi-scene imagery
//!
//! Per-pixel statistics across multiple rasters (e.g., cloud-free median composite).

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Per-pixel median composite across multiple rasters.
///
/// For each pixel position, collects all finite (non-NaN) values across the
/// input rasters, sorts them, and returns the median. This is the standard
/// approach for producing cloud-free composites from satellite time series.
///
/// # Arguments
/// * `rasters` - Slice of rasters (must all have the same dimensions)
///
/// # Returns
/// A single raster where each pixel is the median of corresponding pixels
/// across all inputs. Pixels where all inputs are NaN produce NaN.
pub fn median_composite(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    if rasters.len() < 2 {
        return Err(Error::Other("median_composite requires at least 2 rasters".into()));
    }

    // Check if all rasters have the same dimensions
    let all_same_size = {
        let (r0, c0) = rasters[0].shape();
        rasters.iter().all(|r| r.shape() == (r0, c0))
    };

    if all_same_size {
        // Fast path: same dimensions, direct pixel-wise median
        median_composite_aligned(rasters)
    } else {
        // Different extents: align to union bbox first, then median
        median_composite_unaligned(rasters)
    }
}

/// Fast path: all rasters have identical dimensions.
fn median_composite_aligned(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let (rows, cols) = rasters[0].shape();
    let n = rasters.len();
    let nodata = rasters[0].nodata();
    let arrays: Vec<&Array2<f64>> = rasters.iter().map(|r| r.data()).collect();

    let mut output = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    output
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            let mut values = Vec::with_capacity(n);
            for col in 0..cols {
                values.clear();
                for arr in &arrays {
                    let v = arr[[row, col]];
                    if !v.is_finite() {
                        continue;
                    }
                    if let Some(nd) = nodata {
                        if nd.is_finite() && (v - nd).abs() < 1e-10 {
                            continue;
                        }
                    }
                    values.push(v);
                }
                out_row[col] = compute_median(&mut values);
            }
        });

    let mut result = Raster::from_array(output);
    result.set_transform(rasters[0].transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

/// Slow path: rasters have different extents. Compute union bbox,
/// then for each output pixel find the value in each input raster
/// using its GeoTransform.
fn median_composite_unaligned(rasters: &[&Raster<f64>]) -> Result<Raster<f64>> {
    let gt0 = rasters[0].transform();
    let pw = gt0.pixel_width;
    let ph = gt0.pixel_height.abs();

    // Compute union bounding box
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for r in rasters {
        let (bmin_x, bmin_y, bmax_x, bmax_y) = r.bounds();
        min_x = min_x.min(bmin_x);
        min_y = min_y.min(bmin_y);
        max_x = max_x.max(bmax_x);
        max_y = max_y.max(bmax_y);
    }

    let out_cols = ((max_x - min_x) / pw).round() as usize;
    let out_rows = ((max_y - min_y) / ph).round() as usize;

    if out_cols == 0 || out_rows == 0 {
        return Err(Error::Other("Composite output has zero dimensions".into()));
    }

    // For each input raster, precompute the pixel offset in the output grid
    let offsets: Vec<(isize, isize, usize, usize)> = rasters
        .iter()
        .map(|r| {
            let gt = r.transform();
            let col_off = ((gt.origin_x - min_x) / pw).round() as isize;
            let row_off = ((max_y - gt.origin_y) / ph).round() as isize;
            let (rrows, rcols) = r.shape();
            (row_off, col_off, rrows, rcols)
        })
        .collect();

    let n = rasters.len();
    let arrays: Vec<&Array2<f64>> = rasters.iter().map(|r| r.data()).collect();

    let mut output = Array2::<f64>::from_elem((out_rows, out_cols), f64::NAN);

    output
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(out_cols)
        .enumerate()
        .for_each(|(out_row, row_buf)| {
            let mut values = Vec::with_capacity(n);
            for out_col in 0..out_cols {
                values.clear();
                for (i, &(row_off, col_off, rrows, rcols)) in offsets.iter().enumerate() {
                    let src_row = out_row as isize - row_off;
                    let src_col = out_col as isize - col_off;
                    if src_row < 0
                        || src_col < 0
                        || src_row >= rrows as isize
                        || src_col >= rcols as isize
                    {
                        continue;
                    }
                    let v = arrays[i][[src_row as usize, src_col as usize]];
                    if v.is_finite() {
                        values.push(v);
                    }
                }
                row_buf[out_col] = compute_median(&mut values);
            }
        });

    let mut result = Raster::from_array(output);
    result.set_transform(surtgis_core::GeoTransform::new(min_x, max_y, pw, -ph));
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = rasters[0].crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

fn compute_median(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::Raster;
    use surtgis_core::GeoTransform;

    fn make_raster(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let gt = GeoTransform::new(0.0, 0.0, 1.0, -1.0);
        let mut r = Raster::from_array(arr);
        r.set_transform(gt);
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_median_composite_3_images() {
        let r1 = make_raster(vec![vec![1.0, 2.0], vec![3.0, f64::NAN]]);
        let r2 = make_raster(vec![vec![5.0, 4.0], vec![f64::NAN, f64::NAN]]);
        let r3 = make_raster(vec![vec![3.0, 6.0], vec![7.0, f64::NAN]]);

        let result = median_composite(&[&r1, &r2, &r3]).unwrap();
        let d = result.data();

        assert!((d[[0, 0]] - 3.0).abs() < 1e-10); // median of [1,5,3] = 3
        assert!((d[[0, 1]] - 4.0).abs() < 1e-10); // median of [2,4,6] = 4
        assert!((d[[1, 0]] - 5.0).abs() < 1e-10); // median of [3,7] = 5 (mean of 2)
        assert!(d[[1, 1]].is_nan()); // all NaN
    }

    #[test]
    fn test_median_composite_too_few() {
        let r1 = make_raster(vec![vec![1.0]]);
        assert!(median_composite(&[&r1]).is_err());
    }

    fn make_georaster(data: Vec<Vec<f64>>, origin_x: f64, origin_y: f64) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(origin_x, origin_y, 10.0, -10.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_median_composite_different_extents() {
        // Scene 1: 2x3 at origin (0, 20)
        let r1 = make_georaster(vec![
            vec![10.0, 20.0, 30.0],
            vec![40.0, 50.0, 60.0],
        ], 0.0, 20.0);

        // Scene 2: 2x2 at origin (10, 20) — shifted 1 col right, 1 col narrower
        let r2 = make_georaster(vec![
            vec![100.0, 200.0],
            vec![300.0, 400.0],
        ], 10.0, 20.0);

        let result = median_composite(&[&r1, &r2]).unwrap();
        let (rows, cols) = result.shape();

        // Union: x=[0,30], y=[0,20] → 2 rows, 3 cols at 10m
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);

        let d = result.data();
        // col 0: only r1 has data → 10.0, 40.0
        assert!((d[[0, 0]] - 10.0).abs() < 1e-10);
        assert!((d[[1, 0]] - 40.0).abs() < 1e-10);

        // col 1: r1=20, r2=100 → median of [20,100] = 60
        assert!((d[[0, 1]] - 60.0).abs() < 1e-10);
        // col 1, row 1: r1=50, r2=300 → median of [50,300] = 175
        assert!((d[[1, 1]] - 175.0).abs() < 1e-10);

        // col 2: r1=30, r2=200 → median of [30,200] = 115
        assert!((d[[0, 2]] - 115.0).abs() < 1e-10);
    }
}
