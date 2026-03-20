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

    let (rows, cols) = rasters[0].shape();
    for (i, r) in rasters.iter().enumerate().skip(1) {
        let (ri, ci) = r.shape();
        if ri != rows || ci != cols {
            return Err(Error::Other(format!(
                "Raster {} has shape {}x{}, expected {}x{}",
                i, ri, ci, rows, cols
            )));
        }
    }

    let n = rasters.len();
    let nodata = rasters[0].nodata();

    // Collect data arrays
    let arrays: Vec<&Array2<f64>> = rasters.iter().map(|r| r.data()).collect();

    // Compute per-pixel median
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
                    // Skip explicit nodata values (non-NaN case)
                    if let Some(nd) = nodata {
                        if nd.is_finite() && (v - nd).abs() < 1e-10 {
                            continue;
                        }
                    }
                    values.push(v);
                }
                out_row[col] = if values.is_empty() {
                    f64::NAN
                } else {
                    values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = values.len() / 2;
                    if values.len() % 2 == 0 {
                        (values[mid - 1] + values[mid]) / 2.0
                    } else {
                        values[mid]
                    }
                };
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
}
