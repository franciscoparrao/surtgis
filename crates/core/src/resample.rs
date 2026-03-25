//! Raster resampling: align a raster to a reference grid.
//!
//! Resamples a source raster to match the exact grid (origin, cell size,
//! dimensions) of a reference raster. Essential for stacking rasters from
//! different sources (e.g., Copernicus DEM at 30m + Sentinel-2 at 10m).

use ndarray::Array2;

use crate::error::Result;
use crate::raster::Raster;

/// Interpolation method for resampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMethod {
    /// Nearest neighbor — fast, preserves discrete values (classification, SCL)
    NearestNeighbor,
    /// Bilinear interpolation — smooth, best for continuous data (elevation, indices)
    Bilinear,
}

/// Resample a raster to match the grid of a reference raster.
///
/// The output has the same dimensions, transform, and CRS as `reference`.
/// Each output pixel is computed by mapping its geographic coordinate back
/// to the source raster and interpolating.
///
/// # Arguments
/// * `source` - Raster to resample
/// * `reference` - Reference raster defining the target grid
/// * `method` - Interpolation method
///
/// # Example
/// ```ignore
/// // Resample Sentinel-2 band (10m) to match DEM grid (30m)
/// let aligned = resample_to_grid(&s2_band, &dem, ResampleMethod::Bilinear)?;
/// ```
pub fn resample_to_grid(
    source: &Raster<f64>,
    reference: &Raster<f64>,
    method: ResampleMethod,
) -> Result<Raster<f64>> {
    let (out_rows, out_cols) = reference.shape();
    let ref_gt = reference.transform();
    let src_gt = source.transform();
    let (src_rows, src_cols) = source.shape();
    let src_data = source.data();
    let nodata = source.nodata();

    let mut output = Array2::<f64>::from_elem((out_rows, out_cols), f64::NAN);

    for out_row in 0..out_rows {
        for out_col in 0..out_cols {
            // Geographic coordinate of output pixel center
            let geo_x = ref_gt.origin_x + (out_col as f64 + 0.5) * ref_gt.pixel_width;
            let geo_y = ref_gt.origin_y + (out_row as f64 + 0.5) * ref_gt.pixel_height;

            // Map to source pixel coordinates (continuous)
            let src_col_f = (geo_x - src_gt.origin_x) / src_gt.pixel_width - 0.5;
            let src_row_f = (geo_y - src_gt.origin_y) / src_gt.pixel_height - 0.5;

            // Check if source pixel is within bounds
            if src_col_f < -0.5
                || src_row_f < -0.5
                || src_col_f >= src_cols as f64 - 0.5
                || src_row_f >= src_rows as f64 - 0.5
            {
                continue; // NaN (outside source extent)
            }

            output[[out_row, out_col]] = match method {
                ResampleMethod::NearestNeighbor => {
                    let r = src_row_f.round().max(0.0) as usize;
                    let c = src_col_f.round().max(0.0) as usize;
                    let r = r.min(src_rows - 1);
                    let c = c.min(src_cols - 1);
                    let v = src_data[[r, c]];
                    if is_nodata(v, nodata) { f64::NAN } else { v }
                }
                ResampleMethod::Bilinear => {
                    let r0 = src_row_f.floor().max(0.0) as usize;
                    let c0 = src_col_f.floor().max(0.0) as usize;
                    let r1 = (r0 + 1).min(src_rows - 1);
                    let c1 = (c0 + 1).min(src_cols - 1);

                    let dr = src_row_f - r0 as f64;
                    let dc = src_col_f - c0 as f64;
                    let dr = dr.clamp(0.0, 1.0);
                    let dc = dc.clamp(0.0, 1.0);

                    let v00 = src_data[[r0, c0]];
                    let v01 = src_data[[r0, c1]];
                    let v10 = src_data[[r1, c0]];
                    let v11 = src_data[[r1, c1]];

                    // If any neighbor is nodata, fall back to nearest
                    if is_nodata(v00, nodata)
                        || is_nodata(v01, nodata)
                        || is_nodata(v10, nodata)
                        || is_nodata(v11, nodata)
                    {
                        // Nearest neighbor fallback
                        let r = src_row_f.round().max(0.0) as usize;
                        let c = src_col_f.round().max(0.0) as usize;
                        let v = src_data[[r.min(src_rows - 1), c.min(src_cols - 1)]];
                        if is_nodata(v, nodata) { f64::NAN } else { v }
                    } else {
                        // Bilinear interpolation
                        let top = v00 * (1.0 - dc) + v01 * dc;
                        let bot = v10 * (1.0 - dc) + v11 * dc;
                        top * (1.0 - dr) + bot * dr
                    }
                }
            };
        }
    }

    let mut result = Raster::from_array(output);
    result.set_transform(ref_gt.clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = reference.crs() {
        result.set_crs(Some(crs.clone()));
    }
    Ok(result)
}

fn is_nodata(v: f64, nodata: Option<f64>) -> bool {
    if !v.is_finite() {
        return true;
    }
    if let Some(nd) = nodata {
        if nd.is_finite() && (v - nd).abs() < 1e-10 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::GeoTransform;

    fn make_raster(rows: usize, cols: usize, value: f64, gt: GeoTransform) -> Raster<f64> {
        let arr = Array2::from_elem((rows, cols), value);
        let mut r = Raster::from_array(arr);
        r.set_transform(gt);
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_resample_same_grid() {
        // Same grid → output equals input
        let gt = GeoTransform::new(0.0, 100.0, 10.0, -10.0);
        let src = make_raster(10, 10, 42.0, gt);
        let reference = make_raster(10, 10, 0.0, gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::NearestNeighbor).unwrap();
        assert!((result.data()[[5, 5]] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_downsample() {
        // Source 10m → reference 30m (3x downsample)
        let src_gt = GeoTransform::new(0.0, 300.0, 10.0, -10.0);
        let ref_gt = GeoTransform::new(0.0, 300.0, 30.0, -30.0);

        // Source: 30x30 at 10m, all value 100.0
        let src = make_raster(30, 30, 100.0, src_gt);
        // Reference: 10x10 at 30m
        let reference = make_raster(10, 10, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::Bilinear).unwrap();
        assert_eq!(result.shape(), (10, 10));
        // All values should be 100.0 (uniform source)
        assert!((result.data()[[5, 5]] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_resample_with_offset() {
        // Source and reference have different origins (the real-world case)
        let src_gt = GeoTransform::new(100.0, 500.0, 10.0, -10.0);
        let ref_gt = GeoTransform::new(115.0, 485.0, 30.0, -30.0); // offset by 15m,15m

        // Source: gradient where value = col index
        let mut src_data = Array2::<f64>::zeros((40, 40));
        for r in 0..40 {
            for c in 0..40 {
                src_data[[r, c]] = c as f64;
            }
        }
        let mut src = Raster::from_array(src_data);
        src.set_transform(src_gt);
        src.set_nodata(Some(f64::NAN));

        let reference = make_raster(10, 10, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::Bilinear).unwrap();
        assert_eq!(result.shape(), (10, 10));
        // First output pixel center is at (115+15, 485-15) = (130, 470)
        // In source: col = (130 - 100) / 10 - 0.5 = 2.5 → bilinear between col 2 and 3
        let v = result.data()[[0, 0]];
        assert!(v.is_finite());
        assert!((v - 2.5).abs() < 0.5, "Expected ~2.5, got {}", v);
    }

    #[test]
    fn test_resample_outside_bounds() {
        // Reference extends beyond source → NaN for out-of-bounds pixels
        let src_gt = GeoTransform::new(100.0, 200.0, 10.0, -10.0);
        let ref_gt = GeoTransform::new(0.0, 300.0, 10.0, -10.0); // extends far beyond source

        let src = make_raster(10, 10, 42.0, src_gt);
        let reference = make_raster(30, 30, 0.0, ref_gt);

        let result = resample_to_grid(&src, &reference, ResampleMethod::NearestNeighbor).unwrap();
        // Pixel at (0,0) maps to geo (5, 295), source origin is (100, 200) → outside
        assert!(result.data()[[0, 0]].is_nan());
        // Pixel near source area should have value
        // Source covers x=[100,200], y=[100,200]. ref pixel at col=10,row=10 → geo(105,195)
        let v = result.data()[[10, 10]];
        assert!(v.is_finite());
    }
}
