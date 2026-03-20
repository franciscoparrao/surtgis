//! Cloud masking for satellite imagery
//!
//! Apply Sentinel-2 Scene Classification Layer (SCL) to mask clouds and shadows.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Default SCL classes to keep (clear pixels):
/// 4=Vegetation, 5=Bare soil, 6=Water, 11=Snow/Ice
pub const SCL_VALID_DEFAULT: &[u8] = &[4, 5, 6, 11];

/// Apply cloud mask using Sentinel-2 SCL (Scene Classification Layer).
///
/// Pixels where the SCL value is NOT in `valid_classes` are set to NaN.
///
/// # SCL Classes
/// - 0: No data
/// - 1: Saturated/defective
/// - 2: Dark area
/// - 3: Cloud shadow
/// - 4: Vegetation
/// - 5: Bare soil
/// - 6: Water
/// - 7: Cloud low probability
/// - 8: Cloud medium probability
/// - 9: Cloud high probability
/// - 10: Thin cirrus
/// - 11: Snow/Ice
///
/// # Arguments
/// * `data` - Input raster to mask
/// * `scl` - SCL classification raster (same dimensions, integer values as f64)
/// * `valid_classes` - SCL class values to keep
pub fn cloud_mask_scl(
    data: &Raster<f64>,
    scl: &Raster<f64>,
    valid_classes: &[u8],
) -> Result<Raster<f64>> {
    let (rows, cols) = data.shape();
    let (sr, sc) = scl.shape();

    let data_arr = data.data();
    let scl_arr = scl.data();

    // Scale factors for nearest-neighbor resampling when SCL has different
    // resolution (e.g., S2 SCL at 20m vs data at 10m)
    let row_scale = sr as f64 / rows as f64;
    let col_scale = sc as f64 / cols as f64;

    let mut output = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    output
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            // Map data row to SCL row (nearest neighbor)
            let scl_row = ((row as f64 * row_scale).floor() as usize).min(sr - 1);
            for col in 0..cols {
                let scl_col = ((col as f64 * col_scale).floor() as usize).min(sc - 1);
                let scl_val = scl_arr[[scl_row, scl_col]] as u8;
                if valid_classes.contains(&scl_val) {
                    out_row[col] = data_arr[[row, col]];
                }
            }
        });

    let mut result = Raster::from_array(output);
    result.set_transform(data.transform().clone());
    result.set_nodata(Some(f64::NAN));
    if let Some(crs) = data.crs() {
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
        r
    }

    #[test]
    fn test_cloud_mask_scl() {
        // Data raster: all values = 100.0
        let data = make_raster(vec![
            vec![100.0, 100.0, 100.0],
            vec![100.0, 100.0, 100.0],
        ]);
        // SCL: 4=veg, 9=cloud, 5=bare soil, 3=shadow, 6=water, 8=cloud_med
        let scl = make_raster(vec![
            vec![4.0, 9.0, 5.0],
            vec![3.0, 6.0, 8.0],
        ]);

        let result = cloud_mask_scl(&data, &scl, SCL_VALID_DEFAULT).unwrap();
        let d = result.data();

        assert!((d[[0, 0]] - 100.0).abs() < 1e-10); // class 4 = keep
        assert!(d[[0, 1]].is_nan());                  // class 9 = cloud → NaN
        assert!((d[[0, 2]] - 100.0).abs() < 1e-10); // class 5 = keep
        assert!(d[[1, 0]].is_nan());                  // class 3 = shadow → NaN
        assert!((d[[1, 1]] - 100.0).abs() < 1e-10); // class 6 = keep
        assert!(d[[1, 2]].is_nan());                  // class 8 = cloud → NaN
    }

    #[test]
    fn test_cloud_mask_different_resolution() {
        // Data at 10m: 4x4
        let data = make_raster(vec![
            vec![100.0, 100.0, 100.0, 100.0],
            vec![100.0, 100.0, 100.0, 100.0],
            vec![100.0, 100.0, 100.0, 100.0],
            vec![100.0, 100.0, 100.0, 100.0],
        ]);
        // SCL at 20m: 2x2 (half resolution)
        // top-left=4 (veg), top-right=9 (cloud)
        // bottom-left=9 (cloud), bottom-right=5 (soil)
        let scl = make_raster(vec![
            vec![4.0, 9.0],
            vec![9.0, 5.0],
        ]);

        let result = cloud_mask_scl(&data, &scl, SCL_VALID_DEFAULT).unwrap();
        let d = result.data();

        // Top-left quadrant (SCL=4=veg) → keep
        assert!((d[[0, 0]] - 100.0).abs() < 1e-10);
        assert!((d[[0, 1]] - 100.0).abs() < 1e-10);
        assert!((d[[1, 0]] - 100.0).abs() < 1e-10);
        assert!((d[[1, 1]] - 100.0).abs() < 1e-10);

        // Top-right quadrant (SCL=9=cloud) → NaN
        assert!(d[[0, 2]].is_nan());
        assert!(d[[0, 3]].is_nan());
        assert!(d[[1, 2]].is_nan());
        assert!(d[[1, 3]].is_nan());

        // Bottom-left quadrant (SCL=9=cloud) → NaN
        assert!(d[[2, 0]].is_nan());
        assert!(d[[3, 0]].is_nan());

        // Bottom-right quadrant (SCL=5=soil) → keep
        assert!((d[[2, 2]] - 100.0).abs() < 1e-10);
        assert!((d[[3, 3]] - 100.0).abs() < 1e-10);
    }
}
