//! Cloud masking for satellite imagery
//!
//! Pluggable cloud masking strategies for different satellite collections.
//! - Sentinel-2: SCL (Scene Classification Layer, categorical 0-11)
//! - Landsat C2: QA_PIXEL (bitmask with cloud/shadow/snow bits)
//! - HLS S30/L30: Fmask (bitmask, different bit assignment from Landsat QA)
//! - Sentinel-1 SAR: None (radar penetrates clouds)

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use surtgis_core::Result;

/// Default SCL classes to keep (clear pixels):
/// 4=Vegetation, 5=Bare soil, 6=Water, 11=Snow/Ice
pub const SCL_VALID_DEFAULT: &[u8] = &[4, 5, 6, 11];

/// Trait for collection-specific cloud masking strategies
pub trait CloudMaskStrategy: Send + Sync {
    /// Apply cloud mask to data raster using mask asset
    ///
    /// # Arguments
    /// * `data` - Raster to mask (e.g., reflectance band)
    /// * `mask_asset` - Mask raster (e.g., SCL for S2, QA_PIXEL for Landsat)
    ///
    /// Returns raster with clouds/shadows set to NaN
    fn mask(&self, data: &Raster<f64>, mask_asset: &Raster<f64>) -> Result<Raster<f64>>;

    /// Description for logging (e.g., "S2 SCL" or "Landsat QA_PIXEL")
    fn description(&self) -> &str;
}

/// Sentinel-2 SCL (Scene Classification Layer) cloud masking
#[derive(Clone)]
pub struct S2SclMask {
    /// SCL class values to keep (e.g., [4, 5, 6, 11])
    pub keep_classes: Vec<u8>,
}

impl S2SclMask {
    /// Create with default valid classes: vegetation, bare soil, water, snow
    pub fn new() -> Self {
        Self {
            keep_classes: SCL_VALID_DEFAULT.to_vec(),
        }
    }

    /// Create with custom valid classes
    pub fn with_classes(keep_classes: Vec<u8>) -> Self {
        Self { keep_classes }
    }
}

impl Default for S2SclMask {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudMaskStrategy for S2SclMask {
    fn mask(&self, data: &Raster<f64>, scl: &Raster<f64>) -> Result<Raster<f64>> {
        cloud_mask_scl(data, scl, &self.keep_classes)
    }

    fn description(&self) -> &str {
        "S2 SCL (Scene Classification Layer)"
    }
}

/// Landsat Collection 2 QA_PIXEL cloud masking
#[derive(Clone)]
pub struct LandsatQaMask {
    /// Bits to exclude (fill, cloud, shadow, snow, etc.)
    /// Bit 0: fill, Bit 1: dilated cloud, Bit 3: cloud, Bit 4: cloud shadow
    /// Default: 0b0001_1011 = bits 0, 1, 3, 4 set → exclude fill|cloud|shadow
    pub exclude_bits: u16,
}

impl LandsatQaMask {
    /// Create with default excluded bits: fill (0) | dilated cloud (1) | cloud (3) | shadow (4)
    pub fn new() -> Self {
        Self {
            exclude_bits: 0b0001_1011, // bits 0, 1, 3, 4
        }
    }

    /// Create with custom excluded bits
    pub fn with_bits(exclude_bits: u16) -> Self {
        Self { exclude_bits }
    }
}

impl Default for LandsatQaMask {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudMaskStrategy for LandsatQaMask {
    fn mask(&self, data: &Raster<f64>, qa: &Raster<f64>) -> Result<Raster<f64>> {
        cloud_mask_qa_pixel(data, qa, self.exclude_bits)
    }

    fn description(&self) -> &str {
        "Landsat C2 QA_PIXEL (bitmask)"
    }
}

/// Harmonized Landsat-Sentinel (HLS) Fmask cloud masking.
///
/// HLS S30/L30 v2 products carry an Fmask band whose pixel value is a
/// bitmask with the following layout (see the HLS v2 User Guide,
/// LP DAAC, NASA Earthdata):
///
/// - Bit 0: Cirrus
/// - Bit 1: Cloud
/// - Bit 2: Adjacent to cloud / shadow
/// - Bit 3: Cloud shadow
/// - Bit 4: Snow / Ice
/// - Bit 5: Water
/// - Bits 6-7: Aerosol level (00 climatology, 01 low, 10 moderate, 11 high)
///
/// For training Prithvi-EO-2.0 and other HLS-pre-trained models, the
/// standard NASA-IMPACT preprocessing masks any pixel with cloud (bit 1),
/// adjacent-to-cloud (bit 2), or cloud shadow (bit 3) set. Cirrus (bit 0)
/// is *not* excluded by default — HLS Fmask flags cirrus aggressively,
/// and Prithvi tolerates it. Snow (bit 4) is *kept* by default since most
/// downstream tasks treat snow as a valid land-cover category. Water
/// (bit 5) is informational only and never masks pixels.
#[derive(Clone)]
pub struct HlsFmask {
    /// Bits to exclude. Default: 0b0000_1110 = bits 1, 2, 3
    /// (cloud, adjacent-to-cloud, cloud shadow).
    pub exclude_bits: u16,
}

impl HlsFmask {
    /// Default: exclude cloud (bit 1), adjacent-to-cloud (bit 2), cloud shadow (bit 3).
    /// Cirrus (bit 0) kept; snow (bit 4) kept; water (bit 5) ignored.
    pub fn new() -> Self {
        Self {
            exclude_bits: 0b0000_1110,
        }
    }

    /// Conservative variant that also excludes cirrus and snow.
    /// Use this when downstream is sensitive to cirrus contamination or
    /// when snow is not a target class.
    pub fn strict() -> Self {
        Self {
            exclude_bits: 0b0001_1111,
        }
    }

    /// Custom excluded bits — see struct doc for the HLS bit layout.
    pub fn with_bits(exclude_bits: u16) -> Self {
        Self { exclude_bits }
    }
}

impl Default for HlsFmask {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudMaskStrategy for HlsFmask {
    fn mask(&self, data: &Raster<f64>, fmask: &Raster<f64>) -> Result<Raster<f64>> {
        cloud_mask_hls_fmask(data, fmask, self.exclude_bits)
    }

    fn description(&self) -> &str {
        "HLS Fmask (HLS S30/L30 bitmask)"
    }
}

/// SAR cloud masking strategy (no-op: radar penetrates clouds)
#[derive(Clone)]
pub struct NoCloudMask;

impl CloudMaskStrategy for NoCloudMask {
    fn mask(&self, data: &Raster<f64>, _mask_asset: &Raster<f64>) -> Result<Raster<f64>> {
        // SAR data is not affected by clouds, return as-is
        Ok(data.clone())
    }

    fn description(&self) -> &str {
        "None (SAR penetrates clouds)"
    }
}

/// Apply cloud mask using Landsat QA_PIXEL bitmask
///
/// Sets pixels where any excluded bit is set to NaN.
/// Also excludes fill pixels (data == 0 or QA == 0).
///
/// QA_PIXEL bits:
/// - Bit 0: Fill
/// - Bit 1: Dilated Cloud
/// - Bit 3: Cloud
/// - Bit 4: Cloud Shadow
fn cloud_mask_qa_pixel(
    data: &Raster<f64>,
    qa: &Raster<f64>,
    exclude_bits: u16,
) -> Result<Raster<f64>> {
    let (rows, cols) = data.shape();
    let (qr, qc) = qa.shape();

    let data_arr = data.data();
    let qa_arr = qa.data();

    // Nearest-neighbor resampling for potential resolution mismatch
    let row_scale = qr as f64 / rows as f64;
    let col_scale = qc as f64 / cols as f64;

    let mut output = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    output
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            let qa_row = ((row as f64 * row_scale).floor() as usize).min(qr - 1);
            for col in 0..cols {
                let qa_col = ((col as f64 * col_scale).floor() as usize).min(qc - 1);
                let qa_val = qa_arr[[qa_row, qa_col]] as u16;
                let data_val = data_arr[[row, col]];
                // Skip fill pixels: QA==0 (no classification), data==0 (fill value),
                // or any excluded bit set
                if qa_val == 0 || data_val == 0.0 || (qa_val & exclude_bits) != 0 {
                    continue;
                }
                out_row[col] = data_val;
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
                // SCL=0 means no classification data available.
                // Pass the pixel through (assume clear) rather than discard.
                if scl_val == 0 || valid_classes.contains(&scl_val) {
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

/// Apply cloud mask using HLS Fmask bitmask (HLS S30/L30 v2).
///
/// HLS Fmask differs from Landsat C2 QA_PIXEL: same bitmask shape but
/// different bit assignments (cloud is bit 1 in HLS, bit 3 in Landsat).
/// See [`HlsFmask`] doc for the full layout.
///
/// Pixels where any excluded bit is set are set to NaN. Unlike Landsat
/// QA_PIXEL we do NOT treat Fmask == 0 as no-data: 0 means "clear, all
/// bits unset" which is the most-confidence-clear pixel HLS reports.
pub fn cloud_mask_hls_fmask(
    data: &Raster<f64>,
    fmask: &Raster<f64>,
    exclude_bits: u16,
) -> Result<Raster<f64>> {
    let (rows, cols) = data.shape();
    let (fr, fc) = fmask.shape();
    let data_arr = data.data();
    let f_arr = fmask.data();
    let row_scale = fr as f64 / rows as f64;
    let col_scale = fc as f64 / cols as f64;
    let mut output = Array2::<f64>::from_elem((rows, cols), f64::NAN);

    output
        .as_slice_mut()
        .unwrap()
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(row, out_row)| {
            let f_row = ((row as f64 * row_scale).floor() as usize).min(fr - 1);
            for col in 0..cols {
                let f_col = ((col as f64 * col_scale).floor() as usize).min(fc - 1);
                let f_val = f_arr[[f_row, f_col]] as u16;
                if (f_val & exclude_bits) == 0 {
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
    use surtgis_core::GeoTransform;
    use surtgis_core::raster::Raster;

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
        let data = make_raster(vec![vec![100.0, 100.0, 100.0], vec![100.0, 100.0, 100.0]]);
        // SCL: 4=veg, 9=cloud, 5=bare soil, 3=shadow, 6=water, 8=cloud_med
        let scl = make_raster(vec![vec![4.0, 9.0, 5.0], vec![3.0, 6.0, 8.0]]);

        let result = cloud_mask_scl(&data, &scl, SCL_VALID_DEFAULT).unwrap();
        let d = result.data();

        assert!((d[[0, 0]] - 100.0).abs() < 1e-10); // class 4 = keep
        assert!(d[[0, 1]].is_nan()); // class 9 = cloud → NaN
        assert!((d[[0, 2]] - 100.0).abs() < 1e-10); // class 5 = keep
        assert!(d[[1, 0]].is_nan()); // class 3 = shadow → NaN
        assert!((d[[1, 1]] - 100.0).abs() < 1e-10); // class 6 = keep
        assert!(d[[1, 2]].is_nan()); // class 8 = cloud → NaN
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
        let scl = make_raster(vec![vec![4.0, 9.0], vec![9.0, 5.0]]);

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

    #[test]
    fn test_s2_scl_mask_trait() {
        // Test CloudMaskStrategy trait for S2
        let data = make_raster(vec![vec![100.0, 100.0], vec![100.0, 100.0]]);
        let scl = make_raster(vec![vec![4.0, 9.0], vec![5.0, 8.0]]);

        let strategy = S2SclMask::new();
        let result = strategy.mask(&data, &scl).unwrap();
        let d = result.data();

        assert!((d[[0, 0]] - 100.0).abs() < 1e-10); // class 4 = keep
        assert!(d[[0, 1]].is_nan()); // class 9 = cloud → NaN
        assert!((d[[1, 0]] - 100.0).abs() < 1e-10); // class 5 = keep
        assert!(d[[1, 1]].is_nan()); // class 8 = cloud → NaN
    }

    #[test]
    fn test_landsat_qa_mask_trait() {
        // Test CloudMaskStrategy trait for Landsat
        // Default exclude bits: 0 (fill), 1 (dilated cloud), 3 (cloud), 4 (shadow)
        // Also excludes: qa==0 (no classification) and data==0 (fill value)
        //
        // QA=0 → exclude (no classification available)
        // QA=1 → exclude (bit 0 = fill)
        // QA=2 → exclude (bit 1 = dilated cloud)
        // QA=8 → exclude (bit 3 = cloud)
        // QA=16 → exclude (bit 4 = shadow)
        // QA=64 → keep (bit 6 = clear, no excluded bits)

        let data = make_raster(vec![vec![100.0, 100.0, 100.0], vec![100.0, 100.0, 0.0]]);
        let qa = make_raster(vec![
            vec![0.0, 1.0, 64.0], // no-classification, fill, clear
            vec![2.0, 8.0, 64.0], // dilated cloud, cloud, clear+data=0
        ]);

        let strategy = LandsatQaMask::new();
        let result = strategy.mask(&data, &qa).unwrap();
        let d = result.data();

        assert!(d[[0, 0]].is_nan()); // QA=0 → NaN (no classification)
        assert!(d[[0, 1]].is_nan()); // QA=1 (fill bit) → NaN
        assert!((d[[0, 2]] - 100.0).abs() < 1e-10); // QA=64 (clear) → keep
        assert!(d[[1, 0]].is_nan()); // QA=2 (dilated cloud) → NaN
        assert!(d[[1, 1]].is_nan()); // QA=8 (cloud) → NaN
        assert!(d[[1, 2]].is_nan()); // QA=64 but data=0 (fill value) → NaN
    }

    #[test]
    fn test_hls_fmask_default() {
        // HLS Fmask bit layout:
        // bit 0=cirrus, bit 1=cloud, bit 2=adjacent-to-cloud,
        // bit 3=cloud shadow, bit 4=snow, bit 5=water
        //
        // Default exclude bits: 0b0000_1110 = cloud | adjacent | shadow
        // Pixels expected to KEEP: 0 (clear), 1 (cirrus only), 16 (snow), 32 (water)
        // Pixels expected to DROP: 2 (cloud), 4 (adjacent), 8 (shadow), 14 (all three)

        let data = make_raster(vec![
            vec![100.0, 100.0, 100.0, 100.0],
            vec![100.0, 100.0, 100.0, 100.0],
        ]);
        let fmask = make_raster(vec![
            // clear, cirrus only, cloud, adjacent
            vec![0.0, 1.0, 2.0, 4.0],
            // shadow, all three, snow, water
            vec![8.0, 14.0, 16.0, 32.0],
        ]);
        let strategy = HlsFmask::new();
        let result = strategy.mask(&data, &fmask).unwrap();
        let d = result.data();

        assert!((d[[0, 0]] - 100.0).abs() < 1e-10, "clear should be kept");
        assert!(
            (d[[0, 1]] - 100.0).abs() < 1e-10,
            "cirrus alone should be kept by default"
        );
        assert!(d[[0, 2]].is_nan(), "cloud should be masked");
        assert!(d[[0, 3]].is_nan(), "adjacent-to-cloud should be masked");
        assert!(d[[1, 0]].is_nan(), "cloud shadow should be masked");
        assert!(d[[1, 1]].is_nan(), "cloud|adjacent|shadow should be masked");
        assert!(
            (d[[1, 2]] - 100.0).abs() < 1e-10,
            "snow should be kept by default"
        );
        assert!(
            (d[[1, 3]] - 100.0).abs() < 1e-10,
            "water should be kept (informational)"
        );
    }

    #[test]
    fn test_hls_fmask_strict() {
        // Strict variant excludes cirrus and snow on top of defaults.
        let data = make_raster(vec![vec![100.0, 100.0, 100.0, 100.0]]);
        // clear, cirrus, snow, water
        let fmask = make_raster(vec![vec![0.0, 1.0, 16.0, 32.0]]);

        let strategy = HlsFmask::strict();
        let result = strategy.mask(&data, &fmask).unwrap();
        let d = result.data();

        assert!((d[[0, 0]] - 100.0).abs() < 1e-10, "clear should be kept");
        assert!(d[[0, 1]].is_nan(), "cirrus should be masked by strict()");
        assert!(d[[0, 2]].is_nan(), "snow should be masked by strict()");
        assert!(
            (d[[0, 3]] - 100.0).abs() < 1e-10,
            "water is never excluded (bit 5 not in mask)"
        );
    }

    #[test]
    fn test_no_cloud_mask() {
        // Test CloudMaskStrategy trait for SAR (no-op)
        let data = make_raster(vec![vec![100.0, 200.0], vec![300.0, 400.0]]);
        let dummy_mask = make_raster(vec![vec![0.0, 1.0], vec![1.0, 0.0]]);

        let strategy = NoCloudMask;
        let result = strategy.mask(&data, &dummy_mask).unwrap();
        let d = result.data();

        // All values should be preserved (no masking)
        assert!((d[[0, 0]] - 100.0).abs() < 1e-10);
        assert!((d[[0, 1]] - 200.0).abs() < 1e-10);
        assert!((d[[1, 0]] - 300.0).abs() < 1e-10);
        assert!((d[[1, 1]] - 400.0).abs() < 1e-10);
    }
}
