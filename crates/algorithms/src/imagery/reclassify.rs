//! Raster reclassification
//!
//! Reclassify raster values based on value ranges or exact matches.

use ndarray::Array2;
use rayon::prelude::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// A reclassification entry mapping an input range to an output value
#[derive(Debug, Clone)]
pub struct ReclassEntry {
    /// Minimum value (inclusive)
    pub min: f64,
    /// Maximum value (exclusive, except for the last class)
    pub max: f64,
    /// Output value for this class
    pub value: f64,
}

impl ReclassEntry {
    /// Create a new reclassification entry
    pub fn new(min: f64, max: f64, value: f64) -> Self {
        Self { min, max, value }
    }
}

/// Parameters for reclassification
#[derive(Debug, Clone)]
pub struct ReclassifyParams {
    /// Reclassification table (must be sorted by min value)
    pub classes: Vec<ReclassEntry>,
    /// Value for cells that don't match any class
    pub default_value: f64,
}

impl Default for ReclassifyParams {
    fn default() -> Self {
        Self {
            classes: Vec::new(),
            default_value: f64::NAN,
        }
    }
}

/// Reclassify raster values based on a classification table.
///
/// Each cell value is compared against the classification entries.
/// The first matching entry (min <= value < max) determines the output.
///
/// # Arguments
/// * `raster` - Input raster
/// * `params` - Classification table and default value
///
/// # Example
/// ```ignore
/// // NDVI classification
/// let params = ReclassifyParams {
///     classes: vec![
///         ReclassEntry::new(-1.0, 0.0, 1.0),  // Water
///         ReclassEntry::new(0.0, 0.2, 2.0),   // Bare soil
///         ReclassEntry::new(0.2, 0.5, 3.0),   // Sparse vegetation
///         ReclassEntry::new(0.5, 1.0, 4.0),   // Dense vegetation
///     ],
///     default_value: 0.0,
/// };
/// let classified = reclassify(&ndvi_raster, params)?;
/// ```
pub fn reclassify(raster: &Raster<f64>, params: ReclassifyParams) -> Result<Raster<f64>> {
    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();
    let classes = &params.classes;
    let default = params.default_value;

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            for col in 0..cols {
                let val = unsafe { raster.get_unchecked(row, col) };

                if val.is_nan() {
                    continue;
                }
                if let Some(nd) = nodata {
                    if (val - nd).abs() < f64::EPSILON {
                        continue;
                    }
                }

                // Find matching class
                let mut classified = default;
                for entry in classes {
                    if val >= entry.min && val < entry.max {
                        classified = entry.value;
                        break;
                    }
                }

                // Special case: last class includes max value
                if classified == default && !classes.is_empty() {
                    let last = &classes[classes.len() - 1];
                    if (val - last.max).abs() < 1e-10 {
                        classified = last.value;
                    }
                }

                row_data[col] = classified;
            }
            row_data
        })
        .collect();

    let mut output = raster.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_ndvi_raster() -> Raster<f64> {
        // NDVI-like values across a 5x5 grid
        let values = vec![
            -0.5, -0.2, 0.0, 0.1, 0.15,
            0.2, 0.25, 0.3, 0.35, 0.4,
            0.45, 0.5, 0.55, 0.6, 0.65,
            0.7, 0.75, 0.8, 0.85, 0.9,
            -0.1, 0.0, 0.1, 0.5, 1.0,
        ];
        let mut r = Raster::from_vec(values, 5, 5).unwrap();
        r.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        r
    }

    fn ndvi_classes() -> ReclassifyParams {
        ReclassifyParams {
            classes: vec![
                ReclassEntry::new(-1.0, 0.0, 1.0),  // Water
                ReclassEntry::new(0.0, 0.2, 2.0),   // Bare soil
                ReclassEntry::new(0.2, 0.5, 3.0),   // Sparse vegetation
                ReclassEntry::new(0.5, 1.01, 4.0),   // Dense vegetation
            ],
            default_value: 0.0,
        }
    }

    #[test]
    fn test_reclassify_water() {
        let raster = make_ndvi_raster();
        let result = reclassify(&raster, ndvi_classes()).unwrap();

        // -0.5 → class 1 (water)
        assert_eq!(result.get(0, 0).unwrap(), 1.0);
        // -0.2 → class 1 (water)
        assert_eq!(result.get(0, 1).unwrap(), 1.0);
    }

    #[test]
    fn test_reclassify_bare_soil() {
        let raster = make_ndvi_raster();
        let result = reclassify(&raster, ndvi_classes()).unwrap();

        // 0.0 → class 2 (bare soil)
        assert_eq!(result.get(0, 2).unwrap(), 2.0);
        // 0.1 → class 2 (bare soil)
        assert_eq!(result.get(0, 3).unwrap(), 2.0);
    }

    #[test]
    fn test_reclassify_vegetation() {
        let raster = make_ndvi_raster();
        let result = reclassify(&raster, ndvi_classes()).unwrap();

        // 0.3 → class 3 (sparse vegetation)
        assert_eq!(result.get(1, 2).unwrap(), 3.0);
        // 0.7 → class 4 (dense vegetation)
        assert_eq!(result.get(3, 0).unwrap(), 4.0);
    }

    #[test]
    fn test_reclassify_nodata() {
        let mut raster = make_ndvi_raster();
        raster.set(0, 0, f64::NAN).unwrap();

        let result = reclassify(&raster, ndvi_classes()).unwrap();
        assert!(result.get(0, 0).unwrap().is_nan());
    }

    #[test]
    fn test_reclassify_empty_table() {
        let raster = make_ndvi_raster();
        let params = ReclassifyParams {
            classes: vec![],
            default_value: -1.0,
        };

        let result = reclassify(&raster, params).unwrap();

        // All valid cells should get default value
        assert_eq!(result.get(2, 2).unwrap(), -1.0);
    }
}
