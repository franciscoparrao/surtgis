//! Morphological opening (erosion followed by dilation)
//!
//! Removes small bright features (noise, spikes) while preserving
//! the overall shape and size of larger bright regions.

use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

use super::dilate::dilate;
use super::element::StructuringElement;
use super::erode::erode;

/// Parameters for morphological opening
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct OpeningParams {
    /// Structuring element shape
    pub element: StructuringElement,
}


/// Opening algorithm
#[derive(Debug, Clone, Default)]
pub struct Opening;

impl Algorithm for Opening {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = OpeningParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Opening"
    }

    fn description(&self) -> &'static str {
        "Morphological opening (erosion then dilation) to remove small bright features"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        opening(&input, &params.element)
    }
}

/// Perform morphological opening on a raster
///
/// Opening = erode then dilate. Removes small bright features
/// (spots, thin protrusions) while preserving the overall shape
/// of larger structures.
///
/// The resulting NaN border width is `2 * radius` because erosion
/// and dilation each add a `radius`-wide NaN border.
///
/// # Arguments
/// * `raster` - Input raster
/// * `element` - Structuring element defining the neighborhood shape
pub fn opening(raster: &Raster<f64>, element: &StructuringElement) -> Result<Raster<f64>> {
    let eroded = erode(raster, element)?;
    dilate(&eroded, element)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn make_raster(rows: usize, cols: usize, value: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, value);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn test_opening_uniform() {
        let raster = make_raster(11, 11, 5.0);
        let result = opening(&raster, &StructuringElement::Square(1)).unwrap();
        // Interior should be unchanged
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "Uniform opening should preserve value, got {}",
            val
        );
    }

    #[test]
    fn test_opening_removes_bright_spot() {
        let mut raster = make_raster(11, 11, 5.0);
        // Single bright pixel (smaller than element)
        raster.set(5, 5, 100.0).unwrap();

        let result = opening(&raster, &StructuringElement::Square(1)).unwrap();
        // Opening should remove the single bright spot
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "Opening should remove single bright pixel, got {}",
            val
        );
    }

    #[test]
    fn test_opening_preserves_large_bright_region() {
        let mut raster = make_raster(11, 11, 5.0);
        // Large bright block (3x3) - bigger than element kernel
        for r in 4..7 {
            for c in 4..7 {
                raster.set(r, c, 100.0).unwrap();
            }
        }

        let result = opening(&raster, &StructuringElement::Square(1)).unwrap();
        // Center of large region should remain bright
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 100.0).abs() < 1e-10,
            "Opening should preserve large bright region center, got {}",
            val
        );
    }

    #[test]
    fn test_opening_border_width() {
        let raster = make_raster(11, 11, 5.0);
        let result = opening(&raster, &StructuringElement::Square(1)).unwrap();
        // Border of 2*1=2 should be NaN
        assert!(result.get(0, 5).unwrap().is_nan());
        assert!(result.get(1, 5).unwrap().is_nan());
        // Row 2 should be valid
        let val = result.get(2, 5).unwrap();
        assert!(!val.is_nan(), "Row 2 should be valid, got NaN");
    }
}
