//! Morphological closing (dilation followed by erosion)
//!
//! Fills small dark gaps and holes while preserving the overall
//! shape and size of larger dark regions.

use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

use super::dilate::dilate;
use super::element::StructuringElement;
use super::erode::erode;

/// Parameters for morphological closing
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct ClosingParams {
    /// Structuring element shape
    pub element: StructuringElement,
}


/// Closing algorithm
#[derive(Debug, Clone, Default)]
pub struct Closing;

impl Algorithm for Closing {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = ClosingParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Closing"
    }

    fn description(&self) -> &'static str {
        "Morphological closing (dilation then erosion) to fill small dark gaps"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        closing(&input, &params.element)
    }
}

/// Perform morphological closing on a raster
///
/// Closing = dilate then erode. Fills small dark gaps and holes
/// while preserving the overall shape of larger dark structures.
///
/// The resulting NaN border width is `2 * radius` because dilation
/// and erosion each add a `radius`-wide NaN border.
///
/// # Arguments
/// * `raster` - Input raster
/// * `element` - Structuring element defining the neighborhood shape
pub fn closing(raster: &Raster<f64>, element: &StructuringElement) -> Result<Raster<f64>> {
    let dilated = dilate(raster, element)?;
    erode(&dilated, element)
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
    fn test_closing_uniform() {
        let raster = make_raster(11, 11, 5.0);
        let result = closing(&raster, &StructuringElement::Square(1)).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "Uniform closing should preserve value, got {}",
            val
        );
    }

    #[test]
    fn test_closing_fills_dark_spot() {
        let mut raster = make_raster(11, 11, 100.0);
        // Single dark pixel (smaller than element)
        raster.set(5, 5, 1.0).unwrap();

        let result = closing(&raster, &StructuringElement::Square(1)).unwrap();
        // Closing should fill the single dark spot
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 100.0).abs() < 1e-10,
            "Closing should fill single dark pixel, got {}",
            val
        );
    }

    #[test]
    fn test_closing_preserves_large_dark_region() {
        let mut raster = make_raster(11, 11, 100.0);
        // Large dark block (3x3)
        for r in 4..7 {
            for c in 4..7 {
                raster.set(r, c, 1.0).unwrap();
            }
        }

        let result = closing(&raster, &StructuringElement::Square(1)).unwrap();
        // Center of large dark region should remain dark
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 1.0).abs() < 1e-10,
            "Closing should preserve large dark region center, got {}",
            val
        );
    }
}
