//! Morphological dilation (maximum filter)
//!
//! Replaces each pixel with the maximum value in its structuring element
//! neighborhood. Enlarges bright regions and shrinks dark regions.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

use super::element::StructuringElement;

/// Parameters for morphological dilation
#[derive(Debug, Clone)]
#[derive(Default)]
pub struct DilateParams {
    /// Structuring element shape
    pub element: StructuringElement,
}


/// Dilation algorithm
#[derive(Debug, Clone, Default)]
pub struct Dilate;

impl Algorithm for Dilate {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = DilateParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Dilate"
    }

    fn description(&self) -> &'static str {
        "Morphological dilation (maximum filter over structuring element)"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        dilate(&input, &params.element)
    }
}

/// Perform morphological dilation on a raster
///
/// Each output pixel is the maximum value within the structuring element
/// neighborhood. Edge cells (where the kernel extends beyond the raster)
/// and cells with any nodata neighbor are set to NaN.
///
/// # Arguments
/// * `raster` - Input raster
/// * `element` - Structuring element defining the neighborhood shape
pub fn dilate(raster: &Raster<f64>, element: &StructuringElement) -> Result<Raster<f64>> {
    element.validate()?;

    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();
    let offsets = element.offsets();
    let radius = element.radius() as isize;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let center = unsafe { raster.get_unchecked(row, col) };
                if is_nodata_val(center, nodata) {
                    continue;
                }

                // Skip edges where kernel extends beyond bounds
                let r = row as isize;
                let c = col as isize;
                if r - radius < 0
                    || r + radius >= rows as isize
                    || c - radius < 0
                    || c + radius >= cols as isize
                {
                    continue;
                }

                let mut max_val = f64::NEG_INFINITY;
                let mut has_nodata = false;

                for &(dr, dc) in &offsets {
                    let nr = (r + dr) as usize;
                    let nc = (c + dc) as usize;
                    let v = unsafe { raster.get_unchecked(nr, nc) };
                    if is_nodata_val(v, nodata) {
                        has_nodata = true;
                        break;
                    }
                    if v > max_val {
                        max_val = v;
                    }
                }

                if !has_nodata {
                    *row_data_col = max_val;
                }
            }

            row_data
        })
        .collect();

    build_output(raster, rows, cols, output_data)
}

fn is_nodata_val(value: f64, nodata: Option<f64>) -> bool {
    if value.is_nan() {
        return true;
    }
    match nodata {
        Some(nd) => (value - nd).abs() < f64::EPSILON,
        None => false,
    }
}

fn build_output(
    template: &Raster<f64>,
    rows: usize,
    cols: usize,
    data: Vec<f64>,
) -> Result<Raster<f64>> {
    let mut output = template.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), data).map_err(|e| Error::Other(e.to_string()))?;
    Ok(output)
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
    fn test_dilate_uniform() {
        let raster = make_raster(7, 7, 5.0);
        let result = dilate(&raster, &StructuringElement::Square(1)).unwrap();
        let val = result.get(3, 3).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "Uniform dilation should preserve value, got {}",
            val
        );
    }

    #[test]
    fn test_dilate_picks_maximum() {
        let mut raster = make_raster(7, 7, 5.0);
        // Place a high value near center
        raster.set(3, 4, 20.0).unwrap();

        let result = dilate(&raster, &StructuringElement::Square(1)).unwrap();
        // Cell (3,3) has neighbor (3,4)=20.0 → max should be 20.0
        let val = result.get(3, 3).unwrap();
        assert!(
            (val - 20.0).abs() < 1e-10,
            "Dilation should pick maximum neighbor, got {}",
            val
        );
    }

    #[test]
    fn test_dilate_edges_nan() {
        let raster = make_raster(7, 7, 5.0);
        let result = dilate(&raster, &StructuringElement::Square(1)).unwrap();
        assert!(result.get(0, 0).unwrap().is_nan());
        assert!(result.get(0, 3).unwrap().is_nan());
        assert!(result.get(3, 0).unwrap().is_nan());
    }

    #[test]
    fn test_dilate_nodata_propagation() {
        let mut raster = make_raster(7, 7, 5.0);
        raster.set_nodata(Some(-9999.0));
        raster.set(3, 3, -9999.0).unwrap();

        let result = dilate(&raster, &StructuringElement::Square(1)).unwrap();
        assert!(result.get(3, 3).unwrap().is_nan());
        assert!(result.get(3, 2).unwrap().is_nan());
        assert!(result.get(2, 3).unwrap().is_nan());
    }

    #[test]
    fn test_dilate_cross_element() {
        let mut raster = make_raster(7, 7, 5.0);
        // Place high value at diagonal from center
        raster.set(2, 2, 99.0).unwrap();

        let result = dilate(&raster, &StructuringElement::Cross(1)).unwrap();
        // Cross doesn't include diagonals, so (3,3) should not see (2,2)
        let val = result.get(3, 3).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "Cross should not include diagonal, got {}",
            val
        );
    }
}
