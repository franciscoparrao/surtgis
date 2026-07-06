//! Morphological erosion (minimum filter)
//!
//! Replaces each pixel with the minimum value in its structuring element
//! neighborhood. Shrinks bright regions and enlarges dark regions.

use crate::maybe_rayon::*;
use crate::statistics::focal_fast::sliding_extreme_1d;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

use super::element::StructuringElement;

/// Parameters for morphological erosion
#[derive(Debug, Clone, Default)]
pub struct ErodeParams {
    /// Structuring element shape
    pub element: StructuringElement,
}

/// Erosion algorithm
#[derive(Debug, Clone, Default)]
pub struct Erode;

impl Algorithm for Erode {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = ErodeParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "Erode"
    }

    fn description(&self) -> &'static str {
        "Morphological erosion (minimum filter over structuring element)"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        erode(&input, &params.element)
    }
}

/// Perform morphological erosion on a raster
///
/// Each output pixel is the minimum value within the structuring element
/// neighborhood. Edge cells (where the kernel extends beyond the raster)
/// and cells with any nodata neighbor are set to NaN.
///
/// # Arguments
/// * `raster` - Input raster
/// * `element` - Structuring element defining the neighborhood shape
pub fn erode(raster: &Raster<f64>, element: &StructuringElement) -> Result<Raster<f64>> {
    element.validate()?;

    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();

    // Fast path: a square structuring element's minimum filter is
    // separable (van Herk 1992 / Gil-Werman 1993), giving O(1) amortized
    // per cell instead of the O(radius^2) brute-force scan over
    // `element.offsets()` below. Cross/Disk/Custom elements aren't
    // separable in general, so they keep the brute-force path.
    let output_data: Vec<f64> = if let StructuringElement::Square(r) = element {
        erode_dilate_square_fast(raster, *r, true)
    } else {
        let offsets = element.offsets();
        let radius = element.radius() as isize;

        (0..rows)
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

                    let mut min_val = f64::INFINITY;
                    let mut has_nodata = false;

                    for &(dr, dc) in &offsets {
                        let nr = (r + dr) as usize;
                        let nc = (c + dc) as usize;
                        let v = unsafe { raster.get_unchecked(nr, nc) };
                        if is_nodata_val(v, nodata) {
                            has_nodata = true;
                            break;
                        }
                        if v < min_val {
                            min_val = v;
                        }
                    }

                    if !has_nodata {
                        *row_data_col = min_val;
                    }
                }

                row_data
            })
            .collect()
    };

    build_output(raster, rows, cols, output_data)
}

/// Fast path for `erode`/`dilate` with a `Square(radius)` structuring
/// element: `take_min = true` computes erosion (minimum filter),
/// `take_min = false` computes dilation (maximum filter).
///
/// Reuses [`sliding_extreme_1d`] (the same van Herk/Gil-Werman primitive
/// `focal_fast::hgw_square_2d` is built on) but layers on erode/dilate's
/// own nodata rule, which differs from `focal_fast`'s: there, nodata
/// cells are simply excluded from the reduction; here, *any* nodata cell
/// anywhere in the window must invalidate the whole output cell (matching
/// the brute-force `has_nodata` short-circuit above bit-for-bit, since
/// min/max only ever return one of the window's actual values verbatim —
/// no arithmetic rounding is introduced by reordering the reduction).
///
/// This is done with two separable passes over the whole raster:
/// - one over the raw values (nodata cells replaced by a sentinel that
///   can never win the reduction) to get the numeric extreme;
/// - one over a 0.0/1.0 "is this cell nodata" indicator, run as a
///   max-filter (i.e. logical OR over the window) to detect whether the
///   window contains any nodata cell at all.
fn erode_dilate_square_fast(raster: &Raster<f64>, radius: usize, take_min: bool) -> Vec<f64> {
    let (rows, cols) = raster.shape();
    let nodata = raster.nodata();
    let pad = if take_min {
        f64::INFINITY
    } else {
        f64::NEG_INFINITY
    };

    let mut values = Array2::<f64>::zeros((rows, cols));
    let mut indicator = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let v = unsafe { raster.get_unchecked(r, c) };
            if is_nodata_val(v, nodata) {
                values[[r, c]] = pad;
                indicator[[r, c]] = 1.0;
            } else {
                values[[r, c]] = v;
            }
        }
    }

    let extreme = hgw_square_pass(&values, radius, take_min);
    let any_nodata = hgw_square_pass(&indicator, radius, false); // OR over the window

    let radius_i = radius as isize;
    let mut out = vec![f64::NAN; rows * cols];
    for r in 0..rows {
        let ri = r as isize;
        if ri - radius_i < 0 || ri + radius_i >= rows as isize {
            continue;
        }
        for c in 0..cols {
            let ci = c as isize;
            if ci - radius_i < 0 || ci + radius_i >= cols as isize {
                continue;
            }
            // Mirrors the brute-force path's separate up-front center
            // check (redundant with `any_nodata` for a Square element,
            // whose window always includes the center offset, but kept
            // so the control flow matches 1:1).
            let center = unsafe { raster.get_unchecked(r, c) };
            if is_nodata_val(center, nodata) {
                continue;
            }
            if any_nodata[[r, c]] > 0.5 {
                continue;
            }
            out[r * cols + c] = extreme[[r, c]];
        }
    }
    out
}

/// Separable van Herk (1992) / Gil-Werman (1993) sliding min/max over a
/// square window of the given `radius`, applied to a plain `Array2<f64>`
/// (rows pass, transpose, columns pass, transpose back) — the same
/// structure as `focal_fast::hgw_square_2d`, but operating on data the
/// caller has already prepared (nodata substituted by a sentinel that
/// never wins the reduction), so it can be reused for both the raw-value
/// pass and the nodata-indicator pass in [`erode_dilate_square_fast`].
fn hgw_square_pass(data: &Array2<f64>, radius: usize, take_min: bool) -> Array2<f64> {
    let (rows, cols) = data.dim();

    // Horizontal pass.
    let row_pass = par_map_rows(rows, cols, |row, out_row| {
        let slice = data.row(row);
        let slice = slice.as_slice().expect("raster row must be contiguous");
        let filtered = sliding_extreme_1d(slice, radius, take_min);
        out_row.copy_from_slice(&filtered);
    });

    // Transpose so the vertical pass can reuse the same 1D routine.
    let mut transposed = Array2::<f64>::zeros((cols, rows));
    for r in 0..rows {
        for c in 0..cols {
            transposed[[c, r]] = row_pass[[r, c]];
        }
    }

    let col_pass = par_map_rows(cols, rows, |c, out_col| {
        let slice = transposed.row(c);
        let slice = slice
            .as_slice()
            .expect("transposed row must be contiguous");
        let filtered = sliding_extreme_1d(slice, radius, take_min);
        out_col.copy_from_slice(&filtered);
    });

    // Transpose back.
    let mut out = Array2::<f64>::zeros((rows, cols));
    for c in 0..cols {
        for r in 0..rows {
            out[[r, c]] = col_pass[[c, r]];
        }
    }
    out
}

fn is_nodata_val(value: f64, nodata: Option<f64>) -> bool {
    if value.is_nan() {
        return true;
    }
    // Exact-bit comparison, matching `RasterElement::is_nodata` in
    // crates/core/src/raster/element.rs: the nodata sentinel is written
    // verbatim on I/O, so equality (not tolerance) is correct — a
    // tolerance-based match corrupts small valid values near the
    // sentinel (e.g. NDVI ~ 0 with nodata = 0.0).
    match nodata {
        Some(nd) => value == nd,
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
    fn test_erode_uniform() {
        let raster = make_raster(7, 7, 5.0);
        let result = erode(&raster, &StructuringElement::Square(1)).unwrap();
        // Interior cell should remain 5.0
        let val = result.get(3, 3).unwrap();
        assert!(
            (val - 5.0).abs() < 1e-10,
            "Uniform erosion should preserve value, got {}",
            val
        );
    }

    #[test]
    fn test_erode_picks_minimum() {
        let mut raster = make_raster(7, 7, 10.0);
        // Place a low value near center
        raster.set(3, 4, 2.0).unwrap();

        let result = erode(&raster, &StructuringElement::Square(1)).unwrap();
        // Cell (3,3) has neighbor (3,4)=2.0 → min should be 2.0
        let val = result.get(3, 3).unwrap();
        assert!(
            (val - 2.0).abs() < 1e-10,
            "Erosion should pick minimum neighbor, got {}",
            val
        );
    }

    #[test]
    fn test_erode_edges_nan() {
        let raster = make_raster(7, 7, 5.0);
        let result = erode(&raster, &StructuringElement::Square(1)).unwrap();
        // Edge cells should be NaN
        assert!(result.get(0, 0).unwrap().is_nan());
        assert!(result.get(0, 3).unwrap().is_nan());
        assert!(result.get(3, 0).unwrap().is_nan());
    }

    #[test]
    fn test_erode_nodata_propagation() {
        let mut raster = make_raster(7, 7, 5.0);
        raster.set_nodata(Some(-9999.0));
        raster.set(3, 3, -9999.0).unwrap();

        let result = erode(&raster, &StructuringElement::Square(1)).unwrap();
        // Center is nodata → output is NaN
        assert!(result.get(3, 3).unwrap().is_nan());
        // Neighbor of nodata → output is NaN
        assert!(result.get(3, 2).unwrap().is_nan());
        assert!(result.get(2, 3).unwrap().is_nan());
    }

    #[test]
    fn test_erode_single_pixel() {
        let raster = make_raster(1, 1, 5.0);
        let result = erode(&raster, &StructuringElement::Square(1)).unwrap();
        // Single pixel cannot have a full neighborhood → NaN
        assert!(result.get(0, 0).unwrap().is_nan());
    }

    #[test]
    fn test_erode_larger_element() {
        let raster = make_raster(11, 11, 10.0);
        let result = erode(&raster, &StructuringElement::Square(2)).unwrap();
        // Radius 2: border of 2 cells should be NaN, interior should be 10.0
        assert!(result.get(1, 1).unwrap().is_nan());
        let val = result.get(5, 5).unwrap();
        assert!(
            (val - 10.0).abs() < 1e-10,
            "Interior should be 10.0, got {}",
            val
        );
    }

    #[test]
    fn test_erode_cross_element() {
        let mut raster = make_raster(7, 7, 10.0);
        // Place low value at diagonal from center
        raster.set(2, 2, 1.0).unwrap();

        let result = erode(&raster, &StructuringElement::Cross(1)).unwrap();
        // Cross doesn't include diagonals, so (3,3) should not see (2,2)
        let val = result.get(3, 3).unwrap();
        assert!(
            (val - 10.0).abs() < 1e-10,
            "Cross should not include diagonal, got {}",
            val
        );
    }

    /// Independent brute-force reference (deliberately not reusing any of
    /// `erode`'s own code) checked against the `Square` fast path on a
    /// raster with scattered nodata, to confirm the "any nodata neighbor
    /// in the window -> NaN" rule survived the switch to the separable
    /// van Herk/Gil-Werman filter.
    #[test]
    fn test_erode_square_fast_path_matches_bruteforce_with_scattered_nodata() {
        let rows = 15;
        let cols = 17;
        let nodata_val = -9999.0;
        let mut raster = Raster::<f64>::new(rows, cols);
        raster.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        raster.set_nodata(Some(nodata_val));

        for r in 0..rows {
            for c in 0..cols {
                let v = ((r * 13 + c * 7) % 23) as f64 * 1.3 - 5.0;
                raster.set(r, c, v).unwrap();
            }
        }
        // Scatter nodata cells, including one right at a corner.
        for &(r, c) in &[(2, 3), (5, 5), (5, 6), (9, 12), (12, 2), (0, 0), (14, 16)] {
            raster.set(r, c, nodata_val).unwrap();
        }

        let radius = 3usize;
        let result = erode(&raster, &StructuringElement::Square(radius)).unwrap();

        let ri = radius as isize;
        for r in 0..rows {
            for c in 0..cols {
                let ir = r as isize;
                let ic = c as isize;
                let expected = if ir - ri < 0
                    || ir + ri >= rows as isize
                    || ic - ri < 0
                    || ic + ri >= cols as isize
                {
                    f64::NAN
                } else {
                    let mut has_nodata = false;
                    let mut min_val = f64::INFINITY;
                    for dr in -ri..=ri {
                        for dc in -ri..=ri {
                            let nr = (ir + dr) as usize;
                            let nc = (ic + dc) as usize;
                            let v = raster.get(nr, nc).unwrap();
                            if v.is_nan() || v == nodata_val {
                                has_nodata = true;
                            } else if v < min_val {
                                min_val = v;
                            }
                        }
                    }
                    if has_nodata { f64::NAN } else { min_val }
                };

                let got = result.get(r, c).unwrap();
                match (expected.is_nan(), got.is_nan()) {
                    (true, true) => {}
                    (false, false) => assert!(
                        (expected - got).abs() < 1e-12,
                        "mismatch at ({r},{c}): expected={expected}, got={got}"
                    ),
                    _ => panic!(
                        "NaN mismatch at ({r},{c}): expected={expected}, got={got}"
                    ),
                }
            }
        }
    }

    #[test]
    fn test_erode_element_larger_than_raster() {
        let raster = make_raster(3, 3, 5.0);
        let result = erode(&raster, &StructuringElement::Square(2)).unwrap();
        // Radius 2 on a 3x3 raster: all cells are edges → all NaN
        for row in 0..3 {
            for col in 0..3 {
                assert!(
                    result.get(row, col).unwrap().is_nan(),
                    "All should be NaN at ({}, {})",
                    row,
                    col
                );
            }
        }
    }
}
