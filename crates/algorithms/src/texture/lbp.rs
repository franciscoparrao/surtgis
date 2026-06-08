//! Local Binary Patterns (LBP).
//!
//! Reference: Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002).
//! "Multiresolution gray-scale and rotation invariant texture
//! classification with local binary patterns."
//! IEEE Transactions on Pattern Analysis and Machine Intelligence
//! 24(7), 971-987.
//!
//! Uses the fixed P=8, R=1 neighbourhood (the 3×3 ring around each
//! pixel). The `Standard` variant emits the raw 8-bit code (0..=255)
//! with neighbours ordered clockwise starting from the top-left
//! pixel — bit 7 is the top-left neighbour, bit 0 is the left
//! neighbour. The `RotationInvariantUniform` variant emits the
//! Ojala riu2 value: the popcount of the code for "uniform"
//! patterns (≤2 circular bit transitions), or P+1 = 9 for
//! non-uniform patterns. riu2 collapses all rotations of the same
//! pattern to a single value and is the recommended feature for
//! rotation-invariant classification.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// LBP variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LbpVariant {
    /// Raw 8-bit code (0..=255). Bit i is set when neighbour i ≥ centre.
    Standard,
    /// Rotation-invariant uniform pattern (riu2, Ojala 2002).
    /// Uniform code (≤2 transitions): popcount (0..=8). Non-uniform: 9.
    RotationInvariantUniform,
}

#[derive(Debug, Clone)]
pub struct LbpParams {
    pub variant: LbpVariant,
}

impl Default for LbpParams {
    fn default() -> Self {
        Self {
            variant: LbpVariant::Standard,
        }
    }
}

/// Compute Local Binary Pattern for each interior pixel of `raster`.
///
/// Border pixels (1-cell margin) and pixels whose centre or any of
/// the 8 immediate neighbours is NaN are written as NaN.
pub fn lbp(raster: &Raster<f64>, params: LbpParams) -> Result<Raster<f64>> {
    let (rows, cols) = raster.shape();
    if rows < 3 || cols < 3 {
        return Err(Error::Algorithm("LBP requires at least 3x3 raster".into()));
    }

    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            if row == 0 || row == rows - 1 {
                return row_data;
            }
            for (col, out) in row_data.iter_mut().enumerate().skip(1).take(cols - 2) {
                let centre = unsafe { raster.get_unchecked(row, col) };
                if centre.is_nan() {
                    continue;
                }
                let neighs = [
                    unsafe { raster.get_unchecked(row - 1, col - 1) }, // bit 7
                    unsafe { raster.get_unchecked(row - 1, col) },     // bit 6
                    unsafe { raster.get_unchecked(row - 1, col + 1) }, // bit 5
                    unsafe { raster.get_unchecked(row, col + 1) },     // bit 4
                    unsafe { raster.get_unchecked(row + 1, col + 1) }, // bit 3
                    unsafe { raster.get_unchecked(row + 1, col) },     // bit 2
                    unsafe { raster.get_unchecked(row + 1, col - 1) }, // bit 1
                    unsafe { raster.get_unchecked(row, col - 1) },     // bit 0
                ];
                if neighs.iter().any(|v| v.is_nan()) {
                    continue;
                }
                let mut code: u8 = 0;
                for (i, &v) in neighs.iter().enumerate() {
                    if v >= centre {
                        code |= 1 << (7 - i);
                    }
                }
                *out = match params.variant {
                    LbpVariant::Standard => code as f64,
                    LbpVariant::RotationInvariantUniform => riu2(code) as f64,
                };
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

/// Number of 0↔1 circular bit transitions in an 8-bit code.
#[inline]
fn transitions(code: u8) -> u32 {
    let rotated = code.rotate_right(1);
    (code ^ rotated).count_ones()
}

/// Rotation-invariant uniform value (riu2).
#[inline]
fn riu2(code: u8) -> u8 {
    if transitions(code) <= 2 {
        code.count_ones() as u8
    } else {
        9
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn raster_from_grid(grid: &[&[f64]]) -> Raster<f64> {
        let rows = grid.len();
        let cols = grid[0].len();
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for (row, row_vals) in grid.iter().enumerate() {
            for (col, &val) in row_vals.iter().enumerate() {
                r.set(row, col, val).unwrap();
            }
        }
        r
    }

    #[test]
    fn standard_all_neighbours_brighter() {
        // Centre 5, every neighbour 9 -> code = 0b11111111 = 255.
        let r = raster_from_grid(&[&[9.0, 9.0, 9.0], &[9.0, 5.0, 9.0], &[9.0, 9.0, 9.0]]);
        let result = lbp(&r, LbpParams::default()).unwrap();
        assert_eq!(result.get(1, 1).unwrap(), 255.0);
    }

    #[test]
    fn standard_all_neighbours_darker() {
        // Centre 9, every neighbour 1 -> code = 0.
        let r = raster_from_grid(&[&[1.0, 1.0, 1.0], &[1.0, 9.0, 1.0], &[1.0, 1.0, 1.0]]);
        let result = lbp(&r, LbpParams::default()).unwrap();
        assert_eq!(result.get(1, 1).unwrap(), 0.0);
    }

    #[test]
    fn standard_known_pattern() {
        // Centre 5. Neighbours clockwise from TL = [9,9,9,1,1,1,9,9].
        // bits (7..0) = 1,1,1,0,0,0,1,1 -> 0b11100011 = 227.
        let r = raster_from_grid(&[&[9.0, 9.0, 9.0], &[9.0, 5.0, 1.0], &[9.0, 1.0, 1.0]]);
        let result = lbp(&r, LbpParams::default()).unwrap();
        assert_eq!(result.get(1, 1).unwrap(), 227.0);
    }

    #[test]
    fn riu2_uniform_one_transition() {
        // Half-bright/half-dark split (top row + right cell bright,
        // rest dark) -> 1 transition pair -> uniform, popcount = 4.
        let r = raster_from_grid(&[&[9.0, 9.0, 9.0], &[1.0, 5.0, 9.0], &[1.0, 1.0, 1.0]]);
        let result = lbp(
            &r,
            LbpParams {
                variant: LbpVariant::RotationInvariantUniform,
            },
        )
        .unwrap();
        assert_eq!(result.get(1, 1).unwrap(), 4.0);
    }

    #[test]
    fn riu2_non_uniform_checkerboard() {
        // Alternating bright/dark neighbours -> code = 0b10101010,
        // 8 transitions -> non-uniform -> 9.
        let r = raster_from_grid(&[&[9.0, 1.0, 9.0], &[1.0, 5.0, 1.0], &[9.0, 1.0, 9.0]]);
        let result = lbp(
            &r,
            LbpParams {
                variant: LbpVariant::RotationInvariantUniform,
            },
        )
        .unwrap();
        assert_eq!(result.get(1, 1).unwrap(), 9.0);
    }

    #[test]
    fn riu2_rotation_invariance() {
        // Two rotations of the same "3 bright in a row" pattern
        // should give the same riu2 value.
        // Pattern A: top-left/top/top-right bright -> 0b11100000.
        let a = raster_from_grid(&[&[9.0, 9.0, 9.0], &[1.0, 5.0, 1.0], &[1.0, 1.0, 1.0]]);
        // Pattern B: top-right/right/bottom-right bright -> 0b00111000.
        let b = raster_from_grid(&[&[1.0, 1.0, 9.0], &[1.0, 5.0, 9.0], &[1.0, 1.0, 9.0]]);
        let params = LbpParams {
            variant: LbpVariant::RotationInvariantUniform,
        };
        let ra = lbp(&a, params.clone()).unwrap();
        let rb = lbp(&b, params).unwrap();
        let va = ra.get(1, 1).unwrap();
        let vb = rb.get(1, 1).unwrap();
        assert_eq!(va, 3.0);
        assert_eq!(va, vb, "rotations of the same pattern must give equal riu2");
    }

    #[test]
    fn border_pixels_are_nan() {
        let r = raster_from_grid(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        let result = lbp(&r, LbpParams::default()).unwrap();
        assert!(result.get(0, 0).unwrap().is_nan());
        assert!(result.get(0, 2).unwrap().is_nan());
        assert!(result.get(2, 1).unwrap().is_nan());
    }

    #[test]
    fn nan_neighbour_propagates() {
        let r = raster_from_grid(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, f64::NAN],
            &[7.0, 8.0, 9.0],
        ]);
        let result = lbp(&r, LbpParams::default()).unwrap();
        assert!(result.get(1, 1).unwrap().is_nan());
    }

    #[test]
    fn raster_too_small_errors() {
        let r = raster_from_grid(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert!(lbp(&r, LbpParams::default()).is_err());
    }

    #[test]
    fn transitions_canonical_examples() {
        // 0b00000000 -> 0 transitions
        assert_eq!(transitions(0b00000000), 0);
        // 0b11111111 -> 0 transitions
        assert_eq!(transitions(0b11111111), 0);
        // 0b11110000 -> 2 transitions (uniform)
        assert_eq!(transitions(0b11110000), 2);
        // 0b10101010 -> 8 transitions (max)
        assert_eq!(transitions(0b10101010), 8);
    }
}
