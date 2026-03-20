//! Raster mosaic: merge multiple rasters into one covering the union extent.

use crate::error::{Error, Result};
use crate::raster::{GeoTransform, Raster, RasterElement};

/// Configuration for mosaic operations.
#[derive(Debug, Clone)]
pub struct MosaicOptions {
    /// Relative tolerance for cell-size compatibility (default: 0.01 = 1%).
    pub cell_size_tolerance: f64,
    /// Whether to require all tiles to have the same CRS (default: true).
    pub require_same_crs: bool,
}

impl Default for MosaicOptions {
    fn default() -> Self {
        Self {
            cell_size_tolerance: 0.01,
            require_same_crs: true,
        }
    }
}

/// Merge multiple rasters into a single raster covering the union bounding box.
///
/// All input rasters must:
/// - Have compatible cell sizes (within `cell_size_tolerance`)
/// - Be in the same CRS (if `require_same_crs` is true)
/// - Be north-up (no rotation)
///
/// Overlap handling: NaN-aware last-write-wins. A source pixel only overwrites
/// the output if it is finite and not nodata. This ensures irregular tile shapes
/// (e.g., Sentinel-2 with masked borders) don't overwrite valid data from
/// adjacent tiles.
///
/// The output raster uses the CRS and nodata from the first tile.
pub fn mosaic<T: RasterElement>(
    tiles: &[&Raster<T>],
    options: Option<MosaicOptions>,
) -> Result<Raster<T>> {
    if tiles.is_empty() {
        return Err(Error::Other("mosaic requires at least 1 raster".into()));
    }

    let opts = options.unwrap_or_default();
    let first = tiles[0];
    let gt0 = first.transform();

    // Validate north-up
    for (i, tile) in tiles.iter().enumerate() {
        if !tile.transform().is_north_up() {
            return Err(Error::Other(format!(
                "Tile {} is not north-up (rotated rasters not supported)",
                i
            )));
        }
    }

    // Validate CRS compatibility
    if opts.require_same_crs {
        if let Some(ref crs0) = first.crs() {
            for (i, tile) in tiles.iter().enumerate().skip(1) {
                if let Some(ref crs_i) = tile.crs() {
                    if !crs0.is_equivalent(crs_i) {
                        return Err(Error::Other(format!(
                            "CRS mismatch: tile 0 has {}, tile {} has {}",
                            crs0, i, crs_i
                        )));
                    }
                }
            }
        }
    }

    // Validate cell size compatibility
    let pw0 = gt0.pixel_width;
    let ph0 = gt0.pixel_height.abs();
    let tol = opts.cell_size_tolerance;

    for (i, tile) in tiles.iter().enumerate().skip(1) {
        let gt = tile.transform();
        let pw_diff = ((gt.pixel_width - pw0) / pw0).abs();
        let ph_diff = ((gt.pixel_height.abs() - ph0) / ph0).abs();
        if pw_diff > tol || ph_diff > tol {
            return Err(Error::Other(format!(
                "Cell size mismatch: tile 0 has ({:.6}, {:.6}), tile {} has ({:.6}, {:.6}) (tolerance: {:.1}%)",
                pw0, ph0, i, gt.pixel_width, gt.pixel_height.abs(), tol * 100.0
            )));
        }
    }

    // Single tile shortcut
    if tiles.len() == 1 {
        return Ok(first.clone());
    }

    // Compute union bounding box
    let mut union_min_x = f64::INFINITY;
    let mut union_min_y = f64::INFINITY;
    let mut union_max_x = f64::NEG_INFINITY;
    let mut union_max_y = f64::NEG_INFINITY;

    for tile in tiles {
        let (min_x, min_y, max_x, max_y) = tile.bounds();
        union_min_x = union_min_x.min(min_x);
        union_min_y = union_min_y.min(min_y);
        union_max_x = union_max_x.max(max_x);
        union_max_y = union_max_y.max(max_y);
    }

    // Compute output dimensions
    let output_cols = ((union_max_x - union_min_x) / pw0).round() as usize;
    let output_rows = ((union_max_y - union_min_y) / ph0).round() as usize;

    if output_cols == 0 || output_rows == 0 {
        return Err(Error::Other("Mosaic output has zero dimensions".into()));
    }

    // Create output raster filled with nodata
    let nodata = first.nodata().unwrap_or_else(T::default_nodata);
    let mut output = Raster::filled(output_rows, output_cols, nodata);
    output.set_transform(GeoTransform::new(
        union_min_x,
        union_max_y, // north-up: origin_y is the top
        pw0,
        -ph0, // negative for north-up
    ));
    if let Some(crs) = first.crs() {
        output.set_crs(Some(crs.clone()));
    }
    output.set_nodata(Some(nodata));

    // Place each tile into the output
    let nodata_opt = Some(nodata);
    for tile in tiles {
        let tgt = tile.transform();
        let col_offset = ((tgt.origin_x - union_min_x) / pw0).round() as isize;
        let row_offset = ((union_max_y - tgt.origin_y) / ph0).round() as isize;

        let (tile_rows, tile_cols) = tile.shape();
        let tile_data = tile.data();

        for r in 0..tile_rows {
            let out_r = row_offset + r as isize;
            if out_r < 0 || out_r >= output_rows as isize {
                continue;
            }
            let out_r = out_r as usize;

            for c in 0..tile_cols {
                let out_c = col_offset + c as isize;
                if out_c < 0 || out_c >= output_cols as isize {
                    continue;
                }
                let out_c = out_c as usize;

                let value = tile_data[[r, c]];

                // Skip nodata values
                if value.is_nodata(nodata_opt) {
                    continue;
                }

                // Skip non-finite floats
                if T::is_float() {
                    if let Some(v) = value.to_f64() {
                        if !v.is_finite() {
                            continue;
                        }
                    }
                }

                // Write valid value to output
                unsafe {
                    output.set_unchecked(out_r, out_c, value);
                }
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_tile(rows: usize, cols: usize, value: f64, origin_x: f64, origin_y: f64) -> Raster<f64> {
        let arr = Array2::from_elem((rows, cols), value);
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(origin_x, origin_y, 10.0, -10.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_mosaic_two_adjacent_horizontal() {
        // Two 10x10 tiles side by side: tile1 at x=0, tile2 at x=100
        let t1 = make_tile(10, 10, 1.0, 0.0, 100.0);
        let t2 = make_tile(10, 10, 2.0, 100.0, 100.0);

        let result = mosaic(&[&t1, &t2], None).unwrap();
        assert_eq!(result.shape(), (10, 20));
        assert!((result.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
        assert!((result.get(0, 10).unwrap() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mosaic_two_adjacent_vertical() {
        // tile1 at top (y=100), tile2 below (y=0)
        let t1 = make_tile(10, 10, 1.0, 0.0, 100.0);
        let t2 = make_tile(10, 10, 2.0, 0.0, 0.0);

        let result = mosaic(&[&t1, &t2], None).unwrap();
        assert_eq!(result.shape(), (20, 10));
        assert!((result.get(0, 0).unwrap() - 1.0).abs() < 1e-10);  // top tile
        assert!((result.get(10, 0).unwrap() - 2.0).abs() < 1e-10); // bottom tile
    }

    #[test]
    fn test_mosaic_overlap_last_wins() {
        // Two 10x10 tiles overlapping by 5 cols
        let t1 = make_tile(10, 10, 1.0, 0.0, 100.0);
        let t2 = make_tile(10, 10, 2.0, 50.0, 100.0); // overlaps at cols 5-10

        let result = mosaic(&[&t1, &t2], None).unwrap();
        assert_eq!(result.shape(), (10, 15));
        assert!((result.get(0, 0).unwrap() - 1.0).abs() < 1e-10);  // t1 only
        assert!((result.get(0, 5).unwrap() - 2.0).abs() < 1e-10);  // overlap: t2 wins
        assert!((result.get(0, 10).unwrap() - 2.0).abs() < 1e-10); // t2 only
    }

    #[test]
    fn test_mosaic_nan_preserves_valid() {
        // t1 has value 1.0 everywhere
        let t1 = make_tile(10, 10, 1.0, 0.0, 100.0);
        // t2 overlaps fully but has NaN everywhere
        let mut t2 = make_tile(10, 10, f64::NAN, 0.0, 100.0);
        t2.set_nodata(Some(f64::NAN));

        let result = mosaic(&[&t1, &t2], None).unwrap();
        assert_eq!(result.shape(), (10, 10));
        // NaN from t2 should NOT overwrite t1's valid data
        assert!((result.get(0, 0).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mosaic_single_tile() {
        let t1 = make_tile(5, 5, 42.0, 0.0, 50.0);
        let result = mosaic(&[&t1], None).unwrap();
        assert_eq!(result.shape(), (5, 5));
        assert!((result.get(2, 2).unwrap() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_mosaic_empty_error() {
        let empty: Vec<&Raster<f64>> = vec![];
        assert!(mosaic(&empty, None).is_err());
    }

    #[test]
    fn test_mosaic_incompatible_cell_size() {
        let t1 = make_tile(10, 10, 1.0, 0.0, 100.0);
        let mut t2 = make_tile(10, 10, 2.0, 100.0, 100.0);
        t2.set_transform(GeoTransform::new(100.0, 100.0, 20.0, -20.0)); // different cell size

        assert!(mosaic(&[&t1, &t2], None).is_err());
    }

    #[test]
    fn test_mosaic_four_tiles_grid() {
        // 2x2 grid of 5x5 tiles
        let tl = make_tile(5, 5, 1.0, 0.0, 100.0);   // top-left
        let tr = make_tile(5, 5, 2.0, 50.0, 100.0);   // top-right
        let bl = make_tile(5, 5, 3.0, 0.0, 50.0);     // bottom-left
        let br = make_tile(5, 5, 4.0, 50.0, 50.0);    // bottom-right

        let result = mosaic(&[&tl, &tr, &bl, &br], None).unwrap();
        assert_eq!(result.shape(), (10, 10));
        assert!((result.get(0, 0).unwrap() - 1.0).abs() < 1e-10);  // top-left
        assert!((result.get(0, 5).unwrap() - 2.0).abs() < 1e-10);  // top-right
        assert!((result.get(5, 0).unwrap() - 3.0).abs() < 1e-10);  // bottom-left
        assert!((result.get(5, 5).unwrap() - 4.0).abs() < 1e-10);  // bottom-right
    }

    #[test]
    fn test_mosaic_preserves_metadata() {
        use crate::crs::CRS;
        let mut t1 = make_tile(5, 5, 1.0, 100.0, 200.0);
        t1.set_crs(Some(CRS::from_epsg(32719)));
        let t2 = make_tile(5, 5, 2.0, 150.0, 200.0);

        let result = mosaic(&[&t1, &t2], None).unwrap();
        assert!(result.crs().is_some());
        assert_eq!(result.crs().unwrap().epsg(), Some(32719));
        assert!(result.nodata().is_some());

        let gt = result.transform();
        assert!((gt.origin_x - 100.0).abs() < 1e-10);
        assert!((gt.origin_y - 200.0).abs() < 1e-10);
    }
}
