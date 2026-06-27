//! Melton ruggedness ratio
//!
//! The Melton ruggedness ratio (Melton 1965) characterizes the relief
//! energy of a drainage basin:
//!
//! ```text
//! MRR = (H_max - H_min) / sqrt(A)
//! ```
//!
//! where `H_max - H_min` is the basin relief and `A` is the basin area.
//! The ratio is dimensionless and independent of the length unit chosen
//! (metres vs kilometres), since relief has units of length and `sqrt(A)`
//! also has units of length.
//!
//! MRR is a first-order screening metric for debris-flow and lahar
//! susceptibility. Following the geomorphic literature (e.g. Wilford et al.
//! 2004; Welsh & Davies 2011), basins are commonly classified as:
//!
//! - `MRR < 0.3`   — flood-dominated (normal fluvial)
//! - `0.3 ≤ MRR < 0.5` — debris-flood transitional
//! - `MRR ≥ 0.5`   — debris-flow prone
//!
//! These thresholds are heuristic and region-dependent; the routine reports
//! the raw ratio and leaves classification to the caller.
//!
//! # References
//! - Melton, M.A. (1965). The geomorphic and paleoclimatic significance of
//!   alluvial deposits in southern Arizona. *Journal of Geology* 73(1), 1–38.
//! - Wilford, D.J., Sakals, M.E., Innes, J.L., Sidle, R.C., Bergerud, W.A.
//!   (2004). Recognition of debris flow, debris flood and flood hazard
//!   through watershed morphometrics. *Landslides* 1, 61–66.

use std::collections::HashMap;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Melton ruggedness ratio for a single basin.
#[derive(Debug, Clone)]
pub struct MeltonRuggedness {
    /// Watershed identifier.
    pub watershed_id: i32,
    /// Number of valid cells contributing to the basin.
    pub area_cells: usize,
    /// Basin area in square metres.
    pub area_m2: f64,
    /// Minimum elevation in the basin (map units, typically metres).
    pub min_elevation: f64,
    /// Maximum elevation in the basin.
    pub max_elevation: f64,
    /// Basin relief: `max_elevation - min_elevation`.
    pub relief: f64,
    /// Melton ruggedness ratio: `relief / sqrt(area_m2)` (dimensionless).
    pub melton_ratio: f64,
}

/// Compute the Melton ruggedness ratio for every basin in a watershed raster.
///
/// Cells are accumulated per watershed in a single pass: the basin area is the
/// valid cell count times the cell area, and the relief is the elevation range
/// over the basin's cells. Cells whose watershed id is `<= 0` or whose
/// elevation is `NaN` are ignored.
///
/// # Arguments
/// * `watersheds` - Watershed id raster (`i32`; values `<= 0` are ignored).
/// * `dem` - Elevation raster aligned with `watersheds`.
/// * `cell_size` - Cell size in metres (must be positive).
///
/// # Returns
/// A vector of [`MeltonRuggedness`], one per basin, sorted by watershed id.
///
/// # Errors
/// Returns an error if `cell_size <= 0` or the rasters have mismatched shapes.
pub fn melton_ruggedness(
    watersheds: &Raster<i32>,
    dem: &Raster<f64>,
    cell_size: f64,
) -> Result<Vec<MeltonRuggedness>> {
    if cell_size <= 0.0 {
        return Err(Error::Other("Cell size must be positive".into()));
    }

    let (rows, cols) = watersheds.shape();
    let (rows_d, cols_d) = dem.shape();
    if rows != rows_d || cols != cols_d {
        return Err(Error::SizeMismatch {
            er: rows,
            ec: cols,
            ar: rows_d,
            ac: cols_d,
        });
    }

    let cell_area = cell_size * cell_size;

    // Per-basin accumulators: (count, min_elev, max_elev).
    let mut acc: HashMap<i32, (usize, f64, f64)> = HashMap::new();

    for row in 0..rows {
        for col in 0..cols {
            let ws_id = unsafe { watersheds.get_unchecked(row, col) };
            if ws_id <= 0 {
                continue;
            }
            let z = unsafe { dem.get_unchecked(row, col) };
            if z.is_nan() {
                continue;
            }

            let entry = acc
                .entry(ws_id)
                .or_insert((0, f64::INFINITY, f64::NEG_INFINITY));
            entry.0 += 1;
            if z < entry.1 {
                entry.1 = z;
            }
            if z > entry.2 {
                entry.2 = z;
            }
        }
    }

    let mut results: Vec<MeltonRuggedness> = Vec::with_capacity(acc.len());

    for (ws_id, (count, min_elev, max_elev)) in acc {
        if count == 0 {
            continue;
        }
        let area_m2 = count as f64 * cell_area;
        let relief = max_elev - min_elev;
        // area_m2 > 0 because count > 0 and cell_area > 0.
        let melton_ratio = relief / area_m2.sqrt();

        results.push(MeltonRuggedness {
            watershed_id: ws_id,
            area_cells: count,
            area_m2,
            min_elevation: min_elev,
            max_elevation: max_elev,
            relief,
            melton_ratio,
        });
    }

    results.sort_by_key(|m| m.watershed_id);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn raster_i32(data: Vec<i32>, rows: usize, cols: usize, cs: f64) -> Raster<i32> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, rows as f64, cs, -cs));
        r
    }

    fn raster_f64(data: Vec<f64>, rows: usize, cols: usize, cs: f64) -> Raster<f64> {
        let arr = Array2::from_shape_vec((rows, cols), data).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, rows as f64, cs, -cs));
        r
    }

    #[test]
    fn test_melton_single_basin_known_value() {
        // 2x2 basin, cell_size = 10 m -> area = 4 * 100 = 400 m2, sqrt = 20 m.
        // Elevations 100..130 -> relief = 30 m. MRR = 30 / 20 = 1.5.
        let ws = raster_i32(vec![1, 1, 1, 1], 2, 2, 10.0);
        let dem = raster_f64(vec![100.0, 110.0, 120.0, 130.0], 2, 2, 10.0);

        let out = melton_ruggedness(&ws, &dem, 10.0).unwrap();
        assert_eq!(out.len(), 1);
        let b = &out[0];
        assert_eq!(b.watershed_id, 1);
        assert_eq!(b.area_cells, 4);
        assert!((b.area_m2 - 400.0).abs() < 1e-9);
        assert!((b.relief - 30.0).abs() < 1e-9);
        assert!((b.melton_ratio - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_melton_unit_independence() {
        // Same geometry/relief expressed at a different cell size keeps MRR
        // identical: relief / sqrt(area) is scale-of-unit invariant only when
        // the cell size scales the area consistently. Here we verify the ratio
        // is independent of how many cells discretise a fixed physical basin.
        // Basin A: 2x2 cells @ 10 m (area 400 m2), relief 20 m -> 20/20 = 1.0.
        let ws_a = raster_i32(vec![1, 1, 1, 1], 2, 2, 10.0);
        let dem_a = raster_f64(vec![0.0, 20.0, 0.0, 20.0], 2, 2, 10.0);
        let a = &melton_ruggedness(&ws_a, &dem_a, 10.0).unwrap()[0];
        assert!((a.melton_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_melton_two_basins_and_nodata() {
        // Left column basin 1, right column basin 2, top-right is nodata (0).
        let ws = raster_i32(vec![1, 0, 1, 2], 2, 2, 5.0);
        let dem = raster_f64(vec![10.0, 999.0, 40.0, 50.0], 2, 2, 5.0);
        let out = melton_ruggedness(&ws, &dem, 5.0).unwrap();
        assert_eq!(out.len(), 2);

        // Basin 1: cells (0,0)=10, (1,0)=40 -> relief 30, area 2*25=50.
        let b1 = &out[0];
        assert_eq!(b1.watershed_id, 1);
        assert_eq!(b1.area_cells, 2);
        assert!((b1.relief - 30.0).abs() < 1e-9);
        assert!((b1.melton_ratio - 30.0 / 50.0_f64.sqrt()).abs() < 1e-9);

        // Basin 2: single cell (1,1)=50 -> relief 0, area 25.
        let b2 = &out[1];
        assert_eq!(b2.watershed_id, 2);
        assert_eq!(b2.area_cells, 1);
        assert!((b2.relief).abs() < 1e-9);
        assert!((b2.melton_ratio).abs() < 1e-9);
    }

    #[test]
    fn test_melton_nan_elevation_skipped() {
        let ws = raster_i32(vec![1, 1, 1, 1], 2, 2, 10.0);
        let dem = raster_f64(vec![100.0, f64::NAN, 120.0, 130.0], 2, 2, 10.0);
        let out = melton_ruggedness(&ws, &dem, 10.0).unwrap();
        // Only 3 valid cells -> area 300 m2, relief 30, MRR = 30/sqrt(300).
        let b = &out[0];
        assert_eq!(b.area_cells, 3);
        assert!((b.melton_ratio - 30.0 / 300.0_f64.sqrt()).abs() < 1e-9);
    }

    #[test]
    fn test_melton_rejects_bad_cell_size() {
        let ws = raster_i32(vec![1], 1, 1, 1.0);
        let dem = raster_f64(vec![1.0], 1, 1, 1.0);
        assert!(melton_ruggedness(&ws, &dem, 0.0).is_err());
    }

    #[test]
    fn test_melton_size_mismatch() {
        let ws = raster_i32(vec![1, 1, 1, 1], 2, 2, 10.0);
        let dem = raster_f64(vec![1.0; 9], 3, 3, 10.0);
        assert!(melton_ruggedness(&ws, &dem, 10.0).is_err());
    }
}
