//! Basin morphometric parameters
//!
//! Computes geometric shape metrics for drainage basins (watersheds):
//! - Area (cells and m2)
//! - Perimeter (cells and m)
//! - Circularity ratio: 4*pi*A / P^2 (1.0 for a circle)
//! - Elongation ratio: (2/P) * sqrt(A/pi)
//! - Compactness coefficient: P / (2*sqrt(pi*A))
//!
//! These metrics characterize watershed shape, which influences
//! hydrological response (peak flow timing, flood susceptibility).

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};
use std::collections::HashMap;

/// Morphometric parameters for a single basin
#[derive(Debug, Clone)]
pub struct BasinMorphometry {
    /// Watershed identifier
    pub watershed_id: i32,
    /// Area in cells
    pub area_cells: usize,
    /// Area in square meters
    pub area_m2: f64,
    /// Perimeter in cell edges
    pub perimeter_cells: usize,
    /// Perimeter in meters
    pub perimeter_m: f64,
    /// Circularity ratio: 4*pi*A / P^2 (1.0 = perfect circle)
    pub circularity: f64,
    /// Elongation ratio: (2/P) * sqrt(A/pi)
    pub elongation: f64,
    /// Compactness coefficient: P / (2*sqrt(pi*A))
    pub compactness: f64,
}

/// Compute morphometric parameters for all basins in a watershed raster.
///
/// Uses the same perimeter counting method as `patch_metrics`: a cell edge
/// is counted as perimeter if it borders a different watershed or the raster
/// boundary.
///
/// # Arguments
/// * `watersheds` - Watershed ID raster (i32, values <= 0 are ignored)
/// * `cell_size` - Cell size in meters
///
/// # Returns
/// Vector of `BasinMorphometry` for each watershed found
pub fn basin_morphometry(
    watersheds: &Raster<i32>,
    cell_size: f64,
) -> Result<Vec<BasinMorphometry>> {
    if cell_size <= 0.0 {
        return Err(Error::Other("Cell size must be positive".into()));
    }

    let (rows, cols) = watersheds.shape();
    let cell_area = cell_size * cell_size;

    // Accumulate area and perimeter per watershed
    let mut area: HashMap<i32, usize> = HashMap::new();
    let mut perimeter: HashMap<i32, usize> = HashMap::new();

    for row in 0..rows {
        for col in 0..cols {
            let ws_id = unsafe { watersheds.get_unchecked(row, col) };
            if ws_id <= 0 {
                continue;
            }

            *area.entry(ws_id).or_insert(0) += 1;

            // Count perimeter edges: 4 cardinal neighbors
            for &(dr, dc) in &[(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                let nr = row as i32 + dr;
                let nc = col as i32 + dc;
                if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                    // Border edge
                    *perimeter.entry(ws_id).or_insert(0) += 1;
                } else {
                    let neighbor_id =
                        unsafe { watersheds.get_unchecked(nr as usize, nc as usize) };
                    if neighbor_id != ws_id {
                        *perimeter.entry(ws_id).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Compute morphometric indices
    let mut results: Vec<BasinMorphometry> = Vec::new();

    for (&ws_id, &a_cells) in &area {
        if a_cells == 0 {
            continue;
        }

        let p_edges = *perimeter.get(&ws_id).unwrap_or(&0);
        let a_m2 = a_cells as f64 * cell_area;
        let p_m = p_edges as f64 * cell_size;

        let pi = std::f64::consts::PI;

        // Circularity = 4*pi*A / P^2
        let circularity = if p_m > 0.0 {
            4.0 * pi * a_m2 / (p_m * p_m)
        } else {
            0.0
        };

        // Elongation = (2/P) * sqrt(A/pi)
        let elongation = if p_m > 0.0 {
            (2.0 / p_m) * (a_m2 / pi).sqrt()
        } else {
            0.0
        };

        // Compactness = P / (2*sqrt(pi*A))
        let compactness = if a_m2 > 0.0 {
            p_m / (2.0 * (pi * a_m2).sqrt())
        } else {
            0.0
        };

        results.push(BasinMorphometry {
            watershed_id: ws_id,
            area_cells: a_cells,
            area_m2: a_m2,
            perimeter_cells: p_edges,
            perimeter_m: p_m,
            circularity,
            elongation,
            compactness,
        });
    }

    // Sort by watershed ID for deterministic output
    results.sort_by_key(|b| b.watershed_id);

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_basin_morphometry_square() {
        // 4x4 square basin, cell_size=10m
        // Area = 16 * 100 = 1600 m2
        // Perimeter = 16 edges * 10 = 160 m
        let ws_data = vec![1_i32; 16];
        let arr = Array2::from_shape_vec((4, 4), ws_data).unwrap();
        let mut ws = Raster::from_array(arr);
        ws.set_transform(GeoTransform::new(0.0, 4.0, 10.0, -10.0));

        let result = basin_morphometry(&ws, 10.0).unwrap();
        assert_eq!(result.len(), 1);

        let b = &result[0];
        assert_eq!(b.watershed_id, 1);
        assert_eq!(b.area_cells, 16);
        assert!((b.area_m2 - 1600.0).abs() < 1e-6);
        assert_eq!(b.perimeter_cells, 16);
        assert!((b.perimeter_m - 160.0).abs() < 1e-6);

        // Circularity of a square: 4*pi*s^2 / (4s)^2 = pi/4 ≈ 0.785
        assert!(
            (b.circularity - std::f64::consts::PI / 4.0).abs() < 0.01,
            "Square circularity should be ~pi/4, got {}",
            b.circularity
        );
    }

    #[test]
    fn test_basin_morphometry_two_basins() {
        // Two basins: left half = 1, right half = 2
        let rows = 4;
        let cols = 4;
        let mut ws_data = vec![0_i32; 16];
        for row in 0..rows {
            for col in 0..cols {
                ws_data[row * cols + col] = if col < 2 { 1 } else { 2 };
            }
        }
        let arr = Array2::from_shape_vec((rows, cols), ws_data).unwrap();
        let mut ws = Raster::from_array(arr);
        ws.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));

        let result = basin_morphometry(&ws, 1.0).unwrap();
        assert_eq!(result.len(), 2);

        // Both basins should have same area (8 cells)
        assert_eq!(result[0].area_cells, 8);
        assert_eq!(result[1].area_cells, 8);
    }

    #[test]
    fn test_basin_morphometry_single_cell() {
        let ws_data = vec![0, 0, 0, 0, 1, 0, 0, 0, 0];
        let arr = Array2::from_shape_vec((3, 3), ws_data).unwrap();
        let mut ws = Raster::from_array(arr);
        ws.set_transform(GeoTransform::new(0.0, 3.0, 5.0, -5.0));

        let result = basin_morphometry(&ws, 5.0).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].area_cells, 1);
        assert_eq!(result[0].perimeter_cells, 4);
    }
}
