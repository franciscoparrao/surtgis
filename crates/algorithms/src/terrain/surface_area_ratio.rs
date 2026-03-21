//! Surface Area Ratio (Jenness 2004)
//!
//! The Surface Area Ratio (SAR) quantifies terrain roughness by comparing
//! the 3D surface area to the 2D planimetric area within a moving window.
//!
//! For a 3x3 window (radius=1), 8 triangles are formed by the center cell
//! and each pair of adjacent neighbors (clockwise). The 3D area of each
//! triangle is computed via the cross product, and the sum is divided by
//! the 2D planimetric area.
//!
//! SAR = 1.0 for a perfectly flat surface, and increases with roughness.
//!
//! Reference:
//! Jenness, J.S. (2004). Calculating landscape surface area from digital
//! elevation models. *Wildlife Society Bulletin*, 32(3), 829–839.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Surface Area Ratio
#[derive(Debug, Clone)]
pub struct SarParams {
    /// Neighborhood radius in cells (default 1 → 3x3 window)
    pub radius: usize,
}

impl Default for SarParams {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

/// 8 neighbor offsets in clockwise order starting from E
const NEIGHBORS: [(isize, isize); 8] = [
    (0, 1),   // E
    (1, 1),   // SE
    (1, 0),   // S
    (1, -1),  // SW
    (0, -1),  // W
    (-1, -1), // NW
    (-1, 0),  // N
    (-1, 1),  // NE
];

/// Compute the area of a 3D triangle given three points (x, y, z).
/// Area = 0.5 * |cross_product(v1, v2)| where v1 = p1-p0, v2 = p2-p0
#[inline]
fn triangle_area_3d(
    x0: f64, y0: f64, z0: f64,
    x1: f64, y1: f64, z1: f64,
    x2: f64, y2: f64, z2: f64,
) -> f64 {
    let v1x = x1 - x0;
    let v1y = y1 - y0;
    let v1z = z1 - z0;
    let v2x = x2 - x0;
    let v2y = y2 - y0;
    let v2z = z2 - z0;

    // Cross product
    let cx = v1y * v2z - v1z * v2y;
    let cy = v1z * v2x - v1x * v2z;
    let cz = v1x * v2y - v1y * v2x;

    0.5 * (cx * cx + cy * cy + cz * cz).sqrt()
}

/// Compute Surface Area Ratio (3D surface area / 2D planimetric area).
///
/// For radius=1, computes 8 triangles formed by the center cell and
/// consecutive neighbor pairs (clockwise). Each triangle contributes
/// its 3D surface area. The 2D planimetric area is the total window
/// area minus the half-cells at edges (approximated as 2 * cell_size^2
/// for the 8 triangles covering the 3x3 window).
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - SAR parameters (neighborhood radius)
///
/// # Returns
/// Raster with SAR values (>= 1.0; flat = 1.0)
pub fn surface_area_ratio(dem: &Raster<f64>, params: SarParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius;

    if r == 0 {
        return Err(Error::Other("SAR radius must be >= 1".into()));
    }

    let gt = dem.transform();
    let cell_w = gt.pixel_width.abs();
    let cell_h = gt.pixel_height.abs();
    let nodata = dem.nodata();

    // For radius=1, use the 8-triangle Jenness method
    if r != 1 {
        return Err(Error::Other(
            "Currently only radius=1 (3x3 window) is supported for SAR".into(),
        ));
    }

    // 2D planimetric area covered by the 8 triangles around center
    // Each triangle has base = cell_size and height = cell_size (for cardinal)
    // or base/height involve diagonals. Total 2D area = 4 * cell_w * cell_h
    // (the 8 triangles tile the square excluding nothing — they exactly cover
    // the 2x2 cell area around center = 2*cell_w * 2*cell_h / 2 = 2*cell_w*cell_h)
    // Actually: 8 triangles with the center as apex, each spanning 45 degrees,
    // cover the area = 4 * (0.5 * cell_w * cell_h) = 2 * cell_w * cell_h for
    // cardinal triangles plus 4 * (0.5 * cell_w * cell_h) for diagonal.
    // Correct 2D area: sum of 8 triangles projected onto XY plane.

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                // Skip boundary cells
                if row == 0 || row >= rows - 1 || col == 0 || col >= cols - 1 {
                    continue;
                }

                let z_center = unsafe { dem.get_unchecked(row, col) };
                if z_center.is_nan()
                    || nodata.is_some_and(|nd| (z_center - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                let x_center = 0.0;
                let y_center = 0.0;

                let mut total_3d_area = 0.0;
                let mut total_2d_area = 0.0;
                let mut valid = true;

                for i in 0..8 {
                    let j = (i + 1) % 8;
                    let (dr1, dc1) = NEIGHBORS[i];
                    let (dr2, dc2) = NEIGHBORS[j];

                    let nr1 = (row as isize + dr1) as usize;
                    let nc1 = (col as isize + dc1) as usize;
                    let nr2 = (row as isize + dr2) as usize;
                    let nc2 = (col as isize + dc2) as usize;

                    let z1 = unsafe { dem.get_unchecked(nr1, nc1) };
                    let z2 = unsafe { dem.get_unchecked(nr2, nc2) };

                    if z1.is_nan()
                        || z2.is_nan()
                        || nodata.is_some_and(|nd| (z1 - nd).abs() < f64::EPSILON)
                        || nodata.is_some_and(|nd| (z2 - nd).abs() < f64::EPSILON)
                    {
                        valid = false;
                        break;
                    }

                    let x1 = dc1 as f64 * cell_w;
                    let y1 = -(dr1 as f64) * cell_h; // row increases downward
                    let x2 = dc2 as f64 * cell_w;
                    let y2 = -(dr2 as f64) * cell_h;

                    // 3D triangle area
                    total_3d_area += triangle_area_3d(
                        x_center, y_center, z_center,
                        x1, y1, z1,
                        x2, y2, z2,
                    );

                    // 2D triangle area (z=0)
                    total_2d_area += triangle_area_3d(
                        x_center, y_center, 0.0,
                        x1, y1, 0.0,
                        x2, y2, 0.0,
                    );
                }

                if valid && total_2d_area > 1e-20 {
                    row_data[col] = total_3d_area / total_2d_area;
                }
            }
            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_sar_flat_surface() {
        // Flat surface: 3D area == 2D area → SAR = 1.0
        let mut dem = Raster::filled(5, 5, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let result = surface_area_ratio(&dem, SarParams { radius: 1 }).unwrap();
        let val = result.get(2, 2).unwrap();

        assert!(
            (val - 1.0).abs() < 1e-6,
            "Flat surface should have SAR=1.0, got {}",
            val
        );
    }

    #[test]
    fn test_sar_rough_surface() {
        // Rough surface: center elevated above neighbors → SAR > 1.0
        let mut dem = Raster::filled(5, 5, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));
        dem.set(2, 2, 200.0).unwrap(); // Center 100m above neighbors

        let result = surface_area_ratio(&dem, SarParams { radius: 1 }).unwrap();
        let val = result.get(2, 2).unwrap();

        assert!(
            val > 1.0,
            "Rough surface should have SAR > 1.0, got {}",
            val
        );
    }

    #[test]
    fn test_sar_tilted_plane() {
        // Tilted plane: SAR slightly > 1 (depends on slope)
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));
        for row in 0..5 {
            for col in 0..5 {
                // Slope in both directions: z = 10*row + 10*col
                dem.set(row, col, 10.0 * row as f64 + 10.0 * col as f64)
                    .unwrap();
            }
        }

        let result = surface_area_ratio(&dem, SarParams { radius: 1 }).unwrap();
        let val = result.get(2, 2).unwrap();

        assert!(
            val >= 1.0,
            "Tilted plane should have SAR >= 1.0, got {}",
            val
        );
    }
}
