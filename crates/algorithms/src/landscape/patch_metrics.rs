//! Per-patch landscape metrics: area, perimeter, PARA, FRAC.

use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Per-patch statistics.
#[derive(Debug, Clone)]
pub struct PatchStats {
    /// Patch label (from connected components)
    pub label: i32,
    /// Class value from the original classification
    pub class: i64,
    /// Number of cells in the patch
    pub area_cells: usize,
    /// Area in map units squared (cells × cell_width × cell_height)
    pub area_m2: f64,
    /// Number of edge segments (cell faces adjacent to different class or border)
    pub perimeter_edges: usize,
    /// Perimeter in map units
    pub perimeter_m: f64,
    /// Perimeter-Area Ratio = perimeter_m / area_m2
    pub para: f64,
    /// Fractal Dimension Index = 2 × ln(0.25 × perimeter_m) / ln(area_m2)
    pub frac: f64,
}

/// Compute per-patch metrics for all labeled patches.
///
/// # Arguments
/// * `classification` - Original classification raster (f64, values rounded to class IDs)
/// * `labels` - Labeled patch raster from `label_patches()` (i32, 0 = nodata)
/// * `num_patches` - Number of patches (from `label_patches()`)
pub fn patch_metrics(
    classification: &Raster<f64>,
    labels: &Raster<i32>,
    num_patches: usize,
) -> Result<Vec<PatchStats>> {
    let (rows, cols) = classification.shape();
    let (lr, lc) = labels.shape();
    if rows != lr || cols != lc {
        return Err(Error::Other(format!(
            "Classification {}x{} doesn't match labels {}x{}",
            rows, cols, lr, lc
        )));
    }

    let class_data = classification.data();
    let label_data = labels.data();
    let gt = classification.transform();
    let cell_w = gt.pixel_width.abs();
    let cell_h = gt.pixel_height.abs();
    let cell_area = cell_w * cell_h;

    // Accumulate per-patch: area (cell count), perimeter edges, class
    let mut area: Vec<usize> = vec![0; num_patches + 1];
    let mut perimeter: Vec<usize> = vec![0; num_patches + 1];
    let mut patch_class: Vec<i64> = vec![0; num_patches + 1];

    for r in 0..rows {
        for c in 0..cols {
            let l = label_data[[r, c]];
            if l <= 0 {
                continue;
            }
            let li = l as usize;
            area[li] += 1;
            patch_class[li] = class_data[[r, c]].round() as i64;

            // Count perimeter: each of 4 cardinal faces that borders
            // a different class or the raster edge
            let class_val = class_data[[r, c]].round() as i64;
            for &(dr, dc) in &[(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr < 0 || nc < 0 || nr >= rows as i32 || nc >= cols as i32 {
                    // Border edge
                    perimeter[li] += 1;
                } else {
                    let nval = class_data[[nr as usize, nc as usize]];
                    if nval.is_nan() || nval.round() as i64 != class_val {
                        perimeter[li] += 1;
                    }
                }
            }
        }
    }

    // Build PatchStats
    let mut results = Vec::with_capacity(num_patches);
    for i in 1..=num_patches {
        if area[i] == 0 {
            continue;
        }
        let a_m2 = area[i] as f64 * cell_area;
        let p_m = perimeter[i] as f64 * ((cell_w + cell_h) / 2.0); // avg edge length

        let para = if a_m2 > 0.0 { p_m / a_m2 } else { 0.0 };
        let frac = if a_m2 > 1.0 && p_m > 0.0 {
            2.0 * (0.25 * p_m).ln() / a_m2.ln()
        } else {
            1.0 // degenerate case
        };

        results.push(PatchStats {
            label: i as i32,
            class: patch_class[i],
            area_cells: area[i],
            area_m2: a_m2,
            perimeter_edges: perimeter[i],
            perimeter_m: p_m,
            para,
            frac,
        });
    }

    Ok(results)
}

/// Format patch metrics as CSV string.
pub fn patches_to_csv(patches: &[PatchStats]) -> String {
    let mut csv = String::from("label,class,area_cells,area_m2,perimeter_edges,perimeter_m,para,frac\n");
    for p in patches {
        csv.push_str(&format!(
            "{},{},{},{:.2},{},{:.2},{:.6},{:.4}\n",
            p.label, p.class, p.area_cells, p.area_m2,
            p.perimeter_edges, p.perimeter_m, p.para, p.frac
        ));
    }
    csv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landscape::connected_components::{label_patches, Connectivity};
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn make_class(data: Vec<Vec<f64>>, cell_size: f64) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, cell_size, -cell_size));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_square_patch() {
        // 4x4 all class 1 → 1 patch, area=16, perimeter=16 edges
        let r = make_class(vec![
            vec![1.0; 4],
            vec![1.0; 4],
            vec![1.0; 4],
            vec![1.0; 4],
        ], 10.0);
        let (labels, n) = label_patches(&r, Connectivity::Four).unwrap();
        let patches = patch_metrics(&r, &labels, n).unwrap();
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].area_cells, 16);
        assert_eq!(patches[0].perimeter_edges, 16); // 4 sides × 4 edge cells each
        assert!((patches[0].area_m2 - 1600.0).abs() < 1e-6); // 16 × 100
    }

    #[test]
    fn test_rectangle_patch() {
        // 2x5 all class 1 → 1 patch
        let r = make_class(vec![
            vec![1.0; 5],
            vec![1.0; 5],
        ], 1.0);
        let (labels, n) = label_patches(&r, Connectivity::Four).unwrap();
        let patches = patch_metrics(&r, &labels, n).unwrap();
        assert_eq!(patches[0].area_cells, 10);
        assert_eq!(patches[0].perimeter_edges, 14); // 2*(2+5)
    }

    #[test]
    fn test_single_pixel_patch() {
        // Single pixel surrounded by different class
        let r = make_class(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0],
        ], 10.0);
        let (labels, n) = label_patches(&r, Connectivity::Four).unwrap();
        let patches = patch_metrics(&r, &labels, n).unwrap();
        // Find the class-1 patch
        let p1 = patches.iter().find(|p| p.class == 1).unwrap();
        assert_eq!(p1.area_cells, 1);
        assert_eq!(p1.perimeter_edges, 4);
        assert!((p1.frac - 1.0).abs() < 1e-6); // degenerate
    }
}
