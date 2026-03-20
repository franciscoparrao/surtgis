//! Class-level and landscape-level metrics: SHDI, SIDI, AI, COHESION.

use super::patch_metrics::PatchStats;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};
use std::collections::HashMap;

/// Landscape-level metrics (whole landscape).
#[derive(Debug, Clone)]
pub struct LandscapeMetrics {
    /// Shannon Diversity Index: -Σ(p_i × ln(p_i))
    pub shdi: f64,
    /// Simpson Diversity Index: 1 - Σ(p_i²)
    pub sidi: f64,
    /// Total number of patches
    pub num_patches: usize,
    /// Number of distinct classes
    pub num_classes: usize,
    /// Total landscape area in map units²
    pub total_area_m2: f64,
    /// Total number of valid cells
    pub total_cells: usize,
}

/// Per-class metrics.
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    /// Class value
    pub class: i64,
    /// Total area of this class in map units²
    pub area_m2: f64,
    /// Proportion of landscape occupied by this class
    pub proportion: f64,
    /// Number of patches of this class
    pub num_patches: usize,
    /// Mean patch area for this class
    pub mean_patch_area_m2: f64,
    /// Aggregation Index: (g_ii / max_g_ii) × 100
    pub ai: f64,
    /// Patch Cohesion Index
    pub cohesion: f64,
}

/// Compute global landscape-level metrics from a classification raster.
///
/// Calculates SHDI, SIDI, and basic composition statistics.
/// Does not require labeled patches — works directly on the classification.
pub fn landscape_metrics(classification: &Raster<f64>) -> Result<LandscapeMetrics> {
    let (rows, cols) = classification.shape();
    let data = classification.data();
    let nodata = classification.nodata();

    // Count cells per class
    let mut class_counts: HashMap<i64, usize> = HashMap::new();
    let mut total_valid = 0usize;

    for r in 0..rows {
        for c in 0..cols {
            let v = data[[r, c]];
            if v.is_nan() {
                continue;
            }
            if let Some(nd) = nodata {
                if nd.is_finite() && (v - nd).abs() < 1e-10 {
                    continue;
                }
            }
            *class_counts.entry(v.round() as i64).or_insert(0) += 1;
            total_valid += 1;
        }
    }

    if total_valid == 0 {
        return Err(Error::Other("No valid cells in classification".into()));
    }

    let gt = classification.transform();
    let cell_area = gt.pixel_width.abs() * gt.pixel_height.abs();

    // Compute proportions and diversity indices
    let mut shdi = 0.0f64;
    let mut sidi = 0.0f64;

    for &count in class_counts.values() {
        let p = count as f64 / total_valid as f64;
        if p > 0.0 {
            shdi -= p * p.ln();
            sidi += p * p;
        }
    }
    sidi = 1.0 - sidi;

    Ok(LandscapeMetrics {
        shdi,
        sidi,
        num_patches: 0, // filled by caller if patches are available
        num_classes: class_counts.len(),
        total_area_m2: total_valid as f64 * cell_area,
        total_cells: total_valid,
    })
}

/// Compute per-class metrics from classification, labels, and patch stats.
///
/// Requires pre-computed patches from `patch_metrics()`.
pub fn class_metrics(
    classification: &Raster<f64>,
    patches: &[PatchStats],
) -> Result<Vec<ClassMetrics>> {
    let (rows, cols) = classification.shape();
    let data = classification.data();
    let nodata = classification.nodata();
    let gt = classification.transform();
    let cell_area = gt.pixel_width.abs() * gt.pixel_height.abs();

    // Count total valid cells
    let mut total_valid = 0usize;
    for r in 0..rows {
        for c in 0..cols {
            let v = data[[r, c]];
            if v.is_nan() {
                continue;
            }
            if let Some(nd) = nodata {
                if nd.is_finite() && (v - nd).abs() < 1e-10 {
                    continue;
                }
            }
            total_valid += 1;
        }
    }

    let total_area = total_valid as f64 * cell_area;

    // Aggregate patches by class
    let mut class_patches: HashMap<i64, Vec<&PatchStats>> = HashMap::new();
    for p in patches {
        class_patches.entry(p.class).or_default().push(p);
    }

    // Count same-class adjacencies for AI
    let mut class_adjacencies: HashMap<i64, usize> = HashMap::new();
    let mut class_cells: HashMap<i64, usize> = HashMap::new();

    for r in 0..rows {
        for c in 0..cols {
            let v = data[[r, c]];
            if v.is_nan() {
                continue;
            }
            if let Some(nd) = nodata {
                if nd.is_finite() && (v - nd).abs() < 1e-10 {
                    continue;
                }
            }
            let cls = v.round() as i64;
            *class_cells.entry(cls).or_insert(0) += 1;

            // Check right and down neighbors (avoid double-counting)
            for &(dr, dc) in &[(0i32, 1i32), (1, 0)] {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nc >= 0 && (nr as usize) < rows && (nc as usize) < cols {
                    let nv = data[[nr as usize, nc as usize]];
                    if !nv.is_nan() && nv.round() as i64 == cls {
                        *class_adjacencies.entry(cls).or_insert(0) += 1;
                    }
                }
            }
        }
    }

    // Build per-class metrics
    let mut results: Vec<ClassMetrics> = Vec::new();

    for (&cls, cpatches) in &class_patches {
        let n_cells = class_cells.get(&cls).copied().unwrap_or(0);
        let area = n_cells as f64 * cell_area;
        let proportion = if total_area > 0.0 {
            area / total_area
        } else {
            0.0
        };
        let num_p = cpatches.len();
        let mean_area = if num_p > 0 {
            area / num_p as f64
        } else {
            0.0
        };

        // AI = (g_ii / max_g_ii) × 100
        // max_g_ii for n cells = 2n - 2√n (approximate for compact arrangement)
        // More precisely: for a grid, max adjacencies = 2*n - cols - rows of bounding rect
        // Simplified: max_g_ii = 2*(n_cells) - 2*ceil(sqrt(n_cells))
        let g_ii = class_adjacencies.get(&cls).copied().unwrap_or(0) as f64;
        let n = n_cells as f64;
        let max_g_ii = if n > 1.0 {
            2.0 * n - 2.0 * n.sqrt().ceil()
        } else {
            1.0
        };
        let ai = if max_g_ii > 0.0 {
            (g_ii / max_g_ii * 100.0).min(100.0)
        } else {
            0.0
        };

        // COHESION = [1 - Σ(p_j) / Σ(p_j × √a_j)] × [1 - 1/√A] × 100
        // where p_j = perimeter of patch j (in cells), a_j = area of patch j (in cells)
        let sum_p: f64 = cpatches.iter().map(|p| p.perimeter_edges as f64).sum();
        let sum_p_sqrt_a: f64 = cpatches
            .iter()
            .map(|p| p.perimeter_edges as f64 * (p.area_cells as f64).sqrt())
            .sum();
        let total_a = total_valid as f64;

        let cohesion = if sum_p_sqrt_a > 0.0 && total_a > 1.0 {
            let term1 = 1.0 - sum_p / sum_p_sqrt_a;
            let term2 = 1.0 - 1.0 / total_a.sqrt();
            if term2 > 0.0 {
                (term1 / term2 * 100.0).clamp(0.0, 100.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        results.push(ClassMetrics {
            class: cls,
            area_m2: area,
            proportion,
            num_patches: num_p,
            mean_patch_area_m2: mean_area,
            ai,
            cohesion,
        });
    }

    // Sort by class
    results.sort_by_key(|m| m.class);
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::landscape::connected_components::{label_patches, Connectivity};
    use crate::landscape::patch_metrics::patch_metrics as compute_patches;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn make_class(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 1.0, -1.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_shdi_two_equal_classes() {
        // 50/50 split → SHDI = ln(2) ≈ 0.693
        let r = make_class(vec![
            vec![1.0, 1.0, 2.0, 2.0],
            vec![1.0, 1.0, 2.0, 2.0],
        ]);
        let lm = landscape_metrics(&r).unwrap();
        assert!((lm.shdi - 2.0f64.ln()).abs() < 1e-6);
        assert!((lm.sidi - 0.5).abs() < 1e-6);
        assert_eq!(lm.num_classes, 2);
    }

    #[test]
    fn test_shdi_single_class() {
        let r = make_class(vec![vec![1.0; 10]; 10]);
        let lm = landscape_metrics(&r).unwrap();
        assert!((lm.shdi - 0.0).abs() < 1e-10);
        assert!((lm.sidi - 0.0).abs() < 1e-10);
        assert_eq!(lm.num_classes, 1);
    }

    #[test]
    fn test_ai_compact_class() {
        // All cells same class → high AI
        let r = make_class(vec![vec![1.0; 10]; 10]);
        let (labels, n) = label_patches(&r, Connectivity::Four).unwrap();
        let patches = compute_patches(&r, &labels, n).unwrap();
        let cm = class_metrics(&r, &patches).unwrap();
        assert_eq!(cm.len(), 1);
        assert!(cm[0].ai > 90.0); // highly aggregated
    }

    #[test]
    fn test_ai_dispersed_class() {
        // Checkerboard → low AI
        let mut data = vec![vec![0.0; 10]; 10];
        for r in 0..10 {
            for c in 0..10 {
                data[r][c] = ((r + c) % 2) as f64;
            }
        }
        let r = make_class(data);
        let (labels, n) = label_patches(&r, Connectivity::Four).unwrap();
        let patches = compute_patches(&r, &labels, n).unwrap();
        let cm = class_metrics(&r, &patches).unwrap();
        // Both classes should have AI = 0 (no same-class adjacencies in checkerboard)
        for c in &cm {
            assert!(c.ai < 1.0, "AI should be ~0 for checkerboard, got {}", c.ai);
        }
    }

    #[test]
    fn test_cohesion_compact() {
        // Single large patch → high cohesion
        let r = make_class(vec![vec![1.0; 20]; 20]);
        let (labels, n) = label_patches(&r, Connectivity::Four).unwrap();
        let patches = compute_patches(&r, &labels, n).unwrap();
        let cm = class_metrics(&r, &patches).unwrap();
        assert!(cm[0].cohesion > 90.0);
    }
}
