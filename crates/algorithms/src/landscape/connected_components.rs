//! Connected component labeling for classification rasters.
//!
//! Identifies contiguous patches of the same class using Union-Find.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Connectivity type for patch identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    /// 4-connected: cardinal neighbors only (N, S, E, W)
    Four,
    /// 8-connected: cardinal + diagonal neighbors
    Eight,
}

/// Label connected patches in a classification raster.
///
/// Each contiguous group of cells with the same class value gets a unique
/// integer label (1..N). NaN/nodata cells are labeled 0.
///
/// Uses two-pass Union-Find algorithm: O(n·α(n)) ≈ O(n).
///
/// # Returns
/// Tuple of (labeled raster with i32 patch IDs, number of patches found).
pub fn label_patches(
    classification: &Raster<f64>,
    connectivity: Connectivity,
) -> Result<(Raster<i32>, usize)> {
    let (rows, cols) = classification.shape();
    if rows == 0 || cols == 0 {
        return Err(Error::Other("Empty raster".into()));
    }

    let data = classification.data();
    let nodata = classification.nodata();

    // Union-Find data structure
    let mut parent: Vec<i32> = Vec::with_capacity(rows * cols / 4);
    let mut rank: Vec<u8> = Vec::new();
    let mut next_label: i32 = 1;

    let mut labels = Array2::<i32>::zeros((rows, cols));

    // Neighbor offsets based on connectivity
    let offsets_4: &[(isize, isize)] = &[(-1, 0), (0, -1)]; // up, left
    let offsets_8: &[(isize, isize)] = &[(-1, -1), (-1, 0), (-1, 1), (0, -1)];
    let offsets = match connectivity {
        Connectivity::Four => offsets_4,
        Connectivity::Eight => offsets_8,
    };

    // Initialize parent[0] = 0 (unused sentinel)
    parent.push(0);
    rank.push(0);

    // Pass 1: assign provisional labels
    for r in 0..rows {
        for c in 0..cols {
            let val = data[[r, c]];

            // Skip nodata
            if val.is_nan() {
                continue;
            }
            if let Some(nd) = nodata {
                if nd.is_finite() && (val - nd).abs() < 1e-10 {
                    continue;
                }
            }

            let class_val = val.round() as i64;

            // Check neighbors already processed
            let mut min_label: Option<i32> = None;
            let mut neighbor_labels: Vec<i32> = Vec::new();

            for &(dr, dc) in offsets {
                let nr = r as isize + dr;
                let nc = c as isize + dc;
                if nr < 0 || nc < 0 || nr >= rows as isize || nc >= cols as isize {
                    continue;
                }
                let nr = nr as usize;
                let nc = nc as usize;

                let nval = data[[nr, nc]];
                if nval.is_nan() {
                    continue;
                }
                let nclass = nval.round() as i64;

                if nclass == class_val && labels[[nr, nc]] != 0 {
                    let nl = find(&mut parent, labels[[nr, nc]]);
                    neighbor_labels.push(nl);
                    min_label = Some(match min_label {
                        Some(m) => m.min(nl),
                        None => nl,
                    });
                }
            }

            if let Some(ml) = min_label {
                labels[[r, c]] = ml;
                // Union all neighbor labels
                for &nl in &neighbor_labels {
                    if nl != ml {
                        union(&mut parent, &mut rank, ml, nl);
                    }
                }
            } else {
                // New label: extend parent/rank arrays
                let l = next_label;
                while parent.len() <= l as usize {
                    parent.push(parent.len() as i32);
                    rank.push(0);
                }
                next_label += 1;
                labels[[r, c]] = l;
            }
        }
    }

    // Pass 2: resolve labels and compact
    let mut label_map: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    let mut final_count: i32 = 0;

    for r in 0..rows {
        for c in 0..cols {
            let l = labels[[r, c]];
            if l == 0 {
                continue;
            }
            let root = find(&mut parent, l);
            let final_label = *label_map.entry(root).or_insert_with(|| {
                final_count += 1;
                final_count
            });
            labels[[r, c]] = final_label;
        }
    }

    let mut result = Raster::from_array(labels);
    result.set_transform(classification.transform().clone());
    if let Some(crs) = classification.crs() {
        result.set_crs(Some(crs.clone()));
    }
    result.set_nodata(Some(0));

    Ok((result, final_count as usize))
}

// Union-Find helpers

fn find(parent: &mut [i32], mut x: i32) -> i32 {
    while parent[x as usize] != x {
        // Path compression
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut [i32], rank: &mut [u8], a: i32, b: i32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    // Union by rank
    if rank[ra as usize] < rank[rb as usize] {
        parent[ra as usize] = rb;
    } else if rank[ra as usize] > rank[rb as usize] {
        parent[rb as usize] = ra;
    } else {
        parent[rb as usize] = ra;
        rank[ra as usize] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    fn make_class(data: Vec<Vec<f64>>) -> Raster<f64> {
        let rows = data.len();
        let cols = data[0].len();
        let flat: Vec<f64> = data.into_iter().flatten().collect();
        let arr = Array2::from_shape_vec((rows, cols), flat).unwrap();
        let mut r = Raster::from_array(arr);
        r.set_transform(GeoTransform::new(0.0, 0.0, 10.0, -10.0));
        r.set_nodata(Some(f64::NAN));
        r
    }

    #[test]
    fn test_checkerboard_4connected() {
        // Checkerboard: each pixel is its own patch with 4-connectivity
        let r = make_class(vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 1.0, 2.0],
            vec![1.0, 2.0, 1.0],
        ]);
        let (labels, count) = label_patches(&r, Connectivity::Four).unwrap();
        assert_eq!(count, 9); // 9 isolated patches
        // Each cell should have a unique label
        let mut seen = std::collections::HashSet::new();
        for r in 0..3 {
            for c in 0..3 {
                let l = labels.get(r, c).unwrap();
                assert!(l > 0);
                seen.insert(l);
            }
        }
        assert_eq!(seen.len(), 9);
    }

    #[test]
    fn test_checkerboard_8connected() {
        // With 8-connectivity, diagonal same-class cells connect
        let r = make_class(vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 1.0, 2.0],
            vec![1.0, 2.0, 1.0],
        ]);
        let (labels, count) = label_patches(&r, Connectivity::Eight).unwrap();
        // Class 1: corners + center = 5 cells, all 8-connected → 1 patch
        // Class 2: 4 cells on edges, all 8-connected → 1 patch
        assert_eq!(count, 2);

        let l_center = labels.get(1, 1).unwrap();
        let l_corner = labels.get(0, 0).unwrap();
        assert_eq!(l_center, l_corner); // same patch
    }

    #[test]
    fn test_three_blobs() {
        let r = make_class(vec![
            vec![1.0, 1.0, 0.0, 2.0, 2.0],
            vec![1.0, 1.0, 0.0, 2.0, 2.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![3.0, 3.0, 0.0, 3.0, 3.0],
        ]);
        let (_, count) = label_patches(&r, Connectivity::Four).unwrap();
        // class 1: 1 blob, class 2: 1 blob, class 0: 1 connected region, class 3: 2 blobs (separated)
        assert_eq!(count, 5);
    }

    #[test]
    fn test_nodata_excluded() {
        let r = make_class(vec![
            vec![1.0, f64::NAN, 1.0],
            vec![f64::NAN, f64::NAN, f64::NAN],
            vec![1.0, f64::NAN, 1.0],
        ]);
        let (labels, count) = label_patches(&r, Connectivity::Four).unwrap();
        assert_eq!(count, 4); // 4 isolated class-1 cells
        assert_eq!(labels.get(1, 1).unwrap(), 0); // NaN → 0
    }

    #[test]
    fn test_single_class() {
        let r = make_class(vec![
            vec![5.0, 5.0, 5.0],
            vec![5.0, 5.0, 5.0],
        ]);
        let (_, count) = label_patches(&r, Connectivity::Four).unwrap();
        assert_eq!(count, 1);
    }
}
