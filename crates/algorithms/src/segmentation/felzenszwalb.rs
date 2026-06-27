//! Felzenszwalb–Huttenlocher graph segmentation.
//!
//! Reference: Felzenszwalb, P. F., & Huttenlocher, D. P. (2004).
//! "Efficient Graph-Based Image Segmentation." International
//! Journal of Computer Vision 59(2), 167-181.
//!
//! Builds an 8-connected pixel graph weighted by the Euclidean
//! distance between per-band feature vectors, sorts the edges in
//! ascending weight order, and processes them with Union-Find. The
//! merge rule is
//!
//!   w(e) ≤ min(Int(C₁) + k/|C₁|, Int(C₂) + k/|C₂|)
//!
//! where `Int(C)` is the maximum-edge weight in the MST of `C` so
//! far (which, given the ascending traversal, is just the weight of
//! the merging edge at the time of the last merge inside `C`), and
//! `k` is the user-controlled scale parameter. Larger `k` → larger
//! merged segments. After the merging pass, components smaller
//! than `min_size` are folded into their cheapest neighbour.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

#[derive(Debug, Clone)]
pub struct FelzenszwalbParams {
    /// Scale parameter `k`. Roughly the smallest edge weight that
    /// can cross a segment boundary. Higher → larger segments.
    pub scale: f64,
    /// Minimum allowed component size in pixels. Smaller components
    /// are merged into their cheapest 8-neighbour after the main pass.
    pub min_size: usize,
}

impl Default for FelzenszwalbParams {
    fn default() -> Self {
        Self {
            scale: 1.0,
            min_size: 20,
        }
    }
}

/// Felzenszwalb–Huttenlocher segmentation.
///
/// All band rasters must share the same shape. Output is a label
/// raster (`Raster<i32>`) with one dense integer per segment
/// (`1..=N`). Pixels with NaN in any band are emitted as `0`
/// (`nodata` is set to `Some(0)`) — this matches the
/// `connected_components` convention and is what
/// `statistics::zonal_statistics` expects.
pub fn felzenszwalb(bands: &[&Raster<f64>], params: FelzenszwalbParams) -> Result<Raster<i32>> {
    if bands.is_empty() {
        return Err(Error::Algorithm(
            "Felzenszwalb needs at least one band".into(),
        ));
    }
    if params.scale <= 0.0 {
        return Err(Error::Algorithm("Felzenszwalb scale must be > 0".into()));
    }
    let (rows, cols) = bands[0].shape();
    for b in bands.iter().skip(1) {
        if b.shape() != (rows, cols) {
            return Err(Error::Algorithm(
                "Felzenszwalb bands must share shape".into(),
            ));
        }
    }
    if rows < 2 || cols < 2 {
        return Err(Error::Algorithm(
            "Felzenszwalb requires raster >= 2x2".into(),
        ));
    }

    let n_bands = bands.len();
    let n_px = rows * cols;

    // Build per-pixel feature vector and validity mask.
    let mut feat = vec![0.0f64; n_bands * n_px];
    let mut valid = vec![true; n_px];
    for (b, raster) in bands.iter().enumerate() {
        for row in 0..rows {
            for col in 0..cols {
                let v = unsafe { raster.get_unchecked(row, col) };
                let p = row * cols + col;
                if v.is_finite() {
                    feat[b * n_px + p] = v;
                } else {
                    valid[p] = false;
                }
            }
        }
    }

    // 8-connected edges, each pixel adding only the four edges to
    // (row+0,col+1), (row+1,col-1), (row+1,col), (row+1,col+1) so
    // every edge appears once.
    let offsets: [(isize, isize); 4] = [(0, 1), (1, -1), (1, 0), (1, 1)];
    let edge_weight = |a: usize, b: usize| -> f64 {
        let mut s = 0.0;
        for k in 0..n_bands {
            let d = feat[k * n_px + a] - feat[k * n_px + b];
            s += d * d;
        }
        s.sqrt()
    };

    let mut edges: Vec<(f64, u32, u32)> = Vec::with_capacity(4 * n_px);
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if !valid[p] {
                continue;
            }
            for (dr, dc) in offsets.iter() {
                let nr = row as isize + dr;
                let nc = col as isize + dc;
                if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                    continue;
                }
                let q = (nr as usize) * cols + (nc as usize);
                if !valid[q] {
                    continue;
                }
                let w = edge_weight(p, q);
                edges.push((w, p as u32, q as u32));
            }
        }
    }
    edges.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Union-Find with size + int_diff per root.
    let mut parent: Vec<u32> = (0..n_px as u32).collect();
    let mut size: Vec<u32> = vec![1; n_px];
    let mut int_diff: Vec<f64> = vec![0.0; n_px];

    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            let gp = parent[parent[x as usize] as usize];
            parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    let k_scale = params.scale;
    for &(w, u, v) in edges.iter() {
        let ru = find(&mut parent, u);
        let rv = find(&mut parent, v);
        if ru == rv {
            continue;
        }
        let su = size[ru as usize] as f64;
        let sv = size[rv as usize] as f64;
        let tu = int_diff[ru as usize] + k_scale / su;
        let tv = int_diff[rv as usize] + k_scale / sv;
        let threshold = tu.min(tv);
        if w <= threshold {
            // Union by size (large absorbs small).
            let (new_root, old_root) = if su >= sv { (ru, rv) } else { (rv, ru) };
            parent[old_root as usize] = new_root;
            size[new_root as usize] = (su + sv) as u32;
            int_diff[new_root as usize] = w;
        }
    }

    // Post-process: merge components smaller than min_size into
    // their cheapest neighbour via a second edge pass.
    if params.min_size > 1 {
        for &(_w, u, v) in edges.iter() {
            let ru = find(&mut parent, u);
            let rv = find(&mut parent, v);
            if ru == rv {
                continue;
            }
            let su = size[ru as usize];
            let sv = size[rv as usize];
            if (su as usize) < params.min_size || (sv as usize) < params.min_size {
                let (new_root, old_root) = if su >= sv { (ru, rv) } else { (rv, ru) };
                parent[old_root as usize] = new_root;
                size[new_root as usize] = su + sv;
                // int_diff stays — only matters for the main pass.
            }
        }
    }

    // Compact root ids into dense 1..=N range; 0 = nodata.
    let mut remap: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
    let mut next: i32 = 1;
    let mut labels_flat = vec![0i32; n_px];
    for p in 0..n_px {
        if !valid[p] {
            continue;
        }
        let r = find(&mut parent, p as u32);
        let id = *remap.entry(r).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        labels_flat[p] = id;
    }

    let mut data = Array2::<i32>::zeros((rows, cols));
    for (idx, v) in labels_flat.into_iter().enumerate() {
        let row = idx / cols;
        let col = idx % cols;
        data[[row, col]] = v;
    }

    let mut out = Raster::from_array(data);
    out.set_transform(bands[0].transform().clone());
    if let Some(crs) = bands[0].crs() {
        out.set_crs(Some(crs.clone()));
    }
    out.set_nodata(Some(0));
    Ok(out)
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
    fn flat_raster_one_segment() {
        let mut r = Raster::filled(10, 10, 1.0);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        let out = felzenszwalb(&[&r], FelzenszwalbParams::default()).unwrap();
        let first = out.get(0, 0).unwrap();
        for row in 0..10 {
            for col in 0..10 {
                assert_eq!(
                    out.get(row, col).unwrap(),
                    first,
                    "flat raster should yield a single segment"
                );
            }
        }
    }

    #[test]
    fn two_distinct_regions_two_segments() {
        // Left half = 0, right half = 100. With small k the high-cost
        // boundary edges (weight 100) exceed the merge threshold
        // (k / size ≪ 100), so the two halves stay separate. With
        // min_size below 50 we don't trigger the post-pass merge.
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                r.set(row, col, if col < 5 { 0.0 } else { 100.0 }).unwrap();
            }
        }
        let out = felzenszwalb(
            &[&r],
            FelzenszwalbParams {
                scale: 1.0,
                min_size: 10,
            },
        )
        .unwrap();
        let left = out.get(0, 0).unwrap();
        let right = out.get(0, 9).unwrap();
        assert_ne!(left, right);
        for row in 0..10 {
            assert_eq!(out.get(row, 0).unwrap(), left);
            assert_eq!(out.get(row, 9).unwrap(), right);
        }
    }

    #[test]
    fn min_size_post_pass_absorbs_singletons() {
        // A 7x7 image with a single high-contrast pixel surrounded
        // by a flat background. Without min_size the singleton would
        // be its own segment; min_size=5 forces a merge into the
        // background.
        let mut r = Raster::filled(7, 7, 0.0);
        r.set_transform(GeoTransform::new(0.0, 7.0, 1.0, -1.0));
        r.set(3, 3, 50.0).unwrap();
        let out_no_min = felzenszwalb(
            &[&r],
            FelzenszwalbParams {
                scale: 0.1,
                min_size: 1,
            },
        )
        .unwrap();
        let out_with_min = felzenszwalb(
            &[&r],
            FelzenszwalbParams {
                scale: 0.1,
                min_size: 5,
            },
        )
        .unwrap();
        // With no min_size the singleton sits in its own segment.
        let singleton = out_no_min.get(3, 3).unwrap();
        let bg = out_no_min.get(0, 0).unwrap();
        assert_ne!(singleton, bg);
        // With min_size=5 it gets absorbed.
        assert_eq!(
            out_with_min.get(3, 3).unwrap(),
            out_with_min.get(0, 0).unwrap()
        );
    }

    #[test]
    fn nan_pixels_emit_sentinel() {
        let mut r = raster_from_grid(&[&[0.0, 0.0, 0.0], &[0.0, f64::NAN, 0.0], &[0.0, 0.0, 0.0]]);
        let _ = &mut r; // settle the borrow
        let out = felzenszwalb(&[&r], FelzenszwalbParams::default()).unwrap();
        assert_eq!(out.get(1, 1).unwrap(), 0);
        assert!(out.nodata().is_some_and(|nd| nd == 0));
    }

    #[test]
    fn multi_band_smoke() {
        let r0 = raster_from_grid(&[
            &[0.0, 0.0, 5.0, 5.0],
            &[0.0, 0.0, 5.0, 5.0],
            &[0.0, 0.0, 5.0, 5.0],
            &[0.0, 0.0, 5.0, 5.0],
        ]);
        let r1 = raster_from_grid(&[
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0, 1.0],
        ]);
        let out = felzenszwalb(
            &[&r0, &r1],
            FelzenszwalbParams {
                scale: 1.0,
                min_size: 1,
            },
        )
        .unwrap();
        let left = out.get(0, 0).unwrap();
        let right = out.get(0, 3).unwrap();
        assert_ne!(left, right, "spectral difference should split blocks");
    }

    #[test]
    fn mismatched_shapes_errors() {
        let r0 = Raster::filled(5, 5, 0.0);
        let r1 = Raster::filled(5, 6, 0.0);
        assert!(felzenszwalb(&[&r0, &r1], FelzenszwalbParams::default()).is_err());
    }
}
