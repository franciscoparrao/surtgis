//! SLIC (Simple Linear Iterative Clustering) superpixels.
//!
//! Reference: Achanta, R., Shaji, A., Smith, K., Lucchi, A.,
//! Fua, P., & Süsstrunk, S. (2012). "SLIC Superpixels Compared to
//! State-of-the-art Superpixel Methods." IEEE Transactions on
//! Pattern Analysis and Machine Intelligence 34(11), 2274-2282.
//!
//! The algorithm runs k-means in `(n_bands + 2)`-dimensional space:
//! per-band feature values plus the spatial `(x, y)` coordinates.
//! The composite distance
//!
//!   D² = d_feat² + (m / S)² · d_space²
//!
//! controls the spatial-vs-spectral trade-off (`m = compactness`,
//! `S = sqrt(N / K)` = nominal superpixel side). Each cluster
//! centre only searches the `2S × 2S` window around itself, which
//! is what makes SLIC linear in the number of pixels.
//!
//! Per-band feature values are normalised to `[0, 1]` before
//! clustering so that the conventional compactness range
//! (`m ≈ 10` for "balanced", up to `40` for "very compact") works
//! regardless of input units.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

#[derive(Debug, Clone)]
pub struct SlicParams {
    /// Approximate target number of superpixels. The actual count is
    /// rounded to the nearest grid that fits the raster shape.
    pub n_segments: usize,
    /// Spatial-vs-spectral weight. Higher = more compact (geometric)
    /// superpixels; lower = more spectrally homogeneous. Default 10.
    pub compactness: f64,
    /// Maximum k-means iterations. Default 10 matches the original
    /// paper — SLIC converges fast.
    pub max_iter: usize,
    /// Enforce connectivity of each label as a post-process. Without
    /// this a small fraction of pixels can end up on islands far from
    /// their cluster centre. Default true.
    pub enforce_connectivity: bool,
}

impl Default for SlicParams {
    fn default() -> Self {
        Self {
            n_segments: 100,
            compactness: 10.0,
            max_iter: 10,
            enforce_connectivity: true,
        }
    }
}

/// Compute SLIC superpixels from one or more bands.
///
/// All band rasters must share the same shape. The output label
/// raster has the same shape and inherits the geotransform and CRS
/// from `bands[0]`. Labels are `1..=N` for the `N` superpixels
/// produced (`0` is reserved as the nodata sentinel — this matches
/// `connected_components` and is what
/// `statistics::zonal_statistics` expects). Pixels where any band
/// is NaN are emitted as `0`.
pub fn slic(bands: &[&Raster<f64>], params: SlicParams) -> Result<Raster<i32>> {
    if bands.is_empty() {
        return Err(Error::Algorithm("SLIC needs at least one band".into()));
    }
    if params.n_segments < 2 {
        return Err(Error::Algorithm("SLIC n_segments must be >= 2".into()));
    }
    if params.compactness <= 0.0 {
        return Err(Error::Algorithm("SLIC compactness must be > 0".into()));
    }
    let (rows, cols) = bands[0].shape();
    for b in bands.iter().skip(1) {
        if b.shape() != (rows, cols) {
            return Err(Error::Algorithm("SLIC bands must share shape".into()));
        }
    }
    if rows < 2 || cols < 2 {
        return Err(Error::Algorithm("SLIC requires raster >= 2x2".into()));
    }

    let n_bands = bands.len();
    let n_px = rows * cols;

    // Per-band normalisation to [0, 1] over finite values. NaN cells
    // are tracked via `valid_mask` and skipped end-to-end.
    let mut feat = vec![0.0f64; n_bands * n_px];
    let mut valid_mask = vec![true; n_px];
    for (b, raster) in bands.iter().enumerate() {
        let mut vmin = f64::INFINITY;
        let mut vmax = f64::NEG_INFINITY;
        for row in 0..rows {
            for col in 0..cols {
                let v = unsafe { raster.get_unchecked(row, col) };
                if v.is_finite() {
                    vmin = vmin.min(v);
                    vmax = vmax.max(v);
                } else {
                    valid_mask[row * cols + col] = false;
                }
            }
        }
        let range = if vmax > vmin { vmax - vmin } else { 1.0 };
        for row in 0..rows {
            for col in 0..cols {
                let v = unsafe { raster.get_unchecked(row, col) };
                let p = row * cols + col;
                feat[b * n_px + p] = if v.is_finite() {
                    (v - vmin) / range
                } else {
                    0.0
                };
            }
        }
    }

    // Grid step S ≈ sqrt(N/K). Round to ensure a sensible number of
    // seeds even when K is much smaller than N.
    let step = ((n_px as f64 / params.n_segments as f64).sqrt())
        .max(1.0)
        .round() as usize;
    let step = step.max(1);
    let n_rows_seed = (rows + step - 1) / step;
    let n_cols_seed = (cols + step - 1) / step;
    let n_clusters = n_rows_seed * n_cols_seed;

    // Initialise seeds on a regular grid centred in each cell.
    // Each centroid carries (band features..., y, x, count).
    let mut centroids = vec![0.0f64; n_clusters * (n_bands + 2)];
    let mut counts = vec![0.0f64; n_clusters];
    let dim = n_bands + 2;
    let half_step = step / 2;
    for sr in 0..n_rows_seed {
        for sc in 0..n_cols_seed {
            let cy = (sr * step + half_step).min(rows - 1);
            let cx = (sc * step + half_step).min(cols - 1);
            let k = sr * n_cols_seed + sc;
            for b in 0..n_bands {
                centroids[k * dim + b] = feat[b * n_px + cy * cols + cx];
            }
            centroids[k * dim + n_bands] = cy as f64;
            centroids[k * dim + n_bands + 1] = cx as f64;
        }
    }

    // Spatial-distance weight (m/S)².
    let m = params.compactness;
    let s = step as f64;
    let spatial_w2 = (m / s) * (m / s);

    let mut labels: Vec<i32> = vec![-1; n_px];
    let mut min_dist = vec![f64::INFINITY; n_px];

    for _iter in 0..params.max_iter {
        for d in &mut min_dist {
            *d = f64::INFINITY;
        }
        for v in &mut labels {
            *v = -1;
        }

        // Assignment: each centroid only searches its 2S×2S window.
        for k in 0..n_clusters {
            let cy = centroids[k * dim + n_bands];
            let cx = centroids[k * dim + n_bands + 1];
            if cy.is_nan() || cx.is_nan() {
                continue; // dead cluster (all-NaN window)
            }
            let r0 = (cy - s).max(0.0) as usize;
            let c0 = (cx - s).max(0.0) as usize;
            let r1 = ((cy + s) as usize + 1).min(rows);
            let c1 = ((cx + s) as usize + 1).min(cols);

            for row in r0..r1 {
                for col in c0..c1 {
                    let p = row * cols + col;
                    if !valid_mask[p] {
                        continue;
                    }
                    let mut d_feat2 = 0.0;
                    for b in 0..n_bands {
                        let diff = feat[b * n_px + p] - centroids[k * dim + b];
                        d_feat2 += diff * diff;
                    }
                    let dy = row as f64 - cy;
                    let dx = col as f64 - cx;
                    let d_space2 = dy * dy + dx * dx;
                    let d2 = d_feat2 + spatial_w2 * d_space2;
                    if d2 < min_dist[p] {
                        min_dist[p] = d2;
                        labels[p] = k as i32;
                    }
                }
            }
        }

        // Update: centroid = mean of assigned pixels.
        for v in centroids.iter_mut() {
            *v = 0.0;
        }
        for c in counts.iter_mut() {
            *c = 0.0;
        }
        for row in 0..rows {
            for col in 0..cols {
                let p = row * cols + col;
                let k = labels[p];
                if k < 0 {
                    continue;
                }
                let k = k as usize;
                for b in 0..n_bands {
                    centroids[k * dim + b] += feat[b * n_px + p];
                }
                centroids[k * dim + n_bands] += row as f64;
                centroids[k * dim + n_bands + 1] += col as f64;
                counts[k] += 1.0;
            }
        }
        for k in 0..n_clusters {
            if counts[k] > 0.0 {
                for j in 0..dim {
                    centroids[k * dim + j] /= counts[k];
                }
            } else {
                // Mark dead cluster so the assignment step skips it.
                centroids[k * dim + n_bands] = f64::NAN;
                centroids[k * dim + n_bands + 1] = f64::NAN;
            }
        }
    }

    // Optional connectivity enforcement: relabel small disconnected
    // components into a 4-neighbour they touch. This is a simplified
    // version of the original SLIC post-pass — components smaller
    // than `S²/4` are merged into the largest adjacent component.
    if params.enforce_connectivity {
        labels = enforce_connectivity_4(&labels, rows, cols, (step * step / 4).max(1));
    }

    // Compact label space to a dense `1..=N` range; 0 = nodata.
    let mut remap: std::collections::HashMap<i32, i32> = std::collections::HashMap::new();
    let mut next: i32 = 1;
    for v in labels.iter_mut() {
        if *v < 0 {
            *v = 0;
            continue;
        }
        let mapped = *remap.entry(*v).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        *v = mapped;
    }

    let mut data = Array2::<i32>::zeros((rows, cols));
    for (idx, v) in labels.into_iter().enumerate() {
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

/// Enforce 4-connectivity: any connected component smaller than
/// `min_size` is merged into the largest adjacent component (by
/// neighbour-count vote). Returns the rewritten label vector.
fn enforce_connectivity_4(labels: &[i32], rows: usize, cols: usize, min_size: usize) -> Vec<i32> {
    let n = rows * cols;
    // Component id per pixel, identified by Union-Find over 4-conn
    // edges whose endpoints share the same SLIC label.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    for row in 0..rows {
        for col in 0..cols {
            let p = row * cols + col;
            if labels[p] < 0 {
                continue;
            }
            if row > 0 && labels[(row - 1) * cols + col] == labels[p] {
                union(&mut parent, p, (row - 1) * cols + col);
            }
            if col > 0 && labels[row * cols + (col - 1)] == labels[p] {
                union(&mut parent, p, row * cols + (col - 1));
            }
        }
    }
    // Component sizes.
    let mut size = vec![0usize; n];
    let mut comp_label = vec![0i32; n];
    for p in 0..n {
        if labels[p] < 0 {
            continue;
        }
        let r = find(&mut parent, p);
        size[r] += 1;
        comp_label[r] = labels[p];
    }

    // For each small component (size < min_size), vote its dominant
    // 4-neighbour foreign label and re-tag every pixel in the small
    // component to that label.
    let mut out = labels.to_vec();
    // Build component -> pixel-list mapping for the small ones.
    let mut small_pixels: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for p in 0..n {
        if labels[p] < 0 {
            continue;
        }
        let r = find(&mut parent, p);
        if size[r] < min_size {
            small_pixels.entry(r).or_default().push(p);
        }
    }
    for (_root, pixels) in &small_pixels {
        let own = labels[pixels[0]];
        let mut votes: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
        for &p in pixels {
            let row = p / cols;
            let col = p % cols;
            let mut try_neighbour = |nr: usize, nc: usize| {
                let np = nr * cols + nc;
                let nl = labels[np];
                if nl >= 0 && nl != own {
                    *votes.entry(nl).or_insert(0) += 1;
                }
            };
            if row > 0 {
                try_neighbour(row - 1, col);
            }
            if row + 1 < rows {
                try_neighbour(row + 1, col);
            }
            if col > 0 {
                try_neighbour(row, col - 1);
            }
            if col + 1 < cols {
                try_neighbour(row, col + 1);
            }
        }
        if let Some((&winner, _)) = votes.iter().max_by_key(|&(_, &v)| v) {
            for &p in pixels {
                out[p] = winner;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    fn ramp_band(rows: usize, cols: usize, dir: char) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        for row in 0..rows {
            for col in 0..cols {
                let v = match dir {
                    'x' => col as f64,
                    'y' => row as f64,
                    _ => 0.0,
                };
                r.set(row, col, v).unwrap();
            }
        }
        r
    }

    fn flat_band(rows: usize, cols: usize, val: f64) -> Raster<f64> {
        let mut r = Raster::filled(rows, cols, val);
        r.set_transform(GeoTransform::new(0.0, rows as f64, 1.0, -1.0));
        r
    }

    #[test]
    fn slic_uniform_raster_produces_grid_segments() {
        // A perfectly flat band → the spectral term is zero → the
        // assignment is dominated by spatial distance → each pixel
        // joins its closest grid seed → the segmentation matches the
        // seed grid.
        let r = flat_band(20, 20, 7.0);
        let result = slic(
            &[&r],
            SlicParams {
                n_segments: 16,
                compactness: 10.0,
                max_iter: 5,
                enforce_connectivity: true,
            },
        )
        .unwrap();
        // 20x20 / 16 ≈ 25 → step ≈ 5; expect a 4×4 grid of segments.
        let mut seen = std::collections::HashSet::new();
        for row in 0..20 {
            for col in 0..20 {
                let v = result.get(row, col).unwrap();
                assert!(v >= 1, "uniform raster should produce labels >= 1");
                seen.insert(v);
            }
        }
        assert!(
            (12..=20).contains(&seen.len()),
            "expected ≈16 segments on uniform raster, got {}",
            seen.len()
        );
    }

    #[test]
    fn slic_two_blobs_separate_at_low_compactness() {
        // Half-and-half raster split vertically. With low compactness
        // the spectral term dominates → all left-half pixels share
        // labels distinct from all right-half pixels.
        let mut r = Raster::new(20, 20);
        r.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                r.set(row, col, if col < 10 { 0.0 } else { 1.0 }).unwrap();
            }
        }
        let result = slic(
            &[&r],
            SlicParams {
                n_segments: 4,
                compactness: 1.0,
                max_iter: 10,
                enforce_connectivity: true,
            },
        )
        .unwrap();
        // Collect the label sets of each half.
        let mut left: std::collections::HashSet<i32> = std::collections::HashSet::new();
        let mut right: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for row in 0..20 {
            for col in 0..20 {
                let v = result.get(row, col).unwrap();
                if col < 10 {
                    left.insert(v);
                } else {
                    right.insert(v);
                }
            }
        }
        // The two halves should never share a label at this compactness.
        for v in &left {
            assert!(
                !right.contains(v),
                "label {} leaked across blob boundary",
                v
            );
        }
    }

    #[test]
    fn slic_multi_band_shape_check() {
        let r0 = ramp_band(15, 15, 'x');
        let r1 = ramp_band(15, 15, 'y');
        let result = slic(&[&r0, &r1], SlicParams::default()).unwrap();
        assert_eq!(result.shape(), (15, 15));
    }

    #[test]
    fn slic_nan_pixels_get_sentinel() {
        let mut r = Raster::filled(10, 10, 1.0);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        r.set(5, 5, f64::NAN).unwrap();
        let result = slic(
            &[&r],
            SlicParams {
                n_segments: 4,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.get(5, 5).unwrap(), 0);
        assert!(result.nodata().is_some_and(|nd| nd == 0));
    }

    #[test]
    fn slic_mismatched_shapes_errors() {
        let r0 = flat_band(10, 10, 0.0);
        let r1 = flat_band(10, 12, 0.0);
        assert!(slic(&[&r0, &r1], SlicParams::default()).is_err());
    }

    #[test]
    fn slic_rejects_empty_bands() {
        assert!(slic(&[], SlicParams::default()).is_err());
    }
}
