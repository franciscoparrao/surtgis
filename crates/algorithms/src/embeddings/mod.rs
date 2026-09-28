//! Foundation-model embeddings as first-class rasters.
//!
//! Products such as AlphaEarth Foundations (64-d, 10 m, yearly), TESSERA
//! (128-d) or any encoder you run yourself ship as multi-band rasters
//! whose bands are the dimensions of one vector per cell. This module
//! treats a band stack as a field of vectors and offers the operations
//! that make such a field useful without a GPU or a notebook:
//!
//! - [`vector_at`] / [`mean_vector`]: the vector of one cell, or the mean
//!   over a set of cells (a polygon, a class sample), as a reference;
//! - [`similarity`]: per-cell cosine similarity, dot product or Euclidean
//!   distance to a reference vector — the "find more like this" map;
//! - [`PcaModel`]: a PCA fitted once (on a sample, so it is cheap and
//!   stable across tiles) and projected anywhere, the standard way to see
//!   a 64-d field as an RGB image; it serialises to JSON so a server can
//!   fit it per source and reuse it per tile;
//! - [`norm`]: the L2 norm per cell, to check whether a product is
//!   unit-normalised (then dot product = cosine).
//!
//! Quantised products (int8 with a linear scale) need no de-quantisation
//! for cosine similarity or PCA: a global scale cancels in the cosine and
//! only rescales the principal axes. Apply the product's scale/offset via
//! band math first if the physical values matter (dot product, distance).
//!
//! Nodata: a cell is valid only when every band is finite; otherwise the
//! result is NaN.

use crate::maybe_rayon::*;
use serde::{Deserialize, Serialize};
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// How a stored (quantised) product maps to the vectors it represents.
///
/// Cosine similarity is invariant to a *linear* scale, so quantised
/// products need no de-quantisation for it; a non-linear code such as
/// AlphaEarth's does, and dot products and distances always do.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Dequantize {
    /// Values as stored.
    #[default]
    None,
    /// `f = v * scale + offset`.
    Linear {
        /// Multiplier.
        scale: f64,
        /// Added after scaling.
        offset: f64,
    },
    /// AlphaEarth Foundations int8 code: `f = sign(v) · (v / 127.5)²`,
    /// which restores unit-norm vectors (nodata −128 must already be NaN).
    AlphaEarth,
}

impl Dequantize {
    /// Parse `none` | `alphaearth` | `linear:SCALE[,OFFSET]`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        match s.to_ascii_lowercase().as_str() {
            "none" | "" => return Some(Self::None),
            "alphaearth" | "aef" => return Some(Self::AlphaEarth),
            _ => {}
        }
        let rest = s.strip_prefix("linear:")?;
        let mut it = rest.split(',');
        let scale: f64 = it.next()?.trim().parse().ok()?;
        let offset: f64 = match it.next() {
            Some(o) => o.trim().parse().ok()?,
            None => 0.0,
        };
        Some(Self::Linear { scale, offset })
    }

    /// Apply to one stored value.
    #[inline]
    pub fn apply(self, v: f64) -> f64 {
        match self {
            Self::None => v,
            Self::Linear { scale, offset } => v * scale + offset,
            Self::AlphaEarth => {
                let u = v / 127.5;
                u.signum() * u * u
            }
        }
    }

    /// Apply to every cell of a stack, in place.
    pub fn apply_stack(self, bands: &mut [Raster<f64>]) {
        if self == Self::None {
            return;
        }
        for b in bands {
            b.data_mut().mapv_inplace(|v| self.apply(v));
        }
    }
}

/// How two vectors are compared in [`similarity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SimilarityMetric {
    /// `⟨v, r⟩ / (‖v‖ ‖r‖)` in `[-1, 1]`. Scale-free: works on quantised
    /// products as stored.
    #[default]
    Cosine,
    /// `⟨v, r⟩`; equals cosine when the product is unit-normalised.
    Dot,
    /// `‖v − r‖` (≥ 0; 0 = identical).
    Euclidean,
}

impl SimilarityMetric {
    /// Parse `cosine` | `dot` | `euclidean`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cosine" | "cos" => Some(Self::Cosine),
            "dot" => Some(Self::Dot),
            "euclidean" | "l2" | "distance" => Some(Self::Euclidean),
            _ => None,
        }
    }

    /// Natural colormap domain of the metric.
    pub fn default_range(self) -> Option<(f64, f64)> {
        match self {
            Self::Cosine => Some((-1.0, 1.0)),
            Self::Dot | Self::Euclidean => None,
        }
    }
}

/// Parameters of [`similarity`].
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SimilarityParams {
    /// Comparison metric.
    pub metric: SimilarityMetric,
    /// Applied to every stored value before comparing (the reference is
    /// taken as already de-quantised).
    pub dequantize: Dequantize,
}

fn check_stack(bands: &[&Raster<f64>]) -> Result<(usize, usize)> {
    let Some(first) = bands.first() else {
        return Err(Error::Algorithm("embeddings: stack has no bands".into()));
    };
    let (rows, cols) = first.shape();
    for b in bands.iter().skip(1) {
        if b.shape() != (rows, cols) {
            return Err(Error::SizeMismatch {
                er: rows,
                ec: cols,
                ar: b.rows(),
                ac: b.cols(),
            });
        }
    }
    Ok((rows, cols))
}

/// The vector at `(row, col)`, `None` when any band is non-finite there
/// or the cell is outside the stack.
pub fn vector_at(bands: &[&Raster<f64>], row: usize, col: usize) -> Option<Vec<f64>> {
    let (rows, cols) = bands.first()?.shape();
    if row >= rows || col >= cols {
        return None;
    }
    let v: Vec<f64> = bands.iter().map(|b| b.data()[[row, col]]).collect();
    v.iter().all(|x| x.is_finite()).then_some(v)
}

/// Mean vector over `cells` (invalid cells skipped). Errors when no cell
/// is valid.
pub fn mean_vector(bands: &[&Raster<f64>], cells: &[(usize, usize)]) -> Result<Vec<f64>> {
    check_stack(bands)?;
    let mut acc = vec![0.0f64; bands.len()];
    let mut n = 0usize;
    for &(r, c) in cells {
        if let Some(v) = vector_at(bands, r, c) {
            for (a, x) in acc.iter_mut().zip(&v) {
                *a += x;
            }
            n += 1;
        }
    }
    if n == 0 {
        return Err(Error::Algorithm(
            "embeddings: no valid cell among the reference cells".into(),
        ));
    }
    for a in &mut acc {
        *a /= n as f64;
    }
    Ok(acc)
}

/// Per-cell similarity of the stack to `reference` (one value per band).
pub fn similarity(
    bands: &[&Raster<f64>],
    reference: &[f64],
    params: SimilarityParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = check_stack(bands)?;
    if reference.len() != bands.len() {
        return Err(Error::Algorithm(format!(
            "embeddings: reference has {} values, stack has {} bands",
            reference.len(),
            bands.len()
        )));
    }
    if !reference.iter().all(|x| x.is_finite()) {
        return Err(Error::Algorithm(
            "embeddings: reference vector has non-finite values".into(),
        ));
    }
    let ref_norm = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
    if params.metric == SimilarityMetric::Cosine && ref_norm == 0.0 {
        return Err(Error::Algorithm(
            "embeddings: reference vector is zero, cosine undefined".into(),
        ));
    }
    let metric = params.metric;
    let dq = params.dequantize;
    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|r| {
            let mut row = vec![f64::NAN; cols];
            let mut v = vec![0.0f64; bands.len()];
            for (c, out) in row.iter_mut().enumerate() {
                let mut ok = true;
                for (k, b) in bands.iter().enumerate() {
                    let x = b.data()[[r, c]];
                    if !x.is_finite() {
                        ok = false;
                        break;
                    }
                    v[k] = dq.apply(x);
                }
                if !ok {
                    continue;
                }
                *out = match metric {
                    SimilarityMetric::Cosine => {
                        let dot: f64 = v.iter().zip(reference).map(|(a, b)| a * b).sum();
                        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                        if n == 0.0 {
                            f64::NAN
                        } else {
                            dot / (n * ref_norm)
                        }
                    }
                    SimilarityMetric::Dot => v.iter().zip(reference).map(|(a, b)| a * b).sum(),
                    SimilarityMetric::Euclidean => v
                        .iter()
                        .zip(reference)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt(),
                };
            }
            row
        })
        .collect();
    let mut out = bands[0].with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() = ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(out)
}

/// L2 norm of the vector at each cell (values as stored).
pub fn norm(bands: &[&Raster<f64>]) -> Result<Raster<f64>> {
    norm_with(bands, Dequantize::None)
}

/// L2 norm of the vector at each cell after `dequantize`.
pub fn norm_with(bands: &[&Raster<f64>], dequantize: Dequantize) -> Result<Raster<f64>> {
    let (rows, cols) = check_stack(bands)?;
    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|r| {
            let mut row = vec![f64::NAN; cols];
            for (c, out) in row.iter_mut().enumerate() {
                let mut s = 0.0;
                let mut ok = true;
                for b in bands {
                    let x = b.data()[[r, c]];
                    if !x.is_finite() {
                        ok = false;
                        break;
                    }
                    let x = dequantize.apply(x);
                    s += x * x;
                }
                if ok {
                    *out = s.sqrt();
                }
            }
            row
        })
        .collect();
    let mut out = bands[0].with_same_meta::<f64>(rows, cols);
    out.set_nodata(Some(f64::NAN));
    *out.data_mut() = ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| Error::Other(e.to_string()))?;
    Ok(out)
}

/// A fitted PCA: band means and the leading principal axes. Fit once on
/// a sample of the field, project anywhere (another tile, another year of
/// the same product) so colours mean the same thing everywhere.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PcaModel {
    /// Per-band mean subtracted before projection.
    pub mean: Vec<f64>,
    /// Principal axes, one `n_bands`-long unit vector per component,
    /// ordered by decreasing eigenvalue.
    pub components: Vec<Vec<f64>>,
    /// Eigenvalue (variance) of each kept component.
    pub eigenvalues: Vec<f64>,
    /// Fraction of total variance of each kept component.
    pub variance_explained: Vec<f64>,
    /// Number of vectors the fit used.
    pub n_samples: usize,
    /// Applied to stored values at fit time and again at projection, so a
    /// model always sees the code it was fitted on.
    #[serde(default)]
    pub dequantize: Dequantize,
}

impl PcaModel {
    /// Fit on the valid vectors of `bands`, keeping `n_components` axes.
    /// `max_samples` caps the vectors used (taken on a regular stride, so
    /// the sample is deterministic and spatially spread); `None` uses all.
    pub fn fit(
        bands: &[&Raster<f64>],
        n_components: usize,
        max_samples: Option<usize>,
    ) -> Result<Self> {
        Self::fit_with(bands, n_components, max_samples, Dequantize::None)
    }

    /// [`PcaModel::fit`] on de-quantised values; the model remembers the
    /// code and applies it in [`PcaModel::project`].
    #[allow(clippy::needless_range_loop)] // matrix index loops read as the maths
    pub fn fit_with(
        bands: &[&Raster<f64>],
        n_components: usize,
        max_samples: Option<usize>,
        dequantize: Dequantize,
    ) -> Result<Self> {
        let (rows, cols) = check_stack(bands)?;
        let d = bands.len();
        if n_components == 0 || n_components > d {
            return Err(Error::Algorithm(format!(
                "embeddings: n_components must be in 1..={d}, got {n_components}"
            )));
        }
        // Stride so that ≈ max_samples cells are visited.
        let total = rows * cols;
        let stride = match max_samples {
            Some(m) if m > 0 && total > m => total.div_ceil(m),
            _ => 1,
        };
        let mut mean = vec![0.0f64; d];
        let mut vectors: Vec<Vec<f64>> = Vec::new();
        let mut idx = 0usize;
        while idx < total {
            let (r, c) = (idx / cols, idx % cols);
            if let Some(mut v) = vector_at(bands, r, c) {
                v.iter_mut().for_each(|x| *x = dequantize.apply(*x));
                for (m, x) in mean.iter_mut().zip(&v) {
                    *m += x;
                }
                vectors.push(v);
            }
            idx += stride;
        }
        let n = vectors.len();
        if n < 2 {
            return Err(Error::Algorithm(
                "embeddings: PCA needs at least two valid vectors".into(),
            ));
        }
        for m in &mut mean {
            *m /= n as f64;
        }
        // Covariance (d × d), accumulated in a fixed order: deterministic.
        let mut cov = vec![vec![0.0f64; d]; d];
        for v in &vectors {
            for i in 0..d {
                let vi = v[i] - mean[i];
                for j in i..d {
                    cov[i][j] += vi * (v[j] - mean[j]);
                }
            }
        }
        let denom = (n - 1) as f64;
        for i in 0..d {
            for j in i..d {
                cov[i][j] /= denom;
                cov[j][i] = cov[i][j];
            }
        }
        let (eigenvalues, eigenvectors) = jacobi_eigen_symmetric(cov);
        let total_var: f64 = eigenvalues.iter().sum();
        let mut order: Vec<usize> = (0..d).collect();
        order.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        let mut components = Vec::with_capacity(n_components);
        let mut kept_values = Vec::with_capacity(n_components);
        for &k in order.iter().take(n_components) {
            let mut axis: Vec<f64> = (0..d).map(|i| eigenvectors[i][k]).collect();
            // Sign convention: largest-magnitude loading positive, so the
            // axes (and the colours) are the same on every fit.
            if let Some(&m) = axis
                .iter()
                .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                && m < 0.0
            {
                axis.iter_mut().for_each(|x| *x = -*x);
            }
            components.push(axis);
            kept_values.push(eigenvalues[k]);
        }
        let variance_explained = kept_values
            .iter()
            .map(|e| if total_var > 0.0 { e / total_var } else { 0.0 })
            .collect();
        Ok(Self {
            mean,
            components,
            eigenvalues: kept_values,
            variance_explained,
            n_samples: n,
            dequantize,
        })
    }

    /// Number of bands the model expects.
    pub fn n_bands(&self) -> usize {
        self.mean.len()
    }

    /// Project the stack: one raster per kept component (scores).
    pub fn project(&self, bands: &[&Raster<f64>]) -> Result<Vec<Raster<f64>>> {
        let (rows, cols) = check_stack(bands)?;
        if bands.len() != self.n_bands() {
            return Err(Error::Algorithm(format!(
                "embeddings: model fitted on {} bands, stack has {}",
                self.n_bands(),
                bands.len()
            )));
        }
        let k = self.components.len();
        let dq = self.dequantize;
        let scores: Vec<Vec<f64>> = (0..rows)
            .into_par_iter()
            .flat_map(|r| {
                let mut row = vec![f64::NAN; cols * k];
                let mut v = vec![0.0f64; bands.len()];
                for c in 0..cols {
                    let mut ok = true;
                    for (i, b) in bands.iter().enumerate() {
                        let x = b.data()[[r, c]];
                        if !x.is_finite() {
                            ok = false;
                            break;
                        }
                        v[i] = dq.apply(x) - self.mean[i];
                    }
                    if !ok {
                        continue;
                    }
                    for (j, axis) in self.components.iter().enumerate() {
                        row[c * k + j] = v.iter().zip(axis).map(|(a, b)| a * b).sum();
                    }
                }
                vec![row]
            })
            .collect();
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            let mut data = Vec::with_capacity(rows * cols);
            for row in &scores {
                data.extend((0..cols).map(|c| row[c * k + j]));
            }
            let mut r = bands[0].with_same_meta::<f64>(rows, cols);
            r.set_nodata(Some(f64::NAN));
            *r.data_mut() = ndarray::Array2::from_shape_vec((rows, cols), data)
                .map_err(|e| Error::Other(e.to_string()))?;
            out.push(r);
        }
        Ok(out)
    }

    /// Serialise (for caching next to a source, or shipping to a client).
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("PcaModel serialises")
    }

    /// Parse a model written by [`PcaModel::to_json`].
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| Error::Other(format!("PcaModel: {e}")))
    }
}

/// Eigen-decomposition of a symmetric matrix by cyclic Jacobi rotations.
/// Returns `(eigenvalues, eigenvectors)` with eigenvectors as columns
/// (`v[i][k]` = component `i` of eigenvector `k`). Deterministic.
#[allow(clippy::needless_range_loop)] // matrix index loops read as the maths
fn jacobi_eigen_symmetric(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0f64; n]; n];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let vkp = row[p];
                    let vkq = row[q];
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eigenvalues = (0..n).map(|i| a[i][i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    /// 4-band field: two "classes" (rows < 5 vs ≥ 5) with distinct
    /// directions plus a nodata cell.
    fn stack() -> Vec<Raster<f64>> {
        let (rows, cols) = (10, 8);
        let dirs = [[1.0, 0.0, 0.5, 0.2], [0.0, 1.0, -0.5, 0.3]];
        let mut bands: Vec<Raster<f64>> = (0..4).map(|_| Raster::<f64>::new(rows, cols)).collect();
        for r in 0..rows {
            for c in 0..cols {
                let d = if r < 5 { dirs[0] } else { dirs[1] };
                let jitter = 0.01 * ((r * cols + c) % 7) as f64;
                for (k, b) in bands.iter_mut().enumerate() {
                    b.data_mut()[[r, c]] = 3.0 * d[k] + jitter * (k as f64 + 1.0);
                }
            }
        }
        bands[2].data_mut()[[0, 0]] = f64::NAN;
        for b in &mut bands {
            b.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));
        }
        bands
    }

    #[test]
    fn cosine_similarity_separates_the_classes_and_is_scale_free() {
        let bands = stack();
        let refs: Vec<&Raster<f64>> = bands.iter().collect();
        let reference = vector_at(&refs, 2, 3).unwrap();
        let sim = similarity(&refs, &reference, SimilarityParams::default()).unwrap();
        assert!(sim.data()[[0, 0]].is_nan(), "nodata propagates");
        assert!(sim.data()[[2, 3]] > 0.9999);
        assert!(sim.data()[[1, 5]] > 0.99);
        assert!(
            sim.data()[[8, 5]] < 0.6,
            "other class: {}",
            sim.data()[[8, 5]]
        );
        // Scaling every band by 127 (int8 quantisation) changes nothing.
        let scaled: Vec<Raster<f64>> = bands
            .iter()
            .map(|b| {
                let mut s = b.clone();
                s.data_mut().mapv_inplace(|v| v * 127.0);
                s
            })
            .collect();
        let srefs: Vec<&Raster<f64>> = scaled.iter().collect();
        let sim2 = similarity(&srefs, &reference, SimilarityParams::default()).unwrap();
        for (a, b) in sim.data().iter().zip(sim2.data()) {
            if a.is_finite() {
                assert!((a - b).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn dot_and_euclidean_and_norm() {
        let bands = stack();
        let refs: Vec<&Raster<f64>> = bands.iter().collect();
        let reference = vector_at(&refs, 7, 1).unwrap();
        let p = SimilarityParams {
            metric: SimilarityMetric::Euclidean,
            ..Default::default()
        };
        let d = similarity(&refs, &reference, p).unwrap();
        assert_eq!(d.data()[[7, 1]], 0.0);
        assert!(d.data()[[1, 1]] > 1.0);
        let p = SimilarityParams {
            metric: SimilarityMetric::Dot,
            ..Default::default()
        };
        let dot = similarity(&refs, &reference, p).unwrap();
        let n = norm(&refs).unwrap();
        let expect = n.data()[[7, 1]].powi(2);
        assert!((dot.data()[[7, 1]] - expect).abs() < 1e-9);
        assert!(mean_vector(&refs, &[(0, 0)]).is_err());
        let m = mean_vector(&refs, &[(0, 0), (7, 1), (7, 2)]).unwrap();
        assert_eq!(m.len(), 4);
    }

    #[test]
    fn pca_captures_the_two_directions_and_round_trips() {
        let bands = stack();
        let refs: Vec<&Raster<f64>> = bands.iter().collect();
        let model = PcaModel::fit(&refs, 2, None).unwrap();
        assert_eq!(model.n_bands(), 4);
        assert_eq!(model.n_samples, 79);
        assert!(
            model.variance_explained[0] > 0.9,
            "{:?}",
            model.variance_explained
        );
        assert!(model.eigenvalues[0] >= model.eigenvalues[1]);
        for axis in &model.components {
            let n: f64 = axis.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-9);
        }
        let scores = model.project(&refs).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0].data()[[0, 0]].is_nan());
        // Two classes → PC1 separates them with opposite signs.
        let a = scores[0].data()[[2, 2]];
        let b = scores[0].data()[[8, 2]];
        assert!(a * b < 0.0, "pc1 {a} vs {b}");
        // Subsampled fit gives the same axes up to the sample.
        let sub = PcaModel::fit(&refs, 2, Some(20)).unwrap();
        assert!(
            sub.n_samples <= 21 && sub.n_samples >= 15,
            "{}",
            sub.n_samples
        );
        // serde_json's default float parsing may be 1 ULP off: compare
        // approximately.
        let back = PcaModel::from_json(&model.to_json()).unwrap();
        assert_eq!(back.n_samples, model.n_samples);
        for (a, b) in back.mean.iter().zip(&model.mean) {
            assert!((a - b).abs() < 1e-12);
        }
        for (ca, cb) in back.components.iter().zip(&model.components) {
            for (a, b) in ca.iter().zip(cb) {
                assert!((a - b).abs() < 1e-12);
            }
        }
        assert!(PcaModel::fit(&refs, 5, None).is_err());
    }

    #[test]
    fn dequantize_alphaearth_restores_unit_norm_and_parses() {
        assert_eq!(
            Dequantize::parse("alphaearth"),
            Some(Dequantize::AlphaEarth)
        );
        assert_eq!(Dequantize::parse("none"), Some(Dequantize::None));
        assert_eq!(
            Dequantize::parse("linear:0.5,1"),
            Some(Dequantize::Linear {
                scale: 0.5,
                offset: 1.0
            })
        );
        assert_eq!(Dequantize::parse("linear:x"), None);
        assert_eq!(Dequantize::AlphaEarth.apply(127.5), 1.0);
        assert_eq!(Dequantize::AlphaEarth.apply(-127.5), -1.0);
        assert_eq!(Dequantize::AlphaEarth.apply(0.0), 0.0);
        assert!((Dequantize::AlphaEarth.apply(63.75) - 0.25).abs() < 1e-12);
        // A stored code whose dequantised vector is unit-norm: codes
        // ±90.156 in two dims → 0.5 + 0.5.
        let c = 127.5 * 0.5f64.sqrt();
        let mut a = Raster::<f64>::new(1, 1);
        let mut b = Raster::<f64>::new(1, 1);
        a.data_mut()[[0, 0]] = c;
        b.data_mut()[[0, 0]] = -c;
        let mut st = vec![a, b];
        Dequantize::AlphaEarth.apply_stack(&mut st);
        let refs: Vec<&Raster<f64>> = st.iter().collect();
        let n = norm(&refs).unwrap();
        assert!((n.data()[[0, 0]] - 1.0f64.sqrt() * (0.5f64.powi(2) * 2.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn jacobi_matches_known_eigenvalues() {
        let (vals, vecs) = jacobi_eigen_symmetric(vec![vec![2.0, 1.0], vec![1.0, 2.0]]);
        let mut v = vals.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((v[0] - 1.0).abs() < 1e-12 && (v[1] - 3.0).abs() < 1e-12);
        // Columns are orthonormal.
        let dot: f64 = (0..2).map(|i| vecs[i][0] * vecs[i][1]).sum();
        assert!(dot.abs() < 1e-12);
    }
}
