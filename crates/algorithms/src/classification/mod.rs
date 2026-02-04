//! Classification and machine learning algorithms for raster data
//!
//! Unsupervised and supervised classification methods:
//! - **PCA**: Principal Component Analysis (dimensionality reduction)
//! - **K-means**: Unsupervised clustering
//! - **ISODATA**: Iterative self-organizing clustering with split/merge
//! - **Minimum Distance**: Supervised classification (nearest centroid)
//! - **Maximum Likelihood**: Supervised classification (Gaussian MLE)

mod pca;
mod kmeans;
mod isodata;
mod supervised;

pub use pca::{pca, PcaParams, PcaResult};
pub use kmeans::{kmeans_raster, KmeansParams};
pub use isodata::{isodata, IsodataParams};
pub use supervised::{minimum_distance, maximum_likelihood, signatures_from_training, ClassSignature};
