//! Classification and machine learning algorithms for raster data
//!
//! Unsupervised and supervised classification methods:
//! - **PCA**: Principal Component Analysis (dimensionality reduction)
//! - **K-means**: Unsupervised clustering
//! - **ISODATA**: Iterative self-organizing clustering with split/merge
//! - **Minimum Distance**: Supervised classification (nearest centroid)
//! - **Maximum Likelihood**: Supervised classification (Gaussian MLE)

mod isodata;
mod kmeans;
pub(crate) mod pca;
mod supervised;

pub use isodata::{IsodataParams, isodata};
pub use kmeans::{KmeansParams, kmeans_raster};
pub use pca::{PcaParams, PcaResult, pca};
pub use supervised::{
    ClassSignature, maximum_likelihood, minimum_distance, signatures_from_training,
};
