//! Texture and feature extraction algorithms
//!
//! - **GLCM**: Gray-Level Co-occurrence Matrix (Haralick textures)
//! - **LBP**: Local Binary Patterns (Ojala 2002, standard + riu2)
//! - **Sobel**: Edge detection using Sobel operator
//! - **Laplacian**: Second-derivative edge enhancement

mod edge;
mod glcm;
mod lbp;

pub use edge::{laplacian, sobel_edge};
pub use glcm::{GlcmParams, GlcmTexture, haralick_glcm, haralick_glcm_multi};
pub use lbp::{LbpParams, LbpVariant, lbp};
