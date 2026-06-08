//! Pansharpening: fuse a high-resolution panchromatic image with a
//! lower-resolution multispectral image to produce a high-resolution
//! multispectral output.
//!
//! All algorithms here expect the multispectral bands **already
//! upsampled** to pan resolution. The caller is responsible for the
//! upsampling step (use `vector::resample` or any GIS-standard
//! bilinear/cubic upsampler upstream).
//!
//! Common contract: each function takes `pan: &Raster<f64>` and
//! `ms: &[&Raster<f64>]` (each MS band at pan resolution), and
//! returns `Vec<Raster<f64>>` — one sharpened band per input MS
//! band, in the same order.
//!
//! Algorithms:
//! - **Brovey** (Gillespie et al. 1987) — simplest, fastest;
//!   per-band ratio against a synthetic pan from MS. Can saturate
//!   colours.
//! - **PCA** (Chavez et al. 1991) — replaces the first principal
//!   component of MS with histogram-matched pan. Works for
//!   arbitrary band count. Reuses the Jacobi eigensolver from
//!   `classification::pca`.
//! - **Gram-Schmidt** (Laben & Brower 2000, US patent 6,011,875,
//!   expired 2018) — Gram-Schmidt orthogonalisation of MS against
//!   a synthetic low-res pan, replace the first GS component with
//!   real pan, invert.

mod brovey;
mod gram_schmidt;
mod pca_pansharpen;

pub use brovey::brovey;
pub use gram_schmidt::gram_schmidt;
pub use pca_pansharpen::pca_pansharpen;
