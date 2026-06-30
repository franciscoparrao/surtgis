//! # SurtGIS Relief
//!
//! Rayshader-style shaded-relief compositing on top of the terrain primitives
//! that already live in `surtgis-algorithms`. This crate is a *compositor*,
//! not a new set of algorithms — see `SPEC_SURTGIS_RELIEF.md` in the
//! repository root for the full design.
//!
//! ## What lives here
//!
//! - [`ray_shade`] — ray-traced cast shadows via
//!   [`surtgis_algorithms::terrain::horizon_angle_map_fast`], with optional
//!   soft-shadow averaging over multiple sun samples.
//! - [`sphere_shade`] — normal-based hillshade as an intensity layer in
//!   `[0, 1]`, wrapping
//!   [`surtgis_algorithms::terrain::hillshade`].
//!
//! ## What is re-exported
//!
//! Pixel-buffer and PNG output live in
//! [`surtgis_colormap`](surtgis_colormap); we re-export the relevant types
//! at the crate root for ergonomics.
//!
//! ```ignore
//! use surtgis_relief::{ray_shade, sphere_shade, RayShadeParams, SunSample, RgbaImage};
//!
//! let shadow_mask = ray_shade(&dem, &RayShadeParams::default())?;
//! let mut img = RgbaImage::from_intensity(&sphere_shade(&dem, Default::default())?);
//! let shadow_layer = RgbaImage::from_intensity(&shadow_mask);
//! img.multiply(&shadow_layer, 0.5)?;
//! img.save_png("relief.png")?;
//! ```

// Part of the engine ecosystem's stable surface: every public item must be documented.
#![deny(missing_docs)]

use thiserror::Error;

mod ambient;
mod compose;
mod ray_shade_impl;
mod shadow_ray;
mod sphere_shade_impl;
mod water;

pub use ambient::ambient_shade;
pub use compose::ReliefBuilder;
pub use ray_shade_impl::{RayShadeParams, SunSample, ray_shade};
pub use shadow_ray::{cast_shadow_ray_mask, horizon_tan_map};
pub use sphere_shade_impl::sphere_shade;
pub use water::{WaterParams, detect_water, water_depth};

// Re-exports from colormap so users only need to depend on relief.
pub use surtgis_colormap::{ColorScheme, ColormapParams, EncodeError, RgbaImage};

#[cfg(not(target_arch = "wasm32"))]
pub use surtgis_colormap::{rgba_to_png_bytes, save_png};

/// Errors produced by the relief crate.
#[derive(Debug, Error)]
pub enum ReliefError {
    /// An underlying terrain algorithm (in `surtgis-algorithms`/`surtgis-core`) failed.
    #[error("algorithm: {0}")]
    Algorithm(#[from] surtgis_core::Error),
    /// Pixel-buffer or PNG encoding failed.
    #[error("encode: {0}")]
    Encode(#[from] EncodeError),
    /// Two inputs had incompatible dimensions; the message names the mismatch.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// A parameter was outside its valid range; the message names the parameter.
    #[error("invalid parameter: {0}")]
    InvalidParam(String),
}

/// `Result` alias scoped to this crate.
pub type Result<T> = std::result::Result<T, ReliefError>;
