//! # SurtGis Colormap
//!
//! Color mapping and raster-to-RGBA rendering for SurtGis.
//!
//! Provides 8 predefined color schemes ported from the web demo, plus a generic
//! multi-stop interpolation engine. The main entry point is [`raster_to_rgba`]
//! which converts a `Raster<T>` into an RGBA pixel buffer suitable for GPU textures.
//!
//! ## Usage
//!
//! ```ignore
//! use surtgis_colormap::{ColorScheme, ColormapParams, raster_to_rgba};
//!
//! let params = ColormapParams::new(ColorScheme::Terrain);
//! let rgba = raster_to_rgba(&raster, &params);
//! ```

mod scheme;
mod render;

pub use scheme::{ColorScheme, ColorStop, Rgb, evaluate};
pub use render::{raster_to_rgba, auto_params, ColormapParams};
