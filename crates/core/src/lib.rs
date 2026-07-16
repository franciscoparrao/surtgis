//! # SurtGis Core
//!
//! Core types, traits and I/O for the SurtGis geospatial library.
//!
//! This crate provides:
//! - `Raster<T>`: Generic raster grid type
//! - `GeoTransform`: Affine transformation for georeferencing
//! - `CRS`: Coordinate Reference System handling
//! - I/O for common geospatial formats
//! - Streaming ([`StripProcessor`]) and tiling ([`TileGrid`]) for
//!   bounded-memory and windowed processing
//!
//! ## Stability
//!
//! As of v0.15.0, `surtgis-core` is the foundation of the SurtGIS
//! engine ecosystem and its public API is treated as **stable under
//! SemVer**: within a minor series the API only grows, and breaking
//! changes are pre-announced one release ahead in the CHANGELOG under
//! a `Breaking` heading. Sibling crates should depend on
//! `surtgis-core = "0.15"` alone; the CLI and GUI crates carry no
//! stability promise.
//!
//! ### `ndarray` / `geo-types` re-exports
//!
//! [`Raster`]'s data-access methods return `ndarray` types
//! ([`ndarray::Array2`], `ArrayView2`, ...) and [`vector::Feature`]
//! carries a [`geo_types::Geometry`] — both are part of this crate's
//! public API surface, not internal implementation details. Depend on
//! the versions re-exported here (`surtgis_core::ndarray`,
//! `surtgis_core::geo`, `surtgis_core::geo_types`) rather than adding
//! your own `ndarray`/`geo`/`geo-types` dependency, so your code always
//! matches the exact version this crate was built against — Rust's type
//! system treats two different semver-major versions of the same crate
//! as unrelated types even when named identically. A major-version bump
//! of `ndarray`, `geo`, or `geo-types` is a **major** (breaking) bump of
//! `surtgis-core` too, under the same pre-announced `Breaking` policy
//! above.

// The foundation crate's public API is part of the stability contract, so every
// public item must be documented — enforced at build time.
#![deny(missing_docs)]

pub mod crs;
pub mod cube;
pub mod error;
pub mod io;
pub mod mosaic;
pub mod raster;
pub mod resample;
pub mod streaming;
pub mod tiling;
pub mod vector;

/// Re-exported so downstream code can depend on the exact `geo` version
/// used by algorithm-crate vector operations built on this crate.
/// See the crate-level "Stability" section.
pub use geo;
/// Re-exported so downstream code can depend on the exact `geo-types`
/// version this crate's public API ([`vector::Feature`]) uses. See the
/// crate-level "Stability" section.
pub use geo_types;
/// Re-exported so downstream code can depend on the exact `ndarray`
/// version this crate's public API (`Raster::data`, `view`, ...) uses.
/// See the crate-level "Stability" section.
pub use ndarray;

pub use crs::CRS;
pub use cube::{Cube, CubeChunk, CubeSource};
pub use error::{Error, Result};
pub use mosaic::{MosaicOptions, mosaic};
pub use raster::{AnyRaster, DataType, GeoTransform, Raster, RasterCell, RasterElement};
pub use resample::{ResampleMethod, resample_to_grid};
pub use streaming::{GeoRowContext, StripProcessor, WindowAlgorithm};
pub use tiling::{Tile, TileGrid};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::crs::CRS;
    pub use crate::dispatch_any;
    pub use crate::error::{Error, Result};
    pub use crate::raster::{
        AnyRaster, DataType, GeoTransform, Raster, RasterCell, RasterElement, check_aligned,
        check_same_crs, check_same_shape,
    };
}
