//! # SurtGis Core
//!
//! Core types, traits and I/O for the SurtGis geospatial library.
//!
//! This crate provides:
//! - `Raster<T>`: Generic raster grid type
//! - `GeoTransform`: Affine transformation for georeferencing
//! - `CRS`: Coordinate Reference System handling
//! - Algorithm traits for consistent API
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

pub use crs::CRS;
pub use cube::{Cube, CubeChunk};
pub use error::{Error, Result};
pub use mosaic::{MosaicOptions, mosaic};
pub use raster::{AnyRaster, DataType, GeoTransform, Raster, RasterCell, RasterElement};
pub use resample::{ResampleMethod, resample_to_grid};
pub use streaming::{GeoRowContext, StripProcessor, WindowAlgorithm};
pub use tiling::{Tile, TileGrid};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::Algorithm;
    pub use crate::crs::CRS;
    pub use crate::error::{Error, Result};
    pub use crate::raster::{
        AnyRaster, DataType, GeoTransform, Raster, RasterCell, RasterElement, check_aligned,
        check_same_crs, check_same_shape,
    };
    pub use crate::dispatch_any;
}

/// Core trait for all algorithms in SurtGis.
///
/// Algorithms are pure functions that transform input data according to parameters.
pub trait Algorithm {
    /// Input type for the algorithm
    type Input;
    /// Output type for the algorithm
    type Output;
    /// Parameters controlling algorithm behavior
    type Params: Default;
    /// Error type for algorithm execution
    type Error: std::error::Error;

    /// Returns the algorithm name
    fn name(&self) -> &'static str;

    /// Returns a description of what the algorithm does
    fn description(&self) -> &'static str;

    /// Execute the algorithm
    fn execute(
        &self,
        input: Self::Input,
        params: Self::Params,
    ) -> std::result::Result<Self::Output, Self::Error>;

    /// Execute with default parameters
    fn execute_default(
        &self,
        input: Self::Input,
    ) -> std::result::Result<Self::Output, Self::Error> {
        self.execute(input, Self::Params::default())
    }
}

/// Marker trait for algorithms that can be parallelized
pub trait ParallelAlgorithm: Algorithm {
    /// Execute in parallel using available cores
    fn execute_parallel(
        &self,
        input: Self::Input,
        params: Self::Params,
    ) -> std::result::Result<Self::Output, Self::Error>;
}
