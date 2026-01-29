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

pub mod crs;
pub mod error;
pub mod io;
pub mod raster;
pub mod vector;

pub use crs::CRS;
pub use error::{Error, Result};
pub use raster::{GeoTransform, Raster, RasterElement};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::crs::CRS;
    pub use crate::error::{Error, Result};
    pub use crate::raster::{GeoTransform, Raster, RasterElement};
    pub use crate::Algorithm;
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
    fn execute(&self, input: Self::Input, params: Self::Params) -> std::result::Result<Self::Output, Self::Error>;

    /// Execute with default parameters
    fn execute_default(&self, input: Self::Input) -> std::result::Result<Self::Output, Self::Error> {
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
