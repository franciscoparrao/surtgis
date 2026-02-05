//! # SurtGis Parallel
//!
//! Parallel processing strategies for geospatial algorithms.
//!
//! This crate provides:
//! - Tiled processing for large rasters
//! - Row-parallel processing using Rayon
//! - Streaming for memory-efficient processing

#[cfg(feature = "parallel")]
pub mod strategy;
#[cfg(feature = "parallel")]
pub mod tiled;

#[cfg(feature = "parallel")]
pub use strategy::{ParallelStrategy, ProcessingMode};
#[cfg(feature = "parallel")]
pub use tiled::{Tile, TileIterator, TiledProcessor};
