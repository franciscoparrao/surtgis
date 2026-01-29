//! # SurtGis Parallel
//!
//! Parallel processing strategies for geospatial algorithms.
//!
//! This crate provides:
//! - Tiled processing for large rasters
//! - Row-parallel processing using Rayon
//! - Streaming for memory-efficient processing

pub mod strategy;
pub mod tiled;

pub use strategy::{ParallelStrategy, ProcessingMode};
pub use tiled::{Tile, TileIterator, TiledProcessor};
