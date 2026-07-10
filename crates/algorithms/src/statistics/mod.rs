//! Statistical analysis algorithms for raster data
//!
//! - **focal**: Moving window (focal) statistics
//! - **zonal**: Statistics by zones
//! - **autocorrelation**: Spatial autocorrelation (Moran's I, Getis-Ord Gi*)

pub mod autocorrelation;
pub mod focal;
// `pub(crate)` (not private): the O(1)/O(r) building blocks in this module
// (SummedAreaTable, hgw_square_2d, sliding_extreme_1d, ...) are consumed
// directly by terrain/morphology fast paths outside `statistics`, in
// addition to `focal.rs` in this module.
pub(crate) mod focal_fast;
pub mod zonal;

pub use autocorrelation::{GetisOrdResult, MoransIResult, global_morans_i, local_getis_ord};
pub use focal::{FocalParams, FocalStatistic, focal_statistics};
pub use zonal::{ZonalResult, ZonalStatistic, zonal_statistics, zonal_statistics_raster};
