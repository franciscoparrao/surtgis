//! Statistical analysis algorithms for raster data
//!
//! - **focal**: Moving window (focal) statistics
//! - **zonal**: Statistics by zones
//! - **autocorrelation**: Spatial autocorrelation (Moran's I, Getis-Ord Gi*)

pub mod autocorrelation;
pub mod focal;
pub mod zonal;

pub use autocorrelation::{GetisOrdResult, MoransIResult, global_morans_i, local_getis_ord};
pub use focal::{FocalParams, FocalStatistic, focal_statistics};
pub use zonal::{ZonalResult, ZonalStatistic, zonal_statistics};
