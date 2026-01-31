//! Statistical analysis algorithms for raster data
//!
//! - **focal**: Moving window (focal) statistics
//! - **zonal**: Statistics by zones
//! - **autocorrelation**: Spatial autocorrelation (Moran's I, Getis-Ord Gi*)

pub mod focal;
pub mod zonal;
pub mod autocorrelation;

pub use focal::{focal_statistics, FocalStatistic, FocalParams};
pub use zonal::{zonal_statistics, ZonalResult, ZonalStatistic};
pub use autocorrelation::{global_morans_i, local_getis_ord, MoransIResult, GetisOrdResult};
