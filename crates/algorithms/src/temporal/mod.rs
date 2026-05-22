//! Multi-temporal analysis algorithms
//!
//! Per-pixel time-series analysis for satellite imagery:
//! - **Statistics**: mean, std, min, max, count, percentiles across time
//! - **Trend**: linear regression, Mann-Kendall, Sen's slope
//! - **Anomaly**: z-score deviation from a reference period
//! - **Phenology**: vegetation phenology metrics (SOS, EOS, peak, amplitude)

mod anomaly;
mod phenology;
mod statistics;
mod trend;

pub use anomaly::{AnomalyMethod, temporal_anomaly};
pub use phenology::{PhenologyParams, PhenologyResult, vegetation_phenology};
pub use statistics::{
    TemporalStats, temporal_count, temporal_max, temporal_mean, temporal_min, temporal_percentile,
    temporal_stats, temporal_std,
};
pub use trend::{LinearTrendResult, MannKendallResult, linear_trend, mann_kendall, sens_slope};
