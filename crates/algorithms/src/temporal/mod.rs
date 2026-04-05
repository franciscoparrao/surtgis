//! Multi-temporal analysis algorithms
//!
//! Per-pixel time-series analysis for satellite imagery:
//! - **Statistics**: mean, std, min, max, count, percentiles across time
//! - **Trend**: linear regression, Mann-Kendall, Sen's slope
//! - **Anomaly**: z-score deviation from a reference period
//! - **Phenology**: vegetation phenology metrics (SOS, EOS, peak, amplitude)

mod statistics;
mod trend;
mod anomaly;
mod phenology;

pub use statistics::{
    temporal_mean, temporal_std, temporal_min, temporal_max, temporal_count,
    temporal_percentile, temporal_stats, TemporalStats,
};
pub use trend::{
    linear_trend, mann_kendall, sens_slope,
    LinearTrendResult, MannKendallResult,
};
pub use anomaly::{temporal_anomaly, AnomalyMethod};
pub use phenology::{
    vegetation_phenology, PhenologyParams, PhenologyResult,
};
