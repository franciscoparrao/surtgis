//! Hydrological analysis algorithms
//!
//! Algorithms for hydrological modeling from Digital Elevation Models:
//! - Fill sinks: remove depressions for continuous flow
//! - Priority-Flood: optimal O(n log n) depression filling (Barnes 2014)
//! - Flow direction: D8 single flow direction
//! - Flow accumulation: upstream contributing area
//! - Watershed: basin delineation from pour points
//! - HAND: Height Above Nearest Drainage (flood mapping)
//! - Stream network: drainage network extraction

mod advanced;
mod basin_morphometry;
mod breach;
mod drainage_density;
pub(crate) mod fill_sinks;
pub(crate) mod flow_accumulation;
pub(crate) mod flow_direction;
mod flow_direction_dinf;
mod flow_direction_mfd;
mod flow_direction_mfd_adaptive;
mod flow_direction_tfga;
mod hand;
mod hypsometric;
mod nested_depressions;
mod priority_flood;
mod sediment_connectivity;
mod stream_network;
mod watershed;
mod watershed_parallel;

pub use advanced::{
    FloodSimParams, IsobasinParams, flood_fill_simulation, flow_path_length, isobasins,
    strahler_order,
};
pub use basin_morphometry::{BasinMorphometry, basin_morphometry};
pub use breach::{BreachParams, breach_depressions};
pub use drainage_density::{DrainageDensityParams, drainage_density};
pub use fill_sinks::{FillSinks, FillSinksParams, fill_sinks};
pub use flow_accumulation::{FlowAccumulation, flow_accumulation};
pub use flow_direction::{FlowDirection, flow_direction};
pub use flow_direction_dinf::{
    DinfResult, flow_accumulation_dinf, flow_dinf, flow_direction_dinf,
};
pub use flow_direction_mfd::{MfdParams, flow_accumulation_mfd};
pub use flow_direction_mfd_adaptive::{AdaptiveMfdParams, flow_accumulation_mfd_adaptive};
pub use flow_direction_tfga::{TfgaParams, flow_accumulation_tfga};
pub use hand::{HandParams, hand};
pub use hypsometric::hypsometric_integral;
pub use nested_depressions::{
    Depression, NestedDepressionParams, NestedDepressionResult, nested_depressions,
};
pub use priority_flood::{PriorityFlood, PriorityFloodParams, priority_flood, priority_flood_flat};
pub use sediment_connectivity::{SedimentConnectivityParams, sediment_connectivity};
pub use stream_network::{StreamNetworkParams, stream_network};
pub use watershed::{Watershed, WatershedParams, watershed};
pub use watershed_parallel::{ParallelWatershedParams, watershed_parallel};
