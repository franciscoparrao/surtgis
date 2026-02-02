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

mod breach;
pub(crate) mod fill_sinks;
pub(crate) mod flow_accumulation;
pub(crate) mod flow_direction;
mod flow_direction_dinf;
mod flow_direction_mfd;
mod flow_direction_mfd_adaptive;
mod flow_direction_tfga;
mod hand;
mod nested_depressions;
mod priority_flood;
mod stream_network;
mod watershed;
mod watershed_parallel;

pub use breach::{breach_depressions, BreachParams};
pub use fill_sinks::{fill_sinks, FillSinks, FillSinksParams};
pub use flow_accumulation::{flow_accumulation, FlowAccumulation};
pub use flow_direction::{flow_direction, FlowDirection};
pub use flow_direction_dinf::{flow_direction_dinf, flow_dinf, DinfResult};
pub use flow_direction_mfd::{flow_accumulation_mfd, MfdParams};
pub use hand::{hand, HandParams};
pub use priority_flood::{priority_flood, priority_flood_flat, PriorityFlood, PriorityFloodParams};
pub use stream_network::{stream_network, StreamNetworkParams};
pub use watershed::{watershed, Watershed, WatershedParams};
pub use flow_direction_mfd_adaptive::{flow_accumulation_mfd_adaptive, AdaptiveMfdParams};
pub use flow_direction_tfga::{flow_accumulation_tfga, TfgaParams};
pub use nested_depressions::{nested_depressions, NestedDepressionParams, NestedDepressionResult, Depression};
pub use watershed_parallel::{watershed_parallel, ParallelWatershedParams};
