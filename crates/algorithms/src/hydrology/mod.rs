//! Hydrological analysis algorithms
//!
//! Algorithms for hydrological modeling from Digital Elevation Models:
//! - Fill sinks: remove depressions for continuous flow
//! - Flow direction: D8 single flow direction
//! - Flow accumulation: upstream contributing area
//! - Watershed: basin delineation from pour points

mod fill_sinks;
mod flow_accumulation;
mod flow_direction;
mod watershed;

pub use fill_sinks::{fill_sinks, FillSinks, FillSinksParams};
pub use flow_accumulation::{flow_accumulation, FlowAccumulation};
pub use flow_direction::{flow_direction, FlowDirection};
pub use watershed::{watershed, Watershed, WatershedParams};
