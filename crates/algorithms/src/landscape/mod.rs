//! Landscape ecology algorithms
//!
//! Moving-window landscape metrics for categorical raster data:
//! - **Shannon Diversity**: Information entropy index
//! - **Simpson Diversity**: Probability of interspecific encounter
//! - **Patch Density**: Number of distinct patches per unit area
//!
//! Global landscape metrics (scalar output per patch/class/landscape):
//! - **Connected Components**: Label contiguous patches (Union-Find)
//! - **Patch Metrics**: PARA, FRAC, area, perimeter per patch
//! - **Class Metrics**: AI, COHESION per class
//! - **Landscape Metrics**: SHDI, SIDI global

mod class_metrics;
pub mod connected_components;
mod diversity;
pub mod patch_metrics;

pub use class_metrics::{ClassMetrics, LandscapeMetrics, class_metrics, landscape_metrics};
pub use connected_components::{Connectivity, label_patches};
pub use diversity::{DiversityParams, patch_density, shannon_diversity, simpson_diversity};
pub use patch_metrics::{PatchStats, patch_metrics, patches_to_csv};
