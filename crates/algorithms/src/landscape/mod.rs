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

pub mod connected_components;
mod class_metrics;
mod diversity;
pub mod patch_metrics;

pub use class_metrics::{class_metrics, landscape_metrics, ClassMetrics, LandscapeMetrics};
pub use connected_components::{label_patches, Connectivity};
pub use diversity::{patch_density, shannon_diversity, simpson_diversity, DiversityParams};
pub use patch_metrics::{patch_metrics, patches_to_csv, PatchStats};
