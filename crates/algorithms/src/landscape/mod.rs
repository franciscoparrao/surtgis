//! Landscape ecology algorithms
//!
//! Moving-window landscape metrics for categorical raster data:
//! - **Shannon Diversity**: Information entropy index
//! - **Simpson Diversity**: Probability of interspecific encounter
//! - **Patch Density**: Number of distinct patches per unit area

mod diversity;

pub use diversity::{
    shannon_diversity, simpson_diversity, patch_density,
    DiversityParams,
};
