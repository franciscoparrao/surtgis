//! Imagery analysis algorithms
//!
//! Algorithms for remote sensing and spectral analysis:
//! - Spectral indices: NDVI, NDWI, SAVI, EVI, MNDWI, NBR, BSI
//! - Normalized difference: generic two-band index
//! - Band math: arbitrary raster algebra expressions
//! - Reclassify: value-based reclassification

mod band_math;
mod change_detection;
mod index_builder;
mod indices;
mod reclassify;

pub use band_math::{band_math, band_math_binary, BandMathOp};
pub use change_detection::{
    raster_difference, change_vector_analysis, RasterDiffParams,
    CHANGE_DECREASE, CHANGE_NO_CHANGE, CHANGE_INCREASE,
};
pub use index_builder::index_builder;
pub use indices::{
    bsi, evi, evi2, gndvi, mndwi, msavi, nbr, ndbi, ndmi, ndre, ndsi, ndvi, ndwi, ngrdi,
    normalized_difference, reci, savi, EviParams, SaviParams, SpectralIndex,
};
pub use reclassify::{reclassify, ReclassEntry, ReclassifyParams};
