//! Imagery analysis algorithms
//!
//! Algorithms for remote sensing and spectral analysis:
//! - Spectral indices: NDVI, NDWI, SAVI, EVI, MNDWI, NBR, BSI
//! - Normalized difference: generic two-band index
//! - Band math: arbitrary raster algebra expressions
//! - Reclassify: value-based reclassification

mod band_math;
mod indices;
mod reclassify;

pub use band_math::{band_math, band_math_binary, BandMathOp};
pub use indices::{
    bsi, evi, mndwi, nbr, ndvi, ndwi, normalized_difference, savi,
    EviParams, SaviParams, SpectralIndex,
};
pub use reclassify::{reclassify, ReclassEntry, ReclassifyParams};
