//! Imagery analysis algorithms
//!
//! Algorithms for remote sensing and spectral analysis:
//! - Spectral indices: NDVI, NDWI, SAVI, EVI, MNDWI, NBR, BSI
//! - Normalized difference: generic two-band index
//! - Band math: arbitrary raster algebra expressions
//! - Reclassify: value-based reclassification

mod band_math;
mod burn_severity;
mod change_detection;
mod cloud_mask;
mod composite;
mod index_builder;
mod indices;
mod reclassify;

pub use band_math::{BandMathOp, band_math, band_math_binary};
pub use burn_severity::{burn_severity_classify, dnbr};
pub use change_detection::{
    CHANGE_DECREASE, CHANGE_INCREASE, CHANGE_NO_CHANGE, RasterDiffParams, change_vector_analysis,
    raster_difference,
};
pub use cloud_mask::{
    CloudMaskStrategy, LandsatQaMask, NoCloudMask, S2SclMask, SCL_VALID_DEFAULT, cloud_mask_scl,
};
pub use composite::median_composite;
pub use index_builder::index_builder;
pub use indices::{
    EviParams, SaviParams, SpectralIndex, bsi, evi, evi2, gndvi, mndwi, msavi, nbr, ndbi, ndmi,
    ndre, ndsi, ndvi, ndwi, ngrdi, normalized_difference, reci, savi,
};
pub use reclassify::{ReclassEntry, ReclassifyParams, reclassify};
