//! Mathematical morphology algorithms for raster processing
//!
//! Classical morphological operations for image analysis:
//! - **Erosion**: minimum filter (shrinks bright regions)
//! - **Dilation**: maximum filter (expands bright regions)
//! - **Opening**: erosion then dilation (removes small bright features)
//! - **Closing**: dilation then erosion (fills small dark gaps)
//! - **Gradient**: dilation minus erosion (edge detection)
//! - **Top-hat**: original minus opening (bright feature extraction)
//! - **Black-hat**: closing minus original (dark feature extraction)

mod closing;
mod dilate;
mod element;
mod erode;
mod gradient;
mod opening;
mod tophat;

pub use closing::{ClosingParams, closing};
pub use dilate::{DilateParams, dilate};
pub use element::StructuringElement;
pub use erode::{ErodeParams, erode};
pub use gradient::{GradientParams, gradient};
pub use opening::{OpeningParams, opening};
pub use tophat::{BlackHatParams, TopHatParams, black_hat, top_hat};
