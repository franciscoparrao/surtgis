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

pub use closing::{Closing, ClosingParams, closing};
pub use dilate::{Dilate, DilateParams, dilate};
pub use element::StructuringElement;
pub use erode::{Erode, ErodeParams, erode};
pub use gradient::{Gradient, GradientParams, gradient};
pub use opening::{Opening, OpeningParams, opening};
pub use tophat::{BlackHat, BlackHatParams, TopHat, TopHatParams, black_hat, top_hat};
