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

pub use closing::{closing, Closing, ClosingParams};
pub use dilate::{dilate, Dilate, DilateParams};
pub use element::StructuringElement;
pub use erode::{erode, Erode, ErodeParams};
pub use gradient::{gradient, Gradient, GradientParams};
pub use opening::{opening, Opening, OpeningParams};
pub use tophat::{black_hat, top_hat, BlackHat, BlackHatParams, TopHat, TopHatParams};
