//! Raster element traits for generic cell values.
//!
//! Two layers:
//!
//! - [`RasterCell`] — the minimum a type needs to live in a
//!   [`Raster`](super::Raster): copyable, zeroable, with a nodata
//!   convention. This is the bound on the `Raster<T>` type itself.
//!   Notably it does **not** require ordering, so non-ordered cell
//!   types like `Complex<f32>` (feature `complex`) qualify.
//! - [`RasterElement`] — `RasterCell` plus ordering and numeric
//!   casts (`PartialOrd + NumCast`, min/max, f64 conversion). This is
//!   what statistics, resampling, I/O and the algorithm crates bound
//!   on. Every primitive numeric type implements it.
//!
//! Code generic over rasters should bound on `RasterElement` unless
//! it genuinely works for unordered cells.

use num_traits::{NumCast, Zero};
use std::fmt::Debug;

/// Minimum contract for a raster cell value.
///
/// `Zero` provides the fill value for [`Raster::new`](super::Raster::new);
/// the nodata methods define the type's missing-data convention
/// (NaN-based for floats and complex, sentinel-based for integers).
pub trait RasterCell: Copy + Clone + Debug + PartialEq + Zero + Send + Sync + 'static {
    /// Default no-data value for this type
    fn default_nodata() -> Self;

    /// Check if this value represents no-data
    fn is_nodata(&self, nodata: Option<Self>) -> bool;
}

/// Trait for ordered numeric raster values.
///
/// This is the bound used by statistics, resampling and I/O — every
/// operation that needs comparisons or lossless conversion through
/// `f64`.
pub trait RasterElement: RasterCell + PartialOrd + NumCast {
    /// Minimum value representable by this type
    fn min_value() -> Self;

    /// Maximum value representable by this type
    fn max_value() -> Self;

    /// Whether this type is a floating point type
    fn is_float() -> bool;

    /// Convert self to f64
    fn to_f64(self) -> Option<f64> {
        NumCast::from(self)
    }
}

macro_rules! impl_raster_element_int {
    ($t:ty) => {
        impl RasterCell for $t {
            fn default_nodata() -> Self {
                <$t>::MIN
            }

            fn is_nodata(&self, nodata: Option<Self>) -> bool {
                match nodata {
                    Some(nd) => *self == nd,
                    None => false,
                }
            }
        }

        impl RasterElement for $t {
            fn min_value() -> Self {
                <$t>::MIN
            }

            fn max_value() -> Self {
                <$t>::MAX
            }

            fn is_float() -> bool {
                false
            }
        }
    };
}

macro_rules! impl_raster_element_float {
    ($t:ty) => {
        impl RasterCell for $t {
            fn default_nodata() -> Self {
                <$t>::NAN
            }

            fn is_nodata(&self, nodata: Option<Self>) -> bool {
                if self.is_nan() {
                    return true;
                }
                match nodata {
                    Some(nd) => (self - nd).abs() < <$t>::EPSILON * 100.0,
                    None => false,
                }
            }
        }

        impl RasterElement for $t {
            fn min_value() -> Self {
                <$t>::MIN
            }

            fn max_value() -> Self {
                <$t>::MAX
            }

            fn is_float() -> bool {
                true
            }
        }
    };
}

impl_raster_element_int!(i8);
impl_raster_element_int!(i16);
impl_raster_element_int!(i32);
impl_raster_element_int!(i64);
impl_raster_element_int!(u8);
impl_raster_element_int!(u16);
impl_raster_element_int!(u32);
impl_raster_element_int!(u64);
impl_raster_element_float!(f32);
impl_raster_element_float!(f64);

/// Complex cells for interferometric phase rasters (feature `complex`).
///
/// Nodata convention: a cell is nodata when **both** parts are NaN
/// (the [`RasterCell::default_nodata`] value), or when it equals the
/// explicit sentinel.
#[cfg(feature = "complex")]
macro_rules! impl_raster_cell_complex {
    ($t:ty) => {
        impl RasterCell for num_complex::Complex<$t> {
            fn default_nodata() -> Self {
                num_complex::Complex::new(<$t>::NAN, <$t>::NAN)
            }

            fn is_nodata(&self, nodata: Option<Self>) -> bool {
                if self.re.is_nan() && self.im.is_nan() {
                    return true;
                }
                match nodata {
                    Some(nd) => {
                        (self.re - nd.re).abs() < <$t>::EPSILON * 100.0
                            && (self.im - nd.im).abs() < <$t>::EPSILON * 100.0
                    }
                    None => false,
                }
            }
        }
    };
}

#[cfg(feature = "complex")]
impl_raster_cell_complex!(f32);
#[cfg(feature = "complex")]
impl_raster_cell_complex!(f64);
