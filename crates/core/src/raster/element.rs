//! Raster element trait for generic cell values

use num_traits::{NumCast, Zero};
use std::fmt::Debug;

/// Trait for types that can be stored in a raster cell.
///
/// This trait bounds the types that can be used as raster values,
/// ensuring they support necessary numeric operations.
pub trait RasterElement:
    Copy + Clone + Debug + PartialOrd + PartialEq + NumCast + Zero + Send + Sync + 'static
{
    /// Minimum value representable by this type
    fn min_value() -> Self;

    /// Maximum value representable by this type
    fn max_value() -> Self;

    /// Default no-data value for this type
    fn default_nodata() -> Self;

    /// Check if this value represents no-data
    fn is_nodata(&self, nodata: Option<Self>) -> bool;

    /// Whether this type is a floating point type
    fn is_float() -> bool;

    /// Convert self to f64
    fn to_f64(self) -> Option<f64> {
        NumCast::from(self)
    }
}

macro_rules! impl_raster_element_int {
    ($t:ty) => {
        impl RasterElement for $t {
            fn min_value() -> Self {
                <$t>::MIN
            }

            fn max_value() -> Self {
                <$t>::MAX
            }

            fn default_nodata() -> Self {
                <$t>::MIN
            }

            fn is_nodata(&self, nodata: Option<Self>) -> bool {
                match nodata {
                    Some(nd) => *self == nd,
                    None => false,
                }
            }

            fn is_float() -> bool {
                false
            }
        }
    };
}

macro_rules! impl_raster_element_float {
    ($t:ty) => {
        impl RasterElement for $t {
            fn min_value() -> Self {
                <$t>::MIN
            }

            fn max_value() -> Self {
                <$t>::MAX
            }

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
