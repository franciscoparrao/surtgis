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
                // Exact comparison, as in GDAL/rasterio: the sentinel is
                // written verbatim to the file, so if it survives the binary
                // round-trip exact equality is correct. A tolerance-based
                // match corrupts small valid values (e.g. NDVI ≈ 0 with
                // nodata = 0.0) and misses large sentinels off by 1 ULP.
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
                // Exact comparison — see the float impl for rationale.
                match nodata {
                    Some(nd) => self.re == nd.re && self.im == nd.im,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: with nodata = 0.0, small valid values (NDVI ≈ 0,
    /// low reflectances) must NOT be classified as nodata. The old
    /// epsilon-based match treated |v| < 1.19e-5 (f32) as nodata and
    /// the reader then rewrote them to NaN — irreversible corruption.
    #[test]
    fn test_float_nodata_small_valid_values_near_zero_sentinel() {
        let nd = Some(0.0f32);
        assert!(0.0f32.is_nodata(nd));
        assert!(!1e-6f32.is_nodata(nd));
        assert!(!(-1e-6f32).is_nodata(nd));
        assert!(!1e-7f64.is_nodata(Some(0.0f64)));
    }

    #[test]
    fn test_float_nodata_exact_sentinel_matches() {
        let nd = -9999.0f64;
        assert!((-9999.0f64).is_nodata(Some(nd)));
        assert!(!(-9998.9999f64).is_nodata(Some(nd)));
        // GDAL's Float32 default sentinel
        let big = -3.4e38f32;
        assert!(big.is_nodata(Some(big)));
    }

    #[test]
    fn test_float_nan_always_nodata() {
        assert!(f64::NAN.is_nodata(None));
        assert!(f64::NAN.is_nodata(Some(-9999.0)));
        assert!(f32::NAN.is_nodata(None));
    }

    #[test]
    fn test_int_nodata_exact() {
        assert!(255u8.is_nodata(Some(255)));
        assert!(!0u8.is_nodata(Some(255)));
        assert!(!5i32.is_nodata(None));
    }
}
