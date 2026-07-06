//! Type-erased raster wrapper for dtype-preserving GeoTIFF reads.
//!
//! Historically, reading a GeoTIFF of unknown type ahead of time meant
//! calling `read_geotiff::<f64, _>(path, band)` and letting every pixel
//! type collapse to `f64` — correct, but up to 4x more memory than a
//! `u16` DEM actually needs for callers that only want to inspect or
//! pass the data through (metadata reporting, clipping, mosaicking)
//! rather than run floating-point algorithms on it.
//!
//! [`AnyRaster`] lets the I/O boundary defer that choice:
//! `read_geotiff_any` (in [`crate::io`]) returns whichever variant
//! matches the file's native sample format, and callers that do want
//! `f64` can still get it explicitly via [`AnyRaster::to_f64`].
//!
//! This does not change how the `algorithms` crate computes — its 138
//! call sites concretely typed `Raster<f64>` are untouched. It only
//! removes the forced upcast at the point where a file is opened.

use crate::crs::CRS;
use crate::raster::{GeoTransform, Raster, RasterElement};

/// Native pixel data type of a raster, as detected from the source file.
///
/// One variant per [`AnyRaster`] case. Source types without an exact
/// match here (a TIFF's `i8`, `u64`, `i64` or half-precision `f16`
/// samples) are widened losslessly to the narrowest compatible variant
/// at the I/O boundary — see `read_geotiff_any` / `decode_geotiff_any`
/// in `crate::io`'s native backend for the mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 16-bit signed integer
    I16,
    /// 32-bit unsigned integer
    U32,
    /// 32-bit signed integer
    I32,
    /// 32-bit IEEE float
    F32,
    /// 64-bit IEEE float
    F64,
}

impl DataType {
    /// Short, stable name for the dtype (e.g. for CLI `info` output).
    pub fn name(&self) -> &'static str {
        match self {
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::I16 => "i16",
            DataType::U32 => "u32",
            DataType::I32 => "i32",
            DataType::F32 => "f32",
            DataType::F64 => "f64",
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A raster whose cell type ("dtype") is only known at runtime.
///
/// Produced by `read_geotiff_any` (`crate::io`) so a caller can inspect
/// a file's native pixel type — and hold it at that width — without
/// having picked a concrete `T` ahead of time. Algorithms still operate
/// on concrete `Raster<T>`; use [`AnyRaster::to_f64`] (or match out the
/// variant you need) at the boundary where you commit to a type.
#[derive(Debug, Clone)]
pub enum AnyRaster {
    /// 8-bit unsigned integer raster
    U8(Raster<u8>),
    /// 16-bit unsigned integer raster
    U16(Raster<u16>),
    /// 16-bit signed integer raster
    I16(Raster<i16>),
    /// 32-bit unsigned integer raster
    U32(Raster<u32>),
    /// 32-bit signed integer raster
    I32(Raster<i32>),
    /// 32-bit float raster
    F32(Raster<f32>),
    /// 64-bit float raster
    F64(Raster<f64>),
}

/// Dispatch a body over the inner `Raster<T>` of any [`AnyRaster`]
/// variant without hand-writing the 7-arm match at every call site.
///
/// `$any` may be an owned `AnyRaster`, `&AnyRaster` or `&mut AnyRaster`
/// expression — match ergonomics binds `$r` to `Raster<T>`,
/// `&Raster<T>` or `&mut Raster<T>` accordingly, exactly as writing the
/// match by hand would.
///
/// ```ignore
/// use surtgis_core::dispatch_any;
/// use surtgis_core::raster::AnyRaster;
///
/// fn shape_of(any: &AnyRaster) -> (usize, usize) {
///     dispatch_any!(any, r => r.shape())
/// }
/// ```
#[macro_export]
macro_rules! dispatch_any {
    ($any:expr, $r:ident => $body:expr) => {
        match $any {
            $crate::raster::AnyRaster::U8($r) => $body,
            $crate::raster::AnyRaster::U16($r) => $body,
            $crate::raster::AnyRaster::I16($r) => $body,
            $crate::raster::AnyRaster::U32($r) => $body,
            $crate::raster::AnyRaster::I32($r) => $body,
            $crate::raster::AnyRaster::F32($r) => $body,
            $crate::raster::AnyRaster::F64($r) => $body,
        }
    };
}

impl AnyRaster {
    /// Which variant this is — the file's native pixel type (or its
    /// lossless widening; see `read_geotiff_any` in `crate::io`).
    pub fn dtype(&self) -> DataType {
        match self {
            AnyRaster::U8(_) => DataType::U8,
            AnyRaster::U16(_) => DataType::U16,
            AnyRaster::I16(_) => DataType::I16,
            AnyRaster::U32(_) => DataType::U32,
            AnyRaster::I32(_) => DataType::I32,
            AnyRaster::F32(_) => DataType::F32,
            AnyRaster::F64(_) => DataType::F64,
        }
    }

    /// Dimensions as (rows, cols) — delegates to the inner `Raster<T>`.
    pub fn shape(&self) -> (usize, usize) {
        dispatch_any!(self, r => r.shape())
    }

    /// Coordinate reference system, if the source carried one.
    pub fn crs(&self) -> Option<&CRS> {
        dispatch_any!(self, r => r.crs())
    }

    /// Affine geotransform of the raster.
    pub fn transform(&self) -> &GeoTransform {
        dispatch_any!(self, r => r.transform())
    }

    /// Consume `self` and convert the inner raster to `Raster<f64>`,
    /// preserving transform, CRS and nodata.
    ///
    /// This is the explicit opt-in for callers that want the old
    /// always-`f64` behavior — e.g. handing the result to `algorithms`,
    /// which is concretely `Raster<f64>`-typed throughout.
    pub fn to_f64(self) -> Raster<f64> {
        dispatch_any!(self, r => raster_to_f64(r))
    }
}

/// Per-element cast of any `RasterElement` raster to `Raster<f64>`,
/// carrying transform/CRS/nodata across.
///
/// Every `AnyRaster` variant's element type fits `f64` exactly
/// (`u8`/`u16`/`i16`/`u32`/`i32` all need far fewer than its 52
/// mantissa bits; `f32` is a strict subset of `f64`'s range and
/// precision), so the `unwrap_or` fallback below is unreachable for
/// these types today — it only guards against a future variant being
/// added to the enum without a matching numeric guarantee.
fn raster_to_f64<T: RasterElement>(r: Raster<T>) -> Raster<f64> {
    let (rows, cols) = r.shape();
    let transform = *r.transform();
    let crs = r.crs().cloned();
    let nodata = r.nodata().and_then(num_traits::cast::<T, f64>);
    let data: Vec<f64> = r
        .data()
        .iter()
        .map(|&v| num_traits::cast::<T, f64>(v).unwrap_or(f64::NAN))
        .collect();
    let mut out = Raster::from_vec(data, rows, cols).expect("shape preserved by construction");
    out.set_transform(transform);
    out.set_crs(crs);
    out.set_nodata(nodata);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::RasterCell;

    fn sample_u8() -> Raster<u8> {
        let mut r = Raster::from_vec(vec![0u8, 1, 2, 255], 2, 2).unwrap();
        r.set_transform(GeoTransform::new(10.0, 20.0, 1.0, -1.0));
        r
    }

    #[test]
    fn dtype_matches_variant() {
        assert_eq!(AnyRaster::U8(sample_u8()).dtype(), DataType::U8);
        assert_eq!(
            AnyRaster::F64(Raster::<f64>::new(1, 1)).dtype(),
            DataType::F64
        );
        assert_eq!(DataType::U16.name(), "u16");
        assert_eq!(DataType::F32.to_string(), "f32");
    }

    #[test]
    fn shape_crs_transform_delegate_to_inner_raster() {
        let any = AnyRaster::U8(sample_u8());
        assert_eq!(any.shape(), (2, 2));
        assert_eq!(any.transform().origin_x, 10.0);
        assert!(any.crs().is_none());
    }

    #[test]
    fn to_f64_preserves_values_and_metadata_u8() {
        let any = AnyRaster::U8(sample_u8());
        let f = any.to_f64();
        assert_eq!(f.get(0, 0).unwrap(), 0.0);
        assert_eq!(f.get(0, 1).unwrap(), 1.0);
        assert_eq!(f.get(1, 0).unwrap(), 2.0);
        assert_eq!(f.get(1, 1).unwrap(), 255.0);
        assert_eq!(f.transform().origin_x, 10.0);
    }

    #[test]
    fn to_f64_preserves_values_for_every_variant() {
        // u32/i32 exercise the "no precision loss even near the type's
        // extremes" case that a naive `as f32` cast would fail.
        let u16r = AnyRaster::U16(Raster::from_vec(vec![0u16, 1, 2, u16::MAX], 2, 2).unwrap());
        let i16r = AnyRaster::I16(Raster::from_vec(vec![i16::MIN, -1, 0, i16::MAX], 2, 2).unwrap());
        let u32r = AnyRaster::U32(Raster::from_vec(vec![0u32, 1, 2, u32::MAX], 2, 2).unwrap());
        let i32r = AnyRaster::I32(Raster::from_vec(vec![i32::MIN, -1, 0, i32::MAX], 2, 2).unwrap());
        let f32r = AnyRaster::F32(Raster::from_vec(vec![1.5f32, -2.25, 0.0, 3.75], 2, 2).unwrap());
        let f64r = AnyRaster::F64(Raster::from_vec(vec![1.5f64, -2.25, 0.0, 3.75], 2, 2).unwrap());

        assert_eq!(u16r.to_f64().data().iter().copied().collect::<Vec<_>>(), [
            0.0,
            1.0,
            2.0,
            u16::MAX as f64
        ]);
        assert_eq!(
            i16r.to_f64().data().iter().copied().collect::<Vec<_>>(),
            [i16::MIN as f64, -1.0, 0.0, i16::MAX as f64]
        );
        assert_eq!(u32r.to_f64().data().iter().copied().collect::<Vec<_>>(), [
            0.0,
            1.0,
            2.0,
            u32::MAX as f64
        ]);
        assert_eq!(
            i32r.to_f64().data().iter().copied().collect::<Vec<_>>(),
            [i32::MIN as f64, -1.0, 0.0, i32::MAX as f64]
        );
        assert_eq!(
            f32r.to_f64().data().iter().copied().collect::<Vec<_>>(),
            [1.5, -2.25, 0.0, 3.75]
        );
        assert_eq!(
            f64r.to_f64().data().iter().copied().collect::<Vec<_>>(),
            [1.5, -2.25, 0.0, 3.75]
        );
    }

    #[test]
    fn dispatch_any_macro_counts_valid_cells_across_all_seven_variants() {
        // Exercises the macro with a simple "count valid cells" use
        // case over every AnyRaster variant.
        let count_valid = |any: &AnyRaster| -> usize {
            dispatch_any!(any, r => {
                let nodata = r.nodata();
                r.data().iter().filter(|v| !v.is_nodata(nodata)).count()
            })
        };

        let u8r = AnyRaster::U8(Raster::from_vec(vec![1u8, 2, 3, 4], 2, 2).unwrap());
        let u16r = AnyRaster::U16(Raster::from_vec(vec![1u16, 2, 3, 4], 2, 2).unwrap());
        let i16r = AnyRaster::I16(Raster::from_vec(vec![1i16, 2, 3, 4], 2, 2).unwrap());
        let u32r = AnyRaster::U32(Raster::from_vec(vec![1u32, 2, 3, 4], 2, 2).unwrap());
        let i32r = AnyRaster::I32(Raster::from_vec(vec![1i32, 2, 3, 4], 2, 2).unwrap());
        let f32r = AnyRaster::F32(Raster::from_vec(vec![1f32, 2.0, 3.0, 4.0], 2, 2).unwrap());
        let f64r = AnyRaster::F64(Raster::from_vec(vec![1f64, 2.0, 3.0, 4.0], 2, 2).unwrap());

        for any in [&u8r, &u16r, &i16r, &u32r, &i32r, &f32r, &f64r] {
            assert_eq!(count_valid(any), 4, "dtype {}", any.dtype());
        }

        // And with an actual nodata cell present.
        let mut with_nodata = Raster::from_vec(vec![1u8, 2, 3, 4], 2, 2).unwrap();
        with_nodata.set_nodata(Some(3));
        let any = AnyRaster::U8(with_nodata);
        assert_eq!(count_valid(&any), 3);
    }

    #[test]
    fn dispatch_any_macro_mutates_through_mut_ref() {
        // `$body` is inserted verbatim into all 7 match arms (this is a
        // textual macro, not a generic function), so it must type-check
        // for every concrete `T` — `num_traits::cast` from a `f64`
        // literal does that uniformly across int and float variants,
        // where a bare `Some(1.0)` would only compile for the float ones.
        let mut any = AnyRaster::F32(Raster::from_vec(vec![1f32, 2.0, 3.0, 4.0], 2, 2).unwrap());
        dispatch_any!(&mut any, r => {
            r.set_nodata(num_traits::cast::<f64, _>(1.0));
        });
        assert_eq!(any.dtype(), DataType::F32);
        if let AnyRaster::F32(r) = &any {
            assert_eq!(r.nodata(), Some(1.0));
        } else {
            panic!("variant changed unexpectedly");
        }
    }
}
