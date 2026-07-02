//! Validation helpers for multi-raster operations.
//!
//! Every algorithm that combines two or more rasters cell-by-cell
//! (spectral indices, composites, change detection, TWI/SPI, …)
//! implicitly assumes its inputs live on the *same georeferenced
//! grid*. If a caller passes, say, a NIR band in UTM and a Red band
//! in EPSG:4326 that happen to have equal dimensions, a shape-only
//! check accepts them and the output is silent garbage.
//!
//! This module centralises the checks:
//!
//! - [`check_same_shape`] — dimensions only (`(rows, cols)`).
//! - [`check_same_crs`] — EPSG codes, with deliberately lenient
//!   semantics (see below).
//! - [`check_aligned`] — the full contract: shape + geotransform +
//!   CRS. This is what multi-input entry points should call.
//!
//! # CRS comparison semantics
//!
//! CRSs are compared **only when both rasters carry a CRS with a
//! known EPSG code** (`crs().and_then(CRS::epsg)` is `Some` on both
//! sides). If either raster has no CRS, or a CRS expressed only as
//! WKT/PROJ without an EPSG code, the CRS check is *skipped* rather
//! than failed. Rationale: [`crate::crs::CRS::is_equivalent`]
//! compares raw strings, so an EPSG:32719 raster and the same CRS
//! spelled as WKT would be reported as different (a false negative).
//! Until real CRS normalisation exists, only the unambiguous
//! EPSG-vs-EPSG case is enforced; heterogeneous or absent
//! representations are given the benefit of the doubt. The
//! geotransform comparison still catches most practical
//! misalignments in that case.

use crate::error::{Error, Result};
use crate::raster::{Raster, RasterCell};

/// Relative tolerance used when comparing geotransform coefficients.
///
/// Coefficients `a`, `b` are considered equal when
/// `|a - b| <= REL_TOL * max(|a|, |b|, 1.0)`. The `1.0` floor makes
/// the comparison absolute for near-zero coefficients (rotation
/// terms), where a relative test would be meaninglessly strict.
const REL_TOL: f64 = 1e-9;

/// Check that all rasters have the same `(rows, cols)` shape.
///
/// The first raster is the reference; the first raster that
/// disagrees produces an [`Error::ShapeMismatch`] whose `context`
/// names the offending input position. Empty and single-element
/// slices trivially pass.
///
/// # Errors
///
/// [`Error::ShapeMismatch`] if any raster differs in shape from the
/// first one.
pub fn check_same_shape<T: RasterCell>(rasters: &[&Raster<T>]) -> Result<()> {
    let Some(first) = rasters.first() else {
        return Ok(());
    };
    let expected = first.shape();
    for (i, r) in rasters.iter().enumerate().skip(1) {
        let got = r.shape();
        if got != expected {
            return Err(Error::ShapeMismatch {
                expected,
                got,
                context: format!("input raster {i} vs raster 0"),
            });
        }
    }
    Ok(())
}

/// Check that all rasters have compatible CRSs, comparing EPSG codes only.
///
/// Follows the lenient semantics documented at the
/// [module level](self): two rasters conflict only when **both**
/// expose an EPSG code and the codes differ. Rasters without a CRS,
/// or with a CRS that has no EPSG code (WKT/PROJ only), never fail
/// this check.
///
/// # Errors
///
/// [`Error::Misaligned`] naming the two conflicting EPSG codes.
pub fn check_same_crs<T: RasterCell>(rasters: &[&Raster<T>]) -> Result<()> {
    let Some(first) = rasters.first() else {
        return Ok(());
    };
    let epsg0 = first.crs().and_then(|c| c.epsg());
    for (i, r) in rasters.iter().enumerate().skip(1) {
        if let (Some(a), Some(b)) = (epsg0, r.crs().and_then(|c| c.epsg())) {
            if a != b {
                return Err(Error::Misaligned {
                    reason: format!(
                        "CRS mismatch: input raster {i} is EPSG:{b} but raster 0 is EPSG:{a}"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Check that all rasters live on the same georeferenced grid.
///
/// Verifies, in order:
///
/// 1. **Shape** — via [`check_same_shape`].
/// 2. **Geotransform** — all 6 affine coefficients must match the
///    first raster's within a relative tolerance of `1e-9`
///    (absolute for near-zero coefficients).
/// 3. **CRS** — via [`check_same_crs`]; note the deliberately
///    lenient EPSG-only semantics documented at the
///    [module level](self).
///
/// This is the check every multi-raster entry point should run
/// before combining inputs cell-by-cell. Empty and single-element
/// slices trivially pass.
///
/// # Errors
///
/// - [`Error::ShapeMismatch`] if shapes differ.
/// - [`Error::Misaligned`] if geotransforms or EPSG codes differ.
pub fn check_aligned<T: RasterCell>(rasters: &[&Raster<T>]) -> Result<()> {
    check_same_shape(rasters)?;

    let Some(first) = rasters.first() else {
        return Ok(());
    };
    let gt0 = first.transform().to_gdal();
    for (i, r) in rasters.iter().enumerate().skip(1) {
        let gt = r.transform().to_gdal();
        for (a, b) in gt0.iter().zip(gt.iter()) {
            let tol = REL_TOL * a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > tol {
                return Err(Error::Misaligned {
                    reason: format!(
                        "geotransform mismatch: input raster {i} has {gt:?} \
                         but raster 0 has {gt0:?} (GDAL coefficient order)"
                    ),
                });
            }
        }
    }

    check_same_crs(rasters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crs::CRS;
    use crate::raster::GeoTransform;

    fn raster(rows: usize, cols: usize, gt: GeoTransform, crs: Option<CRS>) -> Raster<f64> {
        let mut r: Raster<f64> = Raster::new(rows, cols);
        r.set_transform(gt);
        r.set_crs(crs);
        r
    }

    #[test]
    fn same_shape_and_transform_passes() {
        let gt = GeoTransform::new(500_000.0, 6_200_000.0, 30.0, -30.0);
        let a = raster(10, 20, gt, Some(CRS::from_epsg(32719)));
        let b = raster(10, 20, gt, Some(CRS::from_epsg(32719)));
        assert!(check_aligned(&[&a, &b]).is_ok());
        assert!(check_same_shape(&[&a, &b]).is_ok());
    }

    #[test]
    fn default_constructed_rasters_are_aligned() {
        // Raster::new() uses the same default GeoTransform and no CRS,
        // so freshly built rasters of equal shape must pass.
        let a: Raster<f64> = Raster::new(5, 5);
        let b: Raster<f64> = Raster::new(5, 5);
        assert!(check_aligned(&[&a, &b]).is_ok());
    }

    #[test]
    fn different_shape_fails() {
        let a: Raster<f64> = Raster::new(5, 5);
        let b: Raster<f64> = Raster::new(3, 5);
        let err = check_aligned(&[&a, &b]).unwrap_err();
        match err {
            Error::ShapeMismatch { expected, got, .. } => {
                assert_eq!(expected, (5, 5));
                assert_eq!(got, (3, 5));
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn shifted_transform_fails() {
        let gt_a = GeoTransform::new(0.0, 100.0, 10.0, -10.0);
        let gt_b = GeoTransform::new(10.0, 100.0, 10.0, -10.0); // shifted 1 px east
        let a = raster(10, 10, gt_a, None);
        let b = raster(10, 10, gt_b, None);
        assert!(matches!(
            check_aligned(&[&a, &b]),
            Err(Error::Misaligned { .. })
        ));
    }

    #[test]
    fn tiny_relative_transform_noise_passes() {
        // Float noise well below 1e-9 relative must not trip the check.
        let gt_a = GeoTransform::new(500_000.0, 6_200_000.0, 30.0, -30.0);
        let gt_b = GeoTransform::new(500_000.000_001, 6_200_000.0, 30.0, -30.0);
        let a = raster(4, 4, gt_a, None);
        let b = raster(4, 4, gt_b, None);
        assert!(check_aligned(&[&a, &b]).is_ok());
    }

    #[test]
    fn different_epsg_fails() {
        let gt = GeoTransform::new(0.0, 10.0, 1.0, -1.0);
        let a = raster(10, 10, gt, Some(CRS::from_epsg(32719)));
        let b = raster(10, 10, gt, Some(CRS::from_epsg(4326)));
        assert!(matches!(
            check_aligned(&[&a, &b]),
            Err(Error::Misaligned { .. })
        ));
        assert!(matches!(
            check_same_crs(&[&a, &b]),
            Err(Error::Misaligned { .. })
        ));
    }

    #[test]
    fn missing_crs_on_one_side_passes() {
        // Documented lenient semantics: no CRS (or no EPSG) on either
        // side means the CRS check is skipped, not failed.
        let gt = GeoTransform::new(0.0, 10.0, 1.0, -1.0);
        let a = raster(10, 10, gt, Some(CRS::from_epsg(32719)));
        let b = raster(10, 10, gt, None);
        assert!(check_aligned(&[&a, &b]).is_ok());
        assert!(check_aligned(&[&b, &a]).is_ok());
    }

    #[test]
    fn wkt_without_epsg_does_not_conflict_with_epsg() {
        // EPSG-vs-WKT must not produce a false negative.
        let gt = GeoTransform::new(0.0, 10.0, 1.0, -1.0);
        let a = raster(10, 10, gt, Some(CRS::from_epsg(32719)));
        let b = raster(
            10,
            10,
            gt,
            Some(CRS::from_wkt("PROJCS[\"WGS 84 / UTM 19S\"]")),
        );
        assert!(check_aligned(&[&a, &b]).is_ok());
    }

    #[test]
    fn empty_and_single_input_pass() {
        let a: Raster<f64> = Raster::new(3, 3);
        assert!(check_aligned::<f64>(&[]).is_ok());
        assert!(check_aligned(&[&a]).is_ok());
    }

    #[test]
    fn third_raster_reported_in_context() {
        let a: Raster<f64> = Raster::new(5, 5);
        let b: Raster<f64> = Raster::new(5, 5);
        let c: Raster<f64> = Raster::new(4, 5);
        match check_same_shape(&[&a, &b, &c]).unwrap_err() {
            Error::ShapeMismatch { context, .. } => assert!(context.contains("raster 2")),
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }
}
