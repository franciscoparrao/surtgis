//! Complex-raster helpers for interferometry (feature `complex`).
//!
//! `Raster<Complex<f32>>` works through the [`RasterCell`] bound;
//! these helpers cover the minimal InSAR workflow:
//!
//! - [`complex_from_parts`] / [`complex_to_parts`] — assemble a
//!   complex raster from real/imaginary band rasters and back. This
//!   is also the I/O path: GeoTIFF (and every other writer) handles
//!   the two float bands, so complex rasters persist as `(re, im)`
//!   pairs without a CFloat sample format.
//! - [`magnitude`] / [`phase`] — interferogram amplitude and wrapped
//!   phase (`atan2(im, re)`, in `(-π, π]`) as float rasters.
//!
//! Nodata: a complex cell is nodata when both parts are NaN; the
//! helpers map nodata cells to NaN outputs and vice versa.
//!
//! FFT helpers are deliberately out of scope — spectral processing
//! belongs to insar-rs.

use num_complex::Complex;
use num_traits::Float;

use super::element::{RasterCell, RasterElement};
use super::grid::Raster;
use crate::error::{Error, Result};

fn check_same_grid<T: RasterElement>(re: &Raster<T>, im: &Raster<T>) -> Result<()> {
    if re.shape() != im.shape() {
        return Err(Error::SizeMismatch {
            er: re.rows(),
            ec: re.cols(),
            ar: im.rows(),
            ac: im.cols(),
        });
    }
    if re.transform() != im.transform() {
        return Err(Error::Other(
            "complex: re/im rasters have different transforms".into(),
        ));
    }
    if re.crs() != im.crs() {
        return Err(Error::Other(
            "complex: re/im rasters have different CRS".into(),
        ));
    }
    Ok(())
}

/// Assemble a complex raster from real and imaginary part rasters.
///
/// Both inputs must share grid, transform and CRS. Cells where
/// either part is non-finite become complex nodata (NaN + NaN·i).
pub fn complex_from_parts<T>(re: &Raster<T>, im: &Raster<T>) -> Result<Raster<Complex<T>>>
where
    T: RasterElement + Float,
    Complex<T>: RasterCell,
{
    check_same_grid(re, im)?;
    let (rows, cols) = re.shape();
    let mut out: Raster<Complex<T>> = re.with_same_meta::<Complex<T>>(rows, cols);
    out.set_nodata(Some(Complex::<T>::default_nodata()));
    for row in 0..rows {
        for col in 0..cols {
            let r = unsafe { re.get_unchecked(row, col) };
            let i = unsafe { im.get_unchecked(row, col) };
            let v = if r.is_finite() && i.is_finite() {
                Complex::new(r, i)
            } else {
                Complex::<T>::default_nodata()
            };
            unsafe { out.set_unchecked(row, col, v) };
        }
    }
    Ok(out)
}

/// Split a complex raster into real and imaginary part rasters.
///
/// Nodata cells become NaN in both outputs.
pub fn complex_to_parts<T>(raster: &Raster<Complex<T>>) -> (Raster<T>, Raster<T>)
where
    T: RasterElement + Float,
    Complex<T>: RasterCell,
{
    let (rows, cols) = raster.shape();
    let mut re: Raster<T> = raster.with_same_meta::<T>(rows, cols);
    let mut im: Raster<T> = raster.with_same_meta::<T>(rows, cols);
    re.set_nodata(Some(T::nan()));
    im.set_nodata(Some(T::nan()));
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { raster.get_unchecked(row, col) };
            let (r, i) = if raster.is_nodata(v) {
                (T::nan(), T::nan())
            } else {
                (v.re, v.im)
            };
            unsafe { re.set_unchecked(row, col, r) };
            unsafe { im.set_unchecked(row, col, i) };
        }
    }
    (re, im)
}

/// Per-cell magnitude `|z|` of a complex raster.
pub fn magnitude<T>(raster: &Raster<Complex<T>>) -> Raster<T>
where
    T: RasterElement + Float,
    Complex<T>: RasterCell,
{
    map_to_float(raster, |z| z.norm())
}

/// Per-cell wrapped phase `atan2(im, re)` in `(-π, π]`.
pub fn phase<T>(raster: &Raster<Complex<T>>) -> Raster<T>
where
    T: RasterElement + Float,
    Complex<T>: RasterCell,
{
    map_to_float(raster, |z| z.im.atan2(z.re))
}

fn map_to_float<T, F>(raster: &Raster<Complex<T>>, f: F) -> Raster<T>
where
    T: RasterElement + Float,
    Complex<T>: RasterCell,
    F: Fn(Complex<T>) -> T,
{
    let (rows, cols) = raster.shape();
    let mut out: Raster<T> = raster.with_same_meta::<T>(rows, cols);
    out.set_nodata(Some(T::nan()));
    for row in 0..rows {
        for col in 0..cols {
            let v = unsafe { raster.get_unchecked(row, col) };
            let r = if raster.is_nodata(v) { T::nan() } else { f(v) };
            unsafe { out.set_unchecked(row, col, r) };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster::GeoTransform;

    fn part(rows: usize, cols: usize, scale: f32) -> Raster<f32> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 10.0, -10.0));
        for row in 0..rows {
            for col in 0..cols {
                r.set(row, col, scale * (row * cols + col) as f32).unwrap();
            }
        }
        r
    }

    #[test]
    fn complex_raster_basic_storage() {
        let mut r: Raster<Complex<f32>> = Raster::new(4, 4);
        r.set_transform(GeoTransform::new(0.0, 4.0, 1.0, -1.0));
        r.set(1, 2, Complex::new(3.0, -4.0)).unwrap();
        let v = r.get(1, 2).unwrap();
        assert_eq!(v, Complex::new(3.0, -4.0));
        // Zero fill from Raster::new
        assert_eq!(r.get(0, 0).unwrap(), Complex::new(0.0, 0.0));
        // Nodata convention: both parts NaN
        assert!(Complex::<f32>::default_nodata().is_nodata(None));
        assert!(!Complex::new(f32::NAN, 0.0).is_nodata(None));
    }

    #[test]
    fn parts_roundtrip_with_nodata() {
        let re = part(5, 4, 1.0);
        let mut im = part(5, 4, -0.5);
        im.set(2, 2, f32::NAN).unwrap();

        let z = complex_from_parts(&re, &im).unwrap();
        assert!(z.is_nodata(z.get(2, 2).unwrap()));
        assert_eq!(z.get(1, 1).unwrap(), Complex::new(5.0, -2.5));
        assert_eq!(z.transform(), re.transform());

        let (re2, im2) = complex_to_parts(&z);
        assert_eq!(re2.get(1, 1).unwrap(), 5.0);
        assert_eq!(im2.get(1, 1).unwrap(), -2.5);
        assert!(re2.get(2, 2).unwrap().is_nan());
        assert!(im2.get(2, 2).unwrap().is_nan());
    }

    #[test]
    fn magnitude_and_phase() {
        let mut z: Raster<Complex<f64>> = Raster::new(2, 2);
        z.set_transform(GeoTransform::new(0.0, 2.0, 1.0, -1.0));
        z.set_nodata(Some(Complex::<f64>::default_nodata()));
        z.set(0, 0, Complex::new(3.0, 4.0)).unwrap();
        z.set(0, 1, Complex::new(0.0, 1.0)).unwrap();
        z.set(1, 0, Complex::new(-1.0, 0.0)).unwrap();
        z.set(1, 1, Complex::<f64>::default_nodata()).unwrap();

        let mag = magnitude(&z);
        assert!((mag.get(0, 0).unwrap() - 5.0).abs() < 1e-12);
        assert!(mag.get(1, 1).unwrap().is_nan());

        let ph = phase(&z);
        assert!((ph.get(0, 1).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((ph.get(1, 0).unwrap() - std::f64::consts::PI).abs() < 1e-12);
        assert!(ph.get(1, 1).unwrap().is_nan());
    }

    #[test]
    fn rejects_mismatched_parts() {
        let re = part(5, 4, 1.0);
        let im = part(4, 4, 1.0);
        assert!(complex_from_parts(&re, &im).is_err());

        let mut im_shifted = part(5, 4, 1.0);
        im_shifted.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        assert!(complex_from_parts(&re, &im_shifted).is_err());
    }

    /// Persistence path: complex → (re, im) GeoTIFFs → complex.
    #[test]
    fn geotiff_roundtrip_via_parts() {
        let dir = tempfile::tempdir().unwrap();
        let re = part(6, 5, 1.0);
        let im = part(6, 5, 2.0);
        let z = complex_from_parts(&re, &im).unwrap();

        let (re_out, im_out) = complex_to_parts(&z);
        let re_path = dir.path().join("ifg_re.tif");
        let im_path = dir.path().join("ifg_im.tif");
        crate::io::write_geotiff(&re_out, &re_path, None).unwrap();
        crate::io::write_geotiff(&im_out, &im_path, None).unwrap();

        let re_in: Raster<f32> = crate::io::read_geotiff(&re_path, None).unwrap();
        let im_in: Raster<f32> = crate::io::read_geotiff(&im_path, None).unwrap();
        let z2 = complex_from_parts(&re_in, &im_in).unwrap();

        for row in 0..6 {
            for col in 0..5 {
                assert_eq!(
                    z.get(row, col).unwrap(),
                    z2.get(row, col).unwrap(),
                    "drift at ({}, {})",
                    row,
                    col
                );
            }
        }
    }
}
