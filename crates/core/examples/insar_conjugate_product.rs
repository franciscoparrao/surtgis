//! Foundation smoke test for `insar-rs`: prove that an external
//! interferometry crate can do a full read → conjugate-product →
//! write cycle using **only** the surface `surtgis-core` exposes
//! under the `complex` feature — no InSAR domain logic lives here.
//!
//! Pipeline: two synthetic SLC-like complex rasters (master, slave)
//! persist as (re, im) GeoTIFF pairs, get read back, combined into an
//! interferogram via the elementwise conjugate product
//! `master * conj(slave)`, and the interferogram's magnitude/phase
//! are written out. The conjugate product itself — the one genuinely
//! InSAR-specific step — is implemented here, in the "consumer", not
//! in core.
//!
//! Run with: `cargo run -p surtgis-core --example insar_conjugate_product --features complex`

use num_complex::Complex;

use surtgis_core::io::{read_geotiff, write_geotiff};
use surtgis_core::raster::{complex_from_parts, magnitude, phase};
use surtgis_core::{GeoTransform, Raster};

fn synthetic_part(rows: usize, cols: usize, phase_rate: f32, amp: f32) -> Raster<f32> {
    let mut r: Raster<f32> = Raster::new(rows, cols);
    r.set_transform(GeoTransform::new(500_000.0, 6_300_000.0, 20.0, -20.0));
    r.set_nodata(Some(f32::NAN));
    for row in 0..rows {
        for col in 0..cols {
            let theta = phase_rate * (row + col) as f32;
            r.set(row, col, amp * theta.cos()).unwrap();
        }
    }
    r
}

/// The one InSAR-specific step in this whole example: elementwise
/// conjugate product `a * conj(b)`. This is exactly the kind of logic
/// the SPEC keeps out of core — it belongs to the consumer.
fn conjugate_product(a: &Raster<Complex<f32>>, b: &Raster<Complex<f32>>) -> Raster<Complex<f32>> {
    let (rows, cols) = a.shape();
    let mut out: Raster<Complex<f32>> = a.with_same_meta(rows, cols);
    out.set_nodata(Some(Complex::new(f32::NAN, f32::NAN)));
    for row in 0..rows {
        for col in 0..cols {
            let za = unsafe { a.get_unchecked(row, col) };
            let zb = unsafe { b.get_unchecked(row, col) };
            let v = if za.re.is_nan() || zb.re.is_nan() {
                Complex::new(f32::NAN, f32::NAN)
            } else {
                za * zb.conj()
            };
            unsafe { out.set_unchecked(row, col, v) };
        }
    }
    out
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let (rows, cols) = (16usize, 12usize);

    // Two synthetic SLC-like acquisitions, each persisted as an (re, im) pair
    // — the I/O path core actually supports today.
    let master_re = synthetic_part(rows, cols, 0.15, 1.0);
    let master_im = synthetic_part(rows, cols, 0.15 + std::f32::consts::FRAC_PI_2, 1.0);
    let slave_re = synthetic_part(rows, cols, 0.11, 0.9);
    let slave_im = synthetic_part(rows, cols, 0.11 + std::f32::consts::FRAC_PI_2, 0.9);

    for (name, r) in [
        ("master_re", &master_re),
        ("master_im", &master_im),
        ("slave_re", &slave_re),
        ("slave_im", &slave_im),
    ] {
        write_geotiff(r, dir.path().join(format!("{name}.tif")), None).unwrap();
    }

    // Read back and reassemble as complex rasters — the fundación surface.
    let master_re: Raster<f32> = read_geotiff(dir.path().join("master_re.tif"), None).unwrap();
    let master_im: Raster<f32> = read_geotiff(dir.path().join("master_im.tif"), None).unwrap();
    let slave_re: Raster<f32> = read_geotiff(dir.path().join("slave_re.tif"), None).unwrap();
    let slave_im: Raster<f32> = read_geotiff(dir.path().join("slave_im.tif"), None).unwrap();

    let master = complex_from_parts(&master_re, &master_im).unwrap();
    let slave = complex_from_parts(&slave_re, &slave_im).unwrap();

    // The consumer-side interferometric step.
    let interferogram = conjugate_product(&master, &slave);

    // Core's magnitude/phase helpers, then persist as ordinary float rasters.
    let mag = magnitude(&interferogram);
    let ph = phase(&interferogram);
    write_geotiff(&mag, dir.path().join("ifg_magnitude.tif"), None).unwrap();
    write_geotiff(&ph, dir.path().join("ifg_phase.tif"), None).unwrap();

    let center = interferogram.get(rows / 2, cols / 2).unwrap();
    println!(
        "interferogram[{},{}] = {:.4}{:+.4}i, |z|={:.4}, phase={:.4} rad",
        rows / 2,
        cols / 2,
        center.re,
        center.im,
        mag.get(rows / 2, cols / 2).unwrap(),
        ph.get(rows / 2, cols / 2).unwrap()
    );
    println!("insar-rs foundation smoke test: read -> conjugate-product -> write OK");
}
