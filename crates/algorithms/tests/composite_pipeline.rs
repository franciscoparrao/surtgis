//! Tests for composite pipeline components: cloud mask, resample.
//!
//! Run: cargo test -p surtgis-algorithms --test composite_pipeline

use ndarray::Array2;
use surtgis_algorithms::imagery::{CloudMaskStrategy, S2SclMask};
use surtgis_core::raster::{GeoTransform, Raster};

// =========================================================================
// Cloud Mask: SCL class handling
// =========================================================================

#[test]
fn cloud_mask_scl_class_0_passes_through() {
    let data_arr = Array2::from_elem((10, 10), 3000.0f64);
    let mut data = Raster::from_array(data_arr);
    data.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));

    // SCL class 0 = no classification data
    let scl_arr = Array2::from_elem((10, 10), 0.0f64);
    let mut scl = Raster::from_array(scl_arr);
    scl.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));

    let mask = S2SclMask::new();
    let result = mask.mask(&data, &scl).expect("mask failed");

    let valid = result.data().iter().filter(|v| v.is_finite()).count();
    assert_eq!(valid, 100, "SCL=0 should pass through, got {} valid of 100", valid);
}

#[test]
fn cloud_mask_scl_class_5_bare_soil_passes() {
    let data_arr = Array2::from_elem((10, 10), 3000.0f64);
    let mut data = Raster::from_array(data_arr);
    data.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));

    let scl_arr = Array2::from_elem((10, 10), 5.0f64); // bare soil
    let mut scl = Raster::from_array(scl_arr);
    scl.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));

    let mask = S2SclMask::new();
    let result = mask.mask(&data, &scl).expect("mask failed");

    let valid = result.data().iter().filter(|v| v.is_finite()).count();
    assert_eq!(valid, 100, "SCL=5 (bare soil) should pass");
}

#[test]
fn cloud_mask_scl_rejects_clouds() {
    let data_arr = Array2::from_elem((10, 10), 3000.0f64);
    let mut data = Raster::from_array(data_arr);
    data.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));

    let scl_arr = Array2::from_elem((10, 10), 9.0f64); // cloud_high
    let mut scl = Raster::from_array(scl_arr);
    scl.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));

    let mask = S2SclMask::new();
    let result = mask.mask(&data, &scl).expect("mask failed");

    let valid = result.data().iter().filter(|v| v.is_finite()).count();
    assert_eq!(valid, 0, "SCL=9 (cloud) should be rejected");
}

#[test]
fn cloud_mask_scl_mixed_classes() {
    let data_arr = Array2::from_elem((2, 5), 3000.0f64);
    let mut data = Raster::from_array(data_arr);
    data.set_transform(GeoTransform::new(0.0, 20.0, 10.0, -10.0));

    // Mix: 0=nodata, 4=veg, 5=soil, 8=cloud_med, 9=cloud_high
    let scl_data = vec![0.0, 4.0, 5.0, 8.0, 9.0, 0.0, 6.0, 11.0, 3.0, 7.0];
    let scl_arr = Array2::from_shape_vec((2, 5), scl_data).unwrap();
    let mut scl = Raster::from_array(scl_arr);
    scl.set_transform(GeoTransform::new(0.0, 20.0, 10.0, -10.0));

    let mask = S2SclMask::new(); // valid: [4, 5, 6, 11] + 0 (passthrough)
    let result = mask.mask(&data, &scl).expect("mask failed");

    // Classes that should pass: 0, 4, 5, 0, 6, 11 = 6 pixels
    // Classes that should fail: 8, 9, 3, 7 = 4 pixels
    let valid = result.data().iter().filter(|v| v.is_finite()).count();
    assert_eq!(valid, 6, "Expected 6 valid pixels (0,4,5,0,6,11), got {}", valid);
}

#[test]
fn cloud_mask_scl_different_resolution() {
    // Data at 10m (10x10), SCL at 20m (5x5)
    let data_arr = Array2::from_elem((10, 10), 3000.0f64);
    let mut data = Raster::from_array(data_arr);
    data.set_transform(GeoTransform::new(0.0, 100.0, 10.0, -10.0));

    // SCL 5x5: all bare soil (5)
    let scl_arr = Array2::from_elem((5, 5), 5.0f64);
    let mut scl = Raster::from_array(scl_arr);
    scl.set_transform(GeoTransform::new(0.0, 100.0, 20.0, -20.0));

    let mask = S2SclMask::new();
    let result = mask.mask(&data, &scl).expect("mask failed");

    // All should pass (SCL rescaled from 5x5 to 10x10 via nearest neighbor)
    let valid = result.data().iter().filter(|v| v.is_finite()).count();
    assert_eq!(valid, 100, "Multi-res SCL should work: got {} valid of 100", valid);
}

// =========================================================================
// Resample: NaN-tolerant bilinear
// =========================================================================

#[test]
fn resample_10m_to_30m_basic() {
    // 6x6 at 10m → 2x2 at 30m
    let arr = Array2::from_elem((6, 6), 3000.0f64);
    let mut src = Raster::from_array(arr);
    src.set_transform(GeoTransform::new(0.0, 60.0, 10.0, -10.0));

    let ref_arr = Array2::from_elem((2, 2), 0.0f64);
    let mut reference = Raster::from_array(ref_arr);
    reference.set_transform(GeoTransform::new(0.0, 60.0, 30.0, -30.0));

    let result = surtgis_core::resample_to_grid(
        &src, &reference, surtgis_core::ResampleMethod::Bilinear,
    ).expect("resample failed");

    assert_eq!(result.shape(), (2, 2));

    // All values should be ~3000
    for r in 0..2 {
        for c in 0..2 {
            let v = result.data()[[r, c]];
            assert!(v.is_finite() && (v - 3000.0).abs() < 1.0,
                "pixel ({},{}) = {}, expected ~3000", r, c, v);
        }
    }
}

#[test]
fn resample_nan_tolerant() {
    // 6x6 with NaN border
    let mut arr = Array2::from_elem((6, 6), 3000.0f64);
    arr[[0, 4]] = f64::NAN;
    arr[[0, 5]] = f64::NAN;
    arr[[1, 5]] = f64::NAN;

    let mut src = Raster::from_array(arr);
    src.set_transform(GeoTransform::new(0.0, 60.0, 10.0, -10.0));

    let ref_arr = Array2::from_elem((2, 2), 0.0f64);
    let mut reference = Raster::from_array(ref_arr);
    reference.set_transform(GeoTransform::new(0.0, 60.0, 30.0, -30.0));

    let result = surtgis_core::resample_to_grid(
        &src, &reference, surtgis_core::ResampleMethod::Bilinear,
    ).expect("resample failed");

    // Pixel (0,1) has some NaN neighbors but should still produce a value
    let v01 = result.data()[[0, 1]];
    assert!(v01.is_finite(), "NaN-tolerant bilinear should handle partial NaN neighbors");
}
