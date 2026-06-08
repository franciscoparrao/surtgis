//! Image segmentation algorithms.
//!
//! - **SLIC** superpixels (Achanta et al. 2012). K-means in
//!   (n_bands + 2)-dimensional space with spatial-vs-spectral
//!   trade-off controlled by a compactness parameter.
//! - **Felzenszwalb–Huttenlocher** graph segmentation (Felzenszwalb
//!   & Huttenlocher 2004). Region-merging on an 8-connected
//!   pixel graph with an adaptive threshold.
//!
//! Both algorithms return a label raster (`Raster<i32>`), one
//! integer label per pixel, ready for downstream zonal statistics
//! (`statistics::zonal_statistics`) or patch metrics.

mod felzenszwalb;
mod slic;

pub use felzenszwalb::{FelzenszwalbParams, felzenszwalb};
pub use slic::{SlicParams, slic};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::statistics::zonal_statistics;
    use surtgis_core::GeoTransform;
    use surtgis_core::raster::Raster;

    /// SLIC -> zonal_statistics chain on a half-bright/half-dark
    /// raster. Per-segment band means must concentrate at the two
    /// underlying values (0 and 1).
    #[test]
    fn slic_then_zonal_statistics_recovers_per_segment_means() {
        let mut r = Raster::new(20, 20);
        r.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for row in 0..20 {
            for col in 0..20 {
                r.set(row, col, if col < 10 { 0.0 } else { 1.0 }).unwrap();
            }
        }
        let labels = slic(
            &[&r],
            SlicParams {
                n_segments: 8,
                compactness: 1.0,
                max_iter: 8,
                enforce_connectivity: true,
            },
        )
        .unwrap();
        let stats = zonal_statistics(&r, &labels).unwrap();
        // zone == 0 is the nodata sentinel and is skipped by
        // zonal_statistics — only "real" segments appear.
        assert!(!stats.is_empty());
        for (zone, result) in &stats {
            assert!(*zone >= 1);
            // Every segment lies entirely inside one of the two
            // halves, so its mean must be either 0 or 1 exactly.
            assert!(
                (result.mean - 0.0).abs() < 1e-9 || (result.mean - 1.0).abs() < 1e-9,
                "zone {} has mean {} — straddled the boundary",
                zone,
                result.mean
            );
        }
    }

    /// Felzenszwalb -> zonal_statistics chain on the same raster
    /// should produce exactly the two expected zones (after the
    /// nodata-zone filter) with their respective means.
    #[test]
    fn felzenszwalb_then_zonal_statistics_recovers_two_zones() {
        let mut r = Raster::new(10, 10);
        r.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                r.set(row, col, if col < 5 { 0.0 } else { 100.0 }).unwrap();
            }
        }
        let labels = felzenszwalb(
            &[&r],
            FelzenszwalbParams {
                scale: 1.0,
                min_size: 10,
            },
        )
        .unwrap();
        let stats = zonal_statistics(&r, &labels).unwrap();
        assert_eq!(stats.len(), 2);
        let means: Vec<f64> = stats.values().map(|s| s.mean).collect();
        let has_zero = means.iter().any(|&m| (m - 0.0).abs() < 1e-9);
        let has_hundred = means.iter().any(|&m| (m - 100.0).abs() < 1e-9);
        assert!(
            has_zero && has_hundred,
            "expected per-segment means {{0, 100}}, got {:?}",
            means
        );
    }
}
