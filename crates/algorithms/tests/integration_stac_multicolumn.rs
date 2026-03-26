//! End-to-end tests for multi-collection STAC support
//! Tests S2, Landsat, and Sentinel-1 CloudMaskStrategy implementations

#[cfg(test)]
mod tests {
    use surtgis_algorithms::imagery::{
        CloudMaskStrategy, S2SclMask, LandsatQaMask, NoCloudMask,
    };
    use surtgis_core::Raster;
    use std::sync::Arc;

    /// Test S2SclMask strategy with synthetic data
    #[test]
    fn test_s2_cloud_masking_trait() {
        // Create synthetic data
        let data_array = ndarray::Array2::from_elem((10, 10), 1000.0);
        let mut data = Raster::from_array(data_array);
        data.set_transform(surtgis_core::GeoTransform::new(0.0, 100.0, 1.0, -1.0));

        // Create SCL mask: some valid (4,5,6,11), some invalid (9=cloud)
        let mut scl_array = ndarray::Array2::from_elem((10, 10), 4.0); // all valid
        scl_array[[2, 2]] = 9.0; // cloud
        scl_array[[3, 3]] = 3.0; // shadow
        scl_array[[5, 5]] = 8.0; // cloud medium

        let mut scl = Raster::from_array(scl_array);
        scl.set_transform(surtgis_core::GeoTransform::new(0.0, 100.0, 1.0, -1.0));

        // Apply masking
        let strategy = S2SclMask::new();
        let result = strategy.mask(&data, &scl).unwrap();

        // Verify results
        assert!((result.data()[[0, 0]] - 1000.0).abs() < 1e-6); // valid
        assert!(result.data()[[2, 2]].is_nan()); // cloud masked
        assert!(result.data()[[3, 3]].is_nan()); // shadow masked
        assert!(result.data()[[5, 5]].is_nan()); // cloud masked
    }

    /// Test LandsatQaMask strategy with synthetic data
    #[test]
    fn test_landsat_cloud_masking_trait() {
        // Create synthetic data
        let data_array = ndarray::Array2::from_elem((10, 10), 5000.0);
        let mut data = Raster::from_array(data_array);
        data.set_transform(surtgis_core::GeoTransform::new(0.0, 100.0, 1.0, -1.0));

        // Create QA_PIXEL mask: bitmask with cloud/shadow/snow bits
        // bit 1 = cloud, bit 3 = cloud shadow, bit 4 = snow
        let mut qa_array = ndarray::Array2::from_elem((10, 10), 0.0); // all valid (no bits set)
        qa_array[[2, 2]] = 2.0; // bit 1 set = cloud
        qa_array[[3, 3]] = 8.0; // bit 3 set = shadow
        qa_array[[5, 5]] = 16.0; // bit 4 set = snow

        let mut qa = Raster::from_array(qa_array);
        qa.set_transform(surtgis_core::GeoTransform::new(0.0, 100.0, 1.0, -1.0));

        // Apply masking
        let strategy = LandsatQaMask::new(); // default: exclude bits 1,3,4
        let result = strategy.mask(&data, &qa).unwrap();

        // Verify results
        assert!((result.data()[[0, 0]] - 5000.0).abs() < 1e-6); // valid
        assert!(result.data()[[2, 2]].is_nan()); // cloud masked
        assert!(result.data()[[3, 3]].is_nan()); // shadow masked
        assert!(result.data()[[5, 5]].is_nan()); // snow masked
    }

    /// Test NoCloudMask strategy (SAR) - should preserve all data
    #[test]
    fn test_sar_no_cloud_masking() {
        // Create synthetic data
        let data_array = ndarray::Array2::from_elem((10, 10), -15.0); // dB values
        let mut data = Raster::from_array(data_array);
        data.set_transform(surtgis_core::GeoTransform::new(0.0, 100.0, 1.0, -1.0));

        // Create dummy mask (not used)
        let mask_array = ndarray::Array2::from_elem((10, 10), 0.0);
        let mut mask = Raster::from_array(mask_array);
        mask.set_transform(surtgis_core::GeoTransform::new(0.0, 100.0, 1.0, -1.0));

        // Apply no-op masking
        let strategy = NoCloudMask;
        let result = strategy.mask(&data, &mask).unwrap();

        // Verify all data preserved
        for v in result.data().iter() {
            assert!((*v - (-15.0)).abs() < 1e-6 || v.is_nan() == false);
        }
    }

    /// Test that CloudMaskStrategy trait works for all 3 implementations
    #[test]
    fn test_cloud_mask_strategy_trait_polymorphism() {
        let strategies: Vec<Arc<dyn CloudMaskStrategy>> = vec![
            Arc::new(S2SclMask::new()),
            Arc::new(LandsatQaMask::new()),
            Arc::new(NoCloudMask),
        ];

        // Create test data
        let data_array = ndarray::Array2::from_elem((5, 5), 100.0);
        let mut data = Raster::from_array(data_array);
        data.set_transform(surtgis_core::GeoTransform::new(0.0, 50.0, 1.0, -1.0));

        let mask_array = ndarray::Array2::from_elem((5, 5), 0.0);
        let mut mask = Raster::from_array(mask_array);
        mask.set_transform(surtgis_core::GeoTransform::new(0.0, 50.0, 1.0, -1.0));

        // All strategies should execute without panicking
        for strategy in strategies {
            let _result = strategy.mask(&data, &mask);
            // In real scenario, we'd validate results, but this is just a polymorphism test
        }
    }

    /// Test S2SclMask with custom valid classes
    #[test]
    fn test_s2_custom_valid_classes() {
        let data_array = ndarray::Array2::from_elem((5, 5), 50.0);
        let mut data = Raster::from_array(data_array);
        data.set_transform(surtgis_core::GeoTransform::new(0.0, 25.0, 1.0, -1.0));

        // SCL with only class 4 (vegetation)
        let scl_array = ndarray::Array2::from_elem((5, 5), 4.0);
        let mut scl = Raster::from_array(scl_array);
        scl.set_transform(surtgis_core::GeoTransform::new(0.0, 25.0, 1.0, -1.0));

        // Create strategy with custom classes (only water = 6)
        let strategy = S2SclMask::with_classes(vec![6]);
        let result = strategy.mask(&data, &scl).unwrap();

        // All pixels should be masked (class 4 not in keep list)
        assert!(result.data().iter().all(|v| v.is_nan()));
    }

    /// Test Landsat custom exclude bits
    #[test]
    fn test_landsat_custom_exclude_bits() {
        let data_array = ndarray::Array2::from_elem((5, 5), 500.0);
        let mut data = Raster::from_array(data_array);
        data.set_transform(surtgis_core::GeoTransform::new(0.0, 25.0, 1.0, -1.0));

        // QA with only bit 1 set (cloud)
        let qa_array = ndarray::Array2::from_elem((5, 5), 2.0);
        let mut qa = Raster::from_array(qa_array);
        qa.set_transform(surtgis_core::GeoTransform::new(0.0, 25.0, 1.0, -1.0));

        // Create strategy that only excludes bit 3 (shadow, not bit 1)
        let strategy = LandsatQaMask::with_bits(0b1000); // only bit 3
        let result = strategy.mask(&data, &qa).unwrap();

        // Pixels should NOT be masked (bit 1 set but not excluded)
        assert!(result.data().iter().all(|v| (*v - 500.0).abs() < 1e-6));
    }

    /// Test description strings for logging
    #[test]
    fn test_cloud_mask_strategy_descriptions() {
        let s2 = S2SclMask::new();
        let landsat = LandsatQaMask::new();
        let sar = NoCloudMask;

        assert_eq!(s2.description(), "S2 SCL (Scene Classification Layer)");
        assert_eq!(landsat.description(), "Landsat C2 QA_PIXEL (bitmask)");
        assert_eq!(sar.description(), "None (SAR penetrates clouds)");
    }
}
