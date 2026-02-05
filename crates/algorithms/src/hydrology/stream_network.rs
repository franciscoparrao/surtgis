//! Stream network extraction
//!
//! Extracts a stream network from a flow accumulation raster by
//! thresholding: cells with accumulation >= threshold are classified
//! as stream cells.
//!
//! The output is a binary raster (1 = stream, 0 = non-stream).
//! This is a fundamental prerequisite for many hydrological workflows
//! including HAND, stream ordering, and sub-basin delineation.

use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::Result;

/// Parameters for stream network extraction
#[derive(Debug, Clone)]
pub struct StreamNetworkParams {
    /// Flow accumulation threshold (in cell counts).
    /// Cells with accumulation >= this value are classified as streams.
    /// Default: 1000.0
    pub threshold: f64,
}

impl Default for StreamNetworkParams {
    fn default() -> Self {
        Self { threshold: 1000.0 }
    }
}

/// Extract stream network from flow accumulation.
///
/// # Arguments
/// * `flow_acc` - Flow accumulation raster (from `flow_accumulation`)
/// * `params` - Stream network parameters (threshold)
///
/// # Returns
/// Raster<u8> with 1 = stream cell, 0 = non-stream cell
pub fn stream_network(flow_acc: &Raster<f64>, params: StreamNetworkParams) -> Result<Raster<u8>> {
    let (rows, cols) = flow_acc.shape();
    let threshold = params.threshold;

    let mut output_data = Array2::<u8>::zeros((rows, cols));

    for row in 0..rows {
        for col in 0..cols {
            let acc = unsafe { flow_acc.get_unchecked(row, col) };
            if !acc.is_nan() && acc >= threshold {
                output_data[(row, col)] = 1;
            }
        }
    }

    let mut output = flow_acc.with_same_meta::<u8>(rows, cols);
    output.set_nodata(Some(0));
    *output.data_mut() = output_data;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydrology::{flow_accumulation, flow_direction};
    use crate::hydrology::fill_sinks::{fill_sinks, FillSinksParams};
    use surtgis_core::GeoTransform;

    #[test]
    fn test_stream_network_threshold() {
        // Simple south-sloping DEM: accumulation increases downslope
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for row in 0..10 {
            for col in 0..10 {
                dem.set(row, col, (10 - row) as f64 * 10.0).unwrap();
            }
        }

        let filled = fill_sinks(&dem, FillSinksParams { min_slope: 0.0 }).unwrap();
        let fdir = flow_direction(&filled).unwrap();
        let facc = flow_accumulation(&fdir).unwrap();

        let streams = stream_network(&facc, StreamNetworkParams { threshold: 5.0 }).unwrap();

        // Top row should have 0 accumulation → no stream
        for col in 0..10 {
            assert_eq!(
                streams.get(0, col).unwrap(), 0,
                "Top row should not be stream at col {}",
                col
            );
        }

        // Bottom rows should have high accumulation → stream
        let bottom = streams.get(9, 5).unwrap();
        assert_eq!(bottom, 1, "Bottom center should be stream, got {}", bottom);
    }

    #[test]
    fn test_stream_network_binary_output() {
        // Verify output is only 0 or 1
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64 * 10.0).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let facc = flow_accumulation(&fdir).unwrap();
        let streams = stream_network(&facc, StreamNetworkParams { threshold: 2.0 }).unwrap();

        let (rows, cols) = streams.shape();
        for row in 0..rows {
            for col in 0..cols {
                let val = streams.get(row, col).unwrap();
                assert!(val == 0 || val == 1, "Expected 0 or 1, got {}", val);
            }
        }
    }

    #[test]
    fn test_stream_network_high_threshold_no_streams() {
        let mut dem = Raster::new(5, 5);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        for row in 0..5 {
            for col in 0..5 {
                dem.set(row, col, (5 - row) as f64).unwrap();
            }
        }

        let fdir = flow_direction(&dem).unwrap();
        let facc = flow_accumulation(&fdir).unwrap();

        // Very high threshold: no cell should qualify
        let streams = stream_network(&facc, StreamNetworkParams { threshold: 1000.0 }).unwrap();

        let (rows, cols) = streams.shape();
        for row in 0..rows {
            for col in 0..cols {
                assert_eq!(streams.get(row, col).unwrap(), 0, "No streams expected");
            }
        }
    }
}
