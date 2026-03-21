//! Drainage Density: stream length per unit area in a moving window
//!
//! Drainage density (DD) measures the total length of streams per unit area,
//! computed within a circular (square) moving window. Higher values indicate
//! more dissected terrain with denser drainage networks.
//!
//! DD = (stream_cells_in_window * cell_size) / (window_cells * cell_size^2)
//!    = stream_cells_in_window / (window_cells * cell_size)
//!
//! Units: 1/m (inverse length)
//!
//! Reference:
//! Horton, R.E. (1932). Drainage-basin characteristics. *Transactions of the
//! American Geophysical Union*, 13(1), 350–361.

use crate::maybe_rayon::*;
use ndarray::Array2;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Drainage Density computation
#[derive(Debug, Clone)]
pub struct DrainageDensityParams {
    /// Window radius in cells (default 5)
    pub radius: usize,
    /// Cell size in meters (default 1.0)
    pub cell_size: f64,
}

impl Default for DrainageDensityParams {
    fn default() -> Self {
        Self {
            radius: 5,
            cell_size: 1.0,
        }
    }
}

/// Compute drainage density: stream length per unit area in a moving window.
///
/// # Arguments
/// * `stream_network` - Binary stream raster: 1.0 = stream, 0.0 = non-stream (read as f64)
/// * `params` - Drainage density parameters (radius, cell_size)
///
/// # Returns
/// Raster with drainage density values (1/m)
pub fn drainage_density(
    stream_network: &Raster<f64>,
    params: DrainageDensityParams,
) -> Result<Raster<f64>> {
    let (rows, cols) = stream_network.shape();
    let r = params.radius as isize;
    let cell_size = params.cell_size;
    let nodata = stream_network.nodata();

    if params.radius == 0 {
        return Err(Error::Other("Drainage density radius must be >= 1".into()));
    }
    if cell_size <= 0.0 {
        return Err(Error::Other("Cell size must be positive".into()));
    }

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            let ri = row as isize;

            for col in 0..cols {
                let ci = col as isize;
                let mut stream_cells = 0u32;
                let mut window_cells = 0u32;

                for dr in -r..=r {
                    let nr = ri + dr;
                    if nr < 0 || nr >= rows as isize {
                        continue;
                    }
                    for dc in -r..=r {
                        let nc = ci + dc;
                        if nc < 0 || nc >= cols as isize {
                            continue;
                        }
                        let val =
                            unsafe { stream_network.get_unchecked(nr as usize, nc as usize) };

                        if val.is_nan()
                            || nodata.is_some_and(|nd| (val - nd).abs() < f64::EPSILON)
                        {
                            continue;
                        }

                        window_cells += 1;
                        // Stream cell if value ≈ 1.0
                        if (val - 1.0).abs() < 0.5 {
                            stream_cells += 1;
                        }
                    }
                }

                if window_cells > 0 {
                    // DD = stream_cells / (window_cells * cell_size)
                    row_data[col] =
                        stream_cells as f64 / (window_cells as f64 * cell_size);
                }
            }
            row_data
        })
        .collect();

    let mut output = stream_network.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_drainage_density_no_streams() {
        // No stream cells → DD = 0
        let mut stream = Raster::filled(5, 5, 0.0);
        stream.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let result = drainage_density(
            &stream,
            DrainageDensityParams {
                radius: 1,
                cell_size: 10.0,
            },
        )
        .unwrap();

        let val = result.get(2, 2).unwrap();
        assert!(
            val.abs() < 1e-10,
            "No streams should give DD=0, got {}",
            val
        );
    }

    #[test]
    fn test_drainage_density_all_streams() {
        // All stream cells → DD = 1/cell_size
        let mut stream = Raster::filled(5, 5, 1.0);
        stream.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));

        let result = drainage_density(
            &stream,
            DrainageDensityParams {
                radius: 1,
                cell_size: 10.0,
            },
        )
        .unwrap();

        let val = result.get(2, 2).unwrap();
        // DD = 9/9/10 = 1/10 = 0.1 (3x3 window, all streams, cell_size=10)
        let expected = 1.0 / 10.0;
        assert!(
            (val - expected).abs() < 1e-10,
            "All streams should give DD={}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_drainage_density_center_stream() {
        // Only center cell is a stream in a 3x3 window → DD = 1/(9*cell_size)
        let data = vec![0.0; 25];
        let arr = Array2::from_shape_vec((5, 5), data).unwrap();
        let mut stream = Raster::from_array(arr);
        stream.set_transform(GeoTransform::new(0.0, 5.0, 10.0, -10.0));
        stream.set(2, 2, 1.0).unwrap();

        let result = drainage_density(
            &stream,
            DrainageDensityParams {
                radius: 1,
                cell_size: 10.0,
            },
        )
        .unwrap();

        let val = result.get(2, 2).unwrap();
        // 1 stream cell in 9 total → DD = 1/(9*10) ≈ 0.0111
        let expected = 1.0 / (9.0 * 10.0);
        assert!(
            (val - expected).abs() < 1e-6,
            "Single stream cell: expected DD≈{}, got {}",
            expected,
            val
        );
    }
}
