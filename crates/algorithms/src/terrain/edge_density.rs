//! Edge Density
//!
//! Computes the proportion of edge pixels within a focal window.
//! Edges are detected using the Sobel operator, then a threshold
//! is applied to create a binary edge mask. The density is the
//! fraction of edge pixels within the window.
//!
//! - 0 = no edges in the neighbourhood (smooth terrain)
//! - 1 = all pixels are edges (highly dissected terrain)
//!
//! Reference: WhiteboxTools `EdgeDensity`

use ndarray::Array2;
use crate::maybe_rayon::*;
use crate::texture::sobel_edge;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for edge density.
#[derive(Debug, Clone)]
pub struct EdgeDensityParams {
    /// Neighborhood radius for density calculation (default 3).
    pub radius: usize,
    /// Sobel magnitude threshold to classify a pixel as an edge.
    /// Pixels with Sobel magnitude > threshold are edges.
    /// Default 0.5 (relative to input units).
    pub threshold: f64,
}

impl Default for EdgeDensityParams {
    fn default() -> Self {
        Self {
            radius: 3,
            threshold: 0.5,
        }
    }
}

/// Compute edge density within a focal window.
pub fn edge_density(dem: &Raster<f64>, params: EdgeDensityParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let threshold = params.threshold;

    // Step 1: compute Sobel edge magnitudes
    let edges = sobel_edge(dem)?;
    let edge_nodata = edges.nodata();

    // Step 2: focal proportion of edge pixels
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];
            let nodata = dem.nodata();

            for (col, out) in row_data.iter_mut().enumerate() {
                let center = unsafe { dem.get_unchecked(row, col) };
                if center.is_nan()
                    || nodata.is_some_and(|nd| (center - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                let ri = row as isize;
                let ci = col as isize;
                if ri < r || ri >= rows as isize - r || ci < r || ci >= cols as isize - r {
                    continue;
                }

                let mut edge_count = 0u32;
                let mut total = 0u32;

                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = (ri + dr) as usize;
                        let nc = (ci + dc) as usize;
                        let ev = unsafe { edges.get_unchecked(nr, nc) };
                        if ev.is_nan()
                            || edge_nodata.is_some_and(|nd| (ev - nd).abs() < f64::EPSILON)
                        {
                            continue;
                        }
                        total += 1;
                        if ev > threshold {
                            edge_count += 1;
                        }
                    }
                }

                if total > 0 {
                    *out = edge_count as f64 / total as f64;
                }
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_flat_no_edges() {
        let mut dem = Raster::filled(15, 15, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 15.0, 1.0, -1.0));

        let result = edge_density(&dem, EdgeDensityParams { radius: 2, threshold: 0.5 }).unwrap();
        let v = result.get(7, 7).unwrap();
        assert!((v - 0.0).abs() < 1e-10, "Flat surface should have 0 edge density, got {}", v);
    }

    #[test]
    fn test_range_zero_to_one() {
        let mut dem = Raster::new(20, 20);
        dem.set_transform(GeoTransform::new(0.0, 20.0, 1.0, -1.0));
        for r in 0..20 {
            for c in 0..20 {
                // Checkerboard creates lots of edges
                dem.set(r, c, if (r + c) % 2 == 0 { 100.0 } else { 0.0 }).unwrap();
            }
        }

        let result = edge_density(&dem, EdgeDensityParams { radius: 2, threshold: 0.5 }).unwrap();
        for r in 3..17 {
            for c in 3..17 {
                let v = result.get(r, c).unwrap();
                if !v.is_nan() {
                    assert!(v >= 0.0 && v <= 1.0,
                        "Edge density should be [0,1], got {} at ({},{})", v, r, c);
                }
            }
        }
    }
}
