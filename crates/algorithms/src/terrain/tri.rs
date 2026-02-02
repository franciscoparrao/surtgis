//! Terrain Ruggedness Index (TRI)
//!
//! TRI quantifies the heterogeneity of the terrain by measuring the total
//! elevation change between a grid cell and its neighborhood.
//!
//! Riley et al. (1999) method:
//!   TRI = sqrt( sum( (z_neighbor - z_center)² ) / n )
//!
//! - High TRI → rough, rugged terrain
//! - Low TRI → smooth, flat terrain
//!
//! Riley classification (approximate ranges for 30m DEMs):
//!   0–80 m   : Level
//!   81–116 m : Nearly level
//!   117–161 m: Slightly rugged
//!   162–239 m: Intermediately rugged
//!   240–497 m: Moderately rugged
//!   498–958 m: Highly rugged
//!   959+ m   : Extremely rugged
//!
//! Reference: Riley, S.J., DeGloria, S.D., Elliot, R. (1999)

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Algorithm, Error, Result};

/// Parameters for TRI calculation
#[derive(Debug, Clone)]
pub struct TriParams {
    /// Neighborhood radius in cells (default 1 → 3x3)
    pub radius: usize,
}

impl Default for TriParams {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

/// TRI algorithm
#[derive(Debug, Clone, Default)]
pub struct Tri;

impl Algorithm for Tri {
    type Input = Raster<f64>;
    type Output = Raster<f64>;
    type Params = TriParams;
    type Error = Error;

    fn name(&self) -> &'static str {
        "TRI"
    }

    fn description(&self) -> &'static str {
        "Terrain Ruggedness Index: elevation variability in a neighborhood"
    }

    fn execute(&self, input: Self::Input, params: Self::Params) -> Result<Self::Output> {
        tri(&input, params)
    }
}

/// Calculate Terrain Ruggedness Index
///
/// # Arguments
/// * `dem` - Input DEM raster
/// * `params` - TRI parameters (neighborhood radius)
///
/// # Returns
/// Raster with TRI values (same units as input elevation)
pub fn tri(dem: &Raster<f64>, params: TriParams) -> Result<Raster<f64>> {
    let (rows, cols) = dem.shape();
    let r = params.radius as isize;
    let nodata = dem.nodata();

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let center = unsafe { dem.get_unchecked(row, col) };
                if center.is_nan() || nodata.is_some_and(|nd| (center - nd).abs() < f64::EPSILON)
                {
                    continue;
                }

                let ri = row as isize;
                let ci = col as isize;
                if ri < r || ri >= rows as isize - r || ci < r || ci >= cols as isize - r {
                    continue;
                }

                let mut sum_sq = 0.0;
                let mut count = 0u32;

                for dr in -r..=r {
                    for dc in -r..=r {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = (ri + dr) as usize;
                        let nc = (ci + dc) as usize;
                        let nv = unsafe { dem.get_unchecked(nr, nc) };
                        if !nv.is_nan() && nodata.is_none_or(|nd| (nv - nd).abs() >= f64::EPSILON)
                        {
                            let diff = nv - center;
                            sum_sq += diff * diff;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    *row_data_col = (sum_sq / count as f64).sqrt();
                }
            }

            row_data
        })
        .collect();

    let mut output = dem.with_same_meta::<f64>(rows, cols);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() =
        Array2::from_shape_vec((rows, cols), output_data).map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_tri_flat_surface() {
        let mut dem = Raster::filled(10, 10, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));

        let result = tri(&dem, TriParams::default()).unwrap();
        let val = result.get(5, 5).unwrap();
        assert!(
            val.abs() < 1e-10,
            "Expected TRI=0 for flat surface, got {}",
            val
        );
    }

    #[test]
    fn test_tri_rugged_surface() {
        // Checkerboard pattern: alternating 0 and 100
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, if (r + c) % 2 == 0 { 100.0 } else { 0.0 })
                    .unwrap();
            }
        }

        let result = tri(&dem, TriParams::default()).unwrap();
        let val = result.get(5, 5).unwrap();
        // Cell (5,5) has (5+5)%2=0 → value 100
        // Neighbors: 4 same-parity (100, diff=0) + 4 opposite-parity (0, diff=-100)
        // sum_sq = 4*0 + 4*10000 = 40000, count=8
        // TRI = sqrt(40000/8) = sqrt(5000) ≈ 70.71
        let expected = (40000.0_f64 / 8.0).sqrt();
        assert!(
            (val - expected).abs() < 1e-6,
            "Expected TRI≈{:.2} for checkerboard, got {}",
            expected, val
        );
    }

    #[test]
    fn test_tri_gentle_slope() {
        // Plane z = row → constant gradient of 1.0/cell
        let mut dem = Raster::new(10, 10);
        dem.set_transform(GeoTransform::new(0.0, 10.0, 1.0, -1.0));
        for r in 0..10 {
            for c in 0..10 {
                dem.set(r, c, r as f64).unwrap();
            }
        }

        let result = tri(&dem, TriParams::default()).unwrap();
        let val = result.get(5, 5).unwrap();
        // Neighbors: row-1 (3 cells diff=-1), row (2 cells diff=0), row+1 (3 cells diff=1)
        // sum_sq = 3*1 + 2*0 + 3*1 = 6, count=8
        // TRI = sqrt(6/8) ≈ 0.866
        assert!(
            (val - (6.0_f64 / 8.0).sqrt()).abs() < 1e-6,
            "Expected TRI≈0.866 for gentle slope, got {}",
            val
        );
    }

    #[test]
    fn test_tri_nodata_border() {
        let mut dem = Raster::filled(5, 5, 100.0);
        dem.set_transform(GeoTransform::new(0.0, 5.0, 1.0, -1.0));
        let result = tri(&dem, TriParams::default()).unwrap();
        assert!(result.get(0, 0).unwrap().is_nan());
    }
}
