//! Vector Ruggedness Measure (VRM)
//!
//! Measures terrain ruggedness as the dispersion of surface normal vectors
//! in a neighborhood. Unlike TRI which uses elevation differences, VRM
//! captures both slope and aspect variation simultaneously.
//!
//! ```text
//! VRM = 1 - |R| / n
//! ```
//! where R is the resultant vector of all unit normal vectors in the
//! neighborhood and n is the number of cells.
//!
//! VRM ranges from 0 (flat or uniformly tilted) to 1 (maximally rugged).
//! It is independent of slope magnitude — a steep but uniform hillside
//! has low VRM.
//!
//! Reference:
//! Sappington, J.M., Longshore, K.M. & Thompson, D.B. (2007).
//! Quantifying landscape ruggedness for animal habitat analysis.
//! Journal of Wildlife Management, 71(5), 1419–1425.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for VRM calculation
#[derive(Debug, Clone)]
pub struct VrmParams {
    /// Neighborhood radius in cells (default 1 → 3×3 window)
    pub radius: usize,
}

impl Default for VrmParams {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

/// Compute the Vector Ruggedness Measure.
///
/// # Algorithm
///
/// 1. For each cell, compute slope and aspect from a 3×3 window (Horn's method)
/// 2. Convert to unit surface normal vector: (sin(slope)·sin(aspect), sin(slope)·cos(aspect), cos(slope))
/// 3. In the neighborhood of radius r, sum all normal vectors → resultant R
/// 4. VRM = 1 - |R| / n where n is the number of valid cells
///
/// # Arguments
/// * `dem` — Input DEM raster
/// * `params` — VRM parameters (neighborhood radius)
///
/// # Returns
/// Raster<f64> with VRM values in [0, 1].
pub fn vrm(dem: &Raster<f64>, params: VrmParams) -> Result<Raster<f64>> {
    let rows = dem.rows();
    let cols = dem.cols();

    if rows < 3 || cols < 3 {
        return Err(Error::Algorithm("DEM must be at least 3×3 for VRM".into()));
    }

    let data = dem.data();
    let r = params.radius;

    // Step 1: Compute unit normal vectors (nx, ny, nz) for each cell
    // using Horn's method for slope and aspect
    let mut nx = Array2::from_elem((rows, cols), f64::NAN);
    let mut ny = Array2::from_elem((rows, cols), f64::NAN);
    let mut nz = Array2::from_elem((rows, cols), f64::NAN);

    // Compute normals for interior cells (need 1-cell border for Horn)
    for row in 1..rows - 1 {
        for col in 1..cols - 1 {
            let z1 = data[[row - 1, col - 1]];
            let z2 = data[[row - 1, col]];
            let z3 = data[[row - 1, col + 1]];
            let z4 = data[[row, col - 1]];
            let z6 = data[[row, col + 1]];
            let z7 = data[[row + 1, col - 1]];
            let z8 = data[[row + 1, col]];
            let z9 = data[[row + 1, col + 1]];

            if z1.is_nan() || z2.is_nan() || z3.is_nan()
                || z4.is_nan() || z6.is_nan()
                || z7.is_nan() || z8.is_nan() || z9.is_nan()
            {
                continue;
            }

            // Horn's method (unnormalized gradients)
            let dz_dx = (z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7);
            let dz_dy = (z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3);

            // Slope angle
            let slope = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();

            // Aspect angle (radians, 0=north, clockwise)
            let aspect = if dz_dx.abs() < 1e-15 && dz_dy.abs() < 1e-15 {
                0.0
            } else {
                let a = (-dz_dy).atan2(-dz_dx);
                if a < 0.0 { a + 2.0 * std::f64::consts::PI } else { a }
            };

            // Unit normal vector components
            let sin_s = slope.sin();
            let cos_s = slope.cos();
            nx[[row, col]] = sin_s * aspect.sin();
            ny[[row, col]] = sin_s * aspect.cos();
            nz[[row, col]] = cos_s;
        }
    }

    // Step 2: Compute VRM using neighborhood averaging of normal vectors
    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            // Need border of r+1 (r for neighborhood + 1 for normal computation)
            if row < r + 1 || row >= rows - r - 1 {
                return row_data;
            }

            for col in (r + 1)..(cols - r - 1) {
                let mut sum_x = 0.0;
                let mut sum_y = 0.0;
                let mut sum_z = 0.0;
                let mut count = 0_usize;

                for nr in row.saturating_sub(r)..=(row + r).min(rows - 1) {
                    for nc in col.saturating_sub(r)..=(col + r).min(cols - 1) {
                        let vx = nx[[nr, nc]];
                        if !vx.is_nan() {
                            sum_x += vx;
                            sum_y += ny[[nr, nc]];
                            sum_z += nz[[nr, nc]];
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let resultant = (sum_x * sum_x + sum_y * sum_y + sum_z * sum_z).sqrt();
                    row_data[col] = 1.0 - resultant / count as f64;
                }
            }

            row_data
        })
        .collect();

    let mut output = Raster::new(rows, cols);
    output.set_transform(*dem.transform());
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::raster::GeoTransform;

    fn flat_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64 * 10.0, 10.0, -10.0));
        *r.data_mut() = Array2::from_elem((rows, cols), 100.0);
        r
    }

    fn tilted_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut data = Array2::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                data[[r, c]] = c as f64 * 10.0; // uniform slope in x
            }
        }
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64 * 10.0, 10.0, -10.0));
        *r.data_mut() = data;
        r
    }

    fn rugged_dem(rows: usize, cols: usize) -> Raster<f64> {
        let mut data = Array2::zeros((rows, cols));
        for r in 0..rows {
            for c in 0..cols {
                // Pseudo-random surface with varying slopes and aspects
                data[[r, c]] = ((r * 7 + c * 13) % 17) as f64 * 10.0;
            }
        }
        let mut r = Raster::new(rows, cols);
        r.set_transform(GeoTransform::new(0.0, rows as f64 * 10.0, 10.0, -10.0));
        *r.data_mut() = data;
        r
    }

    #[test]
    fn test_vrm_flat() {
        let dem = flat_dem(10, 10);
        let result = vrm(&dem, VrmParams::default()).unwrap();

        // Flat surface: all normals point straight up → VRM = 0
        for row in 2..8 {
            for col in 2..8 {
                let v = result.get(row, col).unwrap();
                if !v.is_nan() {
                    assert!(
                        v.abs() < 0.01,
                        "VRM on flat should be ~0, got {:.4} at ({},{})",
                        v, row, col
                    );
                }
            }
        }
    }

    #[test]
    fn test_vrm_tilted_uniform() {
        let dem = tilted_dem(10, 10);
        let result = vrm(&dem, VrmParams::default()).unwrap();

        // Uniformly tilted: all normals point same direction → VRM ≈ 0
        for row in 2..8 {
            for col in 2..8 {
                let v = result.get(row, col).unwrap();
                if !v.is_nan() {
                    assert!(
                        v < 0.05,
                        "VRM on uniform slope should be near 0, got {:.4} at ({},{})",
                        v, row, col
                    );
                }
            }
        }
    }

    #[test]
    fn test_vrm_rugged() {
        let dem = rugged_dem(10, 10);
        let result = vrm(&dem, VrmParams::default()).unwrap();

        // Rugged surface: normals point in many directions → VRM > 0
        let center = result.get(5, 5).unwrap();
        assert!(
            !center.is_nan() && center > 0.1,
            "VRM on rugged terrain should be high, got {:.4}",
            center
        );
    }

    #[test]
    fn test_vrm_range() {
        let dem = rugged_dem(10, 10);
        let result = vrm(&dem, VrmParams::default()).unwrap();

        for row in 0..10 {
            for col in 0..10 {
                let v = result.get(row, col).unwrap();
                if !v.is_nan() {
                    assert!(
                        v >= -0.001 && v <= 1.001,
                        "VRM should be in [0,1], got {:.4} at ({},{})",
                        v, row, col
                    );
                }
            }
        }
    }

    #[test]
    fn test_vrm_larger_radius() {
        let dem = rugged_dem(20, 20);
        let r1 = vrm(&dem, VrmParams { radius: 1 }).unwrap();
        let r3 = vrm(&dem, VrmParams { radius: 3 }).unwrap();

        // Both should produce valid results
        let v1 = r1.get(10, 10).unwrap();
        let v3 = r3.get(10, 10).unwrap();
        assert!(!v1.is_nan() && !v3.is_nan());
    }
}
