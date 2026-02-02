//! Multi-Scale Valleyness (MSV)
//!
//! Wang & Laffan (2009): Characterizes diffuse valley features by fitting
//! quadratic surfaces at multiple window sizes and extracting curvature
//! eigenvalues. Detects areas missed by MRVBF, D8, D-inf, and flow accumulation.
//!
//! The algorithm:
//! 1. At each scale (window radius), fit z = ax² + by² + cxy + dx + ey + f
//! 2. Hessian eigenvalues λ₁, λ₂ characterize local shape
//! 3. Valleyness = max(0, min_eigenvalue) — positive min curvature = concave
//! 4. Combine across scales with maximum or weighted sum
//!
//! Reference:
//! Wang, H. & Laffan, S.W. (2009). Multi-scale valleyness. *Computers & Geosciences*.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::Raster;
use surtgis_core::{Error, Result};

/// Parameters for Multi-Scale Valleyness
#[derive(Debug, Clone)]
pub struct MsvParams {
    /// Window radii to use (in cells). Default: [2, 5, 10, 20]
    pub radii: Vec<usize>,
    /// Combination method across scales. Default: Maximum
    pub combination: MsvCombination,
}

/// How to combine valleyness across scales
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MsvCombination {
    /// Take maximum valleyness across all scales
    Maximum,
    /// Take sum (with weight 1/n_scales)
    Mean,
}

impl Default for MsvParams {
    fn default() -> Self {
        Self {
            radii: vec![2, 5, 10, 20],
            combination: MsvCombination::Maximum,
        }
    }
}

/// Multi-Scale Valleyness result
#[derive(Debug)]
pub struct MsvResult {
    /// Combined valleyness map (0 = not valley, positive = valley)
    pub valleyness: Raster<f64>,
    /// Combined ridgeness map (0 = not ridge, positive = ridge)
    pub ridgeness: Raster<f64>,
}

/// Compute Multi-Scale Valleyness.
///
/// For each scale, fits a 2D quadratic surface within a circular window and
/// extracts Hessian eigenvalues. Positive minimum eigenvalue indicates concavity
/// (valley); negative maximum eigenvalue indicates convexity (ridge).
///
/// # Arguments
/// * `dem` — Input DEM
/// * `params` — MSV parameters
///
/// # Returns
/// [`MsvResult`] with valleyness and ridgeness rasters.
pub fn msv(dem: &Raster<f64>, params: MsvParams) -> Result<MsvResult> {
    if params.radii.is_empty() {
        return Err(Error::Algorithm("At least one radius is required".into()));
    }
    for &r in &params.radii {
        if r == 0 {
            return Err(Error::Algorithm("Radii must be > 0".into()));
        }
    }

    let (rows, cols) = dem.shape();
    let cell_size = dem.cell_size();
    let n_scales = params.radii.len();

    let mut valley_combined = Array2::from_elem((rows, cols), 0.0_f64);
    let mut ridge_combined = Array2::from_elem((rows, cols), 0.0_f64);

    for &radius in &params.radii {
        // For each scale, compute valleyness and ridgeness from quadratic fit
        let (scale_valley, scale_ridge) =
            compute_scale_curvature(dem, rows, cols, radius, cell_size);

        for row in 0..rows {
            for col in 0..cols {
                let v = scale_valley[(row, col)];
                let r = scale_ridge[(row, col)];

                if v.is_nan() || r.is_nan() {
                    continue;
                }

                match params.combination {
                    MsvCombination::Maximum => {
                        if v > valley_combined[(row, col)] {
                            valley_combined[(row, col)] = v;
                        }
                        if r > ridge_combined[(row, col)] {
                            ridge_combined[(row, col)] = r;
                        }
                    }
                    MsvCombination::Mean => {
                        valley_combined[(row, col)] += v / n_scales as f64;
                        ridge_combined[(row, col)] += r / n_scales as f64;
                    }
                }
            }
        }
    }

    let mut valleyness = dem.with_same_meta::<f64>(rows, cols);
    valleyness.set_nodata(Some(f64::NAN));
    *valleyness.data_mut() = valley_combined;

    let mut ridgeness = dem.with_same_meta::<f64>(rows, cols);
    ridgeness.set_nodata(Some(f64::NAN));
    *ridgeness.data_mut() = ridge_combined;

    Ok(MsvResult { valleyness, ridgeness })
}

/// Compute curvature eigenvalues at a single scale.
///
/// Fits z = ax² + by² + cxy + dx + ey + f using least-squares within a
/// circular window of given radius. The Hessian is [[2a, c], [c, 2b]].
/// Eigenvalues λ₁ ≥ λ₂ characterize the shape.
fn compute_scale_curvature(
    dem: &Raster<f64>,
    rows: usize, cols: usize,
    radius: usize, cell_size: f64,
) -> (Array2<f64>, Array2<f64>) {
    let r = radius as isize;
    let r2 = (radius as f64 * cell_size).powi(2);

    let valley_ridge: Vec<(f64, f64)> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![(f64::NAN, f64::NAN); cols];

            for col in 0..cols {
                let z0 = unsafe { dem.get_unchecked(row, col) };
                if z0.is_nan() {
                    continue;
                }

                // Collect (x, y, z) in local coordinates
                // Accumulate normal equations for: z = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
                // Using ATA and ATz directly for 6-parameter fit
                let mut ata = [0.0_f64; 21]; // upper triangle of 6x6
                let mut atz = [0.0_f64; 6];

                let mut count = 0_usize;

                for dr in -r..=r {
                    for dc in -r..=r {
                        let nr = row as isize + dr;
                        let nc = col as isize + dc;
                        if nr < 0 || nc < 0 || (nr as usize) >= rows || (nc as usize) >= cols {
                            continue;
                        }

                        let x = dc as f64 * cell_size;
                        let y = dr as f64 * cell_size;

                        // Circular window
                        if x * x + y * y > r2 {
                            continue;
                        }

                        let z = unsafe { dem.get_unchecked(nr as usize, nc as usize) };
                        if z.is_nan() {
                            continue;
                        }

                        let dz = z - z0;
                        let x2 = x * x;
                        let y2 = y * y;
                        let xy = x * y;

                        // Basis: [x², y², xy, x, y, 1]
                        let basis = [x2, y2, xy, x, y, 1.0];

                        // Accumulate ATA (upper triangle, row-major)
                        let mut idx = 0;
                        for i in 0..6 {
                            for j in i..6 {
                                ata[idx] += basis[i] * basis[j];
                                idx += 1;
                            }
                            atz[i] += basis[i] * dz;
                        }

                        count += 1;
                    }
                }

                // Need at least 6 points for 6-parameter fit
                if count < 6 {
                    continue;
                }

                // Solve 6×6 system using Cholesky-like approach
                // We only need coefficients a, b, c for eigenvalues
                if let Some(coeffs) = solve_6x6_upper(&ata, &atz) {
                    let a = coeffs[0];
                    let b = coeffs[1];
                    let c = coeffs[2];

                    // Hessian = [[2a, c], [c, 2b]]
                    // Eigenvalues: λ = (2a + 2b) ± sqrt((2a - 2b)² + 4c²)) / 2
                    let trace = 2.0 * a + 2.0 * b;
                    let det = 4.0 * a * b - c * c;
                    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();

                    let lambda1 = (trace + disc) / 2.0; // larger eigenvalue
                    let lambda2 = (trace - disc) / 2.0; // smaller eigenvalue

                    // Valleyness: concavity → positive λ₂ (both eigenvalues positive = bowl)
                    // At minimum, one positive eigenvalue indicates valley
                    let valley = lambda2.max(0.0);

                    // Ridgeness: convexity → negative λ₁ (both negative = dome)
                    let ridge = (-lambda1).max(0.0);

                    row_data[col] = (valley, ridge);
                }
            }

            row_data
        })
        .collect();

    let mut valley = Array2::from_elem((rows, cols), f64::NAN);
    let mut ridge = Array2::from_elem((rows, cols), f64::NAN);
    for row in 0..rows {
        for col in 0..cols {
            let (v, r) = valley_ridge[row * cols + col];
            valley[(row, col)] = v;
            ridge[(row, col)] = r;
        }
    }

    (valley, ridge)
}

/// Solve 6×6 symmetric positive-definite system from upper triangle.
/// Returns None if system is singular.
fn solve_6x6_upper(ata: &[f64; 21], atz: &[f64; 6]) -> Option<[f64; 6]> {
    // Expand upper triangle to full 6×6 matrix
    let mut m = [[0.0_f64; 7]; 6]; // augmented matrix [A | b]
    let mut idx = 0;
    for i in 0..6 {
        for j in i..6 {
            m[i][j] = ata[idx];
            m[j][i] = ata[idx];
            idx += 1;
        }
        m[i][6] = atz[i];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..6 {
        // Find pivot
        let mut max_val = m[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..6 {
            if m[row][col].abs() > max_val {
                max_val = m[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return None; // Singular
        }

        if max_row != col {
            m.swap(col, max_row);
        }

        let pivot = m[col][col];
        for row in (col + 1)..6 {
            let factor = m[row][col] / pivot;
            for j in col..7 {
                m[row][j] -= factor * m[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0_f64; 6];
    for i in (0..6).rev() {
        let mut sum = m[i][6];
        for j in (i + 1)..6 {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surtgis_core::GeoTransform;

    #[test]
    fn test_msv_valley() {
        // Bowl with flat sides: center is concave, edges are flat
        let mut dem = Raster::new(31, 31);
        dem.set_transform(GeoTransform::new(0.0, 31.0, 1.0, -1.0));
        for row in 0..31 {
            for col in 0..31 {
                let dx = col as f64 - 15.0;
                let dy = row as f64 - 15.0;
                let r = (dx * dx + dy * dy).sqrt();
                // Bowl in center, flat at edges
                if r < 10.0 {
                    dem.set(row, col, r * r * 0.1).unwrap();
                } else {
                    dem.set(row, col, 10.0).unwrap();
                }
            }
        }

        let result = msv(&dem, MsvParams {
            radii: vec![3, 5],
            ..Default::default()
        }).unwrap();

        let center = result.valleyness.get(15, 15).unwrap();
        let flat = result.valleyness.get(15, 28).unwrap();
        assert!(
            center > flat,
            "Valley center should have higher valleyness: center={:.4} vs flat={:.4}",
            center, flat
        );
    }

    #[test]
    fn test_msv_ridge() {
        // Dome in center, flat at edges
        let mut dem = Raster::new(31, 31);
        dem.set_transform(GeoTransform::new(0.0, 31.0, 1.0, -1.0));
        for row in 0..31 {
            for col in 0..31 {
                let dx = col as f64 - 15.0;
                let dy = row as f64 - 15.0;
                let r = (dx * dx + dy * dy).sqrt();
                if r < 10.0 {
                    dem.set(row, col, 100.0 - r * r * 0.1).unwrap();
                } else {
                    dem.set(row, col, 90.0).unwrap();
                }
            }
        }

        let result = msv(&dem, MsvParams {
            radii: vec![3, 5],
            ..Default::default()
        }).unwrap();

        let center = result.ridgeness.get(15, 15).unwrap();
        let flat = result.ridgeness.get(15, 28).unwrap();
        assert!(
            center > flat,
            "Ridge center should have higher ridgeness: center={:.4} vs flat={:.4}",
            center, flat
        );
    }

    #[test]
    fn test_msv_flat_terrain() {
        let mut dem = Raster::filled(21, 21, 100.0_f64);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));

        let result = msv(&dem, MsvParams {
            radii: vec![3],
            ..Default::default()
        }).unwrap();

        let v = result.valleyness.get(10, 10).unwrap();
        let r = result.ridgeness.get(10, 10).unwrap();
        assert!(v.abs() < 0.01, "Flat terrain valleyness should be ~0, got {:.4}", v);
        assert!(r.abs() < 0.01, "Flat terrain ridgeness should be ~0, got {:.4}", r);
    }

    #[test]
    fn test_msv_bowl() {
        // Bowl: z = x² + y² → both eigenvalues positive
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let x = col as f64 - 10.0;
                let y = row as f64 - 10.0;
                dem.set(row, col, x * x + y * y).unwrap();
            }
        }

        let result = msv(&dem, MsvParams {
            radii: vec![4],
            ..Default::default()
        }).unwrap();

        let v = result.valleyness.get(10, 10).unwrap();
        assert!(v > 0.1, "Bowl center should have high valleyness, got {:.4}", v);
    }

    #[test]
    fn test_msv_mean_combination() {
        let mut dem = Raster::new(21, 21);
        dem.set_transform(GeoTransform::new(0.0, 21.0, 1.0, -1.0));
        for row in 0..21 {
            for col in 0..21 {
                let dx = col as f64 - 10.0;
                let dy = row as f64 - 10.0;
                dem.set(row, col, dx * dx + 0.5 * dy * dy).unwrap();
            }
        }

        let result = msv(&dem, MsvParams {
            radii: vec![3, 5],
            combination: MsvCombination::Mean,
        }).unwrap();

        let v = result.valleyness.get(10, 10).unwrap();
        assert!(v > 0.0, "Mean combination should produce positive valleyness, got {:.4}", v);
    }

    #[test]
    fn test_msv_invalid_params() {
        let dem = Raster::filled(10, 10, 100.0_f64);
        assert!(msv(&dem, MsvParams { radii: vec![], ..Default::default() }).is_err());
        assert!(msv(&dem, MsvParams { radii: vec![0], ..Default::default() }).is_err());
    }
}
