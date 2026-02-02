//! Thin Plate Spline (TPS) interpolation
//!
//! Constructs a smooth surface that passes exactly through all sample points
//! while minimizing the bending energy (integral of squared second derivatives).
//!
//! The TPS interpolant has the form:
//! ```text
//! f(x,y) = a₁ + a₂·x + a₃·y + Σᵢ wᵢ · U(‖(x,y) - (xᵢ,yᵢ)‖)
//! ```
//! where U(r) = r²·ln(r) is the TPS radial basis function in 2D.
//!
//! Requires solving an (n+3)×(n+3) linear system, so it is practical
//! for up to ~5000 points. For larger datasets, use IDW or kriging.
//!
//! Reference:
//! Duchon, J. (1976). Interpolation des fonctions de deux variables suivant
//! le principe de la flexion des plaques minces. RAIRO Analyse Numérique.
//! Wahba, G. (1990). Spline Models for Observational Data. SIAM.
//! Florinsky, I.V. (2025). Digital Terrain Analysis, §3.2.

use ndarray::Array2;
use crate::maybe_rayon::*;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::SamplePoint;

/// Parameters for TPS interpolation
#[derive(Debug, Clone)]
pub struct TpsParams {
    /// Output raster rows
    pub rows: usize,
    /// Output raster columns
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
    /// Smoothing parameter (λ ≥ 0). Default 0.0 = exact interpolation.
    /// Positive values add regularization: larger λ → smoother surface
    /// that doesn't pass exactly through sample points (smoothing spline).
    pub smoothing: f64,
}

impl Default for TpsParams {
    fn default() -> Self {
        Self {
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
            smoothing: 0.0,
        }
    }
}

/// TPS radial basis function: U(r) = r² · ln(r), with U(0) = 0
#[inline]
fn tps_kernel(r: f64) -> f64 {
    if r < 1e-15 {
        0.0
    } else {
        r * r * r.ln()
    }
}

/// Perform Thin Plate Spline interpolation from scattered points to a raster grid.
///
/// # Arguments
/// * `points` — Slice of sample points with (x, y, value)
/// * `params` — Output grid specification and smoothing parameter
///
/// # Returns
/// Raster<f64> with interpolated values. Cells outside the convex hull of
/// sample points are extrapolated (TPS naturally handles extrapolation,
/// though values may be unreliable far from data).
///
/// # Errors
/// - If fewer than 3 points are provided
/// - If the linear system is singular (collinear points)
pub fn tps_interpolation(
    points: &[SamplePoint],
    params: TpsParams,
) -> Result<Raster<f64>> {
    let n = points.len();
    if n < 3 {
        return Err(Error::Algorithm(
            "TPS requires at least 3 non-collinear points".into(),
        ));
    }

    // Build the (n+3) × (n+3) system:
    // [K + λI  P] [w]   [z]
    // [Pᵀ      0] [a] = [0]
    let m = n + 3;
    let mut mat = vec![0.0_f64; m * m];
    let mut rhs = vec![0.0_f64; m];

    // Fill K matrix (n × n) with TPS kernel values
    for i in 0..n {
        for j in 0..n {
            if i == j {
                mat[i * m + j] = params.smoothing;
            } else {
                let dx = points[i].x - points[j].x;
                let dy = points[i].y - points[j].y;
                let r = (dx * dx + dy * dy).sqrt();
                mat[i * m + j] = tps_kernel(r);
            }
        }
    }

    // Fill P matrix (n × 3) and Pᵀ (3 × n)
    for i in 0..n {
        // P: rows i, columns n..n+3
        mat[i * m + n] = 1.0;
        mat[i * m + n + 1] = points[i].x;
        mat[i * m + n + 2] = points[i].y;
        // Pᵀ: rows n..n+3, column i
        mat[n * m + i] = 1.0;
        mat[(n + 1) * m + i] = points[i].x;
        mat[(n + 2) * m + i] = points[i].y;
    }

    // Bottom-right 3×3 block is already zero

    // RHS: [z₁, z₂, ..., zₙ, 0, 0, 0]
    for i in 0..n {
        rhs[i] = points[i].value;
    }

    // Solve using Gaussian elimination with partial pivoting
    let coeffs = gauss_solve(m, &mut mat, &mut rhs)?;

    // Extract weights (w₁..wₙ) and polynomial coefficients (a₁, a₂, a₃)
    let weights = &coeffs[..n];
    let a1 = coeffs[n];
    let a2 = coeffs[n + 1];
    let a3 = coeffs[n + 2];

    // Evaluate at each output cell
    let rows = params.rows;
    let cols = params.cols;
    let transform = params.transform;

    let output_data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for col in 0..cols {
                let (x, y) = transform.pixel_to_geo(col, row);

                let mut val = a1 + a2 * x + a3 * y;

                for (i, pt) in points.iter().enumerate() {
                    let dx = x - pt.x;
                    let dy = y - pt.y;
                    let r = (dx * dx + dy * dy).sqrt();
                    val += weights[i] * tps_kernel(r);
                }

                row_data[col] = val;
            }

            row_data
        })
        .collect();

    let mut output = Raster::new(rows, cols);
    output.set_transform(transform);
    output.set_nodata(Some(f64::NAN));
    *output.data_mut() = Array2::from_shape_vec((rows, cols), output_data)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(output)
}

/// Solve Ax = b using Gaussian elimination with partial pivoting.
///
/// Modifies `mat` and `rhs` in place. Returns solution vector.
fn gauss_solve(n: usize, mat: &mut [f64], rhs: &mut [f64]) -> Result<Vec<f64>> {
    // Forward elimination
    for col in 0..n {
        // Find pivot (max absolute value in column)
        let mut max_val = mat[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = mat[row * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-14 {
            return Err(Error::Algorithm(
                "TPS: singular matrix (points may be collinear or duplicate)".into(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                let a = col * n + j;
                let b = max_row * n + j;
                mat.swap(a, b);
            }
            rhs.swap(col, max_row);
        }

        // Eliminate below
        let pivot = mat[col * n + col];
        for row in (col + 1)..n {
            let factor = mat[row * n + col] / pivot;
            mat[row * n + col] = 0.0;
            for j in (col + 1)..n {
                mat[row * n + j] -= factor * mat[col * n + j];
            }
            rhs[row] -= factor * rhs[col];
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for col in (0..n).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..n {
            sum -= mat[col * n + j] * x[j];
        }
        x[col] = sum / mat[col * n + col];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params(rows: usize, cols: usize, extent: (f64, f64, f64, f64)) -> TpsParams {
        let (x_min, y_min, x_max, y_max) = extent;
        let x_res = (x_max - x_min) / cols as f64;
        let y_res = -(y_max - y_min) / rows as f64;
        TpsParams {
            rows,
            cols,
            transform: GeoTransform::new(x_min, y_max, x_res, y_res),
            smoothing: 0.0,
        }
    }

    #[test]
    fn test_tps_exact_interpolation() {
        // TPS should pass exactly through sample points
        let points = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(10.0, 0.0, 20.0),
            SamplePoint::new(0.0, 10.0, 30.0),
            SamplePoint::new(10.0, 10.0, 40.0),
            SamplePoint::new(5.0, 5.0, 25.0),
        ];

        let params = make_params(11, 11, (0.0, 0.0, 10.0, 10.0));
        let result = tps_interpolation(&points, params).unwrap();

        // Check that points near sample locations have approximately correct values
        // Cell (0, 0) corresponds to (0.45, 9.55) approximately
        // Cell (10, 10) corresponds to (9.55, 0.45) approximately
        // The exact cell containing (5, 5) would be around (5, 5)
        let center = result.get(5, 5).unwrap();
        assert!(
            (center - 25.0).abs() < 2.0,
            "Center should be ~25.0, got {:.2}",
            center
        );
    }

    #[test]
    fn test_tps_linear_surface() {
        // For a perfectly linear surface f(x,y) = 2x + 3y + 1,
        // TPS should reproduce it exactly (up to numerical precision)
        let points: Vec<SamplePoint> = vec![
            SamplePoint::new(0.0, 0.0, 1.0),
            SamplePoint::new(10.0, 0.0, 21.0),
            SamplePoint::new(0.0, 10.0, 31.0),
            SamplePoint::new(10.0, 10.0, 51.0),
            SamplePoint::new(5.0, 5.0, 26.0),
            SamplePoint::new(3.0, 7.0, 28.0),
        ];

        let params = make_params(11, 11, (0.0, 0.0, 10.0, 10.0));
        let result = tps_interpolation(&points, params).unwrap();

        // Check a few interior cells
        for row in 2..9 {
            for col in 2..9 {
                let x = col as f64 * (10.0 / 11.0) + 0.5 * (10.0 / 11.0);
                let y = 10.0 - (row as f64 * (10.0 / 11.0) + 0.5 * (10.0 / 11.0));
                let expected = 2.0 * x + 3.0 * y + 1.0;
                let actual = result.get(row, col).unwrap();
                assert!(
                    (actual - expected).abs() < 1.0,
                    "Linear surface at ({},{}) [{:.1},{:.1}]: expected {:.2}, got {:.2}",
                    row, col, x, y, expected, actual
                );
            }
        }
    }

    #[test]
    fn test_tps_smoothing_parameter() {
        // With smoothing > 0, the surface should be smoother
        // and may not pass exactly through sample points
        let points = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(10.0, 0.0, 10.0),
            SamplePoint::new(5.0, 5.0, 100.0), // spike
            SamplePoint::new(0.0, 10.0, 10.0),
            SamplePoint::new(10.0, 10.0, 10.0),
        ];

        let exact_params = make_params(11, 11, (0.0, 0.0, 10.0, 10.0));
        let smooth_params = TpsParams {
            smoothing: 100.0,
            ..make_params(11, 11, (0.0, 0.0, 10.0, 10.0))
        };

        let exact = tps_interpolation(&points, exact_params).unwrap();
        let smooth = tps_interpolation(&points, smooth_params).unwrap();

        // Center value in exact should be ~100, in smooth should be lower
        let exact_center = exact.get(5, 5).unwrap();
        let smooth_center = smooth.get(5, 5).unwrap();

        assert!(
            smooth_center < exact_center,
            "Smoothing should reduce spike: exact={:.1}, smooth={:.1}",
            exact_center, smooth_center
        );
    }

    #[test]
    fn test_tps_too_few_points() {
        let points = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(1.0, 0.0, 20.0),
        ];
        let params = make_params(5, 5, (0.0, 0.0, 1.0, 1.0));
        assert!(tps_interpolation(&points, params).is_err());
    }

    #[test]
    fn test_tps_kernel_function() {
        assert!((tps_kernel(0.0)).abs() < 1e-10, "U(0) should be 0");
        assert!((tps_kernel(1.0)).abs() < 1e-10, "U(1) = 1·ln(1) = 0");

        let u2 = tps_kernel(2.0);
        let expected = 4.0 * 2.0_f64.ln();
        assert!(
            (u2 - expected).abs() < 1e-10,
            "U(2) = 4·ln(2) ≈ {:.4}, got {:.4}",
            expected, u2
        );
    }

    #[test]
    fn test_tps_symmetry() {
        // Symmetric input should produce symmetric output
        let points = vec![
            SamplePoint::new(0.0, 0.0, 10.0),
            SamplePoint::new(10.0, 0.0, 10.0),
            SamplePoint::new(0.0, 10.0, 10.0),
            SamplePoint::new(10.0, 10.0, 10.0),
            SamplePoint::new(5.0, 5.0, 50.0),
        ];

        let params = make_params(11, 11, (0.0, 0.0, 10.0, 10.0));
        let result = tps_interpolation(&points, params).unwrap();

        // Check symmetry: (2,2) should equal (2,8), (8,2), (8,8) approximately
        let v22 = result.get(2, 2).unwrap();
        let v28 = result.get(2, 8).unwrap();
        let v82 = result.get(8, 2).unwrap();
        let v88 = result.get(8, 8).unwrap();

        assert!(
            (v22 - v28).abs() < 0.5 && (v22 - v82).abs() < 0.5 && (v22 - v88).abs() < 0.5,
            "Symmetric input should give symmetric output: {:.1}, {:.1}, {:.1}, {:.1}",
            v22, v28, v82, v88
        );
    }
}
