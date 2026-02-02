//! TIN (Triangulated Irregular Network) interpolation
//!
//! Constructs a Delaunay triangulation from sample points and
//! interpolates within each triangle using barycentric coordinates
//! for linear interpolation.
//!
//! Uses a simple incremental Bowyer-Watson algorithm for triangulation.

use crate::maybe_rayon::*;
use surtgis_core::raster::{GeoTransform, Raster};
use surtgis_core::{Error, Result};

use super::SamplePoint;

/// Parameters for TIN interpolation
#[derive(Debug, Clone)]
pub struct TinParams {
    /// Output raster rows
    pub rows: usize,
    /// Output raster columns
    pub cols: usize,
    /// Output raster geotransform
    pub transform: GeoTransform,
}

impl Default for TinParams {
    fn default() -> Self {
        Self {
            rows: 100,
            cols: 100,
            transform: GeoTransform::default(),
        }
    }
}

/// A triangle defined by three vertex indices
#[derive(Debug, Clone, Copy)]
struct Triangle {
    v0: usize,
    v1: usize,
    v2: usize,
}

/// Circumcircle of a triangle
#[derive(Debug, Clone, Copy)]
struct Circumcircle {
    cx: f64,
    cy: f64,
    radius_sq: f64,
}

/// Compute the circumcircle of three points
fn circumcircle(p0: &SamplePoint, p1: &SamplePoint, p2: &SamplePoint) -> Option<Circumcircle> {
    let ax = p0.x;
    let ay = p0.y;
    let bx = p1.x;
    let by = p1.y;
    let cx = p2.x;
    let cy = p2.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 {
        return None; // Degenerate triangle
    }

    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d;

    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d;

    let dx = ax - ux;
    let dy = ay - uy;

    Some(Circumcircle {
        cx: ux,
        cy: uy,
        radius_sq: dx * dx + dy * dy,
    })
}

/// Compute barycentric coordinates of point (px, py) within triangle (p0, p1, p2)
///
/// Returns (u, v, w) where the interpolated value is u*v0 + v*v1 + w*v2
fn barycentric(
    px: f64,
    py: f64,
    p0: &SamplePoint,
    p1: &SamplePoint,
    p2: &SamplePoint,
) -> (f64, f64, f64) {
    let v0x = p1.x - p0.x;
    let v0y = p1.y - p0.y;
    let v1x = p2.x - p0.x;
    let v1y = p2.y - p0.y;
    let v2x = px - p0.x;
    let v2y = py - p0.y;

    let dot00 = v0x * v0x + v0y * v0y;
    let dot01 = v0x * v1x + v0y * v1y;
    let dot02 = v0x * v2x + v0y * v2y;
    let dot11 = v1x * v1x + v1y * v1y;
    let dot12 = v1x * v2x + v1y * v2y;

    let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    let v = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    let w = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    let u = 1.0 - v - w;

    (u, v, w)
}

/// Build Delaunay triangulation using Bowyer-Watson algorithm
fn delaunay(points: &[SamplePoint]) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }

    // Find bounding box
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for p in points {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta = dx.max(dy).max(1.0);

    // Create super-triangle vertices (indices 0, 1, 2)
    let mut vertices: Vec<SamplePoint> = vec![
        SamplePoint::new(min_x - 10.0 * delta, min_y - delta, 0.0),
        SamplePoint::new(min_x + 0.5 * dx, max_y + 10.0 * delta, 0.0),
        SamplePoint::new(max_x + 10.0 * delta, min_y - delta, 0.0),
    ];

    let mut triangles: Vec<Triangle> = vec![Triangle { v0: 0, v1: 1, v2: 2 }];

    // Add each point incrementally
    for point in points {
        let vi = vertices.len(); // Index of new vertex
        vertices.push(*point);

        // Find triangles whose circumcircle contains the new point
        let mut bad_triangles: Vec<usize> = Vec::new();

        for (ti, tri) in triangles.iter().enumerate() {
            if let Some(cc) = circumcircle(&vertices[tri.v0], &vertices[tri.v1], &vertices[tri.v2])
            {
                let dx = point.x - cc.cx;
                let dy = point.y - cc.cy;
                if dx * dx + dy * dy <= cc.radius_sq {
                    bad_triangles.push(ti);
                }
            }
        }

        // Find boundary polygon of the hole (edges not shared by two bad triangles)
        let mut boundary: Vec<(usize, usize)> = Vec::new();

        for &bi in &bad_triangles {
            let tri = &triangles[bi];
            let edges = [
                (tri.v0, tri.v1),
                (tri.v1, tri.v2),
                (tri.v2, tri.v0),
            ];

            for &(ea, eb) in &edges {
                let shared = bad_triangles.iter().any(|&oi| {
                    if oi == bi {
                        return false;
                    }
                    let other = &triangles[oi];
                    let oe = [
                        (other.v0, other.v1),
                        (other.v1, other.v2),
                        (other.v2, other.v0),
                    ];
                    oe.iter().any(|&(oa, ob)| {
                        (oa == ea && ob == eb) || (oa == eb && ob == ea)
                    })
                });

                if !shared {
                    boundary.push((ea, eb));
                }
            }
        }

        // Remove bad triangles (in reverse order to preserve indices)
        bad_triangles.sort_unstable_by(|a, b| b.cmp(a));
        for bi in bad_triangles {
            triangles.swap_remove(bi);
        }

        // Create new triangles from boundary edges to new vertex
        for &(ea, eb) in &boundary {
            triangles.push(Triangle {
                v0: ea,
                v1: eb,
                v2: vi,
            });
        }
    }

    // Remove triangles that reference super-triangle vertices (0, 1, 2)
    triangles.retain(|tri| {
        tri.v0 >= 3 && tri.v1 >= 3 && tri.v2 >= 3
    });

    // Remap vertex indices (subtract 3 for the super-triangle offset)
    for tri in &mut triangles {
        tri.v0 -= 3;
        tri.v1 -= 3;
        tri.v2 -= 3;
    }

    triangles
}

/// Perform TIN interpolation from scattered points to a raster grid.
///
/// 1. Builds a Delaunay triangulation from sample points
/// 2. For each output cell, finds the enclosing triangle
/// 3. Interpolates linearly using barycentric coordinates
///
/// # Arguments
/// * `points` - Scattered sample points (minimum 3)
/// * `params` - Output grid parameters
///
/// # Returns
/// Raster with interpolated values. Cells outside the convex hull are NaN.
pub fn tin_interpolation(points: &[SamplePoint], params: TinParams) -> Result<Raster<f64>> {
    if points.len() < 3 {
        return Err(Error::Algorithm(
            "TIN requires at least 3 sample points".into(),
        ));
    }

    let rows = params.rows;
    let cols = params.cols;

    // Build triangulation
    let triangles = delaunay(points);

    if triangles.is_empty() {
        return Err(Error::Algorithm(
            "Failed to build triangulation (collinear points?)".into(),
        ));
    }

    // Interpolate each cell
    let data: Vec<f64> = (0..rows)
        .into_par_iter()
        .flat_map(|row| {
            let mut row_data = vec![f64::NAN; cols];

            for (col, row_data_col) in row_data.iter_mut().enumerate() {
                let (px, py) = params.transform.pixel_to_geo(col, row);

                // Find enclosing triangle (brute force; OK for moderate triangle counts)
                for tri in &triangles {
                    let p0 = &points[tri.v0];
                    let p1 = &points[tri.v1];
                    let p2 = &points[tri.v2];

                    let (u, v, w) = barycentric(px, py, p0, p1, p2);

                    // Check if point is inside triangle (with small tolerance)
                    const EPS: f64 = -1e-10;
                    if u >= EPS && v >= EPS && w >= EPS {
                        *row_data_col = u * p0.value + v * p1.value + w * p2.value;
                        break;
                    }
                }
            }

            row_data
        })
        .collect();

    let mut output = Raster::from_vec(data, rows, cols)?;
    output.set_transform(params.transform);
    output.set_nodata(Some(f64::NAN));

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corner_points() -> Vec<SamplePoint> {
        vec![
            SamplePoint::new(0.0, 10.0, 10.0), // top-left
            SamplePoint::new(10.0, 10.0, 20.0), // top-right
            SamplePoint::new(0.0, 0.0, 30.0),   // bottom-left
            SamplePoint::new(10.0, 0.0, 40.0),  // bottom-right
        ]
    }

    fn default_params() -> TinParams {
        TinParams {
            rows: 10,
            cols: 10,
            transform: GeoTransform::new(0.0, 10.0, 1.0, -1.0),
        }
    }

    #[test]
    fn test_delaunay_basic() {
        let points = corner_points();
        let tris = delaunay(&points);

        // 4 points should produce 2 triangles
        assert_eq!(tris.len(), 2, "Expected 2 triangles, got {}", tris.len());
    }

    #[test]
    fn test_tin_at_vertices() {
        let points = corner_points();
        let result = tin_interpolation(&points, default_params()).unwrap();

        // Near (0, 10) → value ~10.0
        let tl = result.get(0, 0).unwrap();
        assert!(
            !tl.is_nan() && (tl - 10.0).abs() < 5.0,
            "Top-left should be ~10, got {}",
            tl
        );
    }

    #[test]
    fn test_tin_linear_interpolation() {
        // Three points forming a tilted plane: z = x + y
        let points = vec![
            SamplePoint::new(0.0, 0.0, 0.0),
            SamplePoint::new(10.0, 0.0, 10.0),
            SamplePoint::new(0.0, 10.0, 10.0),
            SamplePoint::new(10.0, 10.0, 20.0),
        ];

        let result = tin_interpolation(&points, default_params()).unwrap();

        // At center (5, 5): z should be ~10
        let center = result.get(5, 5).unwrap();
        if !center.is_nan() {
            assert!(
                (center - 10.0).abs() < 2.0,
                "Center of plane z=x+y should be ~10, got {}",
                center
            );
        }
    }

    #[test]
    fn test_tin_produces_valid_output() {
        let points = corner_points();
        let result = tin_interpolation(&points, default_params()).unwrap();

        let mut valid_count = 0;
        for row in 0..10 {
            for col in 0..10 {
                let val = result.get(row, col).unwrap();
                if !val.is_nan() {
                    valid_count += 1;
                    assert!(
                        val >= 5.0 && val <= 45.0,
                        "Value {} out of expected range at ({}, {})",
                        val, row, col
                    );
                }
            }
        }

        assert!(
            valid_count > 50,
            "Should have mostly valid cells, got {}/100",
            valid_count
        );
    }

    #[test]
    fn test_tin_too_few_points() {
        let points = vec![
            SamplePoint::new(0.0, 0.0, 1.0),
            SamplePoint::new(1.0, 0.0, 2.0),
        ];

        let result = tin_interpolation(&points, default_params());
        assert!(result.is_err());
    }

    #[test]
    fn test_barycentric_at_vertices() {
        let p0 = SamplePoint::new(0.0, 0.0, 1.0);
        let p1 = SamplePoint::new(10.0, 0.0, 2.0);
        let p2 = SamplePoint::new(0.0, 10.0, 3.0);

        // At p0
        let (u, v, w) = barycentric(0.0, 0.0, &p0, &p1, &p2);
        assert!((u - 1.0).abs() < 1e-10);
        assert!(v.abs() < 1e-10);
        assert!(w.abs() < 1e-10);

        // At p1
        let (u, v, w) = barycentric(10.0, 0.0, &p0, &p1, &p2);
        assert!(u.abs() < 1e-10);
        assert!((v - 1.0).abs() < 1e-10);
        assert!(w.abs() < 1e-10);

        // At centroid (mean of vertices)
        let (u, v, w) = barycentric(
            10.0 / 3.0,
            10.0 / 3.0,
            &p0, &p1, &p2,
        );
        assert!((u - 1.0 / 3.0).abs() < 1e-10);
        assert!((v - 1.0 / 3.0).abs() < 1e-10);
        assert!((w - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_many_points() {
        // Grid of 25 points
        let mut points = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                let x = i as f64 * 2.5;
                let y = j as f64 * 2.5;
                points.push(SamplePoint::new(x, y, x + y));
            }
        }

        let result = tin_interpolation(&points, default_params()).unwrap();

        let mut valid = 0;
        for row in 0..10 {
            for col in 0..10 {
                if !result.get(row, col).unwrap().is_nan() {
                    valid += 1;
                }
            }
        }

        assert!(valid > 50, "Should interpolate most cells, got {}/100", valid);
    }
}
